"""
submission/indexer.py — the inverted index: text analysis, postings
construction, and a compact on-disk format.

Required component (assignment Section 4.1), built from scratch: no Lucene,
Elasticsearch, Pyserini or Whoosh anywhere in here.

Text analysis pipeline (Lecture 2, "At the End of it...")
--------------------------------------------------------
    raw text -> lowercase -> [a-z0-9]+ tokens -> stopword removal
             -> Porter stemming -> index terms

Every stage is a deliberate choice, and every one is shared between indexing
and querying — an index built with one analyser and queried with another
silently loses most of its matches, which is why `analyze()` is the single
entry point both sides call.

Index layout (Retrieval-I "Inverted Index", Retrieval-II "Compression")
----------------------------------------------------------------------
Internally a document is an integer id 0..N-1 assigned in corpus order, and a
term is an integer id assigned by lexicographic rank. The dictionary is a
*sorted array of terms* searched with binary search rather than a hash map:
Retrieval-I's point that the dictionary must stay small and fast for random
access applies here too, and a sorted list of 165K strings costs a fraction of
the memory of the equivalent Python dict.

In memory (both after build() and after load()) the postings are three flat
NumPy arrays — `_doc_ids`, `_freqs`, and `_starts` — where term t's postings
are the slice `[_starts[t], _starts[t+1])`. The forward index has the same
shape with the roles of term and document swapped. Every query-time operation
is a slice of these arrays; nothing is decoded per query.

On disk
-------
    meta.json         collection statistics, analyser settings, and the chunk
                      table for the compressed streams below
    terms.bin         lzma(sorted term strings, newline-joined)
    docids.bin        lzma(external doc_id strings, newline-joined)
    stats.bin         lzma(VByte: df per term ++ length per doc ++ forward
                      count per doc)
    postings.bin      lzma(VByte d-gap stream, all terms concatenated)
    postings_tf.bin   lzma(VByte term frequencies > 1)
    fwd.bin           lzma(VByte term-id-gap stream, all docs concatenated)
    fwd_tf.bin        lzma(VByte term frequencies > 1)

Two ideas do most of the work on size:

  d-gaps.      Doc ids inside a postings list are ascending, so the difference
               between neighbours is small (a term in 30% of documents has a
               mean gap of ~3), and VByte spends one byte on anything < 128.

  tf flag.     72.5% of postings in this corpus have tf = 1. Spending a whole
               byte on each of those is the single biggest waste in a naive
               (gap, tf) layout. Instead the low bit of each gap carries a flag
               "tf > 1", and a separate, much shorter stream holds only the
               frequencies that are actually > 1 (stored as tf - 2, since
               tf >= 2 is known). Gaps are re-derived by shifting the flag off.
               This alone takes the raw postings from 27.0 MB to 19.3 MB.

  rank space.  The forward index stores term ids in *frequency-rank* space:
               the most common term is 0, the next is 1, and so on. A
               document's most frequent terms are overwhelmingly common terms,
               so in rank space their ids — and hence their gaps — are small.
               This takes the forward index from 5.7 MB to 3.9 MB. It costs
               nothing on disk: the rank permutation is a function of the df
               table, which is stored anyway, so `load()` simply recomputes it
               (a stable argsort of df) and maps the ids back.

The streams are then LZMA-compressed as a whole (12.1 MB for the postings) —
which is only possible because nothing is read from disk at query time. The
whole index is decompressed and decoded into the flat arrays once, in `load()`.
That costs ~1 s of load time, and load time is not part of the efficiency
score (assignment Section 7 scores index *build* time and *query* latency);
in exchange the on-disk footprint roughly halves and every query is a pure
array slice.

Compression runs in parallel on `build()`: each stream is cut into chunks that
are LZMA-compressed by a thread pool (`lzma.compress` releases the GIL), so
the wall-clock cost on the 4-core grading machine is a fraction of the ~7 s it
would take serially. Build time IS scored, so that matters.

The pruned forward index
------------------------
An inverted index answers "which documents contain this term". Pseudo-relevance
feedback (`custom_scorer.py`) needs the opposite: "which terms does this
document contain", for the handful of documents the first retrieval pass ranked
highest. Nothing in the inverted index can answer that without scanning all of
it, so a forward index has to be persisted — and a full one would roughly
double the on-disk footprint, since term-id gaps within a document are large
(a document's ~90 terms are scattered across a 165K-term vocabulary) where doc
-id gaps within a postings list are small.

So we store only the `forward_terms_per_doc` highest-frequency terms of each
document. That is exactly the part relevance feedback uses: the expansion terms
RM3 selects are the ones maximising sum_d P(w|d) = tf/|D|, so a term that is
not among a document's most frequent can never be a top expansion candidate
from it. The dev-set ablation (report, Table 4) shows nDCG@10 rising with the
cut-off up to 24 terms per document and flat beyond it, so 24 is where the
index stops buying anything.

Three things are deliberately NOT persisted, because the index-size component
(Section 7) charges for every byte and `retrieve()` never needs them:
  - the raw document text (the starter skeleton's `doc_text` field): BM25 and
    cosine need only term frequencies and lengths;
  - a *full* forward index, for the reason above;
  - precomputed IDF values, BM25 partial scores or document norms, which are
    cheap to recompute from what *is* stored at load time.
"""
import json
import lzma
import os
import re
from bisect import bisect_left
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from submission import codecs
from submission.porter import stem as porter_stem

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_END = re.compile(r"[.!?\n]")


def boost_title(text: str, boost: int) -> str:
    """Repeat a document's opening span so its terms count `boost` times.

    The corpus format stores each document as its title followed by its body,
    so the text up to the first sentence boundary is (approximately) the
    title. Title words are disproportionately what queries ask about, and
    BM25 has no notion of position — the only way to tell it "this part
    matters more" is to raise those terms' frequencies. Measured on five
    collections, counting the opening sentence twice improves nDCG@10 on
    every one of them (report, cross-dataset section).
    """
    if boost <= 1:
        return text
    match = _SENTENCE_END.search(text)
    title = text[: match.start()] if match else text[:120]
    if not title:
        return text
    return text + (" " + title) * (boost - 1)

# A conservative English stopword list. These are the words Zipf's law makes
# ubiquitous (Lecture 2, "Stopwords") — they carry almost no discriminative
# power, they would otherwise dominate the postings file, and BM25 already
# drives their IDF close to zero, so dropping them costs almost no ranking
# quality while removing roughly a third of all postings. Kept deliberately
# short: aggressive lists eat genuinely meaningful query words ("A vitamin",
# "to be or not to be"), and every word removed here is a word no query can
# ever match again.
_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at
be because been before being below between both but by
can cannot could did do does doing down during
each few for from further
had has have having he her here hers herself him himself his how
i if in into is it its itself
just
me more most my myself
no nor not now
of off on once only or other our ours ourselves out over own
same she should so some such
than that the their theirs them themselves then there these they this those through to too
under until up
very
was we were what when where which while who whom why will with would
you your yours yourself yourselves
""".split())

_INDEX_FORMAT_VERSION = 5

# How many of each document's most frequent terms the forward index keeps.
# 0 disables the forward index entirely (and with it relevance feedback).
DEFAULT_FORWARD_TERMS_PER_DOC = 24

# LZMA preset for the on-disk streams. 6 is the library default; higher
# presets buy ~1-2% for several times the compression time, which is charged
# against the build-time score.
_LZMA_PRESET = 6

# Postings encoded per vectorised call; bounds encode-time scratch memory.
_ENCODE_CHUNK_ENTRIES = 2_000_000

# Streams are compressed in chunks of this many bytes so the work can be
# spread across processes. Smaller chunks parallelise better but compress
# slightly worse (each chunk starts with an empty LZMA dictionary).
_COMPRESS_CHUNK_BYTES = 4 * 1024 * 1024


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric-only tokenization.

    Kept as the starter repo defined it — this is the raw token stream,
    *before* stopwording and stemming. `analyze()` is what produces index
    terms; both indexing and querying must go through that.
    """
    return _TOKEN_RE.findall(text.lower())


def analyze(text: str, remove_stopwords: bool = True, stemming: bool = True) -> List[str]:
    """Full text -> index terms. The one analyser both sides of the system use.

    Returns terms in order of occurrence (order is not used by BM25 or cosine,
    but keeping it means a proximity or bigram feature can be layered on later
    without changing the analyser).
    """
    terms = []
    for token in _TOKEN_RE.findall(text.lower()):
        if remove_stopwords and token in _STOPWORDS:
            continue
        terms.append(porter_stem(token) if stemming else token)
    return terms


def _corpus_chunks(path: str, n_chunks: int):
    """Split a JSONL file into byte ranges on line boundaries."""
    total = os.path.getsize(path)
    bounds = [0]
    with open(path, "rb") as f:
        for i in range(1, n_chunks):
            f.seek(total * i // n_chunks)
            f.readline()  # skip to the end of the current line
            bounds.append(f.tell())
    bounds.append(total)
    return [(lo, hi) for lo, hi in zip(bounds, bounds[1:]) if hi > lo]


def _sweep_chunk(args):
    """Tokenise one byte range of the corpus. Runs in a worker process.

    Returns everything build needs, with term ids in a chunk-LOCAL first-seen
    numbering plus the local vocabulary to translate them; the parent remaps
    every chunk onto one global lexicographic numbering, so the merged result
    is identical to a serial sweep.
    """
    path, lo, hi, remove_stopwords, stemming, forward_terms_per_doc, title_boost = args
    from array import array
    from heapq import nlargest
    from operator import itemgetter

    term_to_id: Dict[str, int] = {}
    token_cache: Dict[str, int] = {}
    posting_terms = array("i"); posting_docs = array("i"); posting_freqs = array("i")
    fwd_terms = array("i"); fwd_freqs = array("i"); fwd_counts = array("i")
    doc_ids: List[str] = []; doc_len: List[int] = []

    with open(path, "rb") as f:
        f.seek(lo)
        local_doc = 0
        while f.tell() < hi:
            line = f.readline()
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj["text"]
            if title_boost > 1:
                text = boost_title(text, title_boost)

            raw_counts = Counter(_TOKEN_RE.findall(text.lower()))
            doc_counts: Dict[int, int] = {}
            for token, count in raw_counts.items():
                term_id = token_cache.get(token, -2)
                if term_id == -2:
                    if remove_stopwords and token in _STOPWORDS:
                        term_id = -1
                    else:
                        term = porter_stem(token) if stemming else token
                        term_id = term_to_id.get(term)
                        if term_id is None:
                            term_id = len(term_to_id)
                            term_to_id[term] = term_id
                    token_cache[token] = term_id
                if term_id < 0:
                    continue
                doc_counts[term_id] = doc_counts.get(term_id, 0) + count

            posting_terms.extend(doc_counts.keys())
            posting_freqs.extend(doc_counts.values())
            posting_docs.extend([local_doc] * len(doc_counts))

            if forward_terms_per_doc:
                if len(doc_counts) > forward_terms_per_doc:
                    kept = nlargest(forward_terms_per_doc, doc_counts.items(), key=itemgetter(1))
                    kept.sort()
                else:
                    kept = sorted(doc_counts.items())
                fwd_terms.extend(t for t, _f in kept)
                fwd_freqs.extend(f for _t, f in kept)
                fwd_counts.append(len(kept))

            doc_ids.append(obj["doc_id"])
            doc_len.append(sum(doc_counts.values()))
            local_doc += 1

    vocab = [None] * len(term_to_id)
    for term, tid in term_to_id.items():
        vocab[tid] = term
    return (doc_ids, doc_len, vocab,
            posting_terms.tobytes(), posting_docs.tobytes(), posting_freqs.tobytes(),
            fwd_terms.tobytes(), fwd_freqs.tobytes(), fwd_counts.tobytes())


class InvertedIndex:
    """An inverted index held as flat NumPy arrays, with a pruned forward
    index alongside it.

    `build()` and `load()` both leave the object in the same shape, so the
    scorers never care which one produced it: `postings(term_id)` and
    `forward(doc_id)` are array slices either way.
    """

    def __init__(self):
        # Dictionary: `terms` is sorted; a term's id is its position in it.
        self.terms: List[str] = []
        self.df = np.zeros(0, dtype=np.int64)         # df[term_id]
        self.doc_len = np.zeros(0, dtype=np.int64)    # doc_len[doc_id], in index terms
        self.doc_ids: List[str] = []                  # external doc_id per internal id
        self.N: int = 0                               # number of documents
        self.avg_doc_len: float = 0.0
        self.remove_stopwords: bool = True
        self.stemming: bool = True

        # Postings, term-major: term t owns [_starts[t], _starts[t+1]).
        self._doc_ids = np.zeros(0, dtype=np.int32)
        self._freqs = np.zeros(0, dtype=np.int32)
        self._starts = np.zeros(1, dtype=np.int64)

        # Pruned forward index, doc-major: doc d owns [_fwd_starts[d], _fwd_starts[d+1]).
        self.forward_terms_per_doc: int = 0
        self._fwd_terms = np.zeros(0, dtype=np.int32)
        self._fwd_freqs = np.zeros(0, dtype=np.int32)
        self._fwd_starts = np.zeros(1, dtype=np.int64)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def build(
        self,
        corpus,
        remove_stopwords: bool = True,
        stemming: bool = True,
        forward_terms_per_doc: int = DEFAULT_FORWARD_TERMS_PER_DOC,
        title_boost: int = 1,
    ) -> None:
        """Build the index from an iterable of (doc_id, text) pairs.

        This is a single-pass in-memory inversion (Retrieval-II, "Single-pass
        In-memory Inversion"): one sweep over the collection emitting
        (term_id, doc_id, tf) triples into flat arrays, then one sort on the
        term component to turn a document-ordered stream into a term-ordered
        one. The sort is a *stable* sort on term id alone, which is what keeps
        doc ids ascending inside each postings list for free — that ascending
        order is exactly what makes the d-gaps in `save()` small.

        The alternative (a dict of term -> dict of doc -> tf, as the starter
        skeleton sketches) needs one Python object per posting and roughly an
        order of magnitude more memory on a 171K-document collection.
        """
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.forward_terms_per_doc = forward_terms_per_doc

        term_to_id: Dict[str, int] = {}
        # raw token -> term id, or -1 for "dropped as a stopword". Caching the
        # whole analysis per distinct token means the Porter stemmer runs once
        # per vocabulary *type*, not once per token occurrence — on this corpus
        # that is ~600K stem calls instead of ~30M.
        token_cache: Dict[str, int] = {}

        from array import array
        from heapq import nlargest
        from operator import itemgetter

        posting_terms = array("i")
        posting_docs = array("i")
        posting_freqs = array("i")

        fwd_terms = array("i")
        fwd_freqs = array("i")
        fwd_counts = array("i")

        doc_ids: List[str] = []
        doc_len: List[int] = []

        for internal_id, (external_id, text) in enumerate(corpus):
            if title_boost > 1:
                text = boost_title(text, title_boost)
            # Counting raw tokens first keeps the hot loop in C: only the
            # distinct tokens of a document reach Python-level analysis.
            raw_counts = Counter(_TOKEN_RE.findall(text.lower()))

            doc_counts: Dict[int, int] = {}
            for token, count in raw_counts.items():
                term_id = token_cache.get(token, -2)
                if term_id == -2:
                    if remove_stopwords and token in _STOPWORDS:
                        term_id = -1
                    else:
                        term = porter_stem(token) if stemming else token
                        term_id = term_to_id.get(term)
                        if term_id is None:
                            term_id = len(term_to_id)
                            term_to_id[term] = term_id
                    token_cache[token] = term_id
                if term_id < 0:
                    continue
                # Distinct raw tokens can share a stem ("virus"/"viruses"), so
                # accumulate rather than assign.
                doc_counts[term_id] = doc_counts.get(term_id, 0) + count

            posting_terms.extend(doc_counts.keys())
            posting_freqs.extend(doc_counts.values())
            posting_docs.extend([internal_id] * len(doc_counts))

            if forward_terms_per_doc:
                # Keep this document's most frequent terms for the forward
                # index. Selecting here, while doc_counts is still in hand,
                # avoids a second sort of all ~12M postings into doc-major
                # order later. nlargest is O(n log M) for M kept terms; the
                # term id is the tie-break so the choice is deterministic.
                if len(doc_counts) > forward_terms_per_doc:
                    kept = nlargest(
                        forward_terms_per_doc, doc_counts.items(), key=itemgetter(1)
                    )
                    kept.sort()  # back to ascending term id, for gap coding
                else:
                    kept = sorted(doc_counts.items())
                fwd_terms.extend(t for t, _f in kept)
                fwd_freqs.extend(f for _t, f in kept)
                fwd_counts.append(len(kept))

            doc_ids.append(external_id)
            doc_len.append(sum(doc_counts.values()))

        self.doc_ids = doc_ids
        self.doc_len = np.asarray(doc_len, dtype=np.int64)
        self.N = len(doc_ids)
        self.avg_doc_len = (float(self.doc_len.sum()) / self.N) if self.N else 0.0

        # Term ids were handed out in first-seen order; re-map them to
        # lexicographic order so the dictionary can be a sorted array (binary
        # searchable, and far more compressible — adjacent sorted terms share
        # long prefixes, which is most of why terms.bin is small).
        self.terms = sorted(term_to_id)
        rank_of_old_id = np.zeros(len(self.terms), dtype=np.int64)
        for new_id, term in enumerate(self.terms):
            rank_of_old_id[term_to_id[term]] = new_id

        if forward_terms_per_doc:
            self._build_forward(fwd_terms, fwd_freqs, fwd_counts, rank_of_old_id)
        else:
            self._fwd_starts = np.zeros(self.N + 1, dtype=np.int64)
        del fwd_terms, fwd_freqs, fwd_counts

        self._invert(posting_terms, posting_docs, posting_freqs, rank_of_old_id)
        # The document-ordered triples are dead once inverted; on a large
        # collection they are hundreds of megabytes, and save() is about to
        # want that space for encoding scratch.
        del posting_terms, posting_docs, posting_freqs

    def build_from_file(
        self,
        corpus_path: str,
        remove_stopwords: bool = True,
        stemming: bool = True,
        forward_terms_per_doc: int = DEFAULT_FORWARD_TERMS_PER_DOC,
        title_boost: int = 1,
        workers: int = None,
    ) -> None:
        """`build()`, but reading the corpus file in parallel.

        Tokenisation is two thirds of build time and runs Python-level code,
        so threads cannot parallelise it — processes can. The file is split
        into byte ranges on line boundaries, each worker sweeps its range
        under a chunk-local term numbering, and the parent remaps every chunk
        onto one global lexicographic vocabulary. Because documents keep file
        order and the final vocabulary is sorted, the merged index is
        byte-identical to a serial `build()` (the tests assert this).

        Falls back to the serial path for small files or if a pool cannot be
        started.
        """
        workers = workers or min(4, os.cpu_count() or 1)
        if workers < 2 or os.path.getsize(corpus_path) < 8 * 1024 * 1024:
            return self._build_serial_from_file(
                corpus_path, remove_stopwords, stemming, forward_terms_per_doc, title_boost
            )
        chunks = _corpus_chunks(corpus_path, workers)
        jobs = [(corpus_path, lo, hi, remove_stopwords, stemming, forward_terms_per_doc, title_boost)
                for lo, hi in chunks]
        try:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as pool:
                parts = list(pool.map(_sweep_chunk, jobs))
        except Exception:  # pragma: no cover - environment-dependent
            return self._build_serial_from_file(
                corpus_path, remove_stopwords, stemming, forward_terms_per_doc, title_boost
            )

        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.forward_terms_per_doc = forward_terms_per_doc

        # Global vocabulary: sorted union of the chunk vocabularies — exactly
        # the term set (and order) a serial build would produce.
        self.terms = sorted(set().union(*(set(v) for _d, _l, v, *_ in parts)))
        global_id = {t: i for i, t in enumerate(self.terms)}

        term_arrays, doc_arrays, freq_arrays = [], [], []
        fwd_t, fwd_f, fwd_c = [], [], []
        doc_ids: List[str] = []; doc_len_all = []
        doc_base = 0
        for d_ids, d_len, vocab, pt, pd, pf, ft, ff, fc in parts:
            remap = np.asarray([global_id[t] for t in vocab], dtype=np.int32)
            local_terms = np.frombuffer(pt, dtype=np.int32)
            term_arrays.append(remap[local_terms] if local_terms.size else local_terms)
            doc_arrays.append(np.frombuffer(pd, dtype=np.int32) + doc_base)
            freq_arrays.append(np.frombuffer(pf, dtype=np.int32))
            if forward_terms_per_doc:
                lf = np.frombuffer(ft, dtype=np.int32)
                fwd_t.append(remap[lf] if lf.size else lf)
                fwd_f.append(np.frombuffer(ff, dtype=np.int32))
                fwd_c.append(np.frombuffer(fc, dtype=np.int32))
            doc_ids.extend(d_ids); doc_len_all.extend(d_len)
            doc_base += len(d_ids)

        self.doc_ids = doc_ids
        self.doc_len = np.asarray(doc_len_all, dtype=np.int64)
        self.N = len(doc_ids)
        self.avg_doc_len = (float(self.doc_len.sum()) / self.N) if self.N else 0.0

        identity = np.arange(len(self.terms), dtype=np.int64)
        if forward_terms_per_doc and fwd_t:
            from array import array as _arr
            self._build_forward(
                _np_to_array(np.concatenate(fwd_t)), _np_to_array(np.concatenate(fwd_f)),
                _np_to_array(np.concatenate(fwd_c)), identity,
            )
        else:
            self._fwd_starts = np.zeros(self.N + 1, dtype=np.int64)

        self._invert(
            _np_to_array(np.concatenate(term_arrays)) if term_arrays else b"",
            _np_to_array(np.concatenate(doc_arrays)) if doc_arrays else b"",
            _np_to_array(np.concatenate(freq_arrays)) if freq_arrays else b"",
            identity,
        )

    def _build_serial_from_file(self, corpus_path, remove_stopwords, stemming, fwd, title_boost=1):
        def stream():
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        yield obj["doc_id"], obj["text"]
        self.build(stream(), remove_stopwords, stemming, fwd, title_boost=title_boost)

    def _build_forward(self, fwd_terms, fwd_freqs, fwd_counts, rank_of_old_id) -> None:
        """Re-map the forward index onto lexicographic term ids and restore
        ascending order within each document.

        The terms were collected under first-seen ids, so re-mapping scrambles
        the within-document ordering that gap coding depends on. Re-sorting is
        done per document with one vectorised trick: add the document index
        times (number of terms + 1) to each term id, so a single global sort
        orders by document first and term id second.
        """
        counts = np.frombuffer(fwd_counts, dtype=np.int32).astype(np.int64)
        terms = rank_of_old_id[np.frombuffer(fwd_terms, dtype=np.int32)]
        freqs = np.frombuffer(fwd_freqs, dtype=np.int32)

        n_terms = len(self.terms) + 1
        doc_of_entry = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
        order = np.argsort(doc_of_entry * n_terms + terms, kind="stable")
        self._fwd_terms = terms[order].astype(np.int32)
        self._fwd_freqs = freqs[order].astype(np.int32)
        self._fwd_starts = np.concatenate(([0], np.cumsum(counts)))

    def _invert(self, posting_terms, posting_docs, posting_freqs, rank_of_old_id) -> None:
        """Sort the (term, doc, tf) triples by term and record where each
        term's slice begins."""
        n_terms = len(self.terms)
        terms_np = rank_of_old_id[np.frombuffer(posting_terms, dtype=np.int32)]

        # Stable sort on the term key alone: within a term, the triples stay in
        # the document order they were emitted in, i.e. ascending doc id.
        order = np.argsort(terms_np, kind="stable")
        sorted_terms = terms_np[order]
        del terms_np
        self._doc_ids = np.frombuffer(posting_docs, dtype=np.int32)[order]
        self._freqs = np.frombuffer(posting_freqs, dtype=np.int32)[order]
        del order

        # _starts[t] = index of term t's first posting; searchsorted gives all
        # of them in one pass, and also handles terms with no postings.
        self._starts = np.searchsorted(sorted_terms, np.arange(n_terms + 1)).astype(np.int64)
        self.df = np.diff(self._starts)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def term_id(self, term: str) -> int:
        """Binary search the sorted dictionary. Returns -1 if absent."""
        i = bisect_left(self.terms, term)
        if i < len(self.terms) and self.terms[i] == term:
            return i
        return -1

    def document_frequency(self, term: str) -> int:
        """Number of documents containing `term` at least once.

        Accepts a raw word and analyses it the same way the corpus was
        analysed, so `document_frequency("viruses")` and
        `document_frequency("virus")` both answer for the index term `viru`.
        A stopword, or a word absent from the collection, has df 0.
        """
        analyzed = analyze(term, self.remove_stopwords, self.stemming)
        if len(analyzed) != 1:
            return 0
        tid = self.term_id(analyzed[0])
        return int(self.df[tid]) if tid >= 0 else 0

    def postings(self, term_id: int):
        """Return (doc_ids, term_freqs) for `term_id` as ascending arrays.

        A pure slice of the in-memory arrays — no decoding, no I/O. This is
        what keeps per-query latency to a few milliseconds.
        """
        if term_id < 0:
            return _EMPTY, _EMPTY
        lo, hi = int(self._starts[term_id]), int(self._starts[term_id + 1])
        return self._doc_ids[lo:hi], self._freqs[lo:hi]

    def postings_for(self, term: str):
        """`postings()` keyed by an index term string."""
        return self.postings(self.term_id(term))

    def postings_range(self, term_id: int):
        """(lo, hi) bounds of `term_id`'s slice in the flat postings arrays.

        Lets a scorer that keeps its own per-posting table (e.g. BM25's
        precomputed partial scores) address it with the same slice.
        """
        return int(self._starts[term_id]), int(self._starts[term_id + 1])

    def forward(self, doc_id: int):
        """Return (term_ids, frequencies) for `doc_id`'s most frequent terms.

        This is the pruned forward index — the top `forward_terms_per_doc`
        terms of the document, ascending by term id. Returns empty arrays if
        the index was built without a forward index.
        """
        lo, hi = int(self._fwd_starts[doc_id]), int(self._fwd_starts[doc_id + 1])
        return self._fwd_terms[lo:hi], self._fwd_freqs[lo:hi]

    def iter_postings_blocks(self, terms_per_block: int = 8192):
        """Sweep the whole index, yielding (term_ids, doc_ids, freqs) arrays a
        block of terms at a time.

        Scoring a query only ever touches a handful of postings lists, so this
        is not on the hot path — it exists for the operations that genuinely
        need every posting, chiefly the vector-space document norms in
        `boolean_vsm.py`. Blocks bound the size of the `term_ids` array that
        has to be materialised alongside each slice.
        """
        n_terms = len(self.terms)
        for first in range(0, n_terms, terms_per_block):
            last = min(first + terms_per_block, n_terms)
            lo, hi = int(self._starts[first]), int(self._starts[last])
            if hi == lo:
                continue
            term_ids = np.repeat(np.arange(first, last, dtype=np.int64), self.df[first:last])
            yield term_ids, self._doc_ids[lo:hi], self._freqs[lo:hi]

    def close(self) -> None:
        """Kept for API compatibility; nothing is held open at query time."""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, index_dir: str) -> None:
        """Write the index to `index_dir` in the format documented at the top
        of this module."""
        os.makedirs(index_dir, exist_ok=True)

        # docids.bin is newline-delimited, so a doc_id containing a newline
        # would silently split into two and desynchronise every id after it.
        # Fail loudly at build time instead of returning wrong doc_ids at query
        # time. (Index terms can't hit this: they are [a-z0-9]+ by construction.)
        if any("\n" in doc_id for doc_id in self.doc_ids):
            raise ValueError(
                "a doc_id in the corpus contains a newline, which the index's "
                "doc-id table cannot represent"
            )

        gap_stream, tf_stream = _encode_gap_streams(self._doc_ids, self._freqs, self._starts)

        # Forward index in frequency-rank space (see module docstring), with
        # each document's entries re-sorted so the gaps are ascending again.
        rank_of_term = _rank_of_term(self.df)
        fwd_ranks, fwd_freqs = _resort_within_lists(
            rank_of_term[self._fwd_terms], self._fwd_freqs, self._fwd_starts
        )
        fwd_gap_stream, fwd_tf_stream = _encode_gap_streams(fwd_ranks, fwd_freqs, self._fwd_starts)
        del fwd_ranks, fwd_freqs
        stats = np.concatenate((
            np.asarray(self.df, dtype=np.int64),
            np.asarray(self.doc_len, dtype=np.int64),
            np.diff(self._fwd_starts),
        ))

        streams = {
            "postings.bin": gap_stream,
            "postings_tf.bin": tf_stream,
            "fwd.bin": fwd_gap_stream,
            "fwd_tf.bin": fwd_tf_stream,
            "stats.bin": codecs.vbyte_encode(stats),
            "terms.bin": "\n".join(self.terms).encode("utf-8"),
            "docids.bin": "\n".join(self.doc_ids).encode("utf-8"),
        }
        chunk_table = _compress_streams(streams, index_dir)

        meta = {
            "format_version": _INDEX_FORMAT_VERSION,
            "num_docs": self.N,
            "num_terms": len(self.terms),
            "avg_doc_len": self.avg_doc_len,
            "remove_stopwords": self.remove_stopwords,
            "stemming": self.stemming,
            "forward_terms_per_doc": self.forward_terms_per_doc,
            "chunks": chunk_table,
        }
        with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, separators=(",", ":"))

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """Reconstruct the index from `index_dir` alone, in a fresh process.

        Everything is decompressed and decoded here, once, into the flat
        arrays `postings()` and `forward()` slice. See the module docstring
        for why paying this at load time (unscored) rather than per query
        (scored) is the right trade.
        """
        index = cls()

        with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format_version") != _INDEX_FORMAT_VERSION:
            raise ValueError(
                f"index at {index_dir} was written by format version "
                f"{meta.get('format_version')}, this code reads "
                f"{_INDEX_FORMAT_VERSION}; rebuild it."
            )
        index.N = meta["num_docs"]
        index.avg_doc_len = meta["avg_doc_len"]
        index.remove_stopwords = meta["remove_stopwords"]
        index.stemming = meta["stemming"]
        index.forward_terms_per_doc = meta.get("forward_terms_per_doc", 0)
        n_terms = meta["num_terms"]
        chunks = meta["chunks"]

        def stream(name: str) -> bytes:
            return _decompress_stream(os.path.join(index_dir, name), chunks[name])

        terms_blob = stream("terms.bin")
        index.terms = terms_blob.decode("utf-8").split("\n") if terms_blob else []
        docids_blob = stream("docids.bin")
        index.doc_ids = docids_blob.decode("utf-8").split("\n") if docids_blob else []

        stats = codecs.vbyte_decode(stream("stats.bin"))
        index.df = stats[:n_terms]
        index.doc_len = stats[n_terms : n_terms + index.N]
        fwd_counts = stats[n_terms + index.N :]

        index._starts = np.concatenate(([0], np.cumsum(index.df))).astype(np.int64)
        index._doc_ids, index._freqs = _decode_gap_streams(
            stream("postings.bin"), stream("postings_tf.bin"), index.df
        )

        index._fwd_starts = np.concatenate(([0], np.cumsum(fwd_counts))).astype(np.int64)
        fwd_ranks, fwd_freqs = _decode_gap_streams(
            stream("fwd.bin"), stream("fwd_tf.bin"), fwd_counts
        )
        term_of_rank = np.argsort(-np.asarray(index.df, dtype=np.int64), kind="stable")
        index._fwd_terms, index._fwd_freqs = _resort_within_lists(
            term_of_rank[fwd_ranks], fwd_freqs, index._fwd_starts
        )
        return index


_EMPTY = np.zeros(0, dtype=np.int32)


def _np_to_array(a: np.ndarray):
    """int32 ndarray -> bytes acceptable to np.frombuffer in _invert/_build_forward."""
    return np.ascontiguousarray(a, dtype=np.int32).tobytes()


def _rank_of_term(df) -> np.ndarray:
    """rank_of_term[t] = position of term t when terms are sorted by
    descending df (stable, so ties keep lexicographic order). Deterministic
    given df, which is why it never has to be stored."""
    term_of_rank = np.argsort(-np.asarray(df, dtype=np.int64), kind="stable")
    rank_of_term = np.empty_like(term_of_rank)
    rank_of_term[term_of_rank] = np.arange(term_of_rank.size, dtype=np.int64)
    return rank_of_term


def _resort_within_lists(ids, freqs, starts):
    """Sort each list `ids[starts[i]:starts[i+1]]` ascending, carrying `freqs`
    along. One global argsort on (list index, id) does every list at once."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return _EMPTY, _EMPTY
    counts = np.diff(np.asarray(starts, dtype=np.int64))
    list_of_entry = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    order = np.argsort(list_of_entry * (int(ids.max()) + 1) + ids, kind="stable")
    return ids[order].astype(np.int32), np.asarray(freqs)[order].astype(np.int32)


# ----------------------------------------------------------------------
# Stream coding: (id, tf) lists -> gap stream with tf flag + tf>1 stream
# ----------------------------------------------------------------------
def _encode_gap_streams(ids, freqs, starts):
    """Encode many concatenated ascending id lists into two VByte streams.

    `ids[starts[i]:starts[i+1]]` is list i (ascending). Output:
      gap stream: for every entry, (gap << 1) | (tf > 1), where gap is the
                  difference from the previous entry of the *same* list, or
                  the absolute id for a list's first entry;
      tf stream:  tf - 2 for every entry with tf > 1, in order.
    """
    starts = np.asarray(starts, dtype=np.int64)
    if len(ids) == 0:
        return b"", b""

    # Encode a run of whole lists at a time. The vectorised encoder allocates
    # several int64 temporaries the size of its input, so encoding 12M
    # postings in one call needs ~1 GB of scratch; bounding each call to
    # ~2M entries bounds the scratch instead, at no cost to the output — the
    # streams are plain concatenations and every chunk ends on a list
    # boundary.
    gap_parts, tf_parts = [], []
    n_lists = starts.size - 1
    list_lo = 0
    while list_lo < n_lists:
        list_hi = int(np.searchsorted(starts, starts[list_lo] + _ENCODE_CHUNK_ENTRIES, side="right"))
        list_hi = min(max(list_hi, list_lo + 1), n_lists)
        lo, hi = int(starts[list_lo]), int(starts[list_hi])
        if hi > lo:
            chunk_ids = np.asarray(ids[lo:hi], dtype=np.int64)
            chunk_freqs = np.asarray(freqs[lo:hi], dtype=np.int64)
            chunk_starts = starts[list_lo : list_hi + 1] - lo

            gaps = chunk_ids.copy()
            gaps[1:] -= chunk_ids[:-1]
            first_of_list = chunk_starts[:-1][np.diff(chunk_starts) > 0]
            gaps[first_of_list] = chunk_ids[first_of_list]
            del chunk_ids

            has_tf = chunk_freqs > 1
            gaps <<= 1
            gaps |= has_tf
            gap_parts.append(codecs.vbyte_encode(gaps))
            del gaps
            tf_parts.append(codecs.vbyte_encode(chunk_freqs[has_tf] - 2))
        list_lo = list_hi
    return b"".join(gap_parts), b"".join(tf_parts)


def _decode_gap_streams(gap_stream: bytes, tf_stream: bytes, counts):
    """Inverse of `_encode_gap_streams`. `counts[i]` is the length of list i.

    Absolute ids come back from the gaps with one global cumulative sum minus,
    per entry, the running total that stood at the start of its own list —
    no Python loop over lists.
    """
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return _EMPTY, _EMPTY

    flagged = _vbyte_decode_chunked(gap_stream)
    if flagged.size != total:
        raise ValueError(f"gap stream holds {flagged.size} entries, expected {total}")
    has_tf = (flagged & 1).astype(bool)
    gaps = flagged >> 1

    del flagged
    running = np.cumsum(gaps)
    list_starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    nonempty = counts > 0
    base = np.zeros(counts.size, dtype=np.int64)
    base[nonempty] = running[list_starts[nonempty]] - gaps[list_starts[nonempty]]
    del gaps
    running -= np.repeat(base, counts)
    ids = running.astype(np.int32)
    del running

    freqs = np.ones(total, dtype=np.int32)
    if has_tf.any():
        freqs[has_tf] = (_vbyte_decode_chunked(tf_stream) + 2).astype(np.int32)
    return ids, freqs


def _vbyte_decode_chunked(stream: bytes, chunk_bytes: int = 4 * 1024 * 1024) -> np.ndarray:
    """`codecs.vbyte_decode` over a long stream, a few MB at a time.

    The vectorised decoder allocates several int64 arrays as long as its
    input in *bytes*, so decoding a 19 MB stream in one call peaks at
    ~700 MB. Splitting the stream is safe as long as every cut falls just
    after a byte with the continuation bit set (the last byte of a number),
    which is what the search below arranges.
    """
    if len(stream) <= chunk_bytes:
        return codecs.vbyte_decode(stream)
    buf = np.frombuffer(stream, dtype=np.uint8)
    parts = []
    offset = 0
    while offset < buf.size:
        end = min(offset + chunk_bytes, buf.size)
        if end < buf.size:
            # Back up to the most recent number boundary.
            while end > offset and buf[end - 1] < 128:
                end -= 1
            if end == offset:  # pragma: no cover - a >4 MB single number
                end = min(offset + chunk_bytes, buf.size)
        parts.append(codecs.vbyte_decode(stream, offset, end - offset))
        offset = end
    return np.concatenate(parts)


# ----------------------------------------------------------------------
# Chunked, parallel LZMA
# ----------------------------------------------------------------------
def _lzma_chunk(chunk: bytes) -> bytes:
    return lzma.compress(chunk, preset=_LZMA_PRESET)


def _compress_streams(streams: Dict[str, bytes], index_dir: str) -> Dict[str, List[int]]:
    """LZMA-compress every stream, in chunks, across a process pool, and
    write each as one file. Returns {filename: [compressed chunk sizes]},
    which is all `load()` needs to split the file back into chunks."""
    jobs = []  # (filename, chunk index, raw bytes)
    for name, blob in streams.items():
        if not blob:
            jobs.append((name, 0, b""))
            continue
        for i, offset in enumerate(range(0, len(blob), _COMPRESS_CHUNK_BYTES)):
            jobs.append((name, i, blob[offset : offset + _COMPRESS_CHUNK_BYTES]))

    compressed = _run_parallel(_lzma_chunk, [raw for _n, _i, raw in jobs])

    table: Dict[str, List[int]] = {name: [] for name in streams}
    handles = {name: open(os.path.join(index_dir, name), "wb") for name in streams}
    try:
        for (name, _i, _raw), blob in zip(jobs, compressed):
            handles[name].write(blob)
            table[name].append(len(blob))
    finally:
        for h in handles.values():
            h.close()
    return table


def _run_parallel(fn, items: List[bytes]) -> List[bytes]:
    """Map `fn` over `items` on a thread pool.

    Threads, not processes, on purpose: `lzma.compress` releases the GIL for
    the duration of the call, so threads get genuine multi-core parallelism
    here, with none of the hazards of a process pool (re-importing the main
    module under macOS/Windows spawn, pickling, sandbox restrictions). Falls
    back to serial when there is not enough work to share out.
    """
    big = sum(1 for it in items if len(it) > 256 * 1024)
    workers = min(4, os.cpu_count() or 1)
    if big < 2 or workers < 2:
        return [fn(it) for it in items]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def _decompress_stream(path: str, chunk_sizes: Sequence[int]) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    parts = []
    offset = 0
    for size in chunk_sizes:
        if size:
            parts.append(lzma.decompress(data[offset : offset + size]))
        offset += size
    return b"".join(parts)
