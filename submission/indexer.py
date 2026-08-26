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
access applies here too, and a sorted list of 600K strings costs a fraction of
the memory (and, more importantly for our score, a fraction of the *load
time*) of the equivalent Python dict.

On disk, `index_dir` holds five files:

    meta.json      collection statistics and the analyser settings the index
                   was built with, so load() can reproduce them exactly
    terms.bin      zlib(sorted term strings, newline-joined)
    termstats.bin  zlib(VByte: df per term, then postings byte-length per term)
    postings.bin   VByte d-gaps + term frequencies, one contiguous run per
                   term, in term-id order. NOT compressed as a whole: it is
                   memory-mapped and read a slice at a time, so it has to stay
                   randomly accessible.
    doclen.bin     zlib(VByte document lengths, in internal doc-id order)
    docids.bin     zlib(external doc_id strings, newline-joined)
    fwd.bin        VByte term-id gaps + frequencies, one run per document —
                   a *pruned* forward index (see below)
    fwdlen.bin     zlib(VByte per-document byte lengths into fwd.bin)

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
index stops buying anything: 48 terms per document costs 8 MB more on disk for
no ranking gain at all.

Three things are deliberately NOT persisted, because the index-size component
(Section 7) charges for every byte and `retrieve()` never needs them:
  - the raw document text (the starter skeleton's `doc_text` field): BM25 and
    cosine need only term frequencies and lengths;
  - a *full* forward index, for the reason above;
  - precomputed IDF values or document norms, which are cheap to recompute
    from what *is* stored, and are therefore pure redundancy on disk.
"""
import json
import mmap
import os
import zlib
from bisect import bisect_left
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from submission import codecs
from submission.porter import stem as porter_stem

try:
    import numpy as np

    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    _HAVE_NUMPY = False

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

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

_INDEX_FORMAT_VERSION = 4

# How many of each document's most frequent terms the forward index keeps.
# 0 disables the forward index entirely (and with it relevance feedback).
DEFAULT_FORWARD_TERMS_PER_DOC = 24


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


class InvertedIndex:
    """An inverted index with VByte-compressed postings.

    After `build()` (build-time shape) the postings live in Python lists; after
    `load()` (query-time shape) they live memory-mapped in `postings.bin` and
    are decoded per term on demand. Both shapes answer the same three
    questions — `document_frequency(term)`, `postings(term)`, and the
    collection statistics — so the scorers do not care which one they hold.
    """

    def __init__(self):
        # Dictionary: `terms` is sorted; a term's id is its position in it.
        self.terms: List[str] = []
        self.df: Sequence[int] = []           # df[term_id]
        self.doc_len: Sequence[int] = []      # doc_len[doc_id], in index terms
        self.doc_ids: List[str] = []          # external doc_id per internal id
        self.N: int = 0                       # number of documents
        self.avg_doc_len: float = 0.0
        self.remove_stopwords: bool = True
        self.stemming: bool = True

        # Query-time postings storage (populated by load()).
        self._postings_mm: Optional[mmap.mmap] = None
        self._postings_file = None
        self._offsets: Sequence[int] = []     # byte offset of term_id's list
        self._lengths: Sequence[int] = []     # byte length of term_id's list

        # Build-time postings storage (populated by build()).
        self._build_doc_ids = None            # int32 array, sorted by term
        self._build_freqs = None              # int32 array, aligned with above
        self._build_starts = None             # first index of term_id's slice

        # Pruned forward index (doc -> its most frequent terms). Used only by
        # relevance feedback; see the module docstring.
        self.forward_terms_per_doc: int = 0
        self._fwd_terms = None                # term ids, doc-major, ascending
        self._fwd_freqs = None                # aligned frequencies
        self._fwd_counts = None               # terms kept per doc
        self._fwd_mm = None                   # query-time: mmap of fwd.bin
        self._fwd_file = None
        self._fwd_offsets = None
        self._fwd_lengths = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def build(
        self,
        corpus,
        remove_stopwords: bool = True,
        stemming: bool = True,
        forward_terms_per_doc: int = DEFAULT_FORWARD_TERMS_PER_DOC,
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
                # avoids a second sort of all ~20M postings into doc-major
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
        self.doc_len = doc_len
        self.N = len(doc_ids)
        total_tokens = sum(doc_len)
        self.avg_doc_len = (total_tokens / self.N) if self.N else 0.0

        # Term ids were handed out in first-seen order; re-map them to
        # lexicographic order so the dictionary can be a sorted array (binary
        # searchable, and far more compressible — adjacent sorted terms share
        # long prefixes, which is most of why terms.bin is small).
        self.terms = sorted(term_to_id)
        rank_of_old_id = [0] * len(self.terms)
        for new_id, term in enumerate(self.terms):
            rank_of_old_id[term_to_id[term]] = new_id

        if forward_terms_per_doc:
            self._build_forward(fwd_terms, fwd_freqs, fwd_counts, rank_of_old_id)
        del fwd_terms, fwd_freqs, fwd_counts

        self._invert(posting_terms, posting_docs, posting_freqs, rank_of_old_id)
        # The document-ordered triples are dead once inverted; on a large
        # collection they are hundreds of megabytes, and save() is about to
        # want that space for encoding scratch.
        del posting_terms, posting_docs, posting_freqs

    def _build_forward(self, fwd_terms, fwd_freqs, fwd_counts, rank_of_old_id) -> None:
        """Re-map the forward index onto lexicographic term ids and restore
        ascending order within each document.

        The terms were collected under first-seen ids, so re-mapping scrambles
        the within-document ordering that gap coding depends on. Re-sorting is
        done per document with one vectorised trick: add the document index
        times (number of terms + 1) to each term id, so a single global sort
        orders by document first and term id second.
        """
        if not _HAVE_NUMPY:  # pragma: no cover
            counts = list(fwd_counts)
            terms, freqs, position = [], [], 0
            for count in counts:
                pairs = sorted(
                    (rank_of_old_id[fwd_terms[position + j]], fwd_freqs[position + j])
                    for j in range(count)
                )
                terms.extend(t for t, _f in pairs)
                freqs.extend(f for _t, f in pairs)
                position += count
            self._fwd_terms, self._fwd_freqs, self._fwd_counts = terms, freqs, counts
            return

        counts = np.frombuffer(fwd_counts, dtype=np.int32).astype(np.int64)
        terms = np.asarray(np.frombuffer(fwd_terms, dtype=np.int32), dtype=np.int64)
        freqs = np.asarray(np.frombuffer(fwd_freqs, dtype=np.int32), dtype=np.int64)
        remap = np.asarray(rank_of_old_id, dtype=np.int64)
        terms = remap[terms]

        n_terms = len(self.terms) + 1
        doc_of_entry = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
        order = np.argsort(doc_of_entry * n_terms + terms, kind="stable")
        self._fwd_terms = terms[order]
        self._fwd_freqs = freqs[order]
        self._fwd_counts = counts

    def _invert(self, posting_terms, posting_docs, posting_freqs, rank_of_old_id) -> None:
        """Sort the (term, doc, tf) triples by term and record where each
        term's slice begins."""
        n_terms = len(self.terms)
        if not _HAVE_NUMPY:
            self._invert_py(posting_terms, posting_docs, posting_freqs, rank_of_old_id)
            return

        terms_np = np.frombuffer(posting_terms, dtype=np.int32)
        remap = np.asarray(rank_of_old_id, dtype=np.int32)
        terms_np = remap[terms_np] if terms_np.size else terms_np.astype(np.int32)

        # Stable sort on the term key alone: within a term, the triples stay in
        # the document order they were emitted in, i.e. ascending doc id.
        order = np.argsort(terms_np, kind="stable")
        sorted_terms = terms_np[order]
        del terms_np
        self._build_doc_ids = np.frombuffer(posting_docs, dtype=np.int32)[order]
        self._build_freqs = np.frombuffer(posting_freqs, dtype=np.int32)[order]
        del order

        # starts[t] = index of term t's first posting; searchsorted gives all
        # of them in one pass, and also handles terms with no postings.
        self._build_starts = np.searchsorted(sorted_terms, np.arange(n_terms + 1))
        self.df = np.diff(self._build_starts)

    def _invert_py(self, posting_terms, posting_docs, posting_freqs, rank_of_old_id) -> None:  # pragma: no cover
        """NumPy-free fallback for `_invert`."""
        n_terms = len(self.terms)
        triples = sorted(
            (rank_of_old_id[t], d, f)
            for t, d, f in zip(posting_terms, posting_docs, posting_freqs)
        )
        self._build_doc_ids = [d for _t, d, _f in triples]
        self._build_freqs = [f for _t, _d, f in triples]
        starts = [0] * (n_terms + 1)
        counts = [0] * n_terms
        for t, _d, _f in triples:
            counts[t] += 1
        running = 0
        for t in range(n_terms):
            starts[t] = running
            running += counts[t]
        starts[n_terms] = running
        self._build_starts = starts
        self.df = counts

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

        At query time this decodes the term's VByte slice out of the
        memory-mapped postings file; only the slices for the query's own terms
        are ever touched, which is the whole point of an inverted index.
        """
        if term_id < 0:
            return _empty_pair()
        if self._postings_mm is not None:
            offset = int(self._offsets[term_id])
            length = int(self._lengths[term_id])
            if length == 0:
                return _empty_pair()
            return codecs.decode_postings(self._postings_mm, offset, length)
        # Build-time shape: slice the sorted triples directly.
        start = int(self._build_starts[term_id])
        end = int(self._build_starts[term_id + 1])
        return self._build_doc_ids[start:end], self._build_freqs[start:end]

    def postings_for(self, term: str):
        """`postings()` keyed by an index term string."""
        return self.postings(self.term_id(term))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, index_dir: str) -> None:
        """Write the index to `index_dir` in the compact format documented at
        the top of this module.

        Everything expensive happens in three vectorised passes over the whole
        posting array rather than a Python loop over the ~600K terms: compute
        d-gaps, interleave with term frequencies, VByte-encode the lot in a
        single call. The per-term byte extents needed by `load()` come from
        `vbyte_byte_widths` summed per term with `reduceat`, so no term is ever
        encoded individually.
        """
        os.makedirs(index_dir, exist_ok=True)

        postings_blob, lengths = self._encode_all_postings()

        with open(os.path.join(index_dir, "postings.bin"), "wb") as f:
            f.write(postings_blob)

        # Dictionary. Sorted terms share long prefixes with their neighbours,
        # so zlib on the newline-joined block gets most of what explicit front
        # coding (Retrieval-I) would, for a fraction of the code.
        terms_blob = "\n".join(self.terms).encode("utf-8")
        with open(os.path.join(index_dir, "terms.bin"), "wb") as f:
            f.write(zlib.compress(terms_blob, 9))

        # Per-term statistics: df first, then postings byte length. Both are
        # small, heavily skewed integers, so VByte then zlib is very effective.
        stats = list(self.df) + list(lengths)
        with open(os.path.join(index_dir, "termstats.bin"), "wb") as f:
            f.write(zlib.compress(codecs.vbyte_encode(stats), 9))

        # Document table: lengths (VByte, small values) and the external
        # doc_id strings, which retrieve() needs only to name its top k.
        with open(os.path.join(index_dir, "doclen.bin"), "wb") as f:
            f.write(zlib.compress(codecs.vbyte_encode(list(self.doc_len)), 9))
        # docids.bin is newline-delimited, so a doc_id containing a newline
        # would silently split into two and desynchronise every id after it.
        # Fail loudly at build time instead of returning wrong doc_ids at query
        # time. (Index terms can't hit this: they are [a-z0-9]+ by construction.)
        if any("\n" in doc_id for doc_id in self.doc_ids):
            raise ValueError(
                "a doc_id in the corpus contains a newline, which the index's "
                "doc-id table cannot represent"
            )
        with open(os.path.join(index_dir, "docids.bin"), "wb") as f:
            f.write(zlib.compress("\n".join(self.doc_ids).encode("utf-8"), 9))

        if self.forward_terms_per_doc and self._fwd_counts is not None:
            fwd_blob, fwd_lengths = self._encode_forward()
            with open(os.path.join(index_dir, "fwd.bin"), "wb") as f:
                f.write(fwd_blob)
            with open(os.path.join(index_dir, "fwdlen.bin"), "wb") as f:
                f.write(zlib.compress(codecs.vbyte_encode(list(fwd_lengths)), 9))

        meta = {
            "format_version": _INDEX_FORMAT_VERSION,
            "num_docs": self.N,
            "num_terms": len(self.terms),
            "avg_doc_len": self.avg_doc_len,
            "remove_stopwords": self.remove_stopwords,
            "stemming": self.stemming,
            "forward_terms_per_doc": self.forward_terms_per_doc,
        }
        with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, separators=(",", ":"))

    # How many postings to VByte-encode at a time. Encoding is vectorised, and
    # a vectorised encoder allocates several int64 temporaries the size of its
    # input — doing the whole collection in one call would need gigabytes of
    # scratch space on a large corpus. Chunking bounds that scratch to a fixed
    # cost regardless of collection size, at no measurable speed penalty (the
    # chunks are still millions of values wide). Chunk boundaries always fall
    # on term boundaries, so each term's postings stay contiguous.
    _ENCODE_CHUNK_POSTINGS = 2_000_000

    def _encode_all_postings(self):
        """VByte-encode every postings list into one blob; return it together
        with each term's byte length."""
        if not _HAVE_NUMPY:  # pragma: no cover
            blob = bytearray()
            lengths = []
            for tid in range(len(self.terms)):
                doc_ids, freqs = self.postings(tid)
                encoded = codecs.encode_postings(doc_ids, freqs)
                lengths.append(len(encoded))
                blob.extend(encoded)
            return bytes(blob), lengths

        starts = np.asarray(self._build_starts, dtype=np.int64)
        n_terms = len(self.terms)
        lengths = np.zeros(n_terms, dtype=np.int64)
        chunks = []

        term_lo = 0
        while term_lo < n_terms:
            # Take as many whole terms as fit in one chunk (always at least
            # one, so a single term with a huge postings list still works).
            budget = starts[term_lo] + self._ENCODE_CHUNK_POSTINGS
            term_hi = int(np.searchsorted(starts, budget, side="right"))
            term_hi = min(max(term_hi, term_lo + 1), n_terms)

            lo, hi = int(starts[term_lo]), int(starts[term_hi])
            if hi > lo:
                chunk_starts = starts[term_lo : term_hi + 1] - lo
                doc_ids = np.asarray(self._build_doc_ids[lo:hi], dtype=np.int64)

                # d-gaps: difference against the previous posting, except at
                # the start of each term's list, where the absolute doc id is
                # stored.
                gaps = doc_ids.copy()
                gaps[1:] -= doc_ids[:-1]
                first_of_term = chunk_starts[:-1][np.diff(chunk_starts) > 0]
                gaps[first_of_term] = doc_ids[first_of_term]

                interleaved = np.empty(doc_ids.size * 2, dtype=np.int64)
                interleaved[0::2] = gaps
                interleaved[1::2] = np.asarray(self._build_freqs[lo:hi], dtype=np.int64)
                del doc_ids, gaps

                # Each term occupies postings [starts[t], starts[t+1]), i.e.
                # values [2*starts[t], 2*starts[t+1]) of the interleaved stream.
                cumulative = np.concatenate(([0], np.cumsum(codecs.vbyte_byte_widths(interleaved))))
                lengths[term_lo:term_hi] = np.diff(cumulative[2 * chunk_starts])
                del cumulative

                chunks.append(codecs.vbyte_encode(interleaved))
                del interleaved

            term_lo = term_hi

        return b"".join(chunks), lengths

    def _encode_forward(self):
        """VByte-encode the pruned forward index: per document, term-id gaps
        interleaved with frequencies. Same shape as `_encode_all_postings`,
        with the roles of term and document swapped."""
        if not _HAVE_NUMPY:  # pragma: no cover
            blob, lengths, position = bytearray(), [], 0
            for count in self._fwd_counts:
                chunk = codecs.encode_postings(
                    self._fwd_terms[position : position + count],
                    self._fwd_freqs[position : position + count],
                )
                lengths.append(len(chunk))
                blob.extend(chunk)
                position += count
            return bytes(blob), lengths

        counts = np.asarray(self._fwd_counts, dtype=np.int64)
        starts = np.concatenate(([0], np.cumsum(counts)))
        n_docs = counts.size
        lengths = np.zeros(n_docs, dtype=np.int64)
        chunks = []

        doc_lo = 0
        while doc_lo < n_docs:
            budget = starts[doc_lo] + self._ENCODE_CHUNK_POSTINGS
            doc_hi = int(np.searchsorted(starts, budget, side="right"))
            doc_hi = min(max(doc_hi, doc_lo + 1), n_docs)

            lo, hi = int(starts[doc_lo]), int(starts[doc_hi])
            if hi > lo:
                chunk_starts = starts[doc_lo : doc_hi + 1] - lo
                terms = np.asarray(self._fwd_terms[lo:hi], dtype=np.int64)

                gaps = terms.copy()
                gaps[1:] -= terms[:-1]
                first_of_doc = chunk_starts[:-1][np.diff(chunk_starts) > 0]
                gaps[first_of_doc] = terms[first_of_doc]

                interleaved = np.empty(terms.size * 2, dtype=np.int64)
                interleaved[0::2] = gaps
                interleaved[1::2] = np.asarray(self._fwd_freqs[lo:hi], dtype=np.int64)
                del terms, gaps

                cumulative = np.concatenate(([0], np.cumsum(codecs.vbyte_byte_widths(interleaved))))
                lengths[doc_lo:doc_hi] = np.diff(cumulative[2 * chunk_starts])
                del cumulative

                chunks.append(codecs.vbyte_encode(interleaved))
                del interleaved

            doc_lo = doc_hi

        return b"".join(chunks), lengths

    def forward(self, doc_id: int):
        """Return (term_ids, frequencies) for `doc_id`'s most frequent terms.

        This is the pruned forward index — the top `forward_terms_per_doc`
        terms of the document, ascending by term id. Returns empty arrays if
        the index was built without a forward index.
        """
        if self._fwd_mm is not None:
            length = int(self._fwd_lengths[doc_id])
            if length == 0:
                return _empty_pair()
            return codecs.decode_postings(self._fwd_mm, int(self._fwd_offsets[doc_id]), length)
        if self._fwd_counts is None:
            return _empty_pair()
        starts = np.concatenate(([0], np.cumsum(np.asarray(self._fwd_counts, dtype=np.int64))))
        lo, hi = int(starts[doc_id]), int(starts[doc_id + 1])
        return self._fwd_terms[lo:hi], self._fwd_freqs[lo:hi]

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """Reconstruct the index from `index_dir` alone, in a fresh process.

        Deliberately lazy about the largest file: `postings.bin` is memory
        mapped, not read. Nothing is decoded until a query asks for a specific
        term, which keeps index-load time to the cost of the dictionary and
        document table (tens of milliseconds) instead of the cost of the whole
        collection.
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
        n_terms = meta["num_terms"]

        with open(os.path.join(index_dir, "terms.bin"), "rb") as f:
            terms_blob = zlib.decompress(f.read())
        index.terms = terms_blob.decode("utf-8").split("\n") if terms_blob else []

        with open(os.path.join(index_dir, "termstats.bin"), "rb") as f:
            stats = codecs.vbyte_decode(zlib.decompress(f.read()))
        index.df = stats[:n_terms]
        lengths = stats[n_terms:]
        if _HAVE_NUMPY:
            index._lengths = lengths
            index._offsets = np.concatenate(([0], np.cumsum(lengths)[:-1])) if n_terms else lengths
        else:  # pragma: no cover
            index._lengths = list(lengths)
            offsets, running = [], 0
            for length in index._lengths:
                offsets.append(running)
                running += length
            index._offsets = offsets

        with open(os.path.join(index_dir, "doclen.bin"), "rb") as f:
            index.doc_len = codecs.vbyte_decode(zlib.decompress(f.read()))
        with open(os.path.join(index_dir, "docids.bin"), "rb") as f:
            doc_ids_blob = zlib.decompress(f.read())
        index.doc_ids = doc_ids_blob.decode("utf-8").split("\n") if doc_ids_blob else []

        index.forward_terms_per_doc = meta.get("forward_terms_per_doc", 0)
        fwd_path = os.path.join(index_dir, "fwd.bin")
        if index.forward_terms_per_doc and os.path.exists(fwd_path) and os.path.getsize(fwd_path) > 0:
            with open(os.path.join(index_dir, "fwdlen.bin"), "rb") as f:
                fwd_lengths = codecs.vbyte_decode(zlib.decompress(f.read()))
            index._fwd_lengths = fwd_lengths
            if _HAVE_NUMPY:
                index._fwd_offsets = np.concatenate(([0], np.cumsum(fwd_lengths)[:-1]))
            else:  # pragma: no cover
                offsets, running = [], 0
                for length in fwd_lengths:
                    offsets.append(running)
                    running += length
                index._fwd_offsets = offsets
            index._fwd_file = open(fwd_path, "rb")
            index._fwd_mm = mmap.mmap(index._fwd_file.fileno(), 0, access=mmap.ACCESS_READ)

        postings_path = os.path.join(index_dir, "postings.bin")
        if os.path.getsize(postings_path) > 0:
            index._postings_file = open(postings_path, "rb")
            index._postings_mm = mmap.mmap(
                index._postings_file.fileno(), 0, access=mmap.ACCESS_READ
            )
        else:
            index._postings_mm = b""
            index._offsets = [0] * n_terms
            index._lengths = [0] * n_terms

        return index

    def iter_postings_blocks(self, terms_per_block: int = 8192):
        """Sweep the whole index, yielding (term_ids, doc_ids, freqs) arrays a
        block of terms at a time.

        Scoring a query only ever touches a handful of postings lists, so this
        is not on the hot path — it exists for the operations that genuinely
        need every posting, chiefly the vector-space document norms in
        `boolean_vsm.py`. Going term by term would mean one Python call per
        term (600K of them); decoding a contiguous *block* of terms in a single
        vectorised call instead keeps the sweep to ~100 calls while bounding
        peak memory, which matters on the 8 GB grading machine.

        The trick for recovering absolute doc ids from d-gaps in bulk: take one
        global cumulative sum over the block, then subtract, per posting, the
        running total that had accumulated before its own term's list started.
        """
        n_terms = len(self.terms)
        if n_terms == 0:
            return
        if not _HAVE_NUMPY:  # pragma: no cover
            for term_id in range(n_terms):
                doc_ids, freqs = self.postings(term_id)
                yield [term_id] * len(doc_ids), doc_ids, freqs
            return

        df = np.asarray(self.df, dtype=np.int64)
        for first in range(0, n_terms, terms_per_block):
            last = min(first + terms_per_block, n_terms)
            block_df = df[first:last]
            if block_df.sum() == 0:
                continue

            if self._postings_mm is not None:
                offset = int(self._offsets[first])
                length = int(np.sum(np.asarray(self._lengths[first:last], dtype=np.int64)))
                flat = codecs.vbyte_decode(self._postings_mm, offset, length)
                gaps, freqs = flat[0::2], flat[1::2]
                # Per-term cumulative sums, done as one global cumsum minus the
                # total standing at the start of each term's own list.
                running = np.cumsum(gaps)
                starts = np.concatenate(([0], np.cumsum(block_df)[:-1]))
                base = running[starts] - gaps[starts]
                term_ids = np.repeat(np.arange(first, last, dtype=np.int64), block_df)
                doc_ids = running - np.repeat(base, block_df)
            else:
                lo = int(self._build_starts[first])
                hi = int(self._build_starts[last])
                doc_ids = np.asarray(self._build_doc_ids[lo:hi], dtype=np.int64)
                freqs = np.asarray(self._build_freqs[lo:hi], dtype=np.int64)
                term_ids = np.repeat(np.arange(first, last, dtype=np.int64), block_df)

            yield term_ids, doc_ids, freqs

    def close(self) -> None:
        """Release the memory map. Not needed by the harness (the process
        exits), but keeps the unit tests from leaking file handles."""
        if isinstance(self._postings_mm, mmap.mmap):
            self._postings_mm.close()
        if self._postings_file is not None:
            self._postings_file.close()
        self._postings_mm = None
        self._postings_file = None
        if isinstance(self._fwd_mm, mmap.mmap):
            self._fwd_mm.close()
        if self._fwd_file is not None:
            self._fwd_file.close()
        self._fwd_mm = None
        self._fwd_file = None


def _empty_pair():
    if _HAVE_NUMPY:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return [], []
