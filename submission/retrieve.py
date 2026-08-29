"""
submission/retrieve.py — THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5,
"Submission Interface & Conformance Checking") — do not rename them, change
their signatures, or move them out of this file.

    build_index(corpus_path, index_dir) -> None
        Runs once, in its own process. Analyses the corpus, inverts it, and
        writes the index to `index_dir`. Timed as index build time; the
        resulting on-disk size is its own graded component.

    load_index(index_dir) -> None
        Runs once, in a FRESH process, reading `index_dir` and nothing else.
        Memory-maps the postings and precomputes the query-independent
        scoring tables. Timed as index load time.

    retrieve(query, k) -> list[(doc_id, score)]
        Runs once per query, after load_index(). Returns up to k results,
        sorted by score descending, no doc_id repeated.

How the pieces fit together
---------------------------
    corpus.jsonl
        |  indexer.analyze()      lowercase -> tokenise -> stopword -> Porter
        v
    InvertedIndex.build()         single-pass in-memory inversion
        |  InvertedIndex.save()   VByte d-gaps + tf flag, chunked parallel LZMA
        v
    index_dir/                    ~~~ process boundary ~~~
        |  InvertedIndex.load()   decompress + decode everything into RAM
        v
    bm25.build() / boolean_vsm.build() / custom_scorer.build()
        |
        v
    retrieve()  ->  the ranking the harness scores

The scorer `retrieve()` actually uses is selected by `_SCORER` below. All three
required retrievers stay wired up and callable regardless of which one is
selected, both because the assignment grades them as independent components
(Section 9, "Correctness of required components") and because the report's
comparison table needs to run all of them against the same index.
"""
import os
from typing import List, Optional, Tuple

from submission import bm25, boolean_vsm, custom_scorer
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex

# ---------------------------------------------------------------------------
# Tuned parameters.
#
# These are the values selected by the sweep described in the report, run on
# the released dev topics only — never on the held-out set, which we never see.
# The environment-variable overrides exist so that sweep can vary them without
# editing this file (and so a grader can reproduce a point on the curve); the
# harness sets none of them, so grading always uses the defaults below.
# ---------------------------------------------------------------------------
_SCORER = os.environ.get("A1_SCORER", "custom")
BM25_K1 = float(os.environ.get("A1_K1", "2.0"))
BM25_B = float(os.environ.get("A1_B", "0.6"))

# The custom scorer wraps the same BM25, so it must use the same operating
# point; setting it here keeps one source of truth for the tuned values.
custom_scorer.K1 = BM25_K1
custom_scorer.B = BM25_B

_INDEX: Optional[InvertedIndex] = None


def build_index(corpus_path: str, index_dir: str) -> None:
    """Analyse `corpus_path` and write a queryable index into `index_dir`.

    Everything expensive lives here by design: this call is charged against the
    index-build-time metric, while `retrieve()` is charged against per-query
    latency, and per-query latency is measured once per query rather than once
    per run. Anything that can be precomputed, is.

    The corpus is streamed rather than materialised as a list — on a 190 MB
    corpus.jsonl that is the difference between holding one document in memory
    and holding all 171K.
    """
    os.makedirs(index_dir, exist_ok=True)

    index = InvertedIndex()
    index.build(_stream_corpus(corpus_path))
    index.save(index_dir)


def load_index(index_dir: str) -> None:
    """Reconstruct everything `retrieve()` needs, reading only `index_dir`.

    Runs in a fresh process with no memory of `build_index()`. The whole index
    is decompressed and decoded into flat arrays here (~1 s), and BM25's
    per-posting partial scores are precomputed, so that a query is nothing but
    array slices and one multiply-add per posting. Load time is reported but
    not part of the efficiency score; query latency is — hence the trade.
    """
    global _INDEX
    _INDEX = InvertedIndex.load(index_dir)

    # Each scorer caches its own query-independent tables (IDF vectors, length
    # ratios). None of them are persisted: they are derivable from the index,
    # so writing them out would cost index-size score for nothing.
    bm25.build(_INDEX)
    boolean_vsm.build(_INDEX)
    custom_scorer.build(_INDEX)
    # Materialise the BM25 partial-score table for the tuned (k1, b) now,
    # inside load, so the first query does not pay for it.
    bm25.warm(BM25_K1, BM25_B)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, best first."""
    if _INDEX is None:
        raise RuntimeError(
            "retrieve() called before load_index(); the harness always "
            "calls build_index(corpus_path, index_dir) and then "
            "load_index(index_dir) — in that order, in two separate "
            "processes — before any retrieve() calls. If you're testing "
            "manually, do the same."
        )

    if _SCORER == "vsm":
        return boolean_vsm.vsm_score(query, k)
    if _SCORER == "custom":
        return custom_scorer.score(query, k)
    return bm25.score(query, k, k1=BM25_K1, b=BM25_B)


def _stream_corpus(corpus_path: str):
    """Yield (doc_id, text) pairs one at a time.

    `corpus_utils.load_corpus()` returns the whole corpus as a list, which is
    fine for the toy set and wasteful for the real one. The index only ever
    needs one document at a time, so stream it.
    """
    import json

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["doc_id"], obj["text"]
