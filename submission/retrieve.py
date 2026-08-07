"""
submission/retrieve.py — THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the two functions below.
Their names and signatures are fixed by the assignment (Section 5 of the
assignment spec, "Submission Interface & Conformance Checking") — do not
rename them, change their signatures, or move them out of this file.

    build_index(corpus_path: str) -> None
        Called exactly once, before any retrieve() calls, with the path
        to a corpus.jsonl file (one {"doc_id": ..., "text": ...} JSON
        object per line — see data/README.md). Build whatever index and
        statistics you need here. The harness times this call separately
        and reports it as your "index build time" efficiency metric.

    retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]
        Called once per query, only after build_index() has run. Return
        up to k (doc_id, score) pairs, sorted by score descending
        (highest score = most relevant). This is exactly the ranking the
        harness scores with nDCG@10 / MAP. doc_id values must be ones
        that appeared in the corpus passed to build_index().

This file ships with a trivial, fully-working baseline — return the first
k documents in collection order, ignoring the query entirely — wired up
below as `_baseline_retrieve`. It exists only so that the harness, the CI
conformance check, and the Docker image all work end-to-end from your
very first commit, before you have implemented anything. Its scores will
be close to zero; replace it with real logic.
"""
from typing import List, Optional, Tuple

from submission.corpus_utils import load_corpus

# TODO(you): once implemented, import and use your real scorers, e.g.:
# from submission import bm25, boolean_vsm, language_model, custom_scorer

# ---------------------------------------------------------------------------
# Module-level state. build_index() populates this; retrieve() reads it.
# Plain module globals (rather than a class) keep the required function
# signatures exactly as specified above.
# ---------------------------------------------------------------------------
_CORPUS: Optional[List[Tuple[str, str]]] = None  # [(doc_id, text), ...] in file order


def build_index(corpus_path: str) -> None:
    """Load the corpus and build whatever index structures you need.

    Called exactly once by the harness before any retrieve() calls.
    Heavy one-time work — tokenising the whole corpus, building postings
    lists, computing collection statistics — belongs here, not in
    retrieve(), so it doesn't get charged against your per-query latency.
    """
    global _CORPUS
    _CORPUS = load_corpus(corpus_path)

    # TODO(you): build your inverted index / term statistics here, e.g.:
    #
    #   from submission.indexer import InvertedIndex
    #   index = InvertedIndex()
    #   index.build(_CORPUS)
    #   bm25.build(index)
    #   boolean_vsm.build(index)
    #   language_model.build(index)
    #
    # and store `index` (or whatever you need) in a module-level variable
    # so retrieve() can use it.


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, best first."""
    if _CORPUS is None:
        raise RuntimeError(
            "retrieve() called before build_index(); the harness always "
            "calls build_index(corpus_path) first — if you're testing "
            "manually, call it yourself first too."
        )

    # TODO(you): replace this with a real scorer, e.g.:
    #   return bm25.score(query, k, k1=1.2, b=0.75)
    return _baseline_retrieve(query, k)


# ---------------------------------------------------------------------------
# Trivial reference baseline — DO NOT submit this as your final entry.
# Ignores the query and returns the first k documents in collection order
# with a dummy descending score. Enough to exercise the full harness;
# metrics against it will legitimately be close to zero.
# ---------------------------------------------------------------------------
def _baseline_retrieve(query: str, k: int) -> List[Tuple[str, float]]:
    assert _CORPUS is not None
    top = _CORPUS[:k]
    return [(doc_id, float(len(top) - i)) for i, (doc_id, _text) in enumerate(top)]
