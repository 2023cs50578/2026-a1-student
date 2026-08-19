"""
submission/boolean_vsm.py — Boolean retrieval + vector-space ranking.

Required component (assignment Section 4.1): "supports conjunctive/
disjunctive Boolean queries and a cosine-similarity vector-space ranking
with a TF-IDF weighting scheme of your choice."

Two independent pieces to implement:

1. Boolean retrieval: given a query, treat it as an AND (conjunctive) or
   OR (disjunctive) combination of terms and return the matching document
   set — no ranking, just set membership. Useful as a fast candidate
   filter and as a sanity check ("does my index even find the right
   documents for this query?").

2. Vector-space ranking: represent the query and each candidate document
   as TF-IDF weighted vectors and rank by cosine similarity. A standard
   TF-IDF weight for term t in document d:

       w(t, d) = tf(t, d) * log( N / df(t) )

   (log base is your choice — just be consistent), and cosine similarity
   between query vector q and document vector d:

       sim(q, d) = (q . d) / (||q|| * ||d||)

Both pieces should read from the same InvertedIndex you build in
indexer.py.
"""
from typing import List, Tuple

from submission.indexer import InvertedIndex


def build(index: InvertedIndex) -> None:
    """Optional: precompute anything VSM-specific (e.g. document vector
    norms) from the InvertedIndex built in indexer.py.

    Call this from retrieve.load_index(), not retrieve.build_index() —
    the harness runs those two in separate processes, so any cache this
    creates only needs to exist in the process that also calls
    retrieve(). If you want a precomputed cache to persist across the
    build/load boundary too, write it out via InvertedIndex.save() instead
    (it then counts toward your index-size score) and rebuild the cache
    here from the loaded index."""
    raise NotImplementedError


def boolean_search(query: str, mode: str = "and") -> List[str]:
    """Return the (unranked) list of doc_ids matching `query`, treating it
    as a conjunction (`mode="and"`) or disjunction (`mode="or"`) of its
    terms."""
    raise NotImplementedError


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by
    TF-IDF cosine similarity, highest score first."""
    raise NotImplementedError
