"""
submission/boolean_vsm.py — Boolean retrieval + vector-space ranking.

Required component (assignment Section 4.1). Two independent pieces, both
reading the same `InvertedIndex`.

1. Boolean retrieval (Retrieval-I, "Processing Boolean Queries")
----------------------------------------------------------------
Treat the query as a conjunction or disjunction of its terms and return the
matching document *set* — no scores, just membership. As the lecture notes
point out, this is the operation everything else is built on, and its cost is
what motivates the inverted index in the first place: intersecting two postings
lists is O(n1 + n2) by linear merge instead of O(|V|) bit operations over a
document-term matrix.

The intersection here processes terms in increasing df order. That is the
standard optimisation: the running result set can only shrink, so starting from
the rarest term keeps every subsequent merge as small as possible.

2. Vector-space ranking (cosine on TF-IDF vectors)
--------------------------------------------------
The weighting scheme is SMART **ltc.ltc**: logarithmic term frequency, inverse
document frequency, cosine normalisation, on both the document and query side.

    w(t, d) = (1 + log10 tf(t,d)) * log10(N / df(t))
    w(t, q) = (1 + log10 tf(t,q)) * log10(N / df(t))
    sim(q, d) = (q . d) / (||q|| * ||d||)

Two deliberate departures from the raw `tf * log(N/df)` in the starter
docstring, both standard and both defensible in the oral defense:

  - **log tf, not raw tf.** Raw tf says a document mentioning "vaccine" 20
    times is 20x more about vaccines than one mentioning it once, which is
    plainly false. The log damps that. This is the same intuition BM25
    formalises properly with its k1 saturation — cosine has no k1, so the log
    is where the damping has to live.
  - **cosine normalisation.** Without dividing by ||d||, long documents win
    every query simply by having more non-zero components. This is VSM's only
    length-normalisation mechanism, and it is a blunt one — it normalises by
    vector magnitude regardless of *why* a document is long. BM25's b parameter
    exists precisely because that blunt instrument underperforms, which is the
    comparison the report's Boolean/VSM-vs-BM25 table is meant to show.

Document norms ||d|| require every posting in the collection, not just the
query's. They are therefore computed once, lazily, on the first cosine query
and cached for the process's lifetime, rather than persisted — recomputing them
costs a couple of seconds, while storing 171K float32s on disk would cost the
index-size component permanently for something derivable.
"""
import math
from collections import Counter
from typing import List, Optional, Tuple

from submission.indexer import InvertedIndex, analyze

try:
    import numpy as np

    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    _HAVE_NUMPY = False

_INDEX: Optional[InvertedIndex] = None
_IDF = None        # log10(N / df) per term id
_DOC_NORM = None   # ||d|| per doc id; computed lazily on first cosine query


def build(index: InvertedIndex) -> None:
    """Bind the loaded index and precompute the VSM IDF vector.

    Called from `retrieve.load_index()`. Document norms are deliberately NOT
    computed here — see the module docstring; `_ensure_doc_norms()` does it on
    demand so that a BM25-only query path never pays for them.
    """
    global _INDEX, _IDF, _DOC_NORM
    _INDEX = index
    _DOC_NORM = None

    if _HAVE_NUMPY:
        df = np.asarray(index.df, dtype=np.float64)
        # df is never 0 for a term that made it into the dictionary, but guard
        # anyway so a hand-built index can't produce a divide-by-zero.
        _IDF = np.log10(index.N / np.maximum(df, 1.0))
    else:  # pragma: no cover
        _IDF = [math.log10(index.N / max(d, 1)) for d in index.df]


# ---------------------------------------------------------------------------
# 1. Boolean retrieval
# ---------------------------------------------------------------------------
def boolean_search(query: str, mode: str = "and") -> List[str]:
    """Return the (unranked) list of doc_ids matching `query`, treating it as a
    conjunction (`mode="and"`) or disjunction (`mode="or"`) of its terms.

    Results come back in ascending internal doc id, i.e. corpus order, which is
    the only ordering a Boolean model can justify — it has no notion of "how
    well" a document matches (Retrieval-II, "Drawbacks of Boolean Model").
    """
    index = _require_index()
    mode = mode.lower()
    if mode not in ("and", "or"):
        raise ValueError(f"mode must be 'and' or 'or'; got {mode!r}")

    terms = analyze(query, index.remove_stopwords, index.stemming)
    if not terms:
        return []

    term_ids = [index.term_id(t) for t in dict.fromkeys(terms)]

    if mode == "and":
        # A term absent from the collection has an empty postings list, so the
        # conjunction is empty — no need to touch the index at all.
        if any(tid < 0 for tid in term_ids):
            return []
        # Rarest first: the running intersection only shrinks, so this keeps
        # every subsequent merge as cheap as possible.
        term_ids.sort(key=lambda tid: int(index.df[tid]))
        matched = None
        for tid in term_ids:
            doc_ids = index.postings(tid)[0]
            matched = doc_ids if matched is None else _intersect(matched, doc_ids)
            if len(matched) == 0:
                return []
    else:
        lists = [index.postings(tid)[0] for tid in term_ids if tid >= 0]
        lists = [l for l in lists if len(l) > 0]
        if not lists:
            return []
        matched = _union(lists)

    return [index.doc_ids[int(d)] for d in matched]


def _intersect(left, right):
    """Intersection of two ascending doc-id lists."""
    if _HAVE_NUMPY:
        # np.intersect1d on two already-sorted arrays is a linear merge.
        return np.intersect1d(left, right, assume_unique=True)
    return sorted(set(left) & set(right))  # pragma: no cover


def _union(lists):
    """Union of several ascending doc-id lists, deduplicated and sorted."""
    if _HAVE_NUMPY:
        return np.unique(np.concatenate(lists))
    merged = set()  # pragma: no cover
    for l in lists:  # pragma: no cover
        merged.update(l)
    return sorted(merged)  # pragma: no cover


# ---------------------------------------------------------------------------
# 2. Vector-space (cosine) ranking
# ---------------------------------------------------------------------------
def _ensure_doc_norms():
    """Compute and cache ||d|| for every document.

    One sweep over the whole index, a block of terms at a time (see
    `InvertedIndex.iter_postings_blocks`). Accumulating sum-of-squares per
    document is a scatter-add across blocks, so `np.add.at` is needed here —
    unlike in BM25, a block contains many terms and therefore repeats doc ids.
    """
    global _DOC_NORM
    if _DOC_NORM is not None:
        return _DOC_NORM

    index = _require_index()
    if not _HAVE_NUMPY:  # pragma: no cover
        squares = [0.0] * index.N
        for term_ids, doc_ids, freqs in index.iter_postings_blocks():
            for term_id, doc_id, tf in zip(term_ids, doc_ids, freqs):
                w = (1.0 + math.log10(tf)) * _IDF[term_id]
                squares[doc_id] += w * w
        _DOC_NORM = [math.sqrt(s) or 1.0 for s in squares]
        return _DOC_NORM

    squares = np.zeros(index.N, dtype=np.float64)
    for term_ids, doc_ids, freqs in index.iter_postings_blocks():
        weights = (1.0 + np.log10(freqs.astype(np.float64))) * _IDF[term_ids]
        np.add.at(squares, doc_ids, weights * weights)
    norms = np.sqrt(squares)
    # A document with no indexed terms has norm 0; substitute 1 so the cosine
    # is 0/1 rather than 0/0. Such a document can never match anything anyway.
    norms[norms == 0.0] = 1.0
    _DOC_NORM = norms.astype(np.float32)
    return _DOC_NORM


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by TF-IDF
    cosine similarity, highest score first."""
    return _top_k(vsm_score_array(query), k)


def vsm_score_array(query: str):
    """Cosine similarity of `query` against every document, as a dense array.

    The query norm is a constant across documents, so it cannot change the
    *ranking* — it is divided out anyway so the returned numbers are genuine
    cosines in [0, 1] and are therefore comparable across queries, which
    matters if the custom scorer ever blends them with another signal.
    """
    index = _require_index()
    doc_norms = _ensure_doc_norms()
    scores = np.zeros(index.N, dtype=np.float32) if _HAVE_NUMPY else [0.0] * index.N

    query_counts = Counter(analyze(query, index.remove_stopwords, index.stemming))
    query_weights = {}
    for term, count in query_counts.items():
        term_id = index.term_id(term)
        if term_id < 0:
            continue
        query_weights[term_id] = (1.0 + math.log10(count)) * float(_IDF[term_id])

    query_norm = math.sqrt(sum(w * w for w in query_weights.values()))
    if query_norm == 0.0:
        return scores

    for term_id, query_weight in query_weights.items():
        doc_ids, freqs = index.postings(term_id)
        if len(doc_ids) == 0:
            continue
        if _HAVE_NUMPY:
            doc_weights = (1.0 + np.log10(freqs.astype(np.float32))) * float(_IDF[term_id])
            scores[doc_ids] += (query_weight * doc_weights / doc_norms[doc_ids]).astype(np.float32)
        else:  # pragma: no cover
            for doc_id, tf in zip(doc_ids, freqs):
                w = (1.0 + math.log10(tf)) * _IDF[term_id]
                scores[doc_id] += query_weight * w / doc_norms[doc_id]

    if _HAVE_NUMPY:
        scores /= query_norm
    else:  # pragma: no cover
        scores = [s / query_norm for s in scores]
    return scores


def _top_k(scores, k: int) -> List[Tuple[str, float]]:
    """Shared top-k selection. Identical tie-breaking to `bm25.top_k`."""
    from submission import bm25

    if bm25._INDEX is None:
        bm25.build(_require_index())
    return bm25.top_k(scores, k)


def _require_index() -> InvertedIndex:
    if _INDEX is None:
        raise RuntimeError(
            "boolean_vsm.build(index) must be called before searching; "
            "retrieve.load_index() does this."
        )
    return _INDEX
