"""
submission/bm25.py — Okapi BM25 ranking.

Required component (assignment Section 4.1): "a BM25 implementation with
tunable k1 and b." Based on Robertson & Walker (1992) and the treatment in
Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and
Beyond" (2009).

    score(D, Q) = sum_i  IDF(q_i) * qtf_i * ( tf(q_i, D) * (k1 + 1) )
                          / ( tf(q_i, D) + k1 * (1 - b + b * |D| / avgdl) )

    IDF(q_i) = ln( (N - df(q_i) + 0.5) / (df(q_i) + 0.5) + 1 )

What the two parameters actually do — the thing the oral defense's "predict
the effect" question is about:

  k1 controls **term-frequency saturation**. The tf term above rises towards an
     asymptote of (k1 + 1) rather than growing linearly, so the 20th occurrence
     of a word adds far less than the 2nd. Small k1 saturates almost
     immediately (BM25 tends towards "does the term appear at all", i.e.
     binary-ish, which suits short documents where a repeat is noise); large k1
     keeps rewarding repeats (which suits long documents where genuine topical
     concentration shows up as high tf). k1 = 0 makes tf irrelevant entirely.

  b  controls **document-length normalisation**. |D|/avgdl scales the
     saturation point by how long the document is, so a long document needs
     more occurrences to earn the same score. b = 0 disables length
     normalisation completely (long documents win, because they contain more of
     everything); b = 1 normalises fully (a document is judged purely on term
     *density*). Collections with genuinely varying document lengths — where a
     long document is long because it says more, not because it is padded —
     usually want b well below the textbook 0.75.

The IDF form above is the Robertson-Sparck Jones weight with the +1 inside the
logarithm, which keeps it non-negative even for a term appearing in more than
half the collection (the unsmoothed form goes negative there, and a negative
contribution would let a document be *penalised* for containing a query word).

Both parameters are function arguments, never constants, precisely so the
sweep in the report can vary them without touching this file.
"""
from collections import Counter
from typing import List, Optional, Tuple

import math

from submission.indexer import InvertedIndex, analyze

try:
    import numpy as np

    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    _HAVE_NUMPY = False

# Module state, populated by build() from retrieve.load_index().
_INDEX: Optional[InvertedIndex] = None
_IDF = None            # idf[term_id]
_LEN_RATIO = None      # |D| / avgdl per doc id


def build(index: InvertedIndex) -> None:
    """Precompute the query-independent parts of BM25 from a loaded index.

    Two things are cached here, both of which would otherwise be recomputed on
    every query: the IDF of every term (a pure function of df and N) and each
    document's length ratio |D|/avgdl. Neither is written to disk — they are
    derivable from what `InvertedIndex.save()` already stores, and persisting
    them would only inflate the index-size score for no gain.

    Called from `retrieve.load_index()`, not `retrieve.build_index()`: the two
    run in separate processes and only this one ever calls `score()`.
    """
    global _INDEX, _IDF, _LEN_RATIO
    _INDEX = index

    if _HAVE_NUMPY:
        df = np.asarray(index.df, dtype=np.float64)
        _IDF = np.log(1.0 + (index.N - df + 0.5) / (df + 0.5))
        doc_len = np.asarray(index.doc_len, dtype=np.float64)
        avg = index.avg_doc_len or 1.0
        _LEN_RATIO = (doc_len / avg).astype(np.float32)
    else:  # pragma: no cover
        _IDF = [
            math.log(1.0 + (index.N - d + 0.5) / (d + 0.5)) for d in index.df
        ]
        avg = index.avg_doc_len or 1.0
        _LEN_RATIO = [length / avg for length in index.doc_len]


def score_array(query: str, k1: float = 1.2, b: float = 0.75):
    """BM25 score every document that contains at least one query term.

    Returns a dense score array indexed by internal doc id. Kept separate from
    `score()` so the custom scorer can blend these raw scores with other
    signals (and so relevance feedback can re-score with an expanded query)
    without going through the top-k selection twice.

    The accumulation is the textbook term-at-a-time strategy: walk one query
    term's postings list at a time and add its contribution into a score
    accumulator over documents. Because a term's postings contain each doc id
    at most once, the whole contribution for a term is one vectorised
    scatter-add — no Python loop over documents anywhere.
    """
    return score_array_weighted(query_term_weights(query), k1=k1, b=b)


def query_term_weights(query: str):
    """Analyse `query` into {term_id: query term frequency}.

    Terms absent from the dictionary are dropped here rather than checked for
    later, so every consumer sees only scoreable term ids.
    """
    index = _require_index()
    weights = {}
    counts = Counter(analyze(query, index.remove_stopwords, index.stemming))
    for term, query_tf in counts.items():
        term_id = index.term_id(term)
        if term_id >= 0:
            weights[term_id] = float(query_tf)
    return weights


def score_array_weighted(term_weights, k1: float = 1.2, b: float = 0.75):
    """BM25 over an explicitly weighted query: {term_id: weight}.

    Splitting this out from `score_array()` is what lets relevance feedback
    re-score with an expanded query — the expansion terms are simply extra
    entries in the weight map, carrying fractional weights, and the scoring
    loop neither knows nor cares that they were not in the user's query.
    """
    index = _require_index()
    scores = _zeros(index.N)

    for term_id, weight in term_weights.items():
        if weight <= 0:
            continue
        doc_ids, freqs = index.postings(term_id)
        if len(doc_ids) == 0:
            continue
        _accumulate_term(scores, doc_ids, freqs, term_id, weight, k1, b)

    return scores


def _accumulate_term(scores, doc_ids, freqs, term_id, query_weight, k1, b) -> None:
    """Add one query term's BM25 contribution into the score accumulator."""
    if _HAVE_NUMPY:
        tf = freqs.astype(np.float32)
        # The saturation denominator: k1 scaled by how long this document is
        # relative to the collection average, blended by b.
        norm = k1 * (1.0 - b + b * _LEN_RATIO[doc_ids])
        contribution = (_IDF[term_id] * query_weight) * (tf * (k1 + 1.0)) / (tf + norm)
        scores[doc_ids] += contribution.astype(np.float32)
    else:  # pragma: no cover
        idf = _IDF[term_id]
        for doc_id, tf in zip(doc_ids, freqs):
            norm = k1 * (1.0 - b + b * _LEN_RATIO[doc_id])
            scores[doc_id] += idf * query_weight * (tf * (k1 + 1.0)) / (tf + norm)


def score(query: str, k: int, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, BM25-ranked,
    highest score first."""
    return top_k(score_array(query, k1=k1, b=b), k)


def top_k(scores, k: int) -> List[Tuple[str, float]]:
    """Turn a dense score array into the top k (external doc_id, score) pairs.

    Only documents with a strictly positive score are eligible: a zero means
    the document shares no term with the query, and padding the list with such
    documents would be noise. Ties are broken by internal doc id so the ranking
    is fully deterministic, as the interface contract requires.
    """
    index = _require_index()
    if k <= 0:
        return []

    if not _HAVE_NUMPY:  # pragma: no cover
        ranked = sorted(
            ((s, -i) for i, s in enumerate(scores) if s > 0), reverse=True
        )[:k]
        return [(index.doc_ids[-i], float(s)) for s, i in ranked]

    candidates = np.flatnonzero(scores > 0)
    if candidates.size == 0:
        return []
    candidate_scores = scores[candidates]
    if candidates.size > k:
        # argpartition is O(n) and avoids sorting 171K scores to read 10.
        top = np.argpartition(-candidate_scores, k - 1)[:k]
        candidates = candidates[top]
        candidate_scores = candidate_scores[top]
    # lexsort's last key is primary: sort by descending score, then ascending
    # doc id — deterministic even when scores tie exactly.
    order = np.lexsort((candidates, -candidate_scores))
    return [
        (index.doc_ids[int(candidates[i])], float(candidate_scores[i]))
        for i in order
    ]


def top_k_internal(scores, k: int):
    """Like `top_k`, but returns (internal doc ids, scores) rather than
    external doc_id strings.

    Relevance feedback needs internal ids — they are what index the forward
    index and the document-length table — so converting to external strings and
    back would be pure waste. Same positive-score filter and same deterministic
    tie-break as `top_k`.
    """
    if k <= 0:
        return None
    if not _HAVE_NUMPY:  # pragma: no cover
        ranked = sorted(((s, -i) for i, s in enumerate(scores) if s > 0), reverse=True)[:k]
        if not ranked:
            return None
        return [-i for _s, i in ranked], [s for s, _i in ranked]

    candidates = np.flatnonzero(scores > 0)
    if candidates.size == 0:
        return None
    candidate_scores = scores[candidates]
    if candidates.size > k:
        top = np.argpartition(-candidate_scores, k - 1)[:k]
        candidates = candidates[top]
        candidate_scores = candidate_scores[top]
    order = np.lexsort((candidates, -candidate_scores))
    return candidates[order], candidate_scores[order]


def idf(term_id: int) -> float:
    """The cached BM25 IDF of a term id. Used by the custom scorer."""
    return float(_IDF[term_id])


def _zeros(n: int):
    return np.zeros(n, dtype=np.float32) if _HAVE_NUMPY else [0.0] * n


def _require_index() -> InvertedIndex:
    if _INDEX is None:
        raise RuntimeError(
            "bm25.build(index) must be called before scoring; "
            "retrieve.load_index() does this."
        )
    return _INDEX
