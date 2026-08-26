"""
submission/custom_scorer.py — the combined scorer used for the competition
entry: tuned BM25 followed by RM3 pseudo-relevance feedback.

Optional per the assignment (Section 4.1), but explicitly flagged as "where
separation in the leaderboard tends to happen" — and on the dev topics it is
worth more than every parameter tuning decision put together.

The problem it solves
---------------------
BM25 can only match words the user actually typed. A query like "will SARS-CoV2
infected people develop immunity" cannot reach a document that says
"seroconversion", "antibody response" and "reinfection" but never says
"immunity" — the classic vocabulary mismatch. Relevance feedback fixes this
without any pretrained model or embedding (both out of scope per Section 10):
it reads the vocabulary of the documents the *first* retrieval pass liked, and
uses that to expand the query.

RM3 (Lavrenko & Croft's relevance model, interpolated with the original query)
--------------------------------------------------------------------------
  1. Run BM25. Take the top `FB_DOCS` documents as *pseudo*-relevant — nobody
     has judged them, we simply assume the top of a decent ranking is mostly
     on-topic.
  2. Build a relevance model over terms:

         P(w | R) = sum_{d in R} P(d | Q) * P(w | d)

     with P(w|d) = tf(w,d) / |d| the maximum-likelihood estimate from the
     document, and P(d|Q) the document's share of the total feedback-set score
     — so a document ranked 1st contributes more vocabulary than one ranked
     20th.
  3. Keep the `FB_TERMS` highest-probability terms and renormalise.
  4. **Interpolate with the original query** (this is the "3" in RM3):

         P(w | Q') = alpha * P(w | Q) + (1 - alpha) * P(w | R)

     This step is what makes the technique safe. Pure expansion (alpha = 0)
     drifts: if the top documents are off-topic, the query is replaced by an
     off-topic one and the query is lost entirely. Keeping alpha of the mass on
     the words the user actually typed anchors the result.
  5. Re-run BM25 with that weighted query.

Cost: exactly one extra BM25 pass plus `FB_DOCS` forward-index reads, so query
latency roughly triples in absolute terms — from ~4 ms to ~12 ms — which is
still far inside the efficiency budget.

Failure modes, and what is done about them
------------------------------------------
  - No feedback documents (query matches nothing): fall back to the plain BM25
    ranking rather than expanding from an empty set.
  - Index built without a forward index: same fallback, so the scorer degrades
    to tuned BM25 instead of crashing.
  - Expansion terms that are near-universal in the collection contribute noise
    at every rank; `MIN_EXPANSION_IDF` drops them.

All parameters below were selected by k-fold cross-validation on the released
dev topics (see `scripts/sweep_params.py` and the report); the held-out topics
were never used to pick any of them.
"""
from typing import Dict, List, Optional, Tuple

from submission import bm25
from submission.indexer import InvertedIndex

try:
    import numpy as np

    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    _HAVE_NUMPY = False

_INDEX: Optional[InvertedIndex] = None

# BM25 parameters, overridden by retrieve.py with the swept values.
K1 = 1.2
B = 0.75

# RM3 parameters.
FB_DOCS = 40            # pseudo-relevant documents to read vocabulary from
FB_TERMS = 30           # expansion terms kept from the relevance model
ALPHA = 0.4             # weight retained on the original query
MIN_EXPANSION_IDF = 0.5  # discard expansion terms this common (BM25 idf units)
ENABLE_RM3 = True


def build(index: InvertedIndex) -> None:
    """Called from retrieve.load_index(), not retrieve.build_index()."""
    global _INDEX
    _INDEX = index
    if bm25._INDEX is not index:
        bm25.build(index)


def score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs, best first."""
    index = _require_index()

    original = bm25.query_term_weights(query)
    if not original:
        return []

    first_pass = bm25.score_array_weighted(original, k1=K1, b=B)
    if not (ENABLE_RM3 and index.forward_terms_per_doc and FB_TERMS > 0):
        return bm25.top_k(first_pass, k)

    expanded = expand_query(original, first_pass)
    if expanded is None:
        return bm25.top_k(first_pass, k)

    return bm25.top_k(bm25.score_array_weighted(expanded, k1=K1, b=B), k)


def expand_query(original: Dict[int, float], first_pass) -> Optional[Dict[int, float]]:
    """Build the RM3 query from the original weights and a first-pass ranking.

    Returns {term_id: weight} summing to 1, or None if there is nothing to
    expand from (in which case the caller keeps the first-pass ranking).
    """
    index = _require_index()

    feedback = bm25.top_k_internal(first_pass, FB_DOCS)
    if not feedback:
        return None
    doc_ids, doc_scores = feedback

    total_score = float(sum(doc_scores))
    if total_score <= 0:
        return None

    # Step 2: P(w|R) = sum_d P(d|Q) P(w|d), over the pruned forward index.
    relevance_model: Dict[int, float] = {}
    for doc_id, doc_score in zip(doc_ids, doc_scores):
        doc_weight = float(doc_score) / total_score
        term_ids, freqs = index.forward(int(doc_id))
        if len(term_ids) == 0:
            continue
        # Normalise by the document's true length, not by the retained terms'
        # total: the forward index is pruned, and rescaling by the pruned sum
        # would silently inflate every kept term's probability.
        length = float(index.doc_len[int(doc_id)]) or 1.0
        for term_id, freq in zip(term_ids, freqs):
            term_id = int(term_id)
            relevance_model[term_id] = (
                relevance_model.get(term_id, 0.0) + doc_weight * float(freq) / length
            )

    if not relevance_model:
        return None

    # Step 3: keep the strongest expansion terms. Terms too common to
    # discriminate are dropped first — they would otherwise spend expansion
    # slots on words that match almost every document.
    candidates = [
        (weight, -term_id, term_id)
        for term_id, weight in relevance_model.items()
        if bm25.idf(term_id) >= MIN_EXPANSION_IDF
    ]
    if not candidates:
        return None
    # Sorting on (weight, -term_id) makes ties break on the lower term id,
    # deterministically, as the interface contract requires.
    candidates.sort(reverse=True)
    kept = candidates[:FB_TERMS]

    expansion_mass = sum(weight for weight, _neg, _tid in kept)
    if expansion_mass <= 0:
        return None

    # Step 4: interpolate. Both sides are normalised to sum to 1 first, so
    # ALPHA means what it says regardless of query length or expansion depth.
    original_mass = sum(original.values()) or 1.0
    weights: Dict[int, float] = {}
    for term_id, weight in original.items():
        weights[term_id] = ALPHA * weight / original_mass
    for weight, _neg, term_id in kept:
        weights[term_id] = weights.get(term_id, 0.0) + (1.0 - ALPHA) * weight / expansion_mass

    return weights


def _require_index() -> InvertedIndex:
    if _INDEX is None:
        raise RuntimeError(
            "custom_scorer.build(index) must be called before scoring; "
            "retrieve.load_index() does this."
        )
    return _INDEX
