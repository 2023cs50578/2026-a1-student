"""
submission/language_model.py — query-likelihood language-model ranking.

Required component (assignment Section 4.1): "query-likelihood scoring
with your choice of Jelinek-Mercer or Dirichlet smoothing, with the
smoothing constant exposed as a tunable parameter."

Rank documents by P(Q | D): the probability that a language model built
from document D would generate the query Q. Because raw maximum-likelihood
estimates assign zero probability to any query term absent from D, you
must smooth document language models with the collection language model.

Jelinek-Mercer smoothing (linear interpolation, lambda in [0, 1]):

    P(w | D) = (1 - lambda) * tf(w, D) / |D|  +  lambda * cf(w) / |C|

Dirichlet smoothing (mu > 0, typically ~1000-2000 for natural-language
collections):

    P(w | D) = ( tf(w, D) + mu * cf(w) / |C| ) / ( |D| + mu )

where tf(w, D) is the term frequency of w in D, |D| the length of D,
cf(w) the term's total frequency across the whole collection, and |C| the
total collection length (sum of all document lengths).

Query likelihood is the product over query terms of P(w | D); implement
this as a sum of log-probabilities to avoid numerical underflow:

    log P(Q | D) = sum_i log P(qi | D)

Pick one smoothing method (or implement both and compare them in your
report), and expose the smoothing constant as a parameter, not a
hard-coded value.
"""
from typing import List, Tuple

from submission.indexer import InvertedIndex


def build(index: InvertedIndex) -> None:
    """Optional: precompute anything LM-specific (e.g. collection term
    frequencies, |C|) from the InvertedIndex built in indexer.py. Called
    once, from your retrieve.build_index()."""
    raise NotImplementedError


def score(
    query: str,
    k: int,
    method: str = "jelinek-mercer",
    smoothing_param: float = 0.1,
) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by
    (log) query-likelihood, highest score first.

    `method`: "jelinek-mercer" (smoothing_param = lambda in [0, 1]) or
    "dirichlet" (smoothing_param = mu > 0)."""
    raise NotImplementedError
