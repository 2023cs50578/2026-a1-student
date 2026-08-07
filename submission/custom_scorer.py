"""
submission/custom_scorer.py — optional combined/custom scorer.

Not required, but this is explicitly called out in the assignment
(Section 4.1) as "where separation in the leaderboard tends to happen":
any linear or non-linear combination of your Boolean/VSM, BM25, and
language-model signals, additional features (e.g. proximity/bigram
overlap), or your own heuristic.

If you use this, wire it in from submission/retrieve.py's retrieve()
instead of calling a single scorer directly, and describe what you did
and why in your report (Section 7, "one-paragraph description of your
final competition entry").
"""
from typing import List, Tuple

from submission.indexer import InvertedIndex


def build(index: InvertedIndex) -> None:
    raise NotImplementedError


def score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by your
    own combined/custom scoring function, highest score first."""
    raise NotImplementedError
