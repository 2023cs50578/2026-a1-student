"""
tests/test_metrics.py — unit tests for harness/metrics.py itself, with
hand-computable examples. These exist so you can trust the scoring code
(it's the same code that computes your grade) without having to take our
word for it — every expected value below is derived by hand in the
comments, not just asserted against the implementation's own output.
"""
import math

from harness.metrics import (
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)


def test_ndcg_perfect_ranking_is_one():
    qrels = {"d1": 3, "d2": 2, "d3": 1, "d4": 0}
    ranked = ["d1", "d2", "d3", "d4"]  # already in relevance order
    assert ndcg_at_k(ranked, qrels, k=10) == 1.0


def test_ndcg_no_relevant_docs_retrieved_is_zero():
    qrels = {"d1": 3, "d2": 2}
    ranked = ["d9", "d10"]  # neither judged relevant (in fact unjudged)
    assert ndcg_at_k(ranked, qrels, k=10) == 0.0


def test_ndcg_worked_example():
    # Ranked relevances (in retrieval order): [2, 0, 1]
    # DCG = (2^2-1)/log2(2) + (2^0-1)/log2(3) + (2^1-1)/log2(4)
    #     = 3/1 + 0/1.585 + 1/2
    #     = 3 + 0 + 0.5 = 3.5
    # Ideal order is [2, 1, 0] (sorted descending):
    # IDCG = 3/1 + 1/log2(3) + 0/log2(4) = 3 + 0.6309... = 3.6309...
    qrels = {"a": 2, "b": 0, "c": 1}
    ranked = ["a", "b", "c"]
    expected_dcg = 3.0 + 0.0 + 0.5
    expected_idcg = 3.0 + 1 / math.log2(3) + 0.0
    expected_ndcg = expected_dcg / expected_idcg
    assert math.isclose(ndcg_at_k(ranked, qrels, k=10), expected_ndcg, rel_tol=1e-9)


def test_dcg_at_k_matches_formula_directly():
    gains = [3, 2, 1, 0]
    expected = sum((2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(gains, start=1))
    assert math.isclose(dcg_at_k(gains, 10), expected, rel_tol=1e-9)


def test_average_precision_worked_example():
    # 2 relevant docs total, found at ranks 1 and 3.
    # AP = (precision@1 + precision@3) / 2 = (1/1 + 2/3) / 2 = 0.8333...
    qrels = {"rel1": 1, "irrelevant": 0, "rel2": 1}
    ranked = ["rel1", "irrelevant", "rel2"]
    assert math.isclose(average_precision(ranked, qrels), (1.0 + 2 / 3) / 2, rel_tol=1e-9)


def test_average_precision_no_relevant_in_qrels_is_zero():
    assert average_precision(["a", "b"], {"a": 0, "b": 0}) == 0.0


def test_reciprocal_rank_worked_example():
    qrels = {"a": 0, "b": 0, "c": 1}
    ranked = ["a", "b", "c"]
    assert reciprocal_rank(ranked, qrels) == 1.0 / 3


def test_reciprocal_rank_no_relevant_found_is_zero():
    qrels = {"a": 1}
    ranked = ["b", "c"]
    assert reciprocal_rank(ranked, qrels) == 0.0


def test_precision_at_k_worked_example():
    qrels = {"a": 1, "b": 0, "c": 1, "d": 0}
    ranked = ["a", "b", "c", "d"]
    # top 2: a (rel), b (not) -> 1/2
    assert precision_at_k(ranked, qrels, k=2) == 0.5
    # top 4: a, c relevant out of 4 -> 2/4
    assert precision_at_k(ranked, qrels, k=4) == 0.5
