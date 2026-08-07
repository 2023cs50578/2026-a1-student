"""
harness/metrics.py — the exact metrics your leaderboard score is computed
from: nDCG@10 (primary), MAP (tie-break), plus MRR and P@k for your own
error analysis. This file is shared, readable, and unit-tested (see
tests/test_metrics.py) precisely so there is no ambiguity about how
scoring works — the same code path is used for the public dev leaderboard
and (with a different, undisclosed topic/qrels file) the private held-out
leaderboard at grading time.

Conventions, matching standard trec_eval behaviour:
  - nDCG uses graded relevance and the (2^rel - 1) gain function.
  - MAP, MRR, and P@k use binarised relevance: a document counts as
    relevant if its qrels judgment is > 0.
  - Unjudged documents (not present in qrels for that query) are treated
    as non-relevant (relevance 0), not skipped.
"""
import math
from typing import Dict, List, Tuple

RunType = Dict[str, List[Tuple[str, float]]]  # qid -> [(doc_id, score), ...], best first
QrelsType = Dict[str, Dict[str, int]]  # qid -> {doc_id: relevance}


def _relevance(qrels_for_q: Dict[str, int], doc_id: str) -> int:
    return qrels_for_q.get(doc_id, 0)


def dcg_at_k(gains: List[int], k: int) -> float:
    """gains: relevance values in ranked order (already truncated or not;
    only the first k are used). DCG = sum_i (2^rel_i - 1) / log2(i + 1),
    1-indexed."""
    dcg = 0.0
    for i, rel in enumerate(gains[:k], start=1):
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_doc_ids: List[str], qrels_for_q: Dict[str, int], k: int = 10) -> float:
    if not qrels_for_q:
        return 0.0
    gains = [_relevance(qrels_for_q, doc_id) for doc_id in ranked_doc_ids[:k]]
    dcg = dcg_at_k(gains, k)
    ideal_gains = sorted(qrels_for_q.values(), reverse=True)
    idcg = dcg_at_k(ideal_gains, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(ranked_doc_ids: List[str], qrels_for_q: Dict[str, int]) -> float:
    n_relevant = sum(1 for rel in qrels_for_q.values() if rel > 0)
    if n_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if _relevance(qrels_for_q, doc_id) > 0:
            hits += 1
            precision_sum += hits / i
    return precision_sum / n_relevant


def reciprocal_rank(ranked_doc_ids: List[str], qrels_for_q: Dict[str, int]) -> float:
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if _relevance(qrels_for_q, doc_id) > 0:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked_doc_ids: List[str], qrels_for_q: Dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if _relevance(qrels_for_q, doc_id) > 0)
    return hits / k


def evaluate_run(run: RunType, qrels: QrelsType, k: int = 10) -> Dict:
    """Evaluate `run` against `qrels`. Returns a dict with per-query
    metrics and corpus-level (macro) averages. Queries present in `run`
    but absent from `qrels` are skipped with a warning key so they don't
    silently zero out your average; queries present in `qrels` but
    missing from `run` are scored as 0 across all metrics (you didn't
    answer them)."""
    per_query = {}
    skipped_no_qrels = []

    all_qids = set(qrels.keys()) | set(run.keys())
    for qid in sorted(all_qids):
        if qid not in qrels:
            skipped_no_qrels.append(qid)
            continue
        qrels_for_q = qrels[qid]
        ranked_doc_ids = [doc_id for doc_id, _score in run.get(qid, [])]
        per_query[qid] = {
            "ndcg@10": ndcg_at_k(ranked_doc_ids, qrels_for_q, k=10),
            "map": average_precision(ranked_doc_ids, qrels_for_q),
            "mrr": reciprocal_rank(ranked_doc_ids, qrels_for_q),
            f"p@{k}": precision_at_k(ranked_doc_ids, qrels_for_q, k),
            "num_ranked": len(ranked_doc_ids),
        }

    n = len(per_query)
    if n == 0:
        aggregate = {"ndcg@10": 0.0, "map": 0.0, "mrr": 0.0, f"p@{k}": 0.0}
    else:
        aggregate = {
            "ndcg@10": sum(m["ndcg@10"] for m in per_query.values()) / n,
            "map": sum(m["map"] for m in per_query.values()) / n,
            "mrr": sum(m["mrr"] for m in per_query.values()) / n,
            f"p@{k}": sum(m[f"p@{k}"] for m in per_query.values()) / n,
        }

    return {
        "per_query": per_query,
        "aggregate": aggregate,
        "num_queries_scored": n,
        "skipped_no_qrels": skipped_no_qrels,
    }
