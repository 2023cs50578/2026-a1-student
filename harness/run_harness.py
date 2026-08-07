"""
harness/run_harness.py — the grading harness. This is the exact code path
used for the public dev leaderboard (which you can run yourself) and, with
a different corpus/topics/qrels file that is never distributed, the
private held-out leaderboard at grading time.

Usage:
    python -m harness.run_harness \\
        --corpus data/toy/corpus.jsonl \\
        --queries data/toy/queries_dev.tsv \\
        --qrels data/toy/qrels_dev.txt \\
        --k 10 \\
        --submission submission.retrieve \\
        --run-out runs/dev_run.trec \\
        --report-out runs/dev_report.json \\
        --baseline-run data/toy/baseline_run_dev.trec

Exit codes:
    0  ran to completion (see the printed/JSON report for scores)
    1  interface conformance failure (this is what the CI check greps for)
    2  runtime error while scoring (crash inside build_index/retrieve)
"""
import argparse
import importlib
import inspect
import json
import os
import sys
import time
from typing import Callable, List, Tuple

from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries, read_run, write_run
from harness.leaderboard import summarize

REQUIRED_RETRIEVE_PARAMS = ["query", "k"]


def check_conformance(submission_module) -> List[str]:
    """Returns a list of conformance problems (empty list = conforms)."""
    problems = []

    if not hasattr(submission_module, "build_index"):
        problems.append("submission module is missing a `build_index(corpus_path)` function.")
    if not hasattr(submission_module, "retrieve"):
        problems.append("submission module is missing a `retrieve(query, k)` function.")
        return problems  # can't check signature of something that doesn't exist

    sig = inspect.signature(submission_module.retrieve)
    param_names = list(sig.parameters.keys())
    if param_names[:2] != REQUIRED_RETRIEVE_PARAMS:
        problems.append(
            f"retrieve() must accept (query, k, ...) as its first two parameters "
            f"(in that order); found {param_names}."
        )
    return problems


def run_submission(
    submission_module,
    corpus_path: str,
    queries: List[Tuple[str, str]],
    k: int,
) -> Tuple[dict, float, List[float]]:
    """Returns (run, index_build_seconds, per_query_latencies_seconds)."""
    t0 = time.perf_counter()
    submission_module.build_index(corpus_path)
    build_time = time.perf_counter() - t0

    run = {}
    latencies = []
    for qid, text in queries:
        t0 = time.perf_counter()
        results = submission_module.retrieve(text, k)
        latencies.append(time.perf_counter() - t0)

        if not isinstance(results, list):
            raise TypeError(f"retrieve() must return a list; got {type(results)} for qid={qid}")
        if len(results) > k:
            raise ValueError(f"retrieve() returned {len(results)} results for qid={qid}, more than k={k}")
        for item in results:
            if not (isinstance(item, tuple) and len(item) == 2):
                raise TypeError(f"each result must be a (doc_id, score) pair; got {item!r} for qid={qid}")

        # Defensive: sort by score descending even if the submission forgot to.
        results_sorted = sorted(results, key=lambda pair: pair[1], reverse=True)
        run[qid] = results_sorted

    return run, build_time, latencies


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--submission", default="submission.retrieve", help="dotted module path exposing build_index()/retrieve()")
    parser.add_argument("--run-out", default=None, help="where to write the TREC run file")
    parser.add_argument("--report-out", default=None, help="where to write the JSON report")
    parser.add_argument("--baseline-run", default=None, help="a precomputed baseline run file to compare against")
    parser.add_argument("--run-tag", default="submission")
    args = parser.parse_args()

    try:
        submission_module = importlib.import_module(args.submission)
    except Exception as e:
        print(f"CONFORMANCE FAIL: could not import '{args.submission}': {e}", file=sys.stderr)
        sys.exit(1)

    problems = check_conformance(submission_module)
    if problems:
        print("CONFORMANCE FAIL:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)

    try:
        run, build_time, latencies = run_submission(submission_module, args.corpus, queries, args.k)
    except Exception as e:
        print(f"RUNTIME ERROR while scoring submission: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)

    if args.run_out:
        os.makedirs(os.path.dirname(args.run_out) or ".", exist_ok=True)
        write_run(args.run_out, run, run_tag=args.run_tag)

    eval_result = evaluate_run(run, qrels, k=args.k)

    baseline_summary = None
    if args.baseline_run and os.path.exists(args.baseline_run):
        baseline_run = read_run(args.baseline_run)
        baseline_eval = evaluate_run(baseline_run, qrels, k=args.k)
        baseline_summary = baseline_eval["aggregate"]

    leaderboard_summary = summarize(eval_result["aggregate"], baseline_summary)

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    report = {
        "submission": args.submission,
        "num_queries": len(queries),
        "num_queries_scored": eval_result["num_queries_scored"],
        "skipped_no_qrels": eval_result["skipped_no_qrels"],
        "aggregate_metrics": eval_result["aggregate"],
        "leaderboard": leaderboard_summary,
        "efficiency": {
            "index_build_seconds": build_time,
            "mean_query_latency_seconds": mean_latency,
            "max_query_latency_seconds": max(latencies) if latencies else 0.0,
        },
        "per_query_metrics": eval_result["per_query"],
    }

    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    _print_human_report(report)


def _print_human_report(report: dict) -> None:
    agg = report["aggregate_metrics"]
    eff = report["efficiency"]
    lb = report["leaderboard"]
    print("=" * 60)
    print(f"Submission:           {report['submission']}")
    print(f"Queries scored:       {report['num_queries_scored']} / {report['num_queries']}")
    if report["skipped_no_qrels"]:
        print(f"  (skipped, no qrels: {report['skipped_no_qrels']})")
    print("-" * 60)
    print(f"nDCG@10:              {agg['ndcg@10']:.4f}")
    print(f"MAP:                  {agg['map']:.4f}")
    print(f"MRR:                  {agg['mrr']:.4f}")
    print(f"P@10:                 {agg.get('p@10', float('nan')):.4f}")
    print("-" * 60)
    print(f"Index build time:     {eff['index_build_seconds']:.4f}s")
    print(f"Mean query latency:   {eff['mean_query_latency_seconds'] * 1000:.2f}ms")
    print(f"Max query latency:    {eff['max_query_latency_seconds'] * 1000:.2f}ms")
    print("-" * 60)
    print(f"Provisional score (90% weight, nDCG@10 + MAP): {lb['provisional_score_90pct']:.4f}")
    if "beats_baseline_ndcg@10" in lb:
        print(f"Baseline nDCG@10:     {lb['baseline_ndcg@10']:.4f}  (you beat it: {lb['beats_baseline_ndcg@10']})")
        print(f"Baseline MAP:         {lb['baseline_map']:.4f}  (you beat it: {lb['beats_baseline_map']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
