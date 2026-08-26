#!/usr/bin/env python
"""
scripts/sweep_params.py — parameter search for the report (assignment
Section 8: "your parameter search procedure for k1 and b (include a plot of
nDCG@10 vs. the swept parameter)").

Builds the index ONCE, then evaluates every parameter setting against it
in-process. Going through `harness/run_harness.py` for each point would rebuild
the index every time; the metrics are imported from `harness.metrics` so the
numbers here are computed by exactly the same code the leaderboard uses.

Tuning is done on the released DEV topics only. The held-out topics are never
seen, which is the whole point of the two-tier leaderboard.

Usage:
    python scripts/sweep_params.py --corpus data/full/corpus.jsonl \\
        --queries data/full/queries_dev.tsv --qrels data/full/qrels_dev.txt \\
        --index-dir /tmp/a1_index --sweep k1,b
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries
from submission import bm25, boolean_vsm, custom_scorer
from submission.indexer import InvertedIndex
from submission.retrieve import _stream_corpus


def build_or_reuse(corpus_path: str, index_dir: str, rebuild: bool) -> float:
    """Build the index unless one is already sitting in `index_dir`."""
    meta = os.path.join(index_dir, "meta.json")
    if os.path.exists(meta) and not rebuild:
        print(f"reusing index at {index_dir}")
        return 0.0
    os.makedirs(index_dir, exist_ok=True)
    t0 = time.perf_counter()
    index = InvertedIndex()
    index.build(_stream_corpus(corpus_path))
    index.save(index_dir)
    elapsed = time.perf_counter() - t0
    print(f"built index in {elapsed:.1f}s -> {index_dir}")
    return elapsed


def evaluate(scorer, queries, qrels, k=10):
    """Run `scorer(query_text, k)` over every query and return the aggregate."""
    run = {}
    latencies = []
    for qid, text in queries:
        t0 = time.perf_counter()
        run[qid] = scorer(text, k)
        latencies.append(time.perf_counter() - t0)
    result = evaluate_run(run, qrels, k=k)
    aggregate = dict(result["aggregate"])
    aggregate["mean_latency_ms"] = 1000 * sum(latencies) / max(len(latencies), 1)
    return aggregate, result["per_query"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/full/corpus.jsonl")
    parser.add_argument("--queries", default="data/full/queries_dev.tsv")
    parser.add_argument("--qrels", default="data/full/qrels_dev.txt")
    parser.add_argument("--index-dir", default="/tmp/a1_sweep_index")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", default="runs/sweep.json")
    parser.add_argument(
        "--sweep", default="k1,b",
        help="comma-separated: k1, b, grid, models",
    )
    args = parser.parse_args()

    build_or_reuse(args.corpus, args.index_dir, args.rebuild)

    t0 = time.perf_counter()
    index = InvertedIndex.load(args.index_dir)
    print(f"loaded index in {time.perf_counter() - t0:.3f}s "
          f"(N={index.N}, |V|={len(index.terms)}, avgdl={index.avg_doc_len:.1f})")
    bm25.build(index)
    boolean_vsm.build(index)
    custom_scorer.build(index)

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    print(f"{len(queries)} queries, {len(qrels)} with judgments\n")

    stages = args.sweep.split(",")
    results = {}

    # One-parameter-at-a-time sweeps, each holding the other at its textbook
    # value. This is what the report plots: it shows the shape of the response
    # curve, which a grid alone hides.
    if "k1" in stages:
        print("=== nDCG@10 vs k1  (b = 0.75) ===")
        results["k1"] = []
        for k1 in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0]:
            agg, _ = evaluate(lambda q, k, k1=k1: bm25.score(q, k, k1=k1, b=0.75),
                              queries, qrels, args.k)
            results["k1"].append({"k1": k1, **agg})
            print(f"  k1={k1:<4} nDCG@10={agg['ndcg@10']:.4f}  MAP@10={agg['map@10']:.4f}  "
                  f"P@10={agg['p@10']:.4f}  {agg['mean_latency_ms']:.1f}ms")

    if "b" in stages:
        print("\n=== nDCG@10 vs b  (k1 = 1.2) ===")
        results["b"] = []
        for b in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]:
            agg, _ = evaluate(lambda q, k, b=b: bm25.score(q, k, k1=1.2, b=b),
                              queries, qrels, args.k)
            results["b"].append({"b": b, **agg})
            print(f"  b={b:<5} nDCG@10={agg['ndcg@10']:.4f}  MAP@10={agg['map@10']:.4f}  "
                  f"P@10={agg['p@10']:.4f}  {agg['mean_latency_ms']:.1f}ms")

    # The two parameters interact (raising k1 changes which b is optimal), so
    # the joint grid is what actually picks the operating point.
    if "grid" in stages:
        print("\n=== joint (k1, b) grid, nDCG@10 ===")
        k1_values = [0.6, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        b_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]
        results["grid"] = []
        print("        " + "".join(f"b={b:<7}" for b in b_values))
        best = None
        for k1 in k1_values:
            row = []
            for b in b_values:
                agg, _ = evaluate(lambda q, k, k1=k1, b=b: bm25.score(q, k, k1=k1, b=b),
                                  queries, qrels, args.k)
                results["grid"].append({"k1": k1, "b": b, **agg})
                row.append(agg["ndcg@10"])
                if best is None or agg["ndcg@10"] > best[0]:
                    best = (agg["ndcg@10"], k1, b, agg)
            print(f"  k1={k1:<4}" + "".join(f"{v:<9.4f}" for v in row))
        print(f"\n  best: nDCG@10={best[0]:.4f} at k1={best[1]}, b={best[2]} "
              f"(MAP@10={best[3]['map@10']:.4f})")
        results["best_grid"] = {"ndcg@10": best[0], "k1": best[1], "b": best[2]}

    # The required-components comparison table for the report.
    if "models" in stages:
        print("\n=== model comparison ===")
        results["models"] = {}
        for name, scorer in [
            ("VSM cosine (ltc.ltc)", boolean_vsm.vsm_score),
            ("BM25 (k1=1.2, b=0.75)", lambda q, k: bm25.score(q, k, 1.2, 0.75)),
            ("custom", custom_scorer.score),
        ]:
            agg, _ = evaluate(scorer, queries, qrels, args.k)
            results["models"][name] = agg
            print(f"  {name:<24} nDCG@10={agg['ndcg@10']:.4f}  MAP@10={agg['map@10']:.4f}  "
                  f"MRR={agg['mrr']:.4f}  P@10={agg['p@10']:.4f}  {agg['mean_latency_ms']:.1f}ms")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
