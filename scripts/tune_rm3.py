#!/usr/bin/env python
"""
scripts/tune_rm3.py — choose the RM3 parameters by k-fold cross-validation.

Why not just take the dev-set argmax? Because there are only 50 dev topics and
the RM3 grid has ~150 points: the best cell on 50 queries is partly luck, and
copying it straight into the submission is exactly the dev-set overfitting the
assignment's two-tier leaderboard is designed to punish (Section 3, learning
objective 5).

So: split the dev topics into k folds, pick parameters on k-1 folds, score them
on the held-out fold, and report the mean held-out nDCG@10. That number is an
honest estimate of what the parameters will do on topics we have never seen —
unlike the dev-set argmax, which is an estimate of nothing.

Usage:
    python scripts/tune_rm3.py --index-dir /tmp/a1_sweep_index
"""
import argparse, itertools, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries
from submission import bm25, custom_scorer
from submission.indexer import InvertedIndex


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", default="/tmp/a1_sweep_index")
    ap.add_argument("--queries", default="data/full/queries_dev.tsv")
    ap.add_argument("--qrels", default="data/full/qrels_dev.txt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k1", type=float, default=2.0)
    ap.add_argument("--b", type=float, default=0.6)
    ap.add_argument("--out", default="runs/rm3_cv.json")
    args = ap.parse_args()

    index = InvertedIndex.load(args.index_dir)
    bm25.build(index); custom_scorer.build(index)
    custom_scorer.K1, custom_scorer.B = args.k1, args.b

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    print(f"{len(queries)} dev topics, {args.folds}-fold CV, BM25 k1={args.k1} b={args.b}\n")

    grid = [
        {"FB_DOCS": fd, "FB_TERMS": ft, "ALPHA": a}
        for fd in (10, 20, 30, 40, 50)
        for ft in (10, 20, 30, 50)
        for a in (0.3, 0.4, 0.5, 0.6)
    ]

    # Score every configuration once, per query, and cache. The CV afterwards
    # is then pure bookkeeping over that table.
    per_query = {}
    t0 = time.perf_counter()
    for i, cfg in enumerate(grid):
        for key, value in cfg.items():
            setattr(custom_scorer, key, value)
        run = {qid: custom_scorer.score(text, 10) for qid, text in queries}
        result = evaluate_run(run, qrels, k=10)["per_query"]
        per_query[tuple(cfg.values())] = {q: m["ndcg@10"] for q, m in result.items()}
        if (i + 1) % 20 == 0:
            print(f"  scored {i+1}/{len(grid)} configs ({time.perf_counter()-t0:.0f}s)")

    # BM25-only reference, same folds.
    custom_scorer.ENABLE_RM3 = False
    base_run = {qid: custom_scorer.score(text, 10) for qid, text in queries}
    base_pq = {q: m["ndcg@10"] for q, m in evaluate_run(base_run, qrels, k=10)["per_query"].items()}
    custom_scorer.ENABLE_RM3 = True

    qids = sorted(base_pq)
    folds = [qids[i::args.folds] for i in range(args.folds)]

    def mean(cfg, subset):
        table = per_query[cfg]
        return sum(table[q] for q in subset) / len(subset)

    print("\n=== cross-validation ===")
    held_out, picks = [], []
    for f, test in enumerate(folds):
        train = [q for q in qids if q not in set(test)]
        best = max(per_query, key=lambda c: mean(c, train))
        score = mean(best, test)
        held_out.append(score); picks.append(best)
        print(f"  fold {f}: picked FB_DOCS={best[0]:<3} FB_TERMS={best[1]:<3} ALPHA={best[2]}"
              f"  -> held-out nDCG@10={score:.4f}  (BM25-only {sum(base_pq[q] for q in test)/len(test):.4f})")

    cv_rm3 = sum(held_out) / len(held_out)
    cv_base = sum(base_pq.values()) / len(base_pq)
    print(f"\n  mean held-out nDCG@10 (RM3, params chosen per fold): {cv_rm3:.4f}")
    print(f"  BM25-only nDCG@10 over all dev topics:                {cv_base:.4f}")
    print(f"  honest expected gain from RM3:                        {cv_rm3-cv_base:+.4f}")

    # A single configuration has to ship. Prefer the one with the best WORST
    # fold rather than the best mean: on unseen topics, robustness beats a
    # peak that one lucky fold produced.
    def worst_fold(cfg):
        return min(mean(cfg, fold) for fold in folds)
    robust = max(per_query, key=worst_fold)
    overall = max(per_query, key=lambda c: mean(c, qids))
    print(f"\n  dev-set argmax:        FB_DOCS={overall[0]}, FB_TERMS={overall[1]}, ALPHA={overall[2]}"
          f"  (dev {mean(overall, qids):.4f}, worst fold {worst_fold(overall):.4f})")
    print(f"  best-worst-fold pick:  FB_DOCS={robust[0]}, FB_TERMS={robust[1]}, ALPHA={robust[2]}"
          f"  (dev {mean(robust, qids):.4f}, worst fold {worst_fold(robust):.4f})")

    top = sorted(per_query, key=lambda c: mean(c, qids), reverse=True)[:12]
    print("\n  top 12 configurations by dev-set mean:")
    for c in top:
        print(f"    FB_DOCS={c[0]:<3} FB_TERMS={c[1]:<3} ALPHA={c[2]}  dev={mean(c,qids):.4f}  worst-fold={worst_fold(c):.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"cv_rm3": cv_rm3, "cv_base": cv_base,
                       "dev_argmax": list(overall), "robust": list(robust),
                       "grid": {str(k): sum(v.values())/len(v) for k, v in per_query.items()}}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
