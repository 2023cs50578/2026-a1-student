#!/usr/bin/env python
"""
scripts/error_analysis.py — per-query diagnostics for the report (assignment
Section 8: "an error analysis of 3-5 queries where your system ranks poorly").

For each dev topic it reports nDCG@10 under BM25 alone and under the full
BM25+RM3 entry, the RM3 expansion terms actually chosen, and what sits in the
top ranks — enough to say *why* a query failed rather than just that it did.

Usage:
    python scripts/error_analysis.py --index-dir /tmp/a1_sweep_index --worst 5
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import evaluate_run, ndcg_at_k
from harness.trec_io import read_qrels, read_queries
from submission import bm25, boolean_vsm, custom_scorer
from submission.indexer import InvertedIndex, analyze


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", default="/tmp/a1_sweep_index")
    ap.add_argument("--queries", default="data/full/queries_dev.tsv")
    ap.add_argument("--qrels", default="data/full/qrels_dev.txt")
    ap.add_argument("--corpus", default="data/full/corpus.jsonl",
                    help="only used to print snippets of the top-ranked documents")
    ap.add_argument("--worst", type=int, default=5)
    ap.add_argument("--k1", type=float, default=2.0)
    ap.add_argument("--b", type=float, default=0.6)
    ap.add_argument("--out", default="runs/error_analysis.json")
    args = ap.parse_args()

    index = InvertedIndex.load(args.index_dir)
    bm25.build(index); boolean_vsm.build(index); custom_scorer.build(index)
    custom_scorer.K1, custom_scorer.B = args.k1, args.b

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)

    rows = []
    for qid, text in queries:
        base = bm25.score(text, 10, k1=args.k1, b=args.b)
        full = custom_scorer.score(text, 10)
        q = qrels.get(qid, {})
        original = bm25.query_term_weights(text)
        expanded = custom_scorer.expand_query(
            original, bm25.score_array_weighted(original, k1=args.k1, b=args.b)
        ) or {}
        added = sorted(
            ((w, index.terms[t]) for t, w in expanded.items() if t not in original),
            reverse=True,
        )[:12]
        rows.append({
            "qid": qid,
            "query": text,
            "query_terms": analyze(text),
            "n_relevant": sum(1 for r in q.values() if r > 0),
            "ndcg_bm25": ndcg_at_k([d for d, _ in base], q, 10),
            "ndcg_rm3": ndcg_at_k([d for d, _ in full], q, 10),
            "expansion": [t for _w, t in added],
            "top10_grades": [q.get(d, 0) for d, _ in full],
            "unjudged_in_top10": sum(1 for d, _ in full if d not in q),
        })

    rows.sort(key=lambda r: r["ndcg_rm3"])
    print(f"{'qid':>4} {'nDCG BM25':>10} {'nDCG RM3':>9} {'delta':>7} {'#rel':>5} {'unjudged@10':>12}  query")
    for r in rows:
        print(f"{r['qid']:>4} {r['ndcg_bm25']:>10.4f} {r['ndcg_rm3']:>9.4f} "
              f"{r['ndcg_rm3']-r['ndcg_bm25']:>+7.4f} {r['n_relevant']:>5} "
              f"{r['unjudged_in_top10']:>12}  {r['query'][:60]}")

    print(f"\n=== {args.worst} worst topics in detail ===")
    for r in rows[:args.worst]:
        print(f"\n[{r['qid']}] {r['query']}")
        print(f"  analysed to : {r['query_terms']}")
        print(f"  nDCG@10     : BM25 {r['ndcg_bm25']:.4f} -> RM3 {r['ndcg_rm3']:.4f}")
        print(f"  relevant docs in qrels: {r['n_relevant']}")
        print(f"  top-10 relevance grades: {r['top10_grades']}  ({r['unjudged_in_top10']} unjudged)")
        print(f"  RM3 added   : {', '.join(r['expansion'])}")

    print(f"\n=== biggest RM3 regressions ===")
    for r in sorted(rows, key=lambda r: r["ndcg_rm3"] - r["ndcg_bm25"])[:args.worst]:
        print(f"  [{r['qid']}] {r['ndcg_bm25']:.4f} -> {r['ndcg_rm3']:.4f} "
              f"({r['ndcg_rm3']-r['ndcg_bm25']:+.4f})  {r['query'][:55]}")
        print(f"         added: {', '.join(r['expansion'][:8])}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
