#!/usr/bin/env python
"""
scripts/cross_dataset.py — robustness check across several collections.

The private leaderboard is scored on a collection we never see. Tuning on the
one released dev set (TREC-COVID) optimises for that collection's quirks —
hundreds of relevant documents per topic, long biomedical abstracts — and the
held-out board showed exactly that: 6th of 57 on dev nDCG, 20th of 21 on
held-out nDCG.

The defence against that is not less tuning but tuning for *robustness*: score
every candidate configuration on several unrelated public collections and
prefer the one that is best on average (or has the best worst case), rather
than the one that peaks on any single collection. A setting that generalises
across four different corpora is far more likely to generalise to a fifth.

Each proxy collection is downloaded with scripts/download_full_corpus.py
(`--dataset beir/<name>/test --out data/proxy/<name>`) and evaluated with the
exact submission code and the exact harness metrics.

Usage:
    python scripts/cross_dataset.py --datasets nfcorpus scifact fiqa scidocs arguana
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries
from submission import bm25, custom_scorer
from submission.indexer import InvertedIndex
from submission.retrieve import _stream_corpus


def load_or_build(name, root, cache):
    corpus = os.path.join(root, name, "corpus.jsonl")
    index_dir = os.path.join(cache, name)
    if not os.path.exists(os.path.join(index_dir, "meta.json")):
        t = time.perf_counter()
        ix = InvertedIndex(); ix.build(_stream_corpus(corpus)); ix.save(index_dir)
        print(f"  built {name} in {time.perf_counter()-t:.1f}s "
              f"({sum(os.path.getsize(os.path.join(index_dir,f)) for f in os.listdir(index_dir))/1e6:.1f} MB)")
    return InvertedIndex.load(index_dir)


def run(scorer, queries, qrels):
    out = {qid: scorer(text, 10) for qid, text in queries}
    a = evaluate_run(out, qrels, k=10)["aggregate"]
    return a["ndcg@10"], a["map@10"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["nfcorpus", "scifact", "fiqa", "scidocs", "arguana"])
    ap.add_argument("--root", default="data/proxy")
    ap.add_argument("--cache", default="/tmp/a1_proxy_index")
    ap.add_argument("--include-dev", action="store_true", help="also score TREC-COVID dev (data/full)")
    ap.add_argument("--out", default="runs/cross_dataset.json")
    args = ap.parse_args()

    # Candidate configurations. Each is (label, k1, b, rm3 params or None).
    configs = []
    for k1, b in [(0.9, 0.4), (1.2, 0.75), (1.5, 0.75), (2.0, 0.6)]:
        configs.append((f"BM25 k1={k1} b={b}", k1, b, None))
    for k1, b in [(1.2, 0.75), (2.0, 0.6)]:
        configs.append((f"BM25({k1},{b})+RM3 10/10/0.5", k1, b, (10, 10, 0.5)))
        configs.append((f"BM25({k1},{b})+RM3 10/20/0.6", k1, b, (10, 20, 0.6)))
        configs.append((f"BM25({k1},{b})+RM3 40/30/0.4 (shipped)", k1, b, (40, 30, 0.4)))

    datasets = list(args.datasets)
    roots = {d: args.root for d in datasets}
    if args.include_dev:
        datasets.append("trec-covid"); roots["trec-covid"] = "data"
    results = {}  # config -> dataset -> (ndcg, map)
    stats = {}
    for name in datasets:
        root = roots[name]
        dname = "full" if name == "trec-covid" else name
        print(f"\n== {name}")
        ix = load_or_build(dname, root, args.cache)
        bm25.build(ix); custom_scorer.build(ix)
        queries = read_queries(os.path.join(root, dname, "queries_dev.tsv"))
        qrels = read_qrels(os.path.join(root, dname, "qrels_dev.txt"))
        queries = [(q, t) for q, t in queries if q in qrels]
        rel = [sum(1 for v in qrels[q].values() if v > 0) for q, _ in queries]
        stats[name] = {"docs": ix.N, "queries": len(queries), "avgdl": round(ix.avg_doc_len, 1),
                       "rel_per_query": round(sum(rel) / max(len(rel), 1), 1)}
        print(f"  {ix.N:,} docs, {len(queries)} queries, avgdl {ix.avg_doc_len:.0f}, "
              f"{stats[name]['rel_per_query']} relevant/query")
        for label, k1, b, rm3 in configs:
            custom_scorer.K1, custom_scorer.B = k1, b
            if rm3 is None:
                custom_scorer.ENABLE_RM3 = False
            else:
                custom_scorer.ENABLE_RM3 = True
                custom_scorer.FB_DOCS, custom_scorer.FB_TERMS, custom_scorer.ALPHA = rm3
            nd, mp = run(custom_scorer.score, queries, qrels)
            results.setdefault(label, {})[name] = (nd, mp)
            print(f"  {label:<38} nDCG@10 {nd:.4f}  MAP@10 {mp:.4f}")
        custom_scorer.ENABLE_RM3 = True

    print("\n\n=== nDCG@10 by configuration (proxies only; TREC-COVID shown separately) ===")
    proxies = [d for d in datasets if d != "trec-covid"]
    head = f"{'configuration':<38}" + "".join(f"{d[:9]:>10}" for d in proxies) + f"{'MEAN':>8}{'WORST':>8}"
    if "trec-covid" in datasets: head += f"{'covid':>8}"
    print(head)
    rows = []
    for label in results:
        vals = [results[label][d][0] for d in proxies]
        mean = sum(vals) / len(vals); worst = min(vals)
        line = f"{label:<38}" + "".join(f"{v:10.4f}" for v in vals) + f"{mean:8.4f}{worst:8.4f}"
        if "trec-covid" in datasets: line += f"{results[label]['trec-covid'][0]:8.4f}"
        rows.append((mean, line))
    for _m, line in sorted(rows, reverse=True): print(line)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"stats": stats, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
