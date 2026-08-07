# A1 — Sparse Retrieval Arena: Starter Repository

This is the starter repository for **Assignment 1: Sparse Retrieval
Arena**. If anything here conflicts with the assignment spec document,
the spec document governs the rules (grading, deadlines, integrity); this
repo governs the exact code interface, which the spec explicitly defers
to it for ("exact signature given in the starter repo").

## What you're building

An inverted-index retrieval engine with Boolean/vector-space, BM25, and
language-model scorers, tuned to beat the instructor's baseline (and
ideally the rest of the class) on nDCG@10. Full requirements are in the
assignment spec — start there if you haven't read it yet.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # or your preferred env manager
pip install -r requirements.txt

# Run the full harness against the toy set — should work immediately,
# using the trivial baseline that ships in submission/retrieve.py.
bash scripts/smoke_test.sh
```

You should see a report ending in something like:

```
nDCG@10:              0.29..   (mediocre — this ignores the query entirely, not a real retriever)
...
Provisional score (90% weight, nDCG@10 + MAP): 0.23..
Baseline nDCG@10:     0.86..  (you beat it: False)
```

The trivial baseline isn't near zero — on a 20-document toy corpus, returning
"the first 10 documents" already overlaps with some relevant ones by chance.
It should still be clearly and consistently behind the baseline. That gap is
what your job is to close and then overtake.

## Where to write code

Everything you implement lives in `submission/`:

| File | What goes here |
|---|---|
| `submission/retrieve.py` | **The required entrypoint** (`build_index`, `retrieve`). Wire your real scorer in here — see the `TODO(you)` markers. Do not change its function signatures. |
| `submission/indexer.py` | Your inverted index: postings, document lengths, collection stats. |
| `submission/boolean_vsm.py` | Boolean AND/OR retrieval + TF-IDF cosine vector-space ranking. |
| `submission/bm25.py` | BM25 with tunable `k1`, `b`. |
| `submission/language_model.py` | Query-likelihood ranking with Jelinek-Mercer or Dirichlet smoothing. |
| `submission/custom_scorer.py` | Optional: your own combined/heuristic scorer. |

Every file above has a docstring with the relevant formula and a
reference back to the assignment section it satisfies — read those before
you start.

**You may not use an existing search/indexing library** (Lucene,
Elasticsearch, Pyserini, Whoosh, `rank_bm25`, etc.) inside `submission/`.
Standard libraries for tokenisation/stemming (e.g. NLTK) and numeric
libraries (NumPy) are fine. See the assignment's Academic Integrity
section for the full policy, including AI-use disclosure and code
provenance requirements.

## Running the harness yourself

```bash
python -m harness.run_harness \
  --corpus data/toy/corpus.jsonl \
  --queries data/toy/queries_dev.tsv \
  --qrels data/toy/qrels_dev.txt \
  --baseline-run data/toy/baseline_run_dev.trec \
  --run-out runs/dev_run.trec \
  --report-out runs/dev_report.json
```

This is the *exact* code path (`harness/run_harness.py`,
`harness/metrics.py`) used to compute your leaderboard score — the only
things that differ at real grading time are which corpus/topics/qrels
file is passed in (the released dev set, then later the private held-out
set you never see) and that course infrastructure runs it for you rather
than you running it locally. See `harness/metrics.py` for exactly how
nDCG@10, MAP, MRR, and P@k are computed, and `harness/leaderboard.py` for
how they combine into your leaderboard score.

To test against the real assignment corpus instead of the toy set, run
`python scripts/download_full_corpus.py` first (see `data/README.md`),
then point `--corpus`/`--queries`/`--qrels` at `data/full/` instead.

## Before you push: run the smoke test

```bash
bash scripts/smoke_test.sh
```

This runs the same interface-conformance tests, metrics unit tests, and
full harness pass that CI runs on every push
(`.github/workflows/conformance.yml`). Fix anything it flags before your
conformance freeze (48 hours before the deadline — see
`docs/SUBMISSION_INTERFACE.md`).

## Repository layout

```
.
├── data/
│   ├── toy/                 # small hand-built set for fast local dev (ships here)
│   ├── README.md            # data format + how to get the real corpus
│   └── full/                # created by scripts/download_full_corpus.py (gitignored)
├── submission/               # <-- you write code here
├── harness/                  # scoring code (read-only reference; don't need to edit)
├── tests/                    # conformance + metrics unit tests
├── scripts/
│   ├── download_full_corpus.py
│   └── smoke_test.sh
├── docs/
│   └── SUBMISSION_INTERFACE.md   # the exact, binding interface contract
├── Dockerfile                 # how course staff run every submission
└── .github/workflows/conformance.yml   # what runs on every push
```

## Getting help

Discussing high-level strategy with classmates is fine. Sharing code, a
tuned parameter file, or your `submission/` implementation is not — see
the assignment's Academic Integrity section, and remember every team sits
a short oral defense after the leaderboard closes where you'll be asked
to explain and modify your own submission live.
