# Pinned, minimal image so the grading harness runs your submission
# identically regardless of local setup (assignment Section 5,
# "Containerisation"). Course staff run every submission through this
# same image at grading time.
FROM python:3.11-slim

WORKDIR /repo

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command: run the interface conformance + smoke-test suite
# against the toy set. Course staff override CMD to point at the real
# corpus/topics/qrels for scoring.
CMD ["python", "-m", "harness.run_harness", \
     "--corpus", "data/toy/corpus.jsonl", \
     "--queries", "data/toy/queries_dev.tsv", \
     "--qrels", "data/toy/qrels_dev.txt", \
     "--baseline-run", "data/toy/baseline_run_dev.trec", \
     "--run-out", "runs/dev_run.trec", \
     "--report-out", "runs/dev_report.json"]
