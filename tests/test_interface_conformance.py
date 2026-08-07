"""
tests/test_interface_conformance.py — the exact check the CI conformance
job runs on every push (assignment Section 5, "Continuous conformance
checking"). If this file passes, your submission has the right shape for
the grading harness to run it; it says nothing about ranking quality.

Run it yourself any time with:
    pytest tests/test_interface_conformance.py -v
"""
import os

import pytest

from harness.run_harness import check_conformance, run_submission
from harness.trec_io import read_queries
from submission import retrieve as submission_module

TOY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "toy")
CORPUS_PATH = os.path.join(TOY_DIR, "corpus.jsonl")
QUERIES_PATH = os.path.join(TOY_DIR, "queries_dev.tsv")


def test_required_functions_exist_with_correct_signature():
    problems = check_conformance(submission_module)
    assert not problems, f"Interface conformance problems: {problems}"


def test_build_index_and_retrieve_run_without_crashing():
    queries = read_queries(QUERIES_PATH)
    run, build_time, latencies = run_submission(submission_module, CORPUS_PATH, queries, k=10)

    assert build_time >= 0
    assert len(latencies) == len(queries)
    assert set(run.keys()) == {qid for qid, _text in queries}


def test_retrieve_returns_well_formed_results():
    queries = read_queries(QUERIES_PATH)
    run, _build_time, _latencies = run_submission(submission_module, CORPUS_PATH, queries, k=5)

    # Load valid doc_ids to check against.
    from harness.trec_io import read_corpus
    valid_doc_ids = {doc_id for doc_id, _text in read_corpus(CORPUS_PATH)}

    for qid, results in run.items():
        assert len(results) <= 5, f"qid={qid} returned more than k results"
        scores = [score for _doc_id, score in results]
        assert scores == sorted(scores, reverse=True), f"qid={qid} results not sorted descending"
        for doc_id, score in results:
            assert doc_id in valid_doc_ids, f"qid={qid} returned unknown doc_id {doc_id!r}"
            assert isinstance(score, (int, float)), f"qid={qid} score {score!r} is not numeric"


def test_retrieve_is_reasonably_fast_on_the_toy_set():
    """Not a correctness check — just catches an accidentally quadratic-in-corpus-size
    per-query implementation before it becomes a problem on the real corpus."""
    queries = read_queries(QUERIES_PATH)
    _run, build_time, latencies = run_submission(submission_module, CORPUS_PATH, queries, k=10)
    assert build_time < 30, "Index build on a 20-document toy corpus took >30s — something is off."
    assert max(latencies) < 5, "A single query on a 20-document toy corpus took >5s — something is off."
