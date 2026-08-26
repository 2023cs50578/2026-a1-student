"""
tests/test_retrievers.py — correctness tests for the required components
(assignment Section 9: "Boolean/VSM and BM25 retrievers are both correctly
implemented and independently verifiable — graded by unit tests against known
small examples, not leaderboard score alone").

Everything here is checked against a three-document corpus small enough to work
through by hand, with the expected values derived in the comments rather than
copied from a previous run of this code. A test that just records whatever the
implementation currently prints proves nothing.

The tiny corpus, indexed with stopwording and stemming OFF so the term
statistics are exactly what the raw text says:

    d1  "alpha beta beta"           |d1| = 3
    d2  "alpha gamma"               |d2| = 2
    d3  "beta gamma gamma gamma"    |d3| = 4

    N = 3,  avgdl = 3,  df(alpha) = 2,  df(beta) = 2,  df(gamma) = 2
"""
import math
import os
import tempfile

import pytest

from submission import bm25, boolean_vsm, codecs, custom_scorer
from submission.indexer import InvertedIndex, analyze, tokenize
from submission.porter import stem

TINY_CORPUS = [
    ("d1", "alpha beta beta"),
    ("d2", "alpha gamma"),
    ("d3", "beta gamma gamma gamma"),
]


@pytest.fixture
def tiny_index():
    """A built-and-reloaded index over TINY_CORPUS.

    Deliberately goes through save() and load() rather than handing back the
    freshly built object: every test below then exercises the same query-time
    shape (memory-mapped, VByte-decoded postings) the harness will use.
    """
    index = InvertedIndex()
    index.build(TINY_CORPUS, remove_stopwords=False, stemming=False, forward_terms_per_doc=8)
    with tempfile.TemporaryDirectory() as index_dir:
        index.save(index_dir)
        loaded = InvertedIndex.load(index_dir)
        bm25.build(loaded)
        boolean_vsm.build(loaded)
        custom_scorer.build(loaded)
        yield loaded
        loaded.close()


# ---------------------------------------------------------------------------
# Index construction and statistics
# ---------------------------------------------------------------------------
def test_collection_statistics(tiny_index):
    assert tiny_index.N == 3
    assert tiny_index.terms == ["alpha", "beta", "gamma"]
    assert list(tiny_index.doc_len) == [3, 2, 4]
    assert tiny_index.avg_doc_len == pytest.approx(3.0)


def test_document_frequency(tiny_index):
    assert tiny_index.document_frequency("alpha") == 2
    assert tiny_index.document_frequency("beta") == 2
    assert tiny_index.document_frequency("gamma") == 2
    assert tiny_index.document_frequency("delta") == 0


def test_postings_carry_term_frequencies_in_ascending_doc_order(tiny_index):
    doc_ids, freqs = tiny_index.postings_for("beta")
    assert list(doc_ids) == [0, 2]      # d1 and d3
    assert list(freqs) == [2, 1]        # beta occurs twice in d1, once in d3

    doc_ids, freqs = tiny_index.postings_for("gamma")
    assert list(doc_ids) == [1, 2]
    assert list(freqs) == [1, 3]


def test_postings_survive_the_save_load_round_trip():
    """The property the harness's two-process design actually tests."""
    built = InvertedIndex()
    built.build(TINY_CORPUS, remove_stopwords=False, stemming=False)
    before = {t: (list(built.postings_for(t)[0]), list(built.postings_for(t)[1]))
              for t in built.terms}

    with tempfile.TemporaryDirectory() as index_dir:
        built.save(index_dir)
        loaded = InvertedIndex.load(index_dir)
        try:
            assert loaded.terms == built.terms
            assert list(loaded.doc_ids) == list(built.doc_ids)
            assert list(loaded.doc_len) == list(built.doc_len)
            for term, expected in before.items():
                doc_ids, freqs = loaded.postings_for(term)
                assert (list(doc_ids), list(freqs)) == expected
        finally:
            loaded.close()


# ---------------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------------
def test_tokenizer_lowercases_and_splits_on_non_alphanumerics():
    assert tokenize("SARS-CoV2, and COVID-19!") == ["sars", "cov2", "and", "covid", "19"]


def test_analyzer_removes_stopwords_and_stems():
    # "what", "is", "the", "of" are stopwords; the rest are stemmed.
    assert analyze("What is the origin of COVID-19 vaccines") == [
        "origin", "covid", "19", "vaccin"
    ]


def test_porter_stemmer_conflates_inflections():
    # The point of stemming: these must all collapse to one index term.
    assert stem("vaccine") == stem("vaccines")
    assert stem("connect") == stem("connected") == stem("connecting") == "connect"
    assert stem("ponies") == "poni"      # step 1a: IES -> I
    assert stem("caresses") == "caress"  # step 1a: SSES -> SS
    assert stem("agreed") == "agre"      # step 1b EED -> EE, then step 5a
    assert stem("hopping") == "hop"      # step 1b *d cleanup
    assert stem("relational") == "relat"  # step 2 ATIONAL -> ATE, step 5a
    assert stem("is") == "is"            # too short to strip


# ---------------------------------------------------------------------------
# Boolean retrieval
# ---------------------------------------------------------------------------
def test_boolean_and_returns_only_documents_containing_every_term(tiny_index):
    # alpha is in {d1, d2}; beta is in {d1, d3}; the intersection is {d1}.
    assert boolean_vsm.boolean_search("alpha beta", mode="and") == ["d1"]
    # beta {d1,d3} AND gamma {d2,d3} -> {d3}
    assert boolean_vsm.boolean_search("beta gamma", mode="and") == ["d3"]


def test_boolean_or_returns_the_union(tiny_index):
    assert boolean_vsm.boolean_search("alpha beta", mode="or") == ["d1", "d2", "d3"]
    assert boolean_vsm.boolean_search("alpha", mode="or") == ["d1", "d2"]


def test_boolean_and_with_an_unknown_term_is_empty(tiny_index):
    # A term that appears in no document makes the conjunction unsatisfiable.
    assert boolean_vsm.boolean_search("alpha delta", mode="and") == []
    assert boolean_vsm.boolean_search("alpha delta", mode="or") == ["d1", "d2"]


def test_boolean_rejects_an_unknown_mode(tiny_index):
    with pytest.raises(ValueError):
        boolean_vsm.boolean_search("alpha", mode="xor")


# ---------------------------------------------------------------------------
# BM25 — checked against arithmetic done by hand
# ---------------------------------------------------------------------------
def test_bm25_matches_hand_computed_scores(tiny_index):
    """Query "beta", k1 = 1.2, b = 0.75.

        idf(beta) = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6)

        d1: tf = 2, |d| = 3, |d|/avgdl = 1.0
            norm  = 1.2 * (1 - 0.75 + 0.75 * 1.0) = 1.2
            score = ln(1.6) * 2 * 2.2 / (2 + 1.2)      = ln(1.6) * 1.375

        d3: tf = 1, |d| = 4, |d|/avgdl = 4/3
            norm  = 1.2 * (0.25 + 0.75 * 4/3) = 1.5
            score = ln(1.6) * 1 * 2.2 / (1 + 1.5)      = ln(1.6) * 0.88

        d2 contains no "beta" at all and must not appear.
    """
    idf = math.log(1.6)
    results = bm25.score("beta", k=10, k1=1.2, b=0.75)

    assert [doc_id for doc_id, _s in results] == ["d1", "d3"]
    assert results[0][1] == pytest.approx(idf * 1.375, rel=1e-6)
    assert results[1][1] == pytest.approx(idf * 0.88, rel=1e-6)


def test_bm25_sums_over_query_terms(tiny_index):
    """A two-term query scores each term independently and adds them, so d3
    (which has both beta and gamma) must outrank d1 and d2 (one each)."""
    results = dict(bm25.score("beta gamma", k=10, k1=1.2, b=0.75))
    idf = math.log(1.6)

    # d3: beta tf=1 -> idf*0.88 (as above); gamma tf=3, same norm 1.5:
    #     idf * 3 * 2.2 / (3 + 1.5) = idf * 1.4666...
    assert results["d3"] == pytest.approx(idf * (0.88 + 3 * 2.2 / 4.5), rel=1e-6)
    assert results["d3"] > results["d1"] > results["d2"]


def test_k1_zero_makes_term_frequency_irrelevant(tiny_index):
    """With k1 = 0 the tf factor collapses to tf*1/(tf+0) = 1, so BM25 becomes
    a pure "does the term occur" model — the prediction an examiner would ask
    for when perturbing k1."""
    results = dict(bm25.score("beta", k=10, k1=0.0, b=0.75))
    # Tolerances are float32-scale: scores accumulate in a float32 array,
    # which is ample for ranking and halves the accumulator's memory traffic.
    assert results["d1"] == pytest.approx(results["d3"], rel=1e-6)
    assert results["d1"] == pytest.approx(math.log(1.6), rel=1e-6)


def test_b_zero_disables_length_normalisation(tiny_index):
    """With b = 0 the denominator no longer depends on |d|, so the longer
    document is no longer penalised for its length."""
    with_norm = dict(bm25.score("gamma", k=10, k1=1.2, b=0.75))
    without = dict(bm25.score("gamma", k=10, k1=1.2, b=0.0))

    # d3 (|d|=4, above average) is held back by length normalisation; turning
    # it off must raise d3's score and leave d2 (|d|=2) relatively worse off.
    assert without["d3"] > with_norm["d3"]
    assert without["d2"] < with_norm["d2"]


def test_higher_k1_rewards_repeated_occurrences_more(tiny_index):
    """Raising k1 delays saturation, so the gap between a document with many
    occurrences and one with few must widen."""
    low = dict(bm25.score("gamma", k=10, k1=0.5, b=0.0))
    high = dict(bm25.score("gamma", k=10, k1=3.0, b=0.0))
    # d3 has gamma 3 times, d2 once.
    assert high["d3"] / high["d2"] > low["d3"] / low["d2"]


def test_bm25_idf_is_never_negative(tiny_index):
    """The +1-smoothed RSJ form must stay non-negative even for a term in
    every document — otherwise containing a query word could hurt."""
    index = InvertedIndex()
    index.build([("a", "x y"), ("b", "x z")], remove_stopwords=False, stemming=False)
    with tempfile.TemporaryDirectory() as index_dir:
        index.save(index_dir)
        loaded = InvertedIndex.load(index_dir)
        try:
            bm25.build(loaded)
            # "x" appears in both documents: df == N.
            assert bm25.idf(loaded.term_id("x")) >= 0.0
            assert all(score >= 0 for _d, score in bm25.score("x", k=10))
        finally:
            loaded.close()
            bm25.build(tiny_index)  # restore the fixture's index for later tests


# ---------------------------------------------------------------------------
# Vector-space model
# ---------------------------------------------------------------------------
def test_vsm_cosine_matches_hand_computed_similarity(tiny_index):
    """Query "beta gamma", ltc.ltc weighting, idf = log10(3/2) for every term
    (all three have df = 2).

        w(t,d) = (1 + log10 tf) * idf
        d1: alpha 1 -> idf          beta 2 -> 1.30103 * idf
        d2: alpha 1 -> idf          gamma 1 -> idf
        d3: beta 1 -> idf           gamma 3 -> 1.477121 * idf

    d3 is the only document containing both query terms, and its "gamma"
    weight is the largest in the collection, so it must rank first. d2 shares
    only gamma, d1 only beta.
    """
    idf = math.log10(3 / 2)
    d3_norm = math.sqrt(idf**2 + (1.477121 * idf) ** 2)
    query_norm = math.sqrt(2 * idf**2)
    expected_d3 = (idf * idf + idf * 1.477121 * idf) / (d3_norm * query_norm)

    results = bm25.top_k(boolean_vsm.vsm_score_array("beta gamma"), 10)
    assert [doc_id for doc_id, _s in results][0] == "d3"
    assert dict(results)["d3"] == pytest.approx(expected_d3, rel=1e-5)


def test_vsm_cosine_of_a_query_with_itself_is_one():
    """A document whose text is exactly the query must score cosine 1."""
    index = InvertedIndex()
    index.build([("same", "alpha beta"), ("other", "gamma delta")],
                remove_stopwords=False, stemming=False)
    with tempfile.TemporaryDirectory() as index_dir:
        index.save(index_dir)
        loaded = InvertedIndex.load(index_dir)
        try:
            boolean_vsm.build(loaded)
            bm25.build(loaded)
            results = dict(boolean_vsm.vsm_score("alpha beta", 10))
            assert results["same"] == pytest.approx(1.0, abs=1e-6)
        finally:
            loaded.close()


def test_vsm_ranks_by_similarity_not_by_raw_overlap(tiny_index):
    boolean_vsm.build(tiny_index)
    ranked = [doc_id for doc_id, _s in boolean_vsm.vsm_score("beta gamma", 10)]
    assert ranked[0] == "d3"
    assert set(ranked) == {"d1", "d2", "d3"}


# ---------------------------------------------------------------------------
# Forward index and relevance feedback
# ---------------------------------------------------------------------------
def test_forward_index_returns_a_documents_own_terms(tiny_index):
    term_ids, freqs = tiny_index.forward(2)  # d3 = "beta gamma gamma gamma"
    assert [tiny_index.terms[int(t)] for t in term_ids] == ["beta", "gamma"]
    assert list(freqs) == [1, 3]


def test_forward_index_keeps_only_the_most_frequent_terms():
    """Pruning must keep the highest-tf terms, which is what RM3 reads."""
    index = InvertedIndex()
    index.build([("d", "a a a b b c d e")], remove_stopwords=False,
                stemming=False, forward_terms_per_doc=2)
    term_ids, freqs = index.forward(0)
    assert [index.terms[int(t)] for t in term_ids] == ["a", "b"]
    assert list(freqs) == [3, 2]


def test_rm3_expansion_adds_terms_from_the_feedback_documents(tiny_index):
    """Querying "beta" should pull "gamma" into the expanded query: the top
    BM25 documents for beta are d1 and d3, and d3 is full of gamma."""
    custom_scorer.K1, custom_scorer.B = 1.2, 0.75
    custom_scorer.FB_DOCS, custom_scorer.FB_TERMS, custom_scorer.ALPHA = 2, 5, 0.5
    custom_scorer.MIN_EXPANSION_IDF = 0.0

    original = bm25.query_term_weights("beta")
    expanded = custom_scorer.expand_query(original, bm25.score_array("beta", 1.2, 0.75))

    assert expanded is not None
    expanded_terms = {tiny_index.terms[t] for t in expanded}
    assert "beta" in expanded_terms       # the original query is retained...
    assert "gamma" in expanded_terms      # ...and feedback vocabulary is added
    assert sum(expanded.values()) == pytest.approx(1.0, abs=1e-9)


def test_rm3_falls_back_to_bm25_when_nothing_matches(tiny_index):
    """A query with no indexable terms must return an empty list, not crash."""
    assert custom_scorer.score("zzz", 10) == []


def test_retrieve_never_repeats_a_doc_id(tiny_index):
    """The one thing the harness rejects outright (RUNTIME_ERROR)."""
    for query in ["beta", "beta gamma", "alpha beta gamma"]:
        doc_ids = [doc_id for doc_id, _s in custom_scorer.score(query, 10)]
        assert len(doc_ids) == len(set(doc_ids))


def test_ranking_is_deterministic_across_repeated_calls(tiny_index):
    first = custom_scorer.score("alpha beta gamma", 10)
    second = custom_scorer.score("alpha beta gamma", 10)
    assert first == second


# ---------------------------------------------------------------------------
# Compression codec
# ---------------------------------------------------------------------------
def test_vbyte_matches_the_lecture_worked_example():
    """docIDs 824, 829, 215406 -> gaps 824, 5, 214577, from the Retrieval-II
    "Variable Byte (VB) codes" slide."""
    encoded = codecs.vbyte_encode([824, 5, 214577])
    assert list(encoded) == [0b00000110, 0b10111000,
                            0b10000101,
                            0b00001101, 0b00001100, 0b10110001]


def test_vbyte_round_trips_and_both_implementations_agree():
    values = [0, 1, 127, 128, 129, 16383, 16384, 2**34] + list(range(0, 5000, 7))
    encoded = codecs.vbyte_encode(values)
    assert encoded == codecs.vbyte_encode_py(values)
    assert list(codecs.vbyte_decode(encoded)) == values
    assert codecs.vbyte_decode_py(encoded) == values


def test_vbyte_rejects_negative_values():
    with pytest.raises(ValueError):
        codecs.vbyte_encode([1, -2, 3])


def test_postings_codec_round_trips_doc_ids_and_frequencies():
    doc_ids = [3, 9, 10, 500, 100000]
    freqs = [1, 2, 1, 40, 3]
    encoded = codecs.encode_postings(doc_ids, freqs)
    decoded_ids, decoded_freqs = codecs.decode_postings(encoded, 0, len(encoded))
    assert list(decoded_ids) == doc_ids
    assert list(decoded_freqs) == freqs


def test_gap_encoding_actually_saves_space():
    """The reason postings store d-gaps rather than absolute doc ids: a dense
    postings list must cost close to one byte per doc id, not four."""
    doc_ids = list(range(0, 20000))
    freqs = [1] * len(doc_ids)
    encoded = codecs.encode_postings(doc_ids, freqs)
    assert len(encoded) / len(doc_ids) < 2.1  # ~1 byte per gap + 1 per freq


# ---------------------------------------------------------------------------
# Edge cases the grading corpus could plausibly contain
# ---------------------------------------------------------------------------
def test_documents_with_no_indexable_terms_do_not_break_anything():
    """A document that is entirely stopwords (or empty) has length 0. It must
    index, persist, and simply never match — not divide by zero."""
    index = InvertedIndex()
    index.build([("empty", ""), ("stops", "the of and to"), ("real", "vaccine trial")])
    with tempfile.TemporaryDirectory() as index_dir:
        index.save(index_dir)
        loaded = InvertedIndex.load(index_dir)
        try:
            bm25.build(loaded); boolean_vsm.build(loaded); custom_scorer.build(loaded)
            assert list(loaded.doc_len)[:2] == [0, 0]
            assert [d for d, _s in bm25.score("vaccine", 10)] == ["real"]
            assert boolean_vsm.vsm_score("vaccine", 10)[0][0] == "real"
            assert custom_scorer.score("vaccine", 10)[0][0] == "real"
        finally:
            loaded.close()


def test_query_of_only_stopwords_returns_empty_rather_than_crashing():
    index = InvertedIndex()
    index.build([("d", "vaccine trial")])
    with tempfile.TemporaryDirectory() as index_dir:
        index.save(index_dir)
        loaded = InvertedIndex.load(index_dir)
        try:
            bm25.build(loaded); custom_scorer.build(loaded)
            assert custom_scorer.score("the of and to", 10) == []
            assert bm25.score("the of and to", 10) == []
        finally:
            loaded.close()


def test_retrieve_respects_k():
    index = InvertedIndex()
    index.build([(f"d{i}", "vaccine trial data") for i in range(50)])
    with tempfile.TemporaryDirectory() as index_dir:
        index.save(index_dir)
        loaded = InvertedIndex.load(index_dir)
        try:
            bm25.build(loaded); custom_scorer.build(loaded)
            for k in (0, 1, 5, 10, 100):
                assert len(custom_scorer.score("vaccine", k)) <= k
        finally:
            loaded.close()


def test_a_doc_id_containing_a_newline_is_rejected_loudly():
    index = InvertedIndex()
    index.build([("good", "vaccine"), ("bad\nid", "trial")])
    with tempfile.TemporaryDirectory() as index_dir:
        with pytest.raises(ValueError, match="newline"):
            index.save(index_dir)
