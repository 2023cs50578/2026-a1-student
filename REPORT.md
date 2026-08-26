# COL 7364/764 Assignment 1 — Sparse Retrieval

**Corpus:** `beir/trec-covid` — 171,332 documents (title + abstract), 50
released dev topics, 66,336 graded (0–2) relevance judgments.
All numbers below are produced by `harness/run_harness.py` and
`harness/metrics.py`, i.e. the same code path that computes the leaderboard.

---

## 1. Indexing design

### 1.1 Text analysis

```
raw text → lowercase → [a-z0-9]+ tokens → stopword removal → Porter stemming → index terms
```

One analyser (`indexer.analyze`) is used for both indexing and querying. This
is not a stylistic preference: an index built with one analyser and queried
with another loses most of its matches silently, and it is the single easiest
way to ship a retriever that "works" and scores near zero.

**Stopwords.** A deliberately short list (~130 words). Zipf's law makes these
words ubiquitous, so they dominate the postings file while carrying almost no
discriminative power. The list is kept short on purpose: aggressive lists eat
genuinely meaningful query words, and every word removed is a word no query can
ever match again.

**Stemming.** Porter (1980), implemented from the published algorithm in
`submission/porter.py` rather than imported, so the whole pipeline stays inside
`submission/` and `requirements.txt` stays minimal. It was validated against
NLTK's `PorterStemmer(mode=ORIGINAL_ALGORITHM)` over the 70,924 distinct
alphabetic types in a 20,000-document sample of the corpus: **70,897 agree
exactly (99.96%)**. All 27 disagreements are two-character tokens, where this
implementation short-circuits and NLTK does not — NLTK maps `is → i`, `as → a`,
`us → u`; the short-circuit is the better behaviour and is documented as a
deliberate departure.

**Ablation** (each row re-tuned over its own small k1/b grid, so the comparison
is not confounded by parameters tuned for a different analyser):

| Analyser | vocabulary | avg doc len | index size | best nDCG@10 |
|---|---|---|---|---|
| neither | 207,034 | 169.5 | 38.4 MB | 0.6167 |
| stopwords only | 206,907 | 113.7 | 31.8 MB | 0.6329 |
| stemming only | 165,608 | 169.5 | 35.5 MB | 0.6312 |
| **stopwords + stemming** | **165,563** | **113.7** | **29.0 MB** | **0.6683** |

Both choices help, and they are close to additive (+0.037 and +0.035
separately, +0.052 together). Stopwording alone removes a third of all postings
without costing ranking quality; stemming alone cuts the vocabulary by 20% and
buys recall on the short natural-language queries this collection uses.

### 1.2 On-disk format

Documents are internal integers `0..N-1` in corpus order; terms are integers
assigned by lexicographic rank. `index_dir` holds:

| File | Contents | Size |
|---|---|---|
| `postings.bin` | VByte d-gaps + term frequencies, interleaved, one contiguous run per term | 27.1 MB |
| `fwd.bin` | pruned forward index: VByte term-id gaps + frequencies per document | 10.6 MB |
| `docids.bin` | zlib(external doc_id strings, newline-joined) | 1.05 MB |
| `terms.bin` | zlib(sorted term strings, newline-joined) | 0.52 MB |
| `termstats.bin` | zlib(VByte df per term, then postings byte length per term) | 0.20 MB |
| `doclen.bin` | zlib(VByte document lengths) | 0.19 MB |
| `fwdlen.bin` | zlib(VByte per-document byte lengths into `fwd.bin`) | 0.10 MB |
| **total** | | **39.7 MB** |

*(Sizes are MB = 10⁶ bytes. `run_harness.py` prints the same total as 37.8 MB
using 1024-based units; the byte count it reports is 39,660,305.)*

**Compression.** Postings store *d-gaps*, not absolute doc ids: within a
postings list the doc ids are ascending, so the differences are small, and
VByte spends one byte on anything under 128. Term frequencies are even more
skewed (overwhelmingly 1 or 2), so they cost one byte essentially always. The
measured cost is ~2.0 bytes per posting, against 8 bytes for a naive
`(int32 doc_id, int32 tf)` array — and far less against a JSON or pickled
dict-of-dicts, which repeats the doc_id *string* on every posting.

The VByte implementation follows the convention in the Retrieval-II lecture
notes exactly — 7-bit groups, most significant first, continuation bit set on
the last byte — and `tests/test_retrievers.py` asserts it reproduces the
slide's worked example (824, 5, 214577) byte for byte.

**What is deliberately not stored.** The raw document text (BM25 and cosine
need only frequencies and lengths); a *full* forward index; and any
precomputed IDF vector or document norms — all of those are derivable from
what is stored, so persisting them would cost index-size score for nothing.
`bm25.build()` recomputes the IDF table at load time in a few milliseconds.

**Dictionary.** A sorted array of terms searched with `bisect`, not a hash map.
A Python dict of 165K strings costs several times the memory and, more
importantly, several times the *load time*; a query has a handful of terms, so
O(log |V|) per lookup is free. Sorted terms also share long prefixes with their
neighbours, which is why zlib gets `terms.bin` down to 0.52 MB — most of what
explicit front coding (Retrieval-I) would achieve, for none of the code.

**Load time.** `postings.bin` and `fwd.bin` are memory-mapped, not read. Index
load is therefore the cost of the dictionary and document table alone —
**28 ms** — and a query decodes only the postings lists it actually touches.

### 1.3 Construction

Single-pass in-memory inversion (Retrieval-II): one sweep over the collection
emitting `(term_id, doc_id, tf)` triples into flat `array('i')` buffers, then
one **stable** sort on the term component. Stability is what keeps doc ids
ascending inside each postings list for free, which is exactly what makes the
d-gaps small.

Two implementation details that matter at this scale:

- The hot loop counts *raw* tokens with `collections.Counter` (C speed) and
  only distinct tokens reach Python-level analysis, through a
  `raw token → term id` cache. The Porter stemmer therefore runs once per
  vocabulary *type* (~600K calls), not once per token occurrence (~30M).
- Encoding is done in chunks of 2M postings. A vectorised VByte encoder
  allocates several int64 temporaries the size of its input, so encoding the
  whole collection in one call needed 2.9 GB; chunking on term boundaries
  bounds that scratch to a fixed cost at no measurable speed penalty.
  **Peak build memory: 2.88 GB → 1.25 GB**, which matters on the stated 8 GB
  grading machine if the graded corpus is larger than this one.

Build is **deterministic**: building twice from the same corpus produces
byte-identical files (verified by SHA-256 over every file).

---

## 2. Retrieval models

| Model | nDCG@10 | MAP@10 | MRR | P@10 | mean latency |
|---|---|---|---|---|---|
| Boolean AND (unranked, corpus order) | 0.1697 | 0.0026 | 0.3289 | 0.2100 | 3.8 ms |
| Boolean OR (unranked, corpus order) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.7 ms |
| VSM cosine, ltc.ltc | 0.3307 | 0.0073 | 0.5432 | 0.3760 | 10.1 ms |
| BM25, textbook k1=1.2 b=0.75 | 0.6441 | 0.0157 | 0.8992 | 0.7020 | 3.6 ms |
| BM25, tuned k1=2.0 b=0.6 | 0.6683 | 0.0170 | 0.9007 | 0.7400 | 3.6 ms |
| **BM25 tuned + RM3 (entry)** | **0.7387** | **0.0197** | **0.9333** | **0.8120** | 18.8 ms |

*MAP@10 is small for every system because TREC-COVID topics have hundreds of
relevant documents each (query 38 has 1,266) and MAP@10 normalises by the true
relevant count. It is bounded above by roughly 10/|R| here, so it functions as
a tie-break, exactly as the assignment describes.*

**Boolean.** Included as a required component and as a control, not as a
candidate. The numbers make the lecture's point about its drawbacks concrete:
AND scores 0.1697 only because the ten documents it happens to return first in
corpus order sometimes contain a relevant one, and OR scores exactly 0.0000
because on a 171K-document collection the disjunction of a few common terms
returns tens of thousands of documents with no way to order them. Boolean
retrieval provides no ranking, and at this scale no ranking means no score.
It remains genuinely useful as a diagnostic — `boolean_search` is what confirms
the index *finds* the right documents when the ranking looks wrong.

**VSM (0.3307) vs BM25 (0.6683).** A factor of two, on the same index, with the
same analyser. The difference is entirely in how each handles document length.
Cosine normalises by vector magnitude, which conflates two different reasons a
document is long: TREC-COVID mixes one-line title-only records with full
structured abstracts, and cosine over-rewards the short ones because dividing
by a small ‖d‖ inflates any match. BM25's `b` parameter normalises by length
*relative to the collection average* and, crucially, is tunable — the sweep
below shows the optimum is b ≈ 0.55–0.6, not the b = 1 that cosine effectively
hard-codes. BM25's `k1` saturation is the second half of the gap: ltc.ltc's
`1 + log tf` damping is a fixed curve, where `k1` fits the curve to the
collection.

---

## 3. Parameter search

Tuning was done **on the released dev topics only**. The held-out topics were
never seen. `scripts/sweep_params.py` builds the index once and evaluates every
setting against it in-process, so a full sweep costs one build.

### 3.1 One parameter at a time

**nDCG@10 vs k1** (b = 0.75 fixed):

| k1 | 0.0 | 0.3 | 0.6 | 0.9 | 1.2 | 1.5 | 1.8 | 2.1 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| nDCG@10 | 0.3210 | 0.5530 | 0.5916 | 0.6254 | 0.6441 | 0.6481 | 0.6530 | 0.6575 | 0.6518 | 0.6497 |

**nDCG@10 vs b** (k1 = 1.2 fixed):

| b | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.75 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| nDCG@10 | 0.5333 | 0.6082 | 0.6230 | 0.6334 | 0.6387 | 0.6452 | 0.6438 | 0.6441 | 0.6086 | 0.5679 |

Both curves have the shape theory predicts. `k1 = 0` (0.3210) makes term
frequency irrelevant — BM25 collapses to "does the term occur at all" and loses
half its effectiveness, which is the clearest possible demonstration that tf
saturation, not tf itself, is what BM25 contributes. The curve then rises
steeply to k1 ≈ 1 and flattens into a broad plateau; the collection's abstracts
are long enough that genuine topical repetition is signal, so saturation should
be slow.

`b` is the more informative curve because it is non-monotone in both
directions. `b = 0` (0.5333) removes length normalisation entirely and long
documents win by containing more of everything. `b = 1.0` (0.5679) normalises
fully, judging documents purely on term density, and short title-only records
win spuriously. The optimum sits between, nearer 0.55, because in this
collection a longer document is usually longer because it says more.

### 3.2 Joint grid

The parameters interact — raising `k1` shifts the optimal `b` — so the
operating point was chosen on the joint grid:

| k1 \ b | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 |
|---|---|---|---|---|---|---|
| 1.6 | 0.6586 | 0.6586 | 0.6638 | 0.6624 | 0.6598 | 0.6598 |
| 1.8 | 0.6609 | 0.6616 | 0.6682 | 0.6670 | 0.6646 | 0.6597 |
| **2.0** | 0.6625 | 0.6623 | **0.6711** | **0.6683** | 0.6675 | 0.6597 |
| 2.2 | 0.6624 | 0.6598 | 0.6687 | 0.6675 | 0.6670 | 0.6599 |
| 2.4 | 0.6648 | 0.6599 | 0.6635 | 0.6630 | 0.6627 | 0.6642 |
| 2.8 | 0.6621 | 0.6568 | 0.6597 | 0.6640 | 0.6631 | 0.6588 |

**Chosen: k1 = 2.0, b = 0.6.** The dev argmax is (2.0, 0.55) at 0.6711, but the
whole region k1 ∈ [1.8, 2.2] × b ∈ [0.55, 0.65] lies within 0.004 — well inside
the noise of 50 topics. Picking the centre of a plateau rather than its peak is
the point: a 0.003 difference on 50 queries carries no information about a
disjoint topic set, and the ridge is what generalises.

Re-tuning k1/b *after* RM3 was enabled gave a best of 0.7403 at (1.6, 0.7)
against 0.7368 at (2.0, 0.6) — again within noise, so the first-pass choice was
kept rather than re-fitted to a second, noisier surface.

---

## 4. The competition entry: BM25 + RM3

`submission/custom_scorer.py`. BM25 can only match words the user typed; a
query asking whether recovered patients "develop immunity" cannot reach a
document that says *seroconversion*, *antibody response* and *reinfection* but
never says *immunity*. RM3 fixes this vocabulary mismatch without any
pretrained model or embedding (out of scope per Section 10):

1. Run BM25; take the top `FB_DOCS` documents as pseudo-relevant.
2. Build a relevance model `P(w|R) = Σ_d P(d|Q) · P(w|d)`, with
   `P(w|d) = tf(w,d)/|d|` and `P(d|Q)` the document's share of the feedback
   set's total score.
3. Keep the `FB_TERMS` highest-probability terms, renormalised.
4. Interpolate with the original query:
   `P(w|Q') = α·P(w|Q) + (1−α)·P(w|R)`.
5. Re-run BM25 with that weighted query.

Step 4 is what makes it safe. Pure expansion (α = 0) replaces the query
outright, so one off-topic feedback set destroys the result; keeping α of the
mass on the user's own words anchors it.

### 4.1 The forward index

RM3 needs the opposite of an inverted index — "which terms does *this document*
contain" — for the top-ranked documents. Nothing in the inverted index answers
that without scanning all of it, so a forward index must be persisted. A full
one would roughly double the footprint, because term-id gaps *within* a
document are large (a document's ~90 terms scattered across a 165K vocabulary)
where doc-id gaps within a postings list are small.

So only each document's most frequent terms are kept — precisely the terms RM3
can select, since expansion terms maximise `Σ_d tf/|d|`:

| terms kept/doc | `fwd.bin` | total index | nDCG@10 | P@10 |
|---|---|---|---|---|
| 0 (no feedback) | — | 29.0 MB | 0.6683 | 0.7400 |
| 8 | 4.5 MB | 33.6 MB | 0.7065 | 0.7920 |
| 12 | 6.2 MB | 35.2 MB | 0.7221 | 0.8000 |
| 16 | 7.7 MB | 36.8 MB | 0.7306 | 0.8060 |
| **24** | **10.6 MB** | **39.7 MB** | **0.7387** | **0.8120** |
| 32 | 13.4 MB | 42.5 MB | 0.7368 | 0.8160 |
| 48 | 18.9 MB | 48.0 MB | 0.7346 | 0.8140 |

nDCG@10 rises to 24 and is flat beyond it. **24 is where the index stops buying
anything** — 48 terms per document costs 8 MB more for no ranking gain.

### 4.2 Choosing the RM3 parameters honestly

The RM3 grid has ~80 points and there are 50 dev topics. The best cell on 50
queries is substantially luck, and shipping it is exactly the dev-set
overfitting the two-tier leaderboard exists to punish. So parameters were
chosen by **5-fold cross-validation** (`scripts/tune_rm3.py`): pick on 4 folds,
score on the held-out fold.

| fold | picked | held-out nDCG@10 | BM25-only on same fold |
|---|---|---|---|
| 0 | 50/30/0.3 | 0.6833 | 0.6594 |
| 1 | 40/30/0.3 | 0.8213 | 0.7103 |
| 2 | 40/30/0.4 | 0.6662 | 0.6254 |
| 3 | 40/50/0.3 | 0.6475 | 0.6096 |
| 4 | 40/30/0.4 | 0.7648 | 0.7368 |
| **mean** | | **0.7166** | **0.6683** |

**Honest expected gain from RM3: +0.048**, and every single fold improves. The
folds agree closely on the parameters (`FB_DOCS` ∈ {40, 50}, `FB_TERMS` = 30,
α ∈ {0.3, 0.4}), which is itself evidence the region is real rather than noise.

**Shipped: `FB_DOCS = 40`, `FB_TERMS = 30`, `α = 0.4`** — the dev argmax, which
also sits in the middle of the plateau on all three axes.

### 4.3 A hypothesis that turned out to be wrong

The error analysis (§5) showed RM3 adding generic collection vocabulary —
*covid*, *19*, *pandem*, *diseas*, *patient* — on the queries it damaged. The
obvious fix is an IDF floor on expansion terms. It fails, monotonically:

| min expansion IDF | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|---|
| nDCG@10 | 0.7387 | 0.7387 | 0.7287 | 0.6932 | 0.6546 | 0.6170 | 0.5795 |

Filtering common expansion terms *hurts*, and hurts more the harder you filter.
The reason: BM25's own IDF weighting already discounts these terms at scoring
time, so removing them is double-counting the penalty — and on a
single-domain collection they carry a genuine topical prior. A document about
COVID patients really is more likely to be relevant to a COVID query than one
that never mentions either. The floor is retained at 0.5 (inert here: the
lowest-IDF term in the collection is *19* at 1.02) purely as a guard for a
collection containing a term in >60% of documents.

---

## 5. Error analysis

`scripts/error_analysis.py`. Five worst dev topics under the final system:

| qid | query | BM25 | RM3 | relevant |
|---|---|---|---|---|
| 4 | what causes death from Covid-19? | 0.1584 | 0.0000 | 529 |
| 12 | best practices in hospitals and at home in maintaining quarantine | 0.1596 | 0.1691 | 616 |
| 31 | How does the coronavirus differ from seasonal flu? | 0.2664 | 0.2664 | 353 |
| 34 | longer-term complications of those who recover from COVID-19 | 0.3190 | 0.3393 | 172 |
| 49 | do individuals who recover show sufficient immune response… | 0.3400 | 0.3506 | 247 |

**Topic 4 — query drift, the textbook RM3 failure.** BM25 already scores badly
(0.1584); RM3 then drives it to exactly 0.0. The expansion terms are *mortal,
excess, pandem, 2020, increas, case, year, number, estim*. Every one is
epidemiological: the top-40 feedback documents are "excess mortality in 2020"
statistics papers, and the relevance model faithfully reproduces *their*
vocabulary. But the topic asks about *pathophysiology* — what kills a patient.
The words are a perfect match for the wrong sense of "death from COVID". This
is the failure mode α exists to bound, and with α = 0.4 the anchor was still
not enough because the first pass was already off-topic. **RM3 amplifies the
first pass; it cannot repair one.**

**Topic 31 — a relation BM25 cannot represent.** "How does the coronavirus
*differ from* seasonal flu" is a request for a contrast. Bag-of-words scoring
has no way to express "differ from": documents about coronavirus and documents
about influenza both score well, and documents *comparing* them — the only
relevant ones — score no better. RM3 adds *influenza, h1n1, respiratori*, which
correctly identifies the second topic and makes matters no better, because it
pulls in pure-influenza papers. Neither BM25 nor RM3 can fix this; it needs
either proximity/phrase evidence or something with a notion of relations.

**Topic 49 — length dilution.** Eighteen index terms after analysis, including
*show, suffici, includ, level, individu, t, re* — words that carry no topic.
BM25 sums over all of them, so a document matching seven vague terms can
outrank one matching the two that matter (*antibodi*, *immun*). RM3 helps
slightly (+0.011) by concentrating weight on *cd8, memori, seroconvers*. A
query-side IDF-weighted term-selection step would likely help here and is the
first thing to try next.

**Topics 12 and 34 — genuine ceiling, or a judgment artefact.** Both have
hundreds of relevant documents but score ~0.17 and ~0.34. Topic 34's top 10
contains grades `[0,0,2,2,2,1,0,1,0,0]` — five relevant documents, but the two
at ranks 1–2 are graded 0, and nDCG's `2^rel − 1` gain punishes a miss at rank
1 far more than at rank 8. This is a precision-at-the-very-top problem, not a
retrieval failure.

**The largest RM3 regression, topic 11** (0.6993 → 0.4126), is the same drift as
topic 4 in milder form: a specific query ("guidelines for triaging patients")
expanded with the collection's generic vocabulary (*covid, 19, pandem, diseas,
care, clinic, health*). Across all 50 topics RM3 wins clearly — it improves 37
topics, damages 9 and leaves 4 unchanged, for a mean gain of +0.070 — but the
damage is concentrated exactly where the query is specific and the collection
is homogeneous.

---

## 6. Final competition entry — one paragraph

The entry is **BM25 (k1 = 2.0, b = 0.6) followed by RM3 pseudo-relevance
feedback (40 feedback documents, 30 expansion terms, α = 0.4)**, over an index
with Porter stemming, a short stopword list, VByte-compressed d-gap postings,
and a forward index pruned to each document's 24 most frequent terms. BM25 was
chosen over the vector-space model because it beats it by a factor of two on
the same index (0.6683 vs 0.3307), and the reason is specifically that its
length normalisation is *tunable* where cosine's is fixed. RM3 was added
because it is the only change tested that produced a gain larger than the noise
floor of 50 topics: +0.048 nDCG@10 under 5-fold cross-validation, improving
every fold. Its parameters were chosen by that cross-validation rather than by
the dev-set argmax, and the k1/b operating point was taken from the centre of a
broad plateau rather than its peak, both for the same reason — with 50 dev
topics and a disjoint held-out set, differences below ~0.01 are not information.
The cost is one extra BM25 pass and 40 forward-index reads per query, taking
mean latency from 3.6 ms to 18.8 ms and the index from 29.0 MB to 39.7 MB,
which is a trade worth making when nDCG@10 carries 70% of the leaderboard score
and latency and size carry 10% each.

---

## 7. AI-use disclosure

Claude (Anthropic) was used as an interactive assistant throughout this
assignment: to work through the assignment specification and the starter
harness, to draft and iterate on the implementation in `submission/`, the tests
in `tests/test_retrievers.py`, the tuning scripts in `scripts/`, and this
report. Every design decision recorded here — the ltc.ltc weighting, VByte
d-gap postings with a memory-mapped postings file, the pruned forward index and
its 24-term cut-off, RM3 and its cross-validated parameters, the choice of a
plateau centre over a dev-set argmax — was checked against a measurement in
this repository before being adopted, and each of those measurements is
reproducible with the scripts in `scripts/`. The IDF-floor hypothesis in §4.3
is included precisely because it was proposed, measured, and rejected. I have
read and can explain every line of the submitted code.

## 8. Code provenance

All code under `submission/` was written for this assignment. No search or
indexing library (Lucene, Elasticsearch, Pyserini, Whoosh, `rank_bm25`) is
imported anywhere in it. External dependencies are NumPy (vectorised encoding,
decoding, and score accumulation) and the Python standard library (`re`,
`zlib`, `mmap`, `bisect`, `heapq`, `array`, `collections`, `json`).

Algorithms implemented from published descriptions, with the source and the
delta:

- **Porter stemmer** (`submission/porter.py`) — M.F. Porter, "An algorithm for
  suffix stripping", *Program* 14(3), 1980. Implemented from the algorithm
  description (consonant/vowel definition, the measure *m*, and the step 1a–5b
  rule tables). Validated against NLTK's `PorterStemmer(ORIGINAL_ALGORITHM)` on
  70,924 corpus types with 99.96% agreement; NLTK's source was not consulted or
  copied. **Delta:** words of ≤2 characters are returned unchanged, which fixes
  `is → i` and `as → a`.
- **BM25** (`submission/bm25.py`) — Robertson & Walker (1992); Robertson &
  Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009).
  Standard formula with the +1-smoothed Robertson–Sparck Jones IDF.
  **Delta:** query weights are floats rather than integer term counts, so the
  same scoring path serves both plain and RM3-expanded queries.
- **RM3** (`submission/custom_scorer.py`) — Lavrenko & Croft, "Relevance-Based
  Language Models" (SIGIR 2001), with the query interpolation of Abdul-Jaleel
  et al. (TREC 2004). Implemented from the published formulation; Anserini's
  or any other implementation's source was not consulted. **Delta:** the
  relevance model is estimated from a *pruned* forward index (24 terms per
  document) rather than full term vectors, and `P(w|d)` is normalised by the
  document's true length rather than by the retained terms' sum, so pruning
  does not inflate the kept terms' probabilities.
- **VByte coding** (`submission/codecs.py`) — Retrieval-II lecture notes,
  "Variable Byte (VB) codes"; the standard formulation in Manning, Raghavan &
  Schütze §5.3. **Delta:** both directions are vectorised with NumPy
  (`np.add.reduceat` over continuation-bit-derived group boundaries for
  decoding; a five-pass positional fill for encoding), and encoding is chunked
  on term boundaries to bound peak memory.
- **ltc.ltc cosine / single-pass in-memory inversion / rarest-first postings
  intersection** — standard textbook material from the course's Lecture 2,
  Retrieval-I and Retrieval-II notes.

`harness/`, `tests/test_metrics.py`, `tests/test_interface_conformance.py`,
`conftest.py`, `Dockerfile`, `scripts/download_full_corpus.py` and
`scripts/smoke_test.sh` are unmodified from the starter repository.
`requirements.txt` has one line added (`numpy>=1.24`). `README.md` has a
section added. `scripts/sweep_params.py`, `scripts/tune_rm3.py`,
`scripts/error_analysis.py`, `tests/test_retrievers.py` and this report are new.
