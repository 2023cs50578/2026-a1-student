# Submission Interface Contract

This is the canonical, binding version of the interface referenced in the
assignment spec (Section 5, "Submission Interface & Conformance
Checking"): *"exact signature given in the starter repo."* This file is
that reference.

## What the harness calls

Exactly two functions, in `submission/retrieve.py`, in this order:

```python
def build_index(corpus_path: str) -> None:
    """Called once, before any retrieve() calls."""

def retrieve(query: str, k: int = 10) -> list[tuple[str, float]]:
    """Called once per query, after build_index() has run."""
```

Nothing else about your submission is inspected by the harness — how you
organise `boolean_vsm.py`, `bm25.py`, `language_model.py`,
`custom_scorer.py`, or `indexer.py` internally, what you name your helper
functions, or how you structure your index, is entirely up to you, as
long as `retrieve.py` exposes these two entrypoints with these exact
names and this parameter order.

## Contract details

**`build_index(corpus_path)`**
- `corpus_path` is a path to a `corpus.jsonl` file (see `data/README.md`
  for the format).
- Called exactly once per harness run, before any `retrieve()` call.
- Its wall-clock time is measured and reported as your index-build-time
  efficiency metric — do expensive one-time work here, not in `retrieve()`.
- Must not require network access or any file other than `corpus_path`.

**`retrieve(query, k=10)`**
- `query` is a raw query string (not pre-tokenised).
- `k` is the number of results requested (the harness always passes an
  explicit value; the default is only there so you can call `retrieve()`
  manually while testing).
- Must return a `list` of `(doc_id, score)` tuples:
  - `doc_id` must be a `doc_id` value that appeared in the corpus passed
    to `build_index()`.
  - `score` must be numeric (`int` or `float`); higher = more relevant.
  - The list must have length `<= k`.
  - The harness re-sorts by score descending defensively, but you should
    return it already sorted — an unsorted return is a strong signal
    something is wrong.
- Must be deterministic: the same query, called twice against the same
  index, should return the same ranking. (Ties in score may break
  arbitrarily but consistently.)
- Must not read `corpus_path` again, hit the network, or depend on global
  state from a previous `retrieve()` call beyond what `build_index()` set up.

## What "conformance" means in practice

`tests/test_interface_conformance.py` is the literal check the CI job
(`.github/workflows/conformance.yml`) runs on every push, and the same
checks the harness performs before scoring your submission for real:

1. `submission.retrieve` exposes `build_index` and `retrieve` with the
   right signature (checked via `inspect.signature`, not just
   `hasattr`).
2. `build_index()` then `retrieve()` run on the toy corpus without
   raising.
3. Every `retrieve()` result is well-formed: a list of `(doc_id, score)`
   pairs, length `<= k`, sorted descending, all `doc_id`s valid, all
   `score`s numeric.
4. Nothing pathologically slow happens on a 20-document toy corpus
   (catches accidental quadratic blowups early, before they bite you on
   the real corpus).

None of this checks ranking *quality* — a submission that always returns
an empty list passes conformance trivially and scores 0 on every metric.
Conformance is a floor, not a target.

## Conformance freeze

48 hours before the assignment deadline, your repository's CI conformance
check must be green. Interface problems reported for the first time after
the freeze are graded as a submission error (assignment Section 9,
Grading Rubric — "Interface conformance"), not treated as a harness bug.
Run `scripts/smoke_test.sh` locally any time before then.
