"""
submission/indexer.py — build your inverted index here.

This is one of the required components (assignment Section 4.1): you must
build the inverted index yourself, without an existing search/indexing
library (Lucene, Elasticsearch, Pyserini, Whoosh, etc.).

A `tokenize()` helper is provided below purely so that tokenization is
consistent across your Boolean/VSM, BM25, and language-model scorers —
feel free to replace it (e.g. add stemming or stopword removal), just make
sure every scorer that reads this index was built with the same tokenizer.

Everything else — the postings representation, what per-document and
collection statistics you track, whether you add positions for
proximity/phrase features — is your design decision. `InvertedIndex`
below sketches a minimal, obviously-sufficient shape; you do not have to
use it, but if you do, filling in `build()` and `document_frequency()` is
enough to support Boolean/VSM, BM25, and a unigram language model.
"""
import re
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric-only tokenization."""
    return _TOKEN_RE.findall(text.lower())


class InvertedIndex:
    """A minimal inverted index skeleton. Extend the data structures here
    however your design needs (e.g. term positions for phrase/proximity
    scoring, a more compact postings representation for the efficiency
    bonus) — this is a starting point, not a fixed schema.
    """

    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: term_freq}
        self.doc_len: Dict[str, int] = {}  # doc_id -> number of tokens
        self.doc_text: Dict[str, str] = {}  # doc_id -> raw text (handy for VSM/debugging)
        self.N: int = 0  # number of documents
        self.avg_doc_len: float = 0.0

    def build(self, corpus: List[Tuple[str, str]]) -> None:
        """corpus: list of (doc_id, text) pairs, e.g. from
        submission.corpus_utils.load_corpus().

        TODO(you): tokenize each document, populate self.postings,
        self.doc_len, self.doc_text, self.N, and self.avg_doc_len.
        """
        raise NotImplementedError("Implement InvertedIndex.build() — see assignment Section 4.1.")

    def document_frequency(self, term: str) -> int:
        """Number of documents containing `term` at least once.

        TODO(you): implement using self.postings.
        """
        raise NotImplementedError("Implement InvertedIndex.document_frequency().")
