"""Okapi BM25 over chunk texts, with an optional document mask for date filtering."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from council_rag.text import tokenize


class BM25Index:
    """Classic Okapi BM25 (Robertson et al.) with ``idf = ln(1 + (N - df + 0.5) / (df + 0.5))``."""

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.avgdl = 0.0
        self.n_docs = 0

    def fit(self, texts: Sequence[str]) -> BM25Index:
        doc_terms = [Counter(self.tokenizer(t)) for t in texts]
        self.n_docs = len(doc_terms)
        self.doc_len = np.array([sum(c.values()) for c in doc_terms], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        build: dict[str, tuple[list[int], list[int]]] = {}
        for doc_id, counts in enumerate(doc_terms):
            for term, tf in counts.items():
                ids, tfs = build.setdefault(term, ([], []))
                ids.append(doc_id)
                tfs.append(tf)
        self.postings = {
            term: (np.array(ids, dtype=np.int32), np.array(tfs, dtype=np.float32))
            for term, (ids, tfs) in build.items()
        }
        self.idf = {
            term: math.log(1.0 + (self.n_docs - len(ids) + 0.5) / (len(ids) + 0.5))
            for term, (ids, _) in self.postings.items()
        }
        return self

    def scores(self, query: str, mask: np.ndarray | None = None) -> np.ndarray:
        """BM25 score of every document; masked-out documents get ``-inf``."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        if self.n_docs == 0:
            return out
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in self.tokenizer(query):
            posting = self.postings.get(term)
            if posting is None:
                continue
            ids, tfs = posting
            out[ids] += self.idf[term] * tfs * (self.k1 + 1.0) / (tfs + norm[ids])
        if mask is not None:
            out = np.where(mask, out, -np.inf)
        return out

    def top_k(self, query: str, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Indices and scores of the ``k`` best documents with a positive score."""
        scores = self.scores(query, mask)
        candidates = np.flatnonzero(scores > 0)
        if candidates.size == 0:
            return []
        order = candidates[np.argsort(-scores[candidates], kind="stable")][:k]
        return [(int(i), float(scores[i])) for i in order]
