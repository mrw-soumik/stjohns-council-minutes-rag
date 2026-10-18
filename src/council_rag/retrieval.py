"""Hybrid retrieval: BM25 + dense embeddings fused with reciprocal rank fusion (FR-06).

The meeting-date filter (FR-04) is applied before ranking in both retrievers, so a
filtered query ranks only chunks inside the date range.

Reciprocal rank fusion (Cormack, Clarke and Buettcher, 2009):
``score(d) = sum_i w_i / (k + rank_i(d))`` over the ranked lists that contain ``d``,
with ranks starting at 1 and ``k = 60`` by default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from council_rag.bm25 import BM25Index
from council_rag.chunking import Chunk
from council_rag.embeddings import Embedder
from council_rag.vectorstore import InMemoryVectorStore, VectorStore

Mode = Literal["hybrid", "bm25", "dense"]


@dataclass(frozen=True)
class RetrievalConfig:
    rrf_k: int = 60
    candidates: int = 50  # depth of each list before fusion
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    index_with_context: bool = True  # prepend meeting/item path to indexed text


@dataclass
class SearchResult:
    chunk: Chunk
    rank: int
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    dense_score: float | None = None


def rrf_fuse(
    ranked_lists: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists; ties are broken by first appearance (deterministic)."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranked, weight in zip(ranked_lists, weights, strict=True):
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


class HybridRetriever:
    """Searches a fixed set of chunks with BM25, dense vectors, or both."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        store: VectorStore,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        self.store = store
        self.config = config or RetrievalConfig()
        self._by_id = {c.chunk_id: c for c in self.chunks}
        self._pos = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self._dates = np.array([c.meeting_date for c in self.chunks], dtype="datetime64[D]")
        self.bm25 = BM25Index().fit(
            [c.index_text(self.config.index_with_context) for c in self.chunks]
        )

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
        embedder: Embedder,
        config: RetrievalConfig | None = None,
        store: VectorStore | None = None,
    ) -> HybridRetriever:
        """Create a retriever and load precomputed chunk embeddings into ``store``."""
        store = store or InMemoryVectorStore(dim=embeddings.shape[1])
        store.upsert([c.chunk_id for c in chunks], [c.meeting_date for c in chunks], embeddings)
        return cls(chunks, embedder, store, config)

    def _mask(self, date_from: str | None, date_to: str | None) -> np.ndarray | None:
        if not date_from and not date_to:
            return None
        mask = np.ones(len(self.chunks), dtype=bool)
        if date_from:
            mask &= self._dates >= np.datetime64(date_from, "D")
        if date_to:
            mask &= self._dates <= np.datetime64(date_to, "D")
        return mask

    def search(
        self,
        query: str,
        k: int = 5,
        mode: Mode = "hybrid",
        date_from: str | None = None,
        date_to: str | None = None,
        query_vector: np.ndarray | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            return []
        depth = max(k, self.config.candidates)
        bm25_hits: list[tuple[str, float]] = []
        dense_hits: list[tuple[str, float]] = []
        if mode in ("hybrid", "bm25"):
            mask = self._mask(date_from, date_to)
            bm25_hits = [
                (self.chunks[i].chunk_id, s) for i, s in self.bm25.top_k(query, depth, mask)
            ]
        if mode in ("hybrid", "dense"):
            vector = query_vector if query_vector is not None else self.embedder.embed_query(query)
            dense_hits = self.store.search(vector, depth, date_from, date_to)

        bm25_rank = {cid: r for r, (cid, _) in enumerate(bm25_hits, start=1)}
        dense_rank = {cid: r for r, (cid, _) in enumerate(dense_hits, start=1)}
        bm25_score = dict(bm25_hits)
        dense_score = dict(dense_hits)

        if mode == "bm25":
            fused = bm25_hits
        elif mode == "dense":
            fused = dense_hits
        else:
            fused = rrf_fuse(
                [[c for c, _ in bm25_hits], [c for c, _ in dense_hits]],
                [self.config.bm25_weight, self.config.dense_weight],
                self.config.rrf_k,
            )
        results = []
        for rank, (cid, score) in enumerate(fused[:k], start=1):
            results.append(
                SearchResult(
                    chunk=self._by_id[cid],
                    rank=rank,
                    score=float(score),
                    bm25_rank=bm25_rank.get(cid),
                    dense_rank=dense_rank.get(cid),
                    bm25_score=bm25_score.get(cid),
                    dense_score=dense_score.get(cid),
                )
            )
        return results
