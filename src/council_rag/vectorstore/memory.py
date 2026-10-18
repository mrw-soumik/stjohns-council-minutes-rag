"""Exact cosine search over an in-memory NumPy matrix (the default store)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from council_rag.vectorstore.base import VectorStore


class InMemoryVectorStore(VectorStore):
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._dates = np.array([], dtype="datetime64[D]")
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    def upsert(self, ids: Sequence[str], dates: Sequence[str], vectors: np.ndarray) -> None:
        self._check(ids, dates, vectors)
        vectors = np.asarray(vectors, dtype=np.float32)
        new_ids, new_dates, new_rows = [], [], []
        for chunk_id, day, vec in zip(ids, dates, vectors, strict=True):
            if chunk_id in self._pos:  # update in place
                i = self._pos[chunk_id]
                self._matrix[i] = vec
                self._dates[i] = np.datetime64(day, "D")
            else:
                new_ids.append(chunk_id)
                new_dates.append(day)
                new_rows.append(vec)
        if new_ids:
            start = len(self._ids)
            self._ids.extend(new_ids)
            self._pos.update({cid: start + j for j, cid in enumerate(new_ids)})
            self._dates = np.concatenate([self._dates, np.array(new_dates, dtype="datetime64[D]")])
            self._matrix = np.vstack([self._matrix, np.stack(new_rows)])

    def date_mask(self, date_from: str | None, date_to: str | None) -> np.ndarray:
        mask = np.ones(len(self._ids), dtype=bool)
        if date_from:
            mask &= self._dates >= np.datetime64(date_from, "D")
        if date_to:
            mask &= self._dates <= np.datetime64(date_to, "D")
        return mask

    def search(
        self,
        query: np.ndarray,
        k: int,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[tuple[str, float]]:
        if not self._ids or k <= 0:
            return []
        scores = self._matrix @ np.asarray(query, dtype=np.float32)
        candidates = np.flatnonzero(self.date_mask(date_from, date_to))
        if candidates.size == 0:
            return []
        order = candidates[np.argsort(-scores[candidates], kind="stable")][:k]
        return [(self._ids[i], float(scores[i])) for i in order]

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self.__init__(self.dim)
