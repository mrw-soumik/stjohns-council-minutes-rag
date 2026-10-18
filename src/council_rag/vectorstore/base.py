"""Vector store interface shared by the in-memory and pgvector back ends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class VectorStore(ABC):
    """Stores L2-normalised embeddings keyed by chunk id, with the meeting date for filtering.

    ``search`` returns ``(chunk_id, cosine_similarity)`` pairs, best first. Date bounds
    are inclusive ISO dates (``YYYY-MM-DD``); ``None`` means unbounded (FR-04).
    """

    dim: int

    @abstractmethod
    def upsert(self, ids: Sequence[str], dates: Sequence[str], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        k: int,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[tuple[str, float]]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...

    def _check(self, ids: Sequence[str], dates: Sequence[str], vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        if not (len(ids) == len(dates) == vectors.shape[0]):
            raise ValueError("ids, dates and vectors must have the same length")
