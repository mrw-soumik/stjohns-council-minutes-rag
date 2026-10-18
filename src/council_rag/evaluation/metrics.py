"""Retrieval metrics over ranked chunk ids and per-gold relevance sets."""

from __future__ import annotations

import math
from collections.abc import Sequence


def hit_at_k(ranked: Sequence[str], relevant: Sequence[set[str]], k: int) -> float:
    """1.0 if any relevant chunk (for any gold reference) is in the top ``k``."""
    pool = set().union(*relevant) if relevant else set()
    return 1.0 if any(cid in pool for cid in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Sequence[set[str]], k: int) -> float:
    """Share of gold references with at least one relevant chunk in the top ``k``."""
    if not relevant:
        return 0.0
    top = set(ranked[:k])
    return sum(1 for rel in relevant if rel & top) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Sequence[set[str]], k: int = 10) -> float:
    pool = set().union(*relevant) if relevant else set()
    for rank, cid in enumerate(ranked[:k], start=1):
        if cid in pool:
            return 1.0 / rank
    return 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]); deterministic and dependency-free."""
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (used to show eval noise honestly)."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
