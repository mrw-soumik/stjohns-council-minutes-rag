"""Claude API prices (USD per million tokens) and cost calculation (NFR-04).

Source: the Anthropic `claude-api` skill's model table (cached 2026-10-06). Cache
writes use the 5-minute rate (1.25x input); cache reads use the model's documented
read rate. Re-check https://platform.claude.com/docs/en/about-claude/pricing before
relying on these numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_write: float
    cache_read: float


PRICES_USD_PER_MTOK: dict[str, Price] = {
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_write=5.00, cache_read=0.20),
    "claude-opus-5": Price(input=5.00, output=25.00, cache_write=6.25, cache_read=0.50),
    "claude-opus-4-8": Price(input=5.00, output=25.00, cache_write=6.25, cache_read=0.50),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_write=2.50, cache_read=0.20),
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_write=0.125, cache_read=0.01),
}

# NFR-04 is stated in Canadian dollars. The exchange rate is an assumption, not a
# measurement: set COUNCIL_RAG_USD_TO_CAD to the current Bank of Canada rate.
USD_TO_CAD = float(os.environ.get("COUNCIL_RAG_USD_TO_CAD", "1.40"))


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float:
    """Cost of one model call. Unknown models raise, so a silent $0 can never appear."""
    price = PRICES_USD_PER_MTOK.get(model)
    if price is None:
        raise KeyError(f"No price configured for model {model!r}")
    return (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_creation_input_tokens * price.cache_write
        + cache_read_input_tokens * price.cache_read
    ) / 1_000_000
