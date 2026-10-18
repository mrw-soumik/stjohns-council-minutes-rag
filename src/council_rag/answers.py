"""Answer and citation types shared by every answer backend, plus backend selection.

Backends (``COUNCIL_RAG_LLM_BACKEND``):

* ``local`` (default): an open model run on this machine with MLX
  (:mod:`council_rag.local_llm`). Free to run; grounding is enforced by the app.
* ``claude``: Claude through the official ``anthropic`` SDK with the Citations
  feature (:mod:`council_rag.generation`). Needs an API key and incurs API charges.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from council_rag.chunking import Chunk

NOT_FOUND_MESSAGE = (
    "I could not find this in the Regular Council Meeting minutes that were searched."
)


@dataclass
class Citation:
    document_index: int  # position of the passage in the list sent to the model
    chunk_id: str
    meeting_name: str
    meeting_date: str
    item_ref: str
    item_title: str
    url: str | None
    pdf_url: str | None
    cited_text: str  # exact quoted span (Claude) or a short excerpt of the passage (local)
    start_char: int | None = None  # span inside the passage text, when the backend gives one
    end_char: int | None = None
    answer_start: int = 0  # span of the answer text this citation supports
    answer_end: int = 0


@dataclass
class UsageRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    attempts: int = 1  # model calls for this answer (fallbacks or re-asks)


@dataclass
class Answer:
    # answered | not_found | refused | truncated | invalid_output
    status: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    model: str | None = None
    stop_reason: str | None = None
    usage: UsageRecord | None = None
    latency_s: float | None = None
    detail: str | None = None
    dropped_citations: list[int] = field(default_factory=list)  # ids outside the passages
    raw_replies: list[str] = field(default_factory=list)  # local backend only, for traces

    @property
    def found(self) -> bool:
        return self.status == "answered"


class Answerer(Protocol):
    """Anything that turns a question plus retrieved chunks into an :class:`Answer`."""

    model_name: str

    def answer(self, question: str, chunks: list[Chunk]) -> Answer: ...


def citation_for(chunk: Chunk, index: int, **extra: Any) -> Citation:
    return Citation(
        document_index=index,
        chunk_id=chunk.chunk_id,
        meeting_name=chunk.meeting_name,
        meeting_date=chunk.meeting_date,
        item_ref=chunk.item_ref,
        item_title=chunk.item_title,
        url=chunk.url,
        pdf_url=chunk.pdf_url,
        **{"cited_text": "", **extra},
    )


USAGE_LOG_FIELDS = (
    "ts",
    "status",
    "stop_reason",
    "latency_s",
    "n_documents",
    "n_citations",
    "model",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cost_usd",
    "attempts",
)


class UsageLogger:
    """Appends one JSON line per answer with usage and cost only (NFR-04, NFR-06).

    No question text, IP address, user agent, session or request identifier is written;
    ``tests/test_privacy.py`` (PA-09) enforces the field list.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def write(self, answer: Answer, n_documents: int) -> None:
        if self.path is None or answer.usage is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": answer.status,
            "stop_reason": answer.stop_reason,
            "latency_s": answer.latency_s,
            "n_documents": n_documents,
            "n_citations": len(answer.citations),
            **asdict(answer.usage),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


def configured_backend() -> str:
    return os.environ.get("COUNCIL_RAG_LLM_BACKEND", "local").lower()


def configured_model_label() -> str:
    """Name of the answer model shown in the UI and /health, without loading it (NFR-07)."""
    if configured_backend() == "claude":
        from council_rag.config import CLAUDE_MODEL

        return CLAUDE_MODEL
    from council_rag.local_llm import LOCAL_MODEL_LABEL

    return LOCAL_MODEL_LABEL


def make_answerer(backend: str | None = None, usage_log: Path | None = None) -> Answerer:
    """Build the configured backend (``local`` unless told otherwise)."""
    backend = (backend or configured_backend()).lower()
    if backend == "local":
        from council_rag.local_llm import LocalAnswerer

        return LocalAnswerer(usage_log=usage_log)
    if backend == "claude":
        from council_rag.generation import ClaudeAnswerer

        return ClaudeAnswerer(usage_log=usage_log)
    raise ValueError(f"Unknown answer backend {backend!r}; use 'local' or 'claude'")
