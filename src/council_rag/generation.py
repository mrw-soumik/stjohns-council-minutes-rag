"""Grounded answers with Claude and the Citations feature (FR-01, FR-02, FR-03, NFR-04).

Each retrieved chunk is sent as its own plain-text ``document`` block with citations
enabled, so every citation Claude returns carries ``document_index`` (which chunk)
and ``start_char_index`` / ``end_char_index`` (where in that chunk's text). Those map
back to the meeting, date, agenda item and official URL without any string matching.

Design choices:

* Model ``claude-opus-5-5`` (the skill's default), adaptive thinking (always on for
  this model) with an explicit ``effort``; ``max_tokens`` leaves room for thinking.
* Server-side refusal fallbacks are opted in (``fallbacks="default"`` with the
  ``server-side-fallback-2026-07-01`` beta), and ``stop_reason`` is checked before
  content is read.
* "Not found" (FR-03) has three triggers: retrieval returned nothing (no API call is
  made), Claude replied with the ``NOT_FOUND`` sentinel, or Claude produced text with
  no citations (an uncited answer is withheld rather than shown).
* Usage and cost are logged per request without the question text (NFR-06).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from council_rag.answers import (
    NOT_FOUND_MESSAGE,
    Answer,
    Citation,
    UsageLogger,
    UsageRecord,
    citation_for,
)
from council_rag.chunking import Chunk, long_date
from council_rag.config import CLAUDE_EFFORT, CLAUDE_MODEL
from council_rag.pricing import cost_usd

log = logging.getLogger(__name__)

NOT_FOUND_SENTINEL = "NOT_FOUND"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """\
You answer questions about decisions recorded in the minutes of City of St. John's \
Regular Council Meetings. The user message contains excerpts of the official minutes as \
documents. Each document is one agenda item (or part of one) from one meeting; its title \
gives the meeting date and the item number.

Rules:
- Use only the documents. Do not add facts from general knowledge, even if you believe \
them to be true.
- Support every factual statement with a citation to the documents.
- If the documents do not answer the question, reply with exactly NOT_FOUND and nothing \
else.
- Answer in one to four short sentences of plain Canadian English. Name the meeting date \
for each decision you report.
- Report motion outcomes, vote counts, amounts and by-law or file numbers exactly as \
recorded.
- If the documents show a matter at more than one meeting, give each relevant date \
rather than merging them.
"""


class GenerationError(RuntimeError):
    """The model call failed (network, rate limit, server or configuration error)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def document_title(chunk: Chunk) -> str:
    where = "attendance" if chunk.kind == "attendance" else f"item {chunk.item_ref}"
    return f"{chunk.meeting_name}, {long_date(chunk.meeting_date)}, {where}"


def document_block(chunk: Chunk) -> dict[str, Any]:
    """One retrieved chunk as a citable plain-text document."""
    context = {
        "meeting": chunk.meeting_name,
        "meeting_date": chunk.meeting_date,
        "agenda_item": f"{chunk.item_ref} {chunk.item_title}".strip(),
        "item_path": chunk.item_path,
    }
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": chunk.text},
        "title": document_title(chunk),
        "context": json.dumps(context, ensure_ascii=False),
        "citations": {"enabled": True},
    }


def build_request(
    question: str,
    chunks: Sequence[Chunk],
    model: str = CLAUDE_MODEL,
    effort: str = CLAUDE_EFFORT,
    max_tokens: int = 16000,
) -> dict[str, Any]:
    """Keyword arguments for ``client.beta.messages.create`` (kept pure for testing)."""
    content: list[dict[str, Any]] = [document_block(c) for c in chunks]
    content.append({"type": "text", "text": f"Question: {question.strip()}"})
    return {
        "model": model,
        "max_tokens": max_tokens,
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
        "output_config": {"effort": effort},
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content}],
    }


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def usage_record(message: Any) -> UsageRecord:
    """Tokens and cost of a response, summed over fallback attempts when present."""
    usage = _get(message, "usage")
    served_model = str(_get(message, "model"))
    iterations = _get(usage, "iterations") or []
    rows = iterations if iterations else [usage]
    totals = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    cost = 0.0
    for row in rows:
        model = str(_get(row, "model") or served_model)
        it = int(_get(row, "input_tokens") or 0)
        ot = int(_get(row, "output_tokens") or 0)
        cw = int(_get(row, "cache_creation_input_tokens") or 0)
        cr = int(_get(row, "cache_read_input_tokens") or 0)
        totals["input"] += it
        totals["output"] += ot
        totals["cache_write"] += cw
        totals["cache_read"] += cr
        cost += cost_usd(model, it, ot, cw, cr)
    return UsageRecord(
        model=served_model,
        input_tokens=totals["input"],
        output_tokens=totals["output"],
        cache_creation_input_tokens=totals["cache_write"],
        cache_read_input_tokens=totals["cache_read"],
        cost_usd=round(cost, 6),
        attempts=max(1, len(iterations)),
    )


def parse_message(message: Any, chunks: Sequence[Chunk]) -> Answer:
    """Turn a Messages API response into an :class:`Answer` with mapped citations."""
    stop_reason = _get(message, "stop_reason")
    model = _get(message, "model")
    if stop_reason == "refusal":
        return Answer(
            status="refused",
            text="",
            model=model,
            stop_reason=stop_reason,
            detail="The model declined this request.",
        )
    text_parts: list[str] = []
    citations: list[Citation] = []
    cursor = 0
    for block in _get(message, "content") or []:
        if _get(block, "type") != "text":
            continue  # thinking, fallback markers and other block types are not shown
        piece = str(_get(block, "text") or "")
        start, end = cursor, cursor + len(piece)
        for raw in _get(block, "citations") or []:
            if _get(raw, "type") != "char_location":
                continue
            idx = int(_get(raw, "document_index"))
            if not 0 <= idx < len(chunks):
                log.warning("Citation points at unknown document %s", idx)
                continue
            chunk = chunks[idx]
            citations.append(
                citation_for(
                    chunk,
                    idx,
                    cited_text=str(_get(raw, "cited_text") or ""),
                    start_char=int(_get(raw, "start_char_index")),
                    end_char=int(_get(raw, "end_char_index")),
                    answer_start=start,
                    answer_end=end,
                )
            )
        text_parts.append(piece)
        cursor = end
    raw_text = "".join(text_parts)
    text = raw_text.strip()
    lead = len(raw_text) - len(raw_text.lstrip())
    for c in citations:  # keep answer spans aligned with the stripped text
        c.answer_start = min(max(c.answer_start - lead, 0), len(text))
        c.answer_end = min(max(c.answer_end - lead, 0), len(text))

    if stop_reason == "max_tokens":
        return Answer(
            status="truncated",
            text=text,
            citations=citations,
            model=model,
            stop_reason=stop_reason,
            detail="The response hit max_tokens and is incomplete.",
        )
    if not text or text.strip().strip(".").upper() == NOT_FOUND_SENTINEL:
        return Answer(
            status="not_found", text=NOT_FOUND_MESSAGE, model=model, stop_reason=stop_reason
        )
    if not citations:
        return Answer(
            status="not_found",
            text=NOT_FOUND_MESSAGE,
            model=model,
            stop_reason=stop_reason,
            detail="The model's answer had no citations, so it was withheld.",
        )
    return Answer(
        status="answered", text=text, citations=citations, model=model, stop_reason=stop_reason
    )


class ClaudeAnswerer:
    """Calls Claude through the official ``anthropic`` SDK (the ``claude`` backend)."""

    def __init__(
        self,
        client: Any | None = None,
        model: str = CLAUDE_MODEL,
        effort: str = CLAUDE_EFFORT,
        max_tokens: int = 16000,
        usage_log: Path | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=timeout_s, max_retries=2)
        self.client = client
        self.model = model
        self.model_name = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.logger = UsageLogger(usage_log)

    def answer(self, question: str, chunks: Sequence[Chunk]) -> Answer:
        if not chunks:
            return Answer(
                status="not_found",
                text=NOT_FOUND_MESSAGE,
                detail="No passages matched the question and filters; the model was not called.",
            )
        import anthropic

        request = build_request(question, chunks, self.model, self.effort, self.max_tokens)
        started = time.perf_counter()
        try:
            message = self.client.beta.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise GenerationError("Claude API credentials are missing or invalid", 401) from exc
        except anthropic.PermissionDeniedError as exc:
            raise GenerationError("Claude API key lacks permission for this request", 403) from exc
        except anthropic.NotFoundError as exc:
            raise GenerationError(f"Unknown model or endpoint: {self.model}", 404) from exc
        except anthropic.RateLimitError as exc:
            raise GenerationError("Claude API rate limit reached; try again shortly", 429) from exc
        except anthropic.BadRequestError as exc:
            raise GenerationError(f"Claude API rejected the request: {exc.message}", 400) from exc
        except anthropic.APIStatusError as exc:
            raise GenerationError(f"Claude API error ({exc.status_code})", exc.status_code) from exc
        except anthropic.APIConnectionError as exc:
            raise GenerationError("Could not reach the Claude API") from exc
        latency = time.perf_counter() - started

        answer = parse_message(message, chunks)
        answer.latency_s = round(latency, 3)
        answer.usage = usage_record(message)
        self.logger.write(answer, len(chunks))
        return answer


# Backwards-compatible name used by earlier code and tests.
AnswerGenerator = ClaudeAnswerer
