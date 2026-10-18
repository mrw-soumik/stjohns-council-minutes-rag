"""Generation tests with mocked Messages API responses (no network, no API spend)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types.beta import BetaMessage

from council_rag.chunking import Chunk
from council_rag.generation import (
    FALLBACK_BETA,
    NOT_FOUND_MESSAGE,
    SYSTEM_PROMPT,
    AnswerGenerator,
    GenerationError,
    build_request,
    parse_message,
    usage_record,
)
from council_rag.pricing import cost_usd


class FakeMessages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeClient:
    def __init__(self, response: Any) -> None:
        self.beta = SimpleNamespace(messages=FakeMessages(response))


def message(
    content: list[dict],
    stop_reason: str = "end_turn",
    usage: dict | None = None,
    model: str = "claude-opus-5-5",
) -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage
            or {
                "input_tokens": 4000,
                "output_tokens": 300,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
    )


def cite(chunk: Chunk, idx: int, phrase: str) -> dict:
    start = chunk.text.index(phrase)
    return {
        "type": "char_location",
        "cited_text": phrase,
        "document_index": idx,
        "document_title": "t",
        "start_char_index": start,
        "end_char_index": start + len(phrase),
        "file_id": None,
    }


@pytest.fixture
def chunks(synthetic_minutes) -> list[Chunk]:
    from council_rag.chunking import chunk_minutes

    by_ref = {c.item_ref: c for c in chunk_minutes(synthetic_minutes)}
    return [by_ref["2.1"], by_ref["3.1"], by_ref["4.1.1"]]


def test_request_uses_citable_documents_and_skill_defaults(chunks) -> None:
    req = build_request("Was the home occupation at 12 Example Street approved?", chunks)
    assert req["model"] == "claude-opus-5-5"
    assert req["betas"] == [FALLBACK_BETA] and req["fallbacks"] == "default"
    assert req["output_config"] == {"effort": "medium"}  # explicit, never "format"
    assert "thinking" not in req  # adaptive thinking is always on for this model
    assert req["system"] == SYSTEM_PROMPT
    content = req["messages"][0]["content"]
    docs, question = content[:-1], content[-1]
    assert len(docs) == 3
    for doc, chunk in zip(docs, chunks, strict=True):
        assert doc["type"] == "document"
        assert doc["citations"] == {"enabled": True}  # all documents, as the API requires
        assert doc["source"] == {"type": "text", "media_type": "text/plain", "data": chunk.text}
        assert json.loads(doc["context"])["meeting_date"] == "2024-03-04"
    assert docs[1]["title"] == "Regular Council Meeting, March 4, 2024, item 3.1"
    assert question == {
        "type": "text",
        "text": "Question: Was the home occupation at 12 Example Street approved?",
    }


def test_citations_map_back_to_meeting_item_and_url(chunks) -> None:
    msg = message(
        [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "No. At the March 4, 2024 meeting, ", "citations": None},
            {
                "type": "text",
                "text": "the application was defeated 1 to 3",
                "citations": [cite(chunks[1], 1, "MOTION LOST (1 to 3)")],
            },
            {"type": "text", "text": ".", "citations": None},
        ]
    )
    answer = parse_message(msg, chunks)
    assert answer.status == "answered"
    assert answer.text == "No. At the March 4, 2024 meeting, the application was defeated 1 to 3."
    (c,) = answer.citations
    assert (c.meeting_date, c.item_ref, c.chunk_id) == ("2024-03-04", "3.1", chunks[1].chunk_id)
    assert c.url and c.url.endswith("#AgendaItemAgendaItem8TitleHeader")
    assert chunks[1].text[c.start_char : c.end_char] == c.cited_text
    assert answer.text[c.answer_start : c.answer_end] == "the application was defeated 1 to 3"


def test_not_found_sentinel(chunks) -> None:
    answer = parse_message(message([{"type": "text", "text": "NOT_FOUND"}]), chunks)
    assert answer.status == "not_found" and answer.text == NOT_FOUND_MESSAGE


def test_uncited_answer_is_withheld(chunks) -> None:
    answer = parse_message(message([{"type": "text", "text": "Council approved it."}]), chunks)
    assert answer.status == "not_found"
    assert answer.detail and "no citations" in answer.detail


def test_refusal_is_checked_before_content(chunks) -> None:
    msg = message([{"type": "text", "text": "partial"}], stop_reason="refusal")
    answer = parse_message(msg, chunks)
    assert answer.status == "refused" and answer.text == ""


def test_truncated_response_is_flagged(chunks) -> None:
    msg = message(
        [{"type": "text", "text": "Council", "citations": None}], stop_reason="max_tokens"
    )
    assert parse_message(msg, chunks).status == "truncated"


def test_out_of_range_citation_is_ignored(chunks) -> None:
    bad = cite(chunks[0], 0, "That the Agenda be adopted as presented.") | {"document_index": 9}
    msg = message([{"type": "text", "text": "Adopted.", "citations": [bad]}])
    assert parse_message(msg, chunks).status == "not_found"


def test_generator_skips_api_when_nothing_retrieved() -> None:
    client = FakeClient(RuntimeError("must not be called"))
    answer = AnswerGenerator(client=client).answer("anything", [])
    assert answer.status == "not_found"
    assert client.beta.messages.calls == []


def test_generator_logs_usage_without_question(chunks, tmp_path) -> None:
    msg = message(
        [
            {
                "type": "text",
                "text": "Adopted.",
                "citations": [cite(chunks[0], 0, "MOTION CARRIED (4 to 0)")],
            }
        ]
    )
    log_path = tmp_path / "usage.jsonl"
    generator = AnswerGenerator(client=FakeClient(msg), usage_log=log_path)
    answer = generator.answer("Was the agenda adopted on March 4?", chunks)
    assert answer.status == "answered"
    assert answer.usage is not None
    assert answer.usage.cost_usd == pytest.approx(4000 * 4 / 1e6 + 300 * 20 / 1e6)
    row = json.loads(log_path.read_text().strip())
    assert row["input_tokens"] == 4000 and row["n_documents"] == 3 and row["n_citations"] == 1
    assert "agenda" not in json.dumps(row).lower()  # NFR-06: question text is not stored


def test_usage_sums_fallback_iterations_at_each_models_price() -> None:
    usage = {
        "input_tokens": 4100,
        "output_tokens": 250,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "iterations": [
            {
                "type": "message",
                "model": "claude-opus-5-5",
                "input_tokens": 4000,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            {
                "type": "fallback_message",
                "model": "claude-opus-4-8",
                "input_tokens": 4100,
                "output_tokens": 250,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        ],
    }
    rec = usage_record(
        message([{"type": "text", "text": "x"}], usage=usage, model="claude-opus-4-8")
    )
    assert rec.attempts == 2 and rec.input_tokens == 8100
    assert rec.cost_usd == pytest.approx(4000 * 4e-6 + 4100 * 5e-6 + 250 * 25e-6)


def test_unknown_model_price_fails_loudly() -> None:
    with pytest.raises(KeyError):
        cost_usd("claude-unknown", 1, 1)


def test_api_errors_become_generation_errors(chunks) -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    rate_limited = anthropic.RateLimitError(
        "slow down", response=httpx2.Response(429, request=request), body=None
    )
    with pytest.raises(GenerationError) as info:
        AnswerGenerator(client=FakeClient(rate_limited)).answer("q", chunks)
    assert info.value.status_code == 429
    offline = anthropic.APIConnectionError(request=request)
    with pytest.raises(GenerationError):
        AnswerGenerator(client=FakeClient(offline)).answer("q", chunks)
