"""Local-model backend tests with a scripted generator (no model weights needed)."""

from __future__ import annotations

import json

import pytest

from council_rag.answers import NOT_FOUND_MESSAGE, make_answerer
from council_rag.chunking import EXCERPT_MAX_CHARS, Chunk, chunk_minutes
from council_rag.local_llm import (
    SYSTEM_PROMPT,
    Generation,
    LocalAnswerer,
    build_messages,
    parse_reply,
)


class ScriptedGenerator:
    model_name = "scripted-test-model"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], max_tokens: int) -> Generation:
        self.calls.append(messages)
        return Generation(
            text=self.replies.pop(0), prompt_tokens=900, completion_tokens=40, seconds=0.1
        )


@pytest.fixture
def chunks(synthetic_minutes) -> list[Chunk]:
    by_ref = {c.item_ref: c for c in chunk_minutes(synthetic_minutes)}
    return [by_ref["2.1"], by_ref["3.1"], by_ref["4.1.1"]]


def reply(answer: str, ids: list, found: bool = True) -> str:
    return json.dumps({"answer": answer, "cited_passage_ids": ids, "found": found})


def test_prompt_numbers_passages_with_meeting_and_item(chunks) -> None:
    messages = build_messages("Was the bicycle shop approved?", chunks)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = messages[1]["content"]
    assert "[2] Regular Council Meeting, March 4, 2024. Item 3.1: Notices Published" in user
    assert chunks[1].text in user
    assert user.endswith("Question: Was the bicycle shop approved?")


def test_valid_reply_maps_citations(chunks) -> None:
    gen = ScriptedGenerator(reply("No. It was defeated 1 to 3 on March 4, 2024.", [2]))
    answer = LocalAnswerer(generator=gen).answer("Was the bicycle shop approved?", chunks)
    assert answer.status == "answered"
    (c,) = answer.citations
    assert (c.meeting_date, c.item_ref, c.chunk_id) == ("2024-03-04", "3.1", chunks[1].chunk_id)
    assert c.url and c.url.endswith("#AgendaItemAgendaItem8TitleHeader")
    assert len(c.cited_text) <= EXCERPT_MAX_CHARS and c.cited_text in chunks[1].text
    assert answer.usage and answer.usage.cost_usd == 0.0 and answer.usage.attempts == 1
    assert answer.model == "scripted-test-model"


def test_fenced_json_is_accepted() -> None:
    parsed, error = parse_reply(
        '```json\n{"answer": "Yes.", "cited_passage_ids": [1], "found": true}\n```'
    )
    assert error is None and parsed and parsed.cited_passage_ids == [1]


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("The motion carried.", "no JSON object"),
        ('{"answer": "Yes", "cited_passage_ids": [1], "found": true', "invalid JSON"),
        ('{"answer": "Yes", "cited_passage_ids": ["1"], "found": true}', "schema error"),
        ('{"answer": "Yes", "cited_passage_ids": [1], "found": "yes"}', "schema error"),
        ('{"answer": "Yes", "cited_passage_ids": [1], "found": true, "why": "x"}', "schema error"),
        ('{"answer": "", "cited_passage_ids": [1], "found": true}', "answer is empty"),
        (
            '{"answer": "Yes", "cited_passage_ids": [1], "found": true} Hope this helps!',
            "text after",
        ),
    ],
)
def test_parse_reply_rejects_invalid_output(text: str, fragment: str) -> None:
    parsed, error = parse_reply(text)
    assert parsed is None and error and fragment in error


def test_one_reask_after_invalid_reply(chunks) -> None:
    gen = ScriptedGenerator(
        "Sure! The application was refused.", reply("It was defeated 1 to 3.", [2])
    )
    answer = LocalAnswerer(generator=gen).answer("Was the bicycle shop approved?", chunks)
    assert answer.status == "answered" and answer.usage and answer.usage.attempts == 2
    retry = gen.calls[1]
    assert retry[-2] == {"role": "assistant", "content": "Sure! The application was refused."}
    assert "not valid (no JSON object found)" in retry[-1]["content"]
    assert answer.usage.input_tokens == 1800


def test_two_invalid_replies_give_invalid_output(chunks) -> None:
    gen = ScriptedGenerator("nope", "still not json")
    answer = LocalAnswerer(generator=gen).answer("q", chunks)
    assert answer.status == "invalid_output" and answer.text == NOT_FOUND_MESSAGE
    assert answer.citations == [] and len(gen.calls) == 2


def test_found_false_is_not_found(chunks) -> None:
    gen = ScriptedGenerator(reply("", [], found=False))
    answer = LocalAnswerer(generator=gen).answer("Did council adopt rent control?", chunks)
    assert answer.status == "not_found" and answer.text == NOT_FOUND_MESSAGE


def test_citations_outside_the_passages_are_dropped(chunks) -> None:
    gen = ScriptedGenerator(reply("Adopted 4 to 0.", [9, 1, 1, 0]))
    answer = LocalAnswerer(generator=gen).answer("Was the agenda adopted?", chunks)
    assert answer.status == "answered"
    assert [c.document_index for c in answer.citations] == [0]  # duplicates collapsed
    assert answer.dropped_citations == [9, 0]
    assert answer.detail and "[9, 0]" in answer.detail


def test_answer_with_only_invalid_citations_is_withheld(chunks) -> None:
    gen = ScriptedGenerator(reply("Council approved it.", [7]))
    answer = LocalAnswerer(generator=gen).answer("q", chunks)
    assert answer.status == "not_found" and "withheld" in (answer.detail or "")
    assert answer.dropped_citations == [7]


def test_no_passages_means_no_model_call() -> None:
    gen = ScriptedGenerator()
    answer = LocalAnswerer(generator=gen).answer("anything", [])
    assert answer.status == "not_found" and gen.calls == []


def test_backend_selection(monkeypatch) -> None:
    from council_rag.generation import ClaudeAnswerer
    from council_rag.local_llm import LocalAnswerer as Local

    monkeypatch.delenv("COUNCIL_RAG_LLM_BACKEND", raising=False)
    assert isinstance(make_answerer(), Local)  # default; the model loads lazily
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    assert isinstance(make_answerer("claude"), ClaudeAnswerer)  # constructed, never called
    with pytest.raises(ValueError):
        make_answerer("other")
