from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from council_rag import api
from council_rag.generation import AnswerGenerator, GenerationError
from council_rag.index import IndexMeta
from council_rag.retrieval import HybridRetriever
from tests.test_generation import FakeClient, cite, message


@pytest.fixture
def client(two_meeting_chunks, fake_embedder):
    vectors = fake_embedder.embed_documents([c.index_text() for c in two_meeting_chunks])
    retriever = HybridRetriever.build(two_meeting_chunks, vectors, fake_embedder)
    meta = IndexMeta(
        created_at="2026-01-01T00:00:00+00:00",
        embedding_repo="fake/embedder",
        embedding_revision="0123456789abcdef",
        embedding_file="none",
        max_words=250,
        index_with_context=True,
        n_meetings=2,
        n_chunks=len(two_meeting_chunks),
        date_min="2023-03-06",
        date_max="2024-03-04",
    )
    state = api.AppState()
    state.retriever, state.meta = retriever, meta
    original = api.state
    api.state = state
    try:
        yield TestClient(api.app), state, two_meeting_chunks
    finally:
        api.state = original


def test_health_reports_coverage_disclaimer_and_model(client, monkeypatch) -> None:
    http, _, _ = client
    monkeypatch.delenv("COUNCIL_RAG_LLM_BACKEND", raising=False)
    body = http.get("/health").json()
    assert body["status"] == "ok"
    assert (body["coverage_from"], body["coverage_to"]) == ("2023-03-06", "2024-03-04")
    assert body["answer_model"] == "Qwen3.5-9B (MLX 4-bit)"  # local backend by default
    assert "Not affiliated" in body["disclaimer"]
    monkeypatch.setenv("COUNCIL_RAG_LLM_BACKEND", "claude")
    assert http.get("/health").json()["answer_model"] == "claude-opus-5-5"


def test_search_returns_short_excerpts_and_links(client) -> None:
    http, _, _ = client
    res = http.post("/search", json={"query": "pottery studio", "k": 3})
    assert res.status_code == 200
    first = res.json()["results"][0]
    assert first["meeting_date"] == "2024-03-04" and first["item_ref"] == "3.1"
    assert len(first["excerpt"]) <= 300
    assert first["url"].endswith("#AgendaItemAgendaItem8TitleHeader")


def test_search_date_filter_and_validation(client) -> None:
    http, _, _ = client
    res = http.post("/search", json={"query": "agenda adopted", "date_to": "2023-12-31", "k": 10})
    assert {r["meeting_date"] for r in res.json()["results"]} == {"2023-03-06"}
    assert http.post("/search", json={"query": ""}).status_code == 422
    assert http.post("/search", json={"query": "x", "k": 99}).status_code == 422
    bad_range = {"query": "x", "date_from": "2024-02-01", "date_to": "2024-01-01"}
    assert http.post("/search", json=bad_range).status_code == 422
    assert http.post("/search", json={"query": "x", "mode": "fuzzy"}).status_code == 422


def test_ask_returns_answer_citations_and_passages(client) -> None:
    http, state, chunks = client
    target = next(c for c in chunks if c.item_ref == "3.1" and c.meeting_date == "2024-03-04")

    class Recorder(FakeClient):
        def __init__(self) -> None:
            super().__init__(None)
            self.beta.messages.create = self.create

        def create(self, **kwargs):
            docs = kwargs["messages"][0]["content"][:-1]
            idx = next(i for i, d in enumerate(docs) if d["source"]["data"] == target.text)
            return message(
                [
                    {
                        "type": "text",
                        "text": "It was defeated 1 to 3.",
                        "citations": [cite(target, idx, "MOTION LOST (1 to 3)")],
                    }
                ]
            )

    state.generator = AnswerGenerator(client=Recorder())
    res = http.post("/ask", json={"question": "Was the pottery studio approved?", "k": 4})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "answered"
    assert body["citations"][0]["meeting_date"] == "2024-03-04"
    assert body["citations"][0]["cited_text"] == "MOTION LOST (1 to 3)"
    assert 1 <= len(body["passages"]) <= 4
    assert body["usage"]["model"] == "claude-opus-5-5"


def test_ask_not_found_without_matches(client) -> None:
    http, state, _ = client
    state.generator = AnswerGenerator(client=FakeClient(RuntimeError("must not be called")))
    res = http.post("/ask", json={"question": "anything at all", "date_from": "2030-01-01"})
    assert res.json()["status"] == "not_found"


def test_ask_maps_missing_credentials_to_503(client) -> None:
    http, state, _ = client

    class Broken:
        def answer(self, *a, **k):
            raise GenerationError("Claude API credentials are missing or invalid", 401)

    state.generator = Broken()
    res = http.post("/ask", json={"question": "Was the agenda adopted?"})
    assert res.status_code == 503
