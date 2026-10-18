"""PA-09: logs hold no IP address, client identifier or question text (NFR-06)."""

from __future__ import annotations

import json
import re

from fastapi.testclient import TestClient

from council_rag import api, cli
from council_rag.answers import USAGE_LOG_FIELDS
from council_rag.index import IndexMeta
from council_rag.local_llm import LocalAnswerer
from council_rag.retrieval import HybridRetriever
from tests.test_local_llm import ScriptedGenerator

IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
IPV6 = re.compile(r"\b[0-9a-f]{1,4}(?::[0-9a-f]{0,4}){2,7}\b", re.IGNORECASE)


def test_usage_log_has_no_identifiers(two_meeting_chunks, fake_embedder, tmp_path) -> None:
    vectors = fake_embedder.embed_documents([c.index_text() for c in two_meeting_chunks])
    state = api.AppState()
    state.retriever = HybridRetriever.build(two_meeting_chunks, vectors, fake_embedder)
    # A complete in-memory index, so the API never falls back to the on-disk index
    # (which exists on a developer machine but not in CI).
    state.meta = IndexMeta(
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
    log_path = tmp_path / "usage.jsonl"
    reply = json.dumps(
        {"answer": "It was defeated 1 to 3.", "cited_passage_ids": [1], "found": True}
    )
    state.generator = LocalAnswerer(generator=ScriptedGenerator(reply), usage_log=log_path)
    original = api.state
    api.state = state
    try:
        question = "Was the pottery studio at 12 Example Street approved?"
        res = TestClient(api.app).post(
            "/ask",
            json={"question": question},
            headers={"User-Agent": "privacy-test-agent/1.0", "X-Forwarded-For": "203.0.113.7"},
        )
        assert res.status_code == 200
    finally:
        api.state = original
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert set(row) <= set(USAGE_LOG_FIELDS)
    raw = lines[0].lower()
    for forbidden in (
        "pottery",
        "example street",
        "privacy-test-agent",
        "203.0.113.7",
        "testclient",
    ):
        assert forbidden not in raw
    assert not IPV4.search(lines[0]) and not IPV6.search(lines[0].replace(row["ts"], ""))


def test_server_runs_without_access_log(monkeypatch) -> None:
    captured = {}
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: captured.update(k))
    cli.main(["serve", "--port", "8999"])
    assert captured["access_log"] is False
