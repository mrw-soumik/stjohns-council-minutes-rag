"""Shared fixtures. Nothing here touches the network or City documents."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pytest

from council_rag.chunking import Chunk, chunk_minutes
from council_rag.ingest.html_minutes import Minutes, parse_minutes_html
from council_rag.text import tokenize

FIXTURES = Path(__file__).parent / "fixtures"
MEETING_ID = "00000000-0000-4000-8000-000000000001"
SOURCE_URL = f"https://example.invalid/Meeting.aspx?Id={MEETING_ID}&Agenda=PostMinutes"


class FakeEmbedder:
    """Deterministic bag-of-words embedder (hashed tokens), L2-normalised."""

    dim = 384

    def __init__(self) -> None:
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        self.calls += 1
        return self._vec(text)


@pytest.fixture
def synthetic_html() -> str:
    return (FIXTURES / "minutes_synthetic.html").read_text(encoding="utf-8")


@pytest.fixture
def synthetic_minutes(synthetic_html: str) -> Minutes:
    return parse_minutes_html(
        synthetic_html,
        meeting_id=MEETING_ID,
        meeting_name="Regular Council Meeting",
        meeting_date="2024-03-04",
        source_url=SOURCE_URL,
        pdf_url="https://example.invalid/FileStream.ashx?DocumentId=1",
        pdf_document_id=1,
    )


def make_minutes(meeting_id: str, meeting_date: str, html: str) -> Minutes:
    return parse_minutes_html(
        html,
        meeting_id=meeting_id,
        meeting_name="Regular Council Meeting",
        meeting_date=meeting_date,
        source_url=f"https://example.invalid/Meeting.aspx?Id={meeting_id}",
    )


@pytest.fixture
def two_meeting_chunks(synthetic_html: str) -> list[Chunk]:
    """The synthetic meeting twice, a year apart, with one wording change."""
    a = make_minutes("aaaaaaaa-0000-4000-8000-000000000001", "2023-03-06", synthetic_html)
    b = make_minutes(
        "bbbbbbbb-0000-4000-8000-000000000002",
        "2024-03-04",
        synthetic_html.replace("bicycle repair shop", "pottery studio"),
    )
    return chunk_minutes(a) + chunk_minutes(b)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
