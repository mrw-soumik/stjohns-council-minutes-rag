from __future__ import annotations

import numpy as np
import pytest

from council_rag.demo_export import DemoIndex, export_demo, fnv1a32, quantize_int8
from council_rag.index import IndexMeta, save_index
from council_rag.retrieval import HybridRetriever


@pytest.fixture
def exported(tmp_path, two_meeting_chunks, fake_embedder):
    chunks = two_meeting_chunks
    vectors = fake_embedder.embed_documents([c.index_text() for c in chunks])
    meta = IndexMeta(
        created_at="2026-01-01T00:00:00+00:00",
        embedding_repo="fake/embedder",
        embedding_revision="0" * 40,
        embedding_file="none",
        max_words=250,
        index_with_context=True,
        n_meetings=2,
        n_chunks=len(chunks),
        date_min="2023-03-06",
        date_max="2024-03-04",
    )
    save_index(tmp_path / "index", chunks, vectors, meta)
    summary = export_demo(tmp_path / "index", tmp_path / "demo")
    retriever = HybridRetriever.build(chunks, vectors, fake_embedder)
    return DemoIndex(tmp_path / "demo"), retriever, summary, tmp_path / "demo"


def test_fnv1a32_reference_values() -> None:
    assert fnv1a32("") == 0x811C9DC5
    assert fnv1a32("a") == 0xE40C292C
    assert fnv1a32("foobar") == 0xBF9CF968


def test_int8_quantisation_preserves_direction() -> None:
    rng = np.random.default_rng(0)
    v = rng.normal(size=(20, 384)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q, scales = quantize_int8(v)
    back = q.astype(np.float32) * scales[:, None]
    cos = (back * v).sum(1) / np.linalg.norm(back, axis=1)
    assert cos.min() > 0.999


def test_summary_and_coverage(exported) -> None:
    index, _, summary, _ = exported
    assert summary["meetings"] == 2 and summary["chunks"] == len(index.chunk_ids)
    assert index.meta["coverage"] == {"from": "2023-03-06", "to": "2024-03-04", "meetings": 2}


@pytest.mark.parametrize("query", ["pottery studio", "snow clearing by-law 9999", "garden water"])
def test_demo_bm25_matches_python_retriever(exported, query) -> None:
    index, retriever, _, _ = exported
    want = [(r.chunk.chunk_id, r.score) for r in retriever.search(query, k=8, mode="bm25")]
    got = index.search(query, None, k=8, mode="bm25")
    assert [g[0] for g in got] == [w[0] for w in want]
    assert [g[1] for g in got] == pytest.approx([w[1] for w in want], rel=1e-4)


def test_demo_dense_and_hybrid_agree_with_python(exported, fake_embedder) -> None:
    index, retriever, _, _ = exported
    for query in ["pottery studio", "snow clearing by-law 9999", "community garden plots"]:
        q = fake_embedder.embed_query(query)
        dense_py = [r.chunk.chunk_id for r in retriever.search(query, k=3, mode="dense")]
        dense_demo = [cid for cid, _ in index.search(query, q, k=3, mode="dense")]
        assert dense_demo[0] == dense_py[0]
        hybrid_py = [r.chunk.chunk_id for r in retriever.search(query, k=1, mode="hybrid")]
        assert index.search(query, q, k=1)[0][0] == hybrid_py[0]


def test_demo_date_filter(exported, fake_embedder) -> None:
    index, _, _, _ = exported
    q = fake_embedder.embed_query("agenda adopted")
    hits = index.search("agenda adopted", q, k=20, date_from="2024-01-01")
    assert hits and all(cid.startswith("2024-03-04") for cid, _ in hits)


def test_export_contains_no_full_text(exported) -> None:
    _, _, _, demo_dir = exported
    blob = b"".join(p.read_bytes() for p in demo_dir.iterdir())
    # sentences deep inside a long item must not be published; only excerpts are
    for sentence in (
        "A fourth site will have a tap connected next year",
        "Members of Council discussed enforcement and the hours of the ban",
    ):
        assert sentence.encode() not in blob
    for row in DemoIndex(demo_dir).chunks:
        assert len(row["excerpt"]) <= 300
