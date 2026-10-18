import itertools
import os

import numpy as np
import pytest

from council_rag.retrieval import HybridRetriever, RetrievalConfig, rrf_fuse
from council_rag.vectorstore import InMemoryVectorStore


def test_rrf_fuse_rewards_agreement_and_is_deterministic() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["b", "d"]], k=60)
    assert fused[0][0] == "b"  # in both lists
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert [cid for cid, _ in fused] == ["b", "a", "d", "c"]
    weighted = rrf_fuse([["a"], ["b"]], weights=[0.5, 1.0], k=60)
    assert [cid for cid, _ in weighted] == ["b", "a"]


def _retriever(chunks, embedder, **cfg):
    vectors = embedder.embed_documents([c.index_text() for c in chunks])
    return HybridRetriever.build(chunks, vectors, embedder, RetrievalConfig(**cfg))


def test_modes_return_relevant_chunk_first(two_meeting_chunks, fake_embedder) -> None:
    r = _retriever(two_meeting_chunks, fake_embedder)
    for mode in ("bm25", "dense", "hybrid"):
        top = r.search("pottery studio home occupation", k=3, mode=mode)
        assert top[0].chunk.meeting_date == "2024-03-04", mode
        assert top[0].chunk.item_ref == "3.1", mode
        assert [h.rank for h in top] == list(range(1, len(top) + 1))


def test_hybrid_reports_component_ranks(two_meeting_chunks, fake_embedder) -> None:
    r = _retriever(two_meeting_chunks, fake_embedder)
    top = r.search("snow clearing by-law 9999", k=5, mode="hybrid")
    assert top[0].bm25_rank is not None and top[0].dense_rank is not None
    assert all(a.score >= b.score for a, b in itertools.pairwise(top))


def test_date_filter_applies_before_ranking(two_meeting_chunks, fake_embedder) -> None:
    r = _retriever(two_meeting_chunks, fake_embedder)
    for mode in ("bm25", "dense", "hybrid"):
        hits = r.search(
            "home occupation discretionary use", k=10, mode=mode, date_from="2024-01-01"
        )
        assert hits, mode
        assert all(h.chunk.meeting_date >= "2024-01-01" for h in hits), mode
    # only the 2023 meeting mentions a bicycle shop: filtering happens before ranking,
    # so keyword search inside 2024 finds nothing rather than an out-of-range match
    assert r.search("bicycle repair shop", k=10, mode="bm25", date_from="2024-01-01") == []
    early = r.search("bicycle repair shop", k=1, mode="bm25", date_to="2023-12-31")
    assert early[0].chunk.meeting_date == "2023-03-06"
    assert r.search("bicycle", k=5, date_from="2030-01-01") == []


def test_empty_query_returns_nothing(two_meeting_chunks, fake_embedder) -> None:
    assert _retriever(two_meeting_chunks, fake_embedder).search("   ") == []


def test_in_memory_store_upsert_search_and_filter() -> None:
    store = InMemoryVectorStore(dim=3)
    eye = np.eye(3, dtype=np.float32)
    store.upsert(["a", "b", "c"], ["2023-01-01", "2024-01-01", "2025-01-01"], eye)
    assert store.count() == 3
    assert store.search(np.array([0, 1, 0], dtype=np.float32), 1) == [("b", 1.0)]
    assert [i for i, _ in store.search(eye[0], 3, date_from="2024-01-01")] == ["b", "c"]
    store.upsert(["a"], ["2023-01-01"], np.array([[0, 0, 1]], dtype=np.float32))  # update
    assert store.count() == 3
    assert store.search(eye[2], 1, date_to="2023-12-31") == [("a", 1.0)]
    with pytest.raises(ValueError):
        store.upsert(["x"], ["2023-01-01"], np.ones((1, 4), dtype=np.float32))
    store.clear()
    assert store.count() == 0 and store.search(eye[0], 3) == []


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("PGVECTOR_TEST_DSN"), reason="PGVECTOR_TEST_DSN not set")
def test_pgvector_store_matches_in_memory(two_meeting_chunks, fake_embedder) -> None:
    from council_rag.vectorstore.pgvector import PgVectorStore

    chunks = two_meeting_chunks
    vectors = fake_embedder.embed_documents([c.index_text() for c in chunks])
    ids = [c.chunk_id for c in chunks]
    dates = [c.meeting_date for c in chunks]
    memory = InMemoryVectorStore()
    memory.upsert(ids, dates, vectors)
    pg = PgVectorStore(os.environ["PGVECTOR_TEST_DSN"], table="test_chunk_embeddings")
    try:
        pg.clear()
        pg.upsert(ids, dates, vectors)
        pg.upsert(ids[:3], dates[:3], vectors[:3])  # idempotent upsert
        assert pg.count() == len(ids)
        for query in ("pottery studio", "snow clearing by-law", "community garden water"):
            q = fake_embedder.embed_query(query)
            for lo, hi in ((None, None), ("2024-01-01", None), (None, "2023-12-31")):
                got = pg.search(q, 5, lo, hi)
                want = memory.search(q, 5, lo, hi)
                assert [g[0] for g in got] == [w[0] for w in want]
                assert [g[1] for g in got] == pytest.approx([w[1] for w in want], abs=1e-5)
        # the retriever works end to end on top of pgvector
        r = HybridRetriever.build(chunks, vectors, fake_embedder, store=pg)
        assert r.search("pottery studio", k=1)[0].chunk.meeting_date == "2024-03-04"
    finally:
        pg.drop()
        pg.close()
