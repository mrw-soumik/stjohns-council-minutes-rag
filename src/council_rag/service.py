"""Wiring: load the index into a retriever, optionally backed by pgvector."""

from __future__ import annotations

import os
from pathlib import Path

from council_rag.config import data_paths
from council_rag.embeddings import Embedder, OnnxEmbedder
from council_rag.index import IndexMeta, load_index
from council_rag.retrieval import HybridRetriever, RetrievalConfig
from council_rag.vectorstore import VectorStore


def make_store(dim: int) -> VectorStore | None:
    """``COUNCIL_RAG_VECTOR_STORE=pgvector`` plus ``COUNCIL_RAG_PG_DSN`` selects pgvector."""
    if os.environ.get("COUNCIL_RAG_VECTOR_STORE", "memory").lower() == "pgvector":
        from council_rag.vectorstore.pgvector import PgVectorStore

        dsn = os.environ.get("COUNCIL_RAG_PG_DSN")
        if not dsn:
            raise RuntimeError("COUNCIL_RAG_PG_DSN must be set when using the pgvector store")
        return PgVectorStore(dsn, dim=dim)
    return None  # HybridRetriever.build defaults to the in-memory store


def load_retriever(
    index_dir: Path | None = None,
    embedder: Embedder | None = None,
    config: RetrievalConfig | None = None,
) -> tuple[HybridRetriever, IndexMeta]:
    index_dir = index_dir or data_paths().index
    chunks, embeddings, meta = load_index(index_dir)
    config = config or RetrievalConfig(index_with_context=meta.index_with_context)
    retriever = HybridRetriever.build(
        chunks,
        embeddings,
        embedder or OnnxEmbedder(),
        config=config,
        store=make_store(embeddings.shape[1]),
    )
    return retriever, meta
