"""Vector store back ends: in-memory NumPy (default) and PostgreSQL + pgvector."""

from council_rag.vectorstore.base import VectorStore
from council_rag.vectorstore.memory import InMemoryVectorStore

__all__ = ["InMemoryVectorStore", "VectorStore"]
