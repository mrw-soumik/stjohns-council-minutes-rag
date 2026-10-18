"""PostgreSQL + pgvector store (psycopg 3).

Exact search by default: at a few thousand chunks a sequential scan with the ``<=>``
cosine-distance operator is fast and returns exactly the same ranking as the
in-memory store. ``hnsw=True`` adds an approximate HNSW index for larger corpora;
note that with a date filter an HNSW scan can return fewer than ``k`` rows on older
pgvector versions (0.8 adds iterative scans for this).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from council_rag.vectorstore.base import VectorStore

try:  # optional dependency: pip install ".[pgvector]"
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg import sql
except ImportError:  # pragma: no cover - exercised only without the extra installed
    psycopg = None  # type: ignore[assignment]


class PgVectorStore(VectorStore):
    def __init__(
        self,
        dsn: str,
        table: str = "chunk_embeddings",
        dim: int = 384,
        hnsw: bool = False,
    ) -> None:
        if psycopg is None:
            raise ImportError('PgVectorStore needs the extra: pip install ".[pgvector]"')
        self.dim = dim
        self.table = table
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        ident = sql.Identifier(table)
        self._conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} ("
                " id text PRIMARY KEY,"
                " meeting_date date NOT NULL,"
                " embedding vector({}) NOT NULL)"
            ).format(ident, sql.Literal(dim))
        )
        self._conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (meeting_date)").format(
                sql.Identifier(f"{table}_date_idx"), ident
            )
        )
        if hnsw:
            self._conn.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_cosine_ops)"
                ).format(sql.Identifier(f"{table}_hnsw_idx"), ident)
            )

    def close(self) -> None:
        self._conn.close()

    def upsert(self, ids: Sequence[str], dates: Sequence[str], vectors: np.ndarray) -> None:
        self._check(ids, dates, vectors)
        query = sql.SQL(
            "INSERT INTO {} (id, meeting_date, embedding) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET meeting_date = EXCLUDED.meeting_date, "
            "embedding = EXCLUDED.embedding"
        ).format(sql.Identifier(self.table))
        rows = [
            (chunk_id, day, np.asarray(vec, dtype=np.float32))
            for chunk_id, day, vec in zip(ids, dates, vectors, strict=True)
        ]
        with self._conn.cursor() as cur:
            cur.executemany(query, rows)

    def search(
        self,
        query: np.ndarray,
        k: int,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        stmt = sql.SQL(
            "SELECT id, 1 - (embedding <=> %(q)s) AS score FROM {} "
            "WHERE (%(lo)s::date IS NULL OR meeting_date >= %(lo)s::date) "
            "AND (%(hi)s::date IS NULL OR meeting_date <= %(hi)s::date) "
            "ORDER BY embedding <=> %(q)s, id LIMIT %(k)s"
        ).format(sql.Identifier(self.table))
        rows = self._conn.execute(stmt, {"q": q, "lo": date_from, "hi": date_to, "k": k})
        return [(row[0], float(row[1])) for row in rows.fetchall()]

    def count(self) -> int:
        row = self._conn.execute(
            sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(self.table))
        ).fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> None:
        self._conn.execute(sql.SQL("TRUNCATE {}").format(sql.Identifier(self.table)))

    def drop(self) -> None:
        self._conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(self.table)))
