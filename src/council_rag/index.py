"""Build, save and load the search index (chunks + embeddings) under ``data/index``.

The index holds full chunk text, so it lives in the gitignored data directory. The
public demo gets a reduced export instead (see ``council_rag.demo_export``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from council_rag.chunking import Chunk, ChunkingConfig, chunk_minutes
from council_rag.config import DataPaths
from council_rag.embeddings import Embedder, OnnxEmbedder
from council_rag.ingest.escribe import load_manifest
from council_rag.ingest.html_minutes import Minutes, load_minutes, parse_minutes_html, save_minutes

log = logging.getLogger(__name__)


@dataclass
class IndexMeta:
    created_at: str
    embedding_repo: str
    embedding_revision: str
    embedding_file: str
    max_words: int
    index_with_context: bool
    n_meetings: int
    n_chunks: int
    date_min: str
    date_max: str


def parse_all(paths: DataPaths) -> list[Minutes]:
    """Parse every downloaded minutes page listed in the manifest into ``data/parsed``."""
    manifest = load_manifest(paths)
    parsed: list[Minutes] = []
    for doc in manifest["documents"]:
        html_path = paths.root / doc["html_path"]
        minutes = parse_minutes_html(
            html_path.read_text(encoding="utf-8"),
            meeting_id=doc["meeting_id"],
            meeting_name=doc["meeting_name"],
            meeting_date=doc["meeting_date"],
            source_url=doc["minutes_html_url"],
            pdf_url=doc.get("pdf_url"),
            pdf_document_id=doc.get("pdf_document_id"),
        )
        if not minutes.items:
            log.warning("No agenda items parsed for %s", doc["meeting_date"])
            continue
        save_minutes(minutes, paths.parsed)
        parsed.append(minutes)
    parsed.sort(key=lambda m: (m.meeting_date, m.meeting_id))
    return parsed


def load_parsed(paths: DataPaths) -> list[Minutes]:
    files = sorted(paths.parsed.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No parsed minutes in {paths.parsed}. Run `council-rag parse`.")
    return sorted((load_minutes(f) for f in files), key=lambda m: (m.meeting_date, m.meeting_id))


def make_chunks(minutes: list[Minutes], config: ChunkingConfig | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for m in minutes:
        chunks.extend(chunk_minutes(m, config))
    return chunks


def build_index(
    paths: DataPaths,
    chunking: ChunkingConfig | None = None,
    embedder: Embedder | None = None,
    index_with_context: bool = True,
    out_dir: Path | None = None,
) -> tuple[list[Chunk], np.ndarray, IndexMeta]:
    chunking = chunking or ChunkingConfig()
    embedder = embedder or OnnxEmbedder()
    minutes = load_parsed(paths)
    chunks = make_chunks(minutes, chunking)
    texts = [c.index_text(index_with_context) for c in chunks]
    log.info("Embedding %d chunks", len(texts))
    embeddings = embedder.embed_documents(texts)
    meta = IndexMeta(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        embedding_repo=getattr(embedder, "repo_id", type(embedder).__name__),
        embedding_revision=getattr(embedder, "revision", ""),
        embedding_file=getattr(embedder, "onnx_file", ""),
        max_words=chunking.max_words,
        index_with_context=index_with_context,
        n_meetings=len(minutes),
        n_chunks=len(chunks),
        date_min=min(m.meeting_date for m in minutes),
        date_max=max(m.meeting_date for m in minutes),
    )
    save_index(out_dir or paths.index, chunks, embeddings, meta)
    return chunks, embeddings, meta


def save_index(
    directory: Path, chunks: list[Chunk], embeddings: np.ndarray, meta: IndexMeta
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_json(), ensure_ascii=False) + "\n")
    np.save(directory / "embeddings.npy", embeddings.astype(np.float32))
    (directory / "meta.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")


def load_index(directory: Path) -> tuple[list[Chunk], np.ndarray, IndexMeta]:
    if not (directory / "chunks.jsonl").exists():
        raise FileNotFoundError(f"No index in {directory}. Run `council-rag build-index`.")
    with (directory / "chunks.jsonl").open(encoding="utf-8") as fh:
        chunks = [Chunk.from_json(json.loads(line)) for line in fh if line.strip()]
    embeddings = np.load(directory / "embeddings.npy")
    meta = IndexMeta(**json.loads((directory / "meta.json").read_text(encoding="utf-8")))
    if embeddings.shape[0] != len(chunks):
        raise ValueError("index is inconsistent: embeddings and chunks differ in length")
    return chunks, embeddings, meta
