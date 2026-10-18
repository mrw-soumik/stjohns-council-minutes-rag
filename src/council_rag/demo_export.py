"""Export and read the static search index used by the browser demo (``docs/data``).

What is published (and what is not). The City's minutes are not redistributed: the
export contains no full text. Per chunk it holds meeting metadata, the agenda item
reference, heading and section path, a link to the official minutes page, an
int8-quantised embedding, and BM25 term counts keyed by a 32-bit hash of each term (no
readable word list). Decisions also carry the opening words of the motion (verbatim, at
most ``EXCERPT_MAX_CHARS`` characters) and the recorded outcome; discussion notes,
attendance and routine items carry no quoted text (see ``chunking.excerpt_for``).

Files:

* ``index.json``: metadata, meetings table, chunk table, layout of the binary files,
* ``vectors.bin``: ``int8[n, dim]`` followed by ``float32[n]`` per-vector scales,
* ``postings.bin``: BM25 inverted index (``uint32`` term hashes sorted ascending,
  ``uint32`` offsets, ``uint16`` chunk numbers, ``uint8`` term frequencies, ``uint16``
  document lengths).

:class:`DemoIndex` reads these files and ranks exactly like ``docs/assets/search.js``;
the CI retrieval gate and the Python/JavaScript parity check both use it.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from council_rag.chunking import EXCERPT_MAX_CHARS, excerpt_for
from council_rag.config import QUERY_INSTRUCTION
from council_rag.index import load_index
from council_rag.retrieval import rrf_fuse
from council_rag.text import tokenize

FORMAT_VERSION = 1
BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60
CANDIDATES = 50


def fnv1a32(text: str) -> int:
    """32-bit FNV-1a hash of the UTF-8 bytes (``Math.imul`` version in search.js)."""
    h = 0x811C9DC5
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def quantize_int8(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-vector int8 quantisation: ``v ~= q * scale``."""
    scales = np.abs(vectors).max(axis=1) / 127.0
    scales = np.where(scales == 0, 1.0, scales).astype(np.float32)
    q = np.clip(np.rint(vectors / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales


def _align(buf: bytearray, n: int = 4) -> None:
    while len(buf) % n:
        buf.append(0)


def export_demo(index_dir: Path, out_dir: Path) -> dict[str, Any]:
    chunks, embeddings, meta = load_index(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meetings: list[dict[str, Any]] = []
    meeting_pos: dict[str, int] = {}
    for c in sorted(chunks, key=lambda c: (c.meeting_date, c.meeting_id)):
        if c.meeting_id in meeting_pos:
            continue
        meeting_pos[c.meeting_id] = len(meetings)
        page_url = c.url.split("#", 1)[0] if c.url else None
        meetings.append(
            {
                "id": c.meeting_id,
                "date": c.meeting_date,
                "name": c.meeting_name,
                "url": page_url,
                "pdf": c.pdf_url,
            }
        )

    rows = []
    for c in chunks:
        ex = excerpt_for(c)
        # PA-13: a published excerpt is verbatim and never longer than the cap
        if len(ex.text) > EXCERPT_MAX_CHARS or ex.text not in c.text:
            raise ValueError(f"excerpt for {c.chunk_id} is too long or not verbatim")
        row = {
            "id": c.chunk_id,
            "m": meeting_pos[c.meeting_id],
            "ref": c.item_ref,
            "title": c.item_title,
            "path": " > ".join(c.item_path),
            "anchor": c.item_anchor,
            "kind": c.kind,
        }
        if ex.text:
            row["excerpt"] = ex.text
        if ex.cut:
            row["cut"] = True
        if ex.outcome:
            row["outcome"] = ex.outcome
        rows.append(row)

    # --- vectors -------------------------------------------------------------------
    q, scales = quantize_int8(embeddings.astype(np.float32))
    vec_buf = bytearray(q.tobytes())
    _align(vec_buf)
    scales_offset = len(vec_buf)
    vec_buf += scales.astype("<f4").tobytes()
    (out_dir / "vectors.bin").write_bytes(bytes(vec_buf))

    # --- BM25 postings over hashed terms -------------------------------------------
    postings: dict[int, Counter[int]] = {}
    doc_len = np.zeros(len(chunks), dtype=np.uint16)
    for i, c in enumerate(chunks):
        terms = tokenize(c.index_text(meta.index_with_context))
        doc_len[i] = min(len(terms), 65535)
        for term, tf in Counter(terms).items():
            postings.setdefault(fnv1a32(term), Counter())[i] += tf
    hashes = np.array(sorted(postings), dtype=np.uint32)
    offsets = np.zeros(len(hashes) + 1, dtype=np.uint32)
    doc_ids: list[int] = []
    tfs: list[int] = []
    for j, h in enumerate(hashes):
        for doc, tf in sorted(postings[int(h)].items()):
            doc_ids.append(doc)
            tfs.append(min(tf, 255))
        offsets[j + 1] = len(doc_ids)
    if len(chunks) > 65535:
        raise ValueError("demo format stores chunk numbers as uint16")
    sections: dict[str, tuple[int, int]] = {}
    post_buf = bytearray()
    for name, arr in (
        ("hashes", hashes.astype("<u4")),
        ("offsets", offsets.astype("<u4")),
        ("docs", np.array(doc_ids, dtype="<u2")),
        ("doclen", doc_len.astype("<u2")),
        ("tfs", np.array(tfs, dtype=np.uint8)),
    ):
        _align(post_buf)
        sections[name] = (len(post_buf), int(arr.size))
        post_buf += arr.tobytes()
    (out_dir / "postings.bin").write_bytes(bytes(post_buf))

    index = {
        "format": FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notice": (
            "Independent portfolio project. Not affiliated with or endorsed by the City of "
            "St. John's. Source documents are (c) City of St. John's; this index holds only "
            "metadata, short excerpts, embeddings and hashed term counts."
        ),
        "coverage": {"from": meta.date_min, "to": meta.date_max, "meetings": len(meetings)},
        "embedding": {
            "model": meta.embedding_repo,
            "revision": meta.embedding_revision,
            "dim": int(embeddings.shape[1]),
            "pooling": "cls",
            "normalize": True,
            "query_instruction": QUERY_INSTRUCTION,
            "quantization": "int8 per-vector scale",
        },
        "bm25": {
            "k1": BM25_K1,
            "b": BM25_B,
            "hash": "fnv1a32",
            "n_docs": len(chunks),
            "avgdl": float(doc_len.mean()) if len(chunks) else 0.0,
            "terms": int(hashes.size),
        },
        "rrf": {"k": RRF_K, "candidates": CANDIDATES},
        "files": {
            "vectors": {
                "path": "vectors.bin",
                "count": len(chunks),
                "scales_offset": scales_offset,
            },
            "postings": {"path": "postings.bin", "sections": sections},
        },
        "meetings": meetings,
        "chunks": rows,
    }
    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    return {
        "chunks": len(chunks),
        "meetings": len(meetings),
        "coverage": index["coverage"],
        "bytes": {p.name: p.stat().st_size for p in sorted(out_dir.glob("*")) if p.is_file()},
    }


class DemoIndex:
    """Reads ``docs/data`` and ranks like the browser (BM25 + int8 dense + RRF)."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        self.chunks: list[dict[str, Any]] = self.meta["chunks"]
        self.meetings: list[dict[str, Any]] = self.meta["meetings"]
        self.chunk_ids = [c["id"] for c in self.chunks]
        self.dates = np.array(
            [self.meetings[c["m"]]["date"] for c in self.chunks], dtype="datetime64[D]"
        )
        n, dim = len(self.chunks), self.meta["embedding"]["dim"]
        raw = (self.directory / "vectors.bin").read_bytes()
        self.q = np.frombuffer(raw, dtype=np.int8, count=n * dim).reshape(n, dim)
        off = self.meta["files"]["vectors"]["scales_offset"]
        self.scales = np.frombuffer(raw, dtype="<f4", count=n, offset=off)
        post = (self.directory / "postings.bin").read_bytes()
        sec = self.meta["files"]["postings"]["sections"]

        def view(name: str, dtype: str) -> np.ndarray:
            offset, count = sec[name]
            return np.frombuffer(post, dtype=dtype, count=count, offset=offset)

        self.hashes = view("hashes", "<u4")
        self.offsets = view("offsets", "<u4")
        self.docs = view("docs", "<u2")
        self.doclen = view("doclen", "<u2").astype(np.float32)
        self.tfs = view("tfs", np.uint8).astype(np.float32)
        bm = self.meta["bm25"]
        self.k1, self.b, self.n_docs, self.avgdl = bm["k1"], bm["b"], bm["n_docs"], bm["avgdl"]

    def mask(self, date_from: str | None, date_to: str | None) -> np.ndarray:
        m = np.ones(len(self.chunks), dtype=bool)
        if date_from:
            m &= self.dates >= np.datetime64(date_from, "D")
        if date_to:
            m &= self.dates <= np.datetime64(date_to, "D")
        return m

    def bm25_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        norm = self.k1 * (1.0 - self.b + self.b * self.doclen / max(self.avgdl, 1e-9))
        for term in tokenize(query):
            h = fnv1a32(term)
            j = int(np.searchsorted(self.hashes, h))
            if j >= self.hashes.size or self.hashes[j] != h:
                continue
            lo, hi = int(self.offsets[j]), int(self.offsets[j + 1])
            docs, tf = self.docs[lo:hi].astype(np.int64), self.tfs[lo:hi]
            df = hi - lo
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            scores[docs] += idf * tf * (self.k1 + 1.0) / (tf + norm[docs])
        return scores

    def dense_scores(self, query_vector: np.ndarray) -> np.ndarray:
        q = np.asarray(query_vector, dtype=np.float32)
        return (self.q.astype(np.float32) @ q) * self.scales

    @staticmethod
    def _rank(scores: np.ndarray, mask: np.ndarray, depth: int, positive: bool) -> list[int]:
        idx = np.flatnonzero(mask & (scores > 0)) if positive else np.flatnonzero(mask)
        order = idx[np.argsort(-scores[idx], kind="stable")]
        return [int(i) for i in order[:depth]]

    def search(
        self,
        query: str,
        query_vector: np.ndarray | None,
        k: int = 5,
        mode: str = "hybrid",
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[tuple[str, float]]:
        mask = self.mask(date_from, date_to)
        depth = max(k, self.meta["rrf"]["candidates"])
        bm25_list: list[int] = []
        dense_list: list[int] = []
        if mode in ("hybrid", "bm25"):
            s = self.bm25_scores(query)
            bm25_list = self._rank(s, mask, depth, positive=True)
            if mode == "bm25":
                return [(self.chunk_ids[i], float(s[i])) for i in bm25_list[:k]]
        if mode in ("hybrid", "dense"):
            if query_vector is None:
                raise ValueError("dense search needs a query vector")
            s = self.dense_scores(query_vector)
            dense_list = self._rank(s, mask, depth, positive=False)
            if mode == "dense":
                return [(self.chunk_ids[i], float(s[i])) for i in dense_list[:k]]
        fused = rrf_fuse(
            [[self.chunk_ids[i] for i in bm25_list], [self.chunk_ids[i] for i in dense_list]],
            k=self.meta["rrf"]["k"],
        )
        return fused[:k]
