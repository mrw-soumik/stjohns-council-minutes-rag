"""Retrieval evaluation: hit@1/3/5, recall@5, MRR@10 and latency, per configuration.

Two sources of rankings:

* ``--source full`` (default): rebuilds chunks and embeddings from ``data/parsed`` for
  each configuration (embeddings cached under ``data/index/eval_cache``), judges
  relevance from chunk text, and can write ``eval/qrels.json``.
* ``--source demo``: ranks with the committed browser index in ``docs/data`` and the
  committed ``eval/qrels.json``. Needs no City downloads, so CI runs it as the
  NFR-01 gate.

Every number written to ``eval/results`` comes from an actual run of this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from council_rag.chunking import Chunk, ChunkingConfig
from council_rag.config import data_paths
from council_rag.embeddings import OnnxEmbedder
from council_rag.evaluation.dataset import EvalQuestion, build_qrels, load_questions, relevant_sets
from council_rag.evaluation.metrics import (
    hit_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
    wilson_interval,
)
from council_rag.index import load_parsed, make_chunks
from council_rag.retrieval import HybridRetriever, RetrievalConfig

log = logging.getLogger(__name__)
NFR01_THRESHOLD = 0.85


@dataclass(frozen=True)
class RunConfig:
    name: str
    description: str
    mode: str = "hybrid"
    max_words: int = 250
    with_context: bool = True
    include_notes: bool = True
    rrf_k: int = 60
    candidates: int = 50
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    query_instruction: bool = True
    onnx_file: str = "onnx/model.onnx"
    int8_docs: bool = False
    measure_latency: bool = False


MAIN_CONFIGS = [
    RunConfig("bm25", "BM25 only (keyword)", mode="bm25", measure_latency=True),
    RunConfig("dense", "Dense only (bge-small-en-v1.5)", mode="dense", measure_latency=True),
    RunConfig("hybrid", "Hybrid: BM25 + dense, RRF k=60", mode="hybrid", measure_latency=True),
]

ABLATIONS = [
    RunConfig(
        "hybrid-no-context",
        "Hybrid, chunk text indexed without meeting/item path",
        with_context=False,
    ),
    RunConfig(
        "hybrid-no-notes", "Hybrid, discussion notes dropped (motions only)", include_notes=False
    ),
    RunConfig("hybrid-words-120", "Hybrid, chunks of at most 120 words", max_words=120),
    RunConfig("hybrid-words-400", "Hybrid, chunks of at most 400 words", max_words=400),
    RunConfig("hybrid-rrf-k10", "Hybrid, RRF k=10", rrf_k=10),
    RunConfig("hybrid-rrf-k100", "Hybrid, RRF k=100", rrf_k=100),
    RunConfig("hybrid-bm25x0.5", "Hybrid, BM25 weight 0.5", bm25_weight=0.5),
    RunConfig("hybrid-densex0.5", "Hybrid, dense weight 0.5", dense_weight=0.5),
    RunConfig(
        "dense-no-instruction",
        "Dense, query without the bge instruction prefix",
        mode="dense",
        query_instruction=False,
    ),
    RunConfig(
        "hybrid-no-instruction",
        "Hybrid, query without the bge instruction prefix",
        query_instruction=False,
    ),
    RunConfig(
        "hybrid-int8-docs", "Hybrid, int8-quantised document vectors (demo index)", int8_docs=True
    ),
    RunConfig(
        "hybrid-q8-query",
        "Hybrid, int8-quantised query model (browser q8 weights)",
        onnx_file="onnx/model_quantized.onnx",
    ),
]


def _summarise(
    cfg: RunConfig,
    questions: Sequence[EvalQuestion],
    rankings: dict[str, list[str]],
    rel: dict[str, list[set[str]]],
    rel_item: dict[str, list[set[str]]] | None,
    latencies: list[float],
    n_chunks: int,
    violations: int,
) -> dict[str, Any]:
    answerable = [q for q in questions if q.answerable]
    per_q: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, float]]] = defaultdict(list)
    for q in answerable:
        ranked = rankings[q.id]
        row = {
            "hit@1": hit_at_k(ranked, rel[q.id], 1),
            "hit@3": hit_at_k(ranked, rel[q.id], 3),
            "hit@5": hit_at_k(ranked, rel[q.id], 5),
            "recall@5": recall_at_k(ranked, rel[q.id], 5),
            "mrr@10": reciprocal_rank(ranked, rel[q.id], 10),
        }
        if rel_item is not None:
            row["item_hit@5"] = hit_at_k(ranked, rel_item[q.id], 5)
        first = next(
            (i for i, cid in enumerate(ranked[:10], 1) if cid in set().union(*rel[q.id])), None
        )
        per_q.append({"id": q.id, "type": q.type, "first_relevant_rank": first, **row})
        by_type[q.type].append(row)

    def mean(key: str, rows: Sequence[dict[str, float]]) -> float:
        return round(float(np.mean([r[key] for r in rows])), 4) if rows else float("nan")

    metric_keys = ["hit@1", "hit@3", "hit@5", "recall@5", "mrr@10"]
    hits5 = sum(r["hit@5"] for r in per_q)
    lo, hi = wilson_interval(hits5, len(per_q))
    out: dict[str, Any] = {
        "config": asdict(cfg),
        "n_answerable": len(per_q),
        "n_chunks": n_chunks,
        "metrics": {k: mean(k, per_q) for k in metric_keys},
        "hit@5_ci95": [round(lo, 4), round(hi, 4)],
        "by_type": {
            t: {"n": len(rows), "hit@5": mean("hit@5", rows), "mrr@10": mean("mrr@10", rows)}
            for t, rows in sorted(by_type.items())
        },
        "date_filter_violations": violations,
        "misses_at_5": [r["id"] for r in per_q if r["hit@5"] == 0],
        "per_question": per_q,
    }
    if rel_item is not None:
        out["metrics"]["item_hit@5"] = mean("item_hit@5", per_q)
    if latencies:
        out["latency_ms"] = {
            "mean": round(float(np.mean(latencies)), 2),
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "n": len(latencies),
        }
    return out


class _EmbeddingCache:
    """Caches document embeddings per (model file, indexed texts) on disk."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.embedders: dict[str, OnnxEmbedder] = {}

    def embedder(self, onnx_file: str) -> OnnxEmbedder:
        if onnx_file not in self.embedders:
            self.embedders[onnx_file] = OnnxEmbedder(onnx_file=onnx_file)
        return self.embedders[onnx_file]

    def documents(self, texts: list[str], onnx_file: str = "onnx/model.onnx") -> np.ndarray:
        digest = hashlib.sha256(("\x1e".join([onnx_file, *texts])).encode()).hexdigest()[:16]
        path = self.directory / f"{digest}.npy"
        if path.exists():
            return np.load(path)
        vectors = self.embedder(onnx_file).embed_documents(texts)
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors


def _int8_roundtrip(vectors: np.ndarray) -> np.ndarray:
    from council_rag.demo_export import quantize_int8

    q, scales = quantize_int8(vectors)
    return q.astype(np.float32) * scales[:, None]


def run_full(
    questions: list[EvalQuestion],
    configs: Sequence[RunConfig],
    qrels_out: Path | None = None,
) -> list[dict[str, Any]]:
    paths = data_paths()
    minutes = load_parsed(paths)
    cache = _EmbeddingCache(paths.index / "eval_cache")
    chunk_sets: dict[tuple[int, bool], list[Chunk]] = {}
    results = []
    query_vectors: dict[tuple[str, bool, str], np.ndarray] = {}

    for cfg in configs:
        key = (cfg.max_words, cfg.include_notes)
        if key not in chunk_sets:
            chunk_sets[key] = make_chunks(
                minutes, ChunkingConfig(max_words=cfg.max_words, include_notes=cfg.include_notes)
            )
        chunks = chunk_sets[key]
        texts = [c.index_text(cfg.with_context) for c in chunks]
        docs = cache.documents(texts)  # documents always use the fp32 model
        if cfg.int8_docs:
            docs = _int8_roundtrip(docs)
        embedder = cache.embedder(cfg.onnx_file)
        if not cfg.query_instruction:
            embedder = replace_instruction(embedder, "")
        retriever = HybridRetriever.build(
            chunks,
            docs,
            embedder,
            RetrievalConfig(
                rrf_k=cfg.rrf_k,
                candidates=cfg.candidates,
                bm25_weight=cfg.bm25_weight,
                dense_weight=cfg.dense_weight,
                index_with_context=cfg.with_context,
            ),
        )
        rel = {q.id: relevant_sets(q, chunks) for q in questions if q.answerable}
        rel_item = {q.id: relevant_sets(q, chunks, "item") for q in questions if q.answerable}
        if qrels_out is not None and cfg.name == "hybrid":
            qrels_out.write_text(
                json.dumps(build_qrels(questions, chunks), indent=1) + "\n", encoding="utf-8"
            )
        if cfg.measure_latency and cfg.mode != "bm25":
            retriever.search("warm up the embedding model", k=5, mode=cfg.mode)

        rankings: dict[str, list[str]] = {}
        latencies: list[float] = []
        violations = 0
        by_id = {c.chunk_id: c for c in chunks}
        for q in questions:
            if not q.answerable:
                continue
            qkey = (q.question, cfg.query_instruction, cfg.onnx_file)
            vector = None
            if cfg.mode != "bm25" and not cfg.measure_latency:
                if qkey not in query_vectors:
                    query_vectors[qkey] = embedder.embed_query(q.question)
                vector = query_vectors[qkey]
            started = time.perf_counter()
            hits = retriever.search(
                q.question,
                k=10,
                mode=cfg.mode,
                date_from=q.date_from,
                date_to=q.date_to,
                query_vector=vector,
            )
            if cfg.measure_latency:
                latencies.append((time.perf_counter() - started) * 1000)
            rankings[q.id] = [h.chunk.chunk_id for h in hits]
            for h in hits:
                d = by_id[h.chunk.chunk_id].meeting_date
                if (q.date_from and d < q.date_from) or (q.date_to and d > q.date_to):
                    violations += 1
        summary = _summarise(
            cfg, questions, rankings, rel, rel_item, latencies, len(chunks), violations
        )
        unreachable = [q.id for q in questions if q.answerable and not any(rel[q.id])]
        summary["unreachable_gold"] = unreachable
        results.append(summary)
        log.info("%s: hit@5=%.3f", cfg.name, summary["metrics"]["hit@5"])
    return results


def replace_instruction(embedder: OnnxEmbedder, instruction: str) -> OnnxEmbedder:
    clone = OnnxEmbedder(
        repo_id=embedder.repo_id,
        revision=embedder.revision,
        onnx_file=embedder.onnx_file,
        cache_dir=embedder.cache_dir,
        query_instruction=instruction,
    )
    clone._session, clone._tokenizer = embedder._session, embedder._tokenizer
    clone._input_names = embedder._input_names
    return clone


def run_demo(
    questions: list[EvalQuestion],
    demo_dir: Path,
    qrels_path: Path,
    modes: Sequence[str],
    onnx_file: str = "onnx/model.onnx",
) -> list[dict[str, Any]]:
    from council_rag.demo_export import DemoIndex

    index = DemoIndex(demo_dir)
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    embedder = OnnxEmbedder(onnx_file=onnx_file)
    known = set(index.chunk_ids)
    rel: dict[str, list[set[str]]] = {}
    for q in questions:
        if q.answerable:
            if q.id not in qrels:
                raise KeyError(f"{q.id} missing from {qrels_path}; regenerate qrels")
            rel[q.id] = [set(ids) & known for ids in qrels[q.id]]
    vectors = {q.id: embedder.embed_query(q.question) for q in questions if q.answerable}
    results = []
    for mode in modes:
        cfg = RunConfig(
            f"demo-{mode}",
            f"Committed demo index, {mode}",
            mode=mode,
            int8_docs=True,
            onnx_file=onnx_file,
        )
        rankings, latencies, violations = {}, [], 0
        for q in questions:
            if not q.answerable:
                continue
            started = time.perf_counter()
            hits = index.search(
                q.question, vectors[q.id], k=10, mode=mode, date_from=q.date_from, date_to=q.date_to
            )
            latencies.append((time.perf_counter() - started) * 1000)
            rankings[q.id] = [cid for cid, _ in hits]
            pos = {cid: i for i, cid in enumerate(index.chunk_ids)}
            for cid, _ in hits:
                d = str(index.dates[pos[cid]])
                if (q.date_from and d < q.date_from) or (q.date_to and d > q.date_to):
                    violations += 1
        summary = _summarise(
            cfg, questions, rankings, rel, None, latencies, len(index.chunk_ids), violations
        )
        summary["latency_note"] = "ranking only; query vectors precomputed"
        results.append(summary)
    return results


def write_results(results: list[dict[str, Any]], out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
    }
    path = out_dir / f"{tag}.json"
    path.write_text(
        json.dumps({"environment": env, "runs": results}, indent=1) + "\n", encoding="utf-8"
    )
    return path


def markdown_table(results: list[dict[str, Any]], with_latency: bool = True) -> str:
    head = "| Configuration | hit@1 | hit@3 | hit@5 (95% CI) | recall@5 | MRR@10 |"
    sep = "|---|---|---|---|---|---|"
    if with_latency:
        head += " p50 / p95 latency (ms) |"
        sep += "---|"
    lines = [head, sep]
    for r in results:
        m = r["metrics"]
        lo, hi = r["hit@5_ci95"]
        row = (
            f"| {r['config']['description']} | {m['hit@1']:.3f} | {m['hit@3']:.3f} | "
            f"{m['hit@5']:.3f} ({lo:.2f} to {hi:.2f}) | {m['recall@5']:.3f} | {m['mrr@10']:.3f} |"
        )
        if with_latency:
            lat = r.get("latency_ms")
            row += f" {lat['p50']:.1f} / {lat['p95']:.1f} |" if lat else " n/a |"
        lines.append(row)
    return "\n".join(lines)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument("--source", choices=["full", "demo"], default="full")
    p.add_argument(
        "--configs",
        default="main",
        help="'main', 'ablations', 'all', or comma-separated config names",
    )
    p.add_argument("--demo-dir", default="docs/data")
    p.add_argument("--qrels", default="eval/qrels.json")
    p.add_argument(
        "--write-qrels", action="store_true", help="(full source) write relevance judgements for CI"
    )
    p.add_argument("--demo-modes", default="bm25,dense,hybrid")
    p.add_argument(
        "--query-model",
        default="onnx/model.onnx",
        help="(demo source) ONNX file used for query embeddings",
    )
    p.add_argument("--out", default="eval/results")
    p.add_argument("--tag", default=None, help="results file name (default from source)")
    p.add_argument(
        "--gate",
        type=float,
        default=None,
        help=f"fail if hit@5 of --gate-config is below this (NFR-01: {NFR01_THRESHOLD})",
    )
    p.add_argument("--gate-config", default="hybrid")


def main(args: argparse.Namespace) -> int:
    questions = load_questions(args.questions)
    if args.source == "demo":
        modes = [m.strip() for m in args.demo_modes.split(",") if m.strip()]
        results = run_demo(
            questions, Path(args.demo_dir), Path(args.qrels), modes, onnx_file=args.query_model
        )
        tag = args.tag or "demo_index"
    else:
        lookup = {c.name: c for c in [*MAIN_CONFIGS, *ABLATIONS]}
        if args.configs == "main":
            configs = MAIN_CONFIGS
        elif args.configs == "ablations":
            configs = [replace(MAIN_CONFIGS[2], measure_latency=False), *ABLATIONS]
        elif args.configs == "all":
            configs = [*MAIN_CONFIGS, *ABLATIONS]
        else:
            configs = [lookup[name.strip()] for name in args.configs.split(",")]
        qrels_out = Path(args.qrels) if args.write_qrels else None
        results = run_full(questions, configs, qrels_out)
        tag = args.tag or ("retrieval_main" if args.configs == "main" else "retrieval_ablations")
    path = write_results(results, Path(args.out), tag)
    print(markdown_table(results, with_latency=any("latency_ms" in r for r in results)))
    print(f"\nWrote {path}")

    if args.gate is not None:
        gate_name = args.gate_config if args.source == "full" else f"demo-{args.gate_config}"
        match = [r for r in results if r["config"]["name"] == gate_name]
        if not match:
            print(f"Gate config {gate_name!r} was not run")
            return 2
        value = match[0]["metrics"]["hit@5"]
        verdict = "PASS" if value >= args.gate else "FAIL"
        print(
            f"NFR-01 gate ({gate_name}): hit@5 = {value:.3f}, threshold {args.gate:.2f}: {verdict}"
        )
        if value < args.gate:
            return 1
        if any(r["date_filter_violations"] for r in results):
            print("Date filter returned out-of-range results")
            return 1
    return 0
