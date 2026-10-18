"""Dense embeddings with BAAI/bge-small-en-v1.5 through ONNX Runtime.

The Python pipeline and the browser demo load the same ONNX export
(``Xenova/bge-small-en-v1.5`` at a pinned revision) and apply the same steps:
WordPiece tokenisation from ``tokenizer.json``, CLS pooling, L2 normalisation.
``scripts/embedding_parity.mjs`` checks that transformers.js reproduces these vectors.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from council_rag.config import (
    EMBEDDING_DIM,
    EMBEDDING_MAX_TOKENS,
    EMBEDDING_ONNX_FILE,
    EMBEDDING_REPO,
    EMBEDDING_REVISION,
    QUERY_INSTRUCTION,
    model_cache_dir,
)

log = logging.getLogger(__name__)


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class OnnxEmbedder:
    """bge-small-en-v1.5 sentence embeddings (384 dimensions)."""

    dim = EMBEDDING_DIM

    def __init__(
        self,
        repo_id: str = EMBEDDING_REPO,
        revision: str = EMBEDDING_REVISION,
        onnx_file: str = EMBEDDING_ONNX_FILE,
        cache_dir: Path | None = None,
        max_length: int = EMBEDDING_MAX_TOKENS,
        batch_size: int = 32,
        query_instruction: str = QUERY_INSTRUCTION,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.onnx_file = onnx_file
        self.cache_dir = cache_dir or model_cache_dir()
        self.max_length = max_length
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self._session = None
        self._tokenizer = None
        self._input_names: list[str] = []
        self.truncated = 0  # documents longer than max_length tokens seen so far

    def _load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        kwargs = {"repo_id": self.repo_id, "revision": self.revision, "cache_dir": self.cache_dir}
        tokenizer_path = hf_hub_download(filename="tokenizer.json", **kwargs)
        model_path = hf_hub_download(filename=self.onnx_file, **kwargs)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_truncation(max_length=self.max_length)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 0
        self._session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._tokenizer = tokenizer
        log.info("Loaded %s@%s (%s)", self.repo_id, self.revision[:8], self.onnx_file)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        assert self._tokenizer is not None and self._session is not None
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encodings = self._tokenizer.encode_batch(batch)
            self.truncated += sum(1 for e in encodings if e.overflowing)
            feeds = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
            }
            feeds = {k: v for k, v in feeds.items() if k in self._input_names}
            hidden = self._session.run(None, feeds)[0]  # (batch, seq, dim)
            out[start : start + len(batch)] = hidden[:, 0, :]  # CLS pooling
        return l2_normalize(out)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([f"{self.query_instruction}{text}"])[0]

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Token counts without truncation (for chunk-size diagnostics)."""
        self._load()
        assert self._tokenizer is not None
        self._tokenizer.no_truncation()
        try:
            return [sum(e.attention_mask) for e in self._tokenizer.encode_batch(list(texts))]
        finally:
            self._tokenizer.enable_truncation(max_length=self.max_length)
