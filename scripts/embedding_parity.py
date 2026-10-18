"""Check that transformers.js (browser demo) and ONNX Runtime (Python) embed text alike.

Runs ``scripts/embedding_parity.mjs`` with Node, embeds the same texts in Python with
the fp32 export and the int8 ("q8", ``model_quantized.onnx``) export, and reports
cosine similarity per pair. Results go to ``eval/results/embedding_parity.json``.

Usage: python scripts/embedding_parity.py   (needs ``npm install`` first)
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from council_rag.config import EMBEDDING_REPO, EMBEDDING_REVISION, QUERY_INSTRUCTION
from council_rag.embeddings import OnnxEmbedder

ROOT = Path(__file__).resolve().parents[1]
JS_OUT = ROOT / ".cache" / "parity" / "js_vectors.json"
RESULT = ROOT / "eval" / "results" / "embedding_parity.json"


def _node_version() -> str:
    return subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()


def _js_version() -> str:
    pkg = ROOT / "node_modules" / "@huggingface" / "transformers" / "package.json"
    return json.loads(pkg.read_text(encoding="utf-8"))["version"]


def _ort_version() -> str:
    import onnxruntime

    return onnxruntime.__version__


def main() -> int:
    subprocess.run(
        [
            "node",
            "scripts/embedding_parity.mjs",
            str(JS_OUT),
            EMBEDDING_REVISION,
            QUERY_INSTRUCTION,
        ],
        cwd=ROOT,
        check=True,
    )
    js = json.loads(JS_OUT.read_text(encoding="utf-8"))
    texts = js["texts"]
    # One text per call, as in the browser: the int8 model quantises activations
    # dynamically over the whole batch, so padded batches would change its output.
    py = {
        "fp32": OnnxEmbedder(onnx_file="onnx/model.onnx", batch_size=1).embed_documents(texts),
        "q8": OnnxEmbedder(onnx_file="onnx/model_quantized.onnx", batch_size=1).embed_documents(
            texts
        ),
    }
    report: dict = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": f"{EMBEDDING_REPO}@{EMBEDDING_REVISION}",
        "js_library": f"{js['library']} {_js_version()} on Node {_node_version()}",
        "python": f"onnxruntime {_ort_version()} on Python {platform.python_version()}",
        "n_texts": len(texts),
        "comparisons": {},
    }
    pairs = {
        "python_fp32_vs_js_fp32": (py["fp32"], np.array(js["vectors"]["fp32"])),
        "python_q8_vs_js_q8": (py["q8"], np.array(js["vectors"]["q8"])),
        "js_fp32_vs_js_q8": (np.array(js["vectors"]["fp32"]), np.array(js["vectors"]["q8"])),
    }
    for name, (a, b) in pairs.items():
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        report["comparisons"][name] = {
            "cosine_min": round(float(cos.min()), 6),
            "cosine_mean": round(float(cos.mean()), 6),
            "max_abs_diff": round(float(np.abs(a - b).max()), 6),
        }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["comparisons"], indent=1))
    ok = report["comparisons"]["python_fp32_vs_js_fp32"]["cosine_min"] > 0.999
    print("near-identical" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
