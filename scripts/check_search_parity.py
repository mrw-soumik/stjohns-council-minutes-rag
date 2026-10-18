"""Check that the browser search code ranks exactly like the Python reader of the same index.

For every evaluation question, embeds the query with the q8 model the browser uses
(one text per call, as in the browser), ranks it with :class:`DemoIndex` in all three
modes, then runs ``scripts/check_search_parity.mjs`` (which imports
``docs/assets/search.js``) on the same vectors and compares the top-10 lists. Shared
tokenizer cases and FNV-1a reference values are checked too.

Usage: python scripts/check_search_parity.py   (needs Node; no City downloads)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from council_rag.demo_export import DemoIndex, fnv1a32
from council_rag.embeddings import OnnxEmbedder
from council_rag.evaluation.dataset import load_questions

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / ".cache" / "parity" / "search_cases.json"


def main() -> int:
    index = DemoIndex(ROOT / "docs" / "data")
    embedder = OnnxEmbedder(onnx_file="onnx/model_quantized.onnx", batch_size=1)
    queries = []
    for q in load_questions(ROOT / "eval" / "questions.jsonl"):
        vector = embedder.embed_query(q.question)
        python = {
            mode: [
                cid
                for cid, _ in index.search(
                    q.question, vector, k=10, mode=mode, date_from=q.date_from, date_to=q.date_to
                )
            ]
            for mode in ("bm25", "dense", "hybrid")
        }
        queries.append(
            {
                "id": q.id,
                "question": q.question,
                "date_from": q.date_from,
                "date_to": q.date_to,
                "vector": [float(x) for x in vector],
                "python": python,
            }
        )
    tokenizer = json.loads((ROOT / "tests" / "fixtures" / "tokenizer_cases.json").read_text())
    fnv = [[t, fnv1a32(t)] for t in ("", "a", "foobar", "bylaw", "council", "1736")]
    CASES.parent.mkdir(parents=True, exist_ok=True)
    CASES.write_text(json.dumps({"queries": queries, "tokenizer": tokenizer, "fnv1a": fnv}))
    proc = subprocess.run(
        ["node", "scripts/check_search_parity.mjs", str(CASES)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr.strip(), file=sys.stderr)
        return proc.returncode
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    ok = all(v["same_top1"] == v["n"] for v in summary.values())
    print("browser ranking matches Python" if ok else "MISMATCH in top-1 results")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
