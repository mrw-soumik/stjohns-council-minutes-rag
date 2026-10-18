"""Estimate what the optional Claude backend would cost on the evaluation set (no API calls).

Builds the exact request the ``claude`` backend would send for each question (hybrid
retrieval, same k and date filters as the answer evaluation) and counts its characters.
Tokens are estimated, not measured: input at 3.5 characters per token plus 10% for the
citations feature's document chunking, and output as a range because adaptive thinking
length is unknown until a run. Prices are the claude-opus-5-5 list prices in pricing.py.

Usage: python scripts/estimate_claude_cost.py [-k 6]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from council_rag.config import CLAUDE_MODEL
from council_rag.evaluation.dataset import load_questions
from council_rag.generation import build_request
from council_rag.pricing import PRICES_USD_PER_MTOK, USD_TO_CAD
from council_rag.service import load_retriever

ROOT = Path(__file__).resolve().parents[1]
CHARS_PER_TOKEN = 3.5
CITATIONS_OVERHEAD = 1.10
OUTPUT_SCENARIOS = {"low": 200, "high": 1200}  # visible answer + adaptive thinking tokens


def request_chars(request: dict) -> int:
    total = len(request["system"])
    for block in request["messages"][0]["content"]:
        if block["type"] == "document":
            total += len(block["source"]["data"]) + len(block["title"]) + len(block["context"])
        else:
            total += len(block["text"])
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=6)
    args = parser.parse_args()
    retriever, _ = load_retriever()
    price = PRICES_USD_PER_MTOK[CLAUDE_MODEL]
    chars = []
    for q in load_questions(ROOT / "eval" / "questions.jsonl"):
        hits = retriever.search(q.question, k=args.k, date_from=q.date_from, date_to=q.date_to)
        chars.append(request_chars(build_request(q.question, [h.chunk for h in hits])))
    tokens_in = [c / CHARS_PER_TOKEN * CITATIONS_OVERHEAD for c in chars]
    mean_in = statistics.mean(tokens_in)
    result = {
        "model": CLAUDE_MODEL,
        "k": args.k,
        "questions": len(chars),
        "request_chars": {"mean": round(statistics.mean(chars)), "max": max(chars)},
        "estimated_input_tokens": {"mean": round(mean_in), "max": round(max(tokens_in))},
        "assumptions": {
            "chars_per_token": CHARS_PER_TOKEN,
            "citations_overhead": CITATIONS_OVERHEAD,
            "output_tokens": OUTPUT_SCENARIOS,
            "usd_to_cad": USD_TO_CAD,
        },
        "per_answer_usd": {},
    }
    for name, out_tokens in OUTPUT_SCENARIOS.items():
        usd = (mean_in * price.input + out_tokens * price.output) / 1_000_000
        result["per_answer_usd"][name] = {
            "usd": round(usd, 4),
            "cad": round(usd * USD_TO_CAD, 4),
            "full_eval_run_usd": round(usd * len(chars), 2),
        }
    out = ROOT / "eval" / "results" / "claude_cost_estimate.json"
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
