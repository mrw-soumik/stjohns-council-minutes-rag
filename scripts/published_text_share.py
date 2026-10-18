"""Measure how much of the minutes text the public demo index publishes (README, PA-13).

Compares what ``docs/data/index.json`` holds with the parsed minutes in ``data/`` (local
only, never committed): the verbatim excerpts, the recorded outcomes shown next to them,
and, reported separately, the item titles and section headings shown with every result.
It also re-checks PA-13 against the full source passages: every excerpt is an exact
substring of its passage and no longer than ``EXCERPT_MAX_CHARS``.

Writes ``eval/results/published_text.json``.

Usage: python scripts/published_text_share.py   (needs the parsed minutes and the index)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from council_rag.chunking import EXCERPT_MAX_CHARS
from council_rag.config import data_paths
from council_rag.index import load_index, load_parsed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results" / "published_text.json"


def main() -> int:
    paths = data_paths()
    minutes = load_parsed(paths)
    parsed_chars = sum(len(item.render()) for m in minutes for item in m.items) + sum(
        len(m.attendance_text()) for m in minutes
    )
    source = {c.chunk_id: c.text for c in load_index(paths.index)[0]}
    demo = json.loads((ROOT / "docs" / "data" / "index.json").read_text(encoding="utf-8"))
    rows = demo["chunks"]

    excerpts = [r["excerpt"] for r in rows if r.get("excerpt")]
    outcomes = [r["outcome"] for r in rows if r.get("outcome")]
    violations = [
        r["id"]
        for r in rows
        if r.get("excerpt")
        and (len(r["excerpt"]) > EXCERPT_MAX_CHARS or r["excerpt"] not in source[r["id"]])
    ]
    # titles and headings are shown with every result; count each distinct one once
    titles = {(r["m"], r["anchor"], r["ref"], r["kind"]): r["title"] for r in rows}
    headings = {(r["m"], h) for r in rows for h in r["path"].split(" > ") if h}

    def share(chars: int) -> float:
        return round(chars / parsed_chars, 4)

    ex_chars = sum(map(len, excerpts))
    out_chars = sum(map(len, outcomes))
    title_chars = sum(map(len, titles.values()))
    heading_chars = sum(len(h) for _, h in headings)
    report = {
        "parsed_minutes_chars": parsed_chars,
        "passages": len(rows),
        "excerpt_max_chars": EXCERPT_MAX_CHARS,
        "excerpts": {"count": len(excerpts), "chars": ex_chars, "share": share(ex_chars)},
        "outcomes": {"count": len(outcomes), "chars": out_chars, "share": share(out_chars)},
        "excerpts_and_outcomes_share": share(ex_chars + out_chars),
        "item_titles": {"count": len(titles), "chars": title_chars, "share": share(title_chars)},
        "section_headings": {
            "count": len(headings),
            "chars": heading_chars,
            "share": share(heading_chars),
        },
        "pa13": {
            "checked": len(excerpts),
            "longest": max(map(len, excerpts), default=0),
            "violations": violations,
        },
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
