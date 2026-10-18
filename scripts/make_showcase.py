"""Write docs/data/showcase.json: precomputed answers shown in the browser demo.

The question ids were fixed before the answer evaluation ran, and the answers are copied
as generated (including any mistakes) from that run's results, so the showcase is not
cherry-picked. Answers are short model-written sentences; citations carry only the
meeting date, agenda item, title and official link.

Usage: python scripts/make_showcase.py [eval/results/answers_local_k6]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Chosen before the full run: a mix of question types, including two unanswerable ones.
SHOWCASE_IDS = [
    "vo12",  # who voted against the 2026 budget
    "pa09",  # renaming the June holiday (paraphrased wording)
    "sf10",  # heritage designation of 157-163 Water Street (two meetings)
    "vo10",  # tied vote on 175 Ladysmith Drive
    "vo04",  # outdoor lounge at 75 Airport Heights Drive (lost 0 to 9)
    "nl07",  # 2024 budget total
    "nl13",  # school zone speed limit and hours
    "sf06",  # Poet Laureate appointment
    "sf13",  # Heritage Strategic Plan 2026-2036
    "pa11",  # the City's X account (paraphrased wording)
    "un06",  # rent control: not in the minutes
    "un08",  # 2027 budget: not in the minutes yet
]


# Notes from the read-through in eval/results/answers_local_k6/review.md. Answers are never
# edited; a note is shown next to any answer that the automatic check passed but that contains
# an error.
REVIEW_NOTES = {
    "sf10": "Review note: the second sentence is wrong. The motion to rescind the April 21 "
    "decision was lost 2 to 8 on May 5, 2026, so the decision not to designate stood.",
}


def main() -> int:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eval/results/answers_local_k6"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = {
        json.loads(line)["id"]: json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    from council_rag.answers import NOT_FOUND_MESSAGE

    items = []
    for qid in SHOWCASE_IDS:
        row = rows[qid]
        answered = row["status"] == "answered"
        items.append(
            {
                "id": qid,
                "question": row["question"],
                "status": row["status"],
                "answer": row["answer"] if answered else NOT_FOUND_MESSAGE,
                "citations": [
                    {k: c[k] for k in ("meeting_date", "item_ref", "item_title", "url")}
                    for c in row["citations"]
                ]
                if answered
                else [],
                "auto_check": "passed"
                if row["grade"].get("correct", row["grade"].get("declined"))
                else "failed",
                "review_note": REVIEW_NOTES.get(qid),
            }
        )
    meta = summary["meta"]
    generated_on = datetime.fromisoformat(meta["run_at"]).astimezone().date().isoformat()
    out = {
        "model": meta["model"],
        "generated_on": generated_on,
        "k_passages": meta["k"],
        "source": str(run_dir.relative_to(ROOT)) + "/results.jsonl",
        "note": "Question ids were fixed before the run; answers are shown as generated, with "
        "review notes where a read-through found an error.",
        "items": items,
    }
    path = ROOT / "docs" / "data" / "showcase.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    passed = sum(1 for i in items if i["auto_check"] == "passed")
    noted = sum(1 for i in items if i["review_note"])
    print(f"wrote {path} ({len(items)} answers, {passed} pass the automatic check, {noted} noted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
