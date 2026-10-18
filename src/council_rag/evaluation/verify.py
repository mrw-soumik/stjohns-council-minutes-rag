"""Check the evaluation set against the parsed minutes.

For every answerable question and gold reference:

* the meeting exists and its date matches,
* the agenda item exists,
* every evidence string occurs in that item's text,
* date-filtered questions have their gold meetings inside the filter range.

For every unanswerable question, none of its ``absent_terms`` occurs anywhere in the
corpus. Reference answers must stay short (they are published).
"""

from __future__ import annotations

import argparse
from collections import Counter

from council_rag.config import data_paths
from council_rag.evaluation.dataset import EvalQuestion, load_questions
from council_rag.index import load_parsed
from council_rag.ingest.html_minutes import Minutes
from council_rag.text import contains_normalized

MAX_REFERENCE_CHARS = 220


def check(questions: list[EvalQuestion], minutes: list[Minutes]) -> list[str]:
    problems: list[str] = []
    by_id = {m.meeting_id: m for m in minutes}
    corpus = [(m, m.attendance_text()) for m in minutes]
    corpus += [(m, item.render()) for m in minutes for item in m.items]

    for q in questions:
        if q.reference_answer and len(q.reference_answer) > MAX_REFERENCE_CHARS:
            problems.append(f"{q.id}: reference answer longer than {MAX_REFERENCE_CHARS} chars")
        if not q.answerable:
            if not q.absent_terms:
                problems.append(f"{q.id}: unanswerable question lists no absent_terms")
            for term in q.absent_terms:
                found = [m.meeting_date for m, text in corpus if contains_normalized(text, term)]
                if found:
                    problems.append(f"{q.id}: absent term {term!r} occurs ({sorted(set(found))})")
            continue
        for g in q.gold:
            m = by_id.get(g.meeting_id)
            if m is None:
                problems.append(f"{q.id}: meeting {g.meeting_id} not in corpus")
                continue
            if m.meeting_date != g.meeting_date:
                problems.append(f"{q.id}: meeting date {g.meeting_date} != {m.meeting_date}")
            if g.pdf_document_id is not None and g.pdf_document_id != m.pdf_document_id:
                problems.append(f"{q.id}: pdf_document_id {g.pdf_document_id} mismatch")
            if g.item_ref == "attendance":
                text = m.attendance_text()
            else:
                item = next((i for i in m.items if i.ref == g.item_ref), None)
                if item is None:
                    problems.append(f"{q.id}: item {g.item_ref} not found on {m.meeting_date}")
                    continue
                text = item.render()
            if not g.evidence:
                problems.append(f"{q.id}: gold {g.item_ref} has no evidence strings")
            for ev in g.evidence:
                if not contains_normalized(text, ev):
                    problems.append(
                        f"{q.id}: evidence {ev!r} not in item {g.item_ref} ({m.meeting_date})"
                    )
            if q.date_from and g.meeting_date < q.date_from:
                problems.append(f"{q.id}: gold meeting before date_from")
            if q.date_to and g.meeting_date > q.date_to:
                problems.append(f"{q.id}: gold meeting after date_to")
    return problems


def main(args: argparse.Namespace) -> int:
    questions = load_questions(args.questions)
    minutes = load_parsed(data_paths(getattr(args, "data_dir", None)))
    problems = check(questions, minutes)
    counts = Counter(q.type for q in questions)
    answerable = sum(1 for q in questions if q.answerable)
    print(
        f"{len(questions)} questions ({answerable} answerable, "
        f"{len(questions) - answerable} unanswerable)"
    )
    for t, n in sorted(counts.items()):
        print(f"  {t}: {n}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("All gold references verified against the parsed minutes.")
    return 0
