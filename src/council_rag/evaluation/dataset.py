"""Evaluation questions, gold references and relevance judgements.

Each answerable question names one or more gold references: a meeting, an agenda
item, and short ``evidence`` strings copied verbatim from that item. A retrieved
chunk is *relevant* to a gold reference when it comes from the same meeting and item
and contains every evidence string (case, accent and whitespace-insensitive). This is
the agenda-item analogue of "the correct source page" in NFR-01: the passage the
answer must be read from.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from council_rag.chunking import Chunk
from council_rag.text import contains_normalized

QUESTION_TYPES = (
    "single_fact",
    "number_lookup",
    "vote_outcome",
    "date_filtered",
    "paraphrase",
    "unanswerable",
)


@dataclass
class GoldRef:
    meeting_id: str
    meeting_date: str
    item_ref: str
    evidence: list[str]
    pdf_document_id: int | None = None


@dataclass
class EvalQuestion:
    id: str
    question: str
    type: str
    answerable: bool
    gold: list[GoldRef] = field(default_factory=list)
    reference_answer: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    absent_terms: list[str] = field(default_factory=list)
    # Facts a correct answer must state: every group must match, any string in a group
    # counts (used by the deterministic answer grader in Phase 2).
    answer_facts: list[list[str]] = field(default_factory=list)
    notes: str | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EvalQuestion:
        gold = [GoldRef(**g) for g in data.get("gold", [])]
        return cls(**{**data, "gold": gold})

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        return {k: v for k, v in out.items() if v not in (None, [], "")} | {
            "answerable": self.answerable
        }


def load_questions(path: Path | str) -> list[EvalQuestion]:
    questions = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            q = EvalQuestion.from_json(json.loads(line))
            if q.type not in QUESTION_TYPES:
                raise ValueError(f"line {line_no}: unknown question type {q.type!r}")
            if q.answerable and not q.gold:
                raise ValueError(f"line {line_no}: answerable question {q.id} has no gold")
            questions.append(q)
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return questions


def is_relevant(chunk: Chunk, gold: GoldRef, level: str = "evidence") -> bool:
    if chunk.meeting_id != gold.meeting_id or chunk.item_ref != gold.item_ref:
        return False
    if level == "item":
        return True
    return all(contains_normalized(chunk.text, e) for e in gold.evidence)


def relevant_sets(
    question: EvalQuestion, chunks: Iterable[Chunk], level: str = "evidence"
) -> list[set[str]]:
    """For each gold reference, the ids of chunks judged relevant."""
    chunk_list = list(chunks)
    return [{c.chunk_id for c in chunk_list if is_relevant(c, g, level)} for g in question.gold]


def build_qrels(
    questions: Sequence[EvalQuestion], chunks: Sequence[Chunk], level: str = "evidence"
) -> dict[str, list[list[str]]]:
    """Relevance judgements keyed by question id (committed for CI as ``eval/qrels.json``)."""
    return {
        q.id: [sorted(s) for s in relevant_sets(q, chunks, level)]
        for q in questions
        if q.answerable
    }
