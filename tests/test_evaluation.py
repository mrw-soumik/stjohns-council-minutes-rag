from __future__ import annotations

import json

import pytest

from council_rag.chunking import chunk_minutes
from council_rag.evaluation.dataset import (
    EvalQuestion,
    GoldRef,
    build_qrels,
    is_relevant,
    load_questions,
)
from council_rag.evaluation.metrics import (
    hit_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
    wilson_interval,
)
from council_rag.evaluation.verify import check
from tests.conftest import MEETING_ID


def test_rank_metrics() -> None:
    ranked = ["x", "a2", "y", "b1"]
    rel = [{"a1", "a2"}, {"b1"}]
    assert hit_at_k(ranked, rel, 1) == 0.0
    assert hit_at_k(ranked, rel, 2) == 1.0
    assert recall_at_k(ranked, rel, 2) == 0.5
    assert recall_at_k(ranked, rel, 4) == 1.0
    assert reciprocal_rank(ranked, rel) == 0.5
    assert reciprocal_rank(["x", "y"], rel) == 0.0
    assert recall_at_k(ranked, [], 5) == 0.0


def test_percentile_and_wilson() -> None:
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile([5, 1, 3, 2, 4], 95) == 5
    lo, hi = wilson_interval(52, 60)
    assert 0.75 < lo < 0.80 and 0.92 < hi < 0.95


def _gold(ref: str, *evidence: str) -> GoldRef:
    return GoldRef(
        meeting_id=MEETING_ID, meeting_date="2024-03-04", item_ref=ref, evidence=list(evidence)
    )


def test_relevance_levels(synthetic_minutes) -> None:
    chunks = chunk_minutes(synthetic_minutes)
    target = next(c for c in chunks if c.item_ref == "3.1")
    assert is_relevant(target, _gold("3.1", "MOTION LOST (1 to 3)"))
    assert not is_relevant(target, _gold("3.1", "MOTION CARRIED"))
    assert is_relevant(target, _gold("3.1", "MOTION CARRIED"), level="item")
    q = EvalQuestion(
        id="q1",
        question="?",
        type="vote_outcome",
        answerable=True,
        gold=[_gold("3.1", "bicycle repair shop")],
    )
    assert build_qrels([q], chunks) == {"q1": [[target.chunk_id]]}


def test_load_questions_validates(tmp_path) -> None:
    good = {
        "id": "q1",
        "question": "?",
        "type": "single_fact",
        "answerable": True,
        "gold": [
            {"meeting_id": "m", "meeting_date": "2024-01-01", "item_ref": "1", "evidence": ["x"]}
        ],
    }
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(good) + "\n")
    assert load_questions(path)[0].gold[0].item_ref == "1"
    path.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(path)
    path.write_text(json.dumps({**good, "type": "trivia"}) + "\n")
    with pytest.raises(ValueError, match="unknown question type"):
        load_questions(path)
    path.write_text(json.dumps({**good, "gold": []}) + "\n")
    with pytest.raises(ValueError, match="no gold"):
        load_questions(path)


def test_verify_flags_bad_gold_and_present_absent_terms(synthetic_minutes) -> None:
    ok = EvalQuestion(
        id="ok",
        question="?",
        type="number_lookup",
        answerable=True,
        gold=[_gold("4.1.1", "By-Law No. 9999")],
        reference_answer="By-Law 9999.",
    )
    wrong_item = EvalQuestion(
        id="bad1",
        question="?",
        type="single_fact",
        answerable=True,
        gold=[_gold("2.1", "By-Law No. 9999")],
    )
    no_item = EvalQuestion(
        id="bad2", question="?", type="single_fact", answerable=True, gold=[_gold("9.9", "x")]
    )
    leaky = EvalQuestion(
        id="bad3",
        question="?",
        type="unanswerable",
        answerable=False,
        absent_terms=["community garden"],
    )
    clean = EvalQuestion(
        id="ok2", question="?", type="unanswerable", answerable=False, absent_terms=["light rail"]
    )
    out_of_range = EvalQuestion(
        id="bad4",
        question="?",
        type="date_filtered",
        answerable=True,
        gold=[_gold("2.1", "adopted as presented")],
        date_from="2025-01-01",
    )
    problems = check([ok, wrong_item, no_item, leaky, clean, out_of_range], [synthetic_minutes])
    flagged = {p.split(":")[0] for p in problems}
    assert flagged == {"bad1", "bad2", "bad3", "bad4"}
