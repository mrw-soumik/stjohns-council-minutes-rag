from council_rag.answers import Answer, citation_for
from council_rag.chunking import chunk_minutes
from council_rag.evaluation.answer_eval import facts_present, grade, summarise
from council_rag.evaluation.dataset import EvalQuestion, GoldRef
from tests.conftest import MEETING_ID


def _question(**kw) -> EvalQuestion:
    gold = GoldRef(MEETING_ID, "2024-03-04", "3.1", ["MOTION LOST (1 to 3)"])
    base = {
        "id": "q",
        "question": "?",
        "type": "vote_outcome",
        "answerable": True,
        "gold": [gold],
        "answer_facts": [["1 to 3"], ["defeated", "lost"]],
    }
    return EvalQuestion(**{**base, **kw})


def test_facts_present_needs_every_group() -> None:
    facts = [["1 to 3"], ["defeated", "lost"]]
    assert facts_present("It was LOST, 1 to 3.", facts)
    assert not facts_present("It was lost.", facts)
    assert facts_present("anything", [])


def test_grade_answerable(synthetic_minutes) -> None:
    passages = chunk_minutes(synthetic_minutes)
    gold_chunk = next(c for c in passages if c.item_ref == "3.1")
    other = next(c for c in passages if c.item_ref == "2.1")
    good = Answer(
        "answered",
        "Defeated 1 to 3 on March 4, 2024.",
        citations=[citation_for(gold_chunk, 0), citation_for(other, 1)],
    )
    g = grade(_question(), good, passages)
    assert g["correct"] and g["cites_gold"] and g["facts_ok"]
    assert (g["n_citations"], g["n_supported"]) == (2, 1)
    wrong_cite = Answer("answered", "Defeated 1 to 3.", citations=[citation_for(other, 1)])
    assert not grade(_question(), wrong_cite, passages)["correct"]
    declined = grade(_question(), Answer("not_found", "I could not find this."), passages)
    assert not declined["answered"] and not declined["correct"]


def test_grade_unanswerable_and_summary(synthetic_minutes) -> None:
    passages = chunk_minutes(synthetic_minutes)
    q = _question(answerable=False, type="unanswerable", gold=[], answer_facts=[])
    declined = grade(q, Answer("not_found", ""), passages)
    answered = grade(
        q, Answer("answered", "Yes.", citations=[citation_for(passages[1], 0)]), passages
    )
    assert declined["declined"] and not answered["declined"]
    rows = [
        {
            "id": "u1",
            "type": "unanswerable",
            "answerable": False,
            "status": "not_found",
            "grade": declined,
            "latency_s": 2.0,
            "input_tokens": 900,
            "output_tokens": 20,
        },
        {
            "id": "u2",
            "type": "unanswerable",
            "answerable": False,
            "status": "answered",
            "grade": answered,
            "latency_s": 4.0,
            "input_tokens": 1100,
            "output_tokens": 40,
        },
    ]
    s = summarise(rows)
    assert s["unanswerable"]["declined"]["rate"] == 0.5
    assert s["latency_s"]["p95"] == 4.0 and s["tokens"]["mean_input"] == 1000


def test_pa11_citation_link_check(synthetic_minutes, tmp_path) -> None:
    from council_rag.evaluation.answer_eval import check_citation_links

    meeting_id = "0123abcd-0000-4000-8000-00000000abcd"
    url = (
        f"https://pub-stjohns.escribemeetings.com/Meeting.aspx?Id={meeting_id}"
        "&Agenda=PostMinutes&lang=English#AgendaItemAgendaItem8TitleHeader"
    )
    chunk = next(c for c in chunk_minutes(synthetic_minutes) if c.item_ref == "3.1")
    chunk = type(chunk).from_json({**chunk.to_json(), "meeting_id": meeting_id, "url": url})
    (tmp_path / f"{meeting_id}.html").write_text("<h3 Id='AgendaItemAgendaItem8TitleHeader'>")
    cite = {"chunk_id": chunk.chunk_id, "meeting_date": "2024-03-04", "item_ref": "3.1", "url": url}
    good = {"id": "a", "status": "answered", "citations": [cite]}
    result = check_citation_links([good], {chunk.chunk_id: chunk}, tmp_path)
    assert result["pass"] and result["citations"] == 1
    bad_url = {
        "id": "b",
        "status": "answered",
        "citations": [{**cite, "url": "https://example.com/x"}],
    }
    uncited = {"id": "c", "status": "answered", "citations": []}
    declined = {"id": "d", "status": "not_found", "citations": []}
    result = check_citation_links([bad_url, uncited, declined], {chunk.chunk_id: chunk}, tmp_path)
    assert not result["pass"] and len(result["problems"]) == 2 and result["answered"] == 2


def test_pa12_date_filter_check() -> None:
    from council_rag.evaluation.answer_eval import check_date_filters

    rows = [
        {
            "id": "a",
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
            "passages": ["2024-06-11_7157bbbc_010"],
        },
        {
            "id": "b",
            "date_from": "2024-01-01",
            "date_to": None,
            "passages": ["2023-06-12_7a4fcfc9_020"],
        },
        {"id": "c", "passages": ["2023-01-09_5dbf241d_001"]},
    ]
    result = check_date_filters(rows)
    assert result["filtered_questions"] == 2 and result["violations"] == [
        "b: 2023-06-12_7a4fcfc9_020"
    ]
