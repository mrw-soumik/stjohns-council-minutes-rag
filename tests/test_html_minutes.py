from council_rag.ingest.html_minutes import (
    Minutes,
    block_text,
    clean_text,
    load_minutes,
    save_minutes,
)


def test_header_and_attendance(synthetic_minutes: Minutes) -> None:
    m = synthetic_minutes
    assert m.title == "Minutes of Regular Meeting - City Council"
    assert m.location == "Council Chamber, 4th Floor, City Hall"
    assert m.time_text.startswith("Monday, March 4, 2024 at 3:00")
    assert m.attendance["Present"][0] == "Mayor Alex Example"
    assert m.attendance["Regrets"] == ["Councillor Eli Invented"]
    assert "Staff: Fay Test, City Clerk" in m.attendance_text()
    # the standard land acknowledgement is deliberately not part of the record we index
    assert "standard statement" not in m.attendance_text()


def test_items_hierarchy_refs_and_anchors(synthetic_minutes: Minutes) -> None:
    items = {i.ref: i for i in synthetic_minutes.items}
    assert list(items) == ["1", "2", "2.1", "3", "3.1", "4", "4.1", "4.1.1", "5", "5.1", "6"]
    nested = items["4.1.1"]  # shown as "1." under 4.1; the full reference is derived
    assert nested.counter == "1"
    assert nested.level == 3
    assert nested.path == [
        "4 COMMITTEE REPORTS",
        "4.1 Committee of the Whole Report - February 21, 2024",
    ]
    assert nested.anchor == "AgendaItemAgendaItem10TitleHeader"
    assert items["2"].blocks == []  # heading-only section


def test_motions_votes_and_results(synthetic_minutes: Minutes) -> None:
    items = {i.ref: i for i in synthetic_minutes.items}
    motion = items["3.1"].blocks[0].motion
    assert motion is not None
    assert motion.number == "SJMC-R-2024-03-04/002"
    assert motion.moved_by == "Deputy Mayor Sample"
    assert motion.result == "MOTION LOST (1 to 3)"
    assert [v.label for v in motion.votes] == ["For (1)", "Against (3)"]
    assert (
        motion.votes[1].names == "Mayor Example, Councillor Placeholder, and Councillor Fictional"
    )
    rendered = items["4.1.1"].render()
    assert rendered.startswith("4.1.1 Snow Clearing By-Law Amendment\n")
    assert "That Council adopt By-Law No. 9999" in rendered
    assert "Abstain (1): Councillor Placeholder" in rendered
    assert [b.kind for b in items["4.1.1"].blocks] == ["notes", "motion", "motion"]


def test_attachments_are_excluded(synthetic_minutes: Minutes) -> None:
    text = "\n".join(i.render() for i in synthetic_minutes.items)
    assert ".pdf" not in text


def test_block_text_joins_inline_spans_without_spaces(synthetic_html: str) -> None:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(synthetic_html, "html.parser")
    assert "3:00 P.m." in block_text(soup.select_one(".AgendaMeetingTime"))
    assert clean_text("a\u200bb\u00a0 c\n\n d ") == "ab c\nd"


def test_json_round_trip(synthetic_minutes: Minutes, tmp_path) -> None:
    path = save_minutes(synthetic_minutes, tmp_path)
    again = load_minutes(path)
    assert again == synthetic_minutes
