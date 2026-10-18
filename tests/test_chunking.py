from council_rag.chunking import (
    EXCERPT_MAX_CHARS,
    Chunk,
    ChunkingConfig,
    Excerpt,
    chunk_minutes,
    excerpt_for,
    first_sentence_end,
    long_date,
    make_excerpt,
    outcome_label,
)
from council_rag.ingest.html_minutes import Minutes


def test_one_chunk_per_item_plus_attendance(synthetic_minutes: Minutes) -> None:
    chunks = chunk_minutes(synthetic_minutes)
    assert chunks[0].kind == "attendance"
    refs = [c.item_ref for c in chunks if c.kind == "item"]
    # heading-only sections (2, 3, 4, 4.1, 5) produce no chunk of their own
    assert refs == ["1", "2.1", "3.1", "4.1.1", "5.1", "6"]
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks[1].chunk_id == "2024-03-04_00000000_001"


def test_chunk_metadata_and_links(synthetic_minutes: Minutes) -> None:
    chunk = next(c for c in chunk_minutes(synthetic_minutes) if c.item_ref == "4.1.1")
    assert chunk.url and chunk.url.endswith("#AgendaItemAgendaItem10TitleHeader")
    assert chunk.decisions == ["SJMC-R-2024-03-04/003", "SJMC-R-2024-03-04/004"]
    assert chunk.results == ["MOTION CARRIED (4 to 0)", "MOTION CARRIED (3 to 0)"]
    assert chunk.locator == "item 4.1.1"
    header = chunk.context_header()
    assert header.startswith("Regular Council Meeting, March 4, 2024 > 4 COMMITTEE REPORTS")
    assert chunk.index_text().startswith(header)
    assert chunk.index_text(with_context=False) == chunk.text


def test_long_item_is_split_at_block_boundaries(synthetic_minutes: Minutes) -> None:
    chunks = chunk_minutes(synthetic_minutes, ChunkingConfig(max_words=45))
    parts = [c for c in chunks if c.item_ref == "4.1.1"]
    assert len(parts) >= 2
    assert all(p.text.startswith("4.1.1 Snow Clearing By-Law Amendment") for p in parts)
    # a motion is never split: number, text, votes and result stay in one chunk
    for p in parts:
        if "SJMC-R-2024-03-04/003" in p.text:
            assert "By-Law No. 9999" in p.text and "MOTION CARRIED (4 to 0)" in p.text
    assert {p.parts for p in parts} == {len(parts)}
    assert [p.part for p in parts] == list(range(1, len(parts) + 1))


def test_oversized_block_is_split_by_sentences(synthetic_minutes: Minutes) -> None:
    chunks = chunk_minutes(synthetic_minutes, ChunkingConfig(max_words=40))
    parts = [c for c in chunks if c.item_ref == "5.1"]
    assert len(parts) >= 3
    assert all(c.n_words <= 40 + 8 for c in parts)  # heading words are added on top
    joined = " ".join(p.text for p in parts)
    assert "Twelve plots were added" in joined and "separate budget request" in joined


def test_options_drop_notes_and_attendance(synthetic_minutes: Minutes) -> None:
    chunks = chunk_minutes(
        synthetic_minutes, ChunkingConfig(include_attendance=False, include_notes=False)
    )
    assert all(c.kind == "item" for c in chunks)
    assert [c.item_ref for c in chunks] == ["2.1", "3.1", "4.1.1", "5.1"]


def test_decision_excerpt_is_the_opening_of_the_motion(synthetic_minutes: Minutes) -> None:
    chunk = next(c for c in chunk_minutes(synthetic_minutes) if c.item_ref == "3.1")
    ex = excerpt_for(chunk)
    assert ex.text in chunk.text  # PA-13: verbatim
    assert ex.text == (
        "That Council approve the Discretionary Use application for a Home Occupation at"
    )
    assert ex.cut and len(ex.text) <= EXCERPT_MAX_CHARS
    assert ex.outcome == "lost, 1 to 3"  # recorded result, in plain words


def test_excerpt_skips_notes_and_vote_lists(synthetic_minutes: Minutes) -> None:
    chunk = next(c for c in chunk_minutes(synthetic_minutes) if c.item_ref == "4.1.1")
    ex = excerpt_for(chunk)
    # "No." does not end the sentence; the cut falls at a word boundary within the cap
    assert ex.text == (
        "That Council adopt By-Law No. 9999, the Snow Clearing (Amendment No. 2, 2024)"
    )
    assert ex.cut and ex.outcome == "carried, 4 to 0"
    assert "Members of Council" not in ex.text and "For (4)" not in ex.text


def test_notes_attendance_and_routine_items_are_not_quoted(synthetic_minutes: Minutes) -> None:
    by_ref = {c.item_ref: c for c in chunk_minutes(synthetic_minutes)}
    # attendance, call to order, adoption of agenda, discussion notes, adjournment
    for ref in ("", "1", "2.1", "5.1", "6"):
        assert excerpt_for(by_ref[ref]) == Excerpt("")
        assert make_excerpt(by_ref[ref]) == ""


def test_motion_wording_over_several_lines(synthetic_minutes: Minutes) -> None:
    text = (
        "9.1 Heritage Designation\nSJMC-R-2024-03-04/010\nMoved By Councillor Placeholder\n"
        "Seconded By Councillor Fictional\nThat Council:\n"
        "(1) direct staff to prepare a designation by-law; and\n(2) thank the owners.\n"
        "For (4): Mayor Example\nMOTION CARRIED (4 to 0)"
    )
    chunk = Chunk.from_json({**chunk_minutes(synthetic_minutes)[3].to_json(), "text": text})
    ex = excerpt_for(chunk)
    assert ex.text == "That Council:\n(1) direct staff to prepare a designation by-law"
    assert ex.cut and ex.outcome == "carried, 4 to 0"


def test_first_sentence_end_ignores_abbreviations() -> None:
    cases = {
        "That Council thank No. 5 Squadron. Further, that staff report back.": (
            "That Council thank No. 5 Squadron."
        ),
        "That Council approve X; and that Y.": "That Council approve X",
        "That Council approve the plan for St. John's Harbour. The fee is waived.": (
            "That Council approve the plan for St. John's Harbour."
        ),
        "That Council meet at 4:00 p.m. Monday. Then adjourn.": (
            "That Council meet at 4:00 p.m. Monday."
        ),
    }
    for text, first in cases.items():
        assert text[: first_sentence_end(text)] == first


def test_outcome_labels() -> None:
    assert outcome_label("MOTION CARRIED (10 to 1)") == "carried, 10 to 1"
    assert outcome_label("MOTION LOST") == "lost"
    assert outcome_label("WITHDRAWN") == "withdrawn"
    assert outcome_label("Abstention noted.") is None


def test_every_excerpt_is_verbatim_and_short(synthetic_minutes: Minutes) -> None:
    for chunk in chunk_minutes(synthetic_minutes):
        ex = excerpt_for(chunk)
        assert len(ex.text) <= EXCERPT_MAX_CHARS and ex.text in chunk.text
        assert make_excerpt(chunk) == ex.text
    long = Chunk.from_json(
        {
            **chunk_minutes(synthetic_minutes)[3].to_json(),
            "text": "heading\nThat " + "word " * 400,
        }
    )
    ex = excerpt_for(long)
    assert ex.cut and 0 < len(ex.text) <= EXCERPT_MAX_CHARS and ex.text in long.text


def test_long_date_is_portable() -> None:
    assert long_date("2024-03-04") == "March 4, 2024"
