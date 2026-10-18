from council_rag.chunking import (
    EXCERPT_MAX_CHARS,
    Chunk,
    ChunkingConfig,
    chunk_minutes,
    excerpt_for,
    long_date,
    make_excerpt,
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


def test_excerpt_shows_short_motion_with_votes_verbatim(synthetic_minutes: Minutes) -> None:
    chunk = next(c for c in chunk_minutes(synthetic_minutes) if c.item_ref == "3.1")
    ex = excerpt_for(chunk)
    assert ex.text in chunk.text  # PA-13: verbatim
    assert ex.text.startswith("That Council approve the Discretionary Use application")
    assert ex.text.endswith("MOTION LOST (1 to 3)") or ex.result == "MOTION LOST (1 to 3)"


def test_long_motion_keeps_result_separately(synthetic_minutes: Minutes) -> None:
    chunk = next(c for c in chunk_minutes(synthetic_minutes) if c.item_ref == "4.1.1")
    ex = excerpt_for(chunk, max_chars=120)
    assert ex.text in chunk.text and ex.cut
    assert ex.result == "MOTION CARRIED (4 to 0)"
    assert len(ex.text) + len(ex.result) + 1 <= 120
    assert "For (4)" not in ex.text  # vote lists are not used to fill a cut excerpt


def test_every_excerpt_is_verbatim_and_short(synthetic_minutes: Minutes) -> None:
    for chunk in chunk_minutes(synthetic_minutes):
        ex = excerpt_for(chunk)
        shown = len(ex.text) + (len(ex.result) + 1 if ex.result else 0)
        assert shown <= EXCERPT_MAX_CHARS
        assert ex.text in chunk.text and (ex.result is None or ex.result in chunk.text)
        assert make_excerpt(chunk) == ex.text
    long = Chunk.from_json(
        {
            **chunk_minutes(synthetic_minutes)[0].to_json(),
            "text": "heading\n" + "word " * 400,
            "kind": "item",
        }
    )
    ex = excerpt_for(long)
    assert ex.cut and len(ex.text) <= EXCERPT_MAX_CHARS and ex.text in long.text


def test_long_date_is_portable() -> None:
    assert long_date("2024-03-04") == "March 4, 2024"
