"""Structure-aware chunking of parsed minutes.

Design (documented in the README):

* The unit of citation is the agenda item. Every chunk belongs to exactly one item
  and carries its reference ("8.1.2"), title, ancestor path and the URL fragment of
  the item on the official minutes page.
* Small items become one chunk each. Items longer than ``max_words`` are split at
  block boundaries (discussion notes, each motion), so a motion's number, mover,
  text, votes and result stay together. A single block that is still too long is
  split at line, then sentence boundaries.
* Heading-only items ("6 DEVELOPMENT APPLICATIONS") produce no chunk; their titles
  live on in the ``path`` of their children.
* One extra chunk per meeting holds the attendance (present, regrets, staff).
* ``text`` is the citable text, taken verbatim from the parsed item. ``context``
  (meeting, date, item path) is added only to the text that is indexed, never to the
  text that is quoted.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from council_rag.ingest.html_minutes import AgendaItem, Block, Minutes

_SENTENCE_RE = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z0-9(\u201c\"])")
_DECISION_RE = re.compile(r"\bSJMC-R-\d{4}-\d{2}-\d{2}/\d+\b")
_RESULT_RE = re.compile(r"^(MOTION (CARRIED|LOST|DEFEATED)[^\n]*|WITHDRAWN|CARRIED[^\n]*)$", re.M)


@dataclass(frozen=True)
class ChunkingConfig:
    max_words: int = 250
    include_attendance: bool = True
    include_notes: bool = True


@dataclass
class Chunk:
    chunk_id: str
    meeting_id: str
    meeting_name: str
    meeting_date: str
    item_ref: str
    item_title: str
    item_anchor: str
    item_path: list[str]
    kind: str  # "item" | "attendance"
    part: int  # 1-based part number within the item
    parts: int  # number of parts the item was split into
    text: str
    url: str | None
    pdf_url: str | None
    pdf_document_id: int | None
    decisions: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    @property
    def locator(self) -> str:
        """Human-readable location, e.g. "item 8.1.2 (Downtown Pedestrian Mall ...)"."""
        if self.kind == "attendance":
            return "attendance"
        return f"item {self.item_ref}".strip()

    def context_header(self) -> str:
        """Meeting and item path, used for indexing and as non-citable context."""
        parts = [f"{self.meeting_name}, {long_date(self.meeting_date)}"]
        parts.extend(self.item_path)
        return " > ".join(parts)

    def index_text(self, with_context: bool = True) -> str:
        return f"{self.context_header()}\n{self.text}" if with_context else self.text

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Chunk:
        return cls(**data)


def long_date(iso_date: str) -> str:
    """'2023-02-20' -> 'February 20, 2023' (portable, no platform-specific strftime flags)."""
    d = date.fromisoformat(iso_date)
    return f"{d:%B} {d.day}, {d.year}"


def _word_count(text: str) -> int:
    return len(text.split())


def _split_long_text(text: str, max_words: int) -> list[str]:
    """Split one oversized block at line, then sentence boundaries."""
    pieces: list[str] = []
    for line in text.split("\n"):
        if _word_count(line) <= max_words:
            pieces.append(line)
            continue
        sentences = _SENTENCE_RE.split(line)
        current: list[str] = []
        for sentence in sentences:
            if current and _word_count(" ".join([*current, sentence])) > max_words:
                pieces.append(" ".join(current))
                current = []
            current.append(sentence)
        if current:
            pieces.append(" ".join(current))
    # re-pack consecutive small pieces up to the limit
    packed: list[str] = []
    for piece in pieces:
        if packed and _word_count(packed[-1]) + _word_count(piece) <= max_words:
            packed[-1] = f"{packed[-1]}\n{piece}"
        else:
            packed.append(piece)
    return packed


def _group_blocks(item: AgendaItem, blocks: list[Block], max_words: int) -> list[str]:
    """Greedy grouping of an item's blocks into texts of at most ``max_words`` words."""
    heading_words = _word_count(item.heading)
    budget = max(max_words - heading_words, 40)
    units: list[str] = []
    for block in blocks:
        if _word_count(block.text) <= budget:
            units.append(block.text)
        else:
            units.extend(_split_long_text(block.text, budget))
    groups: list[list[str]] = []
    for unit in units:
        if groups and _word_count("\n".join([*groups[-1], unit])) <= budget:
            groups[-1].append(unit)
        else:
            groups.append([unit])
    return ["\n".join([item.heading, *group]) for group in groups]


def chunk_minutes(minutes: Minutes, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Turn one meeting's parsed minutes into chunks."""
    config = config or ChunkingConfig()
    chunks: list[Chunk] = []
    prefix = f"{minutes.meeting_date}_{minutes.meeting_id[:8]}"

    def make(item: AgendaItem | None, text: str, kind: str, part: int, parts: int) -> Chunk:
        anchor = item.anchor if item else ""
        url = minutes.source_url
        if url and anchor:
            url = f"{url}#{anchor}"
        return Chunk(
            chunk_id=f"{prefix}_{len(chunks):03d}",
            meeting_id=minutes.meeting_id,
            meeting_name=minutes.meeting_name,
            meeting_date=minutes.meeting_date,
            item_ref=item.ref if item else "",
            item_title=item.title if item else "Attendance",
            item_anchor=anchor,
            item_path=list(item.path) if item else [],
            kind=kind,
            part=part,
            parts=parts,
            text=text,
            url=url,
            pdf_url=minutes.pdf_url,
            pdf_document_id=minutes.pdf_document_id,
            decisions=_DECISION_RE.findall(text),
            results=[m.group(0) for m in _RESULT_RE.finditer(text)],
        )

    if config.include_attendance and minutes.attendance:
        chunks.append(make(None, minutes.attendance_text(), "attendance", 1, 1))

    for item in minutes.items:
        blocks = [b for b in item.blocks if b.text and (config.include_notes or b.kind != "notes")]
        if not blocks:
            continue
        texts = _group_blocks(item, blocks, config.max_words)
        for i, text in enumerate(texts, start=1):
            chunks.append(make(item, text, "item", i, len(texts)))
    return chunks


# --- published excerpts -----------------------------------------------------------------
#
# The browser demo and the API show, for every chunk, the agenda item title, its section
# path and a link to the item in the official minutes. Only decisions are quoted: the
# opening words of the motion and its recorded outcome. Discussion notes, attendance and
# routine items are shown by title and link only. This keeps the published text to a
# small share of the minutes (measured by scripts/published_text_share.py).

EXCERPT_MAX_CHARS = 80  # hard cap for any published excerpt (PA-13)

_ROUTINE_TITLE_RE = re.compile(
    r"^\s*(?:(?:adoption|approval) of (?:the )?(?:agenda|minutes)|minutes of\b"
    r"|call to order|adjournment|weekly payment vouchers?)",
    re.IGNORECASE,
)
_VOTE_OR_RESULT_RE = re.compile(
    r"^(?:(?:For|Against|Abstain(?:ed)?|Absent|Recused)\s*\(\d+\)"
    r"|MOTION\b|WITHDRAWN\b|CARRIED\b|DEFEATED\b|LOST\b)"
)
_OUTCOME_RE = re.compile(r"^(?:MOTION )?(CARRIED|LOST|DEFEATED|WITHDRAWN)(?: \((\d+) to (\d+)\))?")
_SENTENCE_END_RE = re.compile(r"[.!?;](?=\s)")
_ABBREVIATIONS = frozenset(
    {
        "approx", "ave", "blvd", "co", "corp", "cres", "dept", "dr", "etc", "hwy", "inc", "jr",
        "ltd", "mr", "mrs", "ms", "mt", "no", "nos", "pl", "rd", "sr", "st", "vs",
    }
)  # fmt: skip
_OPENERS = '(\u201c"'
_TRAILING = ' \n,;:(\u2013-\u201c"'


@dataclass(frozen=True)
class Excerpt:
    """What is published for a chunk besides its title, path and link.

    ``text`` is an exact substring of the chunk text, or "" when nothing is quoted.
    ``cut`` is true when the motion's wording continues after ``text`` (the UI adds an
    ellipsis). ``outcome`` is the recorded result in plain words ("carried, 9 to 0").
    """

    text: str
    cut: bool = False
    outcome: str | None = None


def is_routine(chunk: Chunk) -> bool:
    """Agenda and minutes adoption, payment vouchers, call to order and adjournment."""
    return bool(_ROUTINE_TITLE_RE.match(chunk.item_title))


def outcome_label(result_line: str) -> str | None:
    """'MOTION CARRIED (9 to 0)' -> 'carried, 9 to 0'; 'WITHDRAWN' -> 'withdrawn'."""
    m = _OUTCOME_RE.match(result_line)
    if not m:
        return None
    word = m.group(1).lower()
    return f"{word}, {m.group(2)} to {m.group(3)}" if m.group(2) else word


def first_sentence_end(text: str) -> int:
    """Length of the first sentence of ``text``, ending at a full stop or a semicolon.

    A full stop after an abbreviation or an initial ("St. John's", "No. 5", "p.m.") does
    not end the sentence.
    """
    for m in _SENTENCE_END_RE.finditer(text):
        end = m.start()
        if text[end] == ";":
            return end
        rest = text[end + 1 :].lstrip()
        if not rest or not (rest[0].isupper() or rest[0] in _OPENERS):
            continue
        before = text[:end].split()
        word = before[-1].lower().lstrip(_OPENERS) if before else ""
        if len(word) <= 1 or "." in word or word in _ABBREVIATIONS:
            continue
        return end + 1
    return len(text)


def _clip(text: str, limit: int) -> str:
    """Prefix of ``text`` of at most ``limit`` characters, ending at a word boundary."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    if not text[limit].isspace():
        space = max(head.rfind(" "), head.rfind("\n"))
        if space > limit // 2:
            head = head[:space]
    return head.rstrip(_TRAILING)


def excerpt_for(chunk: Chunk, max_chars: int = EXCERPT_MAX_CHARS) -> Excerpt:
    """Choose what is quoted for a chunk (FR-05, PA-13).

    Decisions are quoted by the opening of their first motion: up to the end of its first
    sentence and at most ``max_chars`` characters, cut at a word boundary, with the
    recorded outcome alongside. The wording may run over several lines ("That Council:"
    followed by a numbered list). Vote lists, discussion notes, attendance and routine
    items are never quoted.
    """
    if chunk.kind != "item" or is_routine(chunk):
        return Excerpt("")
    text = chunk.text
    lines = text.split("\n")
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    # line 0 is the item heading; the motion follows "Seconded By", else the first "That "
    start = next(
        (i + 1 for i in range(1, len(lines) - 1) if lines[i].startswith("Seconded By")), None
    )
    if start is None:
        start = next((i for i in range(1, len(lines)) if lines[i].startswith("That ")), None)
    if start is None:
        return Excerpt("")
    stop = next(
        (i for i in range(start + 1, len(lines)) if _VOTE_OR_RESULT_RE.match(lines[i])),
        len(lines),
    )
    wording = text[offsets[start] : offsets[stop - 1] + len(lines[stop - 1])].rstrip()
    span = _clip(wording[: first_sentence_end(wording)], max_chars)
    outcome = None
    for line in lines[stop:]:
        if line.startswith("Moved By") or _DECISION_RE.fullmatch(line):
            break  # the next motion
        if _RESULT_RE.match(line):
            outcome = outcome_label(line)
            break
    return Excerpt(span, len(span) < len(wording.rstrip(" .")), outcome)


def make_excerpt(chunk: Chunk, max_chars: int = EXCERPT_MAX_CHARS) -> str:
    """The verbatim excerpt text only ("" when the chunk is not quoted)."""
    return excerpt_for(chunk, max_chars).text
