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


EXCERPT_MAX_CHARS = 300


def make_excerpt(chunk: Chunk, max_chars: int = EXCERPT_MAX_CHARS) -> str:
    """A short, quotable excerpt of a chunk (FR-05; the public demo shows nothing longer).

    Motion chunks show the motion wording and its recorded result, which is usually
    the decision a reader is looking for; other chunks show their opening text. The
    item heading is left out because it is displayed separately.
    """
    lines = chunk.text.split("\n")
    body = lines[1:] if chunk.kind == "item" and len(lines) > 1 else lines
    motion = next((line for line in body if line.startswith("That ")), None)
    if motion is not None:
        result = next((line for line in body if _RESULT_RE.match(line)), "")
        text = f"{motion} {result}".strip()
    else:
        text = " ".join(body)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    if " " in cut[-40:]:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:") + "\u2026"
