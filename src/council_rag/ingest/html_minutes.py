"""Parse eSCRIBE "Minutes (HTML)" pages into a structured record.

The rendered minutes follow a stable template:

* ``header.AgendaHeader`` holds the title, date/time, location and attendance lists
  (``Present``, ``Regrets``, ``Staff``, ``Others``),
* every agenda item is a ``div.AgendaItem`` whose heading (``h2`` to ``h5``) carries
  an ``id`` usable as a URL fragment, an ``AgendaItemCounter`` ("6.1") and an
  ``AgendaItemTitle``; child items follow as siblings, so document order plus heading
  level gives the hierarchy,
* item content sits in ``AgendaItemContentRow`` blocks: ``AgendaItemDescription``,
  ``AgendaItemMinutes`` (discussion notes) and ``AgendaItemMotion`` entries with the
  decision number, mover, seconder, motion text, recorded votes and result.

Attachment lists (file names of staff reports) and the standard land acknowledgement
are left out: the first are links to other documents, the second is identical in
every meeting.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

_BLOCK_TAGS = {
    "p",
    "div",
    "li",
    "ul",
    "ol",
    "tr",
    "table",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "section",
    "article",
    "header",
    "blockquote",
    "pre",
}
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u2060"), None)
_HEADING_RE = re.compile(r"^h([1-6])$")


def clean_text(text: str) -> str:
    """Normalise whitespace and invisible characters while keeping the wording."""
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH).replace("\xa0", " ")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def block_text(node: Tag | NavigableString | None) -> str:
    """Text of an element with line breaks at block boundaries and none inside inline runs.

    eSCRIBE splits words across ``span`` elements ("3", ":00", " P", "."), so plain
    ``get_text(" ")`` would insert spurious spaces.
    """
    if node is None:
        return ""
    parts: list[str] = []

    def walk(el: Tag | NavigableString) -> None:
        if isinstance(el, NavigableString):
            if el.__class__.__name__ in {"Comment", "Doctype", "ProcessingInstruction"}:
                return
            parts.append(str(el))
            return
        if el.name in {"script", "style"}:
            return
        if el.name == "br":
            parts.append("\n")
            return
        is_block = el.name in _BLOCK_TAGS
        if is_block:
            parts.append("\n")
        for child in el.children:
            walk(child)
        if el.name == "td":
            parts.append(" ")
        if is_block:
            parts.append("\n")

    walk(node)
    return clean_text("".join(parts))


@dataclass
class Vote:
    label: str  # e.g. "For (11)", "Against (2)", "Abstain (1)"
    names: str


@dataclass
class Motion:
    number: str | None
    moved_by: str | None
    seconded_by: str | None
    pre_text: str
    text: str
    votes: list[Vote]
    result: str | None
    post_text: str

    def render(self) -> str:
        lines: list[str] = []
        if self.pre_text:
            lines.append(self.pre_text)
        if self.number:
            lines.append(self.number)
        if self.moved_by:
            lines.append(f"Moved By {self.moved_by}")
        if self.seconded_by:
            lines.append(f"Seconded By {self.seconded_by}")
        if self.text:
            lines.append(self.text)
        lines.extend(f"{v.label}: {v.names}" if v.names else v.label for v in self.votes)
        if self.result:
            lines.append(self.result)
        if self.post_text:
            lines.append(self.post_text)
        return "\n".join(lines)


@dataclass
class Block:
    """One piece of item content, in document order."""

    kind: str  # "description" | "notes" | "motion"
    text: str
    motion: Motion | None = None


@dataclass
class AgendaItem:
    anchor: str  # heading id, usable as "#<anchor>" on the official page
    counter: str  # as displayed, e.g. "6.1" or "1"
    ref: str  # full reference, e.g. "9.1.1" for a third-level item shown as "1."
    title: str
    level: int  # 1 for top-level sections
    path: list[str] = field(default_factory=list)  # titles of ancestor items
    blocks: list[Block] = field(default_factory=list)

    @property
    def heading(self) -> str:
        return f"{self.ref} {self.title}".strip()

    def render(self) -> str:
        """Citable text of the item: heading line followed by its blocks."""
        return "\n".join([self.heading, *[b.text for b in self.blocks if b.text]])


@dataclass
class Minutes:
    meeting_id: str
    meeting_name: str
    meeting_date: str
    title: str
    time_text: str
    location: str
    attendance: dict[str, list[str]]
    items: list[AgendaItem]
    source_url: str | None = None
    pdf_url: str | None = None
    pdf_document_id: int | None = None

    def attendance_text(self) -> str:
        lines = [self.title, self.time_text, self.location]
        for label, names in self.attendance.items():
            if names:
                lines.append(f"{label}: " + "; ".join(names))
        return "\n".join(line for line in lines if line)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Minutes:
        items = []
        for raw in data["items"]:
            blocks = []
            for b in raw["blocks"]:
                motion = None
                if b.get("motion"):
                    m = dict(b["motion"])
                    m["votes"] = [Vote(**v) for v in m["votes"]]
                    motion = Motion(**m)
                blocks.append(Block(kind=b["kind"], text=b["text"], motion=motion))
            items.append(AgendaItem(**{**raw, "blocks": blocks}))
        return cls(**{**data, "items": items})


def _value(el: Tag | None, selector: str) -> str | None:
    if el is None:
        return None
    found = el.select_one(selector)
    text = block_text(found) if found else ""
    return text or None


def _parse_motion(li: Tag) -> Motion:
    votes: list[Vote] = []
    for row in li.select("table.MotionVoters tr"):
        label_cell = row.select_one("td.VoterVote")
        if label_cell is None:
            continue
        names_cell = row.select_one("td.VotesUsers")
        names = block_text(names_cell).replace(",  and", ", and") if names_cell else ""
        votes.append(Vote(label=block_text(label_cell), names=names))
    return Motion(
        number=_value(li, ".Number .Value"),
        moved_by=_value(li, ".MovedBy .Value"),
        seconded_by=_value(li, ".SecondedBy .Value"),
        pre_text=_value(li, ".PreMotionText") or "",
        text=_value(li, ".MotionText") or "",
        votes=votes,
        result=_value(li, ".MotionResult"),
        post_text=_value(li, ".PostMotionText") or "",
    )


def _item_blocks(item: Tag) -> list[Block]:
    blocks: list[Block] = []
    for row in item.find_all("div", class_="AgendaItemContentRow", recursive=False):
        for el in row.find_all(True, recursive=False):
            classes = set(el.get("class") or [])
            if "AgendaItemMotions" in classes:
                for li in el.find_all("li", class_="AgendaItemMotion", recursive=False):
                    motion = _parse_motion(li)
                    blocks.append(Block(kind="motion", text=motion.render(), motion=motion))
            elif classes & {"AgendaItemAttachmentsList", "visible-xs", "visible-sm"}:
                continue
            elif "AgendaItemMinutes" in classes:
                text = block_text(el)
                if text:
                    blocks.append(Block(kind="notes", text=text))
            else:
                text = block_text(el)
                if text:
                    blocks.append(Block(kind="description", text=text))
    return blocks


def _attendance(header: Tag | None) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if header is None:
        return result
    table = header.select_one(".AgendaHeaderAttendanceTable")
    if table is None:
        return result
    for group in table.find_all("div", recursive=False):
        label_el = group.select_one(".Label")
        value_el = group.select_one(".Value")
        if label_el is None or value_el is None:
            continue
        label = block_text(label_el).rstrip(":").strip()
        names = [block_text(li) for li in value_el.select("li")]
        if not names:
            names = [line for line in block_text(value_el).split("\n") if line]
        result[label] = [n for n in names if n]
    return result


def parse_minutes_html(
    page: str,
    meeting_id: str,
    meeting_name: str,
    meeting_date: str,
    source_url: str | None = None,
    pdf_url: str | None = None,
    pdf_document_id: int | None = None,
) -> Minutes:
    """Parse one rendered minutes page."""
    soup = BeautifulSoup(page, "html.parser")
    header = soup.select_one("header.AgendaHeader")
    title = _value(header, ".AgendaHeaderTitle") or ""
    subtitle = _value(header, ".AgendaHeaderSubTitle") or ""
    time_text = (_value(header, ".AgendaMeetingTime") or "").replace("\n", " ")
    time_text = re.sub(r"\s+", " ", time_text).replace(" ,", ",")
    location = subtitle or (_value(header, ".AgendaMeetingLocation .Location") or "")

    items: list[AgendaItem] = []
    stack: list[AgendaItem] = []  # ancestors of the current item
    for div in soup.select("div.AgendaItems div.AgendaItem"):
        title_row = div.find("div", class_="AgendaItemTitleRow", recursive=False)
        heading = title_row.find(_HEADING_RE) if title_row else None
        if heading is None:
            continue
        level = int(heading.name[1]) - 1
        counter = (_value(heading, ".AgendaItemCounter") or "").rstrip(".").strip()
        item_title = _value(heading, ".AgendaItemTitle") or ""
        while stack and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1] if stack else None
        if parent and parent.ref and counter and "." not in counter and level >= 2:
            ref = f"{parent.ref}.{counter}"
        else:
            ref = counter
        item = AgendaItem(
            anchor=str(heading.get("id") or ""),
            counter=counter,
            ref=ref,
            title=item_title,
            level=level,
            path=[a.heading for a in stack],
            blocks=_item_blocks(div),
        )
        items.append(item)
        stack.append(item)

    return Minutes(
        meeting_id=meeting_id,
        meeting_name=meeting_name,
        meeting_date=meeting_date,
        title=title,
        time_text=time_text,
        location=location,
        attendance=_attendance(header),
        items=items,
        source_url=source_url,
        pdf_url=pdf_url,
        pdf_document_id=pdf_document_id,
    )


def save_minutes(minutes: Minutes, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{minutes.meeting_id}.json"
    path.write_text(json.dumps(minutes.to_json(), indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_minutes(path: Path) -> Minutes:
    return Minutes.from_json(json.loads(path.read_text(encoding="utf-8")))
