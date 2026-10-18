"""Polite downloader for Regular Council Meeting minutes on the City's eSCRIBE portal.

Behaviour (see the project brief):

* at most one request per ``min_interval`` seconds (default 1.1 s),
* a descriptive User-Agent with a contact address,
* every downloaded file is cached under ``data/raw`` and never fetched again,
* calendar listings are cached per year; ``refresh_calendar=True`` re-reads the
  current year so newly published minutes are picked up (FR-07),
* every Regular Council Meeting that is not downloaded is logged with a reason.

Source format. The portal publishes minutes as "Minutes (HTML)" and "Minutes (PDF)".
The PDF endpoint (``FileStream.ashx``) answers automated clients with a Cloudflare
browser challenge (HTTP 403, ``cf-mitigated: challenge``). This project does not try
to get around that control. The HTML minutes on the same portal are served normally
and carry the same structured record (agenda items, motions, votes), so they are the
default source. PDF download stays available as an opt-in that stops at the first
challenge instead of retrying.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx

from council_rag.config import (
    CALENDAR_ENDPOINT,
    ESCRIBE_BASE_URL,
    MEETING_NAME,
    MIN_REQUEST_INTERVAL_S,
    USER_AGENT,
    DataPaths,
)

log = logging.getLogger(__name__)

_DOC_ID_RE = re.compile(r"DocumentId=(\d+)", re.IGNORECASE)
_HTML_MARKER = "AgendaItems"  # present in every rendered minutes page


class ChallengeBlockedError(RuntimeError):
    """The portal answered with a bot challenge instead of the requested document."""


class RateLimiter:
    """Blocks so that consecutive calls to :meth:`wait` are at least ``min_interval`` apart."""

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self.min_interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last = now


@dataclass(frozen=True)
class MeetingRecord:
    """One Regular Council Meeting as listed by the portal."""

    meeting_id: str
    meeting_name: str
    meeting_date: str  # ISO 8601 date (local time of the meeting)
    start: str  # raw ``StartDate`` value, e.g. "2026/01/13 15:00:00"
    meeting_url: str
    minutes_html_url: str | None
    pdf_document_id: int | None
    pdf_url: str | None


@dataclass
class SyncReport:
    """Outcome of one synchronisation run."""

    downloaded: list[MeetingRecord] = field(default_factory=list)
    cached: list[MeetingRecord] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    out_of_scope: list[dict[str, Any]] = field(default_factory=list)
    pdf_downloaded: int = 0
    pdf_blocked: str | None = None
    requests_made: int = 0


def official_pdf_url(document_id: int, base_url: str = ESCRIBE_BASE_URL) -> str:
    return f"{base_url}/FileStream.ashx?DocumentId={document_id}"


def minutes_html_url(meeting_id: str, base_url: str = ESCRIBE_BASE_URL) -> str:
    return f"{base_url}/Meeting.aspx?Id={meeting_id}&Agenda=PostMinutes&lang=English"


def parse_calendar(
    raw_meetings: Iterable[dict[str, Any]],
    base_url: str = ESCRIBE_BASE_URL,
    meeting_name: str = MEETING_NAME,
) -> tuple[list[MeetingRecord], list[dict[str, Any]]]:
    """Select meetings named ``meeting_name`` and locate their minutes documents.

    Returns ``(records, other_meetings)``; ``other_meetings`` lists meetings left out
    of scope (other committees), which the caller can log.
    """
    records: list[MeetingRecord] = []
    others: list[dict[str, Any]] = []
    for meeting in raw_meetings:
        name = html.unescape(str(meeting.get("MeetingName", ""))).strip()
        start = str(meeting.get("StartDate", ""))
        try:
            meeting_date = datetime.strptime(start, "%Y/%m/%d %H:%M:%S").date().isoformat()
        except ValueError:
            meeting_date = start[:10].replace("/", "-")
        if name != meeting_name:
            others.append({"meeting_name": name, "meeting_date": meeting_date})
            continue
        meeting_id = str(meeting.get("ID", ""))
        pdf_id: int | None = None
        has_html = False
        for link in meeting.get("MeetingDocumentLink") or []:
            title = str(link.get("Title", "")).lower()
            if not title.startswith("minutes"):
                continue
            if "html" in title:
                has_html = True
            elif "pdf" in title:
                match = _DOC_ID_RE.search(str(link.get("Url", "")))
                if match:
                    pdf_id = int(match.group(1))
        records.append(
            MeetingRecord(
                meeting_id=meeting_id,
                meeting_name=name,
                meeting_date=meeting_date,
                start=start,
                meeting_url=str(meeting.get("Url", "")),
                minutes_html_url=minutes_html_url(meeting_id, base_url) if has_html else None,
                pdf_document_id=pdf_id,
                pdf_url=official_pdf_url(pdf_id, base_url) if pdf_id else None,
            )
        )
    records.sort(key=lambda r: (r.meeting_date, r.meeting_id))
    return records, others


def year_windows(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Split ``[start, end]`` into calendar-year windows (one listing request each)."""
    for year in range(start.year, end.year + 1):
        yield max(start, date(year, 1, 1)), min(end, date(year, 12, 31))


def _is_challenge(response: httpx.Response) -> bool:
    if response.headers.get("cf-mitigated", "").lower() == "challenge":
        return True
    return response.status_code == 403 and b"Verifying your browser" in response.content[:4000]


class EscribeClient:
    """Minimal, rate-limited HTTP client for the public eSCRIBE endpoints."""

    def __init__(
        self,
        base_url: str = ESCRIBE_BASE_URL,
        user_agent: str = USER_AGENT,
        min_interval: float = MIN_REQUEST_INTERVAL_S,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.base_url = base_url
        self._http = httpx.Client(
            base_url=base_url,
            headers={"User-Agent": user_agent},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )
        self._limiter = limiter or RateLimiter(min_interval)
        self.requests_made = 0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> EscribeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        self._limiter.wait()
        self.requests_made += 1
        response = self._http.request(method, url, **kwargs)
        if _is_challenge(response):
            raise ChallengeBlockedError(
                f"{url} answered with a bot challenge (HTTP {response.status_code}); "
                "not circumvented"
            )
        response.raise_for_status()
        return response

    def calendar(self, start: date, end: date) -> list[dict[str, Any]]:
        response = self._request(
            "POST",
            CALENDAR_ENDPOINT,
            json={"calendarStartDate": start.isoformat(), "calendarEndDate": end.isoformat()},
            headers={"Accept": "application/json"},
        )
        payload = response.json()
        meetings = payload.get("d") if isinstance(payload, dict) else None
        if not isinstance(meetings, list):
            raise ValueError("Unexpected calendar response shape (missing 'd' list)")
        return meetings

    def minutes_html(self, meeting_id: str) -> str:
        response = self._request(
            "GET", f"/Meeting.aspx?Id={meeting_id}&Agenda=PostMinutes&lang=English"
        )
        text = response.text
        if _HTML_MARKER not in text:
            raise ValueError("minutes page has no agenda items (unexpected layout)")
        return text

    def pdf(self, document_id: int) -> bytes:
        response = self._request("GET", f"/FileStream.ashx?DocumentId={document_id}")
        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"response is not a PDF ({response.headers.get('content-type')})")
        return response.content


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _write_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))


def load_calendar(
    client: EscribeClient,
    paths: DataPaths,
    start: date,
    end: date,
    refresh: bool = False,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Return raw meetings for ``[start, end]``, using the per-year cache.

    Cached windows are reused. With ``refresh`` set, windows in the current year are
    requested again so that newly posted minutes are discovered.
    """
    today = today or date.today()
    meetings: list[dict[str, Any]] = []
    for lo, hi in year_windows(start, end):
        cache_file = paths.calendar_cache / f"{lo.isoformat()}_{hi.isoformat()}.json"
        stale = refresh and hi.year >= today.year
        if cache_file.exists() and not stale:
            meetings.extend(json.loads(cache_file.read_text(encoding="utf-8")))
            continue
        log.info("Listing meetings %s to %s", lo, hi)
        window = client.calendar(lo, hi)
        _write_json(cache_file, window)
        meetings.extend(window)
    return meetings


def sync_minutes(
    client: EscribeClient,
    paths: DataPaths,
    start: date,
    end: date,
    refresh_calendar: bool = False,
    include_pdf: bool = False,
    today: date | None = None,
) -> SyncReport:
    """Download the minutes of every Regular Council Meeting in ``[start, end]``."""
    report = SyncReport()
    raw = load_calendar(client, paths, start, end, refresh=refresh_calendar, today=today)
    records, others = parse_calendar(raw, base_url=client.base_url)
    report.out_of_scope.extend(others)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    for record in records:
        if not (start.isoformat() <= record.meeting_date <= end.isoformat()):
            report.skipped.append(_skip(record, "outside requested date range"))
            continue
        if record.meeting_id in seen:
            report.skipped.append(_skip(record, "duplicate listing of the same meeting"))
            continue
        seen.add(record.meeting_id)
        if record.minutes_html_url is None:
            reason = "no minutes published on the portal"
            if record.pdf_document_id is not None:
                reason = "minutes published as PDF only (PDF endpoint not usable, see README)"
            report.skipped.append(_skip(record, reason))
            continue

        html_path = paths.raw_html / f"{record.meeting_id}.html"
        if html_path.exists():
            report.cached.append(record)
        else:
            try:
                page = client.minutes_html(record.meeting_id)
            except (httpx.HTTPError, ValueError, ChallengeBlockedError) as exc:
                report.skipped.append(_skip(record, f"minutes download failed: {exc}"))
                if isinstance(exc, ChallengeBlockedError):
                    break  # stop politely rather than retrying a blocked endpoint
                continue
            _write_atomic(html_path, page.encode("utf-8"))
            report.downloaded.append(record)
            log.info("Downloaded minutes for %s", record.meeting_date)

        pdf_path = paths.raw_pdfs / f"{record.pdf_document_id}.pdf"
        if (
            include_pdf
            and record.pdf_document_id
            and not pdf_path.exists()
            and not (report.pdf_blocked)
        ):
            try:
                _write_atomic(pdf_path, client.pdf(record.pdf_document_id))
                report.pdf_downloaded += 1
            except ChallengeBlockedError as exc:
                report.pdf_blocked = str(exc)
                log.warning("PDF downloads stopped: %s", exc)
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("PDF %s not downloaded: %s", record.pdf_document_id, exc)

        data = html_path.read_bytes()
        entries.append(
            {
                **asdict(record),
                "html_sha256": hashlib.sha256(data).hexdigest(),
                "html_path": str(html_path.relative_to(paths.root)),
                "pdf_path": str(pdf_path.relative_to(paths.root)) if pdf_path.exists() else None,
            }
        )

    report.requests_made = client.requests_made
    dates = sorted(e["meeting_date"] for e in entries)
    coverage = {
        "from": dates[0] if dates else None,
        "to": dates[-1] if dates else None,
        "meetings": len(entries),
    }
    now = datetime.now(UTC).isoformat(timespec="seconds")
    _write_json(
        paths.manifest,
        {
            "generated_at": now,
            "source": client.base_url,
            "meeting_name": MEETING_NAME,
            "date_range": [start.isoformat(), end.isoformat()],
            "coverage": coverage,
            "documents": entries,
            "skipped": report.skipped,
            "out_of_scope_meetings": len(report.out_of_scope),
            "pdf_blocked": report.pdf_blocked,
        },
    )
    # One line per run (FR-07, PA-14): what changed and the resulting coverage.
    paths.logs.mkdir(parents=True, exist_ok=True)
    with (paths.logs / "ingest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "ts": now,
                    "range": [start.isoformat(), end.isoformat()],
                    "downloaded": [r.meeting_date for r in report.downloaded],
                    "cached": len(report.cached),
                    "skipped": len(report.skipped),
                    "requests": report.requests_made,
                    "pdf_blocked": bool(report.pdf_blocked),
                    "coverage": coverage,
                }
            )
            + "\n"
        )
    return report


def _skip(record: MeetingRecord, reason: str) -> dict[str, Any]:
    return {
        "meeting_name": record.meeting_name,
        "meeting_date": record.meeting_date,
        "meeting_id": record.meeting_id,
        "pdf_document_id": record.pdf_document_id,
        "reason": reason,
    }


def load_manifest(paths: DataPaths) -> dict[str, Any]:
    if not paths.manifest.exists():
        raise FileNotFoundError(f"No manifest at {paths.manifest}. Run `council-rag download`.")
    return json.loads(paths.manifest.read_text(encoding="utf-8"))
