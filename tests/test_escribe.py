import json
from datetime import date

import httpx
import pytest

from council_rag.config import data_paths
from council_rag.ingest.escribe import (
    ChallengeBlockedError,
    EscribeClient,
    RateLimiter,
    parse_calendar,
    sync_minutes,
    year_windows,
)

CALENDAR = [
    {
        "ID": "11111111-aaaa-4000-8000-000000000001",
        "MeetingName": "Regular Council Meeting",
        "StartDate": "2024/03/04 15:00:00",
        "Url": "https://example.invalid/m1",
        "MeetingDocumentLink": [
            {"Title": "Agenda (PDF)", "Url": "/FileStream.ashx?DocumentId=10"},
            {"Title": "Minutes (PDF)", "Url": "/FileStream.ashx?DocumentId=11"},
            {"Title": "Minutes (HTML)", "Url": "/Meeting.aspx?Id=1&Agenda=PostMinutes"},
        ],
    },
    {
        "ID": "22222222-aaaa-4000-8000-000000000002",
        "MeetingName": "Regular Council Meeting",
        "StartDate": "2024/03/18 15:00:00",
        "Url": "https://example.invalid/m2",
        "MeetingDocumentLink": [{"Title": "Agenda (PDF)", "Url": "/FileStream.ashx?DocumentId=12"}],
    },
    {
        "ID": "33333333-aaaa-4000-8000-000000000003",
        "MeetingName": "Committee of the Whole",
        "StartDate": "2024/03/06 09:30:00",
        "Url": "https://example.invalid/m3",
        "MeetingDocumentLink": [],
    },
    {
        "ID": "44444444-aaaa-4000-8000-000000000004",
        "MeetingName": "St. John&#39;s Local Immigration Partnership Council",
        "StartDate": "2024/03/07 09:30:00",
        "Url": "https://example.invalid/m4",
        "MeetingDocumentLink": [],
    },
]


def test_parse_calendar_selects_regular_meetings() -> None:
    records, others = parse_calendar(CALENDAR, base_url="https://portal.example")
    assert [r.meeting_date for r in records] == ["2024-03-04", "2024-03-18"]
    first, second = records
    assert first.pdf_document_id == 11
    assert first.pdf_url == "https://portal.example/FileStream.ashx?DocumentId=11"
    assert first.minutes_html_url and "Agenda=PostMinutes" in first.minutes_html_url
    assert second.minutes_html_url is None and second.pdf_document_id is None
    assert {o["meeting_name"] for o in others} == {
        "Committee of the Whole",
        "St. John's Local Immigration Partnership Council",
    }


def test_year_windows() -> None:
    assert list(year_windows(date(2023, 6, 1), date(2024, 2, 1))) == [
        (date(2023, 6, 1), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 2, 1)),
    ]


def test_rate_limiter_spaces_calls() -> None:
    now = [100.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    limiter = RateLimiter(6.0, clock=lambda: now[0], sleep=sleep)
    limiter.wait()
    now[0] += 1.5
    limiter.wait()
    assert slept == [pytest.approx(4.5)]


def _client(handler) -> EscribeClient:
    no_wait = RateLimiter(0.0, sleep=lambda s: None)
    return EscribeClient(
        base_url="https://portal.example", transport=httpx.MockTransport(handler), limiter=no_wait
    )


def test_sync_downloads_once_and_logs_skips(tmp_path, synthetic_html) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        assert request.headers["User-Agent"].startswith("stjohns-council-minutes-rag/")
        if request.url.path.endswith("GetCalendarMeetings"):
            body = json.loads(request.content)
            assert body["calendarStartDate"] == "2024-01-01"
            return httpx.Response(200, json={"d": CALENDAR})
        if request.url.path == "/Meeting.aspx":
            return httpx.Response(200, text=synthetic_html)
        return httpx.Response(404)

    paths = data_paths(tmp_path)
    with _client(handler) as client:
        report = sync_minutes(client, paths, date(2024, 1, 1), date(2024, 12, 31))
    assert [r.meeting_date for r in report.downloaded] == ["2024-03-04"]
    assert [s["reason"] for s in report.skipped] == ["no minutes published on the portal"]
    assert len(report.out_of_scope) == 2
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["documents"][0]["pdf_document_id"] == 11
    assert (tmp_path / manifest["documents"][0]["html_path"]).exists()

    calls.clear()
    with _client(handler) as client:  # second run: everything cached, no requests at all
        report = sync_minutes(client, paths, date(2024, 1, 1), date(2024, 12, 31))
    assert calls == [] and len(report.cached) == 1


def test_challenge_is_detected_and_not_retried(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("GetCalendarMeetings"):
            return httpx.Response(200, json={"d": CALENDAR})
        return httpx.Response(
            403, headers={"cf-mitigated": "challenge"}, text="Verifying your browser"
        )

    paths = data_paths(tmp_path)
    with _client(handler) as client:
        with pytest.raises(ChallengeBlockedError):
            client.pdf(11)
        report = sync_minutes(client, paths, date(2024, 1, 1), date(2024, 12, 31), include_pdf=True)
        assert client.requests_made == 3  # pdf probe, calendar, one minutes page, then stop
    assert "bot challenge" in report.skipped[0]["reason"]
    assert report.downloaded == []
