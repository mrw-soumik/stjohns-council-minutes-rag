"""Answer-quality evaluation with deterministic grading (NFR-02, NFR-03, NFR-04).

For each question: hybrid retrieval (with the question's date filter) -> answer backend
-> grade. Grading needs no model judge:

* ``answered``: the backend returned an answer (not "not found").
* ``cites_gold``: at least one citation points to a passage that contains the gold
  evidence (the same relevance rule as the retrieval evaluation). ``cites_gold_item``
  relaxes this to the right agenda item.
* ``facts_ok``: every ``answer_facts`` group appears in the answer text (any listed
  alternative counts; case, accent and whitespace-insensitive).
* ``correct`` (strict): answered, facts present and gold cited.
* citation precision: share of citations (over answered, answerable questions) that point
  to a gold-evidence passage. A citation to a passage that supports the claim but is not
  the gold item counts as unsupported, so this is a lower bound on NFR-02's "citations
  point to a passage that supports the claim".
* unanswerable questions are graded on ``declined`` (anything but an answer).

Rows are appended to ``<out>/<tag>/results.jsonl`` as each case finishes, and a rerun
skips ids already present. Prompts contain minutes text, so full traces are written under
the gitignored data directory only.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from council_rag.answers import Answer, Answerer, make_answerer
from council_rag.chunking import Chunk
from council_rag.config import data_paths
from council_rag.evaluation.dataset import EvalQuestion, load_questions, relevant_sets
from council_rag.evaluation.metrics import percentile, wilson_interval
from council_rag.retrieval import HybridRetriever
from council_rag.text import contains_normalized

NFR02_CITATION_TARGET = 0.90
NFR02_DECLINE_TARGET = 0.80
NFR03_P95_TARGET_S = 10.0
OFFICIAL_URL = re.compile(
    r"^https://pub-stjohns\.escribemeetings\.com/Meeting\.aspx\?Id=(?P<id>[0-9a-f-]{36})"
    r"&Agenda=PostMinutes&lang=English(?:#(?P<anchor>AgendaItemAgendaItem\d+TitleHeader))?$"
)


def check_citation_links(
    rows: list[dict[str, Any]], chunks_by_id: dict[str, Chunk], html_dir: Path | None
) -> dict[str, Any]:
    """PA-11: every answer cites meeting name, date and item with a valid official URL.

    A URL is valid when it has the official minutes form and, if it carries an anchor,
    the anchor exists in the downloaded page for that meeting.
    """
    answered = [r for r in rows if r.get("status") == "answered"]
    problems: list[str] = []
    n_citations = 0
    for r in answered:
        if not r["citations"]:
            problems.append(f"{r['id']}: answer without citation")
        for c in r["citations"]:
            n_citations += 1
            chunk = chunks_by_id.get(c["chunk_id"])
            match = OFFICIAL_URL.match(c.get("url") or "")
            if chunk is None or not chunk.meeting_name or not c.get("meeting_date"):
                problems.append(f"{r['id']}: citation without meeting name or date")
            elif chunk.kind == "item" and not c.get("item_ref"):
                problems.append(f"{r['id']}: citation without agenda item")
            elif match is None or match["id"] != chunk.meeting_id:
                problems.append(f"{r['id']}: citation URL is not the official minutes page")
            elif match["anchor"] and html_dir is not None:
                page = html_dir / f"{match['id']}.html"
                if not page.exists() or match["anchor"] not in page.read_text(encoding="utf-8"):
                    problems.append(f"{r['id']}: anchor {match['anchor']} not found on the page")
    return {
        "answered": len(answered),
        "citations": n_citations,
        "problems": problems,
        "pass": not problems,
    }


def check_date_filters(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """PA-12: no passage given to the model falls outside the question's date range."""
    violations = []
    for r in rows:
        lo, hi = r.get("date_from"), r.get("date_to")
        for cid in r.get("passages", []):
            day = cid[:10]
            if (lo and day < lo) or (hi and day > hi):
                violations.append(f"{r['id']}: {cid}")
    checked = sum(1 for r in rows if r.get("date_from") or r.get("date_to"))
    return {"filtered_questions": checked, "violations": violations, "pass": not violations}


def facts_present(text: str, facts: list[list[str]]) -> bool:
    return all(any(contains_normalized(text, alt) for alt in group) for group in facts)


def grade(
    question: EvalQuestion,
    answer: Answer,
    passages: list[Chunk],
) -> dict[str, Any]:
    cited_ids = [c.chunk_id for c in answer.citations]
    if not question.answerable:
        return {"declined": answer.status != "answered", "n_citations": len(cited_ids)}
    evidence = set().union(*relevant_sets(question, passages)) if question.gold else set()
    items = set().union(*relevant_sets(question, passages, "item")) if question.gold else set()
    answered = answer.status == "answered"
    supported = sum(1 for cid in cited_ids if cid in evidence)
    supported_item = sum(1 for cid in cited_ids if cid in items)
    facts_ok = answered and facts_present(answer.text, question.answer_facts)
    cites_gold = answered and supported > 0
    return {
        "answered": answered,
        "gold_retrieved": bool(evidence),
        "cites_gold": cites_gold,
        "cites_gold_item": answered and supported_item > 0,
        "facts_ok": facts_ok,
        "correct": bool(answered and facts_ok and cites_gold),
        "n_citations": len(cited_ids),
        "n_supported": supported,
        "n_supported_item": supported_item,
    }


def _rate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    n = len(rows)
    k = sum(1 for r in rows if r["grade"].get(key))
    lo, hi = wilson_interval(k, n)
    return {
        "k": k,
        "n": n,
        "rate": round(k / n, 4) if n else math.nan,
        "ci95": [round(lo, 3), round(hi, 3)],
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if r["status"] != "error"]
    answerable = [r for r in ok_rows if r["answerable"]]
    unanswerable = [r for r in ok_rows if not r["answerable"]]
    answered = [r for r in answerable if r["grade"]["answered"]]
    cits = sum(r["grade"]["n_citations"] for r in answered)
    sup = sum(r["grade"]["n_supported"] for r in answered)
    sup_item = sum(r["grade"]["n_supported_item"] for r in answered)
    lat = [r["latency_s"] for r in ok_rows if r.get("latency_s") is not None]
    by_type: dict[str, dict[str, Any]] = {}
    for t in sorted({r["type"] for r in ok_rows}):
        group = [r for r in ok_rows if r["type"] == t]
        key = "declined" if t == "unanswerable" else "correct"
        by_type[t] = {"n": len(group), key: _rate(group, key)["rate"]}
    return {
        "cases": len(rows),
        "errors": len(rows) - len(ok_rows),
        "answerable": {
            "answered": _rate(answerable, "answered"),
            "correct": _rate(answerable, "correct"),
            "facts_ok": _rate(answerable, "facts_ok"),
            "cites_gold": _rate(answerable, "cites_gold"),
            "cites_gold_item": _rate(answerable, "cites_gold_item"),
            "gold_retrieved": _rate(answerable, "gold_retrieved"),
        },
        "citation_precision": {
            "citations": cits,
            "supported_evidence": round(sup / cits, 4) if cits else math.nan,
            "supported_item": round(sup_item / cits, 4) if cits else math.nan,
        },
        "unanswerable": {"declined": _rate(unanswerable, "declined")},
        "status_counts": {
            s: sum(1 for r in ok_rows if r["status"] == s)
            for s in sorted({r["status"] for r in ok_rows})
        },
        "reasks": sum(1 for r in ok_rows if r.get("attempts", 1) > 1),
        "dropped_citation_cases": sum(1 for r in ok_rows if r.get("dropped_citations")),
        "latency_s": {
            "p50": round(percentile(lat, 50), 2) if lat else None,
            "p95": round(percentile(lat, 95), 2) if lat else None,
            "max": round(max(lat), 2) if lat else None,
        },
        "tokens": {
            "mean_input": round(sum(r["input_tokens"] for r in ok_rows) / max(len(ok_rows), 1)),
            "mean_output": round(sum(r["output_tokens"] for r in ok_rows) / max(len(ok_rows), 1)),
        },
        "by_type": by_type,
    }


def hardware() -> dict[str, str]:
    info = {"python": platform.python_version(), "machine": platform.machine()}
    if platform.system() == "Darwin":

        def sysctl(key: str) -> str:
            res = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
            return res.stdout.strip()

        info["chip"] = sysctl("machdep.cpu.brand_string")
        info["memory_gb"] = str(round(int(sysctl("hw.memsize") or 0) / 2**30))
        info["os"] = f"macOS {platform.mac_ver()[0]}"
    else:
        info["os"] = platform.platform()
    return info


def run(
    questions: list[EvalQuestion],
    retriever: HybridRetriever,
    answerer: Answerer,
    out_dir: Path,
    k: int,
    trace_dir: Path | None = None,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    for q in questions:
        if q.id in done:
            continue
        started = time.perf_counter()
        hits = retriever.search(q.question, k=k, date_from=q.date_from, date_to=q.date_to)
        retrieval_s = time.perf_counter() - started
        passages = [h.chunk for h in hits]
        try:
            answer = answerer.answer(q.question, passages)
            status = answer.status
        except Exception as exc:  # recorded as a harness error, never as a model failure
            row = {
                "id": q.id,
                "status": "error",
                "error": repr(exc)[:300],
                "answerable": q.answerable,
                "type": q.type,
            }
            with (out_dir / "errors.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            continue
        total_s = time.perf_counter() - started
        usage = answer.usage
        row = {
            "id": q.id,
            "type": q.type,
            "answerable": q.answerable,
            "question": q.question,
            "date_from": q.date_from,
            "date_to": q.date_to,
            "status": status,
            "answer": answer.text if status == "answered" else "",
            "citations": [
                {
                    "passage": c.document_index + 1,
                    "chunk_id": c.chunk_id,
                    "meeting_name": c.meeting_name,
                    "meeting_date": c.meeting_date,
                    "item_ref": c.item_ref,
                    "item_title": c.item_title,
                    "url": c.url,
                }
                for c in answer.citations
            ],
            "dropped_citations": answer.dropped_citations,
            "detail": answer.detail,
            "model": answer.model or getattr(answerer, "model_name", None),
            "attempts": usage.attempts if usage else 0,
            "input_tokens": usage.input_tokens if usage else 0,
            "output_tokens": usage.output_tokens if usage else 0,
            "cost_usd": usage.cost_usd if usage else 0.0,
            "retrieval_s": round(retrieval_s, 3),
            "latency_s": round(total_s, 3),
            "passages": [c.chunk_id for c in passages],
            "grade": grade(q, answer, passages),
        }
        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[q.id] = row
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            from council_rag.local_llm import build_messages

            trace = {"messages": build_messages(q.question, passages), "answer": asdict(answer)}
            (trace_dir / f"{q.id}.json").write_text(json.dumps(trace, indent=1), encoding="utf-8")
    return [done[q.id] for q in questions if q.id in done]


def _links(summary: dict[str, Any]) -> str:
    links = summary.get("pa11_citation_links")
    if not links:
        return "not checked"
    ok = links["citations"] - len(links["problems"])
    return f"{ok} of {links['citations']} in {links['answered']} answers"


def _dates(summary: dict[str, Any]) -> str:
    dates = summary.get("pa12_date_filter")
    if not dates:
        return "not checked"
    return f"{len(dates['violations'])} (over {dates['filtered_questions']} filtered questions)"


def markdown(summary: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    a = summary["answerable"]
    cp = summary["citation_precision"]
    dec = summary["unanswerable"]["declined"]
    lat = summary["latency_s"]

    def pct(r: dict[str, Any]) -> str:
        return (
            f"{r['rate']:.3f} ({r['k']}/{r['n']}; 95% CI {r['ci95'][0]:.2f} to {r['ci95'][1]:.2f})"
        )

    hw = meta["hardware"]
    lines = [
        f"## Answers: {meta['model']} (k={meta['k']} passages)",
        "",
        f"Run {meta['run_at']} on {hw.get('chip', hw['machine'])}, {hw.get('memory_gb', '?')} GB, "
        f"{hw['os']}. Greedy decoding; grading is deterministic (no model judge).",
        "",
        "| Metric | Result | Target |",
        "|---|---|---|",
        f"| Answerable: strictly correct (facts + gold cited) | {pct(a['correct'])} | |",
        f"| Answerable: required facts present | {pct(a['facts_ok'])} | |",
        f"| Answerable: cites a gold-evidence passage | {pct(a['cites_gold'])} | |",
        f"| Answerable: answered (not declined) | {pct(a['answered'])} | |",
        f"| Citation precision, gold evidence (lower bound) | {cp['supported_evidence']:.3f} of "
        f"{cp['citations']} citations | NFR-02 >= {NFR02_CITATION_TARGET:.2f} |",
        f"| Citation precision, gold agenda item | {cp['supported_item']:.3f} | |",
        f"| Unanswerable: correctly declined | {pct(dec)} | NFR-02 >= {NFR02_DECLINE_TARGET:.2f} |",
        f"| End-to-end latency p50 / p95 / max (s) | {lat['p50']} / {lat['p95']} / {lat['max']} "
        f"| NFR-03 p95 < {NFR03_P95_TARGET_S:.0f} |",
        "| Cost per answer | $0.00 (local) | NFR-04 < CAD 0.05 |",
        f"| Mean tokens in / out | {summary['tokens']['mean_input']} / "
        f"{summary['tokens']['mean_output']} | |",
        f"| Re-asks after invalid JSON / final invalid outputs | {summary['reasks']} / "
        f"{summary['status_counts'].get('invalid_output', 0)} | |",
        f"| Answers with citations to passages not provided (dropped) | "
        f"{summary['dropped_citation_cases']} | |",
        f"| Citations with meeting, date, item and a working official link (PA-11) | "
        f"{_links(summary)} | all |",
        f"| Passages outside the question's date range (PA-12) | {_dates(summary)} | 0 |",
        "",
        "By question type (strict correctness for answerable types, declines for unanswerable):",
        "",
        "| Type | n | Rate |",
        "|---|---|---|",
    ]
    for t, v in summary["by_type"].items():
        rate = v.get("correct", v.get("declined"))
        lines.append(f"| {t} | {v['n']} | {rate:.2f} |")
    lines.append("")
    return lines


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument(
        "--backend",
        choices=["local", "claude"],
        default="local",
        help="'claude' calls the paid API; the default local model costs nothing",
    )
    p.add_argument("-k", type=int, default=6, help="passages given to the model")
    p.add_argument(
        "--tag", default=None, help="results folder name (default answers_<backend>_k<k>)"
    )
    p.add_argument("--out", default="eval/results")
    p.add_argument("--ids", default=None, help="comma-separated subset of question ids")
    p.add_argument(
        "--summarise-only",
        action="store_true",
        help="recompute summary.json from saved rows without generating",
    )


def main(args: argparse.Namespace) -> int:
    from council_rag.index import load_index
    from council_rag.service import load_retriever

    questions = load_questions(args.questions)
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        questions = [q for q in questions if q.id in wanted]
    tag = args.tag or f"answers_{args.backend}_k{args.k}"
    paths = data_paths()
    out_dir = Path(args.out) / tag
    summary_path = out_dir / "summary.json"
    previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if args.summarise_only:
        rows = [
            json.loads(line)
            for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        known = {q.id for q in questions}
        rows = [r for r in rows if r["id"] in known]
        meta = previous.get("meta", {})
    else:
        retriever, _ = load_retriever(paths.index)
        answerer = make_answerer(args.backend)
        load_started = time.perf_counter()
        if hasattr(answerer, "generator") and hasattr(answerer.generator, "_load"):
            answerer.generator._load()  # model load time is reported apart from answer latency
            answerer.generator.generate([{"role": "user", "content": "Reply with OK."}], 4)
        load_s = round(time.perf_counter() - load_started, 2)
        rows = run(
            questions,
            retriever,
            answerer,
            out_dir,
            args.k,
            trace_dir=paths.root / "eval_traces" / tag,
        )
        meta = {
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "backend": args.backend,
            "model": getattr(answerer, "model_name", args.backend),
            "k": args.k,
            "model_load_and_warmup_s": load_s,
            "hardware": hardware(),
        }
        if previous.get("meta", {}).get("run_at"):
            meta["first_run_at"] = previous["meta"].get("first_run_at", previous["meta"]["run_at"])
    summary = summarise(rows)
    chunks, _, _ = load_index(paths.index)
    summary["pa11_citation_links"] = check_citation_links(
        rows, {c.chunk_id: c for c in chunks}, paths.raw_html
    )
    summary["pa12_date_filter"] = check_date_filters(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {"meta": meta, "summary": summary, "markdown": markdown(summary, meta)}, indent=1
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n".join(markdown(summary, meta)))
    return 0
