"""Command-line interface: ``council-rag <command>``.

Pipeline (FR-07): ``download`` -> ``parse`` -> ``build-index``; ``update`` runs all three
and only fetches what is new. ``search`` works offline; ``ask`` calls the Claude API.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from datetime import date
from pathlib import Path

from council_rag.config import DEFAULT_START_DATE, data_paths


def _date(value: str) -> str:
    return date.fromisoformat(value).isoformat()


def cmd_download(args: argparse.Namespace) -> int:
    from council_rag.ingest.escribe import EscribeClient, sync_minutes

    paths = data_paths(args.data_dir)
    with EscribeClient(min_interval=args.min_interval) as client:
        report = sync_minutes(
            client,
            paths,
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
            refresh_calendar=args.refresh,
            include_pdf=args.pdf,
        )
    print(
        f"Downloaded {len(report.downloaded)}, already cached {len(report.cached)}, "
        f"skipped {len(report.skipped)} Regular Council Meetings "
        f"({report.requests_made} requests; {len(report.out_of_scope)} other meetings ignored)."
    )
    for skip in report.skipped:
        print(f"  skipped {skip['meeting_date']}: {skip['reason']}")
    if report.pdf_blocked:
        print(f"  PDF downloads stopped: {report.pdf_blocked}")
    print(f"Manifest: {paths.manifest}")
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    from council_rag.index import parse_all

    paths = data_paths(args.data_dir)
    minutes = parse_all(paths)
    items = sum(len(m.items) for m in minutes)
    print(f"Parsed {len(minutes)} meetings ({items} agenda items) into {paths.parsed}")
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    from council_rag.chunking import ChunkingConfig
    from council_rag.index import build_index

    paths = data_paths(args.data_dir)
    _, _, meta = build_index(
        paths,
        ChunkingConfig(max_words=args.max_words),
        index_with_context=not args.no_context,
    )
    print(
        f"Indexed {meta.n_chunks} chunks from {meta.n_meetings} meetings "
        f"({meta.date_min} to {meta.date_max}) into {paths.index}"
    )
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    args.refresh = True
    for step in (cmd_download, cmd_parse, cmd_build_index):
        code = step(args)
        if code:
            return code
    return 0


def _print_results(results: list) -> None:
    from council_rag.chunking import make_excerpt

    for r in results:
        c = r.chunk
        print(f"{r.rank}. {c.meeting_date}  {c.meeting_name}  item {c.item_ref}: {c.item_title}")
        print(textwrap.indent(textwrap.fill(make_excerpt(c), 96), "   "))
        print(f"   {c.url}")


def cmd_search(args: argparse.Namespace) -> int:
    from council_rag.service import load_retriever

    retriever, _ = load_retriever(data_paths(args.data_dir).index)
    results = retriever.search(
        args.query, k=args.k, mode=args.mode, date_from=args.date_from, date_to=args.date_to
    )
    if not results:
        print("No matching passages.")
        return 0
    _print_results(results)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from council_rag.generation import AnswerGenerator, GenerationError
    from council_rag.service import load_retriever

    paths = data_paths(args.data_dir)
    retriever, _ = load_retriever(paths.index)
    results = retriever.search(
        args.question, k=args.k, date_from=args.date_from, date_to=args.date_to
    )
    generator = AnswerGenerator(usage_log=paths.logs / "usage.jsonl")
    try:
        answer = generator.answer(args.question, [r.chunk for r in results])
    except GenerationError as exc:
        print(f"Answer generation failed: {exc}", file=sys.stderr)
        return 2
    print(answer.text)
    for i, c in enumerate(answer.citations, start=1):
        print(f'  [{i}] {c.meeting_name}, {c.meeting_date}, item {c.item_ref}: "{c.cited_text}"')
        print(f"      {c.url}")
    if answer.usage:
        u = answer.usage
        print(f"({u.model}; {u.input_tokens} in / {u.output_tokens} out tokens; ${u.cost_usd:.4f})")
    return 0


def cmd_eval_retrieval(args: argparse.Namespace) -> int:
    from council_rag.evaluation.retrieval_eval import main as eval_main

    return eval_main(args)


def cmd_verify_eval(args: argparse.Namespace) -> int:
    from council_rag.evaluation.verify import main as verify_main

    return verify_main(args)


def cmd_export_demo(args: argparse.Namespace) -> int:
    from council_rag.demo_export import export_demo

    summary = export_demo(data_paths(args.data_dir).index, Path(args.out))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("council_rag.api:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="council-rag", description=__doc__)
    parser.add_argument("--data-dir", default=None, help="data directory (default: ./data)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_download_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--start", type=_date, default=DEFAULT_START_DATE)
        p.add_argument("--end", type=_date, default=date.today().isoformat())
        p.add_argument(
            "--min-interval",
            type=float,
            default=6.0,
            help="seconds between requests (default 6; the portal rate-limits bursts)",
        )
        p.add_argument(
            "--pdf",
            action="store_true",
            help="also try the PDF endpoint (stops at the first bot challenge)",
        )

    p = sub.add_parser("download", help="download minutes from the eSCRIBE portal")
    add_download_args(p)
    p.add_argument("--refresh", action="store_true", help="re-list the current year")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("parse", help="parse downloaded minutes into data/parsed")
    p.set_defaults(func=cmd_parse)

    def add_index_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--max-words", type=int, default=250)
        p.add_argument("--no-context", action="store_true", help="index chunk text only")

    p = sub.add_parser("build-index", help="chunk, embed and save the search index")
    add_index_args(p)
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("update", help="download new minutes, parse and rebuild the index")
    add_download_args(p)
    add_index_args(p)
    p.set_defaults(func=cmd_update)

    def add_filters(p: argparse.ArgumentParser) -> None:
        p.add_argument("--from", dest="date_from", type=_date, default=None)
        p.add_argument("--to", dest="date_to", type=_date, default=None)

    p = sub.add_parser("search", help="search passages (no API calls)")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
    add_filters(p)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer a question with citations (calls the Claude API)")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=8)
    add_filters(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval-retrieval", help="run the retrieval evaluation")
    from council_rag.evaluation.retrieval_eval import add_arguments

    add_arguments(p)
    p.set_defaults(func=cmd_eval_retrieval)

    p = sub.add_parser("verify-eval", help="check gold evidence in the evaluation set")
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.set_defaults(func=cmd_verify_eval)

    p = sub.add_parser("export-demo", help="write the browser demo index to docs/data")
    p.add_argument("--out", default="docs/data")
    p.set_defaults(func=cmd_export_demo)

    p = sub.add_parser("serve", help="run the FastAPI service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
