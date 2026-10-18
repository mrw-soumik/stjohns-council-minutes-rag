"""FastAPI service: ``/health``, ``/search`` and ``/ask``.

Run with ``council-rag serve`` (or ``uvicorn council_rag.api:app``). The app loads the
local index on first use; ``/ask`` calls the Claude API and therefore needs
credentials (``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile).
"""

from __future__ import annotations

import time
from datetime import date
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from council_rag import __version__
from council_rag.answers import Answerer, configured_model_label, make_answerer
from council_rag.chunking import EXCERPT_MAX_CHARS, excerpt_for
from council_rag.config import data_paths
from council_rag.generation import GenerationError
from council_rag.index import IndexMeta
from council_rag.retrieval import HybridRetriever, SearchResult

DISCLAIMER = (
    "Independent portfolio project. Not affiliated with or endorsed by the City of "
    "St. John's. Answers are generated from excerpts of the official minutes and can be "
    "wrong; check the linked source before relying on them."
)


class DateRange(BaseModel):
    date_from: date | None = Field(None, description="Earliest meeting date (inclusive)")
    date_to: date | None = Field(None, description="Latest meeting date (inclusive)")

    @model_validator(mode="after")
    def _ordered(self) -> DateRange:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")
        return self

    def bounds(self) -> tuple[str | None, str | None]:
        return (
            self.date_from.isoformat() if self.date_from else None,
            self.date_to.isoformat() if self.date_to else None,
        )


class SearchRequest(DateRange):
    query: str = Field(..., min_length=1, max_length=500)
    k: int = Field(5, ge=1, le=20)
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"


class AskRequest(DateRange):
    question: str = Field(..., min_length=1, max_length=500)
    k: int = Field(8, ge=1, le=20, description="Passages sent to the model")


class Passage(BaseModel):
    rank: int
    chunk_id: str
    meeting_name: str
    meeting_date: date
    item_ref: str
    item_title: str
    excerpt: str = Field(
        ...,
        description=(
            f"Opening words of the motion, verbatim, at most {EXCERPT_MAX_CHARS} characters; "
            "empty for discussion notes, attendance and routine items"
        ),
    )
    outcome: str | None = Field(None, description="Recorded result, e.g. 'carried, 9 to 0'")
    url: str | None = Field(None, description="Official minutes page, anchored at the item")
    pdf_url: str | None = Field(None, description="Official PDF of the minutes, if published")
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[Passage]
    took_ms: float


class CitationOut(BaseModel):
    meeting_name: str
    meeting_date: date
    item_ref: str
    item_title: str
    cited_text: str
    url: str | None
    pdf_url: str | None
    answer_start: int
    answer_end: int


class UsageOut(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class AskResponse(BaseModel):
    status: Literal["answered", "not_found", "refused", "truncated", "invalid_output"]
    answer: str
    citations: list[CitationOut]
    passages: list[Passage]
    model: str | None
    usage: UsageOut | None
    took_ms: float
    disclaimer: str = DISCLAIMER


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    meetings: int
    chunks: int
    coverage_from: date
    coverage_to: date
    embedding_model: str
    answer_model: str
    disclaimer: str = DISCLAIMER


def to_passage(result: SearchResult) -> Passage:
    c = result.chunk
    ex = excerpt_for(c)
    return Passage(
        rank=result.rank,
        chunk_id=c.chunk_id,
        meeting_name=c.meeting_name,
        meeting_date=date.fromisoformat(c.meeting_date),
        item_ref=c.item_ref,
        item_title=c.item_title,
        excerpt=ex.text,
        outcome=ex.outcome,
        url=c.url,
        pdf_url=c.pdf_url,
        score=round(result.score, 6),
        bm25_rank=result.bm25_rank,
        dense_rank=result.dense_rank,
    )


class AppState:
    """Lazily created, process-wide resources (overridable in tests)."""

    def __init__(self) -> None:
        self.retriever: HybridRetriever | None = None
        self.meta: IndexMeta | None = None
        self.generator: Answerer | None = None

    def get_retriever(self) -> tuple[HybridRetriever, IndexMeta]:
        if self.retriever is None or self.meta is None:
            from council_rag.service import load_retriever

            try:
                self.retriever, self.meta = load_retriever()
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return self.retriever, self.meta

    def get_generator(self) -> Answerer:
        if self.generator is None:
            self.generator = make_answerer(usage_log=data_paths().logs / "usage.jsonl")
        return self.generator


state = AppState()
app = FastAPI(
    title="St. John's Council Minutes Assistant",
    version=__version__,
    description="Search and grounded Q&A over City of St. John's Regular Council Meeting minutes.",
)


def retriever_dep() -> tuple[HybridRetriever, IndexMeta]:
    return state.get_retriever()


RetrieverDep = Annotated[tuple[HybridRetriever, IndexMeta], Depends(retriever_dep)]


@app.get("/health", response_model=Health)
def health(dep: RetrieverDep) -> Health:
    _, meta = dep
    return Health(
        status="ok",
        version=__version__,
        meetings=meta.n_meetings,
        chunks=meta.n_chunks,
        coverage_from=date.fromisoformat(meta.date_min),
        coverage_to=date.fromisoformat(meta.date_max),
        embedding_model=f"{meta.embedding_repo}@{meta.embedding_revision[:8]}",
        answer_model=configured_model_label(),
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, dep: RetrieverDep) -> SearchResponse:
    retriever, _ = dep
    lo, hi = req.bounds()
    started = time.perf_counter()
    results = retriever.search(req.query, k=req.k, mode=req.mode, date_from=lo, date_to=hi)
    took = (time.perf_counter() - started) * 1000
    return SearchResponse(
        query=req.query,
        mode=req.mode,
        results=[to_passage(r) for r in results],
        took_ms=round(took, 2),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, dep: RetrieverDep) -> AskResponse:
    retriever, _ = dep
    lo, hi = req.bounds()
    started = time.perf_counter()
    results = retriever.search(req.question, k=req.k, date_from=lo, date_to=hi)
    try:
        generator = state.get_generator()
        answer = generator.answer(req.question, [r.chunk for r in results])
    except GenerationError as exc:
        status = 503 if exc.status_code in (401, 403) else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except ImportError as exc:  # the local backend needs the optional [local] extra (MLX)
        raise HTTPException(status_code=503, detail=f"Answer model unavailable: {exc}") from exc
    took = (time.perf_counter() - started) * 1000
    usage: dict[str, Any] | None = None
    if answer.usage is not None:
        usage = {
            "model": answer.usage.model,
            "input_tokens": answer.usage.input_tokens,
            "output_tokens": answer.usage.output_tokens,
            "cost_usd": answer.usage.cost_usd,
        }
    return AskResponse(
        status=answer.status,  # type: ignore[arg-type]
        answer=answer.text,
        citations=[
            CitationOut(
                meeting_name=c.meeting_name,
                meeting_date=date.fromisoformat(c.meeting_date),
                item_ref=c.item_ref,
                item_title=c.item_title,
                cited_text=c.cited_text,
                url=c.url,
                pdf_url=c.pdf_url,
                answer_start=c.answer_start,
                answer_end=c.answer_end,
            )
            for c in answer.citations
        ],
        passages=[to_passage(r) for r in results],
        model=answer.model,
        usage=UsageOut(**usage) if usage else None,
        took_ms=round(took, 2),
    )
