"""Grounded answers from a local open model run with MLX (the default ``local`` backend).

A local model has no citations API, so grounding is enforced by this module:

1. Retrieved passages are numbered ``[1]`` to ``[k]`` in the prompt, each headed by its
   meeting, date and agenda item.
2. The model must reply with one JSON object ``{"answer", "cited_passage_ids", "found"}``.
   The reply is validated against a strict schema (:class:`ModelReply`). Decoding is
   not grammar-constrained, so an invalid reply gets exactly one re-ask that quotes the
   validation error; a second invalid reply is reported as ``invalid_output``.
3. Cited ids are mapped back to meeting, date, agenda item and official URL. Ids that
   do not refer to a passage that was provided are dropped and recorded in
   ``Answer.dropped_citations``.
4. ``found: false``, or an answer with no valid citation, becomes "not found" (FR-03);
   an uncited answer is never shown.

Generation is greedy (temperature 0) so results are repeatable on the same machine.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from council_rag.answers import NOT_FOUND_MESSAGE, Answer, UsageLogger, UsageRecord, citation_for
from council_rag.chunking import Chunk, long_date, make_excerpt
from council_rag.config import PROJECT_ROOT

log = logging.getLogger(__name__)

LOCAL_MODEL_REPO = os.environ.get("COUNCIL_RAG_LOCAL_MODEL", "mlx-community/Qwen3.5-9B-MLX-4bit")
LOCAL_MODEL_REVISION = os.environ.get(
    "COUNCIL_RAG_LOCAL_MODEL_REVISION", "938d8919941c6e7efd3c7150eff7fe9d12afa631"
)
LOCAL_MODEL_LABEL = os.environ.get("COUNCIL_RAG_LOCAL_MODEL_LABEL", "Qwen3.5-9B (MLX 4-bit)")
DEFAULT_PASSAGES = 6

SYSTEM_PROMPT = """\
You answer questions about decisions recorded in the minutes of City of St. John's \
Regular Council Meetings, using numbered passages from the official minutes.

Reply with one JSON object and nothing else, exactly in this form:
{"answer": "<one to three sentences>", "cited_passage_ids": [<passage numbers>], "found": true}

Rules:
- Use only the passages. Do not add facts from general knowledge.
- Put the number of every passage you used in cited_passage_ids.
- Give the meeting date of each decision you mention. Copy vote counts, amounts, names, \
addresses and by-law or file numbers exactly as written.
- Read motions carefully: a carried motion "to reject" means the request was refused, \
and a lost motion did not pass.
- If the passages do not answer the question, reply exactly:
{"answer": "", "cited_passage_ids": [], "found": false}"""


class ModelReply(BaseModel):
    """The only reply shape the app accepts from the local model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str
    cited_passage_ids: list[int]
    found: bool


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class TextGenerator(Protocol):
    model_name: str

    def generate(self, messages: list[dict[str, str]], max_tokens: int) -> Generation: ...


def passage_header(chunk: Chunk) -> str:
    where = (
        "Attendance" if chunk.kind == "attendance" else f"Item {chunk.item_ref}: {chunk.item_title}"
    )
    return f"{chunk.meeting_name}, {long_date(chunk.meeting_date)}. {where}"


def build_messages(question: str, chunks: Sequence[Chunk]) -> list[dict[str, str]]:
    passages = "\n\n".join(
        f"[{i}] {passage_header(c)}\n{c.text}" for i, c in enumerate(chunks, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passages:\n\n{passages}\n\nQuestion: {question.strip()}"},
    ]


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_reply(text: str) -> tuple[ModelReply | None, str | None]:
    """Extract and validate the JSON object; returns ``(reply, None)`` or ``(None, error)``."""
    cleaned = _FENCE_RE.sub("", text.strip())
    start = cleaned.find("{")
    if start < 0:
        return None, "no JSON object found"
    try:
        obj, end = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON ({exc.msg})"
    if cleaned[start + end :].strip():
        return None, "text after the JSON object"
    try:
        reply = ModelReply.model_validate(obj)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "object"
        return None, f"schema error at {where}: {first['msg']}"
    if reply.found and not reply.answer.strip():
        return None, "found is true but answer is empty"
    return reply, None


class MlxGenerator:
    """Runs an MLX model from the Hugging Face cache inside the project (``.cache/huggingface``)."""

    def __init__(
        self,
        repo_id: str = LOCAL_MODEL_REPO,
        revision: str = LOCAL_MODEL_REVISION,
        label: str = LOCAL_MODEL_LABEL,
        cache_dir: Path | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.model_name = label
        self.cache_dir = (
            cache_dir
            or Path(os.environ.get("HF_HOME", PROJECT_ROOT / ".cache" / "huggingface")) / "hub"
        )
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from huggingface_hub import snapshot_download
        from mlx_lm import load

        path = snapshot_download(self.repo_id, revision=self.revision, cache_dir=self.cache_dir)
        self._model, self._tokenizer = load(path)
        log.info("Loaded %s@%s", self.repo_id, self.revision[:8])

    def generate(self, messages: list[dict[str, str]], max_tokens: int) -> Generation:
        self._load()
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        started = time.perf_counter()
        text, last = "", None
        for chunk in stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
        ):
            text += chunk.text
            last = chunk
        return Generation(
            text=text,
            prompt_tokens=int(last.prompt_tokens) if last else 0,
            completion_tokens=int(last.generation_tokens) if last else 0,
            seconds=time.perf_counter() - started,
        )


class LocalAnswerer:
    """The ``local`` backend: free to run, grounding checked by the application."""

    def __init__(
        self,
        generator: TextGenerator | None = None,
        max_tokens: int = 320,
        usage_log: Path | None = None,
    ) -> None:
        self.generator = generator or MlxGenerator()
        self.model_name = self.generator.model_name
        self.max_tokens = max_tokens
        self.logger = UsageLogger(usage_log)

    def answer(self, question: str, chunks: Sequence[Chunk]) -> Answer:
        chunks = list(chunks)
        if not chunks:
            return Answer(
                status="not_found",
                text=NOT_FOUND_MESSAGE,
                detail="No passages matched the question and filters; the model was not called.",
            )
        started = time.perf_counter()
        messages = build_messages(question, chunks)
        out = self.generator.generate(messages, self.max_tokens)
        calls = [out]
        reply, error = parse_reply(out.text)
        if reply is None:
            retry = [
                *messages,
                {"role": "assistant", "content": out.text},
                {
                    "role": "user",
                    "content": f"Your reply was not valid ({error}). Reply again with only the "
                    'JSON object: {"answer": ..., "cited_passage_ids": [...], "found": ...}',
                },
            ]
            out = self.generator.generate(retry, self.max_tokens)
            calls.append(out)
            reply, error = parse_reply(out.text)
        usage = UsageRecord(
            model=self.model_name,
            input_tokens=sum(c.prompt_tokens for c in calls),
            output_tokens=sum(c.completion_tokens for c in calls),
            cost_usd=0.0,  # runs locally; electricity is not measured
            attempts=len(calls),
        )
        latency = round(time.perf_counter() - started, 3)
        answer = self._to_answer(reply, error, chunks)
        answer.usage, answer.latency_s, answer.model = usage, latency, self.model_name
        answer.raw_replies = [c.text for c in calls]
        self.logger.write(answer, len(chunks))
        return answer

    @staticmethod
    def _to_answer(reply: ModelReply | None, error: str | None, chunks: list[Chunk]) -> Answer:
        if reply is None:
            return Answer(
                status="invalid_output",
                text=NOT_FOUND_MESSAGE,
                detail=f"The model reply failed validation twice ({error}).",
            )
        valid: list[int] = []
        dropped: list[int] = []
        for pid in reply.cited_passage_ids:
            if 1 <= pid <= len(chunks):
                if pid not in valid:
                    valid.append(pid)
            else:
                dropped.append(pid)
        text = reply.answer.strip()
        citations = [
            citation_for(
                chunks[pid - 1],
                pid - 1,
                cited_text=make_excerpt(chunks[pid - 1]),
                answer_start=0,
                answer_end=len(text),
            )
            for pid in valid
        ]
        detail = (
            f"Dropped citations to passages that were not provided: {dropped}" if dropped else None
        )
        if not reply.found:
            return Answer(status="not_found", text=NOT_FOUND_MESSAGE, dropped_citations=dropped)
        if not citations:
            return Answer(
                status="not_found",
                text=NOT_FOUND_MESSAGE,
                detail="The model's answer cited no provided passage, so it was withheld.",
                dropped_citations=dropped,
            )
        return Answer(
            status="answered",
            text=text,
            citations=citations,
            detail=detail,
            dropped_citations=dropped,
        )
