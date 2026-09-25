"""The grounded answering chain, built with LangChain Expression Language (LCEL).

Flow::

    inputs ──► prepare (retrieve + rerank, or whole-document context)
           ──► build fenced, numbered context prompt
           ──► ChatPromptTemplate(system, history, human)
           ──► ChatGoogleGenerativeAI  ──► StrOutputParser
           ──► verify citations

``RagChain.runnable`` exposes the whole flow as one LCEL ``Runnable`` for
non-streaming use; :meth:`RagChain.stream_answer` runs the same steps but
streams tokens from the generation sub-chain with ``astream``.

Citation verification matters as much as the prompt. Instructing a model to
cite is not the same as it citing correctly: it will occasionally emit ``[7]``
when only five sources were supplied. :func:`verify_citations` rewrites the
answer so every surviving ``[n]`` maps to a real chunk, and the returned list
contains exactly the sources the final text points at.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from app.config import Settings, get_settings
from app.rag.prompts import (
    DOCUMENT_TASKS,
    SYSTEM_PROMPT,
    ContextSource,
    build_prompt,
    collect_injection_notes,
    sanitize_question,
)
from app.rag.reranker import Reranker, get_reranker
from app.rag.retriever import HybridRetriever, RetrievedCandidate, dedupe_candidates
from app.services.gemini import GeminiNotConfiguredError, get_chat_model, is_configured

logger = logging.getLogger(__name__)

_MARKER = re.compile(r"\[(\d{1,3})\]")
NOT_FOUND = "I could not find that in the uploaded documents."
_SNIPPET_CHARS = 600


@dataclass(slots=True)
class Citation:
    """A source the final answer points at."""

    marker: int
    document_id: str
    document_name: str
    page: int | None
    chunk_id: str
    chunk_index: int | None
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form used by the SSE payload and the DB column."""
        return {
            "marker": self.marker,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "snippet": self.snippet,
        }


@dataclass(slots=True)
class AnswerResult:
    """Everything the chain produced for one question."""

    answer: str
    task: str = "qa"
    citations: list[Citation] = field(default_factory=list)
    candidates: list[RetrievedCandidate] = field(default_factory=list)
    selected_ids: list[str] = field(default_factory=list)
    reranked: bool = False
    hybrid: bool = True
    injection_notes: list[str] = field(default_factory=list)
    retrieval_ms: int = 0
    rerank_ms: int = 0
    generation_ms: int = 0
    model: str = ""


def extract_markers(text: str) -> list[int]:
    """Return every ``[n]`` marker in ``text``, in order of first appearance."""
    seen: list[int] = []
    for match in _MARKER.finditer(text):
        value = int(match.group(1))
        if value not in seen:
            seen.append(value)
    return seen


def verify_citations(answer: str, sources: Sequence[ContextSource]) -> tuple[str, list[Citation]]:
    """Drop citation markers that do not map to a supplied source.

    Returns:
        ``(cleaned_answer, citations)``. Valid markers are kept verbatim so the
        frontend can turn them into chips; unmapped markers are removed, and
        the citation list holds only referenced sources in first-appearance
        order.
    """
    by_marker = {source.marker: source for source in sources}
    dropped: list[int] = []

    def _replace(match: re.Match[str]) -> str:
        value = int(match.group(1))
        if value in by_marker:
            return match.group(0)
        dropped.append(value)
        return ""

    cleaned = _MARKER.sub(_replace, answer)
    if dropped:
        # Tidy the whitespace/punctuation left behind by removed markers only;
        # a clean answer is returned byte-for-byte so Markdown survives intact.
        cleaned = re.sub(r"(?<=\S)[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
        logger.warning("Dropped hallucinated citation marker(s): %s", sorted(set(dropped)))
    cleaned = cleaned.strip()

    citations = [
        Citation(
            marker=marker,
            document_id=by_marker[marker].document_id,
            document_name=by_marker[marker].document_name,
            page=by_marker[marker].page,
            chunk_id=by_marker[marker].chunk_id,
            chunk_index=by_marker[marker].chunk_index,
            snippet=_snippet(by_marker[marker].text),
        )
        for marker in extract_markers(cleaned)
        if marker in by_marker
    ]
    return cleaned, citations


def _snippet(text: str, limit: int = _SNIPPET_CHARS) -> str:
    """Whitespace-normalised excerpt for the citation popover."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[:limit].rstrip()}..."


def to_history_messages(history: Sequence[tuple[str, str]]) -> list[BaseMessage]:
    """Convert ``(role, content)`` pairs from the database into LangChain messages."""
    messages: list[BaseMessage] = []
    for role, content in history:
        if not content.strip():
            continue
        messages.append(HumanMessage(content) if role == "user" else AIMessage(content))
    return messages


class RagChain:
    """Retrieve (or gather whole documents), generate with Gemini, verify citations."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._retriever = retriever or HybridRetriever(settings=self._settings)
        self._reranker = reranker or get_reranker()
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system}"),
                MessagesPlaceholder("history"),
                ("human", "{prompt}"),
            ]
        )

    # -- LCEL building blocks -----------------------------------------------------
    def generation_chain(self) -> Runnable:
        """``prompt | Gemini | StrOutputParser`` -- the streaming part of the flow."""
        return self._prompt | get_chat_model(self._settings) | StrOutputParser()

    @property
    def runnable(self) -> Runnable:
        """The complete RAG flow as one LCEL runnable (non-streaming).

        Input: ``{"session_id", "question", "task", "document_ids", "history",
        "sources"}`` where ``sources`` may be ``None`` (retrieve) or a
        pre-built list (whole-document tasks). Output: the input dict plus
        ``prepared`` and ``answer`` (an :class:`AnswerResult`).
        """
        prepare = RunnableLambda(self._prepare_step)
        render = RunnableLambda(self._render_step)
        generate = RunnableLambda(self._generate_step)
        return (
            RunnablePassthrough.assign(prepared=prepare)
            | RunnablePassthrough.assign(llm_input=render)
            | RunnablePassthrough.assign(answer=generate)
        )

    async def _prepare_step(
        self, inputs: dict[str, Any]
    ) -> tuple[list[ContextSource], AnswerResult]:
        return await self.prepare(
            inputs["session_id"],
            inputs["question"],
            task=inputs.get("task", "qa"),
            document_ids=inputs.get("document_ids"),
            sources=inputs.get("sources"),
        )

    def _render_step(self, inputs: dict[str, Any]) -> dict[str, Any]:
        sources, _ = inputs["prepared"]
        return {
            "system": SYSTEM_PROMPT,
            "history": to_history_messages(inputs.get("history") or []),
            "prompt": build_prompt(inputs.get("task", "qa"), inputs["question"], sources),
        }

    async def _generate_step(self, inputs: dict[str, Any]) -> AnswerResult:
        sources, result = inputs["prepared"]
        if not sources:
            result.answer = NOT_FOUND
            return result
        started = time.perf_counter()
        raw = await self.generation_chain().ainvoke(inputs["llm_input"])
        result.generation_ms = int((time.perf_counter() - started) * 1000)
        result.answer, result.citations = verify_citations(raw, sources)
        return result

    # -- context preparation ------------------------------------------------------------
    async def prepare(
        self,
        session_id: str,
        question: str,
        *,
        task: str = "qa",
        document_ids: Sequence[str] | None = None,
        sources: Sequence[ContextSource] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[ContextSource], AnswerResult]:
        """Build the numbered context sources for one turn.

        Whole-document tasks (summarize, eli5, key_points, glossary) pass
        ``sources`` gathered from the database in reading order. Questions go
        through hybrid retrieval (dense + BM25, RRF) and optional reranking.

        Raises:
            GeminiNotConfiguredError: If no API key is configured.
        """
        if not is_configured(self._settings):
            raise GeminiNotConfiguredError()

        result = AnswerResult(
            answer="",
            task=task,
            hybrid=self._settings.hybrid_enabled,
            model=self._settings.gemini_model,
        )

        if sources is not None or task in DOCUMENT_TASKS:
            chosen = list(sources or [])
            result.selected_ids = [source.chunk_id for source in chosen]
            result.hybrid = False
        else:
            query = sanitize_question(question)
            started = time.perf_counter()
            candidates = await self._retriever.retrieve(
                session_id, query, top_k=top_k, document_ids=document_ids
            )
            candidates = dedupe_candidates(candidates)
            result.retrieval_ms = int((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            limit = top_k or self._settings.retrieval_top_k
            selected, reranked = await self._reranker.rerank(query, candidates, top_n=limit)
            result.rerank_ms = int((time.perf_counter() - started) * 1000)
            result.reranked = reranked
            result.candidates = candidates
            result.selected_ids = [candidate.chunk_id for candidate in selected]
            chosen = [
                ContextSource(
                    marker=position,
                    document_id=candidate.document_id,
                    document_name=candidate.document_name,
                    page=candidate.page,
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    heading=candidate.heading,
                    chunk_index=_as_int(candidate.metadata.get("chunk_index")),
                )
                for position, candidate in enumerate(selected, start=1)
            ]

        result.injection_notes = collect_injection_notes(source.text for source in chosen)
        if result.injection_notes:
            logger.warning("Injection-like content in context: %s", result.injection_notes)
        return chosen, result

    # -- generation --------------------------------------------------------------------
    async def answer(
        self,
        session_id: str,
        question: str,
        *,
        task: str = "qa",
        document_ids: Sequence[str] | None = None,
        history: Sequence[tuple[str, str]] = (),
        sources: Sequence[ContextSource] | None = None,
    ) -> AnswerResult:
        """Produce a complete, citation-verified answer (non-streaming)."""
        output = await self.runnable.ainvoke(
            {
                "session_id": session_id,
                "question": question,
                "task": task,
                "document_ids": document_ids,
                "history": list(history),
                "sources": sources,
            }
        )
        return output["answer"]

    async def stream_answer(
        self,
        session_id: str,
        question: str,
        *,
        task: str = "qa",
        document_ids: Sequence[str] | None = None,
        history: Sequence[tuple[str, str]] = (),
        sources: Sequence[ContextSource] | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Stream the answer as ``(event_name, payload)`` pairs.

        Events, in order:

        * ``sources`` -- the numbered context passages, before any token;
        * ``token`` -- ``{"text": "..."}`` deltas;
        * ``citations`` -- the verified citation list plus the cleaned answer;
        * ``result`` -- the :class:`AnswerResult` (consumed by the API, not sent);
        * ``error`` -- terminal failure (no further events follow).
        """
        chosen, result = await self.prepare(
            session_id, question, task=task, document_ids=document_ids, sources=sources
        )
        yield "sources", self._sources_payload(result, chosen)

        if not chosen:
            result.answer = NOT_FOUND
            yield "token", {"text": NOT_FOUND}
            yield "citations", {"answer": NOT_FOUND, "citations": []}
            yield "result", result
            return

        llm_input = self._render_step(
            {"prepared": (chosen, result), "question": question, "task": task, "history": history}
        )
        buffer: list[str] = []
        started = time.perf_counter()
        try:
            async for delta in self.generation_chain().astream(llm_input):
                if delta:
                    buffer.append(delta)
                    yield "token", {"text": delta}
        except Exception as exc:  # noqa: BLE001 - reported to the client as an event
            logger.error("Streaming generation failed: %s", exc)
            yield (
                "error",
                {"message": "The model could not complete the answer.", "detail": str(exc)[:300]},
            )
            return

        result.generation_ms = int((time.perf_counter() - started) * 1000)
        raw = "".join(buffer)
        result.answer, result.citations = verify_citations(raw, chosen)
        yield (
            "citations",
            {
                "answer": result.answer,
                "citations": [citation.to_dict() for citation in result.citations],
            },
        )
        yield "result", result

    @staticmethod
    def _sources_payload(result: AnswerResult, sources: Sequence[ContextSource]) -> dict[str, Any]:
        """Payload of the ``sources`` SSE event (shown while the answer streams)."""
        return {
            "task": result.task,
            "hybrid": result.hybrid,
            "reranked": result.reranked,
            "retrieval_ms": result.retrieval_ms,
            "injection_flagged": bool(result.injection_notes),
            "injection_notes": result.injection_notes,
            "sources": [
                {
                    "marker": source.marker,
                    "document_id": source.document_id,
                    "document_name": source.document_name,
                    "page": source.page,
                    "chunk_id": source.chunk_id,
                    "chunk_index": source.chunk_index,
                    "snippet": _snippet(source.text, 240),
                }
                for source in sources
            ],
        }


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


_chain: RagChain | None = None


def get_rag_chain() -> RagChain:
    """Return the process-wide :class:`RagChain`."""
    global _chain
    if _chain is None:
        _chain = RagChain()
    return _chain


def set_rag_chain(chain: RagChain | None) -> None:
    """Replace the singleton (used by tests)."""
    global _chain
    _chain = chain
