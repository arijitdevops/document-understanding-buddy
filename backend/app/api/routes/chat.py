"""``POST /api/sessions/{id}/chat`` -- grounded answers streamed as Server-Sent Events.

Event stream (each event is ``event: <name>`` + ``data: <json>``):

``meta``       ``{"session_id", "user_message_id", "task", "title"}``
``sources``    numbered context passages, sent before the first token
``token``      ``{"text": "..."}`` answer deltas
``citations``  ``{"answer", "citations": [...]}`` verified citations + final text
``done``       ``{"message_id", "generation_ms", "retrieval_ms"}``
``error``      ``{"message", "detail"}`` -- terminal
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.deps import DbSession, enforce_rate_limit, get_chat_session
from app.api.errors import ApiError
from app.config import get_settings
from app.db.models import Document, DocumentStatus, Message, MessageRole, QueryLog, utcnow
from app.db.session import session_scope
from app.rag.chain import AnswerResult, get_rag_chain
from app.rag.document_context import document_sources
from app.rag.prompts import DOCUMENT_TASKS, TASK_DEFAULT_MESSAGES
from app.schemas import ChatRequest
from app.services.gemini import GeminiNotConfiguredError, is_configured

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

_TITLE_CHARS = 60


def sse(event: str, data: Any) -> str:
    """Encode one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post(
    "/sessions/{session_id}/chat",
    dependencies=[Depends(enforce_rate_limit)],
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def chat(session_id: str, body: ChatRequest, db: DbSession) -> StreamingResponse:
    """Ask a question (or run a document task) and stream the grounded answer."""
    settings = get_settings()
    chat_session = await get_chat_session(db, session_id)
    if not is_configured(settings):
        raise GeminiNotConfiguredError()

    task = body.task
    text = body.message.strip()
    if not text:
        if task not in TASK_DEFAULT_MESSAGES:
            raise ApiError(422, "Please type a question.", "empty_message")
        text = TASK_DEFAULT_MESSAGES[task]

    documents = (
        (
            await db.execute(
                select(Document)
                .where(Document.session_id == session_id)
                .order_by(Document.created_at)
            )
        )
        .scalars()
        .all()
    )
    if body.document_ids:
        wanted = set(body.document_ids)
        unknown = wanted - {document.id for document in documents}
        if unknown:
            raise ApiError(400, "Some document_ids do not belong to this chat.", "unknown_document")
        documents = [document for document in documents if document.id in wanted]
    ready = [document for document in documents if document.status is DocumentStatus.READY]
    if not ready:
        if any(document.status is not DocumentStatus.FAILED for document in documents):
            raise ApiError(
                409,
                "Your documents are still being processed. Try again in a moment.",
                "documents_processing",
            )
        raise ApiError(
            400,
            "No ready documents in this chat. Upload a document (or re-ingest a failed one) first.",
            "no_documents",
        )
    ready_ids = [document.id for document in ready]

    history_rows = (
        (
            await db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at.desc())
                .limit(settings.history_messages)
            )
        )
        .scalars()
        .all()
        if settings.history_messages
        else []
    )
    history = [(row.role.value, row.content) for row in reversed(history_rows)]

    user_message = Message(
        session_id=session_id,
        role=MessageRole.USER,
        content=text,
        task=task,
        document_ids=body.document_ids,
    )
    db.add(user_message)
    if chat_session.title == "New chat" and not history:
        chat_session.title = (
            text if len(text) <= _TITLE_CHARS else text[: _TITLE_CHARS - 3].rstrip() + "..."
        )
    chat_session.updated_at = utcnow()
    await db.commit()

    sources = None
    if task in DOCUMENT_TASKS:
        sources = await document_sources(db, ready, max_chars=settings.document_task_max_chars)

    meta = {
        "session_id": session_id,
        "user_message_id": user_message.id,
        "task": task,
        "title": chat_session.title,
    }

    async def event_stream() -> AsyncIterator[str]:
        yield sse("meta", meta)
        result: AnswerResult | None = None
        try:
            async for event, payload in get_rag_chain().stream_answer(
                session_id,
                text,
                task=task,
                document_ids=ready_ids,
                history=history,
                sources=sources,
            ):
                if event == "result":
                    result = payload
                    continue
                yield sse(event, payload)
                if event == "error":
                    return
        except Exception as exc:  # noqa: BLE001 - the stream must end with an event
            logger.exception("Chat failed for session %s", session_id)
            yield sse(
                "error",
                {"message": "Something went wrong while answering.", "detail": str(exc)[:300]},
            )
            return

        if result is None:
            return
        message_id = await _persist_answer(session_id, text, task, body.document_ids, result)
        yield sse(
            "done",
            {
                "message_id": message_id,
                "retrieval_ms": result.retrieval_ms,
                "generation_ms": result.generation_ms,
                "model": result.model,
            },
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _persist_answer(
    session_id: str,
    question: str,
    task: str,
    document_ids: list[str] | None,
    result: AnswerResult,
) -> str | None:
    """Store the assistant message and a query-log row; returns the message id."""
    try:
        async with session_scope() as db:
            message = Message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=result.answer,
                task=task,
                citations=[citation.to_dict() for citation in result.citations],
                document_ids=document_ids,
            )
            db.add(message)
            db.add(
                QueryLog(
                    session_id=session_id,
                    task=task,
                    question=question,
                    model=result.model,
                    retrieved_chunk_ids=result.selected_ids[:100],
                    retrieval_ms=result.retrieval_ms,
                    rerank_ms=result.rerank_ms,
                    generation_ms=result.generation_ms,
                    citation_count=len(result.citations),
                    injection_flagged=int(bool(result.injection_notes)),
                )
            )
            await db.flush()
            return message.id
    except Exception:  # noqa: BLE001 - the answer was already streamed
        logger.exception("Could not persist the answer for session %s", session_id)
        return None
