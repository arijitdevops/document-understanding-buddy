"""Shared route dependencies and lookup helpers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import ApiError, not_found
from app.db.models import ChatSession, Document, IngestionJob
from app.db.session import get_session
from app.schemas import DocumentOut
from app.services.rate_limit import rate_limiter

DbSession = Annotated[AsyncSession, Depends(get_session)]


async def get_chat_session(db: AsyncSession, session_id: str) -> ChatSession:
    """Load a chat session or raise 404."""
    chat = await db.get(ChatSession, session_id)
    if chat is None:
        raise not_found("Chat session")
    return chat


async def get_document(db: AsyncSession, document_id: str) -> Document:
    """Load a document or raise 404."""
    document = await db.get(Document, document_id)
    if document is None:
        raise not_found("Document")
    return document


async def latest_jobs(db: AsyncSession, document_ids: list[str]) -> dict[str, IngestionJob]:
    """Most recent ingestion job per document."""
    if not document_ids:
        return {}
    rows = (
        (
            await db.execute(
                select(IngestionJob)
                .where(IngestionJob.document_id.in_(document_ids))
                .order_by(IngestionJob.created_at)
            )
        )
        .scalars()
        .all()
    )
    latest: dict[str, IngestionJob] = {}
    for job in rows:  # later rows overwrite earlier ones
        latest[job.document_id] = job
    return latest


def document_out(document: Document, job: IngestionJob | None) -> DocumentOut:
    """Serialise a document together with its latest job's progress."""
    return DocumentOut(
        id=document.id,
        session_id=document.session_id,
        original_name=document.original_name,
        mime=document.mime,
        size_bytes=document.size_bytes,
        page_count=document.page_count,
        chunk_count=document.chunk_count,
        status=document.status.value,
        progress=job.progress if job else 0.0,
        stage_detail=job.stage_detail if job else None,
        error=document.error,
        created_at=document.created_at,
    )


async def documents_out(db: AsyncSession, documents: list[Document]) -> list[DocumentOut]:
    """Serialise several documents with one job query."""
    jobs = await latest_jobs(db, [document.id for document in documents])
    return [document_out(document, jobs.get(document.id)) for document in documents]


async def enforce_rate_limit(request: Request) -> None:
    """Per-client sliding-window limit for expensive (LLM) endpoints."""
    key = request.client.host if request.client else "anonymous"
    allowed, _, retry_after = await rate_limiter.check(key)
    if not allowed:
        raise ApiError(
            429,
            f"Too many requests. Try again in {int(retry_after) + 1} seconds.",
            code="rate_limited",
        )
