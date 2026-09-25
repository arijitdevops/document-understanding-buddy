"""Whole-document context for summarise / explain / key-points / glossary tasks.

Similarity search is the wrong tool for "summarise this document": the
question has no topic, so retrieval returns an arbitrary handful of chunks.
These tasks instead read the stored chunks in reading order. When the
documents are longer than ``DOCUMENT_TASK_MAX_CHARS``, each document gets an
equal share of the budget and chunks are sampled evenly from beginning to end,
so the model always sees the introduction, the middle and the conclusion.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentChunk
from app.rag.prompts import ContextSource


def sample_evenly(items: Sequence, count: int) -> list:
    """Pick ``count`` items spread evenly across ``items`` (keeps first and last)."""
    if count >= len(items):
        return list(items)
    if count <= 0:
        return []
    if count == 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(index * step)] for index in range(count)]


async def document_sources(
    session: AsyncSession,
    documents: Sequence[Document],
    *,
    max_chars: int,
) -> list[ContextSource]:
    """Return numbered context sources covering ``documents`` within ``max_chars``."""
    if not documents:
        return []
    share = max(1000, max_chars // len(documents))
    sources: list[ContextSource] = []
    for document in documents:
        rows = (
            (
                await session.execute(
                    select(DocumentChunk)
                    .where(DocumentChunk.document_id == document.id)
                    .order_by(DocumentChunk.chunk_index)
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            continue
        total = sum(len(row.content) for row in rows)
        if total > share:
            average = max(1, total // len(rows))
            rows = sample_evenly(rows, max(1, share // average))
        for row in rows:
            sources.append(
                ContextSource(
                    marker=len(sources) + 1,
                    document_id=document.id,
                    document_name=document.original_name,
                    page=row.page,
                    chunk_id=row.id,
                    text=row.content,
                    heading=row.heading,
                    chunk_index=row.chunk_index,
                )
            )
    return sources
