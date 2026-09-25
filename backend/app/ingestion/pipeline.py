"""Ingestion orchestration: parse -> chunk -> embed + index (LangChain Chroma).

The pipeline is *idempotent*: re-ingesting a document deletes its existing
chunk rows and vectors before writing new ones, so a retry after a partial
failure cannot leave duplicated passages in the index.

It runs outside the request lifecycle (FastAPI ``BackgroundTasks``), owns its
own database sessions, and records progress on an ``IngestionJob`` row that the
frontend polls.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete

from app.config import Settings, get_settings
from app.db.models import Document, DocumentChunk, JobStatus, new_uuid
from app.db.session import session_scope
from app.ingestion.chunker import Chunk, chunk_pages
from app.ingestion.jobs import update_job
from app.ingestion.loaders import (
    DocumentParseError,
    UnsupportedDocumentError,
    load_document,
)
from app.rag.vectorstore import VectorRecord, VectorStore, VectorStoreError, get_vector_store
from app.services.gemini import GeminiCallError, GeminiNotConfiguredError, is_configured
from app.services.storage import StorageService
from app.services.storage import storage as default_storage

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionOutcome:
    """Summary of one pipeline run."""

    document_id: str
    ok: bool
    chunk_count: int = 0
    page_count: int = 0
    error: str | None = None


class IngestionPipeline:
    """Runs one document through the full ingestion flow."""

    def __init__(
        self,
        store: VectorStore | None = None,
        storage_service: StorageService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._store = store or get_vector_store()
        self._storage = storage_service or default_storage

    async def run(self, document_id: str, job_id: str) -> IngestionOutcome:
        """Ingest one document, updating its job row as it progresses.

        Never raises: every failure is recorded on the job and returned in the
        outcome, because the caller is a background task with nowhere to report an
        exception.
        """
        if not is_configured(self._settings):
            message = str(GeminiNotConfiguredError())
            await update_job(job_id, status=JobStatus.FAILED, error=message, progress=1.0)
            return IngestionOutcome(document_id, ok=False, error=message)

        try:
            context = await self._load_context(document_id)
        except LookupError as exc:
            await update_job(job_id, status=JobStatus.FAILED, error=str(exc))
            return IngestionOutcome(document_id, ok=False, error=str(exc))

        session_id, stored_name, mime, original_name = context
        try:
            path = self._storage.path_for(stored_name)
        except Exception as exc:  # noqa: BLE001
            message = f"Stored file is unavailable: {exc}"
            await update_job(job_id, status=JobStatus.FAILED, error=message)
            return IngestionOutcome(document_id, ok=False, error=message)

        try:
            pages = await self._parse(path, mime, job_id)
            chunks = await self._chunk(pages, job_id)
            if not chunks:
                raise DocumentParseError(
                    "No readable text was extracted. If this is a scanned PDF, "
                    "export the pages as images so Gemini vision OCR can run."
                )
            await self._persist(
                document_id=document_id,
                session_id=session_id,
                document_name=original_name,
                chunks=chunks,
                job_id=job_id,
            )
        except (
            UnsupportedDocumentError,
            DocumentParseError,
            GeminiNotConfiguredError,
            GeminiCallError,
            VectorStoreError,
        ) as exc:
            message = str(exc)
            logger.error("Ingestion failed for %s: %s", document_id, message)
            await update_job(job_id, status=JobStatus.FAILED, error=message, progress=1.0)
            await self._mark_document_counts(document_id, pages=0, chunks=0)
            return IngestionOutcome(document_id, ok=False, error=message)
        except Exception as exc:  # noqa: BLE001 - background task safety net
            logger.exception("Unexpected ingestion failure for %s", document_id)
            message = f"Unexpected error during ingestion: {exc}"
            await update_job(job_id, status=JobStatus.FAILED, error=message, progress=1.0)
            return IngestionOutcome(document_id, ok=False, error=message)

        await update_job(
            job_id,
            status=JobStatus.READY,
            progress=1.0,
            stage_detail=f"{len(chunks)} chunks indexed",
        )
        logger.info("Ingested %s: %d pages, %d chunks", original_name, len(pages), len(chunks))
        return IngestionOutcome(
            document_id, ok=True, chunk_count=len(chunks), page_count=len(pages)
        )

    # -- stages ------------------------------------------------------------
    async def _load_context(self, document_id: str) -> tuple[str, str, str, str]:
        """Fetch the fields the pipeline needs from the document row.

        Raises:
            LookupError: If the document no longer exists.
        """
        async with session_scope() as session:
            document = await session.get(Document, document_id)
            if document is None:
                raise LookupError(f"Document {document_id} no longer exists")
            return (
                document.session_id,
                document.stored_name,
                document.mime,
                document.original_name,
            )

    async def _parse(self, path: Path, mime: str, job_id: str):
        """Parse the file into pages."""
        await update_job(
            job_id, status=JobStatus.PARSING, progress=0.05, stage_detail="Extracting text"
        )
        pages = await load_document(path, mime)
        await update_job(job_id, progress=0.25, stage_detail=f"Parsed {len(pages)} page(s)")
        return pages

    async def _chunk(self, pages, job_id: str) -> list[Chunk]:
        """Split pages into chunks on a worker thread."""
        await update_job(
            job_id, status=JobStatus.CHUNKING, progress=0.3, stage_detail="Splitting text"
        )
        chunks = await asyncio.to_thread(
            chunk_pages,
            pages,
            size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        await update_job(job_id, progress=0.4, stage_detail=f"Created {len(chunks)} chunk(s)")
        return chunks

    async def _persist(
        self,
        *,
        document_id: str,
        session_id: str,
        document_name: str,
        chunks: list[Chunk],
        job_id: str,
    ) -> None:
        """Replace the document's chunk rows and vectors.

        Old vectors are deleted first so a crash mid-way leaves an
        under-indexed document (re-ingest fixes it) rather than an index
        containing passages whose rows are gone. Chunks are embedded and
        indexed in batches through LangChain's ``Chroma.add_texts`` so the job
        can report progress between batches.
        """
        await update_job(
            job_id,
            status=JobStatus.EMBEDDING,
            progress=0.45,
            stage_detail=f"Embedding {len(chunks)} chunk(s)",
        )
        await self._store.delete_document(session_id, document_id)

        chunk_ids = [new_uuid() for _ in chunks]
        async with session_scope() as session:
            await session.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )
            session.add_all(
                [
                    DocumentChunk(
                        id=chunk_ids[position],
                        document_id=document_id,
                        chunk_index=chunk.index,
                        page=chunk.page,
                        heading=chunk.heading,
                        content=chunk.content,
                        token_count=chunk.token_count,
                    )
                    for position, chunk in enumerate(chunks)
                ]
            )

        records = [
            VectorRecord(
                id=chunk_ids[position],
                text=chunk.content,
                metadata={
                    "document_id": document_id,
                    "document_name": document_name,
                    "session_id": session_id,
                    "chunk_index": chunk.index,
                    "page": chunk.page,
                    "heading": chunk.heading,
                },
            )
            for position, chunk in enumerate(chunks)
        ]
        batch_size = self._settings.embedding_batch_size
        for start in range(0, len(records), batch_size):
            await self._store.add(session_id, records[start : start + batch_size])
            done = min(start + batch_size, len(records))
            await update_job(
                job_id,
                progress=0.45 + 0.5 * (done / max(1, len(records))),
                stage_detail=f"Embedded {done}/{len(records)} chunk(s)",
            )

        async with session_scope() as session:
            document = await session.get(Document, document_id)
            if document is not None:
                document.chunk_count = len(chunks)
                document.page_count = len({chunk.page for chunk in chunks if chunk.page})

    async def _mark_document_counts(self, document_id: str, *, pages: int, chunks: int) -> None:
        """Reset counters after a failed run so the UI does not show stale totals."""
        try:
            async with session_scope() as session:
                document = await session.get(Document, document_id)
                if document is not None:
                    document.page_count = pages
                    document.chunk_count = chunks
        except Exception:  # noqa: BLE001
            logger.exception("Could not reset counters for %s", document_id)


async def ingest_document(document_id: str, job_id: str) -> IngestionOutcome:
    """Entry point used by ``BackgroundTasks``."""
    pipeline = IngestionPipeline()
    return await pipeline.run(document_id, job_id)


async def purge_document(session_id: str, document_id: str) -> None:
    """Remove a document's vectors. DB rows cascade from the ``documents`` row."""
    try:
        await get_vector_store().delete_document(session_id, document_id)
    except VectorStoreError as exc:
        logger.warning("Could not purge vectors for %s: %s", document_id, exc)
