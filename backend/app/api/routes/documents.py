"""Document endpoints: upload into a chat, list, status, re-ingest, delete, chunks."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, File, Response, UploadFile, status
from sqlalchemy import select

from app.api.deps import DbSession, document_out, documents_out, get_chat_session, get_document
from app.api.errors import ApiError, not_found
from app.db.models import Document, DocumentChunk, DocumentStatus, utcnow
from app.ingestion.jobs import create_job, latest_job
from app.ingestion.pipeline import ingest_document, purge_document
from app.schemas import DocumentOut, ErrorResponse, IngestionJobOut, UploadResult
from app.services.storage import StorageError, storage

logger = logging.getLogger(__name__)
router = APIRouter(tags=["documents"])

MAX_FILES_PER_UPLOAD = 10
_BUSY = {
    DocumentStatus.PENDING,
    DocumentStatus.PARSING,
    DocumentStatus.CHUNKING,
    DocumentStatus.EMBEDDING,
}


@router.get("/sessions/{session_id}/documents", response_model=list[DocumentOut])
async def list_documents(session_id: str, db: DbSession) -> list[DocumentOut]:
    """Documents uploaded into a chat, with ingestion status and progress."""
    await get_chat_session(db, session_id)
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
    return await documents_out(db, list(documents))


@router.post(
    "/sessions/{session_id}/documents",
    response_model=UploadResult,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_documents(
    session_id: str,
    db: DbSession,
    background: BackgroundTasks,
    files: list[UploadFile] = File(..., description="One or more files (multipart field 'files')."),
) -> UploadResult:
    """Upload files into a chat. Ingestion runs in the background.

    Poll ``GET /api/sessions/{id}/documents`` (or ``GET /api/documents/{id}``)
    to follow the status: ``pending -> parsing -> chunking -> embedding ->
    ready`` (or ``failed`` with an ``error``).
    """
    chat = await get_chat_session(db, session_id)
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise ApiError(
            400, f"Upload at most {MAX_FILES_PER_UPLOAD} files at a time.", "too_many_files"
        )

    accepted: list[tuple[Document, str]] = []
    duplicates: list[str] = []
    rejected: list[ErrorResponse] = []
    for upload in files:
        try:
            stored = await storage.save_upload(upload)
        except StorageError as exc:
            rejected.append(ErrorResponse(detail=f"{upload.filename}: {exc}", code=exc.code))
            continue

        existing = (
            await db.execute(
                select(Document).where(
                    Document.session_id == session_id, Document.checksum == stored.checksum
                )
            )
        ).scalar_one_or_none()
        if existing is not None or any(doc.checksum == stored.checksum for doc, _ in accepted):
            storage.delete(stored.stored_name)
            duplicates.append(stored.original_name)
            continue

        document = Document(
            session_id=chat.id,
            original_name=stored.original_name,
            stored_name=stored.stored_name,
            mime=stored.mime,
            size_bytes=stored.size_bytes,
            checksum=stored.checksum,
            status=DocumentStatus.PENDING,
        )
        db.add(document)
        await db.flush()
        job = await create_job(db, document.id)
        accepted.append((document, job.id))

    if not accepted and not duplicates and rejected:
        raise ApiError(400, rejected[0].detail, rejected[0].code)

    chat.updated_at = utcnow()  # moves the chat to the top of the sidebar
    await db.commit()
    for document, job_id in accepted:
        background.add_task(ingest_document, document.id, job_id)
        logger.info("Queued ingestion of %s (%s)", document.original_name, document.id)

    for document, _ in accepted:
        await db.refresh(document)
    return UploadResult(
        documents=[document_out(document, None) for document, _ in accepted],
        duplicates=duplicates,
        rejected=rejected,
    )


@router.get("/documents/{document_id}", response_model=DocumentOut)
async def read_document(document_id: str, db: DbSession) -> DocumentOut:
    """One document with its ingestion status."""
    document = await get_document(db, document_id)
    return document_out(document, await latest_job(db, document_id))


@router.get("/documents/{document_id}/job", response_model=IngestionJobOut)
async def read_job(document_id: str, db: DbSession) -> IngestionJobOut:
    """The latest ingestion job of a document (progress 0..1 and stage)."""
    await get_document(db, document_id)
    job = await latest_job(db, document_id)
    if job is None:
        raise not_found("Ingestion job")
    return IngestionJobOut(
        id=job.id,
        document_id=job.document_id,
        status=job.status.value,
        progress=job.progress,
        stage_detail=job.stage_detail,
        error=job.error,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.post(
    "/documents/{document_id}/reingest",
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reingest_document(
    document_id: str, db: DbSession, background: BackgroundTasks
) -> DocumentOut:
    """Run ingestion again (for example after adding GEMINI_API_KEY)."""
    document = await get_document(db, document_id)
    if document.status in _BUSY:
        raise ApiError(409, "This document is still being processed.", "document_busy")
    document.status = DocumentStatus.PENDING
    document.error = None
    job = await create_job(db, document.id)
    await db.commit()
    await db.refresh(document)
    background.add_task(ingest_document, document.id, job.id)
    return document_out(document, job)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, db: DbSession) -> Response:
    """Delete a document, its chunks, its vectors and the stored file."""
    document = await get_document(db, document_id)
    session_id, stored_name = document.session_id, document.stored_name
    await db.delete(document)
    await db.commit()
    await purge_document(session_id, document_id)
    storage.delete(stored_name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/chunks/{chunk_id}")
async def read_chunk(chunk_id: str, db: DbSession) -> dict:
    """Full text of one chunk -- used by the citation popover's "show more"."""
    chunk = await db.get(DocumentChunk, chunk_id)
    if chunk is None:
        raise not_found("Chunk")
    document = await db.get(Document, chunk.document_id)
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "document_name": document.original_name if document else "",
        "chunk_index": chunk.chunk_index,
        "page": chunk.page,
        "heading": chunk.heading,
        "content": chunk.content,
    }
