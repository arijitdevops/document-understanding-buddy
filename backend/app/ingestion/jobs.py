"""Helpers for reading and updating ingestion job rows.

Each update runs in its own short transaction so progress is visible to the
polling frontend while the pipeline is still working.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentStatus, IngestionJob, JobStatus
from app.db.session import session_scope

logger = logging.getLogger(__name__)

#: Job status -> document status. The two vocabularies are intentionally identical.
_DOCUMENT_STATUS = {
    JobStatus.PENDING: DocumentStatus.PENDING,
    JobStatus.PARSING: DocumentStatus.PARSING,
    JobStatus.CHUNKING: DocumentStatus.CHUNKING,
    JobStatus.EMBEDDING: DocumentStatus.EMBEDDING,
    JobStatus.READY: DocumentStatus.READY,
    JobStatus.FAILED: DocumentStatus.FAILED,
}


def _utcnow() -> datetime:
    """Naive UTC timestamp (MySQL DATETIME columns are timezone-less here)."""
    return datetime.now(UTC).replace(tzinfo=None)


async def create_job(session: AsyncSession, document_id: str) -> IngestionJob:
    """Create a ``pending`` job for ``document_id`` and flush it."""
    job = IngestionJob(document_id=document_id, status=JobStatus.PENDING, progress=0.0)
    session.add(job)
    await session.flush()
    return job


async def latest_job(session: AsyncSession, document_id: str) -> IngestionJob | None:
    """Return the most recent job for a document, if any."""
    result = await session.execute(
        select(IngestionJob)
        .where(IngestionJob.document_id == document_id)
        .order_by(IngestionJob.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def update_job(
    job_id: str,
    *,
    status: JobStatus | None = None,
    progress: float | None = None,
    stage_detail: str | None = None,
    error: str | None = None,
) -> None:
    """Update a job row (and mirror the status onto its document).

    Failures here are logged and swallowed: losing a progress update must not
    abort an otherwise healthy ingestion run.
    """
    try:
        async with session_scope() as session:
            job = await session.get(IngestionJob, job_id)
            if job is None:
                logger.warning("Ingestion job %s vanished mid-run", job_id)
                return
            if status is not None:
                job.status = status
                if status is JobStatus.PARSING and job.started_at is None:
                    job.started_at = _utcnow()
                if status in (JobStatus.READY, JobStatus.FAILED):
                    job.finished_at = _utcnow()
            if progress is not None:
                job.progress = max(0.0, min(1.0, progress))
            if stage_detail is not None:
                job.stage_detail = stage_detail[:255]
            if error is not None:
                job.error = error[:2000]

            document = await session.get(Document, job.document_id)
            if document is not None and status is not None:
                document.status = _DOCUMENT_STATUS[status]
                document.error = error[:2000] if error else None
    except Exception:  # noqa: BLE001 - progress reporting is best-effort
        logger.exception("Could not update ingestion job %s", job_id)
