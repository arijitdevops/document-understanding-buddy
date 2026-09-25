"""ORM models: chat sessions, their documents and chunks, ingestion jobs, messages.

Documents belong to the chat session they were uploaded into. That is what
makes "ask about the files in *this* conversation" work: retrieval is always
filtered to the current session's documents, and deleting a session removes its
files, vectors and history together.

Column types are portable: large text uses ``LONGTEXT`` on MySQL and plain
``TEXT`` elsewhere (SQLite in the test-suite).
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.mysql import DATETIME as MYSQL_DATETIME
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

#: ``TEXT`` everywhere, ``LONGTEXT`` (4 GB) on MySQL where ``TEXT`` caps at 64 KB.
LongText = Text().with_variant(LONGTEXT(), "mysql")

#: Microsecond timestamps on MySQL so messages written in the same second keep order.
PreciseDateTime = DateTime().with_variant(MYSQL_DATETIME(fsp=6), "mysql")


def utcnow() -> datetime:
    """Naive UTC timestamp (DATETIME columns are timezone-less)."""
    return datetime.now(UTC).replace(tzinfo=None)


def new_uuid() -> str:
    """Generate a hex UUID4 primary key."""
    return uuid.uuid4().hex


class DocumentStatus(str, enum.Enum):
    """Lifecycle of a document as ingestion progresses."""

    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


class JobStatus(str, enum.Enum):
    """Lifecycle of an ingestion job (mirrors :class:`DocumentStatus`)."""

    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


class MessageRole(str, enum.Enum):
    """Who produced a chat message."""

    USER = "user"
    ASSISTANT = "assistant"


class ChatSession(TimestampMixin, Base):
    """A conversation plus the documents uploaded into it."""

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="New chat")

    documents: Mapped[list[Document]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Document.created_at",
    )
    messages: Mapped[list[Message]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<ChatSession {self.id} {self.title!r}>"


class Document(TimestampMixin, Base):
    """An uploaded file plus its ingestion state."""

    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_session_status", "session_id", "status"),
        UniqueConstraint("session_id", "checksum", name="uq_documents_session_checksum"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    mime: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(
            DocumentStatus,
            native_enum=False,
            length=16,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        default=DocumentStatus.PENDING,
    )
    error: Mapped[str | None] = mapped_column(Text, default=None)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)

    session: Mapped[ChatSession] = relationship(back_populates="documents")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    jobs: Mapped[list[IngestionJob]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Document {self.original_name!r} status={self.status.value}>"


class DocumentChunk(Base):
    """One retrievable slice of a document, with its page and heading trail."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_document_index"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer, default=None)
    heading: Mapped[str | None] = mapped_column(String(500), default=None)
    content: Mapped[str] = mapped_column(LongText, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DocumentChunk doc={self.document_id} idx={self.chunk_index}>"


class IngestionJob(TimestampMixin, Base):
    """Progress record for one ingestion run of one document."""

    __tablename__ = "ingestion_jobs"
    __table_args__ = (Index("ix_ingestion_jobs_document_status", "document_id", "status"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(
            JobStatus, native_enum=False, length=16, values_callable=lambda e: [m.value for m in e]
        ),
        nullable=False,
        default=JobStatus.PENDING,
    )
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stage_detail: Mapped[str | None] = mapped_column(String(255), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    document: Mapped[Document] = relationship(back_populates="jobs")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<IngestionJob {self.id} {self.status.value} {self.progress:.0%}>"


class Message(Base):
    """A single chat turn. Assistant turns carry structured citations."""

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_session_created", "session_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(
        SAEnum(
            MessageRole,
            native_enum=False,
            length=16,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(LongText, nullable=False)
    task: Mapped[str] = mapped_column(String(32), nullable=False, default="qa")
    citations: Mapped[list[dict] | None] = mapped_column(JSON, default=None)
    document_ids: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(PreciseDateTime, default=utcnow, nullable=False)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class QueryLog(Base):
    """Per-question telemetry: latency per stage and what was retrieved."""

    __tablename__ = "query_logs"
    __table_args__ = (Index("ix_query_logs_created", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    session_id: Mapped[str | None] = mapped_column(String(32), default=None)
    task: Mapped[str] = mapped_column(String(32), nullable=False, default="qa")
    question: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(80), nullable=False)
    retrieved_chunk_ids: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    retrieval_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rerank_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    generation_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    citation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    injection_flagged: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
