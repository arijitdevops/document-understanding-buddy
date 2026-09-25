"""Request and response schemas.

These mirror the payloads documented in the README's API reference, so the
documented shapes and the wire format cannot drift apart.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Task = Literal["qa", "summarize", "eli5", "key_points", "glossary", "compare"]


# ---- common ------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    """Uniform error body returned by every handled failure."""

    detail: str = Field(description="Human-readable explanation.")
    code: str = Field(default="error", description="Stable machine-readable code.")


class ComponentHealth(BaseModel):
    """Health of a single dependency."""

    name: str
    ok: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    """Aggregate health payload for ``GET /api/health``."""

    status: Literal["ok", "degraded"]
    version: str
    environment: str
    model: str
    embedding_model: str
    components: list[ComponentHealth]
    missing_config: list[str] = Field(
        default_factory=list,
        description="Names of absent required settings. Values are never returned.",
    )


# ---- documents ------------------------------------------------------------------------
class DocumentOut(BaseModel):
    """A document row plus its latest ingestion progress."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    original_name: str
    mime: str
    size_bytes: int
    page_count: int
    chunk_count: int
    status: str
    progress: float = 0.0
    stage_detail: str | None = None
    error: str | None = None
    created_at: datetime


class IngestionJobOut(BaseModel):
    """Ingestion progress for one document."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    stage_detail: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class UploadResult(BaseModel):
    """Body of ``POST /api/sessions/{id}/documents`` (HTTP 202)."""

    documents: list[DocumentOut]
    duplicates: list[str] = Field(
        default_factory=list,
        description="Names of files already present in this chat (identical content).",
    )
    rejected: list[ErrorResponse] = Field(
        default_factory=list, description="Files that were refused, with the reason."
    )


# ---- sessions & messages -----------------------------------------------------------
class SessionCreate(BaseModel):
    """Body for ``POST /api/sessions``."""

    title: str | None = Field(default=None, max_length=200)


class SessionUpdate(BaseModel):
    """Body for ``PATCH /api/sessions/{id}``."""

    title: str = Field(min_length=1, max_length=200)


class SessionOut(BaseModel):
    """A chat session in the sidebar list."""

    id: str
    title: str
    document_count: int = 0
    message_count: int = 0
    created_at: datetime
    updated_at: datetime


class CitationOut(BaseModel):
    """A source the answer points at with an inline ``[n]`` marker."""

    marker: int = Field(ge=1, description="Matches the [n] in the answer text.")
    document_id: str
    document_name: str
    page: int | None = None
    chunk_id: str
    chunk_index: int | None = None
    snippet: str


class MessageOut(BaseModel):
    """A persisted chat message."""

    id: str
    role: Literal["user", "assistant"]
    content: str
    task: str = "qa"
    citations: list[CitationOut] = Field(default_factory=list)
    document_ids: list[str] | None = None
    created_at: datetime


class SessionDetail(SessionOut):
    """A session with its documents and full message history."""

    documents: list[DocumentOut]
    messages: list[MessageOut]


class ChatRequest(BaseModel):
    """Body for ``POST /api/sessions/{id}/chat`` (streams Server-Sent Events)."""

    message: str = Field(default="", max_length=4000)
    task: Task = Field(
        default="qa",
        description=(
            "qa = answer a question with retrieval; summarize, eli5 (explain like I'm "
            "new), key_points, glossary and compare read the whole selected documents."
        ),
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Restrict to these documents of the chat. Omit to use all ready documents.",
    )
