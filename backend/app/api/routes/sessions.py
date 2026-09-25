"""Chat session endpoints: list, create, read (with history), rename, delete."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response, status
from sqlalchemy import func, select

from app.api.deps import DbSession, documents_out, get_chat_session
from app.db.models import ChatSession, Document, Message
from app.rag.vectorstore import get_vector_store
from app.schemas import (
    CitationOut,
    MessageOut,
    SessionCreate,
    SessionDetail,
    SessionOut,
    SessionUpdate,
)
from app.services.storage import storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])


def message_out(message: Message) -> MessageOut:
    """Serialise a message row."""
    return MessageOut(
        id=message.id,
        role=message.role.value,
        content=message.content,
        task=message.task,
        citations=[CitationOut(**item) for item in (message.citations or [])],
        document_ids=message.document_ids,
        created_at=message.created_at,
    )


async def _counts(db: DbSession, session_ids: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    if not session_ids:
        return {}, {}
    documents = dict(
        (
            await db.execute(
                select(Document.session_id, func.count(Document.id))
                .where(Document.session_id.in_(session_ids))
                .group_by(Document.session_id)
            )
        ).all()
    )
    messages = dict(
        (
            await db.execute(
                select(Message.session_id, func.count(Message.id))
                .where(Message.session_id.in_(session_ids))
                .group_by(Message.session_id)
            )
        ).all()
    )
    return documents, messages


@router.get("", response_model=list[SessionOut])
async def list_sessions(db: DbSession) -> list[SessionOut]:
    """All chat sessions, most recently active first."""
    sessions = (
        (await db.execute(select(ChatSession).order_by(ChatSession.updated_at.desc())))
        .scalars()
        .all()
    )
    documents, messages = await _counts(db, [chat.id for chat in sessions])
    return [
        SessionOut(
            id=chat.id,
            title=chat.title,
            document_count=documents.get(chat.id, 0),
            message_count=messages.get(chat.id, 0),
            created_at=chat.created_at,
            updated_at=chat.updated_at,
        )
        for chat in sessions
    ]


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(db: DbSession, body: SessionCreate | None = None) -> SessionOut:
    """Start a new, empty chat."""
    chat = ChatSession(title=(body.title.strip() if body and body.title else None) or "New chat")
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return SessionOut(
        id=chat.id, title=chat.title, created_at=chat.created_at, updated_at=chat.updated_at
    )


@router.get("/{session_id}", response_model=SessionDetail)
async def read_session(session_id: str, db: DbSession) -> SessionDetail:
    """A chat with its documents (and ingestion status) and full message history."""
    chat = await get_chat_session(db, session_id)
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
    messages = (
        (
            await db.execute(
                select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
            )
        )
        .scalars()
        .all()
    )
    return SessionDetail(
        id=chat.id,
        title=chat.title,
        document_count=len(documents),
        message_count=len(messages),
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        documents=await documents_out(db, list(documents)),
        messages=[message_out(message) for message in messages],
    )


@router.get("/{session_id}/messages", response_model=list[MessageOut])
async def list_messages(session_id: str, db: DbSession) -> list[MessageOut]:
    """Conversation history of one chat, oldest first."""
    await get_chat_session(db, session_id)
    messages = (
        (
            await db.execute(
                select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
            )
        )
        .scalars()
        .all()
    )
    return [message_out(message) for message in messages]


@router.patch("/{session_id}", response_model=SessionOut)
async def rename_session(session_id: str, body: SessionUpdate, db: DbSession) -> SessionOut:
    """Rename a chat."""
    chat = await get_chat_session(db, session_id)
    chat.title = body.title.strip()
    await db.commit()
    await db.refresh(chat)
    documents, messages = await _counts(db, [chat.id])
    return SessionOut(
        id=chat.id,
        title=chat.title,
        document_count=documents.get(chat.id, 0),
        message_count=messages.get(chat.id, 0),
        created_at=chat.created_at,
        updated_at=chat.updated_at,
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, db: DbSession) -> Response:
    """Delete a chat together with its documents, vectors, files and history."""
    chat = await get_chat_session(db, session_id)
    stored_names = (
        (await db.execute(select(Document.stored_name).where(Document.session_id == session_id)))
        .scalars()
        .all()
    )
    await db.delete(chat)
    await db.commit()
    await get_vector_store().drop_session(session_id)
    for name in stored_names:
        storage.delete(name)
    logger.info("Deleted session %s (%d documents)", session_id, len(stored_names))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
