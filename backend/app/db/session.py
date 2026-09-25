"""Async SQLAlchemy engine and session helpers.

The engine is created lazily so importing the app (for tests or for
``alembic``) never opens a connection. ``get_session`` is the FastAPI
dependency; ``session_scope`` is the equivalent for background tasks that run
outside the request lifecycle.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_tables_ready = False


def _engine_kwargs(url: str) -> dict:
    """Pool options per backend.

    MySQL gets a sized, pre-pinged, recycled pool (MySQL drops idle
    connections after ``wait_timeout``). SQLite -- used by the tests and handy
    for a quick local run -- does not accept pool sizing arguments.
    """
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {
        "pool_pre_ping": True,
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_recycle": 1800,
    }


def get_engine() -> AsyncEngine:
    """Return (creating on first call) the process-wide async engine."""
    global _engine
    if _engine is None:
        logger.info("Creating async database engine")
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.db_echo,
            **_engine_kwargs(settings.database_url),
        )
        if settings.database_url.startswith("sqlite"):
            # SQLite ignores ON DELETE CASCADE unless foreign keys are enabled
            # per connection; MySQL/InnoDB enforces them natively.
            @event.listens_for(_engine.sync_engine, "connect")
            def _enable_sqlite_fks(dbapi_connection, _record) -> None:  # pragma: no cover
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    return _engine


async def init_models() -> None:
    """Create any missing tables (``CREATE TABLE IF NOT EXISTS`` semantics).

    Controlled by ``AUTO_CREATE_TABLES``. For MySQL you can instead apply
    ``backend/schema.sql`` yourself; both produce the same schema.
    """
    from app.db import models  # noqa: F401 - registers the tables on Base.metadata
    from app.db.base import Base

    global _tables_ready
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _tables_ready = True
    logger.info("Database tables verified")


async def _ensure_tables() -> None:
    """Create tables lazily if the database was unreachable at startup."""
    if _tables_ready or not settings.auto_create_tables:
        return
    try:
        await init_models()
    except Exception as exc:  # noqa: BLE001 - the request itself will report the error
        logger.warning("Database still unavailable: %s", exc)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the session factory bound to the shared engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session that rolls back on error."""
    await _ensure_tables()
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except SQLAlchemyError:
            await session.rollback()
            logger.exception("Database error; transaction rolled back")
            raise


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Context manager for background work: commits on success, rolls back on error."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception("Database error in background session; rolled back")
            raise


async def ping_database() -> tuple[bool, str | None]:
    """Run ``SELECT 1``.

    Returns:
        ``(ok, error_message)``. The error message is safe to surface because
        it is truncated and never contains the connection URL.
    """
    from sqlalchemy import text

    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:  # noqa: BLE001 - health check must never raise
        logger.warning("Database health check failed: %s", exc)
        return False, str(exc)[:200]


async def dispose_engine() -> None:
    """Close pooled connections on shutdown."""
    global _engine, _session_factory, _tables_ready
    _tables_ready = False
    if _engine is not None:
        await _engine.dispose()
        logger.info("Database engine disposed")
    _engine = None
    _session_factory = None
