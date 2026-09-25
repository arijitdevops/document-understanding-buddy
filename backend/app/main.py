"""FastAPI application factory.

Run locally with::

    uvicorn app.main:app --reload --port 8000

The app boots without ``GEMINI_API_KEY`` or a reachable database: startup
problems are logged, ``GET /api/health`` reports them, and routes that need
the missing piece fail with a clear JSON error instead of a stack trace.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import api_router
from app.api.errors import install_error_handlers
from app.config import get_settings
from app.db.session import dispose_engine, init_models
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Create tables on startup (if enabled) and release the pool on shutdown."""
    settings = get_settings()
    if settings.auto_create_tables:
        try:
            await init_models()
        except Exception as exc:  # noqa: BLE001 - keep serving; /api/health shows it
            logger.error(
                "Could not initialise the database (%s). Check DATABASE_URL and that "
                "MySQL is running; the API will keep retrying on each request.",
                exc,
            )
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.environment)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Upload documents into a chat and ask questions about them. Answers are "
            "grounded in the documents (RAG with Gemini + LangChain + Chroma) and cite "
            "the passages they come from."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_error_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "docs": "/docs",
            "health": f"{settings.api_prefix}/health",
        }

    return app


app = create_app()
