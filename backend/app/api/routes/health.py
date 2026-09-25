"""``GET /api/health`` -- dependency status and missing configuration."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.config import get_settings
from app.db.session import ping_database
from app.rag.vectorstore import get_vector_store
from app.schemas import ComponentHealth, HealthResponse
from app.services import gemini

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report database, vector store and Gemini status.

    Always returns HTTP 200 so it can be polled by the UI; ``status`` is
    ``degraded`` when anything is wrong, and ``missing_config`` lists the
    *names* of absent settings (for example ``GEMINI_API_KEY``).
    """
    settings = get_settings()
    db_ok, db_error = await ping_database()
    vs_ok, vs_detail = await get_vector_store().health()
    llm_ok, llm_detail = gemini.health(settings)
    missing = [] if gemini.is_configured(settings) else settings.missing_requirements()
    components = [
        ComponentHealth(name="database", ok=db_ok, detail=db_error or "connected"),
        ComponentHealth(name="vector_store", ok=vs_ok, detail=vs_detail),
        ComponentHealth(name="gemini", ok=llm_ok, detail=llm_detail),
    ]
    return HealthResponse(
        status="ok" if all(c.ok for c in components) else "degraded",
        version=__version__,
        environment=settings.environment,
        model=settings.gemini_model,
        embedding_model=settings.gemini_embedding_model,
        components=components,
        missing_config=missing,
    )
