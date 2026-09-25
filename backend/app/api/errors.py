"""Uniform error responses: every handled failure returns ``{"detail", "code"}``."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.rag.vectorstore import VectorStoreError
from app.services.gemini import GeminiNotConfiguredError
from app.services.storage import StorageError

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """An error with an HTTP status and a stable machine-readable code."""

    def __init__(self, status_code: int, detail: str, code: str = "error") -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.code = code


def not_found(what: str) -> ApiError:
    """404 helper."""
    return ApiError(404, f"{what} not found.", code="not_found")


def _body(detail: str, code: str) -> dict[str, str]:
    return {"detail": detail, "code": code}


def install_error_handlers(app: FastAPI) -> None:
    """Register exception handlers on ``app``."""

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=_body(exc.detail, exc.code))

    @app.exception_handler(GeminiNotConfiguredError)
    async def _not_configured(_: Request, exc: GeminiNotConfiguredError) -> JSONResponse:
        return JSONResponse(status_code=503, content=_body(str(exc), "gemini_not_configured"))

    @app.exception_handler(StorageError)
    async def _storage(_: Request, exc: StorageError) -> JSONResponse:
        status = 413 if exc.code == "file_too_large" else 400
        return JSONResponse(status_code=status, content=_body(str(exc), exc.code))

    @app.exception_handler(VectorStoreError)
    async def _vectors(_: Request, exc: VectorStoreError) -> JSONResponse:
        logger.error("Vector store error: %s", exc)
        return JSONResponse(status_code=503, content=_body(str(exc), "vector_store_error"))
