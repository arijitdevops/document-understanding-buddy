"""Aggregate router mounted under ``API_PREFIX`` (default ``/api``)."""

from fastapi import APIRouter

from app.api.routes import chat, documents, health, sessions

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(sessions.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
