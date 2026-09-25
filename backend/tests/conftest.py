"""Test configuration: SQLite + a temporary Chroma directory + fake Gemini."""

from __future__ import annotations

import os
import shutil
import tempfile

_TMP = tempfile.mkdtemp(prefix="doc-buddy-tests-")
os.environ.update(
    {
        "DATABASE_URL": f"sqlite+aiosqlite:///{_TMP}/test.db",
        "CHROMA_PERSIST_DIR": f"{_TMP}/chroma",
        "UPLOAD_DIR": f"{_TMP}/uploads",
        "GEMINI_API_KEY": "",
        "ENVIRONMENT": "test",
        "RATE_LIMIT_PER_MINUTE": "1000",
        "HYBRID_ENABLED": "true",
        "RERANK_ENABLED": "false",
    }
)

import json  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from app.db.session import dispose_engine, init_models  # noqa: E402
from app.main import app  # noqa: E402
from app.services import gemini  # noqa: E402
from tests.fakes import HashingEmbeddings, ScriptedChatModel  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
async def _database():
    await init_models()
    yield
    await dispose_engine()
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture()
def fake_llm() -> ScriptedChatModel:
    """Inject fake Gemini chat + embedding models for one test."""
    model = ScriptedChatModel()
    gemini.set_chat_model(model)
    gemini.set_embeddings(HashingEmbeddings())
    yield model
    gemini.set_chat_model(None)
    gemini.set_embeddings(None)


@pytest.fixture()
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Split a text/event-stream body into ``(event, data)`` pairs."""
    events: list[tuple[str, dict]] = []
    for block in body.strip().split("\n\n"):
        name, data = "message", ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data += line[6:]
        if data:
            events.append((name, json.loads(data)))
    return events
