"""End-to-end API tests: upload -> background ingestion -> streamed, cited answers."""

from __future__ import annotations

import base64

from app.rag.vectorstore import get_vector_store
from app.services import gemini
from tests.conftest import parse_sse
from tests.helpers import make_pdf

SOLAR = (
    "# Solar Panels\n\n"
    "Photovoltaic cells convert sunlight directly into electricity. "
    "A typical residential panel produces about 400 watts at peak sun.\n\n"
    "## Maintenance\n\nPanels should be cleaned twice a year to remove dust and pollen."
)
COFFEE = (
    "# Coffee Brewing\n\n"
    "Espresso is brewed by forcing hot water through finely ground coffee at nine bars of pressure. "
    "The ideal extraction time is between 25 and 30 seconds."
)


async def _new_session(client) -> str:
    response = await client.post("/api/sessions", json={})
    assert response.status_code == 201
    return response.json()["id"]


async def _upload(client, session_id: str, name: str, content: bytes, mime: str = "text/plain"):
    return await client.post(
        f"/api/sessions/{session_id}/documents", files=[("files", (name, content, mime))]
    )


async def test_health_reports_missing_key(client):
    gemini.set_chat_model(None)
    gemini.set_embeddings(None)
    body = (await client.get("/api/health")).json()
    assert body["status"] == "degraded"
    assert body["missing_config"] == ["GEMINI_API_KEY"]
    assert body["model"] == "gemini-2.5-flash"
    assert body["embedding_model"] == "gemini-embedding-001"
    components = {c["name"]: c for c in body["components"]}
    assert components["database"]["ok"] and components["vector_store"]["ok"]
    assert not components["gemini"]["ok"]


async def test_health_ok_with_models(client, fake_llm):
    body = (await client.get("/api/health")).json()
    assert body["status"] == "ok" and body["missing_config"] == []


async def test_session_crud(client):
    session_id = await _new_session(client)
    renamed = await client.patch(f"/api/sessions/{session_id}", json={"title": "Contracts"})
    assert renamed.json()["title"] == "Contracts"
    listed = (await client.get("/api/sessions")).json()
    assert any(s["id"] == session_id for s in listed)
    assert (await client.delete(f"/api/sessions/{session_id}")).status_code == 204
    assert (await client.get(f"/api/sessions/{session_id}")).status_code == 404


async def test_upload_ingest_and_chat_with_citations(client, fake_llm):
    session_id = await _new_session(client)
    upload = await _upload(client, session_id, "solar.md", SOLAR.encode(), "text/markdown")
    assert upload.status_code == 202, upload.text
    document = upload.json()["documents"][0]

    # BackgroundTasks finish before the ASGI call returns, so ingestion is done.
    status = (await client.get(f"/api/documents/{document['id']}")).json()
    assert status["status"] == "ready", status
    assert status["chunk_count"] >= 1 and status["progress"] == 1.0
    assert await get_vector_store().count(session_id, document["id"]) == status["chunk_count"]

    fake_llm.responses = [
        "A panel produces about 400 watts [1]. Clean it twice a year [2]. Also [9]."
    ]
    response = await client.post(
        f"/api/sessions/{session_id}/chat", json={"message": "How much power does a panel produce?"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "meta" and names[1] == "sources" and names[-1] == "done"
    assert "token" in names

    sources = events[1][1]["sources"]
    assert sources and all(s["document_name"] == "solar.md" for s in sources)
    citations = next(data for name, data in events if name == "citations")
    assert [c["marker"] for c in citations["citations"]] == [1, 2][: len(sources)]
    assert "[9]" not in citations["answer"]
    assert citations["citations"][0]["snippet"]

    # The retrieved context reached the model, fenced as untrusted data.
    prompt = fake_llm.last_prompt_text()
    assert "DOCUMENT_CONTEXT" in prompt and "400 watts" in prompt

    detail = (await client.get(f"/api/sessions/{session_id}")).json()
    assert detail["title"] == "How much power does a panel produce?"
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant"]
    assert detail["messages"][1]["citations"][0]["document_name"] == "solar.md"

    # Full chunk text is available for the citation popover.
    chunk = await client.get(f"/api/chunks/{citations['citations'][0]['chunk_id']}")
    assert chunk.status_code == 200 and chunk.json()["document_name"] == "solar.md"

    # A follow-up sees the earlier turns as history.
    fake_llm.responses = ["Twice a year [1]."]
    await client.post(f"/api/sessions/{session_id}/chat", json={"message": "And cleaning?"})
    history_prompt = fake_llm.last_prompt_text()
    assert "How much power does a panel produce?" in history_prompt


async def test_document_tasks_use_whole_document(client, fake_llm):
    session_id = await _new_session(client)
    await _upload(client, session_id, "solar.md", SOLAR.encode(), "text/markdown")
    for task, marker in [
        ("summarize", "Summarise the document"),
        ("eli5", "completely new"),
        ("key_points", "key points"),
        ("glossary", "glossary"),
        ("compare", "Compare the documents"),
    ]:
        fake_llm.responses = ["Solar panels turn light into power [1]."]
        response = await client.post(f"/api/sessions/{session_id}/chat", json={"task": task})
        events = parse_sse(response.text)
        assert events[-1][0] == "done", events
        prompt = fake_llm.last_prompt_text()
        assert marker in prompt
        assert "Photovoltaic" in prompt and "cleaned twice a year" in prompt
    messages = (await client.get(f"/api/sessions/{session_id}/messages")).json()
    assert [m["task"] for m in messages if m["role"] == "user"] == [
        "summarize",
        "eli5",
        "key_points",
        "glossary",
        "compare",
    ]
    assert messages[0]["content"] == "Summarize this document."


async def test_multi_document_scoping_and_delete(client, fake_llm):
    session_id = await _new_session(client)
    solar = (await _upload(client, session_id, "solar.md", SOLAR.encode())).json()["documents"][0]
    coffee = (await _upload(client, session_id, "coffee.txt", COFFEE.encode())).json()["documents"][
        0
    ]

    fake_llm.responses = ["Nine bars [1]."]
    response = await client.post(
        f"/api/sessions/{session_id}/chat",
        json={"message": "What pressure is used?", "document_ids": [solar["id"]]},
    )
    sources = parse_sse(response.text)[1][1]["sources"]
    assert {s["document_name"] for s in sources} == {"solar.md"}

    response = await client.post(
        f"/api/sessions/{session_id}/chat", json={"message": "espresso pressure bars"}
    )
    sources = parse_sse(response.text)[1][1]["sources"]
    assert sources[0]["document_name"] == "coffee.txt"

    # Another chat's documents are never visible.
    other = await _new_session(client)
    assert (await client.post(f"/api/sessions/{other}/chat", json={"message": "espresso?"})).json()[
        "code"
    ] == "no_documents"

    assert (await client.delete(f"/api/documents/{coffee['id']}")).status_code == 204
    assert await get_vector_store().count(session_id, coffee["id"]) == 0
    assert await get_vector_store().count(session_id, solar["id"]) > 0
    remaining = (await client.get(f"/api/sessions/{session_id}/documents")).json()
    assert [d["original_name"] for d in remaining] == ["solar.md"]

    assert (await client.delete(f"/api/sessions/{session_id}")).status_code == 204
    assert await get_vector_store().count(session_id) == 0


async def test_pdf_citations_carry_page_numbers(client, fake_llm):
    session_id = await _new_session(client)
    pdf = make_pdf(["Intro page about nothing in particular.", "The warranty lasts 25 years."])
    upload = await _upload(client, session_id, "warranty.pdf", pdf, "application/pdf")
    assert upload.json()["documents"][0]["mime"] == "application/pdf"
    fake_llm.responses = ["The warranty lasts 25 years [1]."]
    response = await client.post(
        f"/api/sessions/{session_id}/chat", json={"message": "How long is the warranty?"}
    )
    citation = next(d for n, d in parse_sse(response.text) if n == "citations")["citations"][0]
    assert citation["document_name"] == "warranty.pdf" and citation["page"] == 2


async def test_image_upload_uses_vision(client, fake_llm):
    session_id = await _new_session(client)
    fake_llm.responses = ["WHITEBOARD: Sprint goal is to ship search."]
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    upload = await _upload(client, session_id, "board.png", png, "image/png")
    document = upload.json()["documents"][0]
    status = (await client.get(f"/api/documents/{document['id']}")).json()
    assert status["status"] == "ready", status
    first_call = fake_llm.prompts[0][0].content
    assert isinstance(first_call, list) and first_call[1]["type"] == "image_url"


async def test_duplicates_and_rejections(client, fake_llm):
    session_id = await _new_session(client)
    await _upload(client, session_id, "solar.md", SOLAR.encode())
    again = await _upload(client, session_id, "copy.md", SOLAR.encode())
    assert again.status_code == 202 and again.json()["duplicates"] == ["copy.md"]

    bad = await _upload(client, session_id, "virus.exe", b"MZ", "application/octet-stream")
    assert bad.status_code == 400 and bad.json()["code"] == "unsupported_type"
    empty = await _upload(client, session_id, "empty.txt", b"")
    assert empty.status_code == 400 and empty.json()["code"] == "empty_file"


async def test_without_api_key(client):
    session_id = await _new_session(client)
    upload = await _upload(client, session_id, "solar.md", SOLAR.encode())
    document = upload.json()["documents"][0]
    status = (await client.get(f"/api/documents/{document['id']}")).json()
    assert status["status"] == "failed" and "GEMINI_API_KEY" in status["error"]

    chat = await client.post(f"/api/sessions/{session_id}/chat", json={"message": "hi"})
    assert chat.status_code == 503 and chat.json()["code"] == "gemini_not_configured"

    # Once a key (here: fakes) is available, the document can be re-ingested.
    gemini.set_chat_model(None)
    from tests.fakes import HashingEmbeddings, ScriptedChatModel

    gemini.set_chat_model(ScriptedChatModel())
    gemini.set_embeddings(HashingEmbeddings())
    try:
        again = await client.post(f"/api/documents/{document['id']}/reingest")
        assert again.status_code == 202
        status = (await client.get(f"/api/documents/{document['id']}")).json()
        assert status["status"] == "ready"
    finally:
        gemini.set_chat_model(None)
        gemini.set_embeddings(None)


async def test_chat_validation(client, fake_llm):
    session_id = await _new_session(client)
    response = await client.post(f"/api/sessions/{session_id}/chat", json={"message": ""})
    assert response.status_code == 422
    response = await client.post(f"/api/sessions/{session_id}/chat", json={"message": "hi"})
    assert response.status_code == 400 and response.json()["code"] == "no_documents"
    response = await client.post("/api/sessions/nope/chat", json={"message": "hi"})
    assert response.status_code == 404
