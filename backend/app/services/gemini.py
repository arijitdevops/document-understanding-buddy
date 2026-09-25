"""Gemini access through LangChain.

LangChain is the LLM framework for this project. This module is the single
place where the Gemini-backed LangChain components are constructed:

* :func:`get_chat_model` -- a ``ChatGoogleGenerativeAI`` chat model used by the
  RAG chain, the optional reranker and image transcription;
* :func:`get_embeddings` -- a ``GoogleGenerativeAIEmbeddings`` instance used by
  the Chroma vector store (``RETRIEVAL_DOCUMENT`` for passages,
  ``RETRIEVAL_QUERY`` for questions -- LangChain applies the right task type
  for ``embed_documents`` vs ``embed_query``).

Both are created lazily, so the API boots without ``GEMINI_API_KEY``; any
feature that needs Gemini raises :class:`GeminiNotConfiguredError`, which the
API turns into an HTTP 503 with an explanatory message.

Tests (or alternative deployments) can swap in any LangChain
``BaseChatModel`` / ``Embeddings`` with :func:`set_chat_model` and
:func:`set_embeddings`.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class GeminiNotConfiguredError(RuntimeError):
    """Raised when a Gemini call is attempted without an API key."""

    def __init__(self) -> None:
        super().__init__(
            "GEMINI_API_KEY is not set. Add it to your .env file (see .env.example) "
            "and restart the API. GET /api/health lists missing configuration."
        )


class GeminiCallError(RuntimeError):
    """Raised when a Gemini call fails (after LangChain's own retries)."""


_chat_model: BaseChatModel | None = None
_embeddings: Embeddings | None = None
_overridden = False


def is_configured(settings: Settings | None = None) -> bool:
    """``True`` when an API key is present or test doubles were injected."""
    return _overridden or (settings or get_settings()).gemini_configured


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Return the process-wide Gemini chat model.

    Raises:
        GeminiNotConfiguredError: If no API key is configured.
    """
    global _chat_model
    if _chat_model is None:
        settings = settings or get_settings()
        if not settings.gemini_configured:
            raise GeminiNotConfiguredError()
        from langchain_google_genai import ChatGoogleGenerativeAI

        _chat_model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.gemini_temperature,
            max_retries=settings.gemini_max_retries,
            timeout=settings.gemini_timeout_seconds,
        )
        logger.info("Gemini chat model ready (model=%s)", settings.gemini_model)
    return _chat_model


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return the process-wide Gemini embedding model.

    Raises:
        GeminiNotConfiguredError: If no API key is configured.
    """
    global _embeddings
    if _embeddings is None:
        settings = settings or get_settings()
        if not settings.gemini_configured:
            raise GeminiNotConfiguredError()
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key,
        )
        logger.info("Gemini embeddings ready (model=%s)", settings.gemini_embedding_model)
    return _embeddings


def set_chat_model(model: BaseChatModel | None) -> None:
    """Inject a chat model (tests) or reset to lazy construction (``None``)."""
    global _chat_model, _overridden
    _chat_model = model
    _overridden = model is not None or _embeddings is not None


def set_embeddings(embeddings: Embeddings | None) -> None:
    """Inject an embedding model (tests) or reset to lazy construction (``None``)."""
    global _embeddings, _overridden
    _embeddings = embeddings
    _overridden = embeddings is not None or _chat_model is not None


def message_text(message: Any) -> str:
    """Extract plain text from a LangChain message or chunk.

    Gemini responses can arrive as a string or as a list of content blocks;
    both are flattened to text here.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


async def transcribe_image(data: bytes, mime_type: str, *, prompt: str | None = None) -> str:
    """Transcribe / describe an image with Gemini vision via LangChain.

    Returns:
        The transcription, or an empty string when the image holds no text.

    Raises:
        GeminiNotConfiguredError: If no API key is configured.
        GeminiCallError: If the model call failed.
    """
    settings = get_settings()
    model = get_chat_model(settings)
    instruction = prompt or (
        "Transcribe all text visible in this image exactly, preserving reading "
        "order and line breaks. Render tables as pipe-separated rows. After the "
        "transcription, add a line starting with 'Image description:' that "
        "briefly describes any charts, diagrams or photos. If the image contains "
        "no text and nothing worth describing, reply with the single word NO_TEXT."
    )
    encoded = base64.b64encode(data).decode("ascii")
    message = HumanMessage(
        content=[
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
        ]
    )
    try:
        if settings.gemini_vision_model != settings.gemini_model and not _overridden:
            model = model.model_copy(update={"model": settings.gemini_vision_model})
        response = await model.ainvoke([message])
    except GeminiNotConfiguredError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalised for the pipeline
        raise GeminiCallError(f"vision transcription failed: {exc}") from exc
    text = message_text(response).strip()
    return "" if text.upper() == "NO_TEXT" else text


def health(settings: Settings | None = None) -> tuple[bool, str]:
    """Report configuration state without making a network call.

    Only the *name* of the missing variable is reported -- never the key.
    """
    settings = settings or get_settings()
    if not is_configured(settings):
        return False, "GEMINI_API_KEY not set"
    return True, f"model={settings.gemini_model}; embeddings={settings.gemini_embedding_model}"
