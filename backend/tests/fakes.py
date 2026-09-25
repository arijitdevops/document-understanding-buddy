"""Deterministic stand-ins for Gemini so the suite runs offline.

* :class:`HashingEmbeddings` -- a bag-of-words hashing embedding. Texts that
  share words get similar vectors, so dense retrieval behaves sensibly.
* :class:`ScriptedChatModel` -- a LangChain chat model that replies with
  scripted answers (streaming word by word) and records every prompt it saw.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbeddings(Embeddings):
    """Deterministic, normalised bag-of-words vectors (no network)."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in _WORD.findall(text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return (
            [v / norm for v in vector]
            if any(vector)
            else [1.0 / math.sqrt(self.dimensions)] * self.dimensions
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._vector(text)


class ScriptedChatModel(BaseChatModel):
    """Returns queued responses in order (the last one repeats)."""

    responses: list[str] = Field(default_factory=lambda: ["The answer is in the document [1]."])
    prompts: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _next(self, messages: list[BaseMessage]) -> str:
        self.prompts.append(list(messages))
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(self._next(messages)))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        text = self._next(messages)
        for piece in re.findall(r"\S+\s*|\s+", text):
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))

    def last_prompt_text(self) -> str:
        """All message contents of the most recent call, joined."""
        return "\n".join(str(m.content) for m in self.prompts[-1]) if self.prompts else ""
