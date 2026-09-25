"""LLM reranking of fused retrieval candidates.

Fusion gets the right passage into the top ~30; reranking gets it into the top
5 that actually fit in the prompt. The model scores each candidate against the
question in one structured JSON call, which is far cheaper than one call per
passage. The call is a small LCEL pipeline:
``ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser``.

Reranking is off by default (``RERANK_ENABLED=false``) because it costs one
extra model call per question. When reranking is disabled, unavailable, or returns malformed JSON, the
reranker passes the fused ordering through unchanged -- degrading to "slightly
worse ordering" rather than to an error.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings, get_settings
from app.rag.prompts import RERANK_TEMPLATE
from app.rag.retriever import RetrievedCandidate
from app.services.gemini import get_chat_model, is_configured

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_PASSAGE_CHARS = 700


class Reranker:
    """Scores candidates with the generation model and reorders them."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a precise relevance scorer. Reply with JSON only."),
                ("human", "{prompt}"),
            ]
        )

    @property
    def enabled(self) -> bool:
        """``True`` when reranking is configured *and* Gemini is usable."""
        return self._settings.rerank_enabled and is_configured(self._settings)

    async def rerank(
        self,
        question: str,
        candidates: Sequence[RetrievedCandidate],
        *,
        top_n: int | None = None,
    ) -> tuple[list[RetrievedCandidate], bool]:
        """Reorder ``candidates`` by relevance to ``question``.

        Args:
            question: The user's question.
            candidates: Fused candidates, best-first.
            top_n: How many to return (defaults to ``RERANK_TOP_N``).

        Returns:
            ``(ordered_candidates, reranked)``. ``reranked`` is ``False`` when
            the pass-through path was taken.
        """
        limit = top_n or self._settings.rerank_top_n
        if not candidates:
            return [], False
        if not self.enabled:
            logger.debug("Reranking disabled; using fused order")
            return list(candidates[:limit]), False

        prompt = self._build_prompt(question, candidates)
        chain = (
            self._prompt
            | get_chat_model(self._settings).bind(response_mime_type="application/json")
            | StrOutputParser()
        )
        try:
            raw = await chain.ainvoke({"prompt": prompt})
        except Exception as exc:  # noqa: BLE001 - reranking is best-effort
            logger.warning("Reranking failed (%s); falling back to fused order", exc)
            return list(candidates[:limit]), False

        scores = self._parse_scores(raw, len(candidates))
        if not scores:
            logger.warning("Reranker returned unusable JSON; keeping fused order")
            return list(candidates[:limit]), False

        for position, candidate in enumerate(candidates):
            candidate.rerank_score = scores.get(position + 1)

        ordered = sorted(
            candidates,
            key=lambda c: (
                c.rerank_score if c.rerank_score is not None else -1.0,
                c.fused_score,
            ),
            reverse=True,
        )
        logger.info("Reranked %d candidates down to %d", len(candidates), limit)
        return list(ordered[:limit]), True

    def _build_prompt(self, question: str, candidates: Sequence[RetrievedCandidate]) -> str:
        """Render the scoring prompt with numbered, truncated passages."""
        blocks: list[str] = []
        for position, candidate in enumerate(candidates, start=1):
            text = " ".join(candidate.text.split())[:_PASSAGE_CHARS]
            label = candidate.document_name
            if candidate.page:
                label = f"{label}, page {candidate.page}"
            blocks.append(f"[{position}] ({label})\n{text}")
        return RERANK_TEMPLATE.format(question=question, passages="\n\n".join(blocks))

    @staticmethod
    def _parse_scores(raw: str, count: int) -> dict[int, float]:
        """Parse ``{"scores": [{"id": n, "score": x}]}`` defensively.

        Returns an empty dict when nothing usable could be extracted, which the
        caller treats as "keep the fused order".
        """
        if not raw:
            return {}
        payload = raw.strip()
        if not payload.startswith("{"):
            match = _JSON_BLOCK.search(payload)
            if not match:
                return {}
            payload = match.group(0)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return {}

        entries = data.get("scores") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            return {}

        scores: dict[int, float] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                identifier = int(entry["id"])
                score = float(entry["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= identifier <= count:
                scores[identifier] = max(0.0, min(1.0, score))
        return scores


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """Return the process-wide :class:`Reranker`."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def set_reranker(reranker: Reranker | None) -> None:
    """Replace the singleton (used by tests)."""
    global _reranker
    _reranker = reranker
