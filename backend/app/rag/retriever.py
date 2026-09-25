"""Hybrid retrieval: dense vectors + BM25, fused with Reciprocal Rank Fusion.

Dense search finds passages that *mean* the same thing as the question; BM25
finds passages that contain the same rare words -- part numbers, surnames,
error codes -- which embeddings routinely smear away. RRF merges the two
ranked lists without needing the scores to be on a comparable scale:

    score(d) = sum over lists L of  weight(L) / (k + rank_L(d))

``k`` (default 60) damps the influence of the very top ranks so a single list
cannot dominate the fusion. ``hybrid_alpha`` weights dense against lexical:
1.0 is dense-only, 0.0 is BM25-only, 0.5 is an even split.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.rag.vectorstore import VectorRecord, VectorStore, VectorStoreError, get_vector_store

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were what when where which who why will with how does do did can could""".split()
)


@dataclass(slots=True)
class RetrievedCandidate:
    """A chunk surfaced by retrieval, with the provenance of how it got there."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_rank: int | None = None
    dense_score: float = 0.0
    bm25_rank: int | None = None
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None

    @property
    def document_id(self) -> str:
        """Owning document id (empty when metadata is incomplete)."""
        return str(self.metadata.get("document_id", ""))

    @property
    def document_name(self) -> str:
        """Display name of the owning document."""
        return str(self.metadata.get("document_name", "unknown document"))

    @property
    def page(self) -> int | None:
        """1-based page number, when the loader produced one."""
        value = self.metadata.get("page")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def heading(self) -> str | None:
        """Heading trail for the chunk, if any."""
        value = self.metadata.get("heading")
        return str(value) if value else None

    def snippet(self, limit: int = 320) -> str:
        """First ``limit`` characters of the chunk, whitespace-normalised."""
        text = " ".join(self.text.split())
        return text if len(text) <= limit else f"{text[:limit].rstrip()}..."


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed (BM25 input)."""
    return [
        token
        for token in (match.group(0).lower() for match in _TOKEN.finditer(text))
        if token not in _STOPWORDS and len(token) > 1
    ]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    """Fuse ranked id lists into a single score per id.

    Args:
        ranked_lists: Each inner sequence is ordered best-first.
        weights: Per-list weights; defaults to 1.0 for each list.
        k: RRF damping constant. Larger values flatten rank differences.

    Returns:
        ``{id: fused_score}``. Ids absent from a list simply contribute nothing
        from that list.

    Example:
        Two lists, equal weights, ``k=60``::

            dense = ["a", "b", "c"]; lexical = ["b", "a", "d"]
            a -> 1/61 + 1/62 = 0.032523...
            b -> 1/62 + 1/61 = 0.032523...
            c -> 1/63          = 0.015873...
            d -> 1/63          = 0.015873...
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match the number of ranked lists")

    scores: dict[str, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        if weight <= 0:
            continue
        for position, identifier in enumerate(ranked, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + weight / (k + position)
    return scores


class BM25Index:
    """Okapi BM25 over a fixed set of chunks.

    Implemented in-module (about thirty lines) rather than via ``rank_bm25``:
    the ``log(1 + ...)`` IDF variant used here never goes negative, which
    matters for the tiny corpora typical of a chat with one or two uploads.
    """

    def __init__(
        self, records: Sequence[VectorRecord], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.ids = [record.id for record in records]
        self.corpus = [tokenize(record.text) for record in records]
        self._k1 = k1
        self._b = b
        self._build_statistics()

    def _build_statistics(self) -> None:
        """Precompute document frequencies and lengths."""
        self._doc_len = [len(doc) for doc in self.corpus]
        self._avg_len = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        self._freqs: list[dict[str, int]] = []
        self._df: dict[str, int] = {}
        for doc in self.corpus:
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            self._freqs.append(counts)
            for token in counts:
                self._df[token] = self._df.get(token, 0) + 1

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return ``(chunk_id, score)`` for the best ``top_k`` matches."""
        tokens = tokenize(query)
        if not tokens or not self.ids:
            return []
        scores = [self._score(index, tokens) for index in range(len(self.corpus))]
        ranked = sorted(zip(self.ids, scores), key=lambda pair: pair[1], reverse=True)
        return [(identifier, float(score)) for identifier, score in ranked[:top_k] if score > 0]

    def _score(self, index: int, tokens: Sequence[str]) -> float:
        """Okapi BM25 score of document ``index`` for ``tokens``."""
        total = len(self.corpus)
        length = self._doc_len[index] or 1
        freqs = self._freqs[index]
        score = 0.0
        for token in tokens:
            frequency = freqs.get(token, 0)
            if frequency == 0:
                continue
            df = self._df.get(token, 0)
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            denominator = frequency + self._k1 * (
                1 - self._b + self._b * length / (self._avg_len or 1)
            )
            score += idf * (frequency * (self._k1 + 1)) / denominator
        return score


class HybridRetriever:
    """Dense + lexical retrieval with RRF fusion."""

    def __init__(
        self,
        store: VectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._store = store
        self._settings = settings or get_settings()

    @property
    def store(self) -> VectorStore:
        """The vector store (resolved lazily so tests can swap the singleton)."""
        return self._store or get_vector_store()

    async def retrieve(
        self,
        session_id: str,
        query: str,
        *,
        top_k: int | None = None,
        candidates: int | None = None,
        document_ids: Sequence[str] | None = None,
        hybrid: bool | None = None,
    ) -> list[RetrievedCandidate]:
        """Retrieve the best candidate chunks for ``query``.

        Args:
            session_id: Chat session whose documents are searched.
            query: The user's question.
            top_k: Number of candidates to return (defaults to config).
            candidates: Size of the pool fetched from each retriever before
                fusion. Larger pools give the reranker more to work with.
            document_ids: Restrict the search to these documents.
            hybrid: Force hybrid on/off, overriding config.

        Returns:
            Candidates ordered best-first.

        Raises:
            VectorStoreError: If the dense search could not run at all.
        """
        top_k = top_k or self._settings.retrieval_top_k
        pool = max(candidates or self._settings.retrieval_candidates, top_k)
        use_hybrid = self._settings.hybrid_enabled if hybrid is None else hybrid

        dense_hits = await self.store.search(
            session_id, query, top_k=pool, document_ids=document_ids
        )
        by_id: dict[str, RetrievedCandidate] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            by_id[hit.id] = RetrievedCandidate(
                chunk_id=hit.id,
                text=hit.text,
                metadata=hit.metadata,
                dense_rank=rank,
                dense_score=hit.score,
            )
        dense_order = [hit.id for hit in dense_hits]

        lexical_order: list[str] = []
        if use_hybrid:
            lexical_order = await self._lexical(session_id, query, pool, document_ids, by_id)

        alpha = self._settings.hybrid_alpha if use_hybrid else 1.0
        fused = reciprocal_rank_fusion(
            [dense_order, lexical_order],
            weights=[alpha, 1.0 - alpha],
            k=self._settings.rrf_k,
        )
        for chunk_id, score in fused.items():
            if chunk_id in by_id:
                by_id[chunk_id].fused_score = score

        ordered = sorted(by_id.values(), key=lambda c: c.fused_score, reverse=True)
        logger.info(
            "Retrieved %d candidates (dense=%d lexical=%d hybrid=%s) for %r",
            len(ordered),
            len(dense_order),
            len(lexical_order),
            use_hybrid,
            query[:60],
        )
        return ordered[:pool]

    async def _lexical(
        self,
        session_id: str,
        query: str,
        pool: int,
        document_ids: Sequence[str] | None,
        by_id: dict[str, RetrievedCandidate],
    ) -> list[str]:
        """Run BM25 and merge any new chunks into ``by_id``.

        A BM25 failure is logged and degraded to dense-only rather than failing
        the whole request.
        """
        try:
            records = await self.store.fetch_all(session_id, document_ids=document_ids)
        except VectorStoreError as exc:
            logger.warning("Lexical retrieval skipped: %s", exc)
            return []
        if not records:
            return []

        index = BM25Index(records)
        results = index.search(query, pool)
        text_by_id = {record.id: record for record in records}
        order: list[str] = []
        for rank, (chunk_id, score) in enumerate(results, start=1):
            order.append(chunk_id)
            candidate = by_id.get(chunk_id)
            if candidate is None:
                record = text_by_id[chunk_id]
                candidate = RetrievedCandidate(
                    chunk_id=chunk_id, text=record.text, metadata=dict(record.metadata)
                )
                by_id[chunk_id] = candidate
            candidate.bm25_rank = rank
            candidate.bm25_score = score
        return order


def dedupe_candidates(candidates: Iterable[RetrievedCandidate]) -> list[RetrievedCandidate]:
    """Drop near-duplicate chunks (same document, same normalised prefix)."""
    seen: set[tuple[str, str]] = set()
    unique: list[RetrievedCandidate] = []
    for candidate in candidates:
        key = (candidate.document_id, " ".join(candidate.text.split())[:120].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
