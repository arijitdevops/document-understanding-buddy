"""Chroma vector store, accessed through LangChain's ``langchain-chroma``.

One persistent Chroma client is shared by the process; each chat session gets
its own Chroma collection (``session-<id>``) wrapped in a
:class:`langchain_chroma.Chroma` vector store whose embedding function is the
Gemini embedding model. Every record carries ``document_id`` in its metadata,
so a question can be scoped to some of the session's documents without a
second index, and a document can be removed with a metadata-filtered delete.

Chroma's client is synchronous; every call is dispatched with
:func:`asyncio.to_thread` so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings

from app.config import Settings, get_settings
from app.services.gemini import GeminiNotConfiguredError, get_embeddings

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]")


class VectorStoreError(RuntimeError):
    """Raised when Chroma cannot be opened or a call fails."""


@dataclass(slots=True)
class VectorRecord:
    """One chunk to index, plus the metadata needed to render a citation."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VectorHit:
    """A nearest-neighbour result."""

    id: str
    text: str
    metadata: dict[str, Any]
    distance: float

    @property
    def score(self) -> float:
        """Similarity in ``[0, 1]`` derived from the cosine distance."""
        return max(0.0, 1.0 - self.distance)


def collection_key(session_id: str) -> str:
    """Map a chat session id to a Chroma-legal collection name."""
    safe = _SAFE_NAME.sub("-", session_id)[:56]
    return f"session-{safe}"


class VectorStore:
    """Async facade over LangChain ``Chroma`` stores backed by one persistent client."""

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embeddings = embeddings
        self._client: Any | None = None
        self._lock = asyncio.Lock()

    # -- plumbing ------------------------------------------------------------
    def _ensure_client(self) -> Any:
        """Create the persistent Chroma client on first use."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                self._client = chromadb.PersistentClient(
                    path=str(self._settings.chroma_path),
                    settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
                )
                logger.info("Chroma client ready at %s", self._settings.chroma_path)
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise VectorStoreError("chromadb is not installed.") from exc
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Could not open Chroma store: {exc}") from exc
        return self._client

    def _store(self, session_id: str) -> Any:
        """Return a LangChain ``Chroma`` store for one session.

        Raises:
            GeminiNotConfiguredError: If embeddings are needed but no key is set.
        """
        from langchain_chroma import Chroma

        return Chroma(
            collection_name=collection_key(session_id),
            embedding_function=self._embeddings or get_embeddings(self._settings),
            client=self._ensure_client(),
            collection_configuration={"hnsw": {"space": "cosine"}},
        )

    def _raw_collection(self, session_id: str) -> Any | None:
        """Return the raw Chroma collection, or ``None`` if it does not exist.

        Used for operations that need no embeddings (delete, fetch, count), so
        they keep working when ``GEMINI_API_KEY`` is missing.
        """
        client = self._ensure_client()
        try:
            return client.get_collection(name=collection_key(session_id))
        except Exception:  # noqa: BLE001 - chroma raises NotFound-type errors
            return None

    # -- writes ----------------------------------------------------------------
    async def add(self, session_id: str, records: Sequence[VectorRecord]) -> int:
        """Embed and upsert records via ``Chroma.add_texts``. Returns the count."""
        if not records:
            return 0

        def _write() -> int:
            store = self._store(session_id)
            store.add_texts(
                texts=[record.text for record in records],
                metadatas=[_clean_metadata(record.metadata) for record in records],
                ids=[record.id for record in records],
            )
            return len(records)

        async with self._lock:
            try:
                written = await asyncio.to_thread(_write)
            except (VectorStoreError, GeminiNotConfiguredError):
                raise
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Indexing failed: {exc}") from exc
        logger.info("Indexed %d chunks into session %s", written, session_id)
        return written

    async def delete_document(self, session_id: str, document_id: str) -> None:
        """Remove every vector belonging to one document (idempotent)."""

        def _delete() -> None:
            collection = self._raw_collection(session_id)
            if collection is not None:
                collection.delete(where={"document_id": document_id})

        async with self._lock:
            await asyncio.to_thread(_delete)
        logger.info("Deleted vectors for document %s", document_id)

    async def drop_session(self, session_id: str) -> None:
        """Delete a session's whole collection. Missing collections are ignored."""

        def _drop() -> None:
            client = self._ensure_client()
            try:
                client.delete_collection(name=collection_key(session_id))
            except Exception as exc:  # noqa: BLE001 - already gone is fine
                logger.debug("Chroma collection for %s not dropped: %s", session_id, exc)

        async with self._lock:
            await asyncio.to_thread(_drop)

    # -- reads -------------------------------------------------------------------
    async def search(
        self,
        session_id: str,
        query: str,
        *,
        top_k: int = 10,
        document_ids: Sequence[str] | None = None,
    ) -> list[VectorHit]:
        """Dense similarity search with ``Chroma.similarity_search_with_score``.

        The query is embedded by LangChain with the ``RETRIEVAL_QUERY`` task
        type; passages were embedded with ``RETRIEVAL_DOCUMENT``.
        """
        if not query.strip():
            return []

        def _search() -> list[VectorHit]:
            if self._raw_collection(session_id) is None:
                return []
            store = self._store(session_id)
            available = store._collection.count()
            if available == 0:
                return []
            results: list[tuple[LCDocument, float]] = store.similarity_search_with_score(
                query, k=max(1, min(top_k, available)), filter=_document_filter(document_ids)
            )
            return [
                VectorHit(
                    id=str(doc.id or doc.metadata.get("chunk_id", "")),
                    text=doc.page_content,
                    metadata=dict(doc.metadata),
                    distance=float(distance),
                )
                for doc, distance in results
            ]

        try:
            return await asyncio.to_thread(_search)
        except (VectorStoreError, GeminiNotConfiguredError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

    async def fetch_all(
        self, session_id: str, *, document_ids: Sequence[str] | None = None
    ) -> list[VectorRecord]:
        """Return stored chunks (text + metadata); used to build the BM25 index."""
        where = _document_filter(document_ids)

        def _fetch() -> list[VectorRecord]:
            collection = self._raw_collection(session_id)
            if collection is None:
                return []
            raw = collection.get(where=where, include=["documents", "metadatas"])
            ids = raw.get("ids") or []
            documents = raw.get("documents") or []
            metadatas = raw.get("metadatas") or []
            return [
                VectorRecord(
                    id=identifier,
                    text=documents[position] if position < len(documents) else "",
                    metadata=dict(metadatas[position] or {}) if position < len(metadatas) else {},
                )
                for position, identifier in enumerate(ids)
            ]

        try:
            return await asyncio.to_thread(_fetch)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Vector fetch failed: {exc}") from exc

    async def count(self, session_id: str, document_id: str | None = None) -> int:
        """Number of vectors in a session (optionally for one document)."""

        def _count() -> int:
            collection = self._raw_collection(session_id)
            if collection is None:
                return 0
            if document_id is None:
                return int(collection.count())
            return len(collection.get(where={"document_id": document_id}, include=[])["ids"])

        try:
            return await asyncio.to_thread(_count)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector count failed for %s: %s", session_id, exc)
            return 0

    async def health(self) -> tuple[bool, str | None]:
        """Verify the store can be opened and listed."""

        def _probe() -> str:
            client = self._ensure_client()
            return f"{len(client.list_collections())} collection(s)"

        try:
            return True, await asyncio.to_thread(_probe)
        except Exception as exc:  # noqa: BLE001 - health must not raise
            return False, str(exc)[:200]


def _document_filter(document_ids: Sequence[str] | None) -> dict[str, Any] | None:
    """Build a Chroma ``where`` clause restricting results to some documents."""
    if not document_ids:
        return None
    ids = list(dict.fromkeys(document_ids))
    if len(ids) == 1:
        return {"document_id": ids[0]}
    return {"document_id": {"$in": ids}}


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts scalar metadata values; drop ``None`` and stringify the rest."""
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        cleaned[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return cleaned


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the process-wide :class:`VectorStore`."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def set_vector_store(store: VectorStore | None) -> None:
    """Replace the singleton (used by tests)."""
    global _store
    _store = store
