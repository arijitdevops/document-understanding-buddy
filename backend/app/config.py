"""Application configuration loaded from environment variables.

Every setting has a usable default so the app can boot without a ``.env``
file; the only value that genuinely cannot be defaulted is ``GEMINI_API_KEY``.
When it is absent the application still starts, ``/api/health`` reports the
missing key by name, and any route that needs Gemini returns a clear 503
instead of a traceback.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed view over the process environment."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Application ---------------------------------------------------
    app_name: str = "Document Understanding Buddy"
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    api_prefix: str = "/api"

    # ---- Gemini --------------------------------------------------------
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-2.5-flash")
    gemini_embedding_model: str = Field(default="gemini-embedding-001")
    gemini_vision_model: str = Field(default="gemini-2.5-flash")
    gemini_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    gemini_max_retries: int = Field(default=3, ge=0, le=10)
    gemini_timeout_seconds: float = Field(default=120.0, gt=0)

    # ---- Database ------------------------------------------------------
    database_url: str = Field(
        default="mysql+asyncmy://doc_buddy:doc_buddy@localhost:3306/doc_buddy"
    )
    db_echo: bool = Field(default=False)
    db_pool_size: int = Field(default=10, ge=1)
    db_max_overflow: int = Field(default=20, ge=0)
    auto_create_tables: bool = Field(default=True)

    # ---- Storage -------------------------------------------------------
    chroma_persist_dir: str = Field(default="./data/chroma")
    upload_dir: str = Field(default="./data/uploads")
    max_upload_mb: int = Field(default=25, ge=1, le=512)

    # ---- Chunking ------------------------------------------------------
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)

    # ---- Retrieval -----------------------------------------------------
    retrieval_top_k: int = Field(default=6, ge=1, le=50)
    retrieval_candidates: int = Field(default=30, ge=1, le=200)
    hybrid_enabled: bool = Field(default=True)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    rerank_enabled: bool = Field(default=False)
    rerank_top_n: int = Field(default=6, ge=1, le=50)
    embedding_batch_size: int = Field(default=64, ge=1, le=100)

    # ---- Conversation ----------------------------------------------------
    history_messages: int = Field(default=6, ge=0, le=50)
    document_task_max_chars: int = Field(default=60000, ge=2000, le=800000)

    # ---- HTTP ----------------------------------------------------------
    cors_origins: str = Field(default="http://localhost:5173,http://127.0.0.1:5173")
    rate_limit_per_minute: int = Field(default=30, ge=1)

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_below_size(cls, value: int, info) -> int:
        """Overlap must stay strictly below the chunk size or splitting loops."""
        size = info.data.get("chunk_size", 1000)
        if value >= size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    # ---- Derived helpers ------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a list, tolerating spaces and trailing commas."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        """Upload cap in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        """Absolute upload directory, created on first access."""
        return self._resolve(self.upload_dir)

    @property
    def chroma_path(self) -> Path:
        """Absolute Chroma persistence directory, created on first access."""
        return self._resolve(self.chroma_persist_dir)

    @property
    def gemini_configured(self) -> bool:
        """``True`` when an API key is present (value is never logged)."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    def missing_requirements(self) -> list[str]:
        """Names -- never values -- of required settings that are absent."""
        missing: list[str] = []
        if not self.gemini_configured:
            missing.append("GEMINI_API_KEY")
        return missing

    def _resolve(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = (BACKEND_ROOT / path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    settings = Settings()
    missing = settings.missing_requirements()
    if missing:
        logger.warning(
            "Missing configuration: %s. The API will start, but features that "
            "need them return HTTP 503 with an explanatory message.",
            ", ".join(missing),
        )
    return settings


settings = get_settings()
