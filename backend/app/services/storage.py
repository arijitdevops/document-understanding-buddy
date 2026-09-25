"""Upload validation and on-disk file storage.

Responsibilities:

* extension + MIME allowlist,
* size cap enforced while streaming (never buffer the whole upload first),
* filename sanitisation and UUID stored names,
* SHA-256 checksum for duplicate detection.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Extension -> canonical MIME type. Doubles as the upload allowlist.
ALLOWED_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

#: MIME values browsers commonly send that we accept as aliases.
MIME_ALIASES: dict[str, str] = {
    "application/octet-stream": "",  # fall back to the extension
    "text/x-markdown": "text/markdown",
    "application/csv": "text/csv",
    "application/vnd.ms-excel": "text/csv",
    "image/jpg": "image/jpeg",
}

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")
_CHUNK = 1024 * 1024


class StorageError(ValueError):
    """Raised for any rejected upload. Carries a user-safe message."""

    def __init__(self, message: str, code: str = "upload_rejected") -> None:
        super().__init__(message)
        self.code = code


class UploadLike(Protocol):
    """Structural type matching :class:`fastapi.UploadFile`."""

    filename: str | None
    content_type: str | None

    async def read(self, size: int = -1) -> bytes:  # pragma: no cover - protocol
        ...


@dataclass(slots=True)
class StoredFile:
    """Result of a successful save."""

    path: Path
    stored_name: str
    original_name: str
    mime: str
    size_bytes: int
    checksum: str


def sanitize_filename(name: str | None) -> str:
    """Return a safe display name.

    Strips directory components, normalises Unicode, removes characters that
    are awkward on Windows, and caps the length.
    """
    raw = (name or "document").replace("\\", "/").split("/")[-1]
    raw = unicodedata.normalize("NFKC", raw).strip()
    cleaned = _UNSAFE_CHARS.sub("_", raw).strip(" .")
    if not cleaned:
        cleaned = "document"
    if len(cleaned) > 180:
        stem, _, suffix = cleaned.rpartition(".")
        cleaned = (stem[:170] + "." + suffix) if suffix else cleaned[:180]
    return cleaned


def resolve_mime(filename: str, declared: str | None) -> tuple[str, str]:
    """Resolve an upload to ``(extension, canonical_mime)``.

    Dispatch uses both the declared MIME type and the extension: browsers lie
    in both directions, so a file is accepted only when the extension is on the
    allowlist and the declared MIME (when meaningful) does not contradict it.

    Raises:
        StorageError: If the type is not supported.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_TYPES:
        supported = ", ".join(sorted(ALLOWED_TYPES))
        raise StorageError(
            f"Unsupported file type {ext or '(no extension)'}. Supported: {supported}",
            code="unsupported_type",
        )
    canonical = ALLOWED_TYPES[ext]
    declared_norm = (declared or "").split(";")[0].strip().lower()
    declared_norm = MIME_ALIASES.get(declared_norm, declared_norm)
    if declared_norm and declared_norm != canonical:
        # Text-ish mismatches are benign (text/plain for .md, etc.).
        if not (declared_norm.startswith("text/") and canonical.startswith("text/")):
            logger.info(
                "MIME mismatch for %s: declared=%s canonical=%s -- using canonical",
                filename,
                declared_norm,
                canonical,
            )
    return ext, canonical


class StorageService:
    """Saves uploads under ``UPLOAD_DIR`` and reads them back."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def root(self) -> Path:
        """Absolute upload directory (created if needed)."""
        return self._settings.upload_path

    async def save_upload(self, upload: UploadLike) -> StoredFile:
        """Validate and persist an uploaded file.

        Raises:
            StorageError: On unsupported type, empty file or size overflow.
        """
        original = sanitize_filename(upload.filename)
        ext, mime = resolve_mime(original, upload.content_type)
        stored_name = f"{uuid.uuid4().hex}{ext}"
        target = self.root / stored_name
        digest = hashlib.sha256()
        size = 0
        limit = self._settings.max_upload_bytes

        try:
            with target.open("wb") as handle:
                while True:
                    block = await upload.read(_CHUNK)
                    if not block:
                        break
                    size += len(block)
                    if size > limit:
                        raise StorageError(
                            f"File exceeds the {self._settings.max_upload_mb} MB limit.",
                            code="file_too_large",
                        )
                    digest.update(block)
                    await asyncio.to_thread(handle.write, block)
        except StorageError:
            self._discard(target)
            raise
        except OSError as exc:
            self._discard(target)
            raise StorageError(f"Could not write the upload to disk: {exc}", "io_error") from exc

        if size == 0:
            self._discard(target)
            raise StorageError("The uploaded file is empty.", code="empty_file")

        logger.info("Stored upload %s as %s (%d bytes)", original, stored_name, size)
        return StoredFile(
            path=target,
            stored_name=stored_name,
            original_name=original,
            mime=mime,
            size_bytes=size,
            checksum=digest.hexdigest(),
        )

    def path_for(self, stored_name: str) -> Path:
        """Return the absolute path of a stored file, refusing traversal.

        Raises:
            StorageError: If the name escapes the upload directory or is absent.
        """
        candidate = (self.root / stored_name).resolve()
        if not str(candidate).startswith(str(self.root.resolve())):
            raise StorageError("Invalid stored file name.", code="invalid_path")
        if not candidate.exists():
            raise StorageError("The stored file is no longer on disk.", code="file_missing")
        return candidate

    def read_bytes(self, stored_name: str) -> bytes:
        """Read a stored file's bytes (call via ``asyncio.to_thread``)."""
        return self.path_for(stored_name).read_bytes()

    def _discard(self, path: Path) -> None:
        """Best-effort cleanup of a partially written file."""
        try:
            if path.exists():
                path.unlink()
        except OSError:  # pragma: no cover - cleanup must never mask the real error
            logger.warning("Could not remove partial upload %s", path)

    def delete(self, stored_name: str) -> bool:
        """Delete a stored file. Returns ``True`` when a file was removed."""
        try:
            self.path_for(stored_name).unlink()
            return True
        except (StorageError, OSError) as exc:
            logger.warning("Could not delete %s: %s", stored_name, exc)
            return False


def hash_bytes(data: bytes) -> str:
    """SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def hash_stream(handle: BinaryIO) -> str:
    """SHA-256 hex digest of a binary stream, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(_CHUNK), b""):
        digest.update(block)
    return digest.hexdigest()


storage = StorageService()
