"""Logging configuration.

Development gets a readable single-line format; anything else gets JSON so
log aggregators can parse it. Call :func:`configure_logging` once at startup.
"""

from __future__ import annotations

import json
import logging
import logging.config
import sys
from typing import Any

_SENSITIVE_HINTS = ("api_key", "apikey", "authorization", "password", "secret", "token")


class JsonFormatter(logging.Formatter):
    """Minimal structured formatter with no third-party dependency."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            if any(hint in key.lower() for hint in _SENSITIVE_HINTS):
                payload[key] = "***redacted***"
            else:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", environment: str = "development") -> None:
    """Install handlers for the root logger.

    Args:
        level: Log level name, e.g. ``"DEBUG"``. Unknown names fall back to INFO.
        environment: ``"development"`` selects the human-readable formatter.
    """
    resolved = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    if environment.lower() == "development":
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    else:
        handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)

    # These libraries are chatty at INFO and drown out our own logs.
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(max(resolved, logging.WARNING))
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    # google-genai warns about automatic function calling on every streamed
    # request; this app never uses function calling.
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
