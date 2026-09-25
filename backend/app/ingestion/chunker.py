"""Recursive character chunking with page and heading provenance.

The splitter walks a list of separators from coarsest to finest
(``paragraph -> line -> sentence -> word -> character``) and only falls back to
a finer one when a piece is still too large. That keeps paragraphs and
sentences intact, which matters more for citation quality than exact chunk
sizes: a chunk that ends mid-sentence produces a snippet the user cannot read.

Every chunk carries the page number it came from and the heading trail that was
in force at that point, so a citation can say "page 4, Pricing > Enterprise".
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.loaders import ParsedPage

logger = logging.getLogger(__name__)

#: Coarse-to-fine separators used by the recursive splitter.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")

#: Average characters per token for English prose; used by the token-aware variant.
CHARS_PER_TOKEN = 4

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{2,80})$")


@dataclass(slots=True)
class Chunk:
    """A retrievable slice of a document."""

    index: int
    content: str
    page: int | None = None
    heading: str | None = None
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.content)


def estimate_tokens(text: str) -> int:
    """Estimate token count.

    Uses ``tiktoken`` when it is installed (Gemini's tokenizer differs, but the
    ratio is close enough for budgeting) and a characters-per-token heuristic
    otherwise. This never calls the network.
    """
    if not text:
        return 0
    try:  # pragma: no cover - optional dependency
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:  # noqa: BLE001 - fall back to the heuristic
        return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def detect_heading(line: str) -> tuple[int, str] | None:
    """Return ``(level, title)`` when ``line`` looks like a heading, else ``None``."""
    stripped = line.strip()
    if not stripped or len(stripped) > 200:
        return None
    md = _MD_HEADING.match(stripped)
    if md:
        return len(md.group(1)), md.group(2).strip()
    numbered = _NUMBERED_HEADING.match(stripped)
    if numbered:
        return numbered.group(1).count(".") + 1, stripped
    if (
        stripped.isupper()
        and 3 <= len(stripped) <= 80
        and not stripped.endswith((".", ",", ";"))
        and any(ch.isalpha() for ch in stripped)
    ):
        return 1, stripped.title()
    return None


def _heading_trail(stack: list[tuple[int, str]]) -> str | None:
    """Render a heading stack as ``"Parent > Child"``."""
    return " > ".join(title for _, title in stack) if stack else None


def segment_by_heading(text: str) -> list[tuple[str | None, str]]:
    """Split ``text`` into ``(heading_trail, body)`` sections.

    Headings themselves are kept at the top of their section so the embedded
    text retains the topic words.
    """
    sections: list[tuple[str | None, str]] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    current = _heading_trail(stack)

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current, body))
        buffer.clear()

    for line in text.splitlines():
        heading = detect_heading(line)
        if heading is None:
            buffer.append(line)
            continue
        flush()
        level, title = heading
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        current = _heading_trail(stack)
        buffer.append(line)
    flush()
    return sections or [(None, text.strip())]


def _merge_splits(splits: Sequence[str], separator: str, size: int, overlap: int) -> list[str]:
    """Greedily pack ``splits`` into ``size``-bounded strings with ``overlap`` carry-over."""
    merged: list[str] = []
    window: list[str] = []
    total = 0
    sep_len = len(separator)

    for piece in splits:
        piece_len = len(piece)
        addition = piece_len + (sep_len if window else 0)
        if window and total + addition > size:
            merged.append(separator.join(window).strip())
            # Drop from the front until the carried tail fits the overlap budget
            # and the incoming piece will fit.
            while window and (total > overlap or total + addition > size):
                removed = window.pop(0)
                total -= len(removed) + (sep_len if window else 0)
        window.append(piece)
        total += addition if len(window) > 1 else piece_len

    if window:
        merged.append(separator.join(window).strip())
    return [item for item in merged if item]


def split_text(
    text: str,
    size: int,
    overlap: int,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[str]:
    """Recursively split ``text`` into pieces of at most ``size`` characters.

    Args:
        text: Text to split.
        size: Maximum characters per piece.
        overlap: Characters of trailing context repeated in the next piece.
        separators: Coarse-to-fine boundaries to try.

    Returns:
        Non-empty pieces in document order.

    Raises:
        ValueError: If ``overlap`` is not smaller than ``size``.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    separator = separators[-1]
    remaining: Sequence[str] = ()
    for position, candidate in enumerate(separators):
        if candidate == "":
            separator = ""
            remaining = ()
            break
        if candidate in text:
            separator = candidate
            remaining = separators[position + 1 :]
            break

    pieces = text.split(separator) if separator else list(text)
    output: list[str] = []
    batch: list[str] = []
    for piece in pieces:
        if len(piece) <= size:
            batch.append(piece)
            continue
        if batch:
            output.extend(_merge_splits(batch, separator, size, overlap))
            batch = []
        if remaining:
            output.extend(split_text(piece, size, overlap, remaining))
        else:  # no finer separator left: hard-cut with overlap
            step = max(1, size - overlap)
            output.extend(piece[start : start + size] for start in range(0, len(piece), step))
    if batch:
        output.extend(_merge_splits(batch, separator, size, overlap))
    return [item for item in output if item.strip()]


def chunk_pages(
    pages: Sequence[ParsedPage],
    *,
    size: int = 1000,
    overlap: int = 150,
    min_chunk_chars: int = 40,
) -> list[Chunk]:
    """Chunk parsed pages, preserving page numbers and heading trails.

    Args:
        pages: Output of :func:`app.ingestion.loaders.load_document`.
        size: Maximum characters per chunk.
        overlap: Characters repeated between neighbouring chunks.
        min_chunk_chars: Fragments shorter than this are appended to the
            previous chunk on the same page rather than stored alone -- a
            two-word chunk is noise in the index.

    Returns:
        Chunks numbered from 0 across the whole document.
    """
    chunks: list[Chunk] = []
    for page in pages:
        if page.is_empty:
            continue
        for heading, body in segment_by_heading(page.text):
            carried = ""
            for piece in split_text(body, size, overlap):
                if carried:
                    piece = f"{carried}\n{piece}".strip()
                    carried = ""
                if len(piece) < min_chunk_chars:
                    # Too small to stand alone: fold it into the previous chunk
                    # from the same page, or carry it into the next one.
                    if chunks and chunks[-1].page == page.page_number:
                        previous = chunks[-1]
                        previous.content = f"{previous.content}\n{piece}"
                        previous.token_count = estimate_tokens(previous.content)
                    else:
                        carried = piece
                    continue
                chunks.append(
                    Chunk(
                        index=len(chunks),
                        content=piece,
                        page=page.page_number,
                        heading=heading,
                        token_count=estimate_tokens(piece),
                        metadata={
                            "parser": page.metadata.get("parser", "unknown"),
                            **{
                                key: value
                                for key, value in page.metadata.items()
                                if key in {"sheet", "slide_title", "row_offset"}
                            },
                        },
                    )
                )
            if carried:
                if chunks and chunks[-1].page == page.page_number:
                    previous = chunks[-1]
                    previous.content = f"{previous.content}\n{carried}"
                    previous.token_count = estimate_tokens(previous.content)
                else:
                    chunks.append(
                        Chunk(
                            index=len(chunks),
                            content=carried,
                            page=page.page_number,
                            heading=heading,
                            token_count=estimate_tokens(carried),
                            metadata={"parser": page.metadata.get("parser", "unknown")},
                        )
                    )
    logger.info("Produced %d chunks from %d pages", len(chunks), len(pages))
    return chunks


def token_aware_chunk_pages(
    pages: Sequence[ParsedPage],
    *,
    max_tokens: int = 256,
    overlap_tokens: int = 40,
) -> list[Chunk]:
    """Chunk by an approximate *token* budget rather than characters.

    Useful when the embedding model's input limit is the binding constraint.
    Character budgets are derived from the token budget, then any chunk that
    still overshoots is split again and re-indexed.

    Args:
        pages: Parsed pages.
        max_tokens: Target maximum tokens per chunk.
        overlap_tokens: Token overlap between neighbours.

    Returns:
        Chunks whose ``token_count`` is at or below ``max_tokens`` (subject to
        the accuracy of :func:`estimate_tokens`).

    Raises:
        ValueError: If ``overlap_tokens`` is not smaller than ``max_tokens``.
    """
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    char_size = max_tokens * CHARS_PER_TOKEN
    char_overlap = overlap_tokens * CHARS_PER_TOKEN
    coarse = chunk_pages(pages, size=char_size, overlap=char_overlap)

    refined: list[Chunk] = []
    for chunk in coarse:
        if chunk.token_count <= max_tokens:
            chunk.index = len(refined)
            refined.append(chunk)
            continue
        ratio = max(1.0, chunk.token_count / max(1, len(chunk.content) / CHARS_PER_TOKEN))
        tighter = max(100, int(char_size / ratio))
        for piece in split_text(chunk.content, tighter, min(char_overlap, tighter - 1)):
            refined.append(
                Chunk(
                    index=len(refined),
                    content=piece,
                    page=chunk.page,
                    heading=chunk.heading,
                    token_count=estimate_tokens(piece),
                    metadata=dict(chunk.metadata),
                )
            )
    return refined
