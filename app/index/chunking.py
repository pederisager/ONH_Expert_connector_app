"""Utilities for turning staff profile text into embed-ready chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from .models import Chunk

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Collapse whitespace and strip leading/trailing spacing."""
    collapsed = WHITESPACE_RE.sub(" ", value).strip()
    return collapsed


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...

    def detokenize(self, tokens: Sequence[str]) -> str: ...


@dataclass(slots=True)
class SimpleTokenizer:
    """Lightweight tokenizer that works without external dependencies."""

    lowercase: bool = False

    def tokenize(self, text: str) -> list[str]:
        cleaned = normalize_text(text)
        if not cleaned:
            return []
        if self.lowercase:
            cleaned = cleaned.lower()
        return cleaned.split(" ")

    def detokenize(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens)


class Chunker:
    """Splits text into overlapping token windows."""

    def __init__(
        self,
        *,
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        max_chunks: int | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks
        self.tokenizer = tokenizer or SimpleTokenizer()

    def chunk_text(
        self,
        *,
        staff_slug: str,
        text: str,
        source_url: str,
        metadata: dict[str, str] | None = None,
    ) -> list[Chunk]:
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return []

        step = self.chunk_size - self.chunk_overlap
        total = len(tokens)
        chunks: list[Chunk] = []

        for index, start in enumerate(range(0, total, step)):
            end = min(start + self.chunk_size, total)
            window = tokens[start:end]
            if not window:
                continue
            chunk_id = f"{staff_slug}-{index:04d}"
            chunk_text = self.tokenizer.detokenize(window)
            chunk = Chunk(
                staff_slug=staff_slug,
                chunk_id=chunk_id,
                text=chunk_text,
                order=index,
                token_count=len(window),
                source_url=source_url,
                metadata=dict(metadata or {}),
            )
            chunks.append(chunk)
            if self.max_chunks is not None and len(chunks) >= self.max_chunks:
                break

        return chunks

    def estimate_tokens(self, text: str) -> int:
        """Return an estimated token count for planning chunk distribution."""
        return len(self.tokenizer.tokenize(text))
