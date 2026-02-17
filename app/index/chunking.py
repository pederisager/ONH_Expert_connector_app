"""Utilities for turning staff profile text into embed-ready chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from .models import Chunk

WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


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
        min_chunk_tokens: int = 1,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size.")
        if min_chunk_tokens <= 0:
            raise ValueError("min_chunk_tokens must be >= 1.")
        if min_chunk_tokens > chunk_size:
            raise ValueError("min_chunk_tokens must be <= chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks
        self.min_chunk_tokens = min_chunk_tokens
        self.tokenizer = tokenizer or SimpleTokenizer()

    def chunk_text(
        self,
        *,
        staff_slug: str,
        text: str,
        source_url: str,
        metadata: dict[str, object] | None = None,
        source_namespace: str = "profile",
        start_index: int = 0,
        max_chunks: int | None = None,
        min_chunk_tokens: int | None = None,
        allow_short_single_chunk: bool = True,
    ) -> list[Chunk]:
        if start_index < 0:
            raise ValueError("start_index must be >= 0.")
        effective_min_tokens = (
            self.min_chunk_tokens if min_chunk_tokens is None else min_chunk_tokens
        )
        if effective_min_tokens <= 0:
            raise ValueError("min_chunk_tokens must be >= 1.")
        if effective_min_tokens > self.chunk_size:
            raise ValueError("min_chunk_tokens must be <= chunk_size.")
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return []

        step = self.chunk_size - self.chunk_overlap
        total = len(tokens)
        chunks: list[Chunk] = []
        seen_text_signatures: set[str] = set()
        namespace = _normalize_source_namespace(source_namespace)
        chunk_limit = self.max_chunks if max_chunks is None else max_chunks

        for start in range(0, total, step):
            end = min(start + self.chunk_size, total)
            window = tokens[start:end]
            if not window:
                continue
            # Preserve short single-source documents when explicitly allowed while
            # still dropping tiny tail windows.
            if len(window) < effective_min_tokens:
                if chunks:
                    continue
                if not allow_short_single_chunk:
                    continue
            global_index = start_index + len(chunks)
            chunk_id = f"{staff_slug}-{namespace}-{global_index:04d}"
            chunk_text = self.tokenizer.detokenize(window)
            text_signature = _chunk_text_signature(chunk_text)
            if text_signature in seen_text_signatures:
                continue
            seen_text_signatures.add(text_signature)
            chunk = Chunk(
                staff_slug=staff_slug,
                chunk_id=chunk_id,
                text=chunk_text,
                order=global_index,
                token_count=len(window),
                source_url=source_url,
                metadata=dict(metadata or {}),
            )
            chunks.append(chunk)
            if chunk_limit is not None and len(chunks) >= chunk_limit:
                break

        return chunks

    def estimate_tokens(self, text: str) -> int:
        """Return an estimated token count for planning chunk distribution."""
        return len(self.tokenizer.tokenize(text))


def _normalize_source_namespace(value: str) -> str:
    normalized = NON_ALNUM_RE.sub("-", (value or "").strip().lower()).strip("-")
    return normalized or "source"


def _chunk_text_signature(value: str) -> str:
    normalized = NON_WORD_RE.sub(" ", (value or "").casefold())
    return normalize_text(normalized)
