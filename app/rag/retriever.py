"""Vector-based retrieval interfaces used during matching."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from app.index.embeddings import EmbeddingBackend
from app.index.models import Chunk
from app.index.vector_store import LocalVectorStore, SearchResult


LOGGER = logging.getLogger("rag.embedding-retriever")


@dataclass(slots=True)
class RetrievalQuery:
    text: str
    department: str | None = None
    top_k: int = 5


@dataclass(slots=True)
class RetrievalResult:
    staff_slug: str
    score: float
    chunks: list[Chunk]
    staff_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class EmbeddingRetriever:
    """Uses a vector store to find relevant chunks for a query."""

    def __init__(
        self,
        *,
        vector_store: LocalVectorStore,
        embedder: EmbeddingBackend,
        min_score: float = 0.1,
        max_chunks_per_staff: int = 3,
        oversample_factor: int = 4,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.min_score = min_score
        self.max_chunks_per_staff = max_chunks_per_staff
        self.oversample_factor = max(1, oversample_factor)
        self._active = True
        self._disabled_reason: str | None = None

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        if not self._active:
            return []
        if not query.text.strip():
            return []

        query_vector = self.embedder.embed_one(query.text).astype(np.float32)
        if query_vector.ndim != 1:
            raise ValueError("embed_one must return a 1D vector")

        oversample = max(query.top_k * self.oversample_factor, query.top_k)
        try:
            raw_results = self.vector_store.search(
                query_vector, top_k=oversample, min_score=self.min_score
            )
        except ValueError as exc:
            self._active = False
            self._disabled_reason = str(exc)
            LOGGER.warning(
                "Vector store search disabled: %s. Rebuild the index to re-enable RAG.",
                exc,
            )
            return []
        grouped = self._group_results(raw_results, query)
        sorted_results = sorted(grouped.values(), key=lambda item: item.score, reverse=True)
        return sorted_results[: query.top_k]

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def _group_results(
        self, results: Sequence[SearchResult], query: RetrievalQuery
    ) -> dict[str, RetrievalResult]:
        grouped: dict[str, RetrievalResult] = {}
        for result in results:
            chunk = result.chunk
            if query.department and chunk.metadata.get("department") != query.department:
                continue

            entry = grouped.get(chunk.staff_slug)
            if entry is None:
                entry = RetrievalResult(
                    staff_slug=chunk.staff_slug,
                    score=result.score,
                    chunks=[chunk],
                    staff_name=str(chunk.metadata.get("name")) if chunk.metadata else None,
                    metadata={"department": chunk.metadata.get("department") if chunk.metadata else None},
                )
                grouped[chunk.staff_slug] = entry
            else:
                entry.score = max(entry.score, result.score)
                if len(entry.chunks) < self.max_chunks_per_staff:
                    entry.chunks.append(chunk)

        return grouped
