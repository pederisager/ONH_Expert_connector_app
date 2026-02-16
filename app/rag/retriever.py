"""Vector-based retrieval interfaces used during matching."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from app.index.embeddings import EmbeddingBackend
from app.index.models import Chunk
from app.index.vector_store import LocalVectorStore, SearchResult

LOGGER = logging.getLogger("rag.embedding-retriever")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", re.UNICODE)


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
        semantic_weight: float = 0.85,
        lexical_weight: float = 0.15,
        source_weights: dict[str, float] | None = None,
        max_chunks_per_source: dict[str, int] | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.min_score = min_score
        self.max_chunks_per_staff = max_chunks_per_staff
        self.oversample_factor = max(1, oversample_factor)
        self.semantic_weight = max(0.0, float(semantic_weight))
        self.lexical_weight = max(0.0, float(lexical_weight))
        self.source_weights = {
            key.lower(): max(0.0, float(value))
            for key, value in (source_weights or {}).items()
        }
        self.max_chunks_per_source = {
            key.lower(): max(0, int(value))
            for key, value in (max_chunks_per_source or {}).items()
        }
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
        lexical_scores = self._compute_lexical_scores(
            raw_results=raw_results,
            query_text=query.text,
        )
        weighted_results = []
        for result in raw_results:
            lexical_score = lexical_scores.get(result.chunk.chunk_id, 0.0)
            semantic_score = max(0.0, float(result.score))
            hybrid_score = (
                self.semantic_weight * semantic_score
                + self.lexical_weight * lexical_score
            )
            source_weight = self._source_weight(result.chunk)
            weighted_score = hybrid_score * source_weight
            weighted_results.append(
                (
                    result,
                    weighted_score,
                    semantic_score,
                    lexical_score,
                    hybrid_score,
                    source_weight,
                )
            )
        grouped = self._group_results(weighted_results, query)
        sorted_results = sorted(
            grouped.values(), key=lambda item: item.score, reverse=True
        )
        return sorted_results[: query.top_k]

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def _group_results(
        self,
        results: Sequence[
            tuple[SearchResult, float, float, float, float, float]
        ],
        query: RetrievalQuery,
    ) -> dict[str, RetrievalResult]:
        grouped: dict[str, RetrievalResult] = {}
        source_counts_by_staff: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        ordered = sorted(results, key=lambda item: item[1], reverse=True)
        for (
            result,
            weighted_score,
            semantic_score,
            lexical_score,
            hybrid_score,
            source_weight,
        ) in ordered:
            chunk = result.chunk
            chunk_metadata = chunk.metadata
            if (
                query.department
                and chunk_metadata.get("department") != query.department
            ):
                continue

            source_kind = self._source_kind(chunk)
            entry = grouped.get(chunk.staff_slug)
            if entry is None:
                department = (
                    chunk_metadata.get("department") if chunk_metadata else None
                )
                entry = RetrievalResult(
                    staff_slug=chunk.staff_slug,
                    score=weighted_score,
                    chunks=[],
                    staff_name=(
                        str(chunk_metadata.get("name")) if chunk_metadata else None
                    ),
                    metadata={
                        "department": department,
                        "semantic_score": semantic_score,
                        "lexical_score": lexical_score,
                        "hybrid_score": hybrid_score,
                        "source_weight": source_weight,
                    },
                )
                grouped[chunk.staff_slug] = entry
            if weighted_score > entry.score:
                entry.score = weighted_score
            entry.metadata["semantic_score"] = max(
                float(entry.metadata.get("semantic_score", 0.0)),
                semantic_score,
            )
            entry.metadata["lexical_score"] = max(
                float(entry.metadata.get("lexical_score", 0.0)),
                lexical_score,
            )
            entry.metadata["hybrid_score"] = max(
                float(entry.metadata.get("hybrid_score", 0.0)),
                hybrid_score,
            )
            entry.metadata["source_weight"] = max(
                float(entry.metadata.get("source_weight", 0.0)),
                source_weight,
            )

            if len(entry.chunks) >= self.max_chunks_per_staff:
                continue
            per_source_limit = self.max_chunks_per_source.get(source_kind)
            current_source_count = source_counts_by_staff[chunk.staff_slug][source_kind]
            if per_source_limit is not None and current_source_count >= per_source_limit:
                continue
            entry.chunks.append(chunk)
            source_counts_by_staff[chunk.staff_slug][source_kind] += 1

        return grouped

    def _source_weight(self, chunk: Chunk) -> float:
        source_kind = self._source_kind(chunk)
        return self.source_weights.get(source_kind, 1.0)

    def _compute_lexical_scores(
        self, *, raw_results: Sequence[SearchResult], query_text: str
    ) -> dict[str, float]:
        if not raw_results or self.lexical_weight <= 0:
            return {}

        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return {}
        query_terms = sorted(set(query_tokens))

        doc_tokens: list[list[str]] = [self._chunk_tokens(item.chunk) for item in raw_results]
        doc_counts: list[Counter[str]] = [Counter(tokens) for tokens in doc_tokens]
        doc_lengths = [len(tokens) for tokens in doc_tokens]
        if not doc_lengths:
            return {}
        avg_doc_len = sum(doc_lengths) / max(1, len(doc_lengths))

        term_doc_frequency = {
            term: sum(1 for counts in doc_counts if counts.get(term, 0) > 0)
            for term in query_terms
        }

        k1 = 1.5
        b = 0.75
        raw_scores: dict[str, float] = {}
        max_score = 0.0
        total_docs = len(doc_counts)

        for result, counts, doc_len in zip(raw_results, doc_counts, doc_lengths):
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf <= 0:
                    continue
                doc_freq = term_doc_frequency.get(term, 0)
                idf = math.log1p(
                    (total_docs - doc_freq + 0.5) / (doc_freq + 0.5)
                )
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (
                    1.0 - b + b * (doc_len / max(avg_doc_len, 1.0))
                )
                score += idf * (numerator / max(denominator, 1e-9))
            if score > 0:
                raw_scores[result.chunk.chunk_id] = score
                max_score = max(max_score, score)

        if max_score <= 0:
            return {}
        return {
            chunk_id: score / max_score for chunk_id, score in raw_scores.items()
        }

    @staticmethod
    def _chunk_tokens(chunk: Chunk) -> list[str]:
        metadata = chunk.metadata or {}
        parts = [chunk.text]
        source_title = metadata.get("source_title")
        if isinstance(source_title, str) and source_title.strip():
            parts.append(source_title)
        tags = metadata.get("tags")
        if isinstance(tags, list):
            parts.extend(str(tag) for tag in tags if isinstance(tag, str) and tag.strip())
        return EmbeddingRetriever._tokenize(" ".join(parts))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text or "")]

    @staticmethod
    def _source_kind(chunk: Chunk) -> str:
        metadata = chunk.metadata or {}
        source_kind = str(metadata.get("source_kind") or "profile")
        return source_kind.strip().lower() or "profile"
