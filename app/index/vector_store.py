"""Lightweight vector store for local development and testing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .embeddings import normalize_embeddings
from .models import Chunk


VECTORS_FILENAME = "vectors.npy"
METADATA_FILENAME = "metadata.json"


@dataclass(slots=True)
class SearchResult:
    chunk: Chunk
    score: float


class LocalVectorStore:
    """Stores embeddings on disk and provides cosine-similarity search."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        self._dimension: int | None = None
        self._load_if_present()

    def _load_if_present(self) -> None:
        vectors_path = self.index_dir / VECTORS_FILENAME
        metadata_path = self.index_dir / METADATA_FILENAME
        if not vectors_path.exists() or not metadata_path.exists():
            return
        vectors = np.load(vectors_path)
        with metadata_path.open("r", encoding="utf-8") as handle:
            raw_chunks = json.load(handle)
        self._vectors = vectors.astype("float32")
        self._dimension = self._vectors.shape[1]
        self._chunks = [self._chunk_from_dict(item) for item in raw_chunks]

    def add(self, embeddings: np.ndarray, chunks: Sequence[Chunk]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")
        if len(chunks) == 0:
            return

        embeddings = embeddings.astype("float32")

        if self._vectors is None:
            self._vectors = embeddings
            self._chunks = list(chunks)
            self._dimension = embeddings.shape[1]
        else:
            if embeddings.shape[1] != self._dimension:
                raise ValueError("Embedding dimensionality does not match existing index.")
            self._vectors = np.vstack([self._vectors, embeddings])
            self._chunks.extend(chunks)

    def clear(self) -> None:
        self._vectors = None
        self._chunks = []
        self._dimension = None

    def persist(self) -> None:
        if self._vectors is None:
            raise RuntimeError("No vectors to persist.")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / VECTORS_FILENAME, self._vectors)
        with (self.index_dir / METADATA_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump([self._chunk_to_dict(chunk) for chunk in self._chunks], handle, ensure_ascii=False, indent=2)

    def search(
        self,
        query: np.ndarray,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        if self._vectors is None or not self._chunks:
            return []
        if query.ndim != 1:
            raise ValueError("query must be a 1D embedding vector.")
        if self._dimension is None or query.shape[0] != self._dimension:
            raise ValueError("query dimensionality does not match the index.")

        normalized_index = normalize_embeddings(self._vectors)
        normalized_query = normalize_embeddings(query.reshape(1, -1))[0]
        scores = normalized_index @ normalized_query

        top_indices = scores.argsort()[::-1]
        results: list[SearchResult] = []

        for idx in top_indices[:top_k]:
            score = float(scores[idx])
            if score < min_score:
                continue
            results.append(SearchResult(chunk=self._chunks[idx], score=score))

        return results

    def __len__(self) -> int:
        return len(self._chunks)

    @staticmethod
    def _chunk_to_dict(chunk: Chunk) -> dict[str, object]:
        return {
            "staff_slug": chunk.staff_slug,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "order": chunk.order,
            "token_count": chunk.token_count,
            "source_url": chunk.source_url,
            "metadata": chunk.metadata,
        }

    @staticmethod
    def _chunk_from_dict(payload: dict[str, object]) -> Chunk:
        return Chunk(
            staff_slug=str(payload["staff_slug"]),
            chunk_id=str(payload["chunk_id"]),
            text=str(payload["text"]),
            order=int(payload["order"]),
            token_count=int(payload["token_count"]),
            source_url=str(payload["source_url"]),
            metadata=dict(payload.get("metadata") or {}),
        )

