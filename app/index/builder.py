"""Index builder that transforms staff records into chunked embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .chunking import Chunker
from .embeddings import EmbeddingBackend
from .models import Chunk, IndexPaths, StaffRecord
from .vector_store import LocalVectorStore


@dataclass(slots=True)
class BuildSummary:
    processed_staff: int
    total_chunks: int
    skipped_staff: list[str]


class StaffIndexBuilder:
    """Coordinates chunk generation, embedding, and persistence."""

    def __init__(
        self,
        *,
        paths: IndexPaths,
        chunker: Chunker,
        embedder: EmbeddingBackend,
        vector_store: LocalVectorStore | None = None,
    ) -> None:
        self.paths = paths
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store or LocalVectorStore(paths.vectors_dir)

    def build(self, records: Sequence[StaffRecord]) -> BuildSummary:
        self._ensure_directories()

        all_chunks: list[Chunk] = []
        skipped: list[str] = []
        per_staff_stats: list[dict[str, object]] = []

        for record in records:
            chunks = self._chunks_for_record(record)
            if not chunks:
                skipped.append(record.slug)
                continue
            all_chunks.extend(chunks)
            per_staff_stats.append({"slug": record.slug, "chunk_count": len(chunks)})
            self._write_chunks_snapshot(record.slug, chunks)

        if all_chunks:
            embeddings = self.embedder.embed([chunk.text for chunk in all_chunks])
            if embeddings.shape[0] != len(all_chunks):
                raise ValueError("Embedder returned mismatched number of vectors.")
            self.vector_store.clear()
            self.vector_store.add(embeddings, all_chunks)
            self.vector_store.persist()
        else:
            self.vector_store.clear()

        self._write_manifest(per_staff_stats, len(all_chunks))

        return BuildSummary(
            processed_staff=len(per_staff_stats),
            total_chunks=len(all_chunks),
            skipped_staff=skipped,
        )

    def _ensure_directories(self) -> None:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.vectors_dir.mkdir(parents=True, exist_ok=True)
        self.paths.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.paths.manifests_dir.mkdir(parents=True, exist_ok=True)

    def _chunks_for_record(self, record: StaffRecord) -> list[Chunk]:
        metadata = {
            "name": record.name,
            "department": record.department,
            "title": record.title,
            "tags": record.tags,
            "profile_url": record.profile_url,
        }
        source_url = record.primary_source().url if record.primary_source() else ""
        chunks = self.chunker.chunk_text(
            staff_slug=record.slug,
            text=record.summary,
            source_url=source_url,
            metadata={
                "department": record.department,
                "title": record.title,
                "profile_url": record.profile_url,
            },
        )
        # Attach richer metadata post-chunking to avoid shared references.
        for chunk in chunks:
            chunk.metadata.setdefault("name", record.name)
            chunk.metadata.setdefault("tags", record.tags)
        return chunks

    def _write_chunks_snapshot(self, slug: str, chunks: Iterable[Chunk]) -> None:
        payload = [self._chunk_to_dict(chunk) for chunk in chunks]
        target = self.paths.chunks_dir / f"{slug}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_manifest(self, stats: list[dict[str, object]], total_chunks: int) -> None:
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "staff": stats,
            "total_chunks": total_chunks,
        }
        target = self.paths.manifests_dir / "manifest.json"
        target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _chunk_to_dict(chunk: Chunk) -> dict[str, object]:
        data = {
            "staff_slug": chunk.staff_slug,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "order": chunk.order,
            "token_count": chunk.token_count,
            "source_url": chunk.source_url,
        }
        if chunk.metadata:
            data["metadata"] = chunk.metadata
        return data


class DummyEmbeddingBackend:
    """Fallback embedder used when no real embedding model is available."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for index, text in enumerate(texts):
            seed = float(len(text) + index)
            vector = np.full(self.dimension, seed, dtype=np.float32)
            vectors.append(vector)
        if not vectors:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.vstack(vectors)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
