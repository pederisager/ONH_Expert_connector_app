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
        chunks: list[Chunk] = []
        primary_source = record.primary_source()
        summary_source_url = primary_source.url if primary_source else record.profile_url
        base_metadata = {
            "department": record.department,
            "title": record.title,
            "profile_url": record.profile_url,
            "source_title": record.name,
            "source_kind": "profile",
            "tags": list(record.tags),
        }

        summary_chunks = self.chunker.chunk_text(
            staff_slug=record.slug,
            text=record.summary,
            source_url=summary_source_url,
            metadata=base_metadata,
        )
        chunks.extend(summary_chunks)

        if record.nva_publications:
            chunks.extend(self._chunks_from_nva_publications(record))

        default_tags = list(record.tags)
        for chunk in chunks:
            chunk.metadata.setdefault("name", record.name)
            existing_tags = chunk.metadata.get("tags")
            if isinstance(existing_tags, list) and existing_tags:
                # Preserve order while de-duplicating.
                seen: set[str] = set()
                deduped: list[str] = []
                for tag in existing_tags:
                    lowered = tag.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    deduped.append(tag)
                chunk.metadata["tags"] = deduped
            else:
                chunk.metadata["tags"] = list(default_tags)
        return chunks

    def _chunks_from_nva_publications(self, record: StaffRecord) -> list[Chunk]:
        nva_chunks: list[Chunk] = []
        for result in record.nva_publications:
            text = result.as_text()
            if not text:
                continue
            source_url = result.source_url or record.profile_url
            combined_tags = list(dict.fromkeys([*record.tags, *(result.tags or [])])) if (record.tags or result.tags) else []
            metadata = {
                "department": record.department,
                "title": record.title,
                "profile_url": record.profile_url,
                "source_title": result.title or "NVA-publikasjon",
                "source_kind": "nva",
                "nva_publication_id": result.publication_id,
                "nva_year": result.year,
                "nva_category": result.category,
                "venue": result.venue,
                "tags": combined_tags,
            }
            nva_chunks.extend(
                self.chunker.chunk_text(
                    staff_slug=record.slug,
                    text=text,
                    source_url=source_url,
                    metadata=metadata,
                )
            )
        return nva_chunks

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
