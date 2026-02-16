"""Index builder that transforms staff records into chunked embeddings."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np

from .chunking import Chunker
from .embeddings import EmbeddingBackend
from .models import Chunk, IndexPaths, StaffRecord
from .staff_info_loader import (
    StaffInfo,
    extract_method_tags,
    synthetic_page_text,
)
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
        staff_info: dict[str, StaffInfo] | None = None,
        max_chunks_per_source: dict[str, int] | None = None,
    ) -> None:
        self.paths = paths
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store or LocalVectorStore(paths.vectors_dir)
        self.staff_info = staff_info or {}
        self.max_chunks_per_source = {
            key.lower(): value
            for key, value in (max_chunks_per_source or {}).items()
            if value >= 0
        }

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
        source_offsets: dict[str, int] = defaultdict(int)
        source_counts: dict[str, int] = defaultdict(int)
        primary_source = record.primary_source()
        summary_source_url = (
            primary_source.url if primary_source else record.profile_url
        )
        info = self.staff_info.get(record.name.lower())

        extra_tags: list[str] = []
        if info:
            extra_tags.extend(info.expertise_domains)
            extra_tags.extend(info.research_focus)
            extra_tags.extend(extract_method_tags(info))

        default_tags = list(record.tags)
        base_metadata = {
            "department": record.department,
            "title": record.title,
            "profile_url": record.profile_url,
            "source_title": record.name,
            "source_kind": "profile",
            "tags": list(default_tags),
        }

        # Synthetic staff_info page (high-signal curated data).
        if info:
            synthetic_text = synthetic_page_text(info, include_methods=True)
            synthetic_tags = list(dict.fromkeys([*default_tags, *extra_tags]))
            synthetic_metadata = {
                **base_metadata,
                "source_kind": "staffinfo",
                "source_title": record.name,
                "tags": synthetic_tags,
            }
            synthetic_chunks = self._chunk_with_source_budget(
                source_kind="staffinfo",
                source_namespace="staffinfo",
                staff_slug=record.slug,
                text=synthetic_text,
                source_url=f"staffinfo://{record.slug}",
                metadata=synthetic_metadata,
                source_offsets=source_offsets,
                source_counts=source_counts,
            )
            chunks.extend(synthetic_chunks)

        summary_chunks = self._chunk_with_source_budget(
            source_kind="profile",
            source_namespace="profile",
            staff_slug=record.slug,
            text=record.summary,
            source_url=summary_source_url,
            metadata=base_metadata,
            source_offsets=source_offsets,
            source_counts=source_counts,
        )
        chunks.extend(summary_chunks)

        if record.nva_publications:
            chunks.extend(
                self._chunks_from_nva_publications(
                    record,
                    source_offsets=source_offsets,
                    source_counts=source_counts,
                )
            )

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

    def _chunks_from_nva_publications(
        self,
        record: StaffRecord,
        *,
        source_offsets: dict[str, int],
        source_counts: dict[str, int],
    ) -> list[Chunk]:
        nva_chunks: list[Chunk] = []
        for result in record.nva_publications:
            text = result.as_text()
            if not text:
                continue
            source_url = result.source_url or record.profile_url
            combined_tags = (
                list(dict.fromkeys([*record.tags, *(result.tags or [])]))
                if (record.tags or result.tags)
                else []
            )
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
                self._chunk_with_source_budget(
                    source_kind="nva",
                    source_namespace="nva",
                    staff_slug=record.slug,
                    text=text,
                    source_url=source_url,
                    metadata=metadata,
                    source_offsets=source_offsets,
                    source_counts=source_counts,
                )
            )
        return nva_chunks

    def _chunk_with_source_budget(
        self,
        *,
        source_kind: str,
        source_namespace: str,
        staff_slug: str,
        text: str,
        source_url: str,
        metadata: dict[str, object],
        source_offsets: dict[str, int],
        source_counts: dict[str, int],
    ) -> list[Chunk]:
        normalized_kind = source_kind.lower()
        configured_limit = self.max_chunks_per_source.get(normalized_kind)
        remaining = None
        if configured_limit is not None:
            consumed = source_counts.get(normalized_kind, 0)
            remaining = max(0, configured_limit - consumed)
            if remaining == 0:
                return []

        new_chunks = self.chunker.chunk_text(
            staff_slug=staff_slug,
            text=text,
            source_url=source_url,
            metadata=metadata,
            source_namespace=source_namespace,
            start_index=source_offsets.get(normalized_kind, 0),
            max_chunks=remaining,
        )
        source_offsets[normalized_kind] = source_offsets.get(normalized_kind, 0) + len(
            new_chunks
        )
        source_counts[normalized_kind] = source_counts.get(normalized_kind, 0) + len(
            new_chunks
        )
        return new_chunks

    def _write_chunks_snapshot(self, slug: str, chunks: Iterable[Chunk]) -> None:
        payload = [self._chunk_to_dict(chunk) for chunk in chunks]
        target = self.paths.chunks_dir / f"{slug}.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _write_manifest(
        self, stats: list[dict[str, object]], total_chunks: int
    ) -> None:
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "staff": stats,
            "total_chunks": total_chunks,
        }
        target = self.paths.manifests_dir / "manifest.json"
        target.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

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
