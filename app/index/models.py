"""Data structures shared by the index-building pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class CristinResultSnippet:
    """Lightweight representation of a Cristin/NVA research result."""

    cristin_result_id: str
    title: str | None = None
    summary: str | None = None
    venue: str | None = None
    category_name: str | None = None
    year: str | None = None
    tags: list[str] = field(default_factory=list)
    source_url: str | None = None

    def as_text(self) -> str:
        parts: list[str] = []
        if self.title:
            parts.append(self.title.strip())
        details: list[str] = []
        if self.year:
            details.append(self.year)
        if self.venue:
            details.append(self.venue)
        if self.category_name and self.category_name not in details:
            details.append(self.category_name)
        if details:
            parts.append(f"({', '.join(details)})")
        if self.summary:
            parts.append(self.summary.strip())
        if self.tags:
            parts.append(f"Nøkkelord: {', '.join(self.tags)}.")
        return " ".join(part for part in parts if part).strip()


@dataclass(slots=True)
class SourceLink:
    """Represents a single source URL associated with a staff member."""

    url: str
    kind: str = "web"


@dataclass(slots=True)
class StaffRecord:
    """Canonical information required to index a staff member."""

    slug: str
    name: str
    title: str
    department: str
    profile_url: str
    summary: str
    sources: list[SourceLink] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    cristin_results: list[CristinResultSnippet] = field(default_factory=list)

    def primary_source(self) -> SourceLink | None:
        return self.sources[0] if self.sources else None


@dataclass(slots=True)
class Chunk:
    """Normalized chunk ready for embedding and vector storage."""

    staff_slug: str
    chunk_id: str
    text: str
    order: int
    token_count: int
    source_url: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Mapping[str, Any]:
        payload = {
            "staff_slug": self.staff_slug,
            "chunk_id": self.chunk_id,
            "order": self.order,
            "token_count": self.token_count,
            "source_url": self.source_url,
        }
        payload.update(self.metadata)
        return payload


@dataclass(slots=True)
class IndexPaths:
    """Helper for organizing on-disk artifacts produced by indexing."""

    root: Path

    @property
    def vectors_dir(self) -> Path:
        return self.root / "vectors"

    @property
    def chunks_dir(self) -> Path:
        return self.root / "chunks"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"
