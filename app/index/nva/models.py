"""Data models for NVA ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NvaContributor:
    name: str | None = None
    role: str | None = None
    identifier: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "NvaContributor":
        identity = payload.get("identity") if isinstance(payload, dict) else None
        name = None
        identifier = None
        if isinstance(identity, dict):
            name = identity.get("name")
            identifier = identity.get("id")
        role = None
        role_payload = payload.get("role") if isinstance(payload, dict) else None
        if isinstance(role_payload, dict):
            role = role_payload.get("type") or role_payload.get("code")
        return cls(name=name, role=role, identifier=identifier)


@dataclass(slots=True)
class NvaPublicationRecord:
    employee_id: str
    employee_name: str
    publication_id: str
    title: str | None
    abstract: str | None
    venue: str | None
    category: str | None
    year: str | None
    tags: list[str] = field(default_factory=list)
    contributors: list[NvaContributor] = field(default_factory=list)
    source_url: str | None = None
    language: str | None = None
    doi: str | None = None
    handle: str | None = None
    extra_context: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "nva_publication_id": self.publication_id,
            "title": self.title,
            "abstract": self.abstract,
            "venue": self.venue,
            "category": self.category,
            "year": self.year,
            "tags": self.tags,
            "contributors": [
                {
                    "name": contributor.name,
                    "role": contributor.role,
                    "identifier": contributor.identifier,
                }
                for contributor in self.contributors
            ],
            "source_url": self.source_url,
            "language": self.language,
            "doi": self.doi,
            "handle": self.handle,
            "extra_context": self.extra_context,
        }
