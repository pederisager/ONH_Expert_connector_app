"""Data models for Cristin ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ContributorSummary:
    role: str | None = None
    first_name: str | None = None
    surname: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ContributorSummary":
        role = None
        roles = payload.get("roles")
        if isinstance(roles, list) and roles:
            role_entry = roles[0]
            if isinstance(role_entry, dict):
                role = role_entry.get("role_code") or role_entry.get("role_name")
        return cls(
            role=role,
            first_name=payload.get("first_name"),
            surname=payload.get("surname"),
        )


@dataclass(slots=True)
class ResultRecord:
    employee_id: str
    employee_name: str
    cristin_result_id: str
    title: str | None
    summary: str | None
    venue: str | None
    category_code: str | None
    category_name: str | None
    year_published: str | None
    tags: list[str] = field(default_factory=list)
    contributors: list[ContributorSummary] = field(default_factory=list)
    source_url: str | None = None
    extra_context: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "cristin_result_id": self.cristin_result_id,
            "title": self.title,
            "summary": self.summary,
            "venue": self.venue,
            "category_code": self.category_code,
            "category_name": self.category_name,
            "year_published": self.year_published,
            "tags": self.tags,
            "contributors": [
                {"role": contributor.role, "first_name": contributor.first_name, "surname": contributor.surname}
                for contributor in self.contributors
            ],
            "source_url": self.source_url,
            "extra_context": self.extra_context,
        }
