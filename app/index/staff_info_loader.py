"""Loader and helpers for staff_info.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

METHOD_KEYWORDS = {
    "kvalitativ",
    "kvalitative",
    "kvantitativ",
    "kvantitative",
    "ipa",
    "randomisert",
    "randomiserte",
    "survey",
    "meta-analyse",
    "meta analyse",
    "longitudinell",
    "eksperimentell",
    "klinisk metode",
    "case-studie",
    "casestudie",
    "mixed methods",
    "mixed-methods",
    "tverrsnitt",
}


@dataclass(slots=True)
class StaffInfo:
    name: str
    job_title: str = ""
    expertise_domains: list[str] = field(default_factory=list)
    teaching_courses: list[str] = field(default_factory=list)
    research_focus: list[str] = field(default_factory=list)
    other_relevant_expertise: list[str] = field(default_factory=list)


def load_staff_info(path: Path | str = "staff_info.json") -> dict[str, StaffInfo]:
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    entries = payload.get("staff", [])
    info: dict[str, StaffInfo] = {}
    for entry in entries:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        info[name.lower()] = StaffInfo(
            name=name,
            job_title=str(entry.get("job_title") or "").strip(),
            expertise_domains=_normalize_list(entry.get("expertise_domains")),
            teaching_courses=_normalize_list(entry.get("teaching_courses")),
            research_focus=_normalize_list(entry.get("research_focus")),
            other_relevant_expertise=_normalize_list(
                entry.get("other_relevant_expertise")
            ),
        )
    return info


def _normalize_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def extract_method_tags(info: StaffInfo) -> list[str]:
    corpus = " ".join(
        [
            info.job_title,
            *info.teaching_courses,
            *info.research_focus,
            *info.expertise_domains,
            *info.other_relevant_expertise,
        ]
    ).lower()
    tags = []
    for keyword in METHOD_KEYWORDS:
        if keyword in corpus:
            tags.append(keyword)
    return list(dict.fromkeys(tags))


def synthetic_page_text(info: StaffInfo, include_methods: bool = True) -> str:
    parts: list[str] = []
    if info.job_title:
        parts.append(f"Rolle: {info.job_title}.")
    if info.teaching_courses:
        parts.append(f"Kurs: {', '.join(info.teaching_courses)}.")
    if info.research_focus:
        parts.append(f"Forskning: {', '.join(info.research_focus)}.")
    if info.expertise_domains:
        parts.append(f"Ekspertise: {', '.join(info.expertise_domains)}.")
    if include_methods:
        method_tags = extract_method_tags(info)
        if method_tags:
            parts.append(f"Metode: {', '.join(method_tags)}.")
    if info.other_relevant_expertise:
        parts.append(f"Annet: {', '.join(info.other_relevant_expertise)}.")
    return " ".join(parts).strip()
