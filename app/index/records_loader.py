"""Helpers for loading curated staff records into `StaffRecord` objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from app.config_loader import StaffEntry, load_staff_entries

from .models import SourceLink, StaffRecord


def load_curated_records(
    *,
    staff_yaml_path: Path | None = None,
    records_jsonl_path: Path | None = None,
) -> list[StaffRecord]:
    """Merge staff metadata (YAML) with narrative summaries (JSONL)."""

    staff_entries = load_staff_entries(staff_yaml_path)
    summaries_by_slug = _load_records_jsonl(records_jsonl_path)

    missing_slugs: list[str] = []
    records: list[StaffRecord] = []

    for entry in staff_entries:
        slug = _slug_from_profile_url(entry.profile_url)
        raw_summary = summaries_by_slug.get(slug)
        if raw_summary is None:
            missing_slugs.append(slug)
            continue

        summary_text = str(raw_summary.get("summary") or "").strip()
        if not summary_text:
            missing_slugs.append(slug)
            continue

        title = str(raw_summary.get("title") or "").strip() or "Fagperson"
        json_sources = raw_summary.get("sources") or []
        merged_sources = _dedupe_preserve_order([*entry.sources, *json_sources])

        record = StaffRecord(
            slug=slug,
            name=entry.name,
            title=title,
            department=entry.department,
            profile_url=entry.profile_url,
            summary=summary_text,
            sources=[SourceLink(url=url) for url in merged_sources],
            tags=list(entry.tags),
        )
        records.append(record)

    if missing_slugs:
        slugs_str = ", ".join(sorted(set(missing_slugs)))
        raise ValueError(
            "Fant ikke oppsummeringer for følgende slugs i data/staff_records.jsonl: "
            f"{slugs_str}. Kjør `python -m app.index.refresh_staff` for å regenerere filen,"
            " eller oppdater JSONL manuelt."
        )

    return records


def _slug_from_profile_url(url: str) -> str:
    trimmed = url.strip().rstrip("/")
    if not trimmed:
        raise ValueError("Profil-URL kan ikke være tom.")
    return trimmed.rsplit("/", 1)[-1]


def _load_records_jsonl(path: Path | None) -> dict[str, dict[str, object]]:
    target = path or Path("data/staff_records.jsonl")
    if not target.exists():
        raise FileNotFoundError(
            f"Mangler {target}. Kjør `python -m app.index.refresh_staff` for å generere filen."
        )

    summaries: dict[str, dict[str, object]] = {}
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            slug = str(payload.get("slug") or "").strip()
            if not slug:
                raise ValueError(f"Ugyldig JSONL-linje uten slug: {text[:80]}...")
            summaries[slug] = payload
    return summaries


def _dedupe_preserve_order(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        normalized = url.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered
