"""Helpers for loading curated staff records into `StaffRecord` objects."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from app.config_loader import StaffEntry, load_staff_entries

from .models import CristinResultSnippet, SourceLink, StaffRecord


LOGGER = logging.getLogger(__name__)
DEFAULT_CRISTIN_RESULTS_PATH = Path("data/cristin/results.jsonl")
MAX_CRISTIN_RESULTS = 5


def load_curated_records(
    *,
    staff_yaml_path: Path | None = None,
    records_jsonl_path: Path | None = None,
    cristin_results_path: Path | None = None,
    max_cristin_results: int = MAX_CRISTIN_RESULTS,
) -> list[StaffRecord]:
    """Merge staff metadata (YAML) with narrative summaries (JSONL)."""

    staff_entries = load_staff_entries(staff_yaml_path)
    summaries_by_slug = _load_records_jsonl(records_jsonl_path)
    cristin_by_employee = _load_cristin_results(
        cristin_results_path,
        max_results=max(1, max_cristin_results),
    )

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

        cristin_results: list[CristinResultSnippet] = []
        nva_id = _extract_nva_person_id(entry)
        if nva_id:
            cristin_results = list(cristin_by_employee.get(nva_id, []))

        record = StaffRecord(
            slug=slug,
            name=entry.name,
            title=title,
            department=entry.department,
            profile_url=entry.profile_url,
            summary=summary_text,
            sources=[SourceLink(url=url) for url in merged_sources],
            tags=list(entry.tags),
            cristin_results=cristin_results,
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


def _load_cristin_results(
    path: Path | None,
    *,
    max_results: int,
) -> dict[str, list[CristinResultSnippet]]:
    target = path or DEFAULT_CRISTIN_RESULTS_PATH
    if not target.exists():
        LOGGER.info("Fant ikke Cristin-data på %s. Hopper over forskningsresultater i RAG.", target)
        return {}

    grouped: dict[str, list[CristinResultSnippet]] = {}
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            employee_id = _coerce_str(payload.get("employee_id"))
            result_id = _coerce_str(payload.get("cristin_result_id"))
            if not employee_id or not result_id:
                continue
            snippet = CristinResultSnippet(
                cristin_result_id=result_id,
                title=_coerce_str(payload.get("title")),
                summary=_coerce_str(payload.get("summary")),
                venue=_coerce_str(payload.get("venue")),
                category_name=_coerce_str(payload.get("category_name")),
                year=_coerce_str(payload.get("year_published")),
                tags=[tag for tag in payload.get("tags", []) if isinstance(tag, str)],
                source_url=_normalize_cristin_source_url(
                    _coerce_str(payload.get("source_url")),
                    result_id,
                ),
            )
            grouped.setdefault(employee_id, []).append(snippet)

    for employee_id, snippets in grouped.items():
        snippets.sort(key=_cristin_sort_key, reverse=True)
        grouped[employee_id] = snippets[:max_results]
    return grouped


def _extract_nva_person_id(entry: StaffEntry) -> str | None:
    for source in entry.sources:
        parsed = urlparse(source)
        host = parsed.netloc.lower()
        if "nva.sikt.no" not in host:
            continue
        parts = [part for part in Path(parsed.path).parts if part]
        if not parts:
            continue
        if "research-profile" in parts:
            candidate = parts[-1]
            if candidate:
                return candidate
    return None


def _normalize_cristin_source_url(url: str | None, result_id: str) -> str:
    if url:
        return url
    return f"https://api.cristin.no/v2/results/{result_id}"


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _cristin_sort_key(snippet: CristinResultSnippet) -> tuple[int, int, int, str]:
    has_summary = 1 if snippet.summary else 0
    tag_bonus = min(len(snippet.tags), 5)
    year_value = _year_to_int(snippet.year)
    title = snippet.title or ""
    return (has_summary, tag_bonus, year_value, title.lower())


def _year_to_int(year: str | None) -> int:
    if not year:
        return 0
    try:
        return int(year)
    except ValueError:
        return 0


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
