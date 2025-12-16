"""Normalization helpers for NVA publication payloads."""

from __future__ import annotations

from typing import Any

from .models import NvaContributor, NvaPublicationRecord


def _select_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for candidate in value.values():
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    if isinstance(value, list):
        for candidate in value:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _extract_year(
    entity_description: dict[str, Any], payload: dict[str, Any]
) -> str | None:
    publication_date = entity_description.get("publicationDate")
    if isinstance(publication_date, dict):
        year = publication_date.get("year")
        if isinstance(year, str) and year.strip():
            return year.strip()
    for field in ("publishedDate", "createdDate", "modifiedDate"):
        value = payload.get(field)
        if isinstance(value, str) and len(value) >= 4:
            return value[:4]
    return None


def _extract_venue(reference: dict[str, Any]) -> str | None:
    publication_context = (
        reference.get("publicationContext") if isinstance(reference, dict) else None
    )
    if isinstance(publication_context, dict):
        publisher = publication_context.get("publisher")
        if isinstance(publisher, dict):
            name = publisher.get("name") or publisher.get("title")
            if isinstance(name, str) and name.strip():
                return name.strip()
        series = publication_context.get("series")
        if isinstance(series, dict):
            series_title = series.get("title") or series.get("name")
            if isinstance(series_title, str) and series_title.strip():
                return series_title.strip()
        context_title = publication_context.get("title")
        if isinstance(context_title, str) and context_title.strip():
            return context_title.strip()
        context_type = publication_context.get("type")
        if isinstance(context_type, str) and context_type.strip():
            return context_type.strip()
    return None


def _extract_category(reference: dict[str, Any]) -> str | None:
    if not isinstance(reference, dict):
        return None
    publication_instance = reference.get("publicationInstance")
    if isinstance(publication_instance, dict):
        instance_type = publication_instance.get("type")
        if isinstance(instance_type, str) and instance_type.strip():
            return instance_type.strip()
    publication_context = reference.get("publicationContext")
    if isinstance(publication_context, dict):
        context_type = publication_context.get("type")
        if isinstance(context_type, str) and context_type.strip():
            return context_type.strip()
    return None


def _extract_tags(entity_description: dict[str, Any]) -> list[str]:
    raw_tags = entity_description.get("tags")
    tags: list[str] = []
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            if not isinstance(tag, str):
                continue
            normalized = tag.strip()
            if normalized and normalized.lower() not in {t.lower() for t in tags}:
                tags.append(normalized)
    return tags


def _extract_contributors(entity_description: dict[str, Any]) -> list[NvaContributor]:
    contributors = entity_description.get("contributors")
    if not isinstance(contributors, list):
        return []
    parsed: list[NvaContributor] = []
    for contributor in contributors:
        if isinstance(contributor, dict):
            parsed.append(NvaContributor.from_payload(contributor))
    return parsed


def _extract_language(entity_description: dict[str, Any]) -> str | None:
    language = entity_description.get("language")
    if isinstance(language, str) and language.strip():
        return language.strip()
    return None


def _collect_extra_context(reference: dict[str, Any]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if not isinstance(reference, dict):
        return extra
    publication_instance = reference.get("publicationInstance")
    if isinstance(publication_instance, dict):
        instance_type = publication_instance.get("type")
        if isinstance(instance_type, str) and instance_type.strip():
            extra["publication_instance"] = instance_type.strip()
        pages = publication_instance.get("pages")
        if isinstance(pages, dict):
            page_count = pages.get("pages") or pages.get("pageCount")
            if page_count:
                extra["page_count"] = str(page_count)
    publication_context = reference.get("publicationContext")
    if isinstance(publication_context, dict):
        context_type = publication_context.get("type")
        if isinstance(context_type, str) and context_type.strip():
            extra["publication_context"] = context_type.strip()
    return extra


def normalize_publication(
    *,
    employee_id: str,
    employee_name: str,
    hit_payload: dict[str, Any],
) -> NvaPublicationRecord:
    if not isinstance(hit_payload, dict):
        raise TypeError("hit_payload must be a dict")

    publication_id = _select_text(hit_payload.get("identifier")) or _select_text(
        hit_payload.get("id")
    )
    if not publication_id:
        raise ValueError("Missing publication identifier in hit payload")
    if "/" in publication_id:
        publication_id = publication_id.rstrip("/").split("/")[-1]

    entity_description = hit_payload.get("entityDescription") or {}
    if not isinstance(entity_description, dict):
        entity_description = {}
    reference = entity_description.get("reference") or {}
    if not isinstance(reference, dict):
        reference = {}

    title = _select_text(entity_description.get("mainTitle"))
    abstract = _select_text(entity_description.get("abstract")) or _select_text(
        entity_description.get("alternativeAbstracts")
    )
    venue = _extract_venue(reference)
    category = _extract_category(reference)
    year = _extract_year(entity_description, hit_payload)
    tags = _extract_tags(entity_description)
    contributors = _extract_contributors(entity_description)
    language = _extract_language(entity_description)
    doi = _select_text(reference.get("doi"))
    handle = _select_text(hit_payload.get("handle"))
    source_url = (
        _select_text(hit_payload.get("id"))
        or f"https://api.nva.unit.no/publication/{publication_id}"
    )

    extra_context = _collect_extra_context(reference)

    return NvaPublicationRecord(
        employee_id=employee_id,
        employee_name=employee_name,
        publication_id=publication_id,
        title=title,
        abstract=abstract,
        venue=venue,
        category=category,
        year=year,
        tags=tags,
        contributors=contributors,
        source_url=source_url,
        language=language,
        doi=doi,
        handle=handle,
        extra_context=extra_context,
    )
