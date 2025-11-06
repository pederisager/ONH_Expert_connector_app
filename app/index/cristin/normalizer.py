"""Normalization utilities for Cristin API payloads."""

from __future__ import annotations

from typing import Any, Iterable

from .models import ContributorSummary, ResultRecord

SUMMARY_CANDIDATES = (
    "abstract",
    "description",
    "popular_scientific_summary",
    "academic_summary",
    "popular_scientific_presentation",
    "academic_presentation",
)


def select_localized_text(value: Any, lang: str) -> str | None:
    """Return text for the requested language from Cristin multi-language fields."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        primary = value.get(lang)
        if isinstance(primary, str) and primary.strip():
            return primary.strip()
        for fallback in ("en", "nb", "nn"):
            candidate = value.get(fallback)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        for candidate in value.values():
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    if isinstance(value, list):
        joined = " ".join(filter(None, (select_localized_text(item, lang) for item in value)))
        return joined or None
    return None


def extract_summary(payload: dict[str, Any], lang: str) -> str | None:
    for key in SUMMARY_CANDIDATES:
        if key in payload:
            summary = select_localized_text(payload[key], lang)
            if summary:
                return summary
    return None


def extract_tags(payload: dict[str, Any], lang: str) -> list[str]:
    tags: list[str] = []
    classification = payload.get("classification")
    if isinstance(classification, dict):
        for field_name in ("keywords", "fields_of_science", "disciplines"):
            field_val = classification.get(field_name)
            if isinstance(field_val, list):
                for entry in field_val:
                    if isinstance(entry, dict):
                        tag = select_localized_text(entry.get("name"), lang)
                        if tag:
                            tags.append(tag)
    category = payload.get("category")
    if isinstance(category, dict):
        cat_name = select_localized_text(category.get("name"), lang)
        if cat_name:
            tags.append(cat_name)
    unique = []
    seen = set()
    for tag in tags:
        if tag.lower() not in seen:
            seen.add(tag.lower())
            unique.append(tag)
    return unique


def extract_venue(payload: dict[str, Any], lang: str) -> str | None:
    journal = payload.get("journal")
    if isinstance(journal, dict):
        name = select_localized_text(journal.get("name"), lang)
        if name:
            return name
    channel = payload.get("channel")
    if isinstance(channel, dict):
        title = select_localized_text(channel.get("title"), lang)
        if title:
            return title
    publisher = payload.get("publisher")
    if isinstance(publisher, dict):
        name = select_localized_text(publisher.get("name"), lang)
        if name:
            return name
    part_of = payload.get("part_of")
    if isinstance(part_of, dict):
        title = select_localized_text(part_of.get("title"), lang)
        if title:
            return title
    media_type = payload.get("media_type")
    if isinstance(media_type, dict):
        title = select_localized_text(media_type.get("code_name"), lang)
        if title:
            return title
    degree = payload.get("degree")
    if isinstance(degree, dict):
        title = select_localized_text(degree.get("name"), lang)
        if title:
            return title
    return None


def extract_contributors(payload: dict[str, Any]) -> list[ContributorSummary]:
    contributors_field = payload.get("contributors")
    if isinstance(contributors_field, dict):
        preview = contributors_field.get("preview")
        if isinstance(preview, list):
            return [
                ContributorSummary.from_payload(entry)
                for entry in preview
                if isinstance(entry, dict)
            ]
    return []


def collect_extra_context(payload: dict[str, Any], lang: str) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    original_language = payload.get("original_language")
    if isinstance(original_language, str):
        extra["original_language"] = original_language
    media_type = payload.get("media_type")
    if isinstance(media_type, dict):
        extra["media_type"] = select_localized_text(media_type.get("code_name"), lang) or media_type.get("code")
    pages = payload.get("pages")
    if isinstance(pages, dict):
        page_range = [pages.get("from"), pages.get("to")]
        if any(page_range):
            extra["pages"] = "-".join(filter(None, page_range))
        count = pages.get("count")
        if count:
            extra["page_count"] = count
    project_codes = payload.get("project_codes")
    if isinstance(project_codes, list) and project_codes:
        extra["project_codes"] = project_codes
    funding = payload.get("funding")
    if isinstance(funding, list) and funding:
        extra["funding"] = funding
    return extra


def normalize_result(
    *,
    employee_id: str,
    employee_name: str,
    listing_payload: dict[str, Any] | None,
    detail_payload: dict[str, Any],
    lang: str,
) -> ResultRecord:
    base_payload = detail_payload or listing_payload or {}
    cristin_result_id = base_payload.get("cristin_result_id")
    if cristin_result_id is None:
        raise ValueError("Missing cristin_result_id in payload")

    title = select_localized_text(base_payload.get("title"), lang)
    if title is None and listing_payload:
        title = select_localized_text(listing_payload.get("title"), lang)

    summary = extract_summary(base_payload, lang)
    if summary is None and listing_payload:
        summary = extract_summary(listing_payload, lang)

    venue = extract_venue(base_payload, lang)
    if venue is None and listing_payload:
        venue = extract_venue(listing_payload, lang)

    category = base_payload.get("category") or {}
    category_code = category.get("code") if isinstance(category, dict) else None
    category_name = select_localized_text(category.get("name"), lang) if isinstance(category, dict) else None

    tags = extract_tags(base_payload, lang)

    contributors = extract_contributors(base_payload)

    source_url = base_payload.get("url")
    if not source_url and listing_payload:
        source_url = listing_payload.get("url")

    year_published = base_payload.get("year_published")
    if year_published is None and listing_payload:
        year_published = listing_payload.get("year_published")

    extra_context = collect_extra_context(base_payload, lang)

    return ResultRecord(
        employee_id=employee_id,
        employee_name=employee_name,
        cristin_result_id=str(cristin_result_id),
        title=title,
        summary=summary,
        venue=venue,
        category_code=category_code,
        category_name=category_name,
        year_published=str(year_published) if year_published is not None else None,
        tags=tags,
        contributors=contributors,
        source_url=source_url,
        extra_context=extra_context,
    )
