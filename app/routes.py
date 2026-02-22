"""API routers for the ONH Expert Connector."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import unicodedata
from typing import Any, Literal, Sequence

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .cache_manager import CacheManager
from .config_loader import AppConfig
from .fetch_utils import FetchNotAllowedError, FetchUtils
from .index.models import Chunk
from .language_utils import (
    LanguageContext,
    NoOpTranslator,
    Translator,
    build_language_context,
)
from .match_engine import (
    MatchEngine,
    PageContent,
    StaffDocument,
    StaffProfile,
    extract_themes,
    tokenize,
)
from .nva_lookup import extract_pub_id, preferred_nva_url
from .rag import EmbeddingRetriever, RetrievalQuery

router = APIRouter()
logger = logging.getLogger(__name__)
PAGE_TEXT_CHAR_LIMIT = 8000
CITATION_SNIPPET_LIMIT = 3000
PREVIEW_CHAR_LIMIT = 1200
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
DEFAULT_CITATION_SOURCE_PRIORITY = ["nva", "profile", "staffinfo"]
MATCH_MODE_PUBLICATION = "publication_grounded"
MATCH_MODE_PROFILE = "profile_grounded"
DEFAULT_MATCH_MODE = MATCH_MODE_PUBLICATION
MATCH_MODES = {MATCH_MODE_PUBLICATION, MATCH_MODE_PROFILE}


# --------------------------------------------------------------------------- #
# Pydantic schemas
# --------------------------------------------------------------------------- #


class AnalyzeTopicResponse(BaseModel):
    themes: list[str]
    normalized_preview: str = Field(alias="normalizedPreview")


class MatchRequest(BaseModel):
    themes: list[str]
    department: str | None = None
    mode: Literal["publication_grounded", "profile_grounded"] = DEFAULT_MATCH_MODE


class Citation(BaseModel):
    id: str
    title: str
    url: str
    snippet: str


class MatchItem(BaseModel):
    id: str
    name: str
    department: str
    profile_url: str
    score: float
    why: str
    why_by_lang: dict[str, str] | None = Field(default=None, alias="whyByLang")
    citations: list[Citation]
    score_breakdown: dict[str, float] = Field(alias="scoreBreakdown")
    keywords: list[str] = Field(default_factory=list)


@router.head("/queue", include_in_schema=False)
async def queue_probe_head() -> Response:
    """Silence health-check HEAD probes from local reverse proxies."""

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/queue", include_in_schema=False)
async def queue_probe_get() -> dict[str, str]:
    return {"status": "idle"}


class MatchResponse(BaseModel):
    results: list[MatchItem]
    total: int


class ConfigResponse(BaseModel):
    departments: list[str]
    ui: dict[str, Any]
    security: dict[str, Any]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _validate_text_input(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mangler tematisk innhold. Skriv tekst for tema.",
        )
    return normalized


def _build_normalized_preview(text: str, limit: int = PREVIEW_CHAR_LIMIT) -> str:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    preview_parts: list[str] = []
    total = 0
    truncated = False

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        projected = total + len(sentence) + (1 if preview_parts else 0)
        if preview_parts and projected > limit:
            truncated = True
            break
        preview_parts.append(sentence)
        total = projected
        if total >= limit:
            truncated = total > limit
            break

    if not preview_parts:
        preview = normalized[:limit]
        truncated = len(normalized) > len(preview)
    else:
        preview = " ".join(preview_parts)

    if len(preview) > limit:
        truncated = True
        preview = preview[:limit]
        last_space = preview.rfind(" ")
        if last_space > int(limit * 0.6):
            preview = preview[:last_space]

    preview = preview.strip()
    if not preview:
        return ""

    if truncated or len(preview) < len(normalized):
        return preview.rstrip(",; ") + "..."
    return preview


def _resolve_user_language(request: Request, app_config: AppConfig) -> str:
    header_lang = request.headers.get("x-ui-language") or request.headers.get(
        "x-language"
    )
    query_lang = request.query_params.get("lang")
    return (header_lang or query_lang or app_config.ui.language or "no").lower()


def _extract_citation_snippet(
    text: str, themes: Sequence[str], limit: int = CITATION_SNIPPET_LIMIT
) -> str:
    """Pick the most relevant portion of a chunk for downstream LLM use."""

    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return ""
    if len(normalized) <= limit:
        return normalized

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    theme_tokens = {
        token for theme in themes for token in tokenize(theme) if len(token) > 2
    }

    best_snippet = normalized[:limit].rstrip()
    best_score = -1 if theme_tokens else 0

    if sentences:
        for start_idx in range(len(sentences)):
            window: list[str] = []
            total_len = 0
            total_score = 0

            for sentence in sentences[start_idx:]:
                projected = total_len + len(sentence) + (1 if window else 0)
                if window and projected > limit:
                    break
                window.append(sentence)
                total_len = projected

                if theme_tokens:
                    lowered = sentence.lower()
                    total_score += sum(lowered.count(token) for token in theme_tokens)

            if not window:
                continue

            if total_score > best_score or (
                total_score == best_score and total_len > len(best_snippet)
            ):
                best_score = total_score
                best_snippet = " ".join(window)

    if len(best_snippet) > limit:
        best_snippet = best_snippet[:limit]
        last_space = best_snippet.rfind(" ")
        if last_space > int(limit * 0.6):
            best_snippet = best_snippet[:last_space]

    return best_snippet.strip()


def _staff_doc_cache_key(profile: StaffProfile, max_pages: int) -> str:
    digest = hashlib.sha1(
        f"{profile.profile_url}|{max_pages}".encode("utf-8")
    ).hexdigest()
    return f"staffdoc::{digest}"


def _remember_staff_document(
    *,
    cache_key: str,
    sources: Sequence[str],
    pages: Sequence[PageContent],
    memory_cache: dict[str, dict[str, object]] | None,
) -> None:
    if memory_cache is None or not pages:
        return
    memory_cache[cache_key] = {
        "sources": list(sources),
        "pages": tuple(pages),
    }


def _load_cached_staff_document(
    *,
    profile: StaffProfile,
    cache_manager: CacheManager,
    max_pages: int,
    memory_cache: dict[str, dict[str, object]] | None,
) -> StaffDocument | None:
    cache_key = _staff_doc_cache_key(profile, max_pages)
    expected_sources = profile.sources[:max_pages]

    if memory_cache is not None:
        entry = memory_cache.get(cache_key)
        if entry and entry.get("sources") == expected_sources:
            pages = list(entry.get("pages") or [])
            if pages:
                return StaffDocument(profile=profile, pages=pages)

    cached = cache_manager.get(cache_key)
    if not cached:
        return None
    if cached.get("sources") != expected_sources:
        return None
    pages_payload = cached.get("pages") or []
    if not pages_payload:
        return None
    try:
        pages = [PageContent(**page) for page in pages_payload if page.get("text")]
    except (TypeError, ValueError):
        return None
    if not pages:
        return None
    _remember_staff_document(
        cache_key=cache_key,
        sources=expected_sources,
        pages=pages,
        memory_cache=memory_cache,
    )
    return StaffDocument(profile=profile, pages=pages)


def _store_staff_document(
    *,
    profile: StaffProfile,
    cache_manager: CacheManager,
    max_pages: int,
    pages: Sequence[PageContent],
    memory_cache: dict[str, dict[str, object]] | None,
) -> None:
    if not pages:
        return
    sources = profile.sources[:max_pages]
    payload = {
        "sources": sources,
        "pages": [
            {
                "url": page.url,
                "title": page.title,
                "text": page.text,
            }
            for page in pages
        ],
    }
    cache_key = _staff_doc_cache_key(profile, max_pages)
    cache_manager.set(cache_key, payload)
    _remember_staff_document(
        cache_key=cache_key, sources=sources, pages=pages, memory_cache=memory_cache
    )


async def _fetch_staff_documents(
    *,
    staff_profiles: Sequence[StaffProfile],
    fetch_utils: FetchUtils,
    cache_manager: CacheManager,
    max_pages: int,
    memory_cache: dict[str, dict[str, object]] | None = None,
) -> list[StaffDocument]:
    doc_map: dict[str, StaffDocument] = {}
    pending_profiles: list[StaffProfile] = []

    for profile in staff_profiles:
        cached = _load_cached_staff_document(
            profile=profile,
            cache_manager=cache_manager,
            max_pages=max_pages,
            memory_cache=memory_cache,
        )
        if cached is not None:
            doc_map[profile.profile_url] = cached
        else:
            pending_profiles.append(profile)

    if pending_profiles:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            tasks = [
                _fetch_single_staff(
                    profile,
                    client,
                    fetch_utils,
                    cache_manager,
                    max_pages,
                    memory_cache,
                )
                for profile in pending_profiles
            ]
            fetched = await asyncio.gather(*tasks)
        for profile, document in zip(pending_profiles, fetched, strict=False):
            if document.pages:
                doc_map[profile.profile_url] = document

    ordered: list[StaffDocument] = []
    for profile in staff_profiles:
        document = doc_map.get(profile.profile_url)
        if document is not None:
            ordered.append(document)
    return ordered


async def warm_staff_document_cache(state) -> None:
    try:
        staff_profiles: list[StaffProfile] = state.staff_profiles
        fetch_utils: FetchUtils = state.fetch_utils
        cache_manager: CacheManager = state.cache_manager
        max_pages: int = state.app_config.fetch.max_pages_per_staff
        memory_cache = getattr(state, "staff_document_cache", None)
    except AttributeError:
        return

    if not staff_profiles:
        return

    await _fetch_staff_documents(
        staff_profiles=staff_profiles,
        fetch_utils=fetch_utils,
        cache_manager=cache_manager,
        max_pages=max_pages,
        memory_cache=memory_cache,
    )


async def _fetch_single_staff(
    profile: StaffProfile,
    client: httpx.AsyncClient,
    fetch_utils: FetchUtils,
    cache_manager: CacheManager,
    max_pages: int,
    memory_cache: dict[str, dict[str, object]] | None,
) -> StaffDocument:
    def _as_plain_text(value: Any) -> str:
        if value is None:
            return ""
        if value.__class__ is str:
            return value
        return str(value)

    pages: list[PageContent] = []
    for url in profile.sources[:max_pages]:
        cache_key = f"fetch::{url}"
        cached = cache_manager.get(cache_key)
        if cached:
            pages.append(PageContent(**cached))
            continue
        try:
            page_data = await fetch_utils.fetch_page(client, url)
        except FetchNotAllowedError:
            continue
        except httpx.HTTPError:
            continue
        cleaned = {
            "url": _as_plain_text(page_data.get("url", url)),
            "title": _as_plain_text(page_data.get("title")),
            "text": _as_plain_text(page_data.get("text"))[:PAGE_TEXT_CHAR_LIMIT],
        }
        cache_manager.set(cache_key, cleaned)
        pages.append(PageContent(**cleaned))
    document = StaffDocument(profile=profile, pages=pages)
    _store_staff_document(
        profile=profile,
        cache_manager=cache_manager,
        max_pages=max_pages,
        pages=pages,
        memory_cache=memory_cache,
    )
    return document


def _hash_id(name: str, profile_url: str) -> str:
    digest = hashlib.sha1(f"{name}:{profile_url}".encode("utf-8")).hexdigest()
    return digest[:12]


def _lookup_precomputed_summary(
    *,
    precomputed: dict[str, str],
    profile_url: str,
    name: str,
) -> str:
    summary = precomputed.get(profile_url) or precomputed.get(name)
    return (summary or "").strip()


def _normalize_summary_lang(user_lang: str) -> str:
    normalized = (user_lang or "no").lower()
    return "en" if normalized.startswith("en") else "no"


def _build_precomputed_summaries_by_lang(
    *,
    precomputed_by_lang: dict[str, dict[str, str]],
    profile_url: str,
    name: str,
) -> dict[str, str] | None:
    """Return available precomputed summaries keyed by language, or None if empty."""
    summary_no = _lookup_precomputed_summary(
        precomputed=precomputed_by_lang.get("no", {}),
        profile_url=profile_url,
        name=name,
    )
    summary_en = _lookup_precomputed_summary(
        precomputed=precomputed_by_lang.get("en", {}),
        profile_url=profile_url,
        name=name,
    )
    summaries = {
        key: value
        for key, value in {"no": summary_no, "en": summary_en}.items()
        if value
    }
    return summaries or None


def _lookup_precomputed_summary_by_lang(
    *,
    precomputed_by_lang: dict[str, dict[str, str]],
    user_lang: str,
    profile_url: str,
    name: str,
    default_lang: str = "no",
) -> str:
    """Pick summary in `user_lang`, falling back to `default_lang` then empty."""
    lang = _normalize_summary_lang(user_lang)
    summary = _lookup_precomputed_summary(
        precomputed=precomputed_by_lang.get(lang, {}),
        profile_url=profile_url,
        name=name,
    )
    if summary:
        return summary
    if default_lang and default_lang != lang:
        return _lookup_precomputed_summary(
            precomputed=precomputed_by_lang.get(default_lang, {}),
            profile_url=profile_url,
            name=name,
        )
    return ""


def _choose_why_text(
    *,
    precomputed: str,
    fallback_explanation: str | None,
    staff_name: str,
    themes: Sequence[str],
    language_ctx: LanguageContext,
) -> str:
    """Return the staff card summary (why) in the requested language without runtime translation."""
    if precomputed:
        return precomputed
    if language_ctx.user_lang.startswith("en"):
        return _localize_explanation(
            text=None,
            name=staff_name,
            themes=themes,
            language_ctx=language_ctx,
        )
    return _localize_explanation(
        text=fallback_explanation,
        name=staff_name,
        themes=themes,
        language_ctx=language_ctx,
    )


def _rag_query_from_themes(
    themes: Sequence[str],
    synonym_expansion_map: dict[str, Sequence[str]] | None = None,
) -> str:
    expanded_terms = _expand_themes_for_query(
        themes,
        synonym_expansion_map=synonym_expansion_map,
    )
    return " ".join(expanded_terms).strip()


def _match_via_retriever(
    *,
    retriever: EmbeddingRetriever,
    payload: MatchRequest,
    match_mode: str,
    max_candidates: int,
    translator: Translator,
    language_ctx: LanguageContext,
    query_text: str,
    citation_source_priority: Sequence[str],
    min_query_overlap_per_citation: int,
    scoring_weights: dict[str, float],
    mode_scoring_profiles: dict[str, dict[str, Any]],
    exact_keyword_promotion_config: dict[str, Any],
    overexposure_penalty_config: dict[str, Any],
    profile_staffinfo_min_query_overlap_per_citation: int | None = None,
    category_intent_penalty_config: dict[str, Any] | None = None,
    concept_keyword_map: dict[str, Sequence[str]] | None = None,
    synonym_expansion_map: dict[str, Sequence[str]] | None = None,
) -> list[MatchItem]:
    if not query_text:
        return []
    prepared_query = query_text
    if language_ctx.translate_for_embedding:
        prepared_query = translator.translate(
            query_text,
            source_lang=language_ctx.query_lang,
            target_lang=language_ctx.embed_lang,
        )
    try:
        results = retriever.retrieve(
            RetrievalQuery(
                text=prepared_query,
                department=payload.department,
                top_k=max_candidates,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Retriever failed for '%s': %s", query_text, exc)
        return []

    items: list[MatchItem] = []
    for result in results:
        item = _retrieval_result_to_match_item(
            result,
            payload.themes,
            match_mode=match_mode,
            language_ctx=language_ctx,
            citation_source_priority=citation_source_priority,
            min_query_overlap_per_citation=min_query_overlap_per_citation,
            profile_staffinfo_min_query_overlap_per_citation=(
                profile_staffinfo_min_query_overlap_per_citation
            ),
            scoring_weights=scoring_weights,
            mode_scoring_profiles=mode_scoring_profiles,
            exact_keyword_promotion_config=exact_keyword_promotion_config,
            overexposure_penalty_config=overexposure_penalty_config,
            concept_keyword_map=concept_keyword_map,
            synonym_expansion_map=synonym_expansion_map,
            category_intent_penalty_config=category_intent_penalty_config,
        )
        if item:
            items.append(item)
    items.sort(
        key=lambda item: (
            float(item.score_breakdown.get("exact_keyword_match", 0.0)),
            item.score,
        ),
        reverse=True,
    )
    return items


def _retrieval_result_to_match_item(
    result,
    themes: Sequence[str],
    *,
    match_mode: str,
    language_ctx: LanguageContext,
    citation_source_priority: Sequence[str],
    min_query_overlap_per_citation: int,
    scoring_weights: dict[str, float],
    mode_scoring_profiles: dict[str, dict[str, Any]],
    profile_staffinfo_min_query_overlap_per_citation: int | None = None,
    exact_keyword_promotion_config: dict[str, Any] | None = None,
    overexposure_penalty_config: dict[str, Any] | None = None,
    category_intent_penalty_config: dict[str, Any] | None = None,
    concept_keyword_map: dict[str, Sequence[str]] | None = None,
    synonym_expansion_map: dict[str, Sequence[str]] | None = None,
) -> MatchItem | None:
    if not result.chunks:
        return None

    primary_chunk = result.chunks[0]
    department = str(
        primary_chunk.metadata.get("department")
        or result.metadata.get("department")
        or ""
    )
    profile_url = str(
        primary_chunk.metadata.get("profile_url") or primary_chunk.source_url or ""
    )
    display_name = result.staff_name or str(
        primary_chunk.metadata.get("name") or result.staff_slug
    )
    citations = _chunks_to_citations(
        result.chunks,
        themes,
        source_priority=citation_source_priority,
        min_query_overlap=min_query_overlap_per_citation,
        profile_staffinfo_min_query_overlap=(
            profile_staffinfo_min_query_overlap_per_citation
        ),
        match_mode=match_mode,
        concept_keyword_map=concept_keyword_map,
        synonym_expansion_map=synonym_expansion_map,
    )
    keywords = _collect_chunk_keywords(result.chunks)
    scoring_chunks = _chunks_for_scoring(result.chunks)
    scoring_keywords = _collect_chunk_keywords(scoring_chunks)
    semantic_score = max(
        0.0,
        min(1.0, float(result.metadata.get("semantic_score", result.score))),
    )
    lexical_retrieval_score = max(
        0.0,
        min(1.0, float(result.metadata.get("lexical_score", 0.0))),
    )
    hybrid_retrieval_score = max(
        0.0,
        min(1.0, float(result.metadata.get("hybrid_score", result.score))),
    )
    keyword_score = _keyword_overlap_score(
        scoring_chunks,
        themes,
        concept_keyword_map=concept_keyword_map,
        synonym_expansion_map=synonym_expansion_map,
    )
    tag_score = _tag_overlap_score(
        scoring_keywords,
        themes,
        concept_keyword_map=concept_keyword_map,
        synonym_expansion_map=synonym_expansion_map,
    )
    method_score = _method_overlap_score(scoring_keywords, themes)
    expanded_theme_terms = _expanded_theme_tokens(
        themes,
        concept_keyword_map=concept_keyword_map,
        synonym_expansion_map=synonym_expansion_map,
    )
    base_theme_terms = {
        token.lower()
        for theme in themes
        for token in tokenize(theme)
        if len(token.strip()) > 1
    }
    expansion_terms_count = max(0, len(expanded_theme_terms) - len(base_theme_terms))
    exact_keyword_match, exact_keyword_match_count = _exact_keyword_match_features(
        keywords=scoring_keywords,
        themes=themes,
        exact_keyword_promotion_config=exact_keyword_promotion_config,
    )
    mode_bonus = _mode_score_bonus(
        chunks=result.chunks,
        citations=citations,
        themes=themes,
        match_mode=match_mode,
        mode_scoring_profiles=mode_scoring_profiles,
        concept_keyword_map=concept_keyword_map,
        synonym_expansion_map=synonym_expansion_map,
    )
    overexposure_penalty = _overexposure_score_penalty(
        chunks=result.chunks,
        citations=citations,
        themes=themes,
        match_mode=match_mode,
        keyword_score=keyword_score,
        tag_score=tag_score,
        method_score=method_score,
        overexposure_penalty_config=overexposure_penalty_config or {},
        concept_keyword_map=concept_keyword_map,
        synonym_expansion_map=synonym_expansion_map,
    )
    category_intent_penalty = _category_intent_score_penalty(
        chunks=result.chunks,
        department=department,
        themes=themes,
        concept_keyword_map=concept_keyword_map,
        synonym_expansion_map=synonym_expansion_map,
        category_intent_penalty_config=category_intent_penalty_config,
    )
    semantic_weight = max(0.0, float(scoring_weights.get("semantic", 1.0)))
    keyword_weight = max(0.0, float(scoring_weights.get("keywords", 0.1)))
    tag_weight = max(0.0, float(scoring_weights.get("tags", 0.15)))
    method_weight = max(0.0, float(scoring_weights.get("methods", 0.15)))
    exact_keyword_bonus = _exact_keyword_bonus(
        exact_keyword_match=exact_keyword_match,
        exact_keyword_promotion_config=exact_keyword_promotion_config,
    )
    raw_score = min(
        1.0,
        semantic_weight * semantic_score
        + keyword_weight * keyword_score
        + tag_weight * tag_score
        + method_weight * method_score
        + mode_bonus
        + exact_keyword_bonus,
    )
    adjusted_score = max(0.0, raw_score - overexposure_penalty - category_intent_penalty)

    if language_ctx.user_lang.startswith("en"):
        why_default = f"{display_name} matches {', '.join(themes) or 'the topic'} based on semantic similarity."
        if keywords:
            why_default += f" Keywords: {', '.join(keywords[:4])}."
    else:
        why_default = f"{display_name} matcher {', '.join(themes) or 'temaet'} basert på semantisk treff."
        if keywords:
            why_default += f" Nøkkelord: {', '.join(keywords[:4])}."
    return MatchItem(
        id=_hash_id(display_name, profile_url or result.staff_slug),
        name=display_name,
        department=department,
        profile_url=profile_url or primary_chunk.source_url or "",
        score=round(adjusted_score, 4),
        why=why_default,
        citations=citations,
        scoreBreakdown={
            "semantic": round(semantic_score, 4),
            "lexical": round(lexical_retrieval_score, 4),
            "retrieval": round(hybrid_retrieval_score, 4),
            "keywords": round(keyword_score, 4),
            "tags": round(tag_score, 4),
            "methods": round(method_score, 4),
            "exact_keyword_match": float(exact_keyword_match),
            "exact_keyword_match_count": float(exact_keyword_match_count),
            "expanded_query_terms_count": float(expansion_terms_count),
            "exact_keyword_bonus": round(exact_keyword_bonus, 4),
            "mode_bonus": round(mode_bonus, 4),
            "overexposure_penalty": round(overexposure_penalty, 4),
            "category_intent_penalty": round(category_intent_penalty, 4),
        },
        keywords=keywords,
    )


def _nva_registration_url(pub_id: str) -> str:
    return f"https://nva.sikt.no/registration/{pub_id}"


def _resolve_citation_url(chunk: Chunk) -> str:
    metadata = chunk.metadata or {}
    # 1) Prefer original/DOI URLs when present.
    doi = str(metadata.get("doi") or "").strip()
    if doi:
        if doi.lower().startswith("http"):
            return doi
        return f"https://doi.org/{doi}"

    source_url = str(
        chunk.source_url
        or metadata.get("source_url")
        or metadata.get("profile_url")
        or ""
    ).strip()
    if source_url and "doi.org" in source_url.lower():
        return source_url

    publication_id = (
        metadata.get("nva_publication_id")
        or extract_pub_id(source_url)
        or metadata.get("id")
        or ""
    )
    publication_id = str(publication_id).strip()
    if publication_id:
        preferred = preferred_nva_url(publication_id, source_url)
        if preferred and "doi.org" in preferred.lower():
            return preferred
        return _nva_registration_url(publication_id)
    return source_url


def _source_kind_for_chunk(chunk: Chunk) -> str:
    metadata = chunk.metadata or {}
    source_kind = str(metadata.get("source_kind") or "").strip().lower()
    if source_kind:
        return source_kind
    url = (chunk.source_url or "").lower()
    if url.startswith("staffinfo://"):
        return "staffinfo"
    if "oslonyehoyskole.no" in url:
        return "profile"
    return "nva"


def _priority_by_source_kind(source_priority: Sequence[str]) -> dict[str, int]:
    ordering = [item.strip().lower() for item in source_priority if item.strip()]
    if not ordering:
        ordering = list(DEFAULT_CITATION_SOURCE_PRIORITY)
    return {source_kind: idx for idx, source_kind in enumerate(ordering)}


def _normalize_expansion_map(
    expansion_map: dict[str, Sequence[str]] | None,
) -> dict[str, set[str]]:
    normalized: dict[str, set[str]] = {}
    if not expansion_map:
        return normalized
    for key, values in expansion_map.items():
        key_tokens = [token.lower() for token in tokenize(str(key)) if len(token) > 1]
        if not key_tokens:
            continue
        expansions = {token.lower() for token in key_tokens}
        for value in values or []:
            expansions.update(
                token.lower() for token in tokenize(str(value)) if len(token) > 1
            )
        for token in key_tokens:
            normalized.setdefault(token, set()).update(expansions)
    return normalized


def _expanded_theme_tokens(
    themes: Sequence[str],
    concept_keyword_map: dict[str, Sequence[str]] | None = None,
    synonym_expansion_map: dict[str, Sequence[str]] | None = None,
) -> set[str]:
    base_tokens = {
        token.lower()
        for theme in themes
        for token in tokenize(theme)
        if len(token.strip()) > 1
    }
    if not base_tokens:
        return set()
    expanded = set(base_tokens)
    for mapping in (
        _normalize_expansion_map(concept_keyword_map),
        _normalize_expansion_map(synonym_expansion_map),
    ):
        for token in list(expanded):
            expanded.update(mapping.get(token, set()))
    return expanded


def _expand_themes_for_query(
    themes: Sequence[str],
    synonym_expansion_map: dict[str, Sequence[str]] | None = None,
) -> list[str]:
    expanded_tokens = _expanded_theme_tokens(
        themes,
        synonym_expansion_map=synonym_expansion_map,
    )
    theme_tokens = [
        token.lower() for theme in themes for token in tokenize(theme) if len(token) > 1
    ]
    ordered: list[str] = []
    seen: set[str] = set()
    for token in theme_tokens + sorted(expanded_tokens):
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _normalize_exact_keyword_value(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    folded = normalized.casefold()
    compact = "".join(
        char if (char.isalnum() or char.isspace()) else " "
        for char in folded
    )
    return " ".join(compact.split())


def _normalize_exact_keyword_promotion_config(
    exact_keyword_promotion_config: dict[str, Any] | None,
) -> dict[str, float | bool | int]:
    if exact_keyword_promotion_config is None:
        return {
            "enabled": False,
            "keyword_bonus": 0.0,
            "min_token_length": 2,
        }
    config = exact_keyword_promotion_config
    return {
        "enabled": bool(config.get("enabled", True)),
        "keyword_bonus": max(0.0, float(config.get("keyword_bonus", 0.35))),
        "min_token_length": max(1, int(config.get("min_token_length", 2))),
    }


def _exact_keyword_term_set(values: Sequence[str], *, min_token_length: int) -> set[str]:
    terms: set[str] = set()
    for value in values:
        normalized = _normalize_exact_keyword_value(value)
        if not normalized:
            continue
        if len(normalized.replace(" ", "")) >= min_token_length:
            terms.add(normalized)
        for token in normalized.split():
            if len(token) >= min_token_length:
                terms.add(token)
    return terms


def _exact_keyword_match_features(
    *,
    keywords: Sequence[str],
    themes: Sequence[str],
    exact_keyword_promotion_config: dict[str, Any] | None,
) -> tuple[bool, int]:
    config = _normalize_exact_keyword_promotion_config(exact_keyword_promotion_config)
    if not bool(config.get("enabled")):
        return False, 0

    min_token_length = int(config.get("min_token_length", 2))
    keyword_terms = _exact_keyword_term_set(keywords, min_token_length=min_token_length)
    theme_terms = _exact_keyword_term_set(themes, min_token_length=min_token_length)
    if not keyword_terms or not theme_terms:
        return False, 0

    overlap_count = len(keyword_terms & theme_terms)
    return overlap_count > 0, overlap_count


def _exact_keyword_bonus(
    *,
    exact_keyword_match: bool,
    exact_keyword_promotion_config: dict[str, Any] | None,
) -> float:
    config = _normalize_exact_keyword_promotion_config(exact_keyword_promotion_config)
    if not bool(config.get("enabled")) or not exact_keyword_match:
        return 0.0
    return float(config.get("keyword_bonus", 0.35))


def _chunk_metadata_tags(chunk: Chunk) -> list[str]:
    metadata = chunk.metadata or {}
    raw_tags = metadata.get("tags")
    if not isinstance(raw_tags, list):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for tag in raw_tags:
        if not isinstance(tag, str):
            continue
        normalized = tag.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        tags.append(normalized)
    return tags


def _query_overlap_count(snippet: str, theme_tokens: set[str]) -> int:
    if not snippet or not theme_tokens:
        return 0
    snippet_tokens = {token.lower() for token in tokenize(snippet) if len(token) > 2}
    return len(theme_tokens & snippet_tokens)


def _augment_citation_snippet_for_publication(
    *,
    chunk: Chunk,
    snippet: str,
    theme_tokens: set[str],
    limit: int = CITATION_SNIPPET_LIMIT,
) -> str:
    tags = _chunk_metadata_tags(chunk)
    if not tags:
        return snippet

    ranked_tags = sorted(
        tags,
        key=lambda tag: (-_query_overlap_count(tag, theme_tokens), len(tag)),
    )
    matched_tags = [
        tag for tag in ranked_tags if _query_overlap_count(tag, theme_tokens) > 0
    ]
    selected_tags = (matched_tags or ranked_tags)[:6]
    if not selected_tags:
        return snippet

    addition = f"Nokkelord: {', '.join(selected_tags)}"
    if addition.lower() in snippet.lower():
        return snippet

    combined = f"{snippet} {addition}".strip()
    if len(combined) <= limit:
        return combined

    truncated = combined[:limit]
    last_space = truncated.rfind(" ")
    if last_space > int(limit * 0.6):
        truncated = truncated[:last_space]
    return truncated.strip()


def _chunks_for_scoring(chunks: Sequence[Chunk]) -> list[Chunk]:
    prioritized = [
        chunk
        for chunk in chunks
        if _source_kind_for_chunk(chunk) in {"nva", "profile"}
    ]
    return prioritized or list(chunks)


def _chunks_to_citations(
    chunks: Sequence[Chunk],
    themes: Sequence[str],
    *,
    source_priority: Sequence[str] | None = None,
    min_query_overlap: int = 1,
    profile_staffinfo_min_query_overlap: int | None = None,
    match_mode: str = DEFAULT_MATCH_MODE,
    concept_keyword_map: dict[str, Sequence[str]] | None = None,
    synonym_expansion_map: dict[str, Sequence[str]] | None = None,
) -> list[Citation]:
    priority = _priority_by_source_kind(
        source_priority or DEFAULT_CITATION_SOURCE_PRIORITY
    )
    sorted_chunks = sorted(
        chunks,
        key=lambda chunk: (
            priority.get(_source_kind_for_chunk(chunk), len(priority)),
            chunk.order,
        ),
    )

    min_overlap = max(0, int(min_query_overlap))
    profile_staffinfo_min_overlap = max(
        min_overlap,
        int(
            profile_staffinfo_min_query_overlap
            if profile_staffinfo_min_query_overlap is not None
            else min_overlap
        ),
    )
    theme_tokens = _expanded_theme_tokens(
        themes,
        concept_keyword_map=concept_keyword_map,
        synonym_expansion_map=synonym_expansion_map,
    )

    candidates: list[tuple[Chunk, str, int, str]] = []
    for chunk in sorted_chunks:
        snippet = _extract_citation_snippet(chunk.text, themes)
        if not snippet:
            continue
        source_kind = _source_kind_for_chunk(chunk)
        required_overlap = (
            profile_staffinfo_min_overlap
            if source_kind in {"profile", "staffinfo"}
            else min_overlap
        )
        overlap = _query_overlap_count(snippet, theme_tokens)
        if (
            match_mode == MATCH_MODE_PUBLICATION
            and source_kind == "nva"
            and overlap < required_overlap
        ):
            snippet = _augment_citation_snippet_for_publication(
                chunk=chunk,
                snippet=snippet,
                theme_tokens=theme_tokens,
            )
            overlap = _query_overlap_count(snippet, theme_tokens)
        if overlap < required_overlap:
            continue
        candidates.append((chunk, snippet, overlap, source_kind))

    selected: list[tuple[Chunk, str, int, str]]
    nva_candidates = [item for item in candidates if item[3] == "nva"]
    if nva_candidates:
        selected = nva_candidates
    elif candidates:
        selected = candidates
    else:
        fallback: list[tuple[Chunk, str, int, str]] = []
        for chunk in sorted_chunks:
            snippet = _extract_citation_snippet(chunk.text, themes)
            if not snippet:
                continue
            source_kind = _source_kind_for_chunk(chunk)
            fallback.append(
                (chunk, snippet, _query_overlap_count(snippet, theme_tokens), source_kind)
            )
        fallback.sort(
            key=lambda item: (
                -item[2],
                priority.get(item[3], len(priority)),
                item[0].order,
            )
        )
        selected = fallback[:3]

    citations: list[Citation] = []
    for idx, (chunk, snippet, _, _) in enumerate(selected, start=1):
        title = str(
            chunk.metadata.get("source_title") or chunk.metadata.get("name") or "Kilde"
        )
        url = _resolve_citation_url(chunk)
        citations.append(
            Citation(
                id=f"[{idx}]",
                title=title or "Kilde",
                url=url,
                snippet=snippet,
            )
        )
    return citations


def _citation_source_kind(citation: Citation) -> str:
    url = (citation.url or "").lower()
    if "doi.org" in url or "nva.sikt.no/registration/" in url:
        return "nva"
    if url.startswith("staffinfo://"):
        return "staffinfo"
    return "profile"


def _chunk_source_coverage(chunks: Sequence[Chunk]) -> dict[str, float]:
    if not chunks:
        return {}
    counts: dict[str, int] = {}
    for chunk in chunks:
        source_kind = _source_kind_for_chunk(chunk)
        counts[source_kind] = counts.get(source_kind, 0) + 1
    total = float(len(chunks))
    return {source_kind: count / total for source_kind, count in counts.items()}


def _citation_overlap_score(
    citations: Sequence[Citation],
    themes: Sequence[str],
    *,
    concept_keyword_map: dict[str, Sequence[str]] | None = None,
    synonym_expansion_map: dict[str, Sequence[str]] | None = None,
) -> float:
    theme_tokens = _expanded_theme_tokens(
        themes,
        concept_keyword_map=concept_keyword_map,
        synonym_expansion_map=synonym_expansion_map,
    )
    if not theme_tokens or not citations:
        return 0.0
    best_overlap = 0.0
    denominator = float(len(theme_tokens))
    for citation in citations:
        overlap = _query_overlap_count(citation.snippet, theme_tokens)
        best_overlap = max(best_overlap, overlap / max(1.0, denominator))
    return min(1.0, best_overlap)


def _normalize_mode_scoring_profile(
    *,
    match_mode: str,
    mode_scoring_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    normalized_mode = (
        match_mode if match_mode in MATCH_MODES else DEFAULT_MATCH_MODE
    )
    profile = mode_scoring_profiles.get(normalized_mode) or {}
    source_kind_boosts_raw = profile.get("source_kind_boosts") or {}
    source_kind_boosts = {}
    if isinstance(source_kind_boosts_raw, dict):
        source_kind_boosts = {
            str(source_kind).strip().lower(): max(0.0, float(boost))
            for source_kind, boost in source_kind_boosts_raw.items()
            if str(source_kind).strip()
        }
    return {
        "source_kind_boosts": source_kind_boosts,
        "citation_overlap_weight": max(
            0.0,
            float(profile.get("citation_overlap_weight", 0.0)),
        ),
        "nva_citation_bonus": max(
            0.0,
            float(profile.get("nva_citation_bonus", 0.0)),
        ),
    }


def _mode_score_bonus(
    *,
    chunks: Sequence[Chunk],
    citations: Sequence[Citation],
    themes: Sequence[str],
    match_mode: str,
    mode_scoring_profiles: dict[str, dict[str, Any]],
    concept_keyword_map: dict[str, Sequence[str]] | None = None,
    synonym_expansion_map: dict[str, Sequence[str]] | None = None,
) -> float:
    profile = _normalize_mode_scoring_profile(
        match_mode=match_mode,
        mode_scoring_profiles=mode_scoring_profiles,
    )
    source_coverage = _chunk_source_coverage(chunks)
    source_bonus = sum(
        float(profile["source_kind_boosts"].get(source_kind, 0.0)) * coverage
        for source_kind, coverage in source_coverage.items()
    )
    citation_overlap_bonus = float(profile["citation_overlap_weight"]) * _citation_overlap_score(
        citations,
        themes,
        concept_keyword_map=concept_keyword_map,
        synonym_expansion_map=synonym_expansion_map,
    )
    has_nva_citation = any(_citation_source_kind(citation) == "nva" for citation in citations)
    nva_bonus = float(profile["nva_citation_bonus"]) if has_nva_citation else 0.0
    return source_bonus + citation_overlap_bonus + nva_bonus


def _normalize_category_intent_penalty_config(
    category_intent_penalty_config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = category_intent_penalty_config or {}
    return {
        "enabled": bool(config.get("enabled", True)),
        "base_penalty": max(0.0, float(config.get("base_penalty", 0.08))),
        "max_penalty": max(0.0, float(config.get("max_penalty", 0.16))),
        "evidence_overlap_threshold": min(
            1.0,
            max(0.0, float(config.get("evidence_overlap_threshold", 0.2))),
        ),
        "intent_signals": config.get("intent_signals") or {},
        "intent_department_map": config.get("intent_department_map") or {},
    }


def _category_intent_score_penalty(
    *,
    chunks: Sequence[Chunk],
    department: str,
    themes: Sequence[str],
    concept_keyword_map: dict[str, Sequence[str]] | None,
    synonym_expansion_map: dict[str, Sequence[str]] | None,
    category_intent_penalty_config: dict[str, Any] | None,
) -> float:
    profile = _normalize_category_intent_penalty_config(category_intent_penalty_config)
    if not bool(profile.get("enabled")):
        return 0.0

    theme_tokens = _expanded_theme_tokens(
        themes,
        concept_keyword_map=concept_keyword_map,
        synonym_expansion_map=synonym_expansion_map,
    )
    if not theme_tokens:
        return 0.0

    intent_signals_raw = profile.get("intent_signals") or {}
    intent_departments_raw = profile.get("intent_department_map") or {}
    if not isinstance(intent_signals_raw, dict):
        return 0.0

    department_norm = _normalize_exact_keyword_value(department)
    chunk_tokens: set[str] = set()
    for chunk in chunks:
        chunk_tokens.update(token.lower() for token in tokenize(chunk.text) if len(token) > 1)
        for tag in _chunk_metadata_tags(chunk):
            chunk_tokens.update(token.lower() for token in tokenize(tag) if len(token) > 1)

    total_penalty = 0.0
    for intent, signals in intent_signals_raw.items():
        signal_tokens = {
            token.lower()
            for signal in (signals or [])
            for token in tokenize(str(signal))
            if len(token) > 1
        }
        if not signal_tokens:
            continue
        query_overlap = len(theme_tokens & signal_tokens)
        if query_overlap == 0:
            continue

        allowed_departments = {
            _normalize_exact_keyword_value(dep)
            for dep in (intent_departments_raw.get(intent, []) if isinstance(intent_departments_raw, dict) else [])
            if str(dep).strip()
        }
        if allowed_departments and any(dep in department_norm for dep in allowed_departments):
            continue

        evidence_overlap = len(chunk_tokens & signal_tokens) / max(1, len(signal_tokens))
        if evidence_overlap >= float(profile["evidence_overlap_threshold"]):
            continue

        total_penalty += float(profile["base_penalty"])

    return min(float(profile["max_penalty"]), max(0.0, total_penalty))


def _normalize_overexposure_penalty_config(
    overexposure_penalty_config: dict[str, Any],
) -> dict[str, float | bool]:
    config = overexposure_penalty_config or {}
    return {
        "enabled": bool(config.get("enabled", True)),
        "low_signal_threshold": min(
            1.0,
            max(0.0, float(config.get("low_signal_threshold", 0.35))),
        ),
        "profile_source_weight": max(
            0.0,
            float(config.get("profile_source_weight", 0.2)),
        ),
        "staffinfo_source_weight": max(
            0.0,
            float(config.get("staffinfo_source_weight", 0.12)),
        ),
        "publication_without_nva_penalty": max(
            0.0,
            float(config.get("publication_without_nva_penalty", 0.04)),
        ),
        "max_penalty": max(
            0.0,
            float(config.get("max_penalty", 0.1)),
        ),
    }


def _overexposure_score_penalty(
    *,
    chunks: Sequence[Chunk],
    citations: Sequence[Citation],
    themes: Sequence[str],
    match_mode: str,
    keyword_score: float,
    tag_score: float,
    method_score: float,
    overexposure_penalty_config: dict[str, Any],
    concept_keyword_map: dict[str, Sequence[str]] | None = None,
    synonym_expansion_map: dict[str, Sequence[str]] | None = None,
) -> float:
    profile = _normalize_overexposure_penalty_config(overexposure_penalty_config)
    if not bool(profile.get("enabled")):
        return 0.0

    support_signal = max(
        0.0,
        min(
            1.0,
            max(
                keyword_score,
                tag_score,
                method_score,
                _citation_overlap_score(
                    citations,
                    themes,
                    concept_keyword_map=concept_keyword_map,
                    synonym_expansion_map=synonym_expansion_map,
                ),
            ),
        ),
    )
    low_signal_threshold = float(profile["low_signal_threshold"])
    if support_signal >= low_signal_threshold:
        return 0.0

    source_coverage = _chunk_source_coverage(chunks)
    dominance_weight = (
        float(profile["profile_source_weight"]) * source_coverage.get("profile", 0.0)
        + float(profile["staffinfo_source_weight"])
        * source_coverage.get("staffinfo", 0.0)
    )
    penalty = max(0.0, low_signal_threshold - support_signal) * dominance_weight

    has_nva_citation = any(_citation_source_kind(citation) == "nva" for citation in citations)
    if match_mode == MATCH_MODE_PUBLICATION and not has_nva_citation:
        penalty += float(profile["publication_without_nva_penalty"])

    return min(float(profile["max_penalty"]), max(0.0, penalty))


def _collect_chunk_keywords(chunks: Sequence[Chunk]) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        raw_tags = chunk.metadata.get("tags") if chunk.metadata else None
        if not isinstance(raw_tags, list):
            continue
        for tag in raw_tags:
            if not isinstance(tag, str):
                continue
            normalized = tag.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            keywords.append(normalized)
    return keywords


def _keyword_overlap_score(
    chunks: Sequence[Chunk],
    themes: Sequence[str],
    *,
    concept_keyword_map: dict[str, Sequence[str]] | None = None,
    synonym_expansion_map: dict[str, Sequence[str]] | None = None,
) -> float:
    theme_tokens = _expanded_theme_tokens(
        themes,
        concept_keyword_map=concept_keyword_map,
        synonym_expansion_map=synonym_expansion_map,
    )
    if not theme_tokens:
        return 0.0
    text_tokens: set[str] = set()
    for chunk in chunks:
        text_tokens.update(
            token.lower() for token in tokenize(chunk.text) if len(token) > 2
        )
    if not text_tokens:
        return 0.0
    overlap = len(theme_tokens & text_tokens)
    return overlap / max(1, len(theme_tokens))


def _tag_overlap_score(
    keywords: Sequence[str],
    themes: Sequence[str],
    *,
    concept_keyword_map: dict[str, Sequence[str]] | None = None,
    synonym_expansion_map: dict[str, Sequence[str]] | None = None,
) -> float:
    if not keywords or not themes:
        return 0.0
    keyword_tokens = {
        token.lower()
        for keyword in keywords
        for token in tokenize(keyword)
        if len(token) > 2
    }
    theme_tokens = _expanded_theme_tokens(
        themes,
        concept_keyword_map=concept_keyword_map,
        synonym_expansion_map=synonym_expansion_map,
    )
    if not keyword_tokens or not theme_tokens:
        return 0.0
    overlap = len(keyword_tokens & theme_tokens)
    return overlap / max(1, len(theme_tokens))


def _method_overlap_score(keywords: Sequence[str], themes: Sequence[str]) -> float:
    if not keywords or not themes:
        return 0.0
    method_query_tokens = {
        token.lower()
        for method_keyword in METHOD_KEYWORDS
        for token in tokenize(method_keyword)
        if len(token) > 2
    }
    method_tokens = {
        token.lower()
        for keyword in keywords
        for token in tokenize(keyword)
        if len(token) > 2 and token.lower() in method_query_tokens
    }
    theme_tokens = _expanded_theme_tokens(themes)
    relevant_theme_method_tokens = theme_tokens & method_query_tokens
    if not method_tokens or not relevant_theme_method_tokens:
        return 0.0
    overlap = len(method_tokens & relevant_theme_method_tokens)
    return overlap / max(1, len(relevant_theme_method_tokens))


def _request_state(request: Request) -> Any:
    return request.app.state


def _localize_explanation(
    *,
    text: str | None,
    name: str,
    themes: Sequence[str],
    language_ctx: LanguageContext,
) -> str:
    """Ensure default explanations respect UI language even without translation."""
    safe_text = (text or "").strip()
    if safe_text:
        return safe_text
    joined = ", ".join(themes) or (
        "the topic" if language_ctx.user_lang.startswith("en") else "temaet"
    )
    if language_ctx.user_lang.startswith("en"):
        return f"{name} matches {joined} based on documented sources."
    return f"{name} matcher {joined} basert på dokumenterte kilder."


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.post("/analyze-topic", response_model=AnalyzeTopicResponse)
async def analyze_topic(request: Request) -> AnalyzeTopicResponse:
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        payload = await request.json()
        text_value = str(payload.get("text") or "")
    else:
        form = await request.form()
        text_value = str(form.get("text") or "")

    combined_text = _validate_text_input(text_value)

    themes = extract_themes(combined_text, top_k=8)
    preview = _build_normalized_preview(combined_text)

    return AnalyzeTopicResponse(themes=themes, normalizedPreview=preview)


@router.post("/match", response_model=MatchResponse)
async def match(request: Request, payload: MatchRequest) -> MatchResponse:
    if not payload.themes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Listen med temaer kan ikke være tom.",
        )

    state = _request_state(request)
    app_config: AppConfig = state.app_config
    language_config = getattr(state, "language_config", app_config.language)
    match_engine: MatchEngine = state.match_engine
    retriever: EmbeddingRetriever | None = getattr(state, "embedding_retriever", None)
    vector_index_ready: bool = getattr(state, "vector_index_ready", False)
    fetch_utils: FetchUtils = state.fetch_utils
    cache_manager: CacheManager = state.cache_manager
    staff_profiles: list[StaffProfile] = state.staff_profiles
    translator: Translator = getattr(state, "translator", None) or NoOpTranslator()
    precomputed_by_lang: dict[str, dict[str, str]] = getattr(
        state, "precomputed_summaries_by_lang", None
    ) or {
        "no": getattr(state, "precomputed_summaries", {}),
        "en": {},
    }

    query_text = _rag_query_from_themes(
        payload.themes,
        synonym_expansion_map=app_config.results.synonym_expansion_map,
    )
    user_lang = _resolve_user_language(request, app_config)
    language_ctx = build_language_context(
        query_text=query_text,
        user_lang=user_lang,
        language_config=language_config,
    )
    default_summary_lang = (
        language_config.default_ui or app_config.ui.language or "no"
    ).lower()

    if retriever and vector_index_ready:
        mode_scoring_profiles = {
            mode: profile.model_dump()
            for mode, profile in app_config.results.mode_scoring_profiles.items()
        }
        rag_results = _match_via_retriever(
            retriever=retriever,
            payload=payload,
            match_mode=payload.mode,
            max_candidates=app_config.results.max_candidates,
            translator=translator,
            language_ctx=language_ctx,
            query_text=query_text,
            citation_source_priority=app_config.results.citation_source_priority,
            min_query_overlap_per_citation=(
                app_config.results.min_query_overlap_per_citation
            ),
            profile_staffinfo_min_query_overlap_per_citation=(
                app_config.results.profile_staffinfo_min_query_overlap_per_citation
            ),
            scoring_weights=app_config.results.scoring_weights.model_dump(),
            mode_scoring_profiles=mode_scoring_profiles,
            exact_keyword_promotion_config=(
                app_config.results.exact_keyword_promotion.model_dump()
            ),
            overexposure_penalty_config=(
                app_config.results.overexposure_penalty.model_dump()
            ),
            category_intent_penalty_config=(
                app_config.results.category_intent_penalty.model_dump()
            ),
            concept_keyword_map=app_config.results.concept_keyword_map,
            synonym_expansion_map=app_config.results.synonym_expansion_map,
        )
        if not retriever.is_active:
            state.vector_index_ready = False
            logger.warning(
                "Vector index disabled (%s). Falling back to legacy matcher.",
                retriever.disabled_reason,
            )
        if rag_results:
            requested_summary_lang = _normalize_summary_lang(language_ctx.user_lang)
            for item in rag_results:
                why_by_lang = _build_precomputed_summaries_by_lang(
                    precomputed_by_lang=precomputed_by_lang,
                    profile_url=item.profile_url,
                    name=item.name,
                )
                if why_by_lang:
                    item.why_by_lang = why_by_lang
                precomputed = _lookup_precomputed_summary_by_lang(
                    precomputed_by_lang=precomputed_by_lang,
                    user_lang=requested_summary_lang,
                    profile_url=item.profile_url,
                    name=item.name,
                    default_lang=default_summary_lang,
                )
                if precomputed:
                    item.why = precomputed
            return MatchResponse(results=rag_results, total=len(rag_results))

    if payload.department:
        filtered_staff = [
            profile
            for profile in staff_profiles
            if profile.department == payload.department
        ]
    else:
        filtered_staff = staff_profiles

    if not filtered_staff:
        return MatchResponse(results=[], total=0)

    documents = await _fetch_staff_documents(
        staff_profiles=filtered_staff,
        fetch_utils=fetch_utils,
        cache_manager=cache_manager,
        max_pages=app_config.fetch.max_pages_per_staff,
        memory_cache=getattr(state, "staff_document_cache", None),
    )

    if not documents:
        return MatchResponse(results=[], total=0)

    ranked = match_engine.rank(
        documents=documents,
        themes=payload.themes,
        department_filter=payload.department,
        max_candidates=app_config.results.max_candidates,
        diversity_weight=app_config.results.diversity_weight,
    )

    filtered_ranked = [
        result
        for result in ranked
        if result.score_breakdown["semantic"] >= app_config.results.min_similarity_score
    ]
    if not filtered_ranked and ranked:
        # Fall back to top matches when semantic similarity is low but other signals
        # (keywords, tags) still indicate relevance. Keeps the UI populated while
        # highlighting that tuning may be required.
        filtered_ranked = [
            candidate for candidate in ranked if candidate.score > 0
        ] or ranked[:1]

    results: list[MatchItem] = []
    for match in filtered_ranked:
        score_breakdown = dict(match.score_breakdown)
        if "keyword" in score_breakdown and "keywords" not in score_breakdown:
            score_breakdown["keywords"] = score_breakdown.pop("keyword")
        why_by_lang = _build_precomputed_summaries_by_lang(
            precomputed_by_lang=precomputed_by_lang,
            profile_url=match.staff.profile_url,
            name=match.staff.name,
        )
        precomputed = _lookup_precomputed_summary_by_lang(
            precomputed_by_lang=precomputed_by_lang,
            user_lang=language_ctx.user_lang,
            profile_url=match.staff.profile_url,
            name=match.staff.name,
            default_lang=default_summary_lang,
        )
        item = MatchItem(
            id=_hash_id(match.staff.name, match.staff.profile_url),
            name=match.staff.name,
            department=match.staff.department,
            profile_url=match.staff.profile_url,
            score=match.score,
            why=_choose_why_text(
                precomputed=precomputed,
                fallback_explanation=match.explanation,
                staff_name=match.staff.name,
                themes=payload.themes,
                language_ctx=language_ctx,
            ),
            whyByLang=why_by_lang,
            citations=[Citation(**citation) for citation in match.citations],
            scoreBreakdown=score_breakdown,
            keywords=list(match.staff.tags),
        )
        results.append(item)

    return MatchResponse(results=results, total=len(results))


@router.get("/config", response_model=ConfigResponse)
async def get_config(request: Request) -> ConfigResponse:
    state = _request_state(request)
    app_config: AppConfig = state.app_config
    staff_profiles: list[StaffProfile] = state.staff_profiles

    departments = sorted({profile.department for profile in staff_profiles})
    return ConfigResponse(
        departments=departments,
        ui={
            "allowDepartmentFilter": app_config.ui.allow_department_filter,
            "language": app_config.ui.language,
            "exportFormats": app_config.ui.export_formats,
        },
        security={
            "maxUploadMb": app_config.security.max_upload_mb,
            "allowFileTypes": app_config.security.allow_file_types,
        },
    )
