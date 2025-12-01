"""API routers for the ONH Expert Connector."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any, Literal, Sequence

import httpx
from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .cache_manager import CacheManager
from .config_loader import AppConfig
from .exporter import ShortlistExporter
from .fetch_utils import FetchNotAllowedError, FetchUtils
from .file_parser import FileParser, UnsupportedFileTypeError
from .index.models import Chunk
from .llm_explainer import LLMExplainer
from .match_engine import (
    MatchEngine,
    PageContent,
    StaffDocument,
    StaffProfile,
    extract_themes,
    tokenize,
)
from .rag import EmbeddingRetriever, RetrievalQuery
from .shortlist_store import ShortlistStore

router = APIRouter()
logger = logging.getLogger(__name__)
PAGE_TEXT_CHAR_LIMIT = 8000
CITATION_SNIPPET_LIMIT = 3000
PREVIEW_CHAR_LIMIT = 1200


# --------------------------------------------------------------------------- #
# Pydantic schemas
# --------------------------------------------------------------------------- #


class AnalyzeTopicResponse(BaseModel):
    themes: list[str]
    normalized_preview: str = Field(alias="normalizedPreview")


class MatchRequest(BaseModel):
    themes: list[str]
    department: str | None = None


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


class ShortlistItem(BaseModel):
    id: str
    name: str
    department: str
    why: str | None = None
    citations: list[Citation] | None = None
    notes: str | None = None
    score: float | None = None
    keywords: list[str] | None = None

    model_config = {"extra": "allow"}


class ShortlistPayload(BaseModel):
    items: list[ShortlistItem]


class ConfigResponse(BaseModel):
    departments: list[str]
    ui: dict[str, Any]
    security: dict[str, Any]


class ExportRequest(BaseModel):
    format: Literal["pdf", "json"]
    metadata: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _validate_text_input(text: str, file_texts: Sequence[str]) -> str:
    combined = " ".join(part for part in [text.strip(), *file_texts] if part)
    if not combined:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mangler tematisk innhold. Skriv tekst eller last opp filer.",
        )
    return combined


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


def _extract_citation_snippet(text: str, themes: Sequence[str], limit: int = CITATION_SNIPPET_LIMIT) -> str:
    """Pick the most relevant portion of a chunk for downstream LLM use."""

    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return ""
    if len(normalized) <= limit:
        return normalized

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    theme_tokens = {
        token
        for theme in themes
        for token in tokenize(theme)
        if len(token) > 2
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

            if (
                total_score > best_score
                or (total_score == best_score and total_len > len(best_snippet))
            ):
                best_score = total_score
                best_snippet = " ".join(window)

    if len(best_snippet) > limit:
        best_snippet = best_snippet[:limit]
        last_space = best_snippet.rfind(" ")
        if last_space > int(limit * 0.6):
            best_snippet = best_snippet[:last_space]

    return best_snippet.strip()


async def _read_files(
    *,
    files: Sequence[UploadFile] | None,
    file_parser: FileParser,
    app_config: AppConfig,
) -> list[str]:
    if not files:
        return []

    allowed_suffixes = {f".{suffix.lower().lstrip('.')}" for suffix in app_config.security.allow_file_types}
    max_bytes = app_config.security.max_upload_mb * 1024 * 1024
    parsed_texts: list[str] = []

    for upload in files:
        if upload.filename is None:
            continue
        suffix = "." + upload.filename.split(".")[-1].lower()
        if allowed_suffixes and suffix not in allowed_suffixes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Filtypen {suffix} er ikke tillatt.",
            )
        content = await upload.read()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Filen {upload.filename} overstiger maks størrelse på {app_config.security.max_upload_mb} MB.",
            )
        try:
            parsed_texts.append(file_parser.parse_bytes(upload.filename, content))
        except UnsupportedFileTypeError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return parsed_texts


def _staff_doc_cache_key(profile: StaffProfile, max_pages: int) -> str:
    digest = hashlib.sha1(f"{profile.profile_url}|{max_pages}".encode("utf-8")).hexdigest()
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
    _remember_staff_document(cache_key=cache_key, sources=sources, pages=pages, memory_cache=memory_cache)


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
        for profile, document in zip(pending_profiles, fetched):
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
            "url": page_data["url"],
            "title": page_data.get("title") or "",
            "text": page_data.get("text", "")[: PAGE_TEXT_CHAR_LIMIT],
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


async def _maybe_generate_explanations(
    results: list[MatchItem],
    llm_explainer: LLMExplainer,
    themes: Sequence[str],
) -> None:
    tasks = []
    for result in results:
        snippets = [citation.snippet for citation in result.citations]
        if not snippets:
            continue
        tasks.append(llm_explainer.generate(result.name, snippets, themes))

    if not tasks:
        return

    explanations = await asyncio.gather(*tasks, return_exceptions=True)
    idx = 0
    for result in results:
        if not result.citations:
            continue
        explanation = explanations[idx]
        idx += 1
        if isinstance(explanation, Exception):
            continue
        cleaned = explanation.strip()
        if cleaned:
            result.why = cleaned


def _rag_query_from_themes(themes: Sequence[str]) -> str:
    return " ".join(theme.strip() for theme in themes if theme.strip()).strip()


def _match_via_retriever(
    *,
    retriever: EmbeddingRetriever,
    payload: MatchRequest,
    max_candidates: int,
) -> list[MatchItem]:
    query_text = _rag_query_from_themes(payload.themes)
    if not query_text:
        return []
    try:
        results = retriever.retrieve(
            RetrievalQuery(
                text=query_text,
                department=payload.department,
                top_k=max_candidates,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Retriever failed for '%s': %s", query_text, exc)
        return []

    items: list[MatchItem] = []
    for result in results:
        item = _retrieval_result_to_match_item(result, payload.themes)
        if item:
            items.append(item)
    return items


def _retrieval_result_to_match_item(result, themes: Sequence[str]) -> MatchItem | None:
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
    display_name = result.staff_name or str(primary_chunk.metadata.get("name") or result.staff_slug)
    citations = _chunks_to_citations(result.chunks, themes)
    keywords = _collect_chunk_keywords(result.chunks)
    semantic_score = max(0.0, min(1.0, float(result.score)))
    keyword_score = _keyword_overlap_score(result.chunks, themes)
    tag_score = _tag_overlap_score(keywords, themes)
    adjusted_score = min(1.0, semantic_score + 0.1 * keyword_score + 0.15 * tag_score)

    why_default = (
        f"{display_name} matcher {', '.join(themes) or 'temaet'} basert på semantisk treff."
    )
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
            "keywords": round(keyword_score, 4),
            "tags": round(tag_score, 4),
        },
        keywords=keywords,
    )


def _resolve_citation_url(chunk: Chunk) -> str:
    metadata = chunk.metadata or {}
    publication_id = metadata.get("nva_publication_id")
    if isinstance(publication_id, str) and publication_id.strip():
        return f"https://nva.sikt.no/registration/{publication_id.strip()}"

    url = chunk.source_url or str(metadata.get("profile_url") or "")
    nva_prefixes = (
        "https://api.nva.unit.no/publication/",
        "https://api.test.nva.aws.unit.no/publication/",
    )
    for prefix in nva_prefixes:
        if url.startswith(prefix):
            pub_id = url.rstrip("/").split("/")[-1]
            if pub_id:
                return f"https://nva.sikt.no/registration/{pub_id}"

    return url


def _chunks_to_citations(chunks: Sequence[Chunk], themes: Sequence[str]) -> list[Citation]:
    citations: list[Citation] = []
    for idx, chunk in enumerate(chunks, start=1):
        snippet = _extract_citation_snippet(chunk.text, themes)
        if not snippet:
            continue
        title = str(chunk.metadata.get("source_title") or chunk.metadata.get("name") or "Kilde")
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


def _keyword_overlap_score(chunks: Sequence[Chunk], themes: Sequence[str]) -> float:
    theme_tokens = {theme.strip().lower() for theme in themes if theme.strip()}
    if not theme_tokens:
        return 0.0
    text_tokens: set[str] = set()
    for chunk in chunks:
        text_tokens.update(tokenize(chunk.text))
    if not text_tokens:
        return 0.0
    overlap = len(theme_tokens & text_tokens)
    return overlap / max(1, len(theme_tokens))


def _tag_overlap_score(keywords: Sequence[str], themes: Sequence[str]) -> float:
    if not keywords or not themes:
        return 0.0
    keyword_tokens = {tag.strip().lower() for tag in keywords if tag.strip()}
    theme_tokens = {theme.strip().lower() for theme in themes if theme.strip()}
    if not keyword_tokens or not theme_tokens:
        return 0.0
    overlap = len(keyword_tokens & theme_tokens)
    return overlap / max(1, len(keyword_tokens))


def _request_state(request: Request):
    return request.app.state


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.post("/analyze-topic", response_model=AnalyzeTopicResponse)
async def analyze_topic(request: Request) -> AnalyzeTopicResponse:
    state = _request_state(request)
    file_parser: FileParser = state.file_parser
    app_config: AppConfig = state.app_config

    content_type = (request.headers.get("content-type") or "").lower()
    files: list[UploadFile] | None = None
    parsed_texts: list[str]

    if "application/json" in content_type:
        payload = await request.json()
        text_value = str(payload.get("text") or "")
        parsed_texts = []
    else:
        form = await request.form()
        text_value = str(form.get("text") or "")
        upload_items = [item for item in form.getlist("files") if isinstance(item, UploadFile)]

        files = upload_items or None
        parsed_texts = await _read_files(files=files, file_parser=file_parser, app_config=app_config)

    combined_text = _validate_text_input(text_value, parsed_texts)

    themes = extract_themes(combined_text, top_k=8)
    preview = _build_normalized_preview(combined_text)

    return AnalyzeTopicResponse(themes=themes, normalizedPreview=preview)


@router.post("/match", response_model=MatchResponse)
async def match(request: Request, payload: MatchRequest) -> MatchResponse:
    if not payload.themes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Listen med temaer kan ikke være tom.")

    state = _request_state(request)
    app_config: AppConfig = state.app_config
    match_engine: MatchEngine = state.match_engine
    retriever: EmbeddingRetriever | None = getattr(state, "embedding_retriever", None)
    vector_index_ready: bool = getattr(state, "vector_index_ready", False)
    fetch_utils: FetchUtils = state.fetch_utils
    cache_manager: CacheManager = state.cache_manager
    staff_profiles: list[StaffProfile] = state.staff_profiles
    llm_explainer: LLMExplainer = state.llm_explainer

    if retriever and vector_index_ready:
        rag_results = _match_via_retriever(
            retriever=retriever,
            payload=payload,
            max_candidates=app_config.results.max_candidates,
        )
        if not retriever.is_active:
            state.vector_index_ready = False
            logger.warning(
                "Vector index disabled (%s). Falling back to legacy matcher.",
                retriever.disabled_reason,
            )
        if rag_results:
            await _maybe_generate_explanations(rag_results, llm_explainer, payload.themes)
            return MatchResponse(results=rag_results, total=len(rag_results))

    if payload.department:
        filtered_staff = [profile for profile in staff_profiles if profile.department == payload.department]
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
        filtered_ranked = [candidate for candidate in ranked if candidate.score > 0] or ranked[:1]

    results: list[MatchItem] = []
    for match in filtered_ranked:
        score_breakdown = dict(match.score_breakdown)
        if "keyword" in score_breakdown and "keywords" not in score_breakdown:
            score_breakdown["keywords"] = score_breakdown.pop("keyword")
        item = MatchItem(
            id=_hash_id(match.staff.name, match.staff.profile_url),
            name=match.staff.name,
            department=match.staff.department,
            profile_url=match.staff.profile_url,
            score=match.score,
            why=match.explanation,
            citations=[Citation(**citation) for citation in match.citations],
            scoreBreakdown=score_breakdown,
            keywords=list(match.staff.tags),
        )
        results.append(item)

    await _maybe_generate_explanations(results, llm_explainer, payload.themes)

    return MatchResponse(results=results, total=len(results))


@router.get("/shortlist", response_model=ShortlistPayload)
async def get_shortlist(request: Request) -> ShortlistPayload:
    state = _request_state(request)
    shortlist_store: ShortlistStore = state.shortlist_store
    items = shortlist_store.load()
    return ShortlistPayload(items=[ShortlistItem.model_validate(item) for item in items])


@router.post("/shortlist", response_model=ShortlistPayload)
async def save_shortlist(request: Request, payload: ShortlistPayload) -> ShortlistPayload:
    state = _request_state(request)
    shortlist_store: ShortlistStore = state.shortlist_store
    shortlist_store.save([item.model_dump(mode="json") for item in payload.items])
    return payload


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


@router.post("/shortlist/export")
async def export_shortlist(request: Request, payload: ExportRequest) -> dict[str, str]:
    state = _request_state(request)
    shortlist_store: ShortlistStore = state.shortlist_store
    exporter: ShortlistExporter = state.shortlist_exporter

    items = shortlist_store.load()
    if not items:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Kortlisten er tom.")

    path = exporter.export(payload.format, items, payload.metadata)
    try:
        project_root = exporter.base_dir.parent.parent
        relative = path.relative_to(project_root)
    except ValueError:
        relative = path.relative_to(exporter.base_dir.parent)
    return {"path": str(relative)}
