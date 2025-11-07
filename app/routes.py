"""API routers for the ONH Expert Connector."""

from __future__ import annotations

import asyncio
import hashlib
import logging
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
from .match_engine import MatchEngine, PageContent, StaffDocument, StaffProfile, extract_themes
from .rag import EmbeddingRetriever, RetrievalQuery
from .shortlist_store import ShortlistStore

router = APIRouter()
logger = logging.getLogger(__name__)
CITATION_SNIPPET_LIMIT = 280


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


async def _fetch_staff_documents(
    *,
    staff_profiles: Sequence[StaffProfile],
    fetch_utils: FetchUtils,
    cache_manager: CacheManager,
    max_pages: int,
) -> list[StaffDocument]:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        tasks = [
            _fetch_single_staff(profile, client, fetch_utils, cache_manager, max_pages)
            for profile in staff_profiles
        ]
        documents = await asyncio.gather(*tasks)
    return [doc for doc in documents if doc.pages]


async def _fetch_single_staff(
    profile: StaffProfile,
    client: httpx.AsyncClient,
    fetch_utils: FetchUtils,
    cache_manager: CacheManager,
    max_pages: int,
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
            "text": page_data.get("text", "")[: fetch_utils.max_bytes_per_page],
        }
        cache_manager.set(cache_key, cleaned)
        pages.append(PageContent(**cleaned))
    return StaffDocument(profile=profile, pages=pages)


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
    citations = _chunks_to_citations(result.chunks)

    why_default = (
        f"{display_name} matcher {', '.join(themes) or 'temaet'} basert på semantisk treff."
    )
    return MatchItem(
        id=_hash_id(display_name, profile_url or result.staff_slug),
        name=display_name,
        department=department,
        profile_url=profile_url or primary_chunk.source_url or "",
        score=result.score,
        why=why_default,
        citations=citations,
        scoreBreakdown={
            "semantic": result.score,
            "keywords": 0.0,
            "tags": 0.0,
        },
    )


def _chunks_to_citations(chunks: Sequence[Chunk]) -> list[Citation]:
    citations: list[Citation] = []
    for idx, chunk in enumerate(chunks, start=1):
        snippet = chunk.text.strip()
        if not snippet:
            continue
        snippet = snippet[:CITATION_SNIPPET_LIMIT].rstrip()
        title = str(chunk.metadata.get("source_title") or chunk.metadata.get("name") or "Kilde")
        url = chunk.source_url or str(chunk.metadata.get("profile_url") or "")
        citations.append(
            Citation(
                id=f"[{idx}]",
                title=title or "Kilde",
                url=url,
                snippet=snippet,
            )
        )
    return citations


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
    preview = combined_text[:600]

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
        item = MatchItem(
            id=_hash_id(match.staff.name, match.staff.profile_url),
            name=match.staff.name,
            department=match.staff.department,
            profile_url=match.staff.profile_url,
            score=match.score,
            why=match.explanation,
            citations=[Citation(**citation) for citation in match.citations],
            scoreBreakdown=match.score_breakdown,
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
