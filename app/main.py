"""FastAPI entry point for the ONH Expert Connector service."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import routes
from .cache_manager import CacheManager
from .config_loader import load_app_config, load_models_config, load_staff_profiles
from .exporter import ShortlistExporter
from .fetch_utils import FetchUtils
from .file_parser import FileParser
from .index import IndexPaths, LocalVectorStore, create_embedding_backend
from .llm_explainer import LLMExplainer
from .language_utils import create_translator
from .match_engine import MatchEngine
from .shortlist_store import ShortlistStore
from .rag import EmbeddingRetriever

LOGGER = logging.getLogger(__name__)


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent


def create_app() -> FastAPI:
    app = FastAPI(
        title="ONH Expert Connector",
        description="Helps ONH teachers find relevant colleagues with grounded explanations.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app_config = load_app_config()
    models_config = load_models_config()
    staff_profiles = load_staff_profiles()

    cache_dir = (PROJECT_ROOT / app_config.cache.directory).resolve()
    cache_manager = CacheManager(
        directory=cache_dir,
        retention_days=app_config.cache.retention_days,
        enabled=app_config.cache.enabled,
    )

    fetch_utils = FetchUtils(
        allowlist_domains=app_config.fetch.allowlist_domains,
        max_kb_per_page=app_config.fetch.max_kb_per_page,
        offline_snapshots_dir=app_config.fetch.offline_snapshots_dir,
    )

    llm_explainer = LLMExplainer(model_config=models_config.llm_model.model_dump())
    translator = create_translator(app_config.language.translation)

    exports_dir = PROJECT_ROOT / "outputs" / "exports"
    shortlist_store = ShortlistStore(exports_dir / "shortlist.json")
    shortlist_exporter = ShortlistExporter(exports_dir)

    match_engine = MatchEngine(
        embedding_model_name=models_config.embedding_model.name,
        embedding_backend=models_config.embedding_model.backend,
        embedding_device=models_config.embedding_model.device or "auto",
        embedding_endpoint=models_config.embedding_model.endpoint,
    )

    index_root = (PROJECT_ROOT / app_config.rag.index_root).resolve()
    index_paths = IndexPaths(root=index_root)
    vector_store = LocalVectorStore(index_paths.vectors_dir)
    retriever_embedder = create_embedding_backend(models_config, app_config=app_config)
    embedding_retriever = EmbeddingRetriever(
        vector_store=vector_store,
        embedder=retriever_embedder,
        min_score=app_config.results.min_similarity_score,
        max_chunks_per_staff=3,
    )
    vector_index_ready = len(vector_store) > 0

    app.state.app_config = app_config
    app.state.models_config = models_config
    app.state.staff_profiles = staff_profiles
    app.state.cache_manager = cache_manager
    app.state.fetch_utils = fetch_utils
    app.state.file_parser = FileParser()
    app.state.match_engine = match_engine
    app.state.llm_explainer = llm_explainer
    app.state.translator = translator
    app.state.language_config = app_config.language
    app.state.shortlist_store = shortlist_store
    app.state.shortlist_exporter = shortlist_exporter
    app.state.vector_store = vector_store
    app.state.embedding_retriever = embedding_retriever
    app.state.vector_index_ready = vector_index_ready
    app.state.staff_document_cache = {}

    app.include_router(routes.router)

    static_dir = APP_ROOT / "static"
    favicon_path = static_dir / "favicon.png"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.on_event("startup")
    async def warm_staff_documents() -> None:  # pragma: no cover - startup hook
        try:
            await routes.warm_staff_document_cache(app.state)
        except Exception as exc:  # noqa: BLE001 - log and continue
            LOGGER.warning("Klarte ikke å varme opp dokumentcache: %s", exc)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_index() -> HTMLResponse:  # pragma: no cover - simple file read
        index_path = static_dir / "index.html"
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/favicon.ico", include_in_schema=False)
    async def serve_favicon() -> FileResponse:
        return FileResponse(favicon_path, media_type="image/png")

    return app


app = create_app()
