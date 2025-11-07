from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.cache_manager import CacheManager
from app.exporter import ShortlistExporter
from app.index import embedder_factory
from app.index.builder import DummyEmbeddingBackend


class TestSentenceTransformerBackend:
    def __init__(self, model_name, batch_size=32, device="cpu"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._dummy = DummyEmbeddingBackend()

    def embed(self, texts):
        return self._dummy.embed(texts)

    def embed_one(self, text):
        return self._dummy.embed_one(text)


embedder_factory.SentenceTransformerBackend = TestSentenceTransformerBackend  # type: ignore[attr-defined]

from app.main import create_app
from app.match_engine import MatchEngine, StaffProfile
from app.shortlist_store import ShortlistStore


async def _prepare_app(
    tmp_path,
    monkeypatch,
    *,
    staff_profiles: list[StaffProfile] | None = None,
) -> tuple[AsyncClient, FastAPI]:
    app = create_app()

    app.state.match_engine = MatchEngine()
    app.state.embedding_retriever = None
    app.state.vector_index_ready = False

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    app.state.cache_manager = CacheManager(directory=cache_dir, retention_days=1, enabled=True)

    exports_dir = tmp_path / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    app.state.shortlist_store = ShortlistStore(exports_dir / "shortlist.json")
    app.state.shortlist_exporter = ShortlistExporter(exports_dir)

    default_profiles = staff_profiles or [
        StaffProfile(
            name="Test Forsker",
            department="Psykologi",
            profile_url="https://example.com/ansatt",
            sources=["https://example.com/ansatt"],
            tags=["psykologi", "helse"],
        )
    ]
    app.state.staff_profiles = default_profiles

    async def fake_fetch_page(self, client_http, url):  # type: ignore[override]
        return {
            "url": url,
            "title": "Stub-kilde",
            "text": "Forskning på psykologi og helse ved Oslo Nye Høyskole.",
        }

    async def fake_generate(self, staff_name, snippets, themes):  # type: ignore[override]
        return f"{staff_name} matcher {', '.join(themes)} basert på testdata."

    monkeypatch.setattr(app.state.fetch_utils.__class__, "fetch_page", fake_fetch_page, raising=False)
    monkeypatch.setattr(app.state.llm_explainer.__class__, "generate", fake_generate, raising=False)

    transport = ASGITransport(app=app)
    async_client = AsyncClient(transport=transport, base_url="http://testserver", timeout=10.0)
    async_client.app = app  # type: ignore[attr-defined]
    return async_client, app


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch):
    async_client, app = await _prepare_app(tmp_path, monkeypatch)
    try:
        yield async_client
    finally:
        await async_client.aclose()


@pytest_asyncio.fixture
async def offline_client(tmp_path, monkeypatch):
    profiles = [
        StaffProfile(
            name="Alex Gillespie",
            department="Psykologi",
            profile_url="https://oslonyehoyskole.no/om-oss/Ansatte/Alex-Gillespie",
            sources=[
                "https://oslonyehoyskole.no/om-oss/Ansatte/Alex-Gillespie",
                "https://nva.sikt.no/research-profile/1306894",
            ],
            tags=["psykologi"],
        )
    ]
    async_client, app = await _prepare_app(tmp_path, monkeypatch, staff_profiles=profiles)

    async def fake_generate(self, staff_name, snippets, themes):  # type: ignore[override]
        return f"{staff_name} matcher {', '.join(themes)} basert på offline testdata."

    monkeypatch.setattr(app.state.llm_explainer.__class__, "generate", fake_generate, raising=False)

    async def offline_get(self, url, timeout=30.0, follow_redirects=True):  # type: ignore[override]
        raise httpx.RequestError("offline", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", offline_get, raising=False)

    try:
        yield async_client
    finally:
        await async_client.aclose()
