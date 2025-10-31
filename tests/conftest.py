from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app.cache_manager import CacheManager
from app.exporter import ShortlistExporter
from app.main import create_app
from app.match_engine import MatchEngine, StaffProfile
from app.shortlist_store import ShortlistStore


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    app = create_app()

    app.state.match_engine = MatchEngine()

    cache_dir = tmp_path / "cache"
    app.state.cache_manager = CacheManager(directory=cache_dir, retention_days=1, enabled=True)

    exports_dir = tmp_path / "exports"
    app.state.shortlist_store = ShortlistStore(exports_dir / "shortlist.json")
    app.state.shortlist_exporter = ShortlistExporter(exports_dir)

    app.state.staff_profiles = [
        StaffProfile(
            name="Test Forsker",
            department="Psykologi",
            profile_url="https://example.com/ansatt",
            sources=["https://example.com/ansatt"],
            tags=["psykologi", "helse"],
        )
    ]

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

    return TestClient(app)


@pytest.fixture
def offline_client(tmp_path, monkeypatch) -> TestClient:
    app = create_app()

    app.state.match_engine = MatchEngine()

    cache_dir = tmp_path / "cache"
    app.state.cache_manager = CacheManager(directory=cache_dir, retention_days=1, enabled=True)

    exports_dir = tmp_path / "exports"
    app.state.shortlist_store = ShortlistStore(exports_dir / "shortlist.json")
    app.state.shortlist_exporter = ShortlistExporter(exports_dir)

    app.state.staff_profiles = [
        profile for profile in app.state.staff_profiles if profile.name == "Alex Gillespie"
    ]
    if not app.state.staff_profiles:
        raise RuntimeError("Alex Gillespie profile missing from staff.yaml")

    async def fake_generate(self, staff_name, snippets, themes):  # type: ignore[override]
        return f"{staff_name} matcher {', '.join(themes)} basert på offline testdata."

    monkeypatch.setattr(app.state.llm_explainer.__class__, "generate", fake_generate, raising=False)

    async def offline_get(self, url, timeout=30.0):  # type: ignore[override]
        raise httpx.RequestError("offline", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", offline_get, raising=False)

    return TestClient(app)
