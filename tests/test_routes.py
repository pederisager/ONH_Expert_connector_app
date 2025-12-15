from __future__ import annotations

import types

import pytest

from app import routes
from app.cache_manager import CacheManager
from app.index.builder import DummyEmbeddingBackend
from app.index.models import Chunk
from app.index.vector_store import LocalVectorStore
from app.match_engine import StaffProfile
from app.rag.retriever import EmbeddingRetriever
from app.fetch_utils import FetchUtils


@pytest.mark.asyncio
async def test_queue_head_returns_204(client) -> None:
    response = await client.head("/queue")
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_analyze_topic_returns_themes(client) -> None:
    response = await client.post(
        "/analyze-topic",
        json={"text": "Psykologi og helse ved Oslo Nye Høyskole."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["themes"]
    assert "normalizedPreview" in data


@pytest.mark.asyncio
async def test_match_endpoint_returns_ranked_result(client) -> None:
    response = await client.post(
        "/match",
        json={"themes": ["psykologi", "helse"], "department": "Psykologi"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"], "Forventer minst ett treff"
    top = data["results"][0]
    assert top["citations"]


@pytest.mark.asyncio
async def test_match_endpoint_uses_precomputed_summary_language(client) -> None:
    response_no = await client.post(
        "/match",
        json={"themes": ["psykologi"], "department": "Psykologi"},
        headers={"X-UI-Language": "no"},
    )
    assert response_no.status_code == 200
    payload_no = response_no.json()["results"][0]
    why_no = payload_no["why"]
    assert "forhåndsgenerert" in why_no.lower()
    assert payload_no.get("whyByLang", {}).get("no")

    response_en = await client.post(
        "/match",
        json={"themes": ["psykologi"], "department": "Psykologi"},
        headers={"X-UI-Language": "en"},
    )
    assert response_en.status_code == 200
    payload_en = response_en.json()["results"][0]
    why_en = payload_en["why"]
    assert "precomputed summary" in why_en.lower()
    assert payload_en.get("whyByLang", {}).get("en")


@pytest.mark.asyncio
async def test_match_endpoint_offline_snapshot(offline_client) -> None:
    response = await offline_client.post(
        "/match",
        json={
            "themes": [
                "psykologi",
                "klinisk",
                "traumebehandling",
                "kognitiv",
            ],
            "department": "Psykologi",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"], "Forventer treff basert på offline snapshot"
    names = {item["name"] for item in data["results"]}
    assert "Alex Gillespie" in names


@pytest.mark.asyncio
async def test_match_endpoint_prefers_rag_when_index_ready(client, tmp_path_factory) -> None:
    index_dir = tmp_path_factory.mktemp("rag_vectors")
    store = LocalVectorStore(index_dir)
    embedder = DummyEmbeddingBackend(dimension=3)

    chunk = Chunk(
        staff_slug="test-forsker",
        chunk_id="test-forsker-0000",
        text="Psykologi og helse forskning ved ONH.",
        order=0,
        token_count=6,
        source_url="https://example.org/test-forsker",
        metadata={
            "name": "RAG Forsker",
            "department": "Psykologi",
            "profile_url": "https://example.org/test-forsker",
            "title": "Førsteamanuensis",
        },
    )
    store.add(embedder.embed([chunk.text]), [chunk])

    retriever = EmbeddingRetriever(vector_store=store, embedder=embedder, min_score=0.0, max_chunks_per_staff=2)
    client.app.state.embedding_retriever = retriever  # type: ignore[attr-defined]
    client.app.state.vector_index_ready = True  # type: ignore[attr-defined]

    response = await client.post(
        "/match",
        json={"themes": ["psykologi", "helse"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"], "RAG-søket skal levere treff"
    top = data["results"][0]
    assert top["name"] == "RAG Forsker"
    assert top["citations"][0]["snippet"].startswith("Psykologi og helse")


@pytest.mark.asyncio
async def test_staff_document_cache_reuse(tmp_path) -> None:
    cache_manager = CacheManager(directory=tmp_path / "cache", retention_days=1, enabled=True)
    fetch_utils = FetchUtils(allowlist_domains=["example.com"], max_kb_per_page=10)

    fetch_calls = {"count": 0}

    async def fake_fetch_page(self, client_http, url):  # type: ignore[override]
        fetch_calls["count"] += 1
        return {"url": url, "title": "Stub", "text": "Lang tekst for caching."}

    fetch_utils.fetch_page = types.MethodType(fake_fetch_page, fetch_utils)

    profile = StaffProfile(
        name="Cache Test",
        department="Helsefag",
        profile_url="https://example.com/profile",
        sources=["https://example.com/source"],
        tags=[],
    )

    docs_first = await routes._fetch_staff_documents(  # type: ignore[attr-defined]
        staff_profiles=[profile],
        fetch_utils=fetch_utils,
        cache_manager=cache_manager,
        max_pages=1,
    )
    assert fetch_calls["count"] == 1
    assert docs_first and docs_first[0].combined_text

    docs_second = await routes._fetch_staff_documents(  # type: ignore[attr-defined]
        staff_profiles=[profile],
        fetch_utils=fetch_utils,
        cache_manager=cache_manager,
        max_pages=1,
    )
    assert fetch_calls["count"] == 1, "Expected cached StaffDocument to skip fetch"
    assert docs_second and docs_second[0].combined_text == docs_first[0].combined_text


def test_build_normalized_preview_prefers_sentences() -> None:
    text = (
        "Første setning beskriver psykologi. Andre setning utdyper helseperspektivet. "
        "Tredje setning er overflødig."
    )
    preview = routes._build_normalized_preview(text, limit=80)
    assert preview.endswith(".")
    assert "overflødig" not in preview


def test_chunks_to_citations_maps_nva_url() -> None:
    chunk = Chunk(
        staff_slug="slug",
        chunk_id="slug-0001",
        text="Resultatet beskriver digital sikkerhet og personvern i undervisning.",
        order=0,
        token_count=12,
        source_url="https://api.nva.unit.no/publication/abc123",
        metadata={
            "name": "Test Forsker",
            "nva_publication_id": "abc123",
        },
    )
    citations = routes._chunks_to_citations([chunk], themes=[])
    assert citations
    assert citations[0].url == "https://nva.sikt.no/registration/abc123"
