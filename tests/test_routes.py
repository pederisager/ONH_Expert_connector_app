from __future__ import annotations

from pathlib import Path

import pytest

from app.index.builder import DummyEmbeddingBackend
from app.index.models import Chunk
from app.index.vector_store import LocalVectorStore
from app.rag.retriever import EmbeddingRetriever


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
async def test_shortlist_persistence_and_export(client, tmp_path) -> None:
    shortlist_payload = {
        "items": [
            {
                "id": "abc123",
                "name": "Test Forsker",
                "department": "Psykologi",
                "why": "Matcher psykologi.",
                "citations": [
                    {
                        "id": "[1]",
                        "title": "Stub",
                        "url": "https://example.com",
                        "snippet": "Forskning på psykologi.",
                    }
                ],
                "notes": "Prioritert kandidat",
            }
        ]
    }

    save_response = await client.post("/shortlist", json=shortlist_payload)
    assert save_response.status_code == 200

    fetch_response = await client.get("/shortlist")
    assert fetch_response.status_code == 200
    assert fetch_response.json()["items"][0]["name"] == "Test Forsker"

    export_response = await client.post(
        "/shortlist/export",
        json={"format": "pdf", "metadata": {"topic": "Psykologi"}},
    )
    assert export_response.status_code == 200
    path_str = export_response.json()["path"]
    exporter_dir = client.app.state.shortlist_exporter.base_dir  # type: ignore[attr-defined]
    export_path = (Path(exporter_dir.parent.parent) / path_str).resolve()
    assert export_path.exists()


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
