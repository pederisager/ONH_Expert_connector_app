from __future__ import annotations

from pathlib import Path

import numpy as np

from app.index.builder import DummyEmbeddingBackend, StaffIndexBuilder
from app.index.chunking import Chunker
from app.index.models import IndexPaths, SourceLink, StaffRecord
from app.index.vector_store import LocalVectorStore


def _make_record(slug: str, summary: str) -> StaffRecord:
    return StaffRecord(
        slug=slug,
        name=f"Navn {slug}",
        title="Førsteamanuensis",
        department="Psykologi",
        summary=summary,
        sources=[SourceLink(url=f"https://example.com/{slug}")],
    )


def test_index_builder_creates_chunks_and_vectors(tmp_path: Path) -> None:
    paths = IndexPaths(root=tmp_path / "index")
    chunker = Chunker(chunk_size=5, chunk_overlap=1)
    embedder = DummyEmbeddingBackend(dimension=4)
    vector_store = LocalVectorStore(paths.vectors_dir)
    builder = StaffIndexBuilder(paths=paths, chunker=chunker, embedder=embedder, vector_store=vector_store)

    records = [
        _make_record("alpha", "Dette er en testtekst som skal chunkes i flere deler."),
        _make_record("bravo", "Kort tekst."),  # should still create a single chunk
        _make_record("empty", ""),
    ]

    summary = builder.build(records)

    assert summary.processed_staff == 2
    assert summary.total_chunks >= 2
    assert "empty" in summary.skipped_staff

    chunk_file = paths.chunks_dir / "alpha.json"
    assert chunk_file.exists()

    manifest = paths.manifests_dir / "manifest.json"
    manifest_data = manifest.read_text(encoding="utf-8")
    assert "alpha" in manifest_data

    vectors_path = paths.vectors_dir / "vectors.npy"
    assert vectors_path.exists()

    reloaded = LocalVectorStore(paths.vectors_dir)
    assert len(reloaded) == summary.total_chunks


def test_dummy_embedder_produces_deterministic_vectors() -> None:
    embedder = DummyEmbeddingBackend(dimension=3)
    texts = ["hei", "verden"]
    vectors_first = embedder.embed(texts)
    vectors_second = embedder.embed(texts)

    assert np.allclose(vectors_first, vectors_second)
    assert vectors_first.shape == (2, 3)
