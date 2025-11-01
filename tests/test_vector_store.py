from __future__ import annotations

import numpy as np
import pytest

from app.index.models import Chunk
from app.index.vector_store import LocalVectorStore


def _make_chunk(slug: str, text: str, idx: int) -> Chunk:
    return Chunk(
        staff_slug=slug,
        chunk_id=f"{slug}-{idx:04d}",
        text=text,
        order=idx,
        token_count=len(text.split()),
        source_url="https://example.com",
        metadata={"department": "Test"},
    )


def test_vector_store_add_and_search(tmp_path) -> None:
    store = LocalVectorStore(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    chunks = [_make_chunk("a", "first text", 0), _make_chunk("b", "second text", 1)]
    store.add(embeddings, chunks)

    query = np.array([1.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=2, min_score=0.1)

    assert len(results) == 1
    assert results[0].chunk.staff_slug == "a"
    assert pytest.approx(results[0].score, rel=1e-3) == 1.0


def test_vector_store_persist_and_reload(tmp_path) -> None:
    store = LocalVectorStore(tmp_path)
    embeddings = np.array([[0.6, 0.8]], dtype=np.float32)
    chunk = _make_chunk("persist", "some text", 0)
    store.add(embeddings, [chunk])
    store.persist()

    # Recreate store to ensure data is loaded from disk
    reloaded = LocalVectorStore(tmp_path)
    assert len(reloaded) == 1
    results = reloaded.search(np.array([0.6, 0.8], dtype=np.float32), top_k=1)
    assert results[0].chunk.chunk_id == chunk.chunk_id


def test_vector_store_validates_dimensions(tmp_path) -> None:
    store = LocalVectorStore(tmp_path)
    embeddings = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    chunk = _make_chunk("dim", "text", 0)
    store.add(embeddings, [chunk])

    with pytest.raises(ValueError):
        store.search(np.array([1.0, 2.0], dtype=np.float32))
