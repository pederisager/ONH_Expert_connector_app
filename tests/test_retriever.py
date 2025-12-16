from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from app.index.models import Chunk
from app.index.vector_store import LocalVectorStore
from app.rag.retriever import EmbeddingRetriever, RetrievalQuery


@dataclass
class KeywordEmbedder:
    keywords: tuple[str, ...] = ("psykologi", "økonomi")

    def embed(self, texts):
        vectors = [self._vectorize(text) for text in texts]
        return (
            np.vstack(vectors)
            if vectors
            else np.empty((0, len(self.keywords)), dtype=np.float32)
        )

    def embed_one(self, text: str):
        return self._vectorize(text)

    def _vectorize(self, text: str) -> np.ndarray:
        values = [1.0 if keyword in text.lower() else 0.0 for keyword in self.keywords]
        return np.asarray(values, dtype=np.float32)


def _make_chunk(slug: str, text: str, score_index: int) -> Chunk:
    return Chunk(
        staff_slug=slug,
        chunk_id=f"{slug}-{score_index:04d}",
        text=text,
        order=score_index,
        token_count=len(text.split()),
        source_url="https://example.com",
        metadata={"department": "Psykologi", "name": f"{slug.title()}"},
    )


def test_embedding_retriever_groups_chunks(tmp_path: Path) -> None:
    store = LocalVectorStore(tmp_path / "vectors")
    embedder = KeywordEmbedder()

    chunks = [
        _make_chunk("alpha", "Psykologi forskning", 0),
        _make_chunk("beta", "Økonomi og ledelse", 0),
        _make_chunk("alpha", "Mer psykologi innhold", 1),
    ]
    embeddings = embedder.embed([chunk.text for chunk in chunks])
    store.add(embeddings, chunks)

    retriever = EmbeddingRetriever(
        vector_store=store, embedder=embedder, min_score=0.1, max_chunks_per_staff=2
    )
    results = retriever.retrieve(RetrievalQuery(text="psykologi", top_k=5))

    assert len(results) == 1
    alpha = results[0]
    assert alpha.staff_slug == "alpha"
    assert len(alpha.chunks) == 2


def test_embedding_retriever_applies_department_filter(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "vectors")
    embedder = KeywordEmbedder()
    chunks = [
        Chunk(
            staff_slug="alpha",
            chunk_id="alpha-0000",
            text="Psykologi",
            order=0,
            token_count=1,
            source_url="https://example.com",
            metadata={"department": "Psykologi"},
        ),
        Chunk(
            staff_slug="beta",
            chunk_id="beta-0000",
            text="Psykologi",
            order=0,
            token_count=1,
            source_url="https://example.com",
            metadata={"department": "Økonomi"},
        ),
    ]
    store.add(embedder.embed([c.text for c in chunks]), chunks)

    retriever = EmbeddingRetriever(vector_store=store, embedder=embedder, min_score=0.0)
    results = retriever.retrieve(
        RetrievalQuery(text="psykologi", department="Psykologi", top_k=5)
    )
    assert len(results) == 1
    assert results[0].staff_slug == "alpha"


def test_embedding_retriever_disables_on_dimension_mismatch(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "vectors")
    builder_embedder = KeywordEmbedder()
    chunks = [_make_chunk("alpha", "psykologi", 0)]
    store.add(builder_embedder.embed([chunks[0].text]), chunks)

    class LongerEmbedder(KeywordEmbedder):
        def _vectorize(self, text: str) -> np.ndarray:
            base = super()._vectorize(text)
            return np.concatenate([base, np.array([1.0], dtype=np.float32)])

    retriever = EmbeddingRetriever(vector_store=store, embedder=LongerEmbedder())
    results = retriever.retrieve(RetrievalQuery(text="psykologi", top_k=1))
    assert results == []
    assert retriever.is_active is False
    assert "dimensionality" in (retriever.disabled_reason or "")
