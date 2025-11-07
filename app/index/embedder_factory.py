"""Factory helpers for creating embedding backends based on configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import httpx
import numpy as np

from app.config_loader import AppConfig, ModelsConfig

from .builder import DummyEmbeddingBackend
from .embeddings import EmbeddingBackend, SentenceTransformerBackend

LOGGER = logging.getLogger("rag.embedder-factory")


def create_embedding_backend(
    models_config: ModelsConfig,
    *,
    app_config: AppConfig | None = None,
) -> EmbeddingBackend:
    """Instantiate the embedding backend declared in models.yaml."""

    embedding_model = models_config.embedding_model
    backend = (embedding_model.backend or "").replace("_", "-").lower()

    if backend in {"sentence-transformers", "sentence transformers", "st"}:
        batch_size = app_config.rag.embedding_batch_size if app_config else 32
        preferred_device = (embedding_model.device or "cpu").strip() or "cpu"
        return _create_sentence_transformer_backend(
            model_name=embedding_model.name,
            batch_size=batch_size,
            device=preferred_device,
        )

    if backend == "ollama":
        if not embedding_model.endpoint:
            LOGGER.warning("Ollama-backend krever endpoint i models.yaml. Bruker dummy-embedding.")
            return DummyEmbeddingBackend()
        return OllamaEmbeddingBackend(
            model=embedding_model.name,
            endpoint=embedding_model.endpoint,
            fallback=DummyEmbeddingBackend(),
        )

    LOGGER.warning("Ukjent embedding-backend '%s'. Bruker dummy-embedding.", backend or "none")
    return DummyEmbeddingBackend()


@dataclass(slots=True)
class OllamaEmbeddingBackend(EmbeddingBackend):
    """Calls the local Ollama embeddings API."""

    model: str
    endpoint: str
    timeout: float = 60.0
    _base_url: str = field(init=False, repr=False)
    _dimension: int | None = field(init=False, default=None, repr=False)
    fallback: EmbeddingBackend | None = field(default=None, repr=False)
    _fallback_warned: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._base_url = self.endpoint.rstrip("/")

    def embed(self, texts: Sequence[str]):
        try:
            return self._run_embedding(texts)
        except (httpx.HTTPError, OSError) as exc:  # pragma: no cover - network guard
            return self._fallback_or_raise("embed", texts, exc)

    def embed_one(self, text: str):
        try:
            vectors = self._run_embedding([text])
            return vectors[0]
        except (httpx.HTTPError, OSError) as exc:  # pragma: no cover - network guard
            return self._fallback_or_raise("embed_one", text, exc)

    def _run_embedding(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            if self._dimension is None:
                return np.empty((0, 0), dtype=np.float32)
            return np.empty((0, self._dimension), dtype=np.float32)

        vectors = []
        with httpx.Client(timeout=self.timeout) as client:
            for text in texts:
                vectors.append(self._embed_single(client, text))
        return np.vstack(vectors)

    def _embed_single(self, client: httpx.Client, text: str) -> np.ndarray:
        payload = {
            "model": self.model,
            "prompt": text,
        }
        response = client.post(f"{self._base_url}/api/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()
        embedding = np.asarray(data.get("embedding") or [], dtype=np.float32)
        if embedding.ndim != 1:
            raise ValueError("Ollama embeddings-API returnerte uventet format.")
        if self._dimension is None:
            self._dimension = embedding.shape[0]
        elif embedding.shape[0] != self._dimension:
            raise ValueError("Ollama embeddings endret dimensjon mellom kall.")
        return embedding

    def _fallback_or_raise(
        self,
        method: str,
        payload,
        exc: Exception,
    ):
        if not self.fallback:
            raise
        if not self._fallback_warned:
            LOGGER.warning(
                "Ollama-endepunkt %s utilgjengelig (%s). Faller tilbake til %s.",
                self.endpoint,
                exc,
                self.fallback.__class__.__name__,
            )
            self._fallback_warned = True
        if method == "embed":
            return self.fallback.embed(payload)
        if method == "embed_one":
            return self.fallback.embed_one(payload)
        raise RuntimeError("Ukjent embedding-metode for fallback: %s" % method)


def _create_sentence_transformer_backend(
    *,
    model_name: str,
    batch_size: int,
    device: str,
) -> EmbeddingBackend:
    """Instantiate a SentenceTransformer backend with GPU-aware fallbacks."""

    normalized_device = device.lower()
    try:
        return SentenceTransformerBackend(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
        )
    except ImportError as exc:  # pragma: no cover - missing dependency
        LOGGER.warning(
            "sentence-transformers er ikke installert (%s). Bruker dummy-embedding.",
            exc,
        )
        return DummyEmbeddingBackend()
    except RuntimeError as exc:  # pragma: no cover - CUDA/device errors
        if "cuda" in normalized_device or "gpu" in normalized_device:
            LOGGER.warning(
                "CUDA-enhet '%s' utilgjengelig (%s). Prøver igjen på CPU.",
                device,
                exc,
            )
            return _create_sentence_transformer_backend(
                model_name=model_name,
                batch_size=batch_size,
                device="cpu",
            )
        LOGGER.warning(
            "sentence-transformers feilet (%s). Bruker dummy-embedding.",
            exc,
        )
        return DummyEmbeddingBackend()
