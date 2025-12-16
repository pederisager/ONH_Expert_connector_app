"""Embedding backends used by the index builder and retriever."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

LOGGER = logging.getLogger("rag.embeddings")


class EmbeddingBackend(Protocol):
    """Protocol for embedding providers."""

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_one(self, text: str) -> np.ndarray: ...


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length for cosine similarity."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


@dataclass(slots=True)
class SentenceTransformerBackend:
    """Thin wrapper over `sentence-transformers` with graceful CPU fallback."""

    model_name: str
    batch_size: int = 32
    device: str | None = None
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._load_model(self.device)

    def _load_model(self, device: str | None) -> None:
        sentence_transformer_cls = self._resolve_class()
        if sentence_transformer_cls is None:  # pragma: no cover - runtime guard
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerBackend."
            )
        self.device = device
        self._model = sentence_transformer_cls(self.model_name, device=device)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty(
                (0, self._model.get_sentence_embedding_dimension()), dtype=np.float32
            )
        try:
            embeddings = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except RuntimeError as exc:  # pragma: no cover - runtime-only guard
            message = str(exc).lower()
            if (
                self.device
                and self.device.lower() != "cpu"
                and (
                    "no kernel image" in message
                    or "device-side assert triggered" in message
                    or "is not compatible with gpu" in message
                    or "sm_" in message
                )
            ):
                LOGGER.warning(
                    "sentence-transformers on device '%s' failed (%s). Falling back to CPU.",
                    self.device,
                    exc,
                )
                self._load_model("cpu")
                embeddings = self._model.encode(
                    list(texts),
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
            else:
                raise

        return embeddings.astype("float32")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @staticmethod
    def _resolve_class():
        spec = importlib.util.find_spec("sentence_transformers")  # type: ignore[attr-defined]
        if spec is None:
            return None
        module = importlib.import_module("sentence_transformers")
        return getattr(module, "SentenceTransformer", None)
