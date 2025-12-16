from __future__ import annotations

from types import SimpleNamespace

from app.index import embedder_factory
from app.index.builder import DummyEmbeddingBackend


def _models_config(backend: str, device: str | None = None):
    embedding_model = SimpleNamespace(
        name="test-model",
        backend=backend,
        endpoint=None,
        device=device,
    )
    return SimpleNamespace(embedding_model=embedding_model)


def _app_config(batch_size: int = 16):
    return SimpleNamespace(rag=SimpleNamespace(embedding_batch_size=batch_size))


def test_sentence_transformer_cuda_falls_back_to_cpu(monkeypatch):
    class StubBackend:
        def __init__(self, *, model_name, batch_size, device):  # type: ignore[override]
            self.model_name = model_name
            self.batch_size = batch_size
            self.device = device
            if device == "cuda":
                raise RuntimeError("CUDA init failed")

    monkeypatch.setattr(embedder_factory, "SentenceTransformerBackend", StubBackend)
    monkeypatch.setattr(
        embedder_factory, "_probe_torch_capabilities", lambda: (True, False)
    )

    backend = embedder_factory.create_embedding_backend(
        _models_config(backend="sentence_transformers", device="cuda"),
        app_config=_app_config(batch_size=8),
    )

    assert isinstance(backend, StubBackend)
    assert backend.device == "cpu"
    assert backend.batch_size == 8


def test_sentence_transformer_missing_dependency_returns_dummy(monkeypatch):
    def raising_backend(*args, **kwargs):
        raise ImportError("missing sentence-transformers")

    monkeypatch.setattr(embedder_factory, "SentenceTransformerBackend", raising_backend)
    monkeypatch.setattr(
        embedder_factory, "_probe_torch_capabilities", lambda: (False, False)
    )

    backend = embedder_factory.create_embedding_backend(
        _models_config(backend="sentence_transformers", device="cpu"),
        app_config=_app_config(),
    )

    assert isinstance(backend, DummyEmbeddingBackend)


def test_sentence_transformer_auto_prefers_cuda_when_available(monkeypatch):
    class StubBackend:
        def __init__(self, *, model_name, batch_size, device):  # type: ignore[override]
            self.device = device

    monkeypatch.setattr(embedder_factory, "SentenceTransformerBackend", StubBackend)
    monkeypatch.setattr(
        embedder_factory, "_probe_torch_capabilities", lambda: (True, False)
    )

    backend = embedder_factory.create_embedding_backend(
        _models_config(backend="sentence_transformers", device=None),
        app_config=_app_config(batch_size=12),
    )

    assert isinstance(backend, StubBackend)
    assert backend.device == "cuda"


def test_sentence_transformer_auto_falls_back_to_cpu_without_accelerator(monkeypatch):
    class StubBackend:
        def __init__(self, *, model_name, batch_size, device):  # type: ignore[override]
            self.device = device

    monkeypatch.setattr(embedder_factory, "SentenceTransformerBackend", StubBackend)
    monkeypatch.setattr(
        embedder_factory, "_probe_torch_capabilities", lambda: (False, False)
    )

    backend = embedder_factory.create_embedding_backend(
        _models_config(backend="sentence_transformers", device=None),
        app_config=_app_config(batch_size=12),
    )

    assert isinstance(backend, StubBackend)
    assert backend.device == "cpu"
