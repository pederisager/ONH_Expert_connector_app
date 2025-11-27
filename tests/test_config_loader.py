from __future__ import annotations

from app.config_loader import load_app_config, load_models_config, load_staff_entries, load_staff_profiles


def test_load_app_config_has_expected_defaults() -> None:
    config = load_app_config()
    assert config.fetch.max_pages_per_staff == 2
    assert "oslonyehoyskole.no" in config.fetch.allowlist_domains
    assert config.results.max_candidates == 10
    assert config.rag.chunk_size == 400
    assert config.rag.index_root.endswith("data/index")
    assert config.security.max_upload_mb == 10


def test_load_models_config_round_trip() -> None:
    models = load_models_config()
    assert models.llm_model.name.startswith("llama3.1")
    assert models.embedding_model.backend == "sentence_transformers"
    assert models.embedding_model.device in {"auto", "cuda"}


def test_load_staff_profiles_matches_entries() -> None:
    entries = load_staff_entries()
    profiles = load_staff_profiles()
    assert len(entries) == len(profiles) > 0
    assert profiles[0].name == entries[0].name
    assert profiles[0].sources
