from __future__ import annotations

from app.config_loader import (
    load_app_config,
    load_models_config,
    load_staff_entries,
    load_staff_profiles,
)


def test_load_app_config_has_expected_defaults() -> None:
    config = load_app_config()
    assert config.fetch.max_pages_per_staff == 2
    assert "oslonyehoyskole.no" in config.fetch.allowlist_domains
    assert config.results.max_candidates == 10
    assert config.rag.chunk_size == 400
    assert config.rag.index_root.endswith("data/index")
    assert config.language.embedding_language_mode == "multilingual"
    assert config.language.translation.enabled is False
    assert config.security.max_upload_mb == 10


def test_load_models_config_round_trip() -> None:
    models = load_models_config()
    assert models.llm_model.name.startswith("llama3.1")
    assert models.llm_model.timeout == 120
    assert models.llm_model.api_key is None
    assert models.llm_model.api_key_env is None
    assert models.embedding_model.backend == "sentence_transformers"
    assert "multilingual" in models.embedding_model.name
    assert models.embedding_model.device in {"auto", "cuda"}


def test_load_staff_profiles_matches_entries() -> None:
    entries = load_staff_entries()
    profiles = load_staff_profiles()
    assert len(entries) == len(profiles) > 0
    assert profiles[0].name == entries[0].name
    assert profiles[0].sources


def test_load_models_config_supports_kebab_case_groq_fields(tmp_path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
llm_model:
  name: "llama-3.3-70b-versatile"
  backend: "groq"
  endpoint: "https://api.groq.com/openai/v1"
  timeout: 90
  api-key-env: "GROQ_API_KEY"
embedding_model:
  name: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  backend: "sentence_transformers"
  device: "cpu"
""".strip(),
        encoding="utf-8",
    )

    models = load_models_config(config_path)
    assert models.llm_model.backend == "groq"
    assert models.llm_model.timeout == 90
    assert models.llm_model.api_key_env == "GROQ_API_KEY"
