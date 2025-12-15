"""Utilities for loading project configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .match_engine import StaffProfile


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class StaffEntry(BaseModel):
    name: str
    department: str
    profile_url: str = Field(alias="profile_url")
    sources: list[str]
    tags: list[str] = Field(default_factory=list)


class FetchConfig(BaseModel):
    max_pages_per_staff: int = Field(alias="max-pages-per-staff", default=2)
    max_kb_per_page: int = Field(alias="max-kb-per-page", default=100)
    allowlist_domains: list[str] = Field(alias="allowlist-domains", default_factory=list)
    offline_snapshots_dir: str | None = Field(alias="offline-snapshots-dir", default=None)


class CacheConfig(BaseModel):
    enabled: bool = True
    retention_days: int = Field(alias="retention-days", default=14)
    directory: str = "data/cache"


class ResultsConfig(BaseModel):
    max_candidates: int = Field(alias="max-candidates", default=10)
    min_similarity_score: float = Field(alias="min-similarity-score", default=0.25)
    diversity_weight: float = Field(alias="diversity-weight", default=0.1)


class RagConfig(BaseModel):
    index_root: str = Field(alias="index-root", default="data/index")
    chunk_size: int = Field(alias="chunk-size", default=400)
    chunk_overlap: int = Field(alias="chunk-overlap", default=60)
    max_chunks_per_profile: int = Field(alias="max-chunks-per-profile", default=40)
    embedding_model: str = Field(
        alias="embedding-model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    embedding_batch_size: int = Field(alias="embedding-batch-size", default=32)


class TranslationConfig(BaseModel):
    enabled: bool = False
    provider: str = "none"
    model_name: str | None = Field(alias="model-name", default=None)
    endpoint: str | None = None
    device: str | None = None
    timeout_seconds: float = Field(alias="timeout-seconds", default=20.0)
    cache_size: int = Field(alias="cache-size", default=256)
    max_chars: int = Field(alias="max-chars", default=4000)


class LanguageConfig(BaseModel):
    default_ui: str = Field(alias="default-ui", default="no")
    detection_enabled: bool = Field(alias="detection-enabled", default=True)
    embedding_language_mode: str = Field(alias="embedding-language-mode", default="multilingual")
    llm_language_mode: str = Field(alias="llm-language-mode", default="match-user")
    translation: TranslationConfig = Field(default_factory=TranslationConfig)


class UIConfig(BaseModel):
    allow_department_filter: bool = Field(alias="allow-department-filter", default=True)
    language: str = "no"
    export_formats: list[str] = Field(alias="export-formats", default_factory=list)


class SecurityConfig(BaseModel):
    allow_file_types: list[str] = Field(alias="allow-file-types", default_factory=list)
    max_upload_mb: int = Field(alias="max-upload-mb", default=10)


class AppConfig(BaseModel):
    fetch: FetchConfig
    cache: CacheConfig
    results: ResultsConfig
    rag: RagConfig
    ui: UIConfig
    language: LanguageConfig = Field(default_factory=LanguageConfig)
    security: SecurityConfig


class ModelConfig(BaseModel):
    name: str
    backend: str
    endpoint: str | None = None
    purpose: str | None = None
    device: str | None = None


class ModelsConfig(BaseModel):
    llm_model: ModelConfig = Field(alias="llm_model")
    embedding_model: ModelConfig = Field(alias="embedding_model")


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_staff_entries(path: Path | None = None) -> list[StaffEntry]:
    """Return staff entries from `staff.yaml` as validated models."""
    target = path or DATA_DIR / "staff.yaml"
    data = _load_yaml(target)
    if not isinstance(data, list):
        raise ValueError("staff.yaml must contain a list of staff records.")
    return [StaffEntry.model_validate(item) for item in data]


def load_staff_profiles(path: Path | None = None) -> list[StaffProfile]:
    return [
        StaffProfile(
            name=entry.name,
            department=entry.department,
            profile_url=entry.profile_url,
            sources=entry.sources,
            tags=entry.tags,
        )
        for entry in load_staff_entries(path)
    ]


def load_app_config(path: Path | None = None) -> AppConfig:
    """Return application config from `app.config.yaml`."""
    target = path or DATA_DIR / "app.config.yaml"
    return AppConfig.model_validate(_load_yaml(target))


def load_models_config(path: Path | None = None) -> ModelsConfig:
    """Return model selection config from `models.yaml`."""
    target = path or DATA_DIR / "models.yaml"
    return ModelsConfig.model_validate(_load_yaml(target))
