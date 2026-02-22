"""Utilities for loading project configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

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
    allowlist_domains: list[str] = Field(
        alias="allowlist-domains", default_factory=list
    )
    offline_snapshots_dir: str | None = Field(
        alias="offline-snapshots-dir", default=None
    )


class CacheConfig(BaseModel):
    enabled: bool = True
    retention_days: int = Field(alias="retention-days", default=14)
    directory: str = "data/cache"


class ResultsConfig(BaseModel):
    class ResultsScoringWeights(BaseModel):
        semantic: float = Field(default=1.0, ge=0.0)
        keywords: float = Field(default=0.1, ge=0.0)
        tags: float = Field(default=0.15, ge=0.0)
        methods: float = Field(default=0.15, ge=0.0)

    class ResultsModeScoringProfile(BaseModel):
        source_kind_boosts: dict[str, float] = Field(
            alias="source-kind-boosts",
            default_factory=dict,
        )
        citation_overlap_weight: float = Field(
            alias="citation-overlap-weight",
            default=0.0,
            ge=0.0,
        )
        nva_citation_bonus: float = Field(
            alias="nva-citation-bonus",
            default=0.0,
            ge=0.0,
        )

    class ResultsOverexposurePenalty(BaseModel):
        enabled: bool = Field(default=True)
        low_signal_threshold: float = Field(
            alias="low-signal-threshold",
            default=0.35,
            ge=0.0,
            le=1.0,
        )
        profile_source_weight: float = Field(
            alias="profile-source-weight",
            default=0.2,
            ge=0.0,
        )
        staffinfo_source_weight: float = Field(
            alias="staffinfo-source-weight",
            default=0.12,
            ge=0.0,
        )
        publication_without_nva_penalty: float = Field(
            alias="publication-without-nva-penalty",
            default=0.04,
            ge=0.0,
        )
        max_penalty: float = Field(
            alias="max-penalty",
            default=0.1,
            ge=0.0,
        )

    class ResultsCategoryIntentPenalty(BaseModel):
        enabled: bool = Field(default=True)
        base_penalty: float = Field(alias="base-penalty", default=0.08, ge=0.0)
        max_penalty: float = Field(alias="max-penalty", default=0.16, ge=0.0)
        evidence_overlap_threshold: float = Field(
            alias="evidence-overlap-threshold",
            default=0.2,
            ge=0.0,
            le=1.0,
        )
        intent_signals: dict[str, list[str]] = Field(
            alias="intent-signals",
            default_factory=dict,
        )
        intent_department_map: dict[str, list[str]] = Field(
            alias="intent-department-map",
            default_factory=dict,
        )

    class ResultsExactKeywordPromotion(BaseModel):
        enabled: bool = Field(default=True)
        keyword_bonus: float = Field(alias="keyword-bonus", default=0.35, ge=0.0)
        min_token_length: int = Field(alias="min-token-length", default=2, ge=1)

    max_candidates: int = Field(alias="max-candidates", default=10)
    min_similarity_score: float = Field(alias="min-similarity-score", default=0.25)
    diversity_weight: float = Field(alias="diversity-weight", default=0.1)
    citation_source_priority: list[str] = Field(
        alias="citation-source-priority",
        default_factory=lambda: ["nva", "profile", "staffinfo"],
    )
    min_query_overlap_per_citation: int = Field(
        alias="min-query-overlap-per-citation",
        default=1,
    )
    profile_staffinfo_min_query_overlap_per_citation: int = Field(
        alias="profile-staffinfo-min-query-overlap-per-citation",
        default=1,
        ge=0,
    )
    scoring_weights: ResultsScoringWeights = Field(
        alias="scoring-weights",
        default_factory=ResultsScoringWeights,
    )
    mode_scoring_profiles: dict[str, ResultsModeScoringProfile] = Field(
        alias="mode-scoring-profiles",
        default_factory=lambda: {
            "publication_grounded": {
                "source-kind-boosts": {"nva": 0.18, "profile": 0.03, "staffinfo": 0.0},
                "citation-overlap-weight": 0.12,
                "nva-citation-bonus": 0.08,
            },
            "profile_grounded": {
                "source-kind-boosts": {"nva": 0.03, "profile": 0.14, "staffinfo": 0.09},
                "citation-overlap-weight": 0.06,
                "nva-citation-bonus": 0.0,
            },
        },
    )
    exact_keyword_promotion: ResultsExactKeywordPromotion = Field(
        alias="exact-keyword-promotion",
        default_factory=ResultsExactKeywordPromotion,
    )
    concept_keyword_map: dict[str, list[str]] = Field(
        alias="concept-keyword-map",
        default_factory=dict,
    )
    synonym_expansion_map: dict[str, list[str]] = Field(
        alias="synonym-expansion-map",
        default_factory=dict,
    )
    overexposure_penalty: ResultsOverexposurePenalty = Field(
        alias="overexposure-penalty",
        default_factory=ResultsOverexposurePenalty,
    )
    category_intent_penalty: ResultsCategoryIntentPenalty = Field(
        alias="category-intent-penalty",
        default_factory=ResultsCategoryIntentPenalty,
    )


class RagConfig(BaseModel):
    index_root: str = Field(alias="index-root", default="data/index")
    chunk_size: int = Field(alias="chunk-size", default=400)
    chunk_overlap: int = Field(alias="chunk-overlap", default=60)
    max_chunks_per_profile: int = Field(alias="max-chunks-per-profile", default=40)
    hybrid_semantic_weight: float = Field(
        alias="hybrid-semantic-weight",
        default=0.85,
        ge=0.0,
    )
    hybrid_lexical_weight: float = Field(
        alias="hybrid-lexical-weight",
        default=0.15,
        ge=0.0,
    )
    source_weights: dict[str, float] = Field(
        alias="source-weights",
        default_factory=lambda: {"nva": 1.0, "profile": 0.8, "staffinfo": 0.45},
    )
    max_chunks_per_source_per_staff: dict[str, int] = Field(
        alias="max-chunks-per-source-per-staff",
        default_factory=lambda: {"nva": 3, "profile": 1, "staffinfo": 1},
    )
    embedding_model: str = Field(
        alias="embedding-model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
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
    embedding_language_mode: str = Field(
        alias="embedding-language-mode", default="multilingual"
    )
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
    model_config = ConfigDict(populate_by_name=True)

    name: str
    backend: str
    endpoint: str | None = None
    purpose: str | None = None
    device: str | None = None
    timeout: float = 120.0
    api_key: str | None = Field(alias="api-key", default=None)
    api_key_env: str | None = Field(alias="api-key-env", default=None)


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
