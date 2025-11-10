"""Hybrid matching engine combining keyword and embedding similarity."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Basic Norwegian + English stopwords to keep theme extraction focused.
STOPWORDS = {
    "og",
    "i",
    "på",
    "for",
    "med",
    "til",
    "som",
    "en",
    "et",
    "de",
    "der",
    "eller",
    "av",
    "om",
    "the",
    "a",
    "an",
    "is",
    "are",
    "this",
    "that",
    "of",
    "in",
    "from",
    "at",
    "by",
    "også",
    "men",
    "enn",
    "ikke",
    "har",
    "kan",
    "skal",
}

TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StaffProfile:
    name: str
    department: str
    profile_url: str
    sources: list[str]
    tags: list[str]


@dataclass(slots=True)
class PageContent:
    url: str
    title: str
    text: str


MAX_COMBINED_TEXT_CHARS = 6000


@dataclass(slots=True)
class StaffDocument:
    profile: StaffProfile
    pages: list[PageContent]
    _combined_cache: str | None = field(init=False, default=None, repr=False)

    @property
    def combined_text(self) -> str:
        if self._combined_cache is None:
            combined = "\n".join(page.text for page in self.pages if page.text)
            if len(combined) > MAX_COMBINED_TEXT_CHARS:
                combined = combined[:MAX_COMBINED_TEXT_CHARS]
            self._combined_cache = combined
        return self._combined_cache


@dataclass(slots=True)
class MatchResult:
    staff: StaffProfile
    score: float
    explanation: str
    citations: list[dict[str, str]]
    score_breakdown: dict[str, float]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def extract_themes(text: str, top_k: int = 8) -> list[str]:
    tokens = [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2]
    counts = Counter(tokens)
    ordered = [word for word, _ in counts.most_common(top_k * 2)]
    unique: list[str] = []
    for word in ordered:
        if word not in unique:
            unique.append(word)
        if len(unique) >= top_k:
            break
    return unique


def _sentence_split(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _hostname(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or url


class MatchEngine:
    """Hybrid scoring across TF-IDF semantics, keyword overlap, and tags."""

    def __init__(
        self,
        *,
        embedding_model_name: str | None = None,
        embedding_backend: str | None = None,
        embedding_device: str = "auto",
        embedding_endpoint: str | None = None,
    ) -> None:
        normalized_backend = (embedding_backend or "").replace("_", "-").lower()
        if normalized_backend in {"sentence-transformers", "sentence transformers", "st"}:
            normalized_backend = "sentence-transformers"
        elif not normalized_backend:
            normalized_backend = None

        if not embedding_model_name:
            normalized_backend = None

        requested_device = (embedding_device or "auto").lower()
        if requested_device == "gpu":
            requested_device = "cuda"

        self.embedding_model_name = embedding_model_name
        self.embedding_backend = normalized_backend
        self.embedding_device_requested = requested_device or "auto"
        self.embedding_endpoint = embedding_endpoint.rstrip("/") if embedding_endpoint else None

        self._embedder: Any | None = None
        self._embedder_attempted = False
        self._embedder_lock = Lock()
        self._embedding_device_resolved: str | None = None
        self._embedding_warning_logged = False
        self._last_similarity_backend = "tfidf"

    def _log_embedding_warning(self, message: str, *args: Any) -> None:
        if not self._embedding_warning_logged:
            logger.warning("MatchEngine: " + message, *args)
            self._embedding_warning_logged = True

    def _resolve_device(self, requested: str) -> str:
        try:
            import torch  # type: ignore[import-not-found]
        except Exception:
            return "cpu"

        has_mps_backend = getattr(torch.backends, "mps", None)

        if requested in {"cuda", "gpu"}:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "mps":
            return "mps" if has_mps_backend and torch.backends.mps.is_available() else "cpu"
        if requested in {"auto", "", "cpu"}:
            if torch.cuda.is_available():
                return "cuda"
            if has_mps_backend and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return requested

    def _get_sentence_transformer(self):
        if self.embedding_backend != "sentence-transformers":
            return None
        if not self.embedding_model_name:
            return None
        if self._embedder_attempted:
            return self._embedder

        with self._embedder_lock:
            if self._embedder_attempted:
                return self._embedder
            self._embedder_attempted = True

            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover - import guard
                self._log_embedding_warning("sentence-transformers unavailable (%s)", exc)
                self._embedder = None
                return None

            resolved = self._resolve_device(self.embedding_device_requested)
            try:
                self._embedder = SentenceTransformer(self.embedding_model_name, device=resolved)
                self._embedding_device_resolved = resolved
            except Exception as exc:  # pragma: no cover - runtime guard
                self._log_embedding_warning(
                    "failed to load embedding model '%s' on %s (%s)",
                    self.embedding_model_name,
                    resolved,
                    exc,
                )
                self._embedder = None
            return self._embedder

    def _embedding_similarity_via_sentence_transformers(
        self, docs: list[StaffDocument], theme_text: str
    ) -> np.ndarray | None:
        embedder = self._get_sentence_transformer()
        if embedder is None:
            return None

        try:
            from sentence_transformers import util  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import guard
            self._log_embedding_warning("sentence-transformers util unavailable (%s)", exc)
            return None

        try:
            theme_vec = embedder.encode(
                theme_text,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            doc_vecs = embedder.encode(
                [doc.combined_text for doc in docs],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            similarities = util.cos_sim(doc_vecs, theme_vec).cpu().numpy().reshape(-1)
            self._last_similarity_backend = "sentence-transformers"
            return similarities
        except Exception as exc:  # pragma: no cover - runtime guard
            self._log_embedding_warning("sentence-transformers inference failed (%s)", exc)
            return None

    def _embedding_similarity_via_ollama(
        self, docs: list[StaffDocument], theme_text: str
    ) -> np.ndarray | None:
        if not self.embedding_endpoint or not self.embedding_model_name:
            return None

        base_url = self.embedding_endpoint
        texts = [theme_text] + [doc.combined_text for doc in docs]
        vectors: list[np.ndarray] = []

        try:
            with httpx.Client(timeout=60.0) as client:
                for text in texts:
                    response = client.post(
                        f"{base_url}/api/embeddings",
                        json={"model": self.embedding_model_name, "prompt": text},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    embedding = payload.get("embedding")
                    if not embedding:
                        raise ValueError("Missing embedding from Ollama response.")
                    vectors.append(np.asarray(embedding, dtype=float))
        except Exception as exc:
            self._log_embedding_warning("Ollama embeddings unavailable (%s)", exc)
            return None

        if not vectors:
            return None

        theme_vec, *doc_vecs = vectors
        if not doc_vecs:
            return None

        doc_matrix = np.vstack(doc_vecs)
        theme_norm = float(np.linalg.norm(theme_vec)) or 1.0
        doc_norms = np.linalg.norm(doc_matrix, axis=1)
        doc_norms = np.where(doc_norms == 0.0, 1.0, doc_norms)
        similarities = (doc_matrix @ theme_vec) / (doc_norms * theme_norm)
        similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
        self._last_similarity_backend = "ollama"
        return similarities

    def _embedding_similarity(self, docs: list[StaffDocument], theme_text: str) -> np.ndarray | None:
        if self.embedding_backend == "sentence-transformers":
            return self._embedding_similarity_via_sentence_transformers(docs, theme_text)
        if self.embedding_backend == "ollama":
            return self._embedding_similarity_via_ollama(docs, theme_text)
        return None

    def _tfidf_similarity(self, theme_text: str, docs: list[StaffDocument]) -> np.ndarray:
        corpus = [theme_text] + [doc.combined_text for doc in docs]
        vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            self._last_similarity_backend = "tfidf"
            return np.zeros(len(docs))
        theme_vector = tfidf_matrix[0]
        doc_vectors = tfidf_matrix[1:]
        similarities = cosine_similarity(doc_vectors, theme_vector).reshape(-1)
        self._last_similarity_backend = "tfidf"
        return similarities

    def rank(
        self,
        documents: Iterable[StaffDocument],
        themes: Sequence[str],
        department_filter: str | None,
        max_candidates: int = 10,
        diversity_weight: float = 0.1,
    ) -> list[MatchResult]:
        docs = [doc for doc in documents if doc.combined_text.strip()]
        if not docs or not themes:
            return []

        theme_text = " ".join(themes)
        semantic_scores = self._embedding_similarity(docs, theme_text)
        if semantic_scores is None:
            semantic_scores = self._tfidf_similarity(theme_text, docs)
        if semantic_scores is None or len(semantic_scores) == 0:
            return []

        theme_tokens = set(themes)
        results: list[MatchResult] = []

        for index, doc in enumerate(docs):
            combined_text = doc.combined_text
            tokens = tokenize(combined_text)
            if not tokens:
                continue

            token_counts = Counter(tokens)
            keyword_hits = sum(token_counts.get(theme, 0) for theme in theme_tokens)
            keyword_score = min(1.0, keyword_hits / max(1, len(tokens) / 20))

            tags_lower = {tag.lower() for tag in doc.profile.tags}
            tag_overlap = len(tags_lower & theme_tokens)
            tag_score = tag_overlap / max(1, len(tags_lower)) if tags_lower else 0.0

            department_bonus = 0.1 if department_filter and doc.profile.department == department_filter else 0.0

            semantic_score = float(semantic_scores[index])

            raw_score = (
                0.55 * semantic_score
                + 0.25 * keyword_score
                + 0.15 * tag_score
                + department_bonus
            )
            score = max(0.0, min(1.0, raw_score))

            citations = self._build_citations(doc, theme_tokens)
            explanation = self._build_explanation(doc.profile, citations, themes)

            results.append(
                MatchResult(
                    staff=doc.profile,
                    score=round(score * 100, 2),
                    explanation=explanation,
                    citations=citations,
                    score_breakdown={
                        "semantic": round(semantic_score, 4),
                        "keywords": round(keyword_score, 4),
                        "tags": round(tag_score, 4),
                        "department_bonus": round(department_bonus, 4),
                    },
                )
            )

        results.sort(key=lambda result: result.score, reverse=True)
        results = results[:max_candidates]

        # Diversity nudge: reduce score slightly for duplicate departments after the first.
        if diversity_weight > 0:
            seen_departments: dict[str, int] = {}
            for result in results:
                dept = result.staff.department
                seen_departments[dept] = seen_departments.get(dept, 0) + 1
                penalty = (seen_departments[dept] - 1) * diversity_weight * 5
                result.score = max(0.0, round(result.score - penalty, 2))

        return results

    @property
    def similarity_backend(self) -> str:
        """Return the most recent semantic similarity backend used."""
        return self._last_similarity_backend

    def _build_citations(self, doc: StaffDocument, themes: set[str]) -> list[dict[str, str]]:
        citations: list[dict[str, str]] = []
        citation_id = 1
        for page in doc.pages:
            sentences = _sentence_split(page.text)
            for sentence in sentences:
                lowered = sentence.lower()
                if themes and not any(theme in lowered for theme in themes):
                    continue
                citations.append(
                    {
                        "id": f"[{citation_id}]",
                        "title": page.title or _hostname(page.url),
                        "url": page.url,
                        "snippet": sentence[:300].rstrip(),
                    }
                )
                citation_id += 1
                if citation_id > 3:
                    break
            if citation_id > 3:
                break
        return citations

    def _build_explanation(
        self, profile: StaffProfile, citations: Sequence[dict[str, str]], themes: Sequence[str]
    ) -> str:
        if not citations:
            top_theme = themes[0] if themes else "temaet"
            return (
                f"{profile.name} matcher {top_theme} basert på registrerte kilder, "
                "men ingen konkrete sitater ble hentet."
            )

        first = citations[0]
        top_theme = themes[0] if themes else "temaet"
        explanation_parts = [
            f"{profile.name} omtaler {top_theme} i {first['title']} ({first['url']})."
        ]
        if len(citations) > 1:
            second = citations[1]
            explanation_parts.append(
                f"Ytterligere dokumentasjon finnes i {second['title']} ({second['url']})."
            )
        explanation_parts.append("Se kildeutdragene for konkrete formuleringer.")
        return " ".join(explanation_parts)
