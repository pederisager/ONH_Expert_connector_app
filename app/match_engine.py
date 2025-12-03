"""Hybrid matching engine combining keyword and embedding similarity."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .nva_lookup import extract_pub_id, preferred_nva_url

try:  # Optional, lightweight stemming for better Norwegian/English grouping.
    from nltk.stem.snowball import SnowballStemmer
except Exception:  # pragma: no cover - dependency is optional
    SnowballStemmer = None  # type: ignore

# Norwegian + English stopwords and common request fillers to keep themes focused.
STOPWORDS = {
    "og",
    "i",
    "\u00e5",
    "pa",
    "for",
    "med",
    "til",
    "som",
    "\u00e5",
    "sa",
    "en",
    "et",
    "ei",
    "de",
    "der",
    "dermed",
    "derfor",
    "eller",
    "av",
    "om",
    "men",
    "mens",
    "enn",
    "ikke",
    "ingen",
    "alle",
    "har",
    "kan",
    "skal",
    "skulle",
    "kunne",
    "m\u00e5",
    "ma",
    "m\u00e5tte",
    "vil",
    "ville",
    "b\u00f8r",
    "bor",
    "burde",
    "vi",
    "oss",
    "deg",
    "dere",
    "du",
    "jeg",
    "meg",
    "han",
    "hun",
    "det",
    "den",
    "disse",
    "dette",
    "v\u00e5r",
    "v\u00e5rt",
    "v\u00e5re",
    "deres",
    "b\u00e5de",
    "baade",
    "blant",
    "blant annet",
    "via",
    "vha",
    "gir",
    "skjer",
    "utenom",
    "blir",
    "etter",
    "over",
    "under",
    "hos",
    "fra",
    "mellom",
    "uten",
    "innen",
    "innenfor",
    "behov",
    "trenger",
    "\u00f8nsker",
    "onsker",
    "ser",
    "bare",
    "hele",
    "gjerne",
    "noen",
    "noe",
    "please",
    "need",
    "needs",
    "needed",
    "needing",
    "want",
    "wants",
    "wanting",
    "looking",
    "searching",
    "seeking",
    "seek",
    "about",
    "into",
    "onto",
    "with",
    "without",
    "between",
    "during",
    "before",
    "after",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "should",
    "could",
    "would",
    "must",
    "might",
    "may",
    "s\u00f8k",
    "s\u00f8ker",
    "soker",
    "hjelp",
    "assistanse",
    "assistance",
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "of",
    "in",
    "from",
    "at",
    "by",
}

FILLER_PREFIXES = {
    "looking",
    "searching",
    "seeking",
    "need",
    "needs",
    "needing",
    "trenger",
    "\u00f8nsker",
    "onsker",
    "m\u00e5",
    "ma",
    "skal",
    "kan",
    "want",
    "wants",
    "please",
}

FILLER_SINGLE = {
    "help",
    "hjelp",
    "support",
    "st\u00f8tte",
    "stotte",
    "assistanse",
    "assistance",
}

FILLER_PATTERNS = [
    re.compile(r"^(looking|searching|seeking)\s+for\b"),
    re.compile(
        r"^(need|needs|needing|trenger|\u00f8nsker|onsker|m\u00e5|ma|skal)\s+(hjelp|help|support|st\u00f8tte|stotte|assistanse|assistance)\b"
    ),
    re.compile(r"^(kan|could|can)\s+noen\b"),
    re.compile(r"^please\b"),
]

TOKEN_PATTERN = re.compile(r"[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff][A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff'-]*", re.UNICODE)
BREAK_TOKENS = {"og", "men", "både", "baade", "samt", "pluss"}
MAX_PHRASE_TOKENS = 6
PHRASE_LENGTH_PENALTY_START = 4

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


def _normalize_token(token: str) -> str:
    normalized = unicodedata.normalize("NFKC", token).lower().strip("'-_ ")
    return normalized


@lru_cache(maxsize=4)
def _get_stemmer(lang: str):
    if SnowballStemmer is None:
        return None
    try:
        return SnowballStemmer(lang)
    except Exception:  # pragma: no cover - optional dependency may be absent
        return None


def _rule_based_stem(token: str) -> str:
    suffixes = (
        "ende",
        "ende",
        "ande",
        "ene",
        "ing",
        "ene",
        "ene",
        "ene",
        "het",
        "skap",
        "else",
        "ende",
        "ere",
        "ere",
        "ene",
        "er",
        "et",
        "en",
        "e",
        "s",
    )
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _stem_token(token: str) -> str:
    if not token:
        return token
    lang = "norwegian" if any(ch in "æøå" for ch in token) or token.endswith(
        ("ing", "ende", "ene", "het", "skap")
    ) else "english"
    stemmer = _get_stemmer(lang)
    if stemmer is not None:
        try:
            return stemmer.stem(token)
        except Exception:  # pragma: no cover - defensive
            pass
    return _rule_based_stem(token)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_PATTERN.findall(text):
        normalized = _normalize_token(raw)
        if normalized:
            tokens.append(normalized)
    return tokens


def _is_meaningful_token(token: str) -> bool:
    if not token or len(token) <= 2:
        return False
    if token in STOPWORDS:
        return False
    if token.isdigit() or any(char.isdigit() for char in token):
        return False
    return True


def _is_filler_phrase(words: Sequence[str]) -> bool:
    if not words:
        return True
    if len(words) == 1 and words[0] in FILLER_SINGLE:
        return True
    if words[0] in FILLER_PREFIXES:
        return True
    phrase = " ".join(words)
    return any(pattern.match(phrase) for pattern in FILLER_PATTERNS)


def _split_into_phrases(tokens: Sequence[str]) -> list[tuple[list[str], int]]:
    phrases: list[tuple[list[str], int]] = []
    current: list[str] = []
    start_idx = 0
    for index, token in enumerate(tokens):
        if token in BREAK_TOKENS and current:
            if not _is_filler_phrase(current):
                phrases.append((current.copy(), start_idx))
            current.clear()
            continue
        if not _is_meaningful_token(token):
            if current and not _is_filler_phrase(current):
                phrases.append((current.copy(), start_idx))
            current.clear()
            continue
        if not current:
            start_idx = index
        current.append(token)
        if len(current) >= MAX_PHRASE_TOKENS:
            if not _is_filler_phrase(current):
                phrases.append((current.copy(), start_idx))
            current.clear()
    if current and not _is_filler_phrase(current):
        phrases.append((current.copy(), start_idx))
    return phrases


def _score_phrase(words: Sequence[str], counts: Counter[str], stem_map: dict[str, str]) -> float:
    if not words:
        return 0.0
    stems = [stem_map.get(word, word) for word in words]
    frequency = float(sum(counts.get(stem, 0) for stem in stems))
    if frequency == 0:
        return 0.0
    length_bonus = 1.0 + 0.3 * max(0, len(words) - 1)
    diversity_bonus = 1.0 + 0.1 * max(0, len(set(stems)) - 1)
    length_penalty = 0.15 * max(0, len(words) - PHRASE_LENGTH_PENALTY_START)
    return frequency * (length_bonus + diversity_bonus / 2) - length_penalty


def extract_themes(text: str, top_k: int = 8) -> list[str]:
    tokens = tokenize(text)
    meaningful_tokens = [token for token in tokens if _is_meaningful_token(token)]
    if not meaningful_tokens:
        return []

    stem_map = {token: _stem_token(token) for token in meaningful_tokens}
    stemmed_tokens = [stem_map[token] for token in meaningful_tokens]
    counts = Counter(stemmed_tokens)

    stem_to_surface: dict[str, str] = {}
    for token in meaningful_tokens:
        stem = stem_map[token]
        if stem not in stem_to_surface:
            stem_to_surface[stem] = token

    scored_phrases: list[tuple[float, int, str]] = []
    seen: set[str] = set()
    for words, start_idx in _split_into_phrases(tokens):
        normalized_phrase = " ".join(words).strip()
        if not normalized_phrase or normalized_phrase in seen:
            continue
        score = _score_phrase(words, counts, stem_map)
        if score <= 0:
            continue
        scored_phrases.append((score, start_idx, normalized_phrase))
        seen.add(normalized_phrase)

    scored_phrases.sort(key=lambda item: (-item[0], item[1]))
    themes: list[str] = [phrase for _, _, phrase in scored_phrases[:top_k]]

    if len(themes) < top_k:
        for stem, _ in counts.most_common(top_k * 2):
            candidate = stem_to_surface.get(stem, stem)
            if candidate not in themes:
                themes.append(candidate)
            if len(themes) >= top_k:
                break

    return themes[:top_k]


def _sentence_split(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _hostname(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or url


def _normalize_source_url(url: str) -> str:
    """Prefer stable/public URLs for citations."""
    if not url:
        return url

    lowered = url.lower().strip()
    # Preserve DOI links.
    if "doi.org" in lowered:
        return url

    # Map NVA API endpoints to DOI/working API URLs.
    nva_prefixes = (
        "https://api.nva.unit.no/publication/",
        "https://api.test.nva.aws.unit.no/publication/",
    )
    for prefix in nva_prefixes:
        if lowered.startswith(prefix):
            pub_id = lowered.rstrip("/").split("/")[-1]
            if pub_id:
                preferred = preferred_nva_url(pub_id, url)
                if preferred:
                    return preferred
            return url

    pub_id = extract_pub_id(url)
    if pub_id:
        preferred = preferred_nva_url(pub_id, url)
        if preferred:
            return preferred

    return url


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
                        "url": _normalize_source_url(page.url),
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
                f"{profile.name} matcher {top_theme} basert pÃ¥ registrerte kilder, "
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




