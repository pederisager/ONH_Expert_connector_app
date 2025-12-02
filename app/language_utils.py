"""Language utilities: detection, translation stubs, and context wiring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Protocol, Sequence

try:  # pragma: no cover - optional dependency
    import langid
except Exception:  # pragma: no cover - optional dependency
    langid = None

LOGGER = logging.getLogger("language")


class Translator(Protocol):
    """Minimal translation interface."""

    def translate(self, text: str, *, source_lang: str | None, target_lang: str) -> str:
        ...

    def translate_batch(
        self, texts: Sequence[str], *, source_lang: str | None, target_lang: str
    ) -> list[str]:
        ...

    @property
    def is_noop(self) -> bool:  # pragma: no cover - simple property
        ...


@dataclass(slots=True)
class LanguageContext:
    """Resolved languages for a request and which hops need translation."""

    user_lang: str
    query_lang: str
    embed_lang: str
    llm_lang: str
    translation_enabled: bool
    translate_for_embedding: bool
    translate_for_llm_input: bool
    translate_llm_output: bool


def detect_language(text: str, *, default: str = "no") -> str:
    """Return ISO language code for text, falling back to `default`."""

    cleaned = (text or "").strip()
    if not cleaned:
        return default
    if langid is None:
        return default
    try:
        code, _ = langid.classify(cleaned)
        if code:
            # normalize e.g. 'nb'/'nn' to 'no' for UI parity
            if code in {"nb", "nn"}:
                return "no"
            return code
    except Exception:  # pragma: no cover - defensive
        return default
    return default


def build_language_context(
    *,
    query_text: str,
    user_lang: str,
    language_config,
) -> LanguageContext:
    """Resolve per-request language routing decisions."""

    default_ui = (getattr(language_config, "default_ui", None) or "no").lower()
    requested_user_lang = (user_lang or default_ui or "no").lower()
    detection_enabled = bool(getattr(language_config, "detection_enabled", True))

    detected_query_lang = (
        detect_language(query_text, default=requested_user_lang) if detection_enabled else requested_user_lang
    )

    embedding_mode = (getattr(language_config, "embedding_language_mode", "multilingual") or "multilingual").lower()
    llm_mode = (getattr(language_config, "llm_language_mode", "match-user") or "match-user").lower()

    embed_lang = "en" if embedding_mode == "en-only" else detected_query_lang
    llm_lang = "en" if llm_mode == "en-only" else requested_user_lang

    translation_cfg = getattr(language_config, "translation", None)
    translation_enabled = bool(getattr(translation_cfg, "enabled", False))

    translate_for_embedding = translation_enabled and embedding_mode == "en-only" and detected_query_lang != "en"
    translate_for_llm_input = translation_enabled and llm_mode == "en-only" and detected_query_lang != "en"
    translate_llm_output = translation_enabled and llm_mode == "en-only" and requested_user_lang != "en"

    return LanguageContext(
        user_lang=requested_user_lang,
        query_lang=detected_query_lang,
        embed_lang=embed_lang,
        llm_lang=llm_lang,
        translation_enabled=translation_enabled,
        translate_for_embedding=translate_for_embedding,
        translate_for_llm_input=translate_for_llm_input,
        translate_llm_output=translate_llm_output,
    )


class NoOpTranslator:
    """Identity translator used when translation is disabled or unavailable."""

    @property
    def is_noop(self) -> bool:  # pragma: no cover - trivial
        return True

    def translate(self, text: str, *, source_lang: str | None, target_lang: str) -> str:
        return text

    def translate_batch(
        self, texts: Sequence[str], *, source_lang: str | None, target_lang: str
    ) -> list[str]:
        return list(texts)


class TransformersTranslator:
    """Local translation using a pre-downloaded HuggingFace seq2seq model."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("transformers is required for TransformersTranslator") from exc

        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    @property
    def is_noop(self) -> bool:  # pragma: no cover - trivial
        return False

    @lru_cache(maxsize=256)
    def translate(self, text: str, *, source_lang: str | None, target_lang: str) -> str:
        if not text:
            return ""
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        generated_tokens = self._model.generate(
            **inputs,
            max_length=self.max_length + 32,
            num_beams=4,
        )
        return self._tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def translate_batch(
        self, texts: Sequence[str], *, source_lang: str | None, target_lang: str
    ) -> list[str]:
        return [self.translate(text, source_lang=source_lang, target_lang=target_lang) for text in texts]


def _resolve_device(preferred: str | None) -> str:
    requested = (preferred or "cpu").lower()
    if requested in {"gpu", "cuda"}:
        try:  # pragma: no cover - runtime guard
            import torch  # type: ignore[import-not-found]
        except Exception:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested or "cpu"


def create_translator(config) -> Translator:
    """Instantiate translator based on TranslationConfig-like object."""

    if not getattr(config, "enabled", False):
        return NoOpTranslator()

    provider = (getattr(config, "provider", "") or "none").lower()
    model_name = getattr(config, "model_name", None) or getattr(config, "model", None)
    device = _resolve_device(getattr(config, "device", "cpu"))

    if provider in {"transformers", "local"}:
        if not model_name:
            LOGGER.warning("Translation provider '%s' enabled uten model_name; bruker no-op.", provider)
            return NoOpTranslator()
        try:
            return TransformersTranslator(model_name=model_name, device=device)
        except Exception as exc:  # pragma: no cover - optional dependency / missing weights
            LOGGER.warning("Klarte ikke A� laste oversettelsesmodell '%s': %s. Oversetter ikke.", model_name, exc)
            return NoOpTranslator()

    LOGGER.warning("Ukjent oversettelses-provider '%s'. Oversetter ikke.", provider or "none")
    return NoOpTranslator()
