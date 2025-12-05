"""LLM-backed explanation generator with optional translation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence

import httpx

logger = logging.getLogger(__name__)


class LLMExplainer:
    """Thin wrapper around the configured LLM backend."""

    _REFUSAL_MARKERS = (
        "i'm sorry",
        "i am sorry",
        "i cannot",
        "i can't",
        "cannot comply",
        "cannot assist",
        "not able to",
        "as an ai",
        "unable to provide",
    )

    def __init__(self, model_config: Dict[str, Any]) -> None:
        self.model_config = model_config
        self.backend = model_config.get("backend", "ollama")
        self.model_name = model_config.get("name")
        self.endpoint = model_config.get("endpoint", "http://localhost:11434")
        self.timeout = float(model_config.get("timeout", 120.0))

    async def generate(
        self,
        staff_name: str,
        snippets: Sequence[str],
        themes: Sequence[str],
        *,
        prompt_lang: str = "no",
        output_lang: str | None = None,
        translator=None,
    ) -> str:
        normalized_prompt_lang = (prompt_lang or "no").lower()
        output_lang = (output_lang or normalized_prompt_lang).lower()

        def _build_safe_prompt(local_snippets: Sequence[str]) -> str:
            return self._build_prompt(
                staff_name,
                local_snippets,
                themes,
                language=normalized_prompt_lang,
            )

        fallback = self._fallback_explanation(
            staff_name,
            snippets,
            themes,
            language=output_lang,
            translator=translator,
            source_lang=normalized_prompt_lang,
        )

        # Two attempts: full evidence then minimal evidence, to reduce failures from large payloads.
        attempts = [
            ("full", snippets),
            ("minimal", snippets[:1]),
        ]

        for attempt_label, attempt_snippets in attempts:
            prompt = _build_safe_prompt(attempt_snippets)
            if not prompt:
                logger.debug("LLM prompt skipped (no model configured) for %s", staff_name)
                return fallback

            if self.backend == "ollama" and self.model_name:
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(
                            f"{self.endpoint}/api/generate",
                            json={
                                "model": self.model_name,
                                "prompt": prompt,
                                "stream": False,
                            },
                        )
                        response.raise_for_status()
                        data = response.json()
                except httpx.TimeoutException as exc:
                    logger.warning(
                        "LLM generate timed out (%ss) for %s via %s on %s evidence",
                        self.timeout,
                        staff_name,
                        self.model_name,
                        attempt_label,
                    )
                    continue
                except httpx.HTTPStatusError as exc:
                    content = exc.response.text if exc.response else ""
                    logger.warning(
                        "LLM HTTP %s for %s via %s on %s evidence: %s",
                        exc.response.status_code if exc.response else "unknown",
                        staff_name,
                        self.model_name,
                        attempt_label,
                        content[:200],
                    )
                    continue
                except Exception as exc:
                    logger.warning(
                        "LLM generate failed for %s via %s on %s evidence (%s): %s",
                        staff_name,
                        self.model_name,
                        attempt_label,
                        type(exc).__name__,
                        exc,
                    )
                    continue

                text = (data.get("response") or "").strip()
                if not text or self._looks_like_refusal(text):
                    if not text:
                        logger.debug("LLM returned empty text for %s via %s", staff_name, self.model_name)
                    else:
                        logger.debug(
                            "LLM refusal detected for %s via %s: %s",
                            staff_name,
                            self.model_name,
                            text[:200],
                        )
                    continue
                if output_lang != normalized_prompt_lang and translator:
                    try:
                        text = translator.translate(
                            text,
                            source_lang=normalized_prompt_lang,
                            target_lang=output_lang,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Translation failed for %s (%s -> %s): %s",
                            staff_name,
                            normalized_prompt_lang,
                            output_lang,
                            exc,
                        )
                        continue
                return text

            logger.debug(
                "LLM backend %s unsupported for %s, using fallback", self.backend, staff_name
            )
            return fallback

        return fallback

    def _build_prompt(
        self,
        staff_name: str,
        snippets: Sequence[str],
        themes: Sequence[str],
        *,
        language: str = "no",
    ) -> str:
        if not self.model_name:
            return ""
        themes_text = ", ".join(themes) if themes else "temaet"
        trimmed = [snippet[:300].strip() for snippet in snippets[:3]]
        evidence_lines = "\n".join(f"- {snippet}" for snippet in trimmed if snippet)
        lang = (language or "no").lower()
        if lang.startswith("en"):
            return (
                "You are an assistant writing very short summaries that explain why a "
                "staff member is relevant for a user's topic.\n"
                "Write 1-2 concise sentences (maximum 50 words total) in English.\n"
                "Sentence 1 should introduce the staff member's role and field and "
                "link their work directly to the topic. Sentence 2 (optional) may "
                "give 1-2 concrete examples from the evidence using short noun "
                "phrases.\n"
                "Use only the topics and evidence below; do not invent projects or "
                "expertise, and do not include article titles, long quotes, lists, or "
                "numbers. Respond with only the final summary text, no bullets or "
                "headings.\n"
                f"Staff: {staff_name}\n"
                f"Topics: {themes_text}\n"
                "Evidence:\n"
                f"{evidence_lines}\n"
                "Answer:"
            )

        return (
            "Du er en assistent som skriver svært korte oppsummeringer av hvorfor en "
            "ansatt er relevant for et tema.\n"
            "Skriv 1-2 korte setninger (maks 50 ord totalt) på norsk.\n"
            "Første setning skal beskrive rolle og fagfelt og knytte dette direkte "
            "til temaet. Andre setning (valgfri) kan nevne 1-2 konkrete eksempler "
            "fra bevisene som korte substantivfraser.\n"
            "Bruk bare temaene og bevisene nedenfor; ikke finn opp prosjekter eller "
            "kompetanse, og ikke ta med artikkeltitler, lange sitater, lister eller "
            "tall. Svar kun med selve oppsummeringsteksten, uten punktlister eller "
            "overskrifter.\n"
            f"Ansatt: {staff_name}\n"
            f"Temaer: {themes_text}\n"
            "Bevis:\n"
            f"{evidence_lines}\n"
            "Svar:"
        )

    def _fallback_explanation(
        self,
        staff_name: str,
        snippets: Sequence[str],
        themes: Sequence[str],
        *,
        language: str = "no",
        translator=None,
        source_lang: str | None = None,
    ) -> str:
        lang = (language or "no").lower()
        top_theme = ", ".join(theme for theme in themes[:2] if theme) or (
            "the topic" if lang.startswith("en") else "temaet"
        )
        cleaned_snippets = [self._clean_snippet(snippet) for snippet in snippets[:2]]
        cleaned_snippets = [snippet for snippet in cleaned_snippets if snippet]
        if cleaned_snippets:
            evidence = " ".join(cleaned_snippets)
            text = (
                f"{staff_name} omtaler {top_theme} i dokumentasjonen. {evidence}"
                if not lang.startswith("en")
                else f"{staff_name} covers {top_theme} in the documentation. {evidence}"
            )
        else:
            text = (
                f"{staff_name} matcher {top_theme}, men vi fant ikke konkrete sitater."
                if not lang.startswith("en")
                else f"{staff_name} matches {top_theme}, but we did not find concrete "
                "quotes."
            )
        if translator and source_lang and lang != source_lang:
            try:
                return translator.translate(
                    text,
                    source_lang=source_lang,
                    target_lang=lang,
                )
            except Exception:
                return text
        return text

    @staticmethod
    def _clean_snippet(snippet: str) -> str:
        text = " ".join((snippet or "").strip().split())
        text = text.strip(" '\"")
        if not text:
            return ""
        if len(text) > 220:
            trimmed = text[:220]
            last_space = trimmed.rfind(" ")
            if last_space > 120:
                trimmed = trimmed[:last_space]
            text = trimmed.rstrip(",; ") + "..."
        if not text:
            return ""
        if text[-1] not in ".!?":
            text = text.rstrip(",; ") + "."
        return text

    @classmethod
    def _looks_like_refusal(cls, text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in cls._REFUSAL_MARKERS)
