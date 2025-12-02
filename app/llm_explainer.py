"""LLM-backed explanation generator with optional translation."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import httpx


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
        self.timeout = float(model_config.get("timeout", 60.0))

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

        prompt = self._build_prompt(
            staff_name,
            snippets,
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
        if not prompt:
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
            except Exception:
                return fallback
            text = (data.get("response") or "").strip()
            if not text or self._looks_like_refusal(text):
                return fallback
            if output_lang != normalized_prompt_lang and translator:
                try:
                    text = translator.translate(
                        text,
                        source_lang=normalized_prompt_lang,
                        target_lang=output_lang,
                    )
                except Exception:
                    return fallback
            return text

        # Unsupported backend fallbacks to deterministic explanation.
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
        evidence_lines = "\n".join(f"- {snippet}" for snippet in snippets[:3])
        lang = (language or "no").lower()
        if lang.startswith("en"):
            return (
                "You are an assistant writing short justifications for why a staff member matches a teaching topic.\n"
                "Write 2-4 sentences in English. Use only the evidence below and avoid adding new facts.\n"
                f"Staff: {staff_name}\n"
                f"Topics: {themes_text}\n"
                "Evidence:\n"
                f"{evidence_lines}\n"
                "Answer:"
            )

        return (
            "Du er en assistent som skriver korte begrunnelser for hvorfor en ansatt matcher et undervisningstema.\n"
            "Skriv 2-4 setninger pA� norsk, og bruk bare bevisene nedenfor. "
            "Ikke legg til nye fakta, men referer eksplisitt til dokumentasjonen.\n"
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
                else f"{staff_name} matches {top_theme}, but we did not find concrete quotes."
            )
        if translator and source_lang and lang != source_lang:
            try:
                return translator.translate(text, source_lang=source_lang, target_lang=lang)
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
