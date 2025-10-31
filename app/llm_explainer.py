"""LLM-backed explanation generator."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import httpx


class LLMExplainer:
    """Thin wrapper around the configured LLM backend."""

    def __init__(self, model_config: Dict[str, Any]) -> None:
        self.model_config = model_config
        self.backend = model_config.get("backend", "ollama")
        self.model_name = model_config.get("name")
        self.endpoint = model_config.get("endpoint", "http://localhost:11434")

    async def generate(self, staff_name: str, snippets: Sequence[str], themes: Sequence[str]) -> str:
        prompt = self._build_prompt(staff_name, snippets, themes)
        if not prompt:
            return self._fallback_explanation(staff_name, snippets, themes)

        if self.backend == "ollama":
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
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
                return self._fallback_explanation(staff_name, snippets, themes)
            text = (data.get("response") or "").strip()
            return text or self._fallback_explanation(staff_name, snippets, themes)

        # Unsupported backend fallbacks to deterministic explanation.
        return self._fallback_explanation(staff_name, snippets, themes)

    def _build_prompt(self, staff_name: str, snippets: Sequence[str], themes: Sequence[str]) -> str:
        if not self.model_name:
            return ""
        themes_text = ", ".join(themes) if themes else "temaet"
        evidence_lines = "\n".join(f"- {snippet}" for snippet in snippets[:3])
        return (
            "Du er en assistent som skriver korte begrunnelser for hvorfor en ansatt matcher et undervisningstema.\n"
            "Skriv 2–4 setninger på norsk, og hvil deg kun på bevisene nedenfor. "
            "Inkluder ikke nye fakta, men referer eksplisitt til dokumentasjonen.\n"
            f"Ansatt: {staff_name}\n"
            f"Temaer: {themes_text}\n"
            "Bevis:\n"
            f"{evidence_lines}\n"
            "Svar:"
        )

    def _fallback_explanation(self, staff_name: str, snippets: Sequence[str], themes: Sequence[str]) -> str:
        top_theme = themes[0] if themes else "temaet"
        if snippets:
            snippet = snippets[0][:220].rstrip()
            return (
                f"{staff_name} matcher {top_theme} basert på følgende dokumentasjon: \"{snippet}\"."
                " Se kildeoversikten for flere detaljer."
            )
        return f"{staff_name} matcher {top_theme}, men ingen detaljerte sitat ble tilgjengelige."
