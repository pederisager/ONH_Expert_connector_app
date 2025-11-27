from __future__ import annotations

import pytest

from app.llm_explainer import LLMExplainer


@pytest.mark.asyncio
async def test_llm_explainer_returns_structured_summary_without_model() -> None:
    explainer = LLMExplainer(model_config={"backend": "ollama"})
    snippets = [
        "Detaljert omtale av digital sikkerhet i undervisningsopplegg ved Oslo Nye Høyskole.",
    ]
    summary = await explainer.generate("Test Forsker", snippets, ["digital sikkerhet", "personvern"])
    assert "digital sikkerhet" in summary
    assert "Test Forsker" in summary
