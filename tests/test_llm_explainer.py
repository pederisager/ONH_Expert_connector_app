from __future__ import annotations

import pytest
from app.llm_explainer import LLMExplainer


@pytest.mark.asyncio
async def test_llm_explainer_returns_structured_summary_without_model() -> None:
    explainer = LLMExplainer(model_config={"backend": "ollama"})
    snippets = [
        "Detaljert omtale av digital sikkerhet i undervisningsopplegg ved Oslo Nye HA,yskole.",
    ]
    summary = await explainer.generate(
        "Test Forsker",
        snippets,
        ["digital sikkerhet", "personvern"],
    )
    assert "digital sikkerhet" in summary
    assert "Test Forsker" in summary


def test_build_prompt_returns_empty_when_no_model_name() -> None:
    explainer = LLMExplainer(model_config={"backend": "ollama"})
    prompt = explainer._build_prompt(
        "Test Forsker",
        ["snippet"],
        ["tema"],
        language="no",
    )
    assert prompt == ""


def test_build_prompt_english_includes_new_structure_instructions() -> None:
    explainer = LLMExplainer(model_config={"backend": "ollama", "name": "test-model"})
    prompt = explainer._build_prompt(
        "Test Researcher",
        ["Some evidence about digital health."],
        ["digital health"],
        language="en",
    )
    assert "Write 1-2 concise sentences (maximum 50 words total) in English." in prompt
    assert "Staff: Test Researcher" in prompt
    assert "Topics: digital health" in prompt


def test_build_prompt_norwegian_includes_new_structure_instructions() -> None:
    explainer = LLMExplainer(model_config={"backend": "ollama", "name": "test-modell"})
    prompt = explainer._build_prompt(
        "Test Forsker",
        ["Noe dokumentasjon om digital sikkerhet."],
        ["digital sikkerhet"],
        language="no",
    )
    assert "Skriv 1-2 korte setninger (maks 50 ord totalt) på norsk." in prompt
    assert "Ansatt: Test Forsker" in prompt
    assert "Temaer: digital sikkerhet" in prompt
