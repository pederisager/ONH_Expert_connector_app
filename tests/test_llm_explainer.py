from __future__ import annotations

import httpx
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
    assert "Skriv 1-2 korte setninger (maks 50 ord totalt)" in prompt
    assert "Ansatt: Test Forsker" in prompt
    assert "Temaer: digital sikkerhet" in prompt


@pytest.mark.asyncio
async def test_llm_explainer_groq_chat_completion(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_post(self, url, **kwargs):  # type: ignore[override]
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        captured["json"] = kwargs.get("json")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Kort oppsummering fra Groq."}},
                ]
            },
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    explainer = LLMExplainer(
        model_config={
            "backend": "groq",
            "name": "llama-3.3-70b-versatile",
            "endpoint": "https://api.groq.com/openai/v1",
        }
    )
    summary = await explainer.generate(
        "Test Forsker",
        ["Dokumentasjon om digital sikkerhet."],
        ["digital sikkerhet"],
    )

    assert summary == "Kort oppsummering fra Groq."
    assert str(captured["url"]).endswith("/chat/completions")
    assert captured["headers"] == {
        "Authorization": "Bearer test-groq-key",
        "Content-Type": "application/json",
    }
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_llm_explainer_groq_without_api_key_falls_back(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    explainer = LLMExplainer(
        model_config={
            "backend": "groq",
            "name": "llama-3.3-70b-versatile",
        }
    )
    summary = await explainer.generate(
        "Test Forsker",
        ["Dokumentasjon om digital sikkerhet."],
        ["digital sikkerhet"],
    )

    assert "Test Forsker" in summary
    assert "digital sikkerhet" in summary
