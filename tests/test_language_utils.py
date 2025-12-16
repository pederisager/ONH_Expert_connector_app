from __future__ import annotations

from types import SimpleNamespace

from app.language_utils import build_language_context, detect_language


def _lang_config(
    *,
    embedding_mode: str = "multilingual",
    llm_mode: str = "match-user",
    translation_enabled: bool = False,
):
    translation = SimpleNamespace(enabled=translation_enabled, provider="none")
    return SimpleNamespace(
        default_ui="no",
        detection_enabled=True,
        embedding_language_mode=embedding_mode,
        llm_language_mode=llm_mode,
        translation=translation,
    )


def test_detect_language_returns_default_on_empty() -> None:
    assert detect_language("", default="no") == "no"


def test_build_language_context_flags_translation_for_en_only() -> None:
    cfg = _lang_config(
        embedding_mode="en-only",
        llm_mode="en-only",
        translation_enabled=True,
    )
    ctx = build_language_context(
        query_text="psykologi og helse", user_lang="no", language_config=cfg
    )
    assert ctx.embed_lang == "en"
    assert ctx.llm_lang == "en"
    assert ctx.translate_for_embedding is True
    assert ctx.translate_for_llm_input is True
    assert ctx.translate_llm_output is True
