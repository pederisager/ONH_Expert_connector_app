from __future__ import annotations

from app.match_engine import (
    MAX_COMBINED_TEXT_CHARS,
    MatchEngine,
    PageContent,
    StaffDocument,
    StaffProfile,
    extract_themes,
)


def test_extract_themes_returns_keywords() -> None:
    text = "Dette er en tekst om psykologi, helse og undervisning i psykologi ved ONH."
    themes = extract_themes(text, top_k=5)
    assert "psykologi" in themes


def test_extract_themes_prefers_phrases_over_fillers() -> None:
    text = (
        "Vi trenger undervisning om digital sikkerhet og personvern i helseteknologi. "
        "Digital sikkerhet skal gi studentene praksis i datasikkerhet."
    )
    themes = extract_themes(text, top_k=3)
    assert any(theme == "digital sikkerhet" for theme in themes)
    assert "vi" not in themes


def test_match_engine_ranks_staff_member() -> None:
    profile = StaffProfile(
        name="Test Forsker",
        department="Psykologi",
        profile_url="https://example.com",
        sources=["https://example.com"],
        tags=["psykologi", "helse"],
    )
    page = PageContent(
        url="https://example.com",
        title="Forskning",
        text="Forskeren skriver om psykologi og helseatferd i nyere publikasjoner.",
    )

    engine = MatchEngine()
    results = engine.rank(
        documents=[StaffDocument(profile=profile, pages=[page])],
        themes=["psykologi", "helsatferd"],
        department_filter=None,
        max_candidates=5,
        diversity_weight=0.0,
    )

    assert results
    result = results[0]
    assert result.staff.name == "Test Forsker"
    assert result.score > 0
    assert result.citations


def test_staff_document_combined_text_trims() -> None:
    profile = StaffProfile(
        name="Test",
        department="Helse",
        profile_url="https://example.com/profile",
        sources=["https://example.com/page"],
        tags=[],
    )
    long_text = "A" * (MAX_COMBINED_TEXT_CHARS + 100)
    doc = StaffDocument(profile=profile, pages=[PageContent(url="https://example.com", title="Stub", text=long_text)])
    assert len(doc.combined_text) == MAX_COMBINED_TEXT_CHARS


def test_extract_themes_ignores_filler_phrase() -> None:
    text = "Looking for help with AI ethics and policy in healthcare."
    themes = extract_themes(text, top_k=5)
    assert all("looking" not in theme for theme in themes)
    assert any("ai ethics" in theme or theme == "ethics" for theme in themes)


def test_extract_themes_handles_norwegian_morphology() -> None:
    text = "Digital sikkerhet og datasikkerheten til pasientdata i helseteknologi."
    themes = extract_themes(text, top_k=4)
    assert any(theme.startswith("digital sikkerhet") or theme == "sikkerhet" for theme in themes)


def test_extract_themes_mixed_language_input() -> None:
    text = "Machine learning for helseteknologi and clinical decision support."
    themes = extract_themes(text, top_k=4)
    assert any("machine learning" in theme for theme in themes)
    assert any("helseteknologi" in theme for theme in themes)


def test_extract_themes_limits_clause_length() -> None:
    text = (
        "POSSISJON: Ett triglyserid gir dermed ett monoglyserid og en fettsyre. "
        "Frie fettsyrer både lipase og pankreasenzymer avspalter; fettsyrene transporteres videre."
    )
    themes = extract_themes(text, top_k=6)
    # themes should be concise, not entire clauses
    assert all(len(theme.split()) <= 6 for theme in themes)
    assert any("lipase" in theme for theme in themes)
    assert any("fettsyre" in theme or "fettsyrer" in theme for theme in themes)
