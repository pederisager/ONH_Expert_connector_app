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
