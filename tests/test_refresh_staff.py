from __future__ import annotations

from app.index.refresh_staff import extract_profile_summary


def test_extract_profile_summary_prioritizes_expertise_sections_and_drops_admin_noise() -> None:
    html = """
    <div class="col-md-8 col-lg-8">
        <h2>Jobber med</h2>
        <p>Ledelsespsykologi, arbeidsmiljø og teamutvikling.</p>
        <h2>CV</h2>
        <p>CV oppdatert 2025. Styreverv og administrative roller.</p>
        <h2>Kontaktinformasjon</h2>
        <p>Telefon: 11 22 33 44</p>
        <p>E-post: person@example.org</p>
    </div>
    """

    summary = extract_profile_summary(html)

    assert "Jobber med" in summary
    assert "Ledelsespsykologi, arbeidsmiljø og teamutvikling." in summary
    assert "CV oppdatert 2025" not in summary
    assert "administrative roller" not in summary
    assert "Telefon" not in summary
    assert "E-post" not in summary


def test_extract_profile_summary_keeps_neutral_content_when_no_expertise_heading_exists() -> None:
    html = """
    <div class="col-md-8 col-lg-8">
        <h2>Bakgrunn</h2>
        <p>Har forsket på konfliktmekling i organisasjoner.</p>
        <ul>
            <li>Publisert om beslutningsbias i team.</li>
            <li>Underviser i kommunikasjon og forhandling.</li>
        </ul>
    </div>
    """

    summary = extract_profile_summary(html)

    assert "Bakgrunn" in summary
    assert "Har forsket på konfliktmekling i organisasjoner." in summary
    assert "Publisert om beslutningsbias i team." in summary
    assert "Underviser i kommunikasjon og forhandling." in summary


def test_extract_profile_summary_fallback_prevents_empty_summary_when_only_low_priority_sections_exist() -> None:
    html = """
    <div class="col-md-8 col-lg-8">
        <h2>CV</h2>
        <p>Professor i organisasjonspsykologi siden 2018.</p>
    </div>
    """

    summary = extract_profile_summary(html)

    assert summary.strip() != ""
    assert "Professor i organisasjonspsykologi siden 2018." in summary
