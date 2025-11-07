from __future__ import annotations

from pathlib import Path

from app.index.models import SourceLink
from app.index.refresh_staff import (
    StaffListing,
    StaffProfile,
    dedupe_preserve_order,
    extract_contact_details,
    extract_profile_summary,
    prepare_staff_metadata,
    profiles_to_records,
)


def test_extract_contact_details_recovers_email_and_phone() -> None:
    html = """
    <script>
    var ansatte_e_post = 'test.person@example.com';
    </script>
    <div class="telephone">Telefon: <span>123 45 678</span></div>
    """
    email, phone = extract_contact_details(html)
    assert email == "test.person@example.com"
    assert phone == "123 45 678"


def test_extract_profile_summary_picks_textual_container() -> None:
    html = """
    <div class="col-md-8 col-lg-8">
        <figure><img src="photo.jpg" alt="Bilde" /></figure>
    </div>
    <div class="col-md-8 col-lg-8">
        <p class="lead">Kort ingress.</p>
        <h2>Bakgrunn</h2>
        <p>Relevant erfaring.</p>
        <ul><li>Punkt 1</li></ul>
    </div>
    """
    summary = extract_profile_summary(html)
    assert "Kort ingress." in summary
    assert "Bakgrunn" in summary
    assert "Relevant erfaring." in summary


def _make_profile(
    name: str,
    department: str,
    *,
    title: str = "Rådgiver",
    summary: str = "Kort bio.",
    sources: list[str] | None = None,
    tags: list[str] | None = None,
    email: str | None = None,
    phone: str | None = None,
) -> StaffProfile:
    listing = StaffListing(
        name=name,
        title=title,
        department=department,
        profile_url=f"https://example.org/{name.lower().replace(' ', '-')}",
        slug=name.lower().replace(" ", "-"),
        image_url=None,
        tags=tags or [],
    )
    return StaffProfile(
        listing=listing,
        summary_text=summary,
        html_path=Path("snapshots") / f"{listing.slug}.html",
        sources=sources or [listing.profile_url],
        email=email,
        phone=phone,
    )


def test_prepare_staff_metadata_orders_and_keeps_fields() -> None:
    profiles = [
        _make_profile(
            "Beta Person",
            "Avdeling B",
            sources=["https://example.org/beta", "https://example.org/beta"],
            tags=["tag1"],
            email="beta@example.org",
        ),
        _make_profile(
            "Alpha Person",
            "Avdeling A",
            phone="99999999",
            tags=[],
        ),
    ]

    prepared = prepare_staff_metadata(profiles)
    assert [entry["name"] for entry in prepared] == ["Alpha Person", "Beta Person"]
    alpha_entry = prepared[0]
    assert alpha_entry["department"] == "Avdeling A"
    assert alpha_entry["profile_url"].endswith("/alpha-person")
    assert alpha_entry["sources"] == ["https://example.org/alpha-person"]
    assert "email" not in alpha_entry
    assert alpha_entry["tags"] == []

    beta_entry = prepared[1]
    assert beta_entry["email"] == "beta@example.org"
    assert beta_entry["sources"] == ["https://example.org/beta"]
    assert beta_entry["tags"] == ["tag1"]


def test_profiles_to_records_returns_staff_records() -> None:
    profiles = [
        _make_profile(
            "Gamma Person",
            "Avdeling C",
            summary="En kort tekst.",
            tags=["tema"],
            sources=["https://example.org/gamma", "https://example.org/gamma-2"],
        )
    ]
    records = profiles_to_records(profiles)
    assert len(records) == 1
    record = records[0]
    assert record.slug == "gamma-person"
    assert record.tags == ["tema"]
    assert record.summary == "En kort tekst."
    assert record.profile_url.endswith("/gamma-person")
    assert [link.url for link in record.sources] == [
        "https://example.org/gamma",
        "https://example.org/gamma-2",
    ]
    assert all(isinstance(link, SourceLink) for link in record.sources)


def test_dedupe_preserve_order_removes_duplicates() -> None:
    items = ["https://a", "https://b", "https://a", "", "https://c", "https://b"]
    assert dedupe_preserve_order(items) == ["https://a", "https://b", "https://c"]
