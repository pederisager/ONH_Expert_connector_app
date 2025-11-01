from __future__ import annotations

import json
from pathlib import Path

from app.index.refresh_staff import (
    NvaResolver,
    SnapshotFetcher,
    StaffListing,
    StaffProfile,
)


def _make_staff_profile(
    snapshot_dir: Path, slug: str = "alexander-utne"
) -> StaffProfile:
    listing = StaffListing(
        name="Alexander Utne",
        title="Førsteamanuensis",
        department="Økonomi og administrasjon",
        profile_url="https://oslonyehoyskole.no/om-oss/Ansatte/Alexander-Utne",
        slug=slug,
    )
    profile_html = snapshot_dir / "staff" / slug / "profile.html"
    profile_html.parent.mkdir(parents=True, exist_ok=True)
    profile_html.write_text("<html></html>", encoding="utf-8")
    return StaffProfile(
        listing=listing,
        summary_text="",
        html_path=profile_html,
        sources=[listing.profile_url],
    )


def _write_html_snapshot(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def test_nva_resolver_picks_matching_link(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    resolver = NvaResolver(SnapshotFetcher(snapshot_dir=snapshot_dir, offline=True))
    profile = _make_staff_profile(snapshot_dir)
    search_snapshot = snapshot_dir / "staff" / profile.listing.slug / "nva_search.html"
    _write_html_snapshot(
        search_snapshot,
        """
        <html>
            <body>
                <a class="result" href="/research-profile/564459">
                    <h3>Alexander Utne</h3>
                    <p>Oslo Nye Høyskole</p>
                </a>
                <a class="result" href="/research-profile/999">
                    <h3>Other Person</h3>
                </a>
            </body>
        </html>
        """,
    )

    with resolver.fetcher:
        resolved, unresolved = resolver.attach_profile_urls([profile])

    assert resolved == 1
    assert unresolved == 0
    assert profile.nva_profile_url == "https://nva.sikt.no/research-profile/564459"
    assert profile.sources[-1] == "https://nva.sikt.no/research-profile/564459"


def test_nva_resolver_uses_api_snapshot_when_available(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    resolver = NvaResolver(SnapshotFetcher(snapshot_dir=snapshot_dir, offline=True))
    profile = _make_staff_profile(snapshot_dir, slug="api-match")
    api_snapshot = snapshot_dir / "staff" / profile.listing.slug / "nva_search.json"
    api_snapshot.parent.mkdir(parents=True, exist_ok=True)
    api_snapshot.write_text(
        json.dumps(
            {
                "hits": [
                    {
                        "id": "research-profile/564459",
                        "displayName": "Alexander Utne",
                        "uri": "https://nva.sikt.no/research-profile/564459",
                    },
                    {
                        "id": "research-profile/999",
                        "displayName": "Someone Else",
                        "uri": "https://nva.sikt.no/research-profile/999",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with resolver.fetcher:
        resolved, unresolved = resolver.attach_profile_urls([profile])

    assert resolved == 1
    assert unresolved == 0
    assert profile.nva_profile_url == "https://nva.sikt.no/research-profile/564459"


def test_nva_resolver_returns_none_when_no_candidates(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    resolver = NvaResolver(SnapshotFetcher(snapshot_dir=snapshot_dir, offline=True))
    profile = _make_staff_profile(snapshot_dir, slug="no-match")
    search_snapshot = snapshot_dir / "staff" / profile.listing.slug / "nva_search.html"
    _write_html_snapshot(
        search_snapshot,
        """
        <html>
            <body>
                <a class="result" href="/research-profile/564459">
                    <h3>Completely Different</h3>
                </a>
            </body>
        </html>
        """,
    )

    with resolver.fetcher:
        resolved, unresolved = resolver.attach_profile_urls([profile])

    assert resolved == 0
    assert unresolved == 1
    assert profile.nva_profile_url is None
    assert len(profile.sources) == 1  # original ONH profile only


def test_nva_resolver_skips_api_without_key_when_online(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    fetcher = SnapshotFetcher(snapshot_dir=snapshot_dir, offline=False)
    resolver = NvaResolver(fetcher, api_key=None)
    profile = _make_staff_profile(snapshot_dir, slug="html-only")
    search_snapshot = snapshot_dir / "staff" / profile.listing.slug / "nva_search.html"
    _write_html_snapshot(
        search_snapshot,
        """
        <html>
            <body>
                <a href="/research-profile/564459">Alexander Utne</a>
            </body>
        </html>
        """,
    )

    with resolver.fetcher:
        resolved, unresolved = resolver.attach_profile_urls([profile])

    assert resolved == 1
    assert unresolved == 0
    assert profile.nva_profile_url == "https://nva.sikt.no/research-profile/564459"
