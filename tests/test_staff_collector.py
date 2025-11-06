from __future__ import annotations

import json
from pathlib import Path

from app.index.refresh_staff import (
    CsvStaffEntry,
    SnapshotFetcher,
    StaffCollector,
    load_expert_topics,
    slug_from_profile_url,
)


def _write_snapshot(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_staff_collector_respects_csv_allowlist(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    roster_payload = [
        {
            "field_ansatte_stilling": "Professor",
            "field_ansatte_navn": "Alpha One",
            "field_avdeling": "Dept A",
            "view_node": "/om-oss/Ansatte/Alpha-One",
            "field_top_bilde": "",
            "field_tags": "",
        },
        {
            "field_ansatte_stilling": "Professor",
            "field_ansatte_navn": "Beta Two",
            "field_avdeling": "Dept B",
            "view_node": "/om-oss/Ansatte/Beta-Two",
            "field_top_bilde": "",
            "field_tags": "",
        },
    ]
    _write_snapshot(
        snapshot_dir / "oslo_staff_index.json",
        json.dumps(roster_payload, ensure_ascii=False),
    )

    alpha_profile_url = "https://example.org/om-oss/Ansatte/Alpha-One"
    alpha_slug = slug_from_profile_url(alpha_profile_url)
    alpha_html = """
    <div class="col-md-8 col-lg-8">
        <h1>Alpha One</h1>
        <h3 class="ansatte_stilling">Professor i økonomi</h3>
        <p class="lead">Kort bio.</p>
    </div>
    <div class="sidebar col">
        <div class="telephone"><span>12345678</span></div>
    </div>
    <script>var ansatte_e_post = 'alpha@example.org';</script>
    """
    _write_snapshot(
        snapshot_dir / "staff" / alpha_slug / "profile.html",
        alpha_html,
    )

    csv_entry = CsvStaffEntry(
        name="Alpha One",
        profile_url=alpha_profile_url,
        department="Dept A",
        nva_profile_url="https://nva.sikt.no/research-profile/123",
        slug=alpha_slug,
    )

    fetcher = SnapshotFetcher(snapshot_dir=snapshot_dir, offline=True)
    collector = StaffCollector(
        fetcher,
        csv_entries_order=[csv_entry],
        csv_entries={csv_entry.slug: csv_entry},
        role_keywords=(),
    )

    with fetcher:
        profiles = collector.collect_staff_profiles()

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.listing.slug == alpha_slug
    assert profile.listing.name == "Alpha One"
    assert profile.listing.department == "Dept A"
    assert profile.listing.title == "Professor i økonomi"
    assert profile.email == "alpha@example.org"
    assert profile.nva_profile_url == "https://nva.sikt.no/research-profile/123"
    assert "https://nva.sikt.no/research-profile/123" in profile.sources


def test_load_expert_topics_extracts_topics(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    experts_html = """
    <table class="table">
        <tbody>
            <tr><td><strong>A</strong></td><td>&nbsp;</td></tr>
            <tr>
                <td>Digital markedsføring</td>
                <td><a href="/om-oss/Ansatte/alpha-one">Alpha One</a></td>
            </tr>
            <tr>
                <td>Forskning</td>
                <td>Ingen lenker</td>
            </tr>
        </tbody>
    </table>
    """
    snapshot_path = snapshot_dir / "experts" / "experts.html"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(experts_html, encoding="utf-8")

    profile_url = "https://example.org/om-oss/Ansatte/Alpha-One"
    entry = CsvStaffEntry(
        name="Alpha One",
        profile_url=profile_url,
        department="Dept A",
        nva_profile_url=None,
        slug=slug_from_profile_url(profile_url),
    )

    fetcher = SnapshotFetcher(snapshot_dir=snapshot_dir, offline=True)
    topics = load_expert_topics(fetcher, [entry])

    assert topics == {entry.slug: ["Digital markedsføring"]}
