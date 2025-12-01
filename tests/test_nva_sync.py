from __future__ import annotations

import json
from pathlib import Path

import httpx

from app.index.nva.models import NvaPublicationRecord
from app.index.nva.normalizer import normalize_publication
from app.index.nva.sync import (
    NvaApiKey,
    NvaClientConfig,
    StaffMember,
    sync_nva_publications,
)


def test_normalize_publication_extracts_core_fields() -> None:
    hit = {
        "id": "https://api.nva.unit.no/publication/ABC-123",
        "identifier": "ABC-123",
        "handle": "https://hdl.handle.net/123",
        "entityDescription": {
            "mainTitle": "NVA Title",
            "abstract": "Detailed abstract text.",
            "language": "en",
            "publicationDate": {"year": "2024"},
            "tags": ["Innovation", "Education"],
            "contributors": [
                {
                    "identity": {"id": "https://api.nva.unit.no/cristin/person/1", "name": "Ada Lovelace"},
                    "role": {"type": "Creator"},
                }
            ],
            "reference": {
                "doi": "10.1234/abc",
                "publicationContext": {"type": "Journal", "publisher": {"name": "Expert Journal"}},
                "publicationInstance": {"type": "AcademicArticle"},
            },
        },
    }

    record = normalize_publication(
        employee_id="123",
        employee_name="Test Person",
        hit_payload=hit,
    )

    assert isinstance(record, NvaPublicationRecord)
    assert record.title == "NVA Title"
    assert record.abstract == "Detailed abstract text."
    assert record.venue == "Expert Journal"
    assert record.category == "AcademicArticle"
    assert "Innovation" in record.tags
    assert record.contributors[0].name == "Ada Lovelace"
    assert record.year == "2024"
    assert record.doi == "10.1234/abc"


def test_sync_nva_publications_writes_expected_artifacts(tmp_path: Path) -> None:
    staff_members = [StaffMember(employee_id="999", name="Employee One", department="Dept")]

    hits_payload = [
        {
            "id": "https://api.nva.unit.no/publication/ABC-001",
            "identifier": "ABC-001",
            "entityDescription": {
                "mainTitle": "Listing Title",
                "abstract": "Insightful summary.",
                "publicationDate": {"year": "2023"},
                "tags": ["Analytics"],
                "reference": {
                    "publicationContext": {"type": "Journal", "publisher": {"name": "Applied Sciences"}},
                    "publicationInstance": {"type": "AcademicArticle"},
                },
                "contributors": [{"identity": {"name": "Lin Zhu"}}],
            },
        },
        {
            "id": "https://api.nva.unit.no/publication/ABC-002",
            "identifier": "ABC-002",
            "entityDescription": {
                "mainTitle": "Listing Title",
                "publicationDate": {"year": "2019"},
                "reference": {
                    "publicationInstance": {"type": "AcademicArticle"},
                },
            },
        },
    ]

    token_url = "https://auth.local/token"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL(token_url):
            return httpx.Response(200, json={"access_token": "token-123", "expires_in": 3600})
        if request.url.path.endswith("/search/resources"):
            return httpx.Response(200, json={"hits": hits_payload, "nextResults": None})
        return httpx.Response(404, json={"detail": "not found"})

    transport = httpx.MockTransport(handler)
    session = httpx.Client(transport=transport, base_url="https://api.nva.unit.no")

    output_path = tmp_path / "results.jsonl"
    raw_dir = tmp_path / "raw"

    stats = sync_nva_publications(
        staff_members=staff_members,
        output_path=output_path,
        raw_dir=raw_dir,
        api_key=NvaApiKey(client_id="id", client_secret="secret", token_url=token_url),
        client_config=NvaClientConfig(base_url="https://api.nva.unit.no", results_per_page=100),
        cache=None,
        session=session,
    )

    assert stats.results_normalized == 1
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    record = json.loads(content[0])
    assert record["title"] == "Listing Title"
    assert record["abstract"] == "Insightful summary."
    assert record["venue"] == "Applied Sciences"
    assert record["tags"] == ["Analytics"]
    assert record["nva_publication_id"] == "ABC-001"

    listing_file = raw_dir / "999" / "results_listing.json"
    detail_file = raw_dir / "999" / "ABC-001.json"
    detail_file_dup = raw_dir / "999" / "ABC-002.json"
    assert listing_file.exists()
    assert detail_file.exists()
    assert detail_file_dup.exists()
    assert json.loads(listing_file.read_text(encoding="utf-8")) == hits_payload
