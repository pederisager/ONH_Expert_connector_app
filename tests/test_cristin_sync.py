from __future__ import annotations

import json
from pathlib import Path

import httpx

from app.index.cristin.models import ResultRecord
from app.index.cristin.normalizer import normalize_result
from app.index.cristin.sync import (
    CristinClientConfig,
    StaffMember,
    sync_cristin_results,
)


def test_normalize_result_extracts_core_fields() -> None:
    listing = {
        "cristin_result_id": "RES-1",
        "title": {"en": "Listing Title"},
        "year_published": "2020",
    }
    detail = {
        "cristin_result_id": "RES-1",
        "title": {"en": "Detail Title"},
        "abstract": {"en": "Detailed abstract text."},
        "journal": {"name": {"en": "Expert Journal"}},
        "category": {"code": "ARTICLE", "name": {"en": "Academic article"}},
        "classification": {
            "keywords": [
                {"name": {"en": "Innovation"}},
                {"name": {"en": "Education"}},
            ]
        },
        "contributors": {
            "preview": [
                {"first_name": "Ada", "surname": "Lovelace"},
                {"first_name": "Grace", "surname": "Hopper"},
            ]
        },
        "url": "https://api.cristin.no/v2/results/RES-1",
    }

    record = normalize_result(
        employee_id="123",
        employee_name="Test Person",
        listing_payload=listing,
        detail_payload=detail,
        lang="en",
    )

    assert isinstance(record, ResultRecord)
    assert record.title == "Detail Title"
    assert record.summary == "Detailed abstract text."
    assert record.venue == "Expert Journal"
    assert record.category_code == "ARTICLE"
    assert "Innovation" in record.tags
    assert record.contributors[0].first_name == "Ada"


def test_sync_cristin_results_writes_expected_artifacts(tmp_path: Path) -> None:
    staff_members = [StaffMember(employee_id="999", name="Employee One", department="Dept")]

    listings_payload = [
        {
            "cristin_result_id": "RES-2",
            "title": {"en": "Listing Title"},
            "year_published": "2023",
            "url": "https://api.cristin.no/v2/results/RES-2",
        },
        {
            "cristin_result_id": "RES-3",
            "title": {"en": "Listing Title"},
            "year_published": "2019",
            "url": "https://api.cristin.no/v2/results/RES-3",
        },
    ]

    detail_payload = {
        "cristin_result_id": "RES-2",
        "title": {"en": "Detailed Title"},
        "abstract": {"en": "Insightful summary."},
        "journal": {"name": {"en": "Applied Sciences"}},
        "category": {"code": "ARTICLE", "name": {"en": "Academic article"}},
        "classification": {"keywords": [{"name": {"en": "Analytics"}}]},
        "contributors": {"preview": [{"first_name": "Lin", "surname": "Zhu"}]},
        "url": "https://api.cristin.no/v2/results/RES-2",
    }

    detail_payload_duplicate = {
        "cristin_result_id": "RES-3",
        "title": {"en": "Detailed Title"},
        "category": {"code": "ARTICLE", "name": {"en": "Academic article"}},
        "url": "https://api.cristin.no/v2/results/RES-3",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/persons/999/results"):
            return httpx.Response(200, json=listings_payload)
        if request.url.path.endswith("/results/RES-2"):
            return httpx.Response(200, json=detail_payload)
        if request.url.path.endswith("/results/RES-3"):
            return httpx.Response(200, json=detail_payload_duplicate)
        return httpx.Response(404, json={"detail": "not found"})

    transport = httpx.MockTransport(handler)
    session = httpx.Client(transport=transport, base_url="https://api.cristin.no/v2")

    output_path = tmp_path / "results.jsonl"
    raw_dir = tmp_path / "raw"

    stats = sync_cristin_results(
        staff_members=staff_members,
        output_path=output_path,
        raw_dir=raw_dir,
        client_config=CristinClientConfig(lang="en", per_page=1000),
        cache=None,
        session=session,
    )

    assert stats.results_normalized == 1
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    record = json.loads(content[0])
    assert record["title"] == "Detailed Title"
    assert record["summary"] == "Insightful summary."
    assert record["venue"] == "Applied Sciences"
    assert record["tags"] == ["Analytics", "Academic article"]

    listing_file = raw_dir / "999" / "results_listing.json"
    detail_file = raw_dir / "999" / "RES-2.json"
    detail_file_dup = raw_dir / "999" / "RES-3.json"
    assert listing_file.exists()
    assert detail_file.exists()
    assert detail_file_dup.exists()
    assert json.loads(listing_file.read_text(encoding="utf-8")) == listings_payload
    assert json.loads(detail_file.read_text(encoding="utf-8")) == detail_payload
    assert json.loads(detail_file_dup.read_text(encoding="utf-8")) == detail_payload_duplicate
