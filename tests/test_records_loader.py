from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from app.index.records_loader import load_curated_records


def _write_staff_yaml(path: Path) -> None:
    payload = textwrap.dedent(
        """
        - name: Test Forsker
          department: Psykologi
          profile_url: https://example.org/ansatte/Test-Forsker
          sources:
            - https://example.org/ansatte/Test-Forsker
            - https://nva.sikt.no/research-profile/42
          tags:
            - psykologi
        """
    ).strip()
    path.write_text(payload, encoding="utf-8")


def _write_records_jsonl(path: Path) -> None:
    record = {
        "slug": "Test-Forsker",
        "name": "Test Forsker",
        "title": "Førsteamanuensis",
        "department": "Psykologi",
        "profile_url": "https://example.org/ansatte/Test-Forsker",
        "summary": "Kort sammendrag om psykologi og helse.",
        "sources": ["https://example.org/ansatte/Test-Forsker", "https://nva.sikt.no/profil/123"],
        "tags": ["psykologi"],
    }
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_cristin_jsonl(path: Path) -> None:
    payload = {
        "employee_id": "42",
        "employee_name": "Test Forsker",
        "cristin_result_id": "999",
        "title": "Forskning på psykologi",
        "summary": "Studie om mental helse.",
        "venue": "Tidsskrift for psykologi",
        "category_name": "Academic article",
        "year_published": "2024",
        "tags": ["psykologi", "helse"],
        "source_url": "https://api.cristin.no/v2/results/999",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def test_load_curated_records_merges_metadata(tmp_path) -> None:
    staff_yaml = tmp_path / "staff.yaml"
    records_jsonl = tmp_path / "staff_records.jsonl"
    cristin_jsonl = tmp_path / "cristin.jsonl"

    _write_staff_yaml(staff_yaml)
    _write_records_jsonl(records_jsonl)
    _write_cristin_jsonl(cristin_jsonl)

    records = load_curated_records(
        staff_yaml_path=staff_yaml,
        records_jsonl_path=records_jsonl,
        cristin_results_path=cristin_jsonl,
    )
    assert len(records) == 1

    record = records[0]
    assert record.slug == "Test-Forsker"
    assert record.profile_url.endswith("Test-Forsker")
    assert record.title == "Førsteamanuensis"
    assert record.summary.startswith("Kort sammendrag")
    assert [link.url for link in record.sources][0] == "https://example.org/ansatte/Test-Forsker"
    assert "https://nva.sikt.no/profil/123" in [link.url for link in record.sources]
    assert record.cristin_results
    cristin_entry = record.cristin_results[0]
    assert cristin_entry.cristin_result_id == "999"
    assert "psykologi" in cristin_entry.tags


def test_load_curated_records_requires_matching_summary(tmp_path) -> None:
    staff_yaml = tmp_path / "staff.yaml"
    staff_yaml.write_text(
        textwrap.dedent(
            """
            - name: Manglende Forsker
              department: Psykologi
              profile_url: https://example.org/ansatte/Manglende-Forsker
              sources: []
              tags: []
            """
        ).strip(),
        encoding="utf-8",
    )
    records_jsonl = tmp_path / "staff_records.jsonl"
    _write_records_jsonl(records_jsonl)  # contains Test-Forsker only

    with pytest.raises(ValueError):
        load_curated_records(
            staff_yaml_path=staff_yaml,
            records_jsonl_path=records_jsonl,
        )
