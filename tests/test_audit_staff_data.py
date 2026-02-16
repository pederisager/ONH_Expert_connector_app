from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_staff_data import run_audit


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_audit_flags_likely_wrong_nva_mapping_as_high(tmp_path: Path) -> None:
    staff_info_path = tmp_path / "staff_info.json"
    staff_info_path.write_text(
        json.dumps(
            {
                "staff": [
                    {
                        "name": "Mismatch Person",
                        "job_title": "Associate Professor",
                        "research_focus": ["international relations", "security policy"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    staff_records_path = tmp_path / "data" / "staff_records.jsonl"
    _write_jsonl(
        staff_records_path,
        [
            {
                "slug": "Mismatch-Person",
                "name": "Mismatch Person",
                "department": "Statsvitenskap og internasjonale relasjoner",
                "profile_url": "https://example.com/mismatch",
                "summary": "International relations and foreign policy analysis.",
            }
        ],
    )

    staff_csv_path = tmp_path / "staff.csv"
    staff_csv_path.write_text(
        "Name,ONH_profile,NVA_profile,Department\n"
        "Mismatch Person,https://example.com/mismatch,https://nva.sikt.no/research-profile/999,Statsvitenskap og internasjonale relasjoner\n",
        encoding="utf-8",
    )

    nva_results_path = tmp_path / "data" / "nva" / "results.jsonl"
    _write_jsonl(
        nva_results_path,
        [
            {
                "employee_name": "Mismatch Person",
                "title": "Rare neuroendocrine tumours and oncology outcomes",
                "abstract": "Clinical cancer survival analysis in hospitals.",
            },
            {
                "employee_name": "Mismatch Person",
                "title": "Global surveillance of cancer survival",
                "abstract": "Epidemiology and medicine publication.",
            },
            {
                "employee_name": "Mismatch Person",
                "title": "Cost-effectiveness of thrombolysis in patient cohorts",
                "abstract": "Clinical treatment and hospital outcomes.",
            },
        ],
    )

    report = run_audit(
        staff_info_path=staff_info_path,
        staff_records_path=staff_records_path,
        staff_csv_path=staff_csv_path,
        nva_results_path=nva_results_path,
        chunks_dir=tmp_path / "data" / "index" / "chunks",
    )

    entry = next(item for item in report["staff"] if item["name"] == "Mismatch Person")
    checks = {issue["check"]: issue for issue in entry["issues"]}
    assert checks["nva_domain_outlier"]["severity"] == "high"


def test_audit_flags_chunk_id_collision_as_high(tmp_path: Path) -> None:
    staff_info_path = tmp_path / "staff_info.json"
    staff_info_path.write_text(json.dumps({"staff": []}), encoding="utf-8")

    staff_records_path = tmp_path / "data" / "staff_records.jsonl"
    _write_jsonl(
        staff_records_path,
        [
            {
                "slug": "Duplicate-Chunks",
                "name": "Duplicate Chunks",
                "department": "Psykologi",
                "profile_url": "https://example.com/dup",
                "summary": "Summary text",
            }
        ],
    )

    staff_csv_path = tmp_path / "staff.csv"
    staff_csv_path.write_text(
        "Name,ONH_profile,NVA_profile,Department\n"
        "Duplicate Chunks,https://example.com/dup,,Psykologi\n",
        encoding="utf-8",
    )

    nva_results_path = tmp_path / "data" / "nva" / "results.jsonl"
    _write_jsonl(nva_results_path, [])

    chunks_dir = tmp_path / "data" / "index" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / "Duplicate-Chunks.json").write_text(
        json.dumps(
            [
                {"chunk_id": "Duplicate-Chunks-profile-0000"},
                {"chunk_id": "Duplicate-Chunks-profile-0000"},
            ]
        ),
        encoding="utf-8",
    )

    report = run_audit(
        staff_info_path=staff_info_path,
        staff_records_path=staff_records_path,
        staff_csv_path=staff_csv_path,
        nva_results_path=nva_results_path,
        chunks_dir=chunks_dir,
    )

    entry = next(item for item in report["staff"] if item["name"] == "Duplicate Chunks")
    checks = {issue["check"]: issue for issue in entry["issues"]}
    assert checks["chunk_id_collision"]["severity"] == "high"
