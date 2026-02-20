"""Deterministic data-quality audit for staff profile, staff_info, NVA, and chunks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-zÆØÅæøå]{3,}")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

STOPWORDS = {
    "and",
    "are",
    "av",
    "basert",
    "ble",
    "can",
    "den",
    "det",
    "dette",
    "eller",
    "for",
    "fra",
    "har",
    "hos",
    "hvor",
    "i",
    "ikke",
    "in",
    "is",
    "med",
    "mot",
    "og",
    "om",
    "on",
    "på",
    "som",
    "the",
    "til",
    "ved",
}

MEDICAL_TOKENS = {
    "cancer",
    "clinical",
    "diabetes",
    "epidemiology",
    "health",
    "hospital",
    "medicine",
    "neuroendocrine",
    "oncology",
    "patient",
    "sarcoma",
    "survival",
    "thrombolysis",
    "tumour",
}

HEALTH_DEPARTMENT_TOKENS = {"helse", "syke", "fysio", "klinisk"}
KNOWN_PRIORITY_SEEDS = [
    {
        "name": "Jilwan Soltanpanah",
        "check": "staff_info_vs_profile_divergence",
        "message": "Seeded review target: staff_info profile mismatch must be curated.",
    },
    {
        "name": "Christopher White",
        "check": "nva_domain_outlier",
        "message": "Seeded review target: likely mixed-person NVA mapping needs manual curation.",
    },
]


@dataclass(slots=True)
class StaffRecord:
    name: str
    slug: str
    department: str
    summary: str
    profile_url: str


def _tokenize(value: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(value or "")
        if token and token.lower() not in STOPWORDS
    ]


def _jaccard(a_tokens: set[str], b_tokens: set[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return intersection / max(1, union)


def _char_ngram_counter(value: str, n: int = 3) -> Counter[str]:
    collapsed = " ".join((value or "").lower().split())
    if len(collapsed) < n:
        return Counter()
    return Counter(collapsed[idx : idx + n] for idx in range(len(collapsed) - n + 1))


def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    shared_keys = set(a) & set(b)
    dot = sum(a[key] * b[key] for key in shared_keys)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _slugify(value: str) -> str:
    cleaned = NON_ALNUM_RE.sub("-", (value or "").strip().lower()).strip("-")
    return cleaned or "unknown"


def _load_staff_info(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("staff", [])
    by_name: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        by_name[name.lower()] = entry
    return by_name


def _staff_info_text(entry: dict[str, Any] | None) -> str:
    if not entry:
        return ""
    parts: list[str] = []
    for key in (
        "job_title",
        "expertise_domains",
        "teaching_courses",
        "research_focus",
        "other_relevant_expertise",
    ):
        value = entry.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts)


def _load_staff_records(path: Path) -> dict[str, StaffRecord]:
    records: dict[str, StaffRecord] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            name = str(payload.get("name") or "").strip()
            if not name:
                continue
            record = StaffRecord(
                name=name,
                slug=str(payload.get("slug") or _slugify(name)),
                department=str(payload.get("department") or "").strip(),
                summary=str(payload.get("summary") or "").strip(),
                profile_url=str(payload.get("profile_url") or "").strip(),
            )
            records[name.lower()] = record
    return records


def _load_staff_csv(path: Path) -> dict[str, dict[str, str]]:
    by_name: dict[str, dict[str, str]] = {}
    if not path.exists():
        return by_name
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("Name") or "").strip()
            if not name:
                continue
            by_name[name.lower()] = {
                "nva_profile_url": (row.get("NVA_profile") or "").strip(),
                "department": (row.get("Department") or "").strip(),
                "profile_url": (row.get("ONH_profile") or "").strip(),
            }
    return by_name


def _load_nva_results(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return by_name
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            name = str(payload.get("employee_name") or "").strip()
            if not name:
                continue
            by_name[name.lower()].append(payload)
    return by_name


def _detect_chunk_collisions(chunks_dir: Path) -> dict[str, dict[str, Any]]:
    collisions_by_slug: dict[str, dict[str, Any]] = {}
    if not chunks_dir.exists():
        return collisions_by_slug
    for file_path in sorted(chunks_dir.glob("*.json")):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            collisions_by_slug[file_path.stem] = {
                "duplicate_ids": [],
                "error": "invalid_json",
            }
            continue
        if not isinstance(payload, list):
            continue
        counts = Counter(
            str(item.get("chunk_id") or "")
            for item in payload
            if isinstance(item, dict) and item.get("chunk_id")
        )
        duplicate_ids = sorted([chunk_id for chunk_id, count in counts.items() if count > 1])
        if duplicate_ids:
            collisions_by_slug[file_path.stem] = {
                "duplicate_ids": duplicate_ids,
                "duplicate_count": len(duplicate_ids),
                "total_chunks": len(payload),
            }
    return collisions_by_slug


def _severity_for_divergence(score: float) -> str | None:
    if score < 0.12:
        return "high"
    if score < 0.22:
        return "medium"
    if score < 0.32:
        return "low"
    return None


def _severity_for_alignment(
    *,
    total_publications: int,
    alignment_rate: float,
    medical_share: float,
    is_health_department: bool,
) -> str | None:
    if total_publications < 2:
        return None
    if total_publications >= 3 and (
        alignment_rate < 0.25 or (medical_share >= 0.5 and not is_health_department)
    ):
        return "high"
    if total_publications >= 3 and alignment_rate < 0.45:
        return "medium"
    if alignment_rate < 0.6:
        return "low"
    return None


def _is_health_department(value: str) -> bool:
    lowered = (value or "").lower()
    return any(token in lowered for token in HEALTH_DEPARTMENT_TOKENS)


def _build_issue(
    *,
    check: str,
    severity: str,
    message: str,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "check": check,
        "severity": severity,
        "status": "open",
        "message": message,
        "metrics": metrics or {},
    }


def run_audit(
    *,
    staff_info_path: Path,
    staff_records_path: Path,
    staff_csv_path: Path,
    nva_results_path: Path,
    chunks_dir: Path,
) -> dict[str, Any]:
    staff_info = _load_staff_info(staff_info_path)
    staff_records = _load_staff_records(staff_records_path)
    staff_csv = _load_staff_csv(staff_csv_path)
    nva_by_name = _load_nva_results(nva_results_path)
    chunk_collisions = _detect_chunk_collisions(chunks_dir)

    all_names = sorted(
        set(staff_records.keys()) | set(staff_info.keys()) | set(staff_csv.keys())
    )
    staff_reports: list[dict[str, Any]] = []

    for name_key in all_names:
        record = staff_records.get(name_key)
        csv_entry = staff_csv.get(name_key, {})
        info_entry = staff_info.get(name_key)
        nva_entries = nva_by_name.get(name_key, [])

        display_name = (
            record.name
            if record
            else str(info_entry.get("name"))
            if info_entry
            else name_key
        )
        slug = record.slug if record else _slugify(display_name)
        department = (
            record.department
            if record and record.department
            else csv_entry.get("department")
            or ""
        )
        summary_text = record.summary if record else ""
        info_text = _staff_info_text(info_entry)
        issues: list[dict[str, Any]] = []

        if summary_text and info_text:
            summary_tokens = set(_tokenize(summary_text))
            info_tokens = set(_tokenize(info_text))
            lexical = _jaccard(summary_tokens, info_tokens)
            semantic_lite = _cosine_similarity(
                _char_ngram_counter(summary_text),
                _char_ngram_counter(info_text),
            )
            divergence_score = 0.65 * lexical + 0.35 * semantic_lite
            severity = _severity_for_divergence(divergence_score)
            if severity:
                missing_profile_terms = sorted(list(info_tokens - summary_tokens))[:10]
                issues.append(
                    _build_issue(
                        check="staff_info_vs_profile_divergence",
                        severity=severity,
                        message=(
                            "staff_info content diverges from profile summary "
                            f"(score={divergence_score:.3f})."
                        ),
                        metrics={
                            "divergence_score": round(divergence_score, 4),
                            "lexical_similarity": round(lexical, 4),
                            "semantic_lite_similarity": round(semantic_lite, 4),
                            "sample_staffinfo_only_terms": missing_profile_terms,
                        },
                    )
                )

        if nva_entries:
            domain_text = " ".join([department, summary_text, info_text])
            domain_tokens = set(_tokenize(domain_text))
            aligned = 0
            medical = 0
            publication_scores: list[dict[str, Any]] = []
            for entry in nva_entries:
                pub_text = " ".join(
                    [
                        str(entry.get("title") or ""),
                        str(entry.get("abstract") or ""),
                    ]
                )
                pub_tokens = set(_tokenize(pub_text))
                if not pub_tokens:
                    continue
                overlap = len(domain_tokens & pub_tokens) / max(1, len(pub_tokens))
                publication_scores.append(
                    {
                        "title": str(entry.get("title") or "").strip(),
                        "overlap_ratio": round(overlap, 4),
                    }
                )
                if overlap >= 0.04:
                    aligned += 1
                if MEDICAL_TOKENS & pub_tokens:
                    medical += 1

            total_publications = len(publication_scores)
            if total_publications:
                alignment_rate = aligned / total_publications
                medical_share = medical / total_publications
                severity = _severity_for_alignment(
                    total_publications=total_publications,
                    alignment_rate=alignment_rate,
                    medical_share=medical_share,
                    is_health_department=_is_health_department(department),
                )
                if severity:
                    sample_low_overlap = [
                        item["title"]
                        for item in sorted(
                            publication_scores, key=lambda value: value["overlap_ratio"]
                        )[:5]
                        if item["title"]
                    ]
                    issues.append(
                        _build_issue(
                            check="nva_domain_outlier",
                            severity=severity,
                            message=(
                                "NVA publications weakly align with profile/department terms "
                                f"(alignment_rate={alignment_rate:.2f})."
                            ),
                            metrics={
                                "total_publications": total_publications,
                                "aligned_publications": aligned,
                                "alignment_rate": round(alignment_rate, 4),
                                "medical_publication_share": round(medical_share, 4),
                                "sample_low_overlap_titles": sample_low_overlap,
                            },
                        )
                    )

        nva_profile_url = str(csv_entry.get("nva_profile_url") or "").strip()
        if nva_profile_url and not nva_entries:
            issues.append(
                _build_issue(
                    check="nva_presence_mismatch",
                    severity="medium",
                    message="staff.csv has NVA profile but no local NVA results.",
                    metrics={"nva_profile_url": nva_profile_url},
                )
            )

        collision_details = chunk_collisions.get(slug)
        if collision_details:
            issues.append(
                _build_issue(
                    check="chunk_id_collision",
                    severity="high",
                    message="Duplicate chunk_id values detected in chunk snapshot.",
                    metrics=collision_details,
                )
            )

        if issues:
            staff_reports.append(
                {
                    "name": display_name,
                    "slug": slug,
                    "department": department,
                    "nva_publication_count": len(nva_entries),
                    "issues": issues,
                }
            )

    by_name = {entry["name"].lower(): entry for entry in staff_reports}
    for seed in KNOWN_PRIORITY_SEEDS:
        seeded_name = seed["name"]
        seeded_key = seeded_name.lower()
        target = by_name.get(seeded_key)
        if target is None:
            target = {
                "name": seeded_name,
                "slug": _slugify(seeded_name),
                "department": "",
                "nva_publication_count": len(nva_by_name.get(seeded_key, [])),
                "issues": [],
            }
            staff_reports.append(target)
            by_name[seeded_key] = target
        already_present = any(
            issue.get("check") == seed["check"] for issue in target["issues"]
        )
        if not already_present:
            target["issues"].append(
                _build_issue(
                    check=seed["check"],
                    severity="high",
                    message=seed["message"],
                    metrics={"seeded": True},
                )
            )

    severity_counts = {"high": 0, "medium": 0, "low": 0}
    unresolved_high = 0
    for staff_entry in staff_reports:
        for issue in staff_entry["issues"]:
            severity = str(issue.get("severity") or "").lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
            if severity == "high" and str(issue.get("status") or "open") != "resolved":
                unresolved_high += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "staff_with_issues": len(staff_reports),
            "issue_counts": severity_counts,
            "unresolved_high_count": unresolved_high,
        },
        "staff": sorted(staff_reports, key=lambda item: item["name"].lower()),
    }


def _build_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    issue_counts = summary.get("issue_counts", {})
    lines: list[str] = [
        "# Staff Data Audit",
        "",
        f"- Generated at: {report.get('generated_at', '')}",
        f"- Staff with issues: {summary.get('staff_with_issues', 0)}",
        f"- High issues: {issue_counts.get('high', 0)}",
        f"- Medium issues: {issue_counts.get('medium', 0)}",
        f"- Low issues: {issue_counts.get('low', 0)}",
        f"- Unresolved high issues: {summary.get('unresolved_high_count', 0)}",
        "",
        "## High Severity",
        "",
    ]

    high_rows: list[str] = []
    for staff_entry in report.get("staff", []):
        for issue in staff_entry.get("issues", []):
            if issue.get("severity") == "high":
                high_rows.append(
                    f"- {staff_entry.get('name')} | {issue.get('check')}: {issue.get('message')}"
                )
    if high_rows:
        lines.extend(high_rows)
    else:
        lines.append("- None")

    lines.extend(["", "## Staff Issues", ""])
    for staff_entry in report.get("staff", []):
        lines.append(
            f"### {staff_entry.get('name')} ({staff_entry.get('department') or 'Unknown'})"
        )
        lines.append("")
        for issue in staff_entry.get("issues", []):
            lines.append(
                f"- [{issue.get('severity')}] {issue.get('check')}: {issue.get('message')}"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _write_report(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit staff data consistency across staff_info/profile/NVA/chunks."
    )
    parser.add_argument("--staff-info", type=Path, default=Path("staff_info.json"))
    parser.add_argument(
        "--staff-records", type=Path, default=Path("data/staff_records.jsonl")
    )
    parser.add_argument("--staff-csv", type=Path, default=Path("staff.csv"))
    parser.add_argument(
        "--nva-results", type=Path, default=Path("data/nva/results.jsonl")
    )
    parser.add_argument(
        "--chunks-dir", type=Path, default=Path("data/index/chunks")
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("reports/staff_data_audit.json")
    )
    parser.add_argument(
        "--output-md", type=Path, default=Path("reports/staff_data_audit.md")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_audit(
        staff_info_path=args.staff_info,
        staff_records_path=args.staff_records,
        staff_csv_path=args.staff_csv,
        nva_results_path=args.nva_results,
        chunks_dir=args.chunks_dir,
    )
    _write_report(
        args.output_json,
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    _write_report(args.output_md, _build_markdown(report))
    print(f"Wrote {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
