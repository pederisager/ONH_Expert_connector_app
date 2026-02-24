from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def _split_names(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split("|") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build user64 benchmark YAML from expected-vs-actual CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("reports/query_test_expected_vs_actual.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input_csv.exists():
        raise SystemExit(f"Missing input CSV: {args.input_csv}")

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    queries: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: str(item.get("query_id") or "")):
        query_mode = str(row.get("query_mode") or "profile_grounded").strip() or "profile_grounded"
        required_source_kinds = (
            ["nva"] if query_mode == "publication_grounded" else ["profile", "staffinfo"]
        )
        queries.append(
            {
                "id": str(row.get("query_id") or "").strip(),
                "domain": str(row.get("domain") or "unknown").strip() or "unknown",
                "difficulty": str(row.get("difficulty") or "unknown").strip() or "unknown",
                "query_text": str(row.get("query_text") or "").strip(),
                "query_mode": query_mode,
                "required_source_kinds": required_source_kinds,
                "citation_overlap_rules": {
                    "min_query_term_overlap_per_citation": 1,
                },
                "expected": {
                    "must_include": _split_names(str(row.get("expected_must_include") or "")),
                    "should_include": _split_names(str(row.get("expected_should_include") or "")),
                    "hard_exclude": _split_names(str(row.get("expected_hard_exclude") or "")),
                },
            }
        )

    payload: dict[str, object] = {
        "version": "search_relevance_chatgpt_user64_v1",
        "description": (
            "64-query benchmark derived from the ChatGPT user-test empirical table. "
            "Use for ranking/regression checks against the user-test query set."
        ),
        "source_snapshot": {
            "source_csv": str(args.input_csv).replace("\\", "/"),
        },
        "global_rules": {
            "default_top_k": 10,
            "default_min_query_term_overlap_per_citation": 1,
            "require_citation_for_must_include": True,
        },
        "queries": queries,
        "overexposure_controls": [],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(args.output).replace("\\", "/"),
                "query_count": len(queries),
                "input_csv": str(args.input_csv).replace("\\", "/"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
