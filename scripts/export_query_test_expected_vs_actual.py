from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def _join(values: list[str]) -> str:
    return " | ".join(v.strip() for v in values if v and v.strip())


def _as_bool_text(value: Any) -> str:
    if value is None:
        return ""
    return "true" if bool(value) else "false"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export benchmark expected-vs-actual comparison CSV for user64 queries."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/benchmark_results_user64_baseline.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/query_test_expected_vs_actual.csv"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    benchmark = yaml.safe_load(args.benchmark.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))

    benchmark_queries = {
        str(query.get("id")): query for query in (benchmark.get("queries") or [])
    }
    report_results = {
        str(result.get("query_id")): result for result in (report.get("query_results") or [])
    }
    report_failures = {
        str(item.get("query_id")): item for item in (report.get("query_failures") or [])
    }

    ordered_ids = sorted(benchmark_queries.keys())

    rows: list[dict[str, str]] = []
    for query_id in ordered_ids:
        benchmark_query = benchmark_queries[query_id]
        expected = benchmark_query.get("expected") or {}
        result = report_results.get(query_id, {})
        failure = report_failures.get(query_id, {})

        rows.append(
            {
                "query_id": query_id,
                "query_text": str(benchmark_query.get("query_text") or ""),
                "domain": str(benchmark_query.get("domain") or ""),
                "difficulty": str(benchmark_query.get("difficulty") or ""),
                "query_mode": str(benchmark_query.get("query_mode") or ""),
                "expected_must_include": _join(
                    [str(v) for v in (expected.get("must_include") or [])]
                ),
                "expected_should_include": _join(
                    [str(v) for v in (expected.get("should_include") or [])]
                ),
                "expected_hard_exclude": _join(
                    [str(v) for v in (expected.get("hard_exclude") or [])]
                ),
                "actual_must_include_pass_top3": _as_bool_text(
                    result.get("must_include_pass")
                ),
                "actual_should_hits_top10": str(int(result.get("should_hits") or 0)),
                "actual_should_total": str(int(result.get("should_total") or 0)),
                "actual_hard_exclude_hits_top10": str(
                    int(result.get("hard_exclude_hits") or 0)
                ),
                "actual_hard_exclude_total": str(
                    int(result.get("hard_exclude_total") or 0)
                ),
                "actual_publication_evidence_pass": _as_bool_text(
                    result.get("publication_evidence_pass")
                ),
                "request_error": str(result.get("request_error") or ""),
                "failure_reasons": _join(
                    [str(v) for v in (failure.get("reasons") or [])]
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "output": str(args.output).replace("\\", "/"),
                "rows": len(rows),
                "benchmark": str(args.benchmark).replace("\\", "/"),
                "report": str(args.report).replace("\\", "/"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
