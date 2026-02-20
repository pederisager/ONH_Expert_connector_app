"""Run retrieval benchmark queries against a live ONH Expert Connector instance."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml

TOKEN_RE = re.compile(r"[A-Za-zÆØÅæøå]{3,}")


@dataclass(slots=True)
class QueryOutcome:
    query_id: str
    query_text: str
    domain: str
    query_mode: str
    difficulty: str
    must_include_pass: bool
    publication_evidence_pass: bool | None
    should_hits: int
    should_total: int
    hard_exclude_hits: int
    hard_exclude_total: int
    top10_names: list[str]


def _tokenize(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(value or "") if token}


def _normalize_name(value: str) -> str:
    return " ".join((value or "").split()).strip().lower()


def _citation_source_kind(citation: dict[str, Any]) -> str:
    url = str(citation.get("url") or "").lower()
    if "doi.org" in url or "nva.sikt.no/registration/" in url:
        return "nva"
    if url.startswith("staffinfo://"):
        return "staffinfo"
    return "profile"


def _citation_overlap(citation: dict[str, Any], query_text: str) -> int:
    return len(_tokenize(str(citation.get("snippet") or "")) & _tokenize(query_text))


def _get_query_themes(client: httpx.Client, base_url: str, query_text: str) -> list[str]:
    response = client.post(
        f"{base_url}/analyze-topic",
        json={"text": query_text},
        timeout=30.0,
    )
    response.raise_for_status()
    payload = response.json()
    themes = payload.get("themes") or []
    if isinstance(themes, list) and themes:
        return [str(theme) for theme in themes if str(theme).strip()]
    return [query_text]


def _run_single_query(
    client: httpx.Client,
    base_url: str,
    query: dict[str, Any],
    global_rules: dict[str, Any],
) -> QueryOutcome:
    query_id = str(query.get("id"))
    query_text = str(query.get("query_text") or "")
    domain = str(query.get("domain") or "unknown")
    query_mode = str(query.get("query_mode") or "profile_grounded")
    difficulty = str(query.get("difficulty") or "unknown")
    expected = query.get("expected") or {}

    must_include = [str(item) for item in expected.get("must_include") or []]
    should_include = [str(item) for item in expected.get("should_include") or []]
    hard_exclude = [str(item) for item in expected.get("hard_exclude") or []]

    themes = _get_query_themes(client, base_url, query_text)
    response = client.post(
        f"{base_url}/match",
        json={"themes": themes, "mode": query_mode},
        timeout=60.0,
    )
    response.raise_for_status()
    payload = response.json()
    results = list(payload.get("results") or [])
    top10 = results[: int(global_rules.get("default_top_k", 10))]
    top3 = results[:3]

    top10_names = [_normalize_name(result.get("name") or "") for result in top10]
    top3_names = {_normalize_name(result.get("name") or "") for result in top3}

    must_include_pass = all(_normalize_name(name) in top3_names for name in must_include)

    should_hits = sum(
        1 for name in should_include if _normalize_name(name) in set(top10_names)
    )
    hard_exclude_hits = sum(
        1 for name in hard_exclude if _normalize_name(name) not in set(top10_names)
    )

    publication_evidence_pass: bool | None = None
    if query_mode == "publication_grounded":
        required_sources = {
            str(item).strip().lower() for item in (query.get("required_source_kinds") or [])
        }
        if not required_sources:
            required_sources = {"nva"}
        overlap_rules = query.get("citation_overlap_rules") or {}
        min_overlap = int(
            overlap_rules.get(
                "min_query_term_overlap_per_citation",
                global_rules.get("default_min_query_term_overlap_per_citation", 1),
            )
        )
        publication_evidence_pass = True
        result_by_name = {
            _normalize_name(result.get("name") or ""): result for result in top10
        }
        for staff_name in must_include:
            result = result_by_name.get(_normalize_name(staff_name))
            if result is None:
                publication_evidence_pass = False
                continue
            citations = list(result.get("citations") or [])
            valid = False
            for citation in citations:
                citation_source = _citation_source_kind(citation)
                if citation_source not in required_sources:
                    continue
                if _citation_overlap(citation, query_text) >= min_overlap:
                    valid = True
                    break
            if not valid:
                publication_evidence_pass = False

    return QueryOutcome(
        query_id=query_id,
        query_text=query_text,
        domain=domain,
        query_mode=query_mode,
        difficulty=difficulty,
        must_include_pass=must_include_pass,
        publication_evidence_pass=publication_evidence_pass,
        should_hits=should_hits,
        should_total=len(should_include),
        hard_exclude_hits=hard_exclude_hits,
        hard_exclude_total=len(hard_exclude),
        top10_names=[name for name in top10_names if name],
    )


def _compute_metrics(outcomes: list[QueryOutcome]) -> dict[str, float]:
    must_include_total = len(outcomes)
    must_include_hits = sum(1 for item in outcomes if item.must_include_pass)

    should_total = sum(item.should_total for item in outcomes)
    should_hits = sum(item.should_hits for item in outcomes)

    hard_total = sum(item.hard_exclude_total for item in outcomes)
    hard_hits = sum(item.hard_exclude_hits for item in outcomes)

    publication_items = [
        item for item in outcomes if item.publication_evidence_pass is not None
    ]
    publication_total = len(publication_items)
    publication_hits = sum(
        1 for item in publication_items if item.publication_evidence_pass
    )

    return {
        "MustInclude@3": must_include_hits / must_include_total if must_include_total else 0.0,
        "ShouldInclude@10": should_hits / should_total if should_total else 0.0,
        "HardExcludeRate@10": hard_hits / hard_total if hard_total else 0.0,
        "PublicationEvidencePassRate": (
            publication_hits / publication_total if publication_total else 0.0
        ),
    }


def _compute_metrics_by_key(
    outcomes: list[QueryOutcome], key_fn: Callable[[QueryOutcome], str]
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[QueryOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[str(key_fn(outcome))].append(outcome)
    metrics_by_key: dict[str, dict[str, float]] = {}
    for key, key_outcomes in grouped.items():
        key_metrics = _compute_metrics(key_outcomes)
        key_metrics["QueryCount"] = float(len(key_outcomes))
        metrics_by_key[key] = key_metrics
    return metrics_by_key


def _evaluate_thresholds(
    metrics: dict[str, float], thresholds: dict[str, Any], prefix: str = ""
) -> list[str]:
    failures: list[str] = []
    threshold_map = {
        "must_include_at_3_min": "MustInclude@3",
        "should_include_at_10_min": "ShouldInclude@10",
        "hard_exclude_rate_at_10_min_min": "HardExcludeRate@10",
        "publication_evidence_pass_rate_min": "PublicationEvidencePassRate",
    }
    for threshold_key, metric_key in threshold_map.items():
        threshold_value = thresholds.get(threshold_key)
        if threshold_value is None:
            continue
        actual = float(metrics.get(metric_key, 0.0))
        expected = float(threshold_value)
        if actual < expected:
            label = f"{prefix}{metric_key}" if prefix else metric_key
            failures.append(f"{label}={actual:.3f} below threshold {expected:.3f}")
    return failures


def _build_query_failure_reasons(outcome: QueryOutcome) -> list[str]:
    reasons: list[str] = []
    if not outcome.must_include_pass:
        reasons.append("must_include_missing_top3")
    if outcome.publication_evidence_pass is False:
        reasons.append("publication_evidence_missing")
    should_misses = max(0, outcome.should_total - outcome.should_hits)
    if should_misses:
        reasons.append(f"should_include_missed={should_misses}/{outcome.should_total}")
    hard_violations = max(0, outcome.hard_exclude_total - outcome.hard_exclude_hits)
    if hard_violations:
        reasons.append(f"hard_exclude_violations={hard_violations}/{outcome.hard_exclude_total}")
    return reasons


def _evaluate_overexposure(
    outcomes: list[QueryOutcome], controls: list[dict[str, Any]]
) -> list[str]:
    by_query = {outcome.query_id: outcome for outcome in outcomes}
    violations: list[str] = []
    for control in controls:
        staff = str(control.get("staff") or "").strip()
        if not staff:
            continue
        staff_norm = _normalize_name(staff)
        in_top10 = 0
        in_top3 = 0
        for outcome in outcomes:
            names = outcome.top10_names
            if staff_norm in names:
                in_top10 += 1
            if staff_norm in names[:3]:
                in_top3 += 1
        max_top10 = int(control.get("max_queries_in_top_10", 9999))
        max_top3 = int(control.get("max_queries_in_top_3", 9999))
        if in_top10 > max_top10:
            violations.append(
                f"{staff} appears in top10 for {in_top10} queries (max {max_top10})."
            )
        if in_top3 > max_top3:
            violations.append(
                f"{staff} appears in top3 for {in_top3} queries (max {max_top3})."
            )

        for query_id in control.get("required_absent_query_ids") or []:
            outcome = by_query.get(str(query_id))
            if not outcome:
                continue
            if staff_norm in outcome.top10_names:
                violations.append(
                    f"{staff} present in top10 for required-absent query {query_id}."
                )
    return violations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run search relevance benchmark.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("tests/benchmarks/search_relevance_pilot_v1.yaml"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark_results.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark = yaml.safe_load(args.benchmark.read_text(encoding="utf-8"))
    queries = list(benchmark.get("queries") or [])
    global_rules = dict(benchmark.get("global_rules") or {})

    outcomes: list[QueryOutcome] = []
    with httpx.Client(follow_redirects=True) as client:
        for query in queries:
            outcomes.append(
                _run_single_query(
                    client=client,
                    base_url=args.base_url.rstrip("/"),
                    query=query,
                    global_rules=global_rules,
                )
            )

    metrics = _compute_metrics(outcomes)
    metrics_by_domain = _compute_metrics_by_key(outcomes, key_fn=lambda item: item.domain)
    metrics_by_mode = _compute_metrics_by_key(outcomes, key_fn=lambda item: item.query_mode)
    overexposure_violations = _evaluate_overexposure(
        outcomes,
        list(benchmark.get("overexposure_controls") or []),
    )

    threshold_failures = _evaluate_thresholds(
        metrics=metrics,
        thresholds=dict(benchmark.get("regression_thresholds") or {}),
    )
    mode_threshold_failures: list[str] = []
    mode_thresholds = benchmark.get("mode_regression_thresholds") or {}
    for mode, mode_threshold in mode_thresholds.items():
        mode_metrics = metrics_by_mode.get(str(mode))
        if mode_metrics is None:
            continue
        mode_threshold_failures.extend(
            _evaluate_thresholds(
                metrics=mode_metrics,
                thresholds=dict(mode_threshold or {}),
                prefix=f"{mode}:",
            )
        )

    query_failures = []
    for item in outcomes:
        reasons = _build_query_failure_reasons(item)
        if not reasons:
            continue
        query_failures.append(
            {
                "query_id": item.query_id,
                "query_text": item.query_text,
                "domain": item.domain,
                "query_mode": item.query_mode,
                "difficulty": item.difficulty,
                "failure_score": len(reasons),
                "reasons": reasons,
            }
        )
    query_failures.sort(
        key=lambda row: (-int(row["failure_score"]), str(row["query_id"]))
    )

    output_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_version": benchmark.get("version"),
        "metrics": metrics,
        "metrics_by_domain": metrics_by_domain,
        "metrics_by_mode": metrics_by_mode,
        "threshold_failures": threshold_failures,
        "mode_threshold_failures": mode_threshold_failures,
        "overexposure_violations": overexposure_violations,
        "query_failures": query_failures,
        "query_results": [
            {
                "query_id": item.query_id,
                "query_text": item.query_text,
                "domain": item.domain,
                "query_mode": item.query_mode,
                "difficulty": item.difficulty,
                "must_include_pass": item.must_include_pass,
                "publication_evidence_pass": item.publication_evidence_pass,
                "should_hits": item.should_hits,
                "should_total": item.should_total,
                "hard_exclude_hits": item.hard_exclude_hits,
                "hard_exclude_total": item.hard_exclude_total,
            }
            for item in outcomes
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "overall_metrics": output_payload["metrics"],
                "metrics_by_mode": output_payload["metrics_by_mode"],
                "metrics_by_domain": output_payload["metrics_by_domain"],
            },
            ensure_ascii=False,
        )
    )
    for failure in query_failures:
        print(
            "FAIL_QUERY "
            f"{failure['query_id']} [{failure['domain']}|{failure['query_mode']}|{failure['difficulty']}] "
            + "; ".join(str(reason) for reason in failure["reasons"])
        )

    if threshold_failures or mode_threshold_failures or overexposure_violations:
        for failure in threshold_failures:
            print(f"FAIL: {failure}")
        for failure in mode_threshold_failures:
            print(f"FAIL: {failure}")
        for violation in overexposure_violations:
            print(f"FAIL: {violation}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
