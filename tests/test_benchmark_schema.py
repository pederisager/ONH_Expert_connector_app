from __future__ import annotations

from pathlib import Path

import yaml


BENCHMARK_PATHS = [
    Path("tests/benchmarks/search_relevance_pilot_v1.yaml"),
    Path("tests/benchmarks/search_relevance_100_v1.yaml"),
    Path("tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml"),
]
PUBLICATION_BENCHMARK_PATHS = [
    Path("tests/benchmarks/search_relevance_pilot_v1.yaml"),
    Path("tests/benchmarks/search_relevance_100_v1.yaml"),
]


def test_search_benchmark_dual_mode_schema() -> None:
    valid_modes = {"publication_grounded", "profile_grounded"}
    for path in BENCHMARK_PATHS:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        queries = payload.get("queries") or []
        assert queries, f"{path} must include queries."

        for query in queries:
            assert query.get("query_mode") in valid_modes
            required = query.get("required_source_kinds")
            assert isinstance(required, list)
            assert required
            overlap_rules = query.get("citation_overlap_rules") or {}
            assert "min_query_term_overlap_per_citation" in overlap_rules


def test_publication_grounded_queries_require_nva_source() -> None:
    for path in PUBLICATION_BENCHMARK_PATHS:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        queries = payload.get("queries") or []

        publication_queries = [
            query for query in queries if query.get("query_mode") == "publication_grounded"
        ]
        assert publication_queries
        assert all("nva" in (query.get("required_source_kinds") or []) for query in publication_queries)


def test_search_relevance_100_coverage_targets() -> None:
    path = Path("tests/benchmarks/search_relevance_100_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    queries = payload.get("queries") or []
    assert len(queries) == 100

    by_mode = {"publication_grounded": 0, "profile_grounded": 0}
    by_difficulty: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    for query in queries:
        mode = str(query.get("query_mode"))
        by_mode[mode] = by_mode.get(mode, 0) + 1
        difficulty = str(query.get("difficulty") or "unknown")
        by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1
        domain = str(query.get("domain") or "unknown")
        by_domain[domain] = by_domain.get(domain, 0) + 1

    assert by_mode["publication_grounded"] == 60
    assert by_mode["profile_grounded"] == 40
    assert by_difficulty == {"easy": 35, "medium": 35, "hard": 20, "negative": 10}
    assert all(count == 25 for count in by_domain.values())
