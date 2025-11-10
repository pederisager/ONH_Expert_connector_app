#!/usr/bin/env python3
"""Offline profiling harness for the /match pipeline.

The script avoids spinning up FastAPI and instead invokes the same helper
functions that power /match (document fetch/merge, TF-IDF/embedding scoring,
and LLM explanations). All heavy stages are instrumented so we can see where
CPU time goes when "Finn relevante ansatte" feels sluggish.

Usage examples:

    python scripts/profile_match.py psykologisk traumatologi --department Helsefag

    # Profile five runs against 25 staff, forcing TF-IDF and stubbing the LLM
    python scripts/profile_match.py "klinisk psykologi" "traumebehandling" \\
        --runs 5 --staff-limit 25 --force-tfidf --stub-llm

All profiling happens offline by default (HTML snapshots or synthetic text are
used whenever network access is unavailable)."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import types
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Sequence

if TYPE_CHECKING:  # pragma: no cover - for editors/Mypy only
    from app import routes as routes_module
    from app.llm_explainer import LLMExplainer
    from app.match_engine import MatchEngine


StageFunc = Callable[..., Any]
AsyncStageFunc = Callable[..., Awaitable[Any]]


class StageTimer:
    """Collects duration samples for labelled stages."""

    def __init__(self) -> None:
        self.samples: Dict[str, List[float]] = {}

    def record(self, label: str, duration: float) -> None:
        bucket = self.samples.setdefault(label, [])
        bucket.append(duration)

    def summary(self) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for label, values in sorted(self.samples.items()):
            stats[label] = {
                "count": len(values),
                "mean_ms": statistics.mean(values) * 1000.0,
                "median_ms": statistics.median(values) * 1000.0,
                "max_ms": max(values) * 1000.0,
            }
        return stats


def _wrap_sync(label: str, timer: StageTimer, func: StageFunc) -> StageFunc:
    def wrapper(*args, **kwargs):
        start = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            timer.record(label, perf_counter() - start)

    return wrapper


def _wrap_async(label: str, timer: StageTimer, func: AsyncStageFunc) -> AsyncStageFunc:
    async def wrapper(*args, **kwargs):
        start = perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            timer.record(label, perf_counter() - start)

    return wrapper


def _patch_instrumentation(
    timer: StageTimer,
    routes_module,
    match_engine_cls,
    llm_explainer_cls,
) -> None:
    """Monkey-patch hot stages to capture per-run timings."""

    match_engine_cls._tfidf_similarity = _wrap_sync(  # type: ignore[method-assign]
        "tfidf_vectorization", timer, match_engine_cls._tfidf_similarity
    )
    match_engine_cls._embedding_similarity_via_sentence_transformers = _wrap_sync(  # type: ignore[method-assign]
        "embedding_generation",
        timer,
        match_engine_cls._embedding_similarity_via_sentence_transformers,
    )
    routes_module._fetch_staff_documents = _wrap_async(  # type: ignore[attr-defined]
        "cristin_nva_merge", timer, routes_module._fetch_staff_documents
    )
    llm_explainer_cls.generate = _wrap_async(  # type: ignore[method-assign]
        "llm_explainer", timer, llm_explainer_cls.generate
    )


def _stub_sentence_transformer_backend() -> None:
    from app.index import embedder_factory
    from app.index import embeddings
    from app.index.builder import DummyEmbeddingBackend

    class _StubBackend:
        def __init__(self, *_, **__):
            self._dummy = DummyEmbeddingBackend()

        def embed(self, texts):
            return self._dummy.embed(texts)

        def embed_one(self, text):
            return self._dummy.embed_one(text)

    embedder_factory.SentenceTransformerBackend = _StubBackend  # type: ignore[attr-defined]
    embeddings.SentenceTransformerBackend = _StubBackend  # type: ignore[attr-defined]


def _enable_offline_fetch(fetch_utils, stub_text: str) -> None:
    original = fetch_utils.fetch_page

    async def offline_fetch(self, client_http, url):  # type: ignore[override]
        snapshot = self._load_offline_snapshot(url)
        if snapshot:
            return snapshot
        return {
            "url": url,
            "title": "Offline stub",
            "text": stub_text.format(url=url),
        }

    fetch_utils.fetch_page = types.MethodType(offline_fetch, fetch_utils)
    return original


def _stub_llm_explainer(llm_explainer) -> None:
    async def fake_generate(self, staff_name, snippets, themes):  # type: ignore[override]
        await asyncio.sleep(0)
        joined = ", ".join(themes) or "temaet"
        return f"{staff_name} matcher {joined} (stub)."

    llm_explainer.generate = types.MethodType(fake_generate, llm_explainer)


async def _profile_once(
    *,
    themes: list[str],
    department: str | None,
    max_candidates: int,
    diversity_weight: float,
    timer: StageTimer,
    staff_profiles,
    fetch_utils,
    cache_manager,
    match_engine,
    app_config,
    llm_explainer,
    routes_module,
    memory_cache,
) -> int:
    documents = await routes_module._fetch_staff_documents(  # type: ignore[attr-defined]
        staff_profiles=staff_profiles,
        fetch_utils=fetch_utils,
        cache_manager=cache_manager,
        max_pages=app_config.fetch.max_pages_per_staff,
        memory_cache=memory_cache,
    )
    if not documents:
        return 0

    ranked = match_engine.rank(
        documents=documents,
        themes=themes,
        department_filter=department,
        max_candidates=max_candidates,
        diversity_weight=diversity_weight,
    )
    if not ranked:
        return 0

    snippets_per_staff: list[tuple[str, list[str]]] = []
    for result in ranked:
        snippets = [citation["snippet"] for citation in result.citations if citation.get("snippet")]
        if not snippets:
            continue
        snippets_per_staff.append((result.staff.name, snippets))

    if snippets_per_staff:
        tasks = [llm_explainer.generate(name, snippets, themes) for name, snippets in snippets_per_staff]
        await asyncio.gather(*tasks, return_exceptions=True)

    return len(ranked)


async def run_profile(args) -> dict[str, Any]:
    if args.stub_st_backend:
        _stub_sentence_transformer_backend()

    from app import routes
    from app.llm_explainer import LLMExplainer
    from app.main import create_app
    from app.match_engine import MatchEngine

    app = create_app()
    timer = StageTimer()
    _patch_instrumentation(timer, routes, MatchEngine, LLMExplainer)

    if args.disable_rag:
        app.state.vector_index_ready = False
        app.state.embedding_retriever = None
    if args.force_tfidf:
        engine: MatchEngine = app.state.match_engine
        engine.embedding_backend = None
        engine._embedder = None  # type: ignore[attr-defined]
        engine._embedder_attempted = True  # type: ignore[attr-defined]
    if args.staff_limit is not None and args.staff_limit > 0:
        app.state.staff_profiles = app.state.staff_profiles[: args.staff_limit]

    fetch_utils = app.state.fetch_utils
    if not args.allow_network:
        _enable_offline_fetch(fetch_utils, stub_text="Caching utilgjengelig for {url}")
    if args.stub_llm:
        _stub_llm_explainer(app.state.llm_explainer)

    await routes.warm_staff_document_cache(app.state)

    results: list[int] = []
    total_times: list[float] = []

    for _ in range(args.runs):
        start = perf_counter()
        count = await _profile_once(
            themes=list(args.themes),
            department=args.department,
            max_candidates=app.state.app_config.results.max_candidates,
            diversity_weight=app.state.app_config.results.diversity_weight,
            timer=timer,
            staff_profiles=app.state.staff_profiles,
            fetch_utils=fetch_utils,
            cache_manager=app.state.cache_manager,
            match_engine=app.state.match_engine,
            app_config=app.state.app_config,
            llm_explainer=app.state.llm_explainer,
            routes_module=routes,
            memory_cache=getattr(app.state, "staff_document_cache", None),
        )
        total_times.append(perf_counter() - start)
        results.append(count)

    return {
        "results": results,
        "total_request_ms": [value * 1000.0 for value in total_times],
        "stages": timer.summary(),
    }


def format_report(report: dict[str, Any]) -> str:
    total = report["total_request_ms"]
    total_line = (
        f"Total pipeline: mean={statistics.mean(total):.1f} ms, "
        f"median={statistics.median(total):.1f} ms, max={max(total):.1f} ms"
    )
    lines = [total_line]
    for label, stats in report["stages"].items():
        lines.append(
            f"- {label}: mean={stats['mean_ms']:.1f} ms, median={stats['median_ms']:.1f} ms, "
            f"max={stats['max_ms']:.1f} ms (n={stats['count']})"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile the ONH /match pipeline offline.")
    parser.add_argument("themes", nargs="*", default=["klinisk psykologi", "traumebehandling"], help="Themes to query.")
    parser.add_argument("--department", default=None, help="Optional department filter.")
    parser.add_argument("--runs", type=int, default=3, help="Number of profiling iterations (default: 3).")
    parser.add_argument("--disable-rag", action="store_true", help="Skip vector-store retrieval to mirror fallback mode.")
    parser.add_argument("--force-tfidf", action="store_true", help="Bypass embeddings inside MatchEngine and rely on TF-IDF.")
    parser.add_argument("--stub-st-backend", action="store_true", help="Replace SentenceTransformer backend with a dummy embedder.")
    parser.add_argument("--stub-llm", action="store_true", help="Bypass the LLM explainer with a deterministic stub.")
    parser.add_argument("--staff-limit", type=int, default=20, help="Restrict staff sample size (default: 20).")
    parser.add_argument("--allow-network", action="store_true", help="Permit live HTTP fetches (defaults to offline snapshots/stubs).")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to dump raw JSON timings.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    report = asyncio.run(run_profile(args))
    print(format_report(report))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
