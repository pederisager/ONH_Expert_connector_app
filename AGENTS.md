# Repository Guidelines

## Purpose
This file is the minimum project-specific guidance for agents working in this repo.

## Non-Negotiable Runtime Safety
1. Wrap long-running commands with an explicit timeout.
2. On PowerShell, always use:
   - `Set-ExecutionPolicy -Scope Process Bypass -Force`
   - `scripts/run_with_timeout.ps1 -TimeoutSec <sec> ...`
3. If a required input/artifact is missing, mark task `blocked` in `docs/TODO.md` and stop pretending completion.

## Queue Discipline
1. `docs/TODO.md` is the canonical autonomous work queue.
2. If instructed to work on tasks listed in `docs/TODO.md`, focus on one primary task per run.
3. Update `docs/TODO.md` in the same run with:
   - status transition
   - bounded command log
   - output paths
   - decision (`keep`/`revise`/`revert`)
   - next task id

## Artifact Hygiene (Do Not Commit)
- `reports/model_sweeps/`
- `reports/relevance_tuning/`
- `reports/worker_*`
- `app/**/__pycache__/`
- `data/cache/`

Commit source code, tests, config, and stable benchmark definitions only.

## Product/Domain Guardrails
- Keep shortlist feature removed (no shortlist UI/API reintroduction).
- File uploads remain disabled; keep `/analyze-topic` text-only unless explicitly requested.
- `staff.csv` is the entry point for staff data refresh; propagate via `scripts/update_staff.sh`.
- Precomputed summaries are loaded from `data/precomputed_summaries.json`; do not add online LLM summary generation in `/match`.

## Search Quality Workflow
- Strict benchmark: `tests/benchmarks/search_relevance_100_v1.yaml`
- User-test benchmark: `tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml`
- Rebuild user64 benchmark file from CSV source:
  - `python3 scripts/build_user64_benchmark.py`

When editing retrieval/scoring (`app/routes.py`, `app/rag/retriever.py`, `data/app.config.yaml`), run targeted route/retriever tests and at least one bounded benchmark run.

## Useful Commands
- Tests: `python3 -m pytest`
- Targeted retrieval tests: `python3 -m pytest tests/test_routes.py tests/test_retriever.py tests/test_config_loader.py`
- Strict benchmark: `python3 scripts/run_search_benchmark.py --benchmark tests/benchmarks/search_relevance_100_v1.yaml --base-url http://127.0.0.1:8000 --output reports/benchmark_results_100_latest.json`

## Agent Maintenance
If you change operational workflow/policy for future agents, update this file in the same task.
