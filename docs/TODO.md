# TODO (Autonomous Search Queue)

Last updated: 2026-02-24
Queue state: `active`
Next task id: `SQ-UT-011`
Canonical queue file: `docs/TODO.md`

## Operating Contract
This file is the execution contract for unattended agent runs.

1. Work exactly one primary task per run.
2. Set that task status to `in_progress` before edits.
3. Use bounded commands only:
   - Windows/PowerShell: `Set-ExecutionPolicy -Scope Process Bypass -Force; .\\scripts\\run_with_timeout.ps1 -TimeoutSec <sec> ...`
   - Linux/WSL: `timeout <sec> ...`
4. Do not mark a task `completed` unless its acceptance criteria are met with reproducible artifacts.
5. Update this file in the same run (status, command log, outputs, decision, next task id).

## Anti-Slop Guardrails
1. Never commit generated runtime artifacts:
   - `reports/model_sweeps/`
   - `reports/relevance_tuning/`
   - `reports/worker_*`
   - `app/**/__pycache__/`
   - `data/cache/`
2. If a required input file is missing, set task `blocked` with explicit unblock condition. Do not fabricate completion.
3. Use deterministic script entry points:
   - `python3 scripts/build_user64_benchmark.py`
   - `python3 scripts/run_search_benchmark.py`
   - `python3 scripts/run_embedding_model_sweep.py`
   - `python3 scripts/run_relevance_feedback_tuning.py`

## Current Baseline Snapshot
Best strict run currently committed:
- file: `reports/benchmark_results_100_step06_step08_integrated.json`
- metrics: `MustInclude@3=0.75`, `ShouldInclude@10=0.5102`, `HardExcludeRate@10=0.9767`, `PublicationEvidencePassRate=0.7333`
- gate status: still failing global `ShouldInclude@10`, mode thresholds, and overexposure controls

## Queue Index
| ID | Priority | Status | Title |
|---|---|---|---|
| SQ-UT-011 | P0 | pending | Rebuild and validate user64 benchmark baseline |
| SQ-UT-012 | P0 | pending | Re-run embedding sweep with clean inputs |
| SQ-UT-013 | P1 | pending | Run non-dry relevance tuning sweep |
| SQ-UT-014 | P0 | pending | Improve strict100 failures (should-include + overexposure) |

## Task Ledger

### [ ] SQ-UT-011 Rebuild and validate user64 benchmark baseline (P0)
Status: `pending`
Depends on: `<none>`
Blocked by: `<none>`

Goal:
Restore fully reproducible user64 benchmarking artifacts after cleanup.

Implementation targets:
- `tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml`
- `reports/benchmark_results_user64_latest.json`
- `reports/query_test_expected_vs_actual.csv`

Acceptance criteria:
1. `python3 scripts/build_user64_benchmark.py` succeeds and outputs 64 queries.
2. User64 benchmark run succeeds against local server with no `request_error` entries.
3. `scripts/export_query_test_expected_vs_actual.py --report reports/benchmark_results_user64_latest.json` succeeds.

Run commands (bounded):
- `Set-ExecutionPolicy -Scope Process Bypass -Force; .\\scripts\\run_with_timeout.ps1 -TimeoutSec 120 -FilePath .\\.venv\\Scripts\\python.exe -ArgumentList @('scripts/build_user64_benchmark.py')`
- `Set-ExecutionPolicy -Scope Process Bypass -Force; .\\scripts\\run_with_timeout.ps1 -TimeoutSec 2400 -FilePath .\\.venv\\Scripts\\python.exe -ArgumentList @('-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8101')`
- `Set-ExecutionPolicy -Scope Process Bypass -Force; .\\scripts\\run_with_timeout.ps1 -TimeoutSec 2400 -FilePath .\\.venv\\Scripts\\python.exe -ArgumentList @('scripts/run_search_benchmark.py','--benchmark','tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml','--base-url','http://127.0.0.1:8101','--output','reports/benchmark_results_user64_latest.json')`
- `Set-ExecutionPolicy -Scope Process Bypass -Force; .\\scripts\\run_with_timeout.ps1 -TimeoutSec 120 -FilePath .\\.venv\\Scripts\\python.exe -ArgumentList @('scripts/export_query_test_expected_vs_actual.py','--report','reports/benchmark_results_user64_latest.json')`

### [ ] SQ-UT-012 Re-run embedding sweep with clean inputs (P0)
Status: `pending`
Depends on: `SQ-UT-011`
Blocked by: `<none>`

Goal:
Re-run model sweep using reproducible user64 + strict100 inputs and retain only summary decisions in docs.

Implementation targets:
- `scripts/run_embedding_model_sweep.py`
- `data/models.yaml` (must be restored to original at run end)
- `docs/TODO.md` (result summary only; no generated sweep artifacts committed)

Acceptance criteria:
1. Sweep run uses `tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml` and `tests/benchmarks/search_relevance_100_v1.yaml`.
2. `data/models.yaml` is unchanged after run.
3. Decision summary in TODO includes candidate metrics and selected model/no-selection rationale.

### [ ] SQ-UT-013 Run non-dry relevance tuning sweep (P1)
Status: `pending`
Depends on: `SQ-UT-011`
Blocked by: `<none>`

Goal:
Execute real tuning trials (not dry-run), then capture best-trial recommendation with benchmark evidence.

Implementation targets:
- `scripts/run_relevance_feedback_tuning.py`
- `docs/TODO.md` summary entry

Acceptance criteria:
1. At least 3 non-dry trials complete with user64 + strict100 outputs.
2. A best trial is identified or rejection reason is explicit.
3. No generated `reports/relevance_tuning/*` files are committed.

### [ ] SQ-UT-014 Improve strict100 failures (should-include + overexposure) (P0)
Status: `pending`
Depends on: `SQ-UT-012, SQ-UT-013`
Blocked by: `<none>`

Goal:
Land retrieval/scoring changes that reduce strict100 threshold failures without relaxing thresholds.

Implementation targets:
- `app/routes.py`
- `data/app.config.yaml`
- `tests/test_routes.py`
- `tests/test_retriever.py` (if scoring behavior changes)

Acceptance criteria:
1. Strict benchmark improves `ShouldInclude@10` and does not regress `HardExcludeRate@10`.
2. Overexposure violation count decreases from current baseline.
3. Updated tests cover the new behavior.

## Session Update Log

### 2026-02-24 - QUEUE-CLEANUP
- Status change: queue cleanup pass completed.
- Key changes:
  - Removed committed generated artifacts (`reports/model_sweeps`, `reports/relevance_tuning`, `reports/worker_*`, `app/index/__pycache__`, `data/cache/*`).
  - Added deterministic benchmark builder: `scripts/build_user64_benchmark.py`.
  - Restored `tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml` from committed CSV source.
  - Hardened automation scripts with input preflight checks and improved server readiness handling.
  - Rewrote queue to current, actionable tasks only.
- Next recommended task: `SQ-UT-011`.
