# TODO (Autonomous Agent Queue)

Last updated: 2026-02-23
Queue state: `active`
Next task id: `SQ-UT-008`
Canonical queue file: `docs/TODO.md`

## Agent Autonomy Contract
If a user says "look at TODO.md and work on the next task", this file is the execution contract.
Agents must be able to pick work, execute work, and leave a complete handoff without user intervention.

Rules:
1. Read this file completely before coding.
2. Work exactly one primary task per session.
3. Always set the chosen task to `in_progress` before implementation.
4. Run bounded commands only. Linux/WSL: `timeout <sec> ...`. PowerShell: `scripts/run_with_timeout.ps1 -TimeoutSec <sec> ...`.
5. Update this file in the same task before finishing.
6. Do not leave task status ambiguous.

## Status Vocabulary
- `pending`: ready to start.
- `in_progress`: currently being executed by this session.
- `blocked`: cannot proceed until explicit blocker is resolved.
- `completed`: done and verified against acceptance criteria.
- `revised`: partially landed but requires follow-up changes.

## Deterministic Task Selection Algorithm
1. If any task is `in_progress`, continue that task first.
2. Otherwise choose the highest-priority `pending` task (`P0` before `P1` before `P2`).
3. Break ties by lowest task ID number.
4. If the chosen task has unmet dependencies, mark it `blocked`, add blocker details, and choose the next eligible task.
5. Update `Next task id` at the top of this file before implementation.

## Mandatory TODO.md Updates Before Session End
1. Update `Last updated`.
2. Update `Queue state` and `Next task id`.
3. Update the worked task status (`in_progress -> completed/revised/blocked`) both in `Task Ledger` and in `Queue Index`.
4. Append a new entry under `Session Update Log` containing:
- date
- task id
- status change
- files changed
- exact commands run (with timeout values)
- benchmark/test output paths
- metric deltas or verification summary
- decision (`keep`, `revise`, or `revert`)
- explicit next recommended task id
5. If a task is `blocked`, include unblock condition and create follow-up task(s) if needed.

## No Pending Tasks Protocol
If no `pending` or `in_progress` task exists:
1. Run a queue health check:
- run targeted tests for recently changed areas
- run benchmark(s) relevant to queue goals using bounded commands
2. If failures/regressions exist, create new tasks from the top failure clusters and set `Queue state: active`.
3. If no meaningful failures exist and all exit criteria are met:
- set `Queue state: awaiting_user_input`
- set `Next task id: none`
- append a `Session Update Log` entry titled `QUEUE-IDLE` with proof (commands and outputs)
- stop and wait for user direction

## New Task Template
Use this exact template when adding tasks:

### [ ] SQ-UT-### <short title> (P0|P1|P2)
Status: `pending`
Depends on: `<none|task ids>`
Blocked by: `<none|reason>`

Goal:
<one paragraph>

Implementation targets:
- <file/path and change target>

Acceptance criteria:
1. <criterion>

Run commands (bounded):
- `timeout ...`

## Source Artifacts
- `docs/user_testing/2026-02-20_chatgpt_railway_user_test/ChatGPT user test.pdf`
- `docs/user_testing/2026-02-20_chatgpt_railway_user_test/ChatGPT user test - empirical results.docx`
- `docs/user_testing/2026-02-20_chatgpt_railway_user_test/ChatGPT user test - empirical results.pdf`
- `docs/user_testing/2026-02-20_chatgpt_railway_user_test/README.md`

## Baseline Snapshot from User Test
- Query set size: 64
- Reported misses: 25/64 queries missing the primary expected expert
- Key error classes:
- false negatives for exact ONH expert topics
- false positives/category leakage from generic profile language
- weak top-rank ordering when relevant expert is retrieved
- weak handling of synonyms/abbreviations (e.g. `FN`)
- citation snippets that do not clearly justify relevance
- Recurring problematic pattern in empirical table: several unrelated results appear repeatedly for high-severity queries (especially queries in psychology/HR, IR, and physiotherapy)

## High-Severity Queries to Prioritize
`arbeidsmiljø og hybridarbeid`, `kommunikasjonstrening`, `alkoholmisbruk ungdom`, `kritisk psykologi`, `kognitiv atferdsterapi`, `beslutningsbias`, `teambygging`, `stress og høyaktivitet`, `arbeidskonflikter`, `graviditetstrening`, `humanitærrett`, `FN`, `radikalisering`, `diplomati`, `bistand`, `autoritaere regimer`, `sosialpsykologi`, `ledelsespsykologi`, `stressmestring`, `styrketrening`

## Queue Index
Keep this table synchronized with `Task Ledger` status values.

| ID | Priority | Status | Title |
|---|---|---|---|
| SQ-UT-000 | P0 | completed | Archive user-test artifacts in repo |
| SQ-UT-001 | P0 | completed | Build machine-readable benchmark from empirical table |
| SQ-UT-002 | P0 | revised | Exact keyword hard promotion |
| SQ-UT-003 | P0 | revised | Increase trusted keyword influence + conceptual keyword mapping |
| SQ-UT-004 | P0 | revised | Synonym + abbreviation expansion layer |
| SQ-UT-005 | P0 | revised | Category-aware filtering/down-weighting |
| SQ-UT-006 | P0 | revised | Staff profile preprocessing and noise stripping pipeline |
| SQ-UT-007 | P1 | revised | Evidence snippet quality gate tightening |
| SQ-UT-008 | P0 | revised | Embedding model sweep on RTX 4070 with user64 + strict100 scoring |
| SQ-UT-009 | P1 | revised | Relevance-feedback tuning pass using user64 gold labels |
| SQ-UT-010 | P2 | completed | Link missing local artifact or declare replacement |

## Queue Exit Criteria
Treat the queue as truly complete only when all are true:
1. No `pending`, `in_progress`, `blocked`, or `revised` tasks remain.
2. Latest strict benchmark and user64 benchmark outputs are present and pass agreed thresholds.
3. No unresolved high-severity retrieval failures remain in current benchmark reports.
4. `Session Update Log` contains a final `QUEUE-IDLE` entry with command/output evidence.

## Task Ledger

### ~~[x] SQ-UT-000 Archive user-test artifacts in repo~~
Status: `completed` (2026-02-20)

Scope:
- Added a dedicated archive folder for supplied files.
- Added a local README with provenance and findings summary.

Files changed:
- `docs/user_testing/2026-02-20_chatgpt_railway_user_test/ChatGPT user test.pdf`
- `docs/user_testing/2026-02-20_chatgpt_railway_user_test/ChatGPT user test - empirical results.docx`
- `docs/user_testing/2026-02-20_chatgpt_railway_user_test/ChatGPT user test - empirical results.pdf`
- `docs/user_testing/2026-02-20_chatgpt_railway_user_test/README.md`

### [ ] SQ-UT-001 Build machine-readable benchmark from empirical table (P0)
Status: `completed`
Depends on: `<none>`
Blocked by: `<none>`

Goal:
Create a reproducible benchmark from the 64-query empirical table so improvements can be measured per query and per severity.

Implementation targets:
- Add benchmark file: `tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml`
- Parse/encode fields: query, expected top staff, expected include set, severity, mode hint (`publication_grounded` or `profile_grounded`)
- Add script support for this benchmark in `scripts/run_search_benchmark.py` if schema updates are required

Acceptance criteria:
1. Benchmark file includes all 64 queries from the empirical table.
2. Script runs without schema errors.
3. Output report includes per-query pass/fail and aggregated metrics.

Run commands (bounded):
- `timeout 1200 python3 scripts/run_search_benchmark.py --benchmark tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml --base-url http://127.0.0.1:8000 --output reports/benchmark_results_user64_baseline.json`

### [ ] SQ-UT-002 Exact keyword hard promotion (P0, requester requirement #1)
Status: `revised`
Depends on: `SQ-UT-001`
Blocked by: `Acceptance criterion #2 remains unmet after rerun: user64 baseline metrics unchanged and high-severity exact-topic misses persist.`

Goal:
If query matches staff keyword/tag exactly, promote that staff member to the top tier before weaker semantic matches.

Implementation targets:
- `app/routes.py`: add explicit exact keyword match feature in score pipeline
- `data/app.config.yaml`: add configurable boost/override gates
- `tests/test_routes.py`: add tests for exact-match top ranking (including tie handling)

Acceptance criteria:
1. Exact keyword match always ranks above non-exact matches unless a hard exclusion/safety rule applies.
2. Affected high-severity queries improve in user64 benchmark.
3. No regression on strict exclusion controls.

### [ ] SQ-UT-003 Increase trusted keyword influence + conceptual keyword mapping (P0, requester requirement #2)
Status: `revised`
Depends on: `SQ-UT-002`
Blocked by: `<none>`

Goal:
Increase weighting of author keywords/tags and add controlled conceptual-neighbor matching for near-equivalent concepts.

Implementation targets:
- `data/app.config.yaml`: raise/retune keyword and tag weights
- Add new config block for curated concept map (examples: `diskursanalyse -> kvalitativ metode`, `statistikk -> kvantitativ metode`, `anoreksi -> spiseforstyrrelser`)
- `app/routes.py` and/or `app/rag/retriever.py`: apply concept-map expansion in overlap scoring without flooding recall
- `tests/test_routes.py`, `tests/test_retriever.py`: coverage for exact, near-match, and false-positive controls

Acceptance criteria:
1. Concept-neighbor matches improve recall for mapped terms.
2. Exact keyword behavior from SQ-UT-002 remains intact.
3. Overexposure/false-positive metrics do not regress beyond agreed threshold.

### [ ] SQ-UT-004 Synonym + abbreviation expansion layer (P0, user-test recommendation)
Status: `revised`
Depends on: `SQ-UT-001`
Blocked by: `<none>`

Goal:
Improve retrieval for abbreviations and lexical variants (e.g. `FN`, `autoritaere regimer`, mixed NO/EN terms).

Implementation targets:
- Add lightweight query-expansion module (config-driven, deterministic)
- Seed with high-impact misses from user64 failures
- Wire into both lexical and tag overlap paths
- Add tests for `FN`, `humanitærrett`, `diplomati`, `kognitiv atferdsterapi`

Acceptance criteria:
1. Expanded terms are visible in debug/score breakdown for traceability.
2. Targeted high-severity misses show measurable improvement.
3. Non-target domains are not significantly polluted.

### [ ] SQ-UT-005 Category-aware filtering/down-weighting (P0, user-test recommendation)
Status: `revised`
Depends on: `SQ-UT-001`
Blocked by: `<none>`

Goal:
Reduce category leakage by down-weighting staff from unrelated departments when query intent is clear.

Implementation targets:
- Define category intent signals (from tags, curated map, and query terms)
- Add config-driven penalties in score pipeline (`results.*` config)
- Update route tests with known leakage cases from user64

Acceptance criteria:
1. Fewer unrelated staff in top 5 for high-severity queries.
2. Relevant cross-disciplinary matches remain possible with explicit evidence.
3. Strict benchmark exclusion metrics hold or improve.

### [ ] SQ-UT-006 Staff profile preprocessing and noise stripping pipeline (P0, requester requirement #3)
Status: `revised`
Depends on: `<none>`
Blocked by: `Acceptance criterion #2 requires benchmark verification after retrieval rerun.`

Goal:
Strip non-expertise junk from harvested profile text before indexing.

Implementation targets:
- `app/index/refresh_staff.py`: upgrade `extract_profile_summary` with section-aware filters and boilerplate removal
- Keep and prioritize expertise sections (`Jobber med`, `Forskning`, topic lists), down-weight/remove CV/admin clutter and unrelated site text
- Add tests in `tests/test_refresh_staff.py` with representative noisy HTML fixtures
- Run `python3 scripts/audit_staff_data.py` after pipeline changes

Acceptance criteria:
1. `data/staff_records.jsonl` summaries become shorter and more expertise-dense.
2. Retrieval false positives caused by generic language are reduced.
3. No staff records become empty unless source page truly lacks content.

Run commands (bounded):
- `timeout 1200 python3 -m pytest tests/test_refresh_staff.py`
- `timeout 1200 python3 scripts/audit_staff_data.py`

### [ ] SQ-UT-007 Evidence snippet quality gate tightening (P1, user-test recommendation)
Status: `revised`
Depends on: `SQ-UT-001`
Blocked by: `Acceptance criteria #1-#2 need benchmark-level validation on user64/strict100 after this citation gate change.`

Goal:
Ensure citation snippets directly support the query/topic and avoid generic filler text.

Implementation targets:
- `app/routes.py`: require stronger snippet-query overlap for profile/staffinfo citations
- Keep publication-mode tag augmentation logic intact
- Add tests for snippet relevance and fallback ordering

Acceptance criteria:
1. Snippets for top results mention query terms or mapped equivalents.
2. Unsupported citations are filtered or down-ranked.
3. Existing publication evidence improvements are preserved.

### [ ] SQ-UT-008 Embedding model sweep on RTX 4070 with user64 + strict100 scoring (P0, requester requirement #4)
Status: `revised`
Depends on: `SQ-UT-001, SQ-UT-006`
Blocked by: `No model has yet passed strict100 thresholds; strict100 runs still show intermittent request errors under local server load; BAAI/bge-m3 index build fails due torch<2.6 safety restriction when loading model weights.`

Goal:
Evaluate stronger embedding models and keep the best quality model that is practical on RTX 4070.

Candidate starting set:
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (current baseline)
- `intfloat/multilingual-e5-large`
- `BAAI/bge-m3`
- one additional multilingual candidate that fits VRAM constraints

Implementation targets:
- Add/extend sweep script in `scripts/` to iterate model config, rebuild index, and run both benchmarks
- Store outputs under `reports/model_sweeps/<date>/`
- Record VRAM/runtime/quality metrics for each model

Acceptance criteria:
1. Every candidate has reproducible benchmark output files.
2. Decision memo documents keep/reject rationale based on quality metrics first, speed second.
3. Chosen model/config is reflected in `data/models.yaml` and rebuild instructions.

Run commands (bounded examples):
- `timeout 2400 python3 -m app.index.build --nva-results data/nva/results.jsonl`
- `timeout 2400 python3 scripts/run_search_benchmark.py --benchmark tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml --base-url http://127.0.0.1:8000 --output reports/model_sweeps/<date>/<model>_user64.json`
- `timeout 2400 python3 scripts/run_search_benchmark.py --benchmark tests/benchmarks/search_relevance_100_v1.yaml --base-url http://127.0.0.1:8000 --output reports/model_sweeps/<date>/<model>_strict100.json`

### [ ] SQ-UT-009 Relevance-feedback tuning pass using user64 gold labels (P1, user-test recommendation)
Status: `revised`
Depends on: `SQ-UT-001, SQ-UT-002, SQ-UT-003, SQ-UT-004, SQ-UT-005`
Blocked by: `Full benchmark-backed tuning pass (non-dry user64 + strict100) still pending; current pass added deterministic sweep tooling + dry-run artifacts.`

Goal:
Use user64 labels to tune scoring weights and penalties systematically rather than ad-hoc.

Implementation targets:
- Add tuning notebook/script in `scripts/` to sweep scoring weight ranges
- Keep config-driven outputs in `data/app.config.yaml`
- Document selected settings and deltas

Acceptance criteria:
1. Tuned config improves user64 top-rank and recall metrics.
2. Strict100 guardrails remain acceptable.
3. Tuning process is reproducible from commands in this file.

### [ ] SQ-UT-010 Link missing local artifact (`query_test_expected_vs_actual.csv`) or declare replacement (P2)
Status: `completed` (2026-02-23)
Depends on: `SQ-UT-001`
Blocked by: `<none>`

Goal:
Resolve missing file referenced by requester IDE context so future tasks do not rely on absent data.

Implementation targets:
- Search requester-provided paths for CSV
- If unavailable, generate equivalent export from the new benchmark and document replacement path

Acceptance criteria:
1. Backlog no longer references an unresolved artifact.
2. Replacement file is committed and documented.

## Session Update Log
Add one entry per completed/revised task.

### 2026-02-20 - SQ-UT-000
- Status change: `pending -> completed`
- Files changed:
- `docs/user_testing/2026-02-20_chatgpt_railway_user_test/ChatGPT user test.pdf`
- `docs/user_testing/2026-02-20_chatgpt_railway_user_test/ChatGPT user test - empirical results.docx`
- `docs/user_testing/2026-02-20_chatgpt_railway_user_test/ChatGPT user test - empirical results.pdf`
- `docs/user_testing/2026-02-20_chatgpt_railway_user_test/README.md`
- Command log (with timeout):
- `timeout 20 mkdir -p docs/user_testing/2026-02-20_chatgpt_railway_user_test`
- `timeout 20 cp -f "/mnt/c/Users/pader/Downloads/ChatGPT user test.pdf" docs/user_testing/2026-02-20_chatgpt_railway_user_test/`
- `timeout 20 cp -f "/mnt/c/Users/pader/Downloads/ChatGPT user test - empirical results.docx" docs/user_testing/2026-02-20_chatgpt_railway_user_test/`
- `timeout 20 cp -f "/mnt/c/Users/pader/Downloads/ChatGPT user test - empirical results.pdf" docs/user_testing/2026-02-20_chatgpt_railway_user_test/`
- `timeout 120 /tmp/codex_venv/bin/python <extract script>`
- Benchmark output path: `n/a`
- Decision: keep
- Next recommended task: `SQ-UT-001`

### 2026-02-20 - QUEUE-META-001
- Status change: `n/a` (queue-governance update)
- Files changed:
- `docs/TODO.md`
- `AGENTS.md`
- `docs/search_quality_improvement_backlog.md`
- Command log (with timeout):
- `timeout 20 sed -n '1,260p' docs/TODO.md`
- `timeout 20 rg -n "TODO\\.md|search_quality_live_todo" AGENTS.md docs/*.md`
- Decision: keep
- Next recommended task: `SQ-UT-001`

### 2026-02-21 - SQ-UT-001
- Status change: `pending -> in_progress -> revised`
- Files changed:
- `tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml`
- `scripts/build_user64_benchmark.py`
- `scripts/run_user64_baseline_with_local_server.py`
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 30 -FilePath powershell -ArgumentList @('-NoProfile','-Command','Get-ChildItem tests/benchmarks | Select-Object Name')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 120 -FilePath python -ArgumentList @('scripts/build_user64_benchmark.py')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 1200 -FilePath python -ArgumentList @('scripts/run_search_benchmark.py','--benchmark','tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml','--base-url','http://127.0.0.1:8000','--output','reports/benchmark_results_user64_baseline.json')` (failed: target API not running)
- `scripts/run_with_timeout.ps1 -TimeoutSec 60 -FilePath python -ArgumentList @('-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000')` (failed: `ModuleNotFoundError: No module named 'sklearn'`)
- `scripts/run_with_timeout.ps1 -TimeoutSec 1200 -FilePath python -ArgumentList @('-m','pip','install','-r','requirements.txt')` (failed: Windows access denied in user-site package path)
- Benchmark output path: `reports/benchmark_results_user64_baseline.json` (not generated due environment blocker)
- Metric/verification summary: generated user64 benchmark YAML with 64/64 queries from empirical table; benchmark execution remains blocked by local Python environment setup.
- Decision: revise
- Next recommended task: `SQ-UT-001`

### 2026-02-21 - SQ-UT-001 (completion pass)
- Status change: `revised -> in_progress -> completed`
- Files changed:
- `.venv/` (local project environment)
- `reports/benchmark_results_user64_baseline.json`
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 180 -FilePath python -ArgumentList @('-m','venv','.venv')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 1800 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','pip','install','-r','requirements.txt')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 1800 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('scripts/run_user64_baseline_with_local_server.py')`
- Benchmark output path: `reports/benchmark_results_user64_baseline.json`
- Metric/verification summary: report generated with per-query outcomes and aggregate metrics (`MustInclude@3=0.296875`, `ShouldInclude@10=0.25`) for all 64 queries.
- Decision: keep
- Next recommended task: `SQ-UT-002`

### 2026-02-21 - SQ-UT-002
- Status change: `pending -> in_progress -> revised`
- Files changed:
- `app/routes.py`
- `app/config_loader.py`
- `data/app.config.yaml`
- `tests/test_routes.py`
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 180 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','py_compile','app/routes.py','app/config_loader.py','tests/test_routes.py')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 1200 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','pytest','tests/test_routes.py')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 600 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','pytest','tests/test_config_loader.py','tests/test_routes.py')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 2400 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('scripts/run_user64_baseline_with_local_server.py')` (stalled; terminated)
- Benchmark/test output paths:
- `reports/benchmark_results_user64_baseline.json` (pre-existing baseline; no fresh SQ-UT-002 comparative run produced)
- Metric/verification summary: added exact keyword promotion config + scoring breakdown + rank-tier sort (`exact_keyword_match` before raw score). Added tests proving (a) exact keyword matches outrank non-exact hits and (b) exact-match tie keeps score order. Full targeted suite passed: `27 passed`.
- Decision: revise
- Next recommended task: `SQ-UT-002`

### 2026-02-21 - SQ-UT-002 (validation pass)
- Status change: `revised -> in_progress -> revised`
- Files changed:
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 900 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','pytest','tests/test_routes.py','-k','exact_keyword_promotion','-q')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 1200 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','pytest','tests/test_routes.py','-q')`
- Benchmark/test output paths:
- `reports/benchmark_results_user64_baseline.json` (existing baseline used for acceptance review)
- Metric/verification summary: exact-keyword promotion tests passed (`2 passed`) and full `tests/test_routes.py` passed (`23 passed`), but acceptance criterion #2 remains unmet against current user64 baseline (high-severity exact-topic misses still present).
- Decision: revise
- Next recommended task: `SQ-UT-002`

### 2026-02-22 - SQ-UT-002 (benchmark rerun)
- Status change: `in_progress -> revised`
- Files changed:
- `reports/benchmark_results_user64_baseline.json`
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 2400 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('scripts/run_user64_baseline_with_local_server.py')`
- Benchmark/test output paths:
- `reports/benchmark_results_user64_baseline.json`
- Metric/verification summary: fresh baseline generated (`generated_at: 2026-02-22T00:31:36Z`) with unchanged aggregate metrics (`MustInclude@3=0.296875`, `ShouldInclude@10=0.25`). High-severity exact-topic misses remain (e.g., `U003`, `U008`, `U010`, `U013`, `U017`, `U018`, `U019`).
- Decision: revise
- Next recommended task: `SQ-UT-002`

### 2026-02-22 - SQ-UT-003
- Status change: `pending -> in_progress -> revised`
- Files changed:
- `app/routes.py`
- `app/config_loader.py`
- `data/app.config.yaml`
- `tests/test_routes.py`
- `tests/test_config_loader.py`
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 180 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','py_compile','app/routes.py','app/config_loader.py','tests/test_routes.py','tests/test_config_loader.py')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 1200 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','pytest','tests/test_config_loader.py','tests/test_routes.py','-q')`
- Benchmark/test output paths:
- `pytest stdout` (`28 passed, 16 warnings in 6.70s`)
- Metric/verification summary: increased trusted keyword/tag scoring weights (`keywords 0.2`, `tags 0.25`) and added config-driven concept keyword mapping used in overlap scoring; added test coverage for config loading + concept map overlap.
- Decision: revise
- Next recommended task: `SQ-UT-003`

### 2026-02-22 - SQ-UT-004
- Status change: `pending -> in_progress -> revised`
- Files changed:
- `app/routes.py`
- `app/config_loader.py`
- `data/app.config.yaml`
- `tests/test_routes.py`
- `tests/test_config_loader.py`
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 240 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','py_compile','app/routes.py','app/config_loader.py','tests/test_routes.py','tests/test_config_loader.py')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 1200 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','pytest','tests/test_config_loader.py','tests/test_routes.py','-q')`
- Benchmark/test output paths:
- `pytest stdout` (`31 passed, 16 warnings in 6.79s`)
- Metric/verification summary: added deterministic config-driven synonym/abbreviation expansion layer (including `FN`, `humanitærrett`, `diplomati`, `kognitiv atferdsterapi`), wired expansion into query text + lexical/tag/citation overlap paths, and exposed `expanded_query_terms_count` in score breakdown for traceability.
- Decision: revise
- Next recommended task: `SQ-UT-004`

### 2026-02-22 - SQ-UT-005
- Status change: `pending -> in_progress -> revised`
- Files changed:
- `app/routes.py`
- `app/config_loader.py`
- `data/app.config.yaml`
- `tests/test_routes.py`
- `tests/test_config_loader.py`
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 240 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','py_compile','app/routes.py','app/config_loader.py','tests/test_routes.py','tests/test_config_loader.py')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 1200 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','pytest','tests/test_config_loader.py','tests/test_routes.py','-q')`
- Benchmark/test output paths:
- `pytest stdout` (`33 passed, 16 warnings in 6.78s`)
- Metric/verification summary: added config-driven category-intent penalty to down-weight unrelated department matches while preserving cross-disciplinary results with explicit topical evidence; surfaced `category_intent_penalty` in score breakdown and added targeted regression tests.
- Decision: revise
- Next recommended task: `SQ-UT-005`

### 2026-02-22 - SQ-UT-006
- Status change: `pending -> in_progress -> revised`
- Files changed:
- `app/index/refresh_staff.py`
- `tests/test_refresh_staff.py`
- `reports/staff_data_audit.json`
- `reports/staff_data_audit.md`
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 240 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','py_compile','app/index/refresh_staff.py','tests/test_refresh_staff.py','tests/test_refresh_staff_parsing.py')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 1200 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','pytest','tests/test_refresh_staff.py','tests/test_refresh_staff_parsing.py','-q')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 1200 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('scripts/audit_staff_data.py')`
- Benchmark/test output paths:
- `pytest stdout` (`8 passed, 2 warnings in 0.16s`)
- `reports/staff_data_audit.json`
- `reports/staff_data_audit.md`
- Metric/verification summary: added section-aware summary extraction with expertise-priority cues and low-value CV/admin/contact filtering, plus fallback to avoid empty summaries when pages only contain low-priority sections. Added focused parser tests for noise stripping and fallback behavior. Audit regenerated (`staff_with_issues=87`, `unresolved_high_count=13`) for post-change tracking.
- Decision: revise
- Next recommended task: `SQ-UT-008`

### 2026-02-22 - SQ-UT-008
- Status change: `pending -> in_progress -> revised`
- Files changed:
- `scripts/run_embedding_model_sweep.py`
- `reports/model_sweeps/2026-02-22/sweep_summary.json`
- `reports/model_sweeps/2026-02-22/decision_memo.md`
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 240 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','py_compile','scripts/run_embedding_model_sweep.py')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 240 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('scripts/run_embedding_model_sweep.py','--help')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 240 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('scripts/run_embedding_model_sweep.py','--run-date','2026-02-22','--max-models','4','--dry-run')`
- Benchmark/test output paths:
- `reports/model_sweeps/2026-02-22/sweep_summary.json`
- `reports/model_sweeps/2026-02-22/decision_memo.md`
- Metric/verification summary: added a deterministic sweep runner that updates `data/models.yaml` per candidate, rebuilds index, runs user64 + strict100 benchmarks with bounded timeouts, records command logs, extracts benchmark metrics, captures GPU snapshots, and emits machine-readable summary + decision memo. Dry-run executed for all four candidate models and produced reproducible artifacts; runtime environment reports `NVIDIA GeForce GTX 1060` (not RTX 4070), so full quality/performance comparison remains pending.
- Decision: revise
- Next recommended task: `SQ-UT-007`

### 2026-02-22 - SQ-UT-007
- Status change: `pending -> in_progress -> revised`
- Files changed:
- `app/routes.py`
- `app/config_loader.py`
- `data/app.config.yaml`
- `tests/test_routes.py`
- `tests/test_config_loader.py`
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 240 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','py_compile','app/routes.py','app/config_loader.py','tests/test_routes.py','tests/test_config_loader.py')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 1200 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','pytest','tests/test_config_loader.py','tests/test_routes.py','-q')`
- Benchmark/test output paths:
- `pytest stdout` (`35 passed, 16 warnings in 6.73s`)
- Metric/verification summary: tightened citation evidence gate by enforcing stronger query-overlap minimum specifically for `profile`/`staffinfo` snippets (`profile-staffinfo-min-query-overlap-per-citation: 2`) while preserving publication-mode NVA tag augmentation path. Added fallback ranking to down-rank zero-overlap citations when strict gating yields no direct candidates. Added route tests for stronger profile/staffinfo gating and fallback ordering.
- Decision: revise
- Next recommended task: `SQ-UT-009`

### 2026-02-23 - SQ-UT-009
- Status change: `pending -> in_progress -> revised`
- Files changed:
- `scripts/run_relevance_feedback_tuning.py`
- `reports/relevance_tuning/2026-02-23/tuning_summary.json`
- `reports/relevance_tuning/2026-02-23/decision_memo.md`
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 240 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','py_compile','scripts/run_relevance_feedback_tuning.py')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 240 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('scripts/run_relevance_feedback_tuning.py','--help')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 300 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('scripts/run_relevance_feedback_tuning.py','--run-date','2026-02-23','--max-trials','5','--dry-run')`
- Benchmark/test output paths:
- `reports/relevance_tuning/2026-02-23/tuning_summary.json`
- `reports/relevance_tuning/2026-02-23/decision_memo.md`
- Metric/verification summary: added deterministic relevance-feedback tuning runner that sweeps score/penalty candidates, writes per-trial config snapshots, executes user64 + strict100 benchmarks against a local server (non-dry mode), computes a combined tuning score, and can optionally apply the best trial back to `data/app.config.yaml`. Dry-run produced reproducible 5-trial artifacts (`T00`..`T04`) with no benchmark executions yet.
- Decision: revise
- Next recommended task: `SQ-UT-010`

### 2026-02-23 - SQ-UT-010
- Status change: `pending -> in_progress -> completed`
- Files changed:
- `scripts/export_query_test_expected_vs_actual.py`
- `reports/query_test_expected_vs_actual.csv`
- `docs/user_testing/2026-02-20_chatgpt_railway_user_test/README.md`
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 120 -FilePath powershell -ArgumentList @('-NoProfile','-Command','Get-ChildItem -Path ''C:\Users\pader\.openclaw\workspace'' -Recurse -Filter ''query_test_expected_vs_actual.csv'' | ForEach-Object { $_.FullName }')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 300 -FilePath cmd -ArgumentList @('/c','where /r C:\Users\pader query_test_expected_vs_actual.csv')` (not found)
- `scripts/run_with_timeout.ps1 -TimeoutSec 60 -FilePath git -ArgumentList @('grep','-n','query_test_expected_vs_actual.csv')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 120 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','py_compile','scripts/export_query_test_expected_vs_actual.py')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 120 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('scripts/export_query_test_expected_vs_actual.py')`
- Benchmark/test output paths:
- `reports/query_test_expected_vs_actual.csv`
- Metric/verification summary: the legacy CSV was not present in workspace/home scans, so a deterministic replacement export is now generated from `tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml` + `reports/benchmark_results_user64_baseline.json`. Replacement contains 64 rows (U001-U064) with expected includes and actual benchmark outcomes.
- Decision: keep
- Next recommended task: `SQ-UT-008`

### 2026-02-23 - SQ-UT-008 (full non-dry sweep pass)
- Status change: `revised -> in_progress -> revised`
- Files changed:
- `reports/model_sweeps/2026-02-23/sweep_summary.json`
- `reports/model_sweeps/2026-02-23/decision_memo.md`
- `reports/model_sweeps/2026-02-23/*_user64.json`
- `reports/model_sweeps/2026-02-23/*_strict100.json`
- `reports/model_sweeps/2026-02-23/logs/*`
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 7200 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('scripts/run_embedding_model_sweep.py','--run-date','2026-02-23','--max-models','4')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 60 -FilePath git -ArgumentList @('restore','--worktree','data/models.yaml')`
- Benchmark/test output paths:
- `reports/model_sweeps/2026-02-23/sweep_summary.json`
- `reports/model_sweeps/2026-02-23/decision_memo.md`
- `reports/model_sweeps/2026-02-23/sentence-transformers-paraphrase-multilingual-mpnet-base-v2_user64.json`
- `reports/model_sweeps/2026-02-23/sentence-transformers-paraphrase-multilingual-mpnet-base-v2_strict100.json`
- `reports/model_sweeps/2026-02-23/intfloat-multilingual-e5-large_user64.json`
- `reports/model_sweeps/2026-02-23/intfloat-multilingual-e5-large_strict100.json`
- `reports/model_sweeps/2026-02-23/sentence-transformers-paraphrase-multilingual-minilm-l12-v2_user64.json`
- `reports/model_sweeps/2026-02-23/sentence-transformers-paraphrase-multilingual-minilm-l12-v2_strict100.json`
- Metric/verification summary: completed non-dry sweep across four candidates on local `NVIDIA GeForce GTX 1060` host. User64 metrics: mpnet (`MustInclude@3=0.6875`, `ShouldInclude@10=0.7083`), MiniLM (`0.6094`, `0.6042`), e5-large (`0.5000`, `0.3750`). All strict100 runs failed benchmark thresholds (e.g., mpnet strict100 `MustInclude@3=0.73`, `ShouldInclude@10=0.4898`, `HardExcludeRate@10=0.88`, `PublicationEvidencePassRate=0.6167`). `BAAI/bge-m3` build failed due transformers safety gate requiring torch >=2.6 when loading non-safetensor weights.
- Decision: revise
- Next recommended task: `SQ-UT-008`

### 2026-02-23 - SQ-UT-008 (benchmark reliability + retry classification)
- Status change: `revised -> in_progress -> revised`
- Files changed:
- `scripts/run_embedding_model_sweep.py`
- `reports/model_sweeps/2026-02-23-retry-pass/sweep_summary.json`
- `reports/model_sweeps/2026-02-23-retry-pass/decision_memo.md`
- `reports/model_sweeps/2026-02-23-retry-pass/sentence-transformers-paraphrase-multilingual-mpnet-base-v2_user64.json`
- `reports/model_sweeps/2026-02-23-retry-pass/sentence-transformers-paraphrase-multilingual-mpnet-base-v2_strict100.json`
- `reports/model_sweeps/2026-02-23-retry-pass/logs/*`
- `docs/TODO.md`
- Command log (with timeout):
- `scripts/run_with_timeout.ps1 -TimeoutSec 240 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('-m','py_compile','scripts/run_embedding_model_sweep.py')`
- `scripts/run_with_timeout.ps1 -TimeoutSec 2400 -FilePath .\.venv\Scripts\python.exe -ArgumentList @('scripts/run_embedding_model_sweep.py','--run-date','2026-02-23-retry-pass','--models','sentence-transformers/paraphrase-multilingual-mpnet-base-v2','--max-models','1','--skip-index','--benchmark-retries','1')`
- Benchmark/test output paths:
- `reports/model_sweeps/2026-02-23-retry-pass/sweep_summary.json`
- `reports/model_sweeps/2026-02-23-retry-pass/decision_memo.md`
- `reports/model_sweeps/2026-02-23-retry-pass/sentence-transformers-paraphrase-multilingual-mpnet-base-v2_user64.json`
- `reports/model_sweeps/2026-02-23-retry-pass/sentence-transformers-paraphrase-multilingual-mpnet-base-v2_strict100.json`
- Metric/verification summary: enhanced sweep runner to (a) retry benchmark runs when output contains request errors, (b) distinguish infra failures vs threshold failures in candidate status, and (c) persist benchmark metrics even when benchmark exits non-zero. Smoke non-dry pass confirms strict100 is now flagged as `strict100_infra_failed` with explicit retry note (`strict100 retry 1/1 after 9 request errors`) and retains metrics for diagnosis.
- Decision: revise
- Next recommended task: `SQ-UT-008`
