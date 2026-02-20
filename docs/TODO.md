# TODO (Autonomous Agent Queue)

Last updated: 2026-02-20
Queue state: `active`
Next task id: `SQ-UT-001`
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
| SQ-UT-001 | P0 | pending | Build machine-readable benchmark from empirical table |
| SQ-UT-002 | P0 | pending | Exact keyword hard promotion |
| SQ-UT-003 | P0 | pending | Increase trusted keyword influence + conceptual keyword mapping |
| SQ-UT-004 | P0 | pending | Synonym + abbreviation expansion layer |
| SQ-UT-005 | P0 | pending | Category-aware filtering/down-weighting |
| SQ-UT-006 | P0 | pending | Staff profile preprocessing and noise stripping pipeline |
| SQ-UT-007 | P1 | pending | Evidence snippet quality gate tightening |
| SQ-UT-008 | P0 | pending | Embedding model sweep on RTX 4070 with user64 + strict100 scoring |
| SQ-UT-009 | P1 | pending | Relevance-feedback tuning pass using user64 gold labels |
| SQ-UT-010 | P2 | pending | Link missing local artifact or declare replacement |

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
Status: `pending`
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
Status: `pending`
Depends on: `SQ-UT-001`
Blocked by: `<none>`

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
Status: `pending`
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
Status: `pending`
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
Status: `pending`
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
Status: `pending`
Depends on: `<none>`
Blocked by: `<none>`

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
Status: `pending`
Depends on: `SQ-UT-001`
Blocked by: `<none>`

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
Status: `pending`
Depends on: `SQ-UT-001, SQ-UT-006`
Blocked by: `<none>`

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
Status: `pending`
Depends on: `SQ-UT-001, SQ-UT-002, SQ-UT-003, SQ-UT-004, SQ-UT-005`
Blocked by: `<none>`

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
Status: `pending`
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
