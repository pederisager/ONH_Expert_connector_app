# 2026-02-20 ChatGPT Railway User Test Archive

This folder stores the external user-test artifacts provided on 2026-02-20 for the latest Railway deployment.

## Source files (original)
- `ChatGPT user test.pdf`
- `ChatGPT user test - empirical results.docx`
- `ChatGPT user test - empirical results.pdf`

## Local text extracts (for quick grep/diff)
- `ChatGPT user test.pdf.txt`
- `ChatGPT user test - empirical results.docx.txt`
- `ChatGPT user test - empirical results.pdf.txt`

These `.txt` files are lossy extracts for engineering triage. The PDF/DOCX originals remain the source of truth.

## Key findings captured from the user test
- 64 total queries were executed.
- The summary reports 25/64 queries where the primary expected expert was missing.
- Systematic issues reported: false negatives, false positives/category leakage, weak ranking among relevant candidates, weak abbreviation/synonym handling, and citation snippets that often do not directly justify the match.
- Recurring failure pattern: generic profile language appears to over-score some unrelated staff across many queries.

## Related live backlog
- Canonical Codex task queue: `docs/search_quality_live_todo.md`
- Existing strict 100-query campaign: `docs/search_quality_improvement_backlog.md`

## Missing artifact resolution
- `query_test_expected_vs_actual.csv` was referenced by requester context but was not found in workspace or user home path scans.
- Replacement export committed at `reports/query_test_expected_vs_actual.csv`.
- Rebuild command: `python scripts/export_query_test_expected_vs_actual.py`.
