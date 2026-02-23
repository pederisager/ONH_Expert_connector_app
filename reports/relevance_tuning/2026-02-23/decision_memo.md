# Relevance Feedback Tuning Memo (2026-02-23)

- Dry run: `True`

## Trial outcomes

- **T00 baseline**: `dry_run`
  - weights: semantic=1.0, keywords=0.2, tags=0.25, methods=0.15
  - bonuses: exact_keyword_bonus=0.35, category_base_penalty=0.08
  - note: dry_run: benchmarks not executed
- **T01 keyword_plus_tag_boost**: `dry_run`
  - weights: semantic=1.0, keywords=0.25, tags=0.3, methods=0.15
  - bonuses: exact_keyword_bonus=0.39999999999999997, category_base_penalty=0.08
  - note: dry_run: benchmarks not executed
- **T02 keyword_stronger**: `dry_run`
  - weights: semantic=1.0, keywords=0.30000000000000004, tags=0.25, methods=0.15
  - bonuses: exact_keyword_bonus=0.44999999999999996, category_base_penalty=0.1
  - note: dry_run: benchmarks not executed
- **T03 tag_stronger**: `dry_run`
  - weights: semantic=1.0, keywords=0.2, tags=0.35, methods=0.15
  - bonuses: exact_keyword_bonus=0.35, category_base_penalty=0.08
  - note: dry_run: benchmarks not executed
- **T04 cross_disciplinary_relaxation**: `dry_run`
  - weights: semantic=1.0, keywords=0.2, tags=0.25, methods=0.15
  - bonuses: exact_keyword_bonus=0.35, category_base_penalty=0.05
  - note: dry_run: benchmarks not executed

## Recommended trial

- None (no successful benchmark trials yet)
