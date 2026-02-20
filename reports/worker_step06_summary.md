# Worker Step 06 Summary

- Changed `app/index/chunking.py` to support source-specific strict minimum handling via `allow_short_single_chunk`; NVA callers can now reject undersized singleton chunks.
- Changed `app/index/builder.py` NVA ingestion to skip publications below `min_chunk_tokens_per_source['nva']` before budgeting and to chunk NVA with `allow_short_single_chunk=False`.
- Added tests in `tests/test_chunking.py` and `tests/test_index_builder.py` covering strict singleton filtering and short-NVA drop behavior.
- Rebuilt index after Step 06 changes: 97 staff, 417 chunks.
- Strict benchmark (`reports/benchmark_results_100_step06.json`) vs Step 05:
- `MustInclude@3`: 0.7200 -> 0.7500 (+0.0300)
- `ShouldInclude@10`: 0.5000 -> 0.4898 (-0.0102)
- `HardExcludeRate@10`: 0.9733 -> 0.9800 (+0.0067)
- `PublicationEvidencePassRate`: 0.6500 -> 0.6667 (+0.0167)
- Query failures: 67 -> 68 (+1)
- Unresolved risks:
- Profile-grounded coverage regressed (`ShouldInclude@10` 0.6452 -> 0.5806).
- Overexposure remains failing; `Kjetil Tronvoll` increased top10 frequency (16 -> 21), while `Tore Pedersen` improved slightly (12 -> 11).
