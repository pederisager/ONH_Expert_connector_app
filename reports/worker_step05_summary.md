# Worker Step 05 Revision Summary

- Scope: stronger intent-aware overexposure penalty shape (query-mode multipliers + stronger low-signal demotion).
- Outcome: rejected.
- Benchmark output: `reports/benchmark_results_100_step05r.json`.
- Metrics vs Step 05 baseline:
  - `MustInclude@3`: `0.7200 -> 0.5500` (`-0.1700`)
  - `ShouldInclude@10`: `0.5000 -> 0.2857` (`-0.2143`)
  - `HardExcludeRate@10`: `0.9733 -> 0.9067` (`-0.0667`)
  - `PublicationEvidencePassRate`: `0.6500 -> 0.0000` (`-0.6500`)
  - Query failures: `67 -> 90` (`+23`)
  - Overexposure violations: `2 -> 0`
- Decision:
  - Do not merge this Step 05 revision shape; it over-penalizes and destroys retrieval/evidence quality.
