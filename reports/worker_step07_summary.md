# Worker Step 07 Summary

- Scope: embedding model sweep on integrated Step06+Step08 baseline.
- Candidates benchmarked:
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` -> `reports/benchmark_results_100_step07_candidate_mpnet.json`
  - `intfloat/multilingual-e5-base` -> `reports/benchmark_results_100_step07_candidate_e5_base.json`
- Results:
  - mpnet: `MustInclude@3=0.7500`, `ShouldInclude@10=0.5102`, `HardExcludeRate@10=0.9767`, `PublicationEvidencePassRate=0.7333`, query failures `61`, overexposure violations `2`.
  - e5-base: `MustInclude@3=0.6200`, `ShouldInclude@10=0.5000`, `HardExcludeRate@10=0.9533`, `PublicationEvidencePassRate=0.4333`, query failures `70`, overexposure violations `0`.
- Decision:
  - Keep `paraphrase-multilingual-mpnet-base-v2` as default in `data/models.yaml`.
  - e5-base reduced overexposure but regressed ranking/evidence quality too much for strict gate progress.
