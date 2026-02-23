# Model Sweep Decision Memo (2026-02-23-port-probe-pass)

## Context
- Purpose of this pass: validate benchmark-run reliability improvements (dedicated ephemeral server ports + `/queue` readiness probe) before another full 4-model rerun.
- Candidate scope this pass: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` only.
- Host GPU observed: `NVIDIA GeForce GTX 1060`.

## Candidate outcome
- **sentence-transformers/paraphrase-multilingual-mpnet-base-v2**: `strict100_threshold_failed`
  - user64: MustInclude@3=`0.3125`, ShouldInclude@10=`0.5000`
  - strict100: MustInclude@3=`0.6000`, ShouldInclude@10=`0.3061`, HardExcludeRate@10=`0.9100`, PublicationEvidencePassRate=`0.0000`
  - no request-error-driven infra classification in this pass; failure category is threshold quality.

## Keep / reject rationale (quality first)
- **Keep for continued evaluation:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
  - Reason: reliability gate for server startup improved (threshold failure now surfaced without infra failure classification in this smoke pass).
- **No model promotion in this pass:** quality thresholds are still unmet on strict100.

## Recommended model
- No final switch yet. Continue SQ-UT-008 with a full four-candidate rerun using the new server-port/probe logic.
