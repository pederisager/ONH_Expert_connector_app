# Model Sweep Decision Memo (2026-02-23)

## Context
- Host GPU detected during run: `NVIDIA GeForce GTX 1060` (6GB VRAM), not RTX 4070.
- Sweep command: `python scripts/run_embedding_model_sweep.py --run-date 2026-02-23 --max-models 4`
- Benchmarks executed: user64 (`search_relevance_chatgpt_user64_v1`) and strict100 (`search_relevance_100_v1`).

## Candidate outcomes

- **sentence-transformers/paraphrase-multilingual-mpnet-base-v2**: `strict100_failed`
  - user64: MustInclude@3=`0.6875`, ShouldInclude@10=`0.7083`
  - strict100: MustInclude@3=`0.7300`, ShouldInclude@10=`0.4898`, HardExcludeRate@10=`0.8800`, PublicationEvidencePassRate=`0.6167`
  - strict100 threshold failures include recall/exclusion/evidence targets (and overexposure violations present).

- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**: `strict100_failed`
  - user64: MustInclude@3=`0.6094`, ShouldInclude@10=`0.6042`
  - strict100: MustInclude@3=`0.7300`, ShouldInclude@10=`0.5306`, HardExcludeRate@10=`0.8967`, PublicationEvidencePassRate=`0.5667`
  - strict100 threshold failures include recall/exclusion/evidence targets (and overexposure violations present).

- **intfloat/multilingual-e5-large**: `strict100_failed`
  - user64: MustInclude@3=`0.5000`, ShouldInclude@10=`0.3750`
  - strict100: MustInclude@3=`0.6100`, ShouldInclude@10=`0.3469`, HardExcludeRate@10=`0.8433`, PublicationEvidencePassRate=`0.4000`
  - weakest aggregate quality of tested successful builds in this pass.

- **BAAI/bge-m3**: `build_failed`
  - index build failed while loading model weights.
  - blocker: transformers safety gate requires torch >=2.6 for this weight-loading path (`torch.load` CVE guard).

## Keep / reject decision (this pass)
- **Keep for further consideration:**
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Reject for now:**
  - `intfloat/multilingual-e5-large` (quality below top two in both user64 and strict100)
  - `BAAI/bge-m3` (hard build blocker in current environment)

## Recommended model
- **No final switch yet** (queue task remains revised): none of the successful candidates met strict100 thresholds in this pass.
- Provisional leader by quality among runnable candidates: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
