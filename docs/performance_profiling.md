# Performance Profiling Guide

Use `scripts/profile_match.py` to time the heavy stages inside `/match` without
running FastAPI or touching the network. The script reuses the same helper
functions as the endpoint (Cristin/NVA fetch+merge, TF-IDF/embedding scoring,
LLM explanations) and records each stage in milliseconds.

## Quick start

```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/profile_match.py \
  "klinisk psykologi" "traumebehandling" \
  --runs 5 --staff-limit 50 --disable-rag --force-tfidf \
  --stub-st-backend --stub-llm --save outputs/profiling/sample.json
```

Flags of interest:

- `--staff-limit` – cap the number of staff profiles to keep runs fast while
  still exercising the bottlenecks (defaults to 20).
- `--disable-rag` – mirror the fallback TF-IDF matcher path even if a vector
  index exists.
- `--force-tfidf` – bypass sentence-transformer embeddings entirely when the
  model weights are not available offline.
- `--stub-st-backend` / `--stub-llm` – swap network-dependent components with
  deterministic stubs so profiling remains offline.
- `--save` – emit the raw timings (per run totals and per-stage summaries) as
  JSON under `outputs/profiling/`.

The script prints stage statistics such as:

```
Total pipeline: mean=1441.3 ms, median=1433.5 ms, max=1637.6 ms
- cristin_nva_merge: mean=1370.7 ms, median=1366.3 ms, max=1603.6 ms (n=5)
- tfidf_vectorization: mean=63.0 ms, median=22.0 ms, max=228.8 ms (n=5)
```

Use these numbers to decide which optimizations to ship. The JSON output makes
it easy to compare before/after runs in Git history or spreadsheets.
