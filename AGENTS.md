# Repository Guidelines

## Agent Maintenance
Codex must automatically update this document whenever its changes introduce, remove, or materially alter guidance that future agents need to follow. Make that edit in the same task so the new expectations are captured immediately. Skip touching `AGENTS.md` for routine prompts that leave the policies and workflows unchanged.

## Execution Safeguards
- Wrap every potentially long-running shell command (index builds, sync scripts, uvicorn, pytest, etc.) in an explicit timeout so Codex never relies on a human to stop a hung process. Prefer the POSIX `timeout` utility or command-specific flags (e.g., `pytest --maxfail=1 --timeout=900`).
- Before launching background servers or loops, document in the task notes how to stop them and enforce an upper bound on runtime (e.g., `timeout 1200 uvicorn ...`).
- If a tool lacks built-in timeout support, run it via `timeout <seconds> bash -lc "..."` and surface the timeout value in the response so future agents know the guard is active.
- When writing automation/scripts, bake the timeout policy into the code (e.g., `subprocess.run(..., timeout=300)`), and mention the safeguard in the PR/task summary so reviewers know it exists.

## Project Structure & Module Organization
Keep feature code under `app/` with `main.py` as the FastAPI entry point and routers in `routes.py`. RAG components should be split between the offline index builder (`app/index/` packages) and the online retriever/orchestrator (`app/rag/` helpers) that power `/match`. Legacy utilities (`match_engine.py`, `llm_explainer.py`, `fetch_utils.py`, `file_parser.py`) remain until fully refactored or wrapped by the new flow. Front-end assets live in `app/static/` (plain HTML/CSS/JS). Configuration, curated data, and generated indexes sit in `data/` (`staff.yaml`, `models.yaml`, `app.config.yaml`, `index/`); treat YAML files as the source of truth. Use `models/` for large model checkpoints and `outputs/exports/` for generated PDFs/JSON. Mirror modules with test files under `tests/`.

Legacy references labeled "Cristin" point to the same NVA research data; new ingestion lives under `app/index/nva`, and older Cristin-specific files are kept only for historical reference. `staff.csv` is the user-facing source of staff entries; regenerate downstream files via `scripts/update_staff.sh`.

Staff `tags` in `data/staff.yaml` now surface as "Nokkelord" on the results/cards UI and contribute to the RAG score, so keep them descriptive, deduplicated, and localized where possible.

Staff documents fetched from ONH/NVA sources are cached as whole `StaffDocument` objects (see `routes._fetch_staff_documents`) and warmed during FastAPI startup so repeat "Finn relevante ansatte" queries stay responsive. Preserve this cache and update the invalidation rules if you change how sources are assembled. `StaffDocument.combined_text` is capped at roughly 6k characters to keep TF-IDF costs predictable; adjust tests and docs if you tweak that limit.

Citation snippets returned from `/match` now use a theme-aware sentence window and allow up to ~3000 characters (see `routes.CITATION_SNIPPET_LIMIT`) so the LLM receives full chunk context. Avoid shrinking this limit without revalidating match quality.

## Build, Test, and Development Commands
Default to the python3 command for all scripts/CLI examples on this project (some environments reject python).
Create a virtual environment before installing dependencies: `python3 -m venv .venv && source .venv/bin/activate`. Install Python packages with `pip install -r requirements.txt` (pins torch 2.3.1+cu118 via `--extra-index-url https://download.pytorch.org/whl/cu118` so Pascal GPUs stay supported; do not bump torch without checking sm_61 coverage). Build or refresh the retrieval index after editing `data/staff.yaml` or source content (`python3 -m app.index.build`). Run the API locally with `uvicorn app.main:app --reload` so the static UI served from `app/static` stays current. Run the full suite with `pytest` (add `--cov=app` before merging). Static assets are plain files in `app/static`—no bundler is needed. The default embedding backend is `sentence-transformers` on CUDA; ensure the virtualenv has the pinned CUDA-enabled PyTorch + `sentence-transformers`, or update `data/models.yaml` to another backend (e.g., Ollama) and rebuild the index if you deviate. Leaving `embedding_model.device` empty or set to `auto` now lets the embedder factory auto-detect CUDA (then MPS, then CPU) for both the offline index builder and online retriever—only pin it to `cpu` if you explicitly need to disable accelerators.

- Keep `staff.csv` as the entry point. Run `bash scripts/update_staff.sh [key_file] [base_url]` to refresh `data/staff.yaml`, `data/staff_records.jsonl`, sync NVA publications to `data/nva/results.jsonl`, and rebuild the index. Defaults: `nva_api_keys_test.json`, `https://api.test.nva.aws.unit.no`. API key files are git-ignored; never commit secrets.
- Profile `/match` hot spots with `python3 scripts/profile_match.py ...` (see `docs/performance_profiling.md`). The script reuses the production helpers and records per-stage timings (NVA merge, TF-IDF, embeddings, LLM explainer) so we can justify performance tweaks before merging.

You have access to the mpc tool context7. Use it for documentation lookup whenever that would be beneficial to solving a task. 

## Coding Style & Naming Conventions
Follow PEP 8 with `black` (line length 88) and validate imports/order with `ruff` (`ruff check app tests`). Use type hints for new Python functions. Modules and files are snake_case; classes are PascalCase; constants are UPPER_SNAKE_CASE. Keep JavaScript in `app/static/app.js` formatted with Prettier (2-space indent). Configuration keys in YAML are lower-case kebab-case to match existing examples.

## Testing Guidelines
Write unit tests with `pytest` and `pytest-asyncio` for async endpoints. Name tests after the module under test (e.g., `tests/test_match_engine.py::test_rank_ordering`). Back RAG features with fixtures referencing `data/staff.yaml` and seeding a lightweight index under a temp directory. Maintain â‰¥85% coverage on `app/` and cover the `/match` route in integration tests, ensuring retrieved passages and explanations remain grounded. Use temporary directories when touching cache or index paths.

## Commit & Pull Request Guidelines
Adopt Conventional Commits (`feat:`, `fix:`, `docs:`) and limit subject lines to 72 characters. Reference relevant config or spec files (e.g., `ONH_Expert_Connector_Structure.txt`) in the body. Each PR should describe functional changes, list config updates, and add screenshots or JSON snippets for UI or export tweaks. Request review only when CI (lint + tests) is green and include a checklist confirming cache, data, and security settings were considered.

## Security & Configuration Tips
Never add external domains to `data/app.config.yaml` without approval; keep `allowlist_domains` tight. Store staff metadata solely in `data/staff.yaml` and cite their public sources. Scrub uploads in `fetch_utils.py` for unsupported MIME types and enforce the `max_upload_mb` guardrails. Rotate cache contents by respecting `retention_days`, and document experimental model changes inside `models/README.txt`.

If future agents detect that the NVA search endpoints have stopped responding, assume the service may have been deprecated and immediately notify the requester. Suggest rebuilding the integration against the latest NVA REST API as a fallback.




