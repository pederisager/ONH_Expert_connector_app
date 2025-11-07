# Repository Guidelines

## Agent Maintenance
Codex must automatically update this document whenever its changes introduce, remove, or materially alter guidance that future agents need to follow. Make that edit in the same task so the new expectations are captured immediately. Skip touching `AGENTS.md` for routine prompts that leave the policies and workflows unchanged.

## Project Structure & Module Organization
Keep feature code under `app/` with `main.py` as the FastAPI entry point and routers in `routes.py`. RAG components should be split between the offline index builder (`app/index/` packages) and the online retriever/orchestrator (`app/rag/` helpers) that power `/match`. Legacy utilities (`match_engine.py`, `llm_explainer.py`, `fetch_utils.py`, `file_parser.py`) remain until fully refactored or wrapped by the new flow. Front-end assets live in `app/static/` (plain HTML/CSS/JS). Configuration, curated data, and generated indexes sit in `data/` (`staff.yaml`, `models.yaml`, `app.config.yaml`, `index/`); treat YAML files as the source of truth. Use `models/` for large model checkpoints and `outputs/exports/` for generated PDFs/JSON. Mirror modules with test files under `tests/`.

## Build, Test, and Development Commands
Create a virtual environment before installing dependencies: `python -m venv .venv && source .venv/bin/activate`. Install Python packages with `pip install -r requirements.txt`. Build or refresh the retrieval index after editing `data/staff.yaml` or source content (`python -m app.index.build`). Run the API locally with `uvicorn app.main:app --reload` so the static UI served from `app/static` stays current. Run the full suite with `pytest` (add `--cov=app` before merging). Static assets are plain files in `app/static`—no bundler is needed. The default embedding backend is `sentence-transformers` on CUDA; ensure the virtualenv has CUDA-enabled PyTorch + `sentence-transformers`, or update `data/models.yaml` to another backend (e.g., Ollama) and rebuild the index if you deviate.

## Coding Style & Naming Conventions
Follow PEP 8 with `black` (line length 88) and validate imports/order with `ruff` (`ruff check app tests`). Use type hints for new Python functions. Modules and files are snake_case; classes are PascalCase; constants are UPPER_SNAKE_CASE. Keep JavaScript in `app/static/app.js` formatted with Prettier (2-space indent). Configuration keys in YAML are lower-case kebab-case to match existing examples.

## Testing Guidelines
Write unit tests with `pytest` and `pytest-asyncio` for async endpoints. Name tests after the module under test (e.g., `tests/test_match_engine.py::test_rank_ordering`). Back RAG features with fixtures referencing `data/staff.yaml` and seeding a lightweight index under a temp directory. Maintain ≥85% coverage on `app/` and cover the `/match` route in integration tests, ensuring retrieved passages and explanations remain grounded. Use temporary directories when touching cache or index paths.

## Commit & Pull Request Guidelines
Adopt Conventional Commits (`feat:`, `fix:`, `docs:`) and limit subject lines to 72 characters. Reference relevant config or spec files (e.g., `ONH_Expert_Connector_Structure.txt`) in the body. Each PR should describe functional changes, list config updates, and add screenshots or JSON snippets for UI or export tweaks. Request review only when CI (lint + tests) is green and include a checklist confirming cache, data, and security settings were considered.

## Security & Configuration Tips
Never add external domains to `data/app.config.yaml` without approval; keep `allowlist_domains` tight. Store staff metadata solely in `data/staff.yaml` and cite their public sources. Scrub uploads in `fetch_utils.py` for unsupported MIME types and enforce the `max_upload_mb` guardrails. Rotate cache contents by respecting `retention_days`, and document experimental model changes inside `models/README.txt`.

If future agents detect that the Cristin API endpoints have stopped responding, assume the service may have been deprecated and immediately notify the requester. Suggest rebuilding the integration against the current NVA REST API as a fallback.
