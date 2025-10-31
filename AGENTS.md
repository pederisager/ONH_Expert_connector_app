# Repository Guidelines

## Project Structure & Module Organization
Keep feature code under `app/` with `main.py` as the FastAPI entry point and routers in `routes.py`. Matching utilities live in `match_engine.py`, `llm_explainer.py`, `fetch_utils.py`; parsers remain in `file_parser.py`. Front-end assets live in `app/static/` (plain HTML/CSS/JS). Configuration and curated data sit in `data/` (`staff.yaml`, `models.yaml`, `app.config.yaml`); treat them as source of truth. Use `models/` for large model checkpoints and `outputs/exports/` for generated PDFs/JSON. Mirror modules with test files under `tests/`.

## Build, Test, and Development Commands
Create a virtual environment before installing dependencies: `python -m venv .venv && source .venv/bin/activate`. Install Python packages with `pip install -r requirements.txt`. Run the API locally with `uvicorn app.main:app --reload` so the static UI served from `app/static` stays current. Run the full suite with `pytest` (add `--cov=app` before merging). Static assets are plain files in `app/static/`; no bundler is needed.

## Coding Style & Naming Conventions
Follow PEP 8 with `black` (line length 88) and validate imports/order with `ruff` (`ruff check app tests`). Use type hints for new Python functions. Modules and files are snake_case; classes are PascalCase; constants are UPPER_SNAKE_CASE. Keep JavaScript in `app/static/app.js` formatted with Prettier (2-space indent). Configuration keys in YAML are lower-case kebab-case to match existing examples.

## Testing Guidelines
Write unit tests with `pytest` and `pytest-asyncio` for async endpoints. Name tests after the module under test (e.g., `tests/test_match_engine.py::test_rank_ordering`). Back matcher features with fixtures referencing `data/staff.yaml`. Maintain ≥85% coverage on `app/` and cover the `/match` route in integration tests. Use temporary directories when touching cache paths.

## Commit & Pull Request Guidelines
Adopt Conventional Commits (`feat:`, `fix:`, `docs:`) and limit subject lines to 72 characters. Reference relevant config or spec files (e.g., `ONH_Expert_Connector_Structure.txt`) in the body. Each PR should describe functional changes, list config updates, and add screenshots or JSON snippets for UI or export tweaks. Request review only when CI (lint + tests) is green and include a checklist confirming cache, data, and security settings were considered.

## Security & Configuration Tips
Never add external domains to `data/app.config.yaml` without approval; keep `allowlist_domains` tight. Store staff metadata solely in `data/staff.yaml` and cite their public sources. Scrub uploads in `fetch_utils.py` for unsupported MIME types and enforce the `max_upload_mb` guardrails. Rotate cache contents by respecting `retention_days`, and document experimental model changes inside `models/README.txt`.
