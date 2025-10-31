# ONH Expert Connector

Local-first tool that helps ONH teachers surface relevant colleagues for a course, lecture, or material. The app ingests a topic description and optional files, fetches pre-approved staff sources, and returns a shortlist of staff with grounded explanations and exportable summaries.

## Project structure

- `app/`: FastAPI service and support modules.
- `app/static/`: Plain HTML/CSS/JS single-page frontend.
- `data/`: Hand-maintained YAML configuration files.
- `models/`: Notes and placeholders for local model checkpoints.
- `outputs/exports/`: Generated shortlist PDFs/JSON exports.
- `tests/`: Pytest suite mirroring the `app/` modules.

See `ONH_Expert_Connector_Concept.txt` and `ONH_Expert_Connector_Structure.txt` for full functional specs and UI guidance.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API serves the static UI from `app/static` at the root path.

## Local models

Model choices live in `data/models.yaml`. Download the referenced LLM and embedding checkpoints ahead of time (for example via Ollama) and update the config if you pick different variants.

## Offline test scenario

- The fetcher now falls back to curated offline snapshots under `data/offline_snapshots` when network access is unavailable.
- Use `data/test_inputs/klinisk_psykologi_tema.txt` as a stable prompt: paste its text into the Tema field and optionally upload the same file (TXT is supported).
- Click “Analyser tema”, review the suggested themes, then “Finn relevante ansatte”. You should see at least one psychology profile (e.g. Alex Gillespie) with citations sourced from the offline snapshot.
- Clear the shortlist or rerun with other prompts as needed; cached results live under `data/cache`.

## Testing and linting

```bash
pytest
ruff check app tests
black app tests
```

Aim for ≥85% coverage on `app/` and include `/match` integration tests before merging changes.
