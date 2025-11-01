# ONH Expert Connector

Local-first tool that helps ONH teachers surface relevant colleagues for a course, lecture, or material. The app ingests a topic description and optional files, retrieves supporting passages from a locally indexed knowledge base of staff content (RAG), and returns a shortlist of staff with grounded explanations and exportable summaries.

## Project structure

- `app/`: FastAPI service and support modules.
- `app/static/`: Plain HTML/CSS/JS single-page frontend.
- `data/`: Hand-maintained YAML configuration files and the on-disk retrieval index.
- `models/`: Notes and placeholders for local model checkpoints.
- `outputs/exports/`: Generated shortlist PDFs/JSON exports.
- `tests/`: Pytest suite mirroring the `app/` modules.

See `ONH_Expert_Connector_Concept.txt` and `backend_analysis_process.md` for full functional specs, RAG backend flow, and UI guidance.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API serves the static UI from `app/static` at the root path.

## Daily run checklist

1. **Open a terminal.** Use your preferred shell (macOS Terminal, Windows WSL, etc.).
2. **Go to the project directory.**
   ```bash
   cd /path/to/ONH_Expert_connector_app
   ```
3. **Activate the virtual environment.** If the `.venv` folder is missing, create it with `python -m venv .venv` (or `python3`) first.
   ```bash
   source .venv/bin/activate
   ```
4. **Ensure Ollama is running.** Start the local service in another terminal if needed (examples: `ollama serve`, `launchctl load /Library/LaunchDaemons/com.ollama.ollama.plist`, or `sudo systemctl start ollama`).
5. **Confirm required models are available (first boot or after updates to `data/models.yaml`).**
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.1:8b-instruct-q4_0
  ```
6. **Install or update Python dependencies (first boot or requirements change only).**
  ```bash
  pip install -r requirements.txt
  ```
7. **Build or refresh the retrieval index (after editing `data/staff.yaml` or when sources change).**
   ```bash
   python -m app.index.build  # placeholder command; implement during RAG migration
   ```
8. **Run the API with live reload.**
  ```bash
  uvicorn app.main:app --reload
  ```
9. **Visit the UI.** Open http://127.0.0.1:8000/ in a browser to load the static frontend served by FastAPI.

## Local models

Model choices live in `data/models.yaml`. Download the referenced LLM and embedding checkpoints ahead of time (for example via Ollama) and update the config if you pick different variants.

## Offline test scenario

- The retriever reads passages from the local index in `data/index/` (run the index build step first).
- Use `data/test_inputs/klinisk_psykologi_tema.txt` as a stable prompt: paste its text into the Tema field and optionally upload the same file (TXT is supported).
- Click “Analyser tema”, review the suggested themes, then “Finn relevante ansatte”. You should see at least one psychology profile (e.g. Alex Gillespie) with citations sourced from the indexed passages.
- Clear the shortlist or rerun with other prompts as needed; cached fetch snapshots live under `data/offline_snapshots` and vector data under `data/index/`.

## Testing and linting

```bash
pytest
ruff check app tests
black app tests
```

Aim for ≥85% coverage on `app/` and include `/match` integration tests before merging changes.
