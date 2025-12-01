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
python3 -m venv .venv  # run only to (re)create .venv
source .venv/bin/activate
pip install -r requirements.txt  # only run once
uvicorn app.main:app --reload
```

The API serves the static UI from `app/static` at the root path.

## NVA API keys and data refresh

- Place your NVA API credentials in `nva_api_keys_test.json` or `nva_api_keys_prod.json` (these files are git-ignored). Structure:
  ```json
  {
    "clientId": "<client id>",
    "clientSecret": "<client secret>",
    "tokenUrl": "<oauth token endpoint>"
  }
  ```
- One-shot refresh after editing `staff.csv` (test example):
  ```bash
  bash scripts/update_staff.sh
  ```
  or point at prod:
  ```bash
  bash scripts/update_staff.sh nva_api_keys_prod.json https://api.nva.unit.no
  ```
- The script will: refresh `data/staff.yaml` + `data/staff_records.jsonl` from `staff.csv`, sync NVA publications to `data/nva/results.jsonl`, and rebuild the index under `data/index/`.
  If the GPU build fails, it automatically retries on CPU.
- Fetch publications from the NVA API manually (test example):
  ```bash
  python -m app.index.nva.sync --key-file nva_api_keys_test.json --base-url https://api.test.nva.aws.unit.no --output data/nva/results.jsonl
  ```
- Rebuild the local retrieval index after syncing data:
  ```bash
  python -m app.index.build --nva-results data/nva/results.jsonl
  ```

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
4. **Ensure the embedding backend is ready.** The default configuration uses `sentence-transformers` with CUDA; make sure your virtualenv has `torch`/`sentence-transformers` installed (see “GPU embeddings” below) and that `nvidia-smi` shows your GPU. If you switch back to Ollama, start `ollama serve` separately.
5. **Confirm required models are available (first boot or after updates to `data/models.yaml`).** For SentenceTransformers, download the specified Hugging Face model ahead of time (`python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"`). For Ollama-based setups, run `ollama pull` for the configured models.
6. **Install or update Python dependencies (first boot or requirements change only).**
  ```bash
  pip install -r requirements.txt
  ```
7. **Build or refresh the retrieval index (after editing `data/staff.yaml` or when sources change).**
   ```bash
   python -m app.index.build
   ```
8. **Run the API with live reload.**
  ```bash
  uvicorn app.main:app --reload
  ```
9. **Visit the UI.** Open http://127.0.0.1:8000/ in a browser to load the static frontend served by FastAPI.

## Local models

Model choices live in `data/models.yaml`. Download the referenced LLM and embedding checkpoints ahead of time (for example via Ollama) and update the config if you pick different variants.

## GPU embeddings

The retriever defaults to `sentence-transformers/all-MiniLM-L6-v2` running on CUDA (`data/models.yaml`). To actually leverage your GPU:

1. Install CUDA-enabled PyTorch plus `sentence-transformers` inside the virtualenv:
   ```bash
   pip install --upgrade "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121
   pip install sentence-transformers
   ```
2. Verify access with `python -c "import torch; print(torch.cuda.is_available())"` and `nvidia-smi`.
3. Rebuild the index so embeddings are regenerated on the GPU:
   ```bash
   python -m app.index.build --verbose
   ```
4. Start the API (`uvicorn app.main:app --reload`). If CUDA is missing, `app/index/embedder_factory.py` logs a warning and retries on CPU, so you still get a working retriever.

If you prefer Ollama for embeddings, set `embedding_model.backend` back to `ollama`, configure `endpoint`, and rebuild the index after the Ollama service is running (optionally with GPU acceleration).

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
