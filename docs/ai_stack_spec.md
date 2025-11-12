# AI Stack Specification — ONH Expert Connector

*Version 1.0 — Maintained in `docs/` so non-developers can update it directly. When you edit this file, write in plain language and highlight any open questions so the developer (Codex) can confirm the intended behavior before coding.*

## How to use this document
- **For you (non-dev editor):** Update the bullet points or tables below to describe the behavior you expect. Avoid code-level detail—focus on outcomes, data sources, and constraints. Call out anything you are unsure about under “Open questions”.
- **For Codex (developer):** Treat every instruction here as source-of-truth architecture guidance. If something conflicts with reality, note it in the PR and propose revisions to this spec.
- **When making changes:**
  1. Add today’s date to the changelog entry.
  2. Summarize what changed and why.
  3. Ping Codex (in chat or task description) to review the updates before implementation.

## Changelog
- *2024-XX-XX — Initial draft (Codex).* Covers the five layers shown in the AI stack diagram: Infrastructure, Model, Data, Orchestration, Application.

## Stack overview (read this first)
| Layer | What it is | Primary owners/files | Key decisions we control |
| --- | --- | --- | --- |
| **Application** | FastAPI service + static frontend that teachers interact with. | `app/main.py`, `app/routes.py`, `app/static/` | UI flow, endpoints, shortlist/export behavior, user messaging. |
| **Orchestration** | How requests move through parsing → retrieval → explanation. | `app/routes.py`, `app/match_engine.py`, `app/index/` helpers | Execution order, fallbacks, caching, determinism rules. |
| **Data** | Configuration, snapshots, and vector indexes the system reads. | `data/staff.yaml`, `data/index/`, `data/offline_snapshots/`, `app/index/build.py` | Which staff/content are included, how often we refresh, chunk sizes. |
| **Model** | Embedding + LLM components used in retrieval and explanation. | `data/models.yaml`, `app/index/embedder_factory.py`, `app/rag/llm_explainer.py` | Model families, hardware targets, temperature/top-k settings, deterministic fallbacks. |
| **Infrastructure** | Environment and runtime setup that makes everything run locally. | Local workstation (RTX 4070 target), Python venv, `requirements.txt`, `uvicorn` | Deployment mode (local-only vs. LAN), hardware assumptions, security constraints. |

> **Tip for editors:** If you are not sure which layer a change belongs to, jot it under the “Open questions” section and tag Codex to clarify.

---

## Infrastructure layer
**Mission:** Provide a fully local environment (no cloud dependencies) that can run FastAPI, the retrieval index, and lightweight language models on commodity hardware (target: RTX 4070 laptop/desktop).

**Current stance**
- Runs as a Python virtual environment (`python -m venv .venv`), started with `uvicorn app.main:app --reload` for development.
- Assumes access to CUDA for faster embeddings but must fall back to CPU if unavailable (auto-detected in the embedder factory).
- Network exposure stays on `localhost` by default; no public internet calls occur at runtime (all data is pre-fetched).
- Security guardrails: strict URL allowlist (`data/app.config.yaml`), file upload limits (size/type), and antivirus scan hooks in `fetch_utils.py`.

**Decisions you can request here**
- Allow the app to bind to a different interface (e.g., LAN demo) — specify expected audience and security requirements.
- Approve or deny use of alternative hardware (e.g., CPU-only mode, different GPU) and note any performance tolerance.
- Call out compliance requirements (data retention, logging policy) that Codex should enforce.

**Constraints & risks**
- GPU memory (12 GB) limits model sizes; larger checkpoints require quantization (documented under Model layer).
- Because everything runs locally, storage growth in `data/offline_snapshots/` and `data/index/` must be monitored.
- No background daemons should run without explicit timeout safeguards (see repository AGENTS instructions).

**Open questions**
- _Example_: “Can we support a kiosk mode where the app is accessible on a shared lab computer? What security changes would that need?”

---

## Model layer
**Mission:** Configure and operate the embedding model and the explanation LLM so that retrieval and generated rationales stay fast, local, and grounded in indexed content.

**Current components**
- **Embeddings:** Default is `sentence-transformers/all-MiniLM-L6-v2` (CUDA-preferred). Managed through `data/models.yaml` and instantiated in `app/index/embedder_factory.py`.
- **LLM explainer:** Small local instruction-tuned model (e.g., Llama 3.2 3B Instruct or Mistral 7B Instruct) invoked via `app/rag/llm_explainer.py`. Deterministic fallback generates template-based explanations if the LLM is unavailable.
- **Quantization & runtime:** Models can run via Ollama or Hugging Face. Decisions live in `data/models.yaml`; Codex updates the runtime adapters accordingly.

**Decisions you can request here**
- Swap to a different embedding model (specify name, performance expectations, and whether GPU is required).
- Adjust explanation tone/length — describe desired voice, and Codex will tune prompts or fallback templates.
- Define acceptable latency ceilings for embeddings and explanation generation.

**Constraints & risks**
- Larger LLMs may not fit in memory; must confirm with Codex before upscaling.
- Embedding model changes require rebuilding the vector index (`python -m app.index.build`).
- Prompt adjustments must preserve grounded citations; Codex will highlight if a request risks hallucinations.

**Open questions**
- _Example_: “Should the explanations include Norwegian translations automatically?”

---

## Data layer
**Mission:** Maintain a curated, offline dataset of ONH staff profiles and related materials that the retrieval system can search quickly and safely.

**Current components**
- **Configuration:** `data/staff.yaml` lists staff metadata, allowed source URLs, departments, and tags (“Nøkkelord”).
- **Snapshots:** `data/offline_snapshots/` stores fetched HTML/text from approved URLs, produced by the index builder.
- **Vector index:** `data/index/` contains embeddings and metadata for retrieval (FAISS/Chroma backend, built via `app/index/build.py`).
- **Supporting configs:** `data/app.config.yaml` controls chunk sizes, fetch limits, cache retention; `data/models.yaml` ties data refresh rules to model choices.

**Decisions you can request here**
- Add/remove staff entries or tags — edit `data/staff.yaml` in plain YAML; Codex will validate schema and rebuild the index.
- Set refresh cadence (e.g., “rebuild snapshots monthly”) so Codex can script automation.
- Specify new document types or URL patterns to include (ensure they are publicly accessible and compliant).

**Constraints & risks**
- All sources must be on the approved domain allowlist; expanding the list needs explicit approval.
- Uploaded teacher files are processed temporarily and should not be stored beyond session scope unless specified.
- Large uploads slow down parsing; confirm acceptable size limits before raising them.

**Open questions**
- _Example_: “Do we need to prioritize Norwegian-language sources over English ones when both exist?”

---

## Orchestration layer
**Mission:** Define how user input flows through preprocessing, retrieval, ranking, and explanation so results stay deterministic, reproducible, and quick.

**Current flow (runtime)**
1. **Input handling:** `/analyze-topic` parses teacher text + optional files, extracts candidate themes, and returns editable chips to the UI.
2. **Search:** `/match` takes confirmed themes, runs deterministic retrieval via `app/match_engine.py`, and aggregates scores per staff member.
3. **Explanation:** Retrieves top passages, then calls the LLM explainer with guardrails (marching-ants style highlighting, citation metadata).
4. **Shortlist/export:** UI can add staff to a shortlist saved client-side; exports (PDF/JSON) generated via backend utilities.

**Determinism rules**
- Seeded random states ensure consistent ranking on identical inputs.
- Immediate propagation: any change to input themes re-triggers retrieval without delay.
- Fallback logic keeps explanations functional even if the LLM fails.

**Decisions you can request here**
- Change the ranking criteria (e.g., emphasize department matches more heavily) — describe desired weighting in plain terms.
- Introduce new steps (e.g., manual approval before explanations) — outline the user journey and Codex will map it to code.
- Adjust caching behavior or timeout thresholds for long-running steps.

**Constraints & risks**
- Must preserve marching-ants visualization and clamp behaviors defined in the UI logic.
- Adding asynchronous/background jobs requires timeout safeguards per repository policy.
- Complex re-ranking logic should remain explainable; if you propose a rule, state the rationale so Codex can implement transparent scoring.

**Open questions**
- _Example_: “Should we allow teachers to pin certain staff so they always appear in results?”

---

## Application layer
**Mission:** Deliver a simple, reliable interface for teachers to discover relevant colleagues, grounded in the backend orchestration.

**Current components**
- **Backend:** FastAPI app (`app/main.py`) with routers in `app/routes.py`. Serves static assets and exposes `/analyze-topic`, `/match`, shortlist endpoints.
- **Frontend:** Plain HTML/CSS/JS under `app/static/`. Handles input forms, marching-ants edge animations, slider clamp behavior, shortlist interactions, and export buttons.
- **Exports:** PDF/JSON generated through utilities in `app/` (see `outputs/` for generated files).
- **Testing:** Pytest suite in `tests/` validates retrieval logic, caching, and endpoint responses.

**Decisions you can request here**
- UI copy updates — provide exact text and indicate where it belongs (e.g., “Replace the button label on the home screen with ‘Finn eksperter’”).
- Workflow tweaks — describe the before/after experience (e.g., “Add a confirmation modal before exporting PDF”).
- Accessibility/visual adjustments — specify goals (contrast, keyboard navigation) so Codex can translate them into HTML/CSS changes.

**Constraints & risks**
- Must keep existing invariants: deterministic propagation, immediate source updates, marching-ants animation, ephemeral clamp behavior.
- Avoid introducing heavy new frontend dependencies without discussion; stick to vanilla JS unless justified.
- Any backend change must maintain API compatibility unless the UI spec also changes.

**Open questions**
- _Example_: “Do we want to add an onboarding tooltip for first-time users?”

---

## Coordination checklist for editors
When you finish editing this spec:
1. Review each layer for conflicting requests. If something contradicts another section, highlight it.
2. Update the changelog with today’s date and your summary.
3. Share the updated file with Codex so they can confirm feasibility and create implementation tasks.
4. Expect Codex to reference this document in PR descriptions to show how changes align with the agreed stack.

*This document is meant to stay high-level and accessible. If you need deeper technical details (e.g., function names, performance metrics), ask Codex to create supporting notes in `docs/reference/` rather than bloating this spec.*
