# Backend Analysis Workflow

## Plain-Language Walkthrough
- **Staff knowledge base lives locally:** Admins curate `data/staff.yaml` with each person’s profile, department, tags, and approved source links. A background “index builder” script fetches those sources, cleans the text, splits them into short passages, and stores them—together with semantic fingerprints—in a local library so the app can search instantly.
- **Teacher request → themes:** When you describe a teaching need (and optionally upload documents), the service cleans the text, finds the main themes, and creates its own semantic fingerprint for the request. Empty submissions are rejected so the system always has real input to work with.
- **Retrieving evidence:** Using that fingerprint and the themes, the app searches the local library for the most relevant passages. Department filters or curator tags narrow the search so only staff from the right area surface.
- **Ranking candidates:** The retrieved passages are grouped by staff member. Each person is scored based on how many strong passages they have, how closely those passages match the themes, and how well they line up with any selected department or tags. Diversity nudges keep the list from showing only a single team.
- **Building explanations:** For the top matches, the app shows the exact snippets it found and sends them—plus the themes—to a local language model. The model writes a short explanation grounded in the supplied snippets. If generation fails, a template message using the same snippets is returned instead.
- **Shortlists and exports:** The UI can still save interesting people to a shortlist, which is stored locally. You can export that list as PDF or JSON with the evidence snippets and any optional metadata (topic text, selected department, themes).

## Developer Notes
- **High-level architecture:**
  - Keep FastAPI as the entry point (`app/main.py` / `app/routes.py`) but introduce two new subsystems: an offline *index builder* that maintains the retrieval corpus, and an online *RAG orchestrator* that handles query-time retrieval + generation.
  - Application state now includes handles to the vector store (e.g., FAISS/Chroma), a passage store (flattened text chunks with metadata), and a retriever component that can execute filtered similarity searches.
- **Index building pipeline:**
  - Triggered manually by admins or as a scheduled job. Reads `data/staff.yaml`, fetches allowed source pages through `FetchUtils`, and stores raw HTML snapshots in `data/offline_snapshots` (for audit/debug).
  - Cleans and chunks content (≈200–400 tokens per chunk, overlap ~20%), then embeds each chunk with the configured embedding model (`data/models.yaml`). Store embeddings + metadata (staff id, department, source URL, tag vector) in the vector store; store normalized chunk text in an accompanying document DB (could be simple SQLite/Parquet files).
  - Records ingestion timestamps so outdated chunks can be refreshed or pruned without rebuilding everything.
  - **Source-specific extraction:**
    - *oslonyehoyskole.no staff pages:* Parse the HTML snapshot with BeautifulSoup, locate `<div class="col-md-8 col-lg-8">`, strip scripts/ads, and capture headings, paragraphs, and bullet lists that describe teaching focus, roles, and qualifications. Keep the cleaned HTML/text as the canonical profile summary for that staff member.
    - *nva.sikt.no research profiles:* Iterate result listings on the profile page (each publication card). For every listed item, capture the title, abstract/summary text, publication year, and URL. Flatten these into discrete passages (e.g., `"{Title} — {Abstract}"`) so the index retains both the high-level topic and the detailed description of the research.
    - Store extraction artifacts (raw HTML + normalized text) under `data/offline_snapshots/<staff_id>/` for traceability. If automated fetching fails or approvals restrict network access, provide a CLI flag to ingest pre-downloaded HTML/JSON files so admins can run the same parser offline.
  - **Automation vs. manual ingestion:**
    - Preferred path is a scripted fetch using the allowlisted domains and the above extractors. The script can run under developer credentials when network access is granted.
    - When automation is blocked (e.g., air-gapped environments), admins can download the relevant pages manually (Save as HTML/JSON), place them in a staging folder, and rerun the index builder in `--offline` mode so the same parsing logic updates the local staff data without live HTTP calls.
- **Request-time flow (`/analyze-topic`, `/match`):**
  - `/analyze-topic` still parses uploads via `FileParser` and extracts themes (`extract_themes`). Additionally, pre-compute a query embedding for later reuse.
  - `/match` loads the pre-computed query embedding; if the client skips `/analyze-topic`, compute it on the fly. Execute a similarity search against the vector index with optional metadata filters (department, curated tags). Retrieve top-N chunks per staff, optionally rerank with cross-encoder or heuristic scoring.
  - Aggregate chunk-level scores into staff-level scores (weighted sum of max chunk similarity, tag overlap, department bonus, diversity penalty). Keep citation snippets and metadata for explanation.
- **Generation (`LLMExplainer`):**
  - Prompt now references chunk metadata (source title, URL, staff context) and includes multiple evidence passages. Strict instructions ensure the model cites provided evidence only.
  - Fallback path remains deterministic for environments without a running LLM.
- **Caching & storage:**
  - Disk cache still used for HTTP fetches. Vector/document stores live under `data/index/` (new) and are loaded once at startup.
  - Shortlist persistence and export pipeline remain unchanged.
- **Admin operations:**
  - Provide CLI/management command (future work) to run the index builder after editing `staff.yaml` so new/updated profiles are embedded before teachers query.
- **Testing touchpoints:**
  - Unit tests cover chunking, embedding adapters, retriever ranking, and RAG prompt construction.
  - Integration tests seed a small vector index fixture and verify `/match` returns staff and explanations grounded in indexed passages.
