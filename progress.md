# Current Task Snapshot

## Overall goal
- Prepare `data/staff.yaml` for the upcoming RAG backend by automatically refreshing staff metadata from public sources.
- Only include ONH personnel with titles containing `Professor`, `Førsteamanuensis`, or `Høyskolelektor`.
- Augment each staff record with two authoritative URLs:
  1. ONH profile page on `oslonyehoyskole.no`.
  2. Matching researcher profile on `nva.sikt.no`, capturing publication titles/abstracts.

## Work completed so far
- **Project planning:** Documented six-step plan (S1–S6) covering roster fetch, detail extraction, NVA lookup, YAML merge, tooling, and testing.
- **Module scaffolding:** Added `app/index/__init__.py` and CLI stub `app/index/refresh_staff.py`.
- **Roster ingestion (S1 & S2):**
  - Implemented `SnapshotFetcher` to download/save snapshots under `data/offline_snapshots/`, with offline reuse support.
  - Implemented `StaffCollector` to call `https://oslonyehoyskole.no/ansatte/ansatte_data` (JSON) and filter entries by role keywords.
  - Normalized department names (first comma-separated value) and produced canonical profile URLs/slugs.
  - For each qualifying staff member, downloaded their profile page and parsed `<div class="col-md-8 col-lg-8">` into a plain-text summary.
  - Stored profile HTML snapshots at `data/offline_snapshots/staff/<slug>/profile.html`.
- **NVA lookup (S3 in progress):**
  - Added `NvaResolver` to prefer the documented `https://api.nva.unit.no/search/resources` endpoint (with cached `nva_search.json` snapshots) while falling back to the public HTML search when API credentials are unavailable.
  - Implemented name-normalisation heuristics (`normalize_person_name`, `tokenize_person_name`, `name_similarity`) so the resolver tolerates diacritics and hyphenated names.
  - Added targeted unit tests (`tests/test_nva_resolver.py`) covering API snapshots, HTML-only fallback, and missing-match scenarios.
- **Index foundations (S4 scaffolding):**
  - Introduced reusable chunking utilities (`app/index/chunking.py`) with tokenizer abstractions and comprehensive tests.
  - Added `LocalVectorStore` and embedding helpers to persist chunk vectors locally, plus normalization helpers and coverage.
  - Implemented `StaffIndexBuilder` (`app/index/builder.py`) that consumes `StaffRecord` data, produces chunk snapshots, writes manifests, and persists vectors (with a deterministic `DummyEmbeddingBackend` for offline runs). Covered by `tests/test_index_builder.py`.
- **Runtime RAG stubs (S5 preview):**
  - Added `app/rag/retriever.py` providing an embedding-based retriever that groups chunk hits per staff member with department filtering.
  - Added `tests/test_retriever.py` exercising retrieval grouping and filter handling.
- **Route test hardening:** Migrated FastAPI route tests to async `httpx.AsyncClient` with JSON payload support to eliminate multipart timeouts (`tests/test_routes.py`, `tests/conftest.py`), and added a JSON fallback path in `/analyze-topic` so CI stays deterministic.

## Next focus (S3 onwards)
- Once the API key arrives, validate live requests against `https://api.nva.unit.no/search/resources` and record representative JSON snapshots for regression tests.
- Capture per-staff NVA profile pages (HTML/JSON) and extract publication metadata required for the future index builder.
- Reconcile new data with existing `data/staff.yaml` (merge, sort by department/name, remove outdated entries).
- Implement diff report + dry-run output before writing YAML.
- Wire the new index builder into a `python -m app.index.build` CLI that pulls summaries from snapshots + resolver output, then emits fresh vectors/chunks.
- Add tests covering the YAML merge/diff logic and NVA JSON parsing once the API path is confirmed.
- Integrate the retriever stub with `/match`, gradually retiring the legacy match engine in favour of chunk-level evidence.

## Outstanding considerations / blockers
- Need consistent network access for:
  - Fetching `https://oslonyehoyskole.no/ansatte/ansatte_data` (already working).
  - Fetching each ONH profile page (already working).
  - Sending authenticated requests to the NVA API (pending API key); HTML fallback remains available meanwhile.
- Ensure script gracefully handles missing or ambiguous NVA matches (e.g., prompt or log for manual follow-up).
- Confirm embedders and vector store paths when moving from dummy embeddings to the production model defined in `data/models.yaml`.
