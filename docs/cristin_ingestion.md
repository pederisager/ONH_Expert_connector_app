# Cristin Research Results Ingestion

This module downloads research results for Oslo Nye Høyskole staff using the Cristin REST API v2. The ingested dataset powers downstream RAG indexing by providing per-employee topical expertise summaries.

## Workflow Summary

1. Read `staff.csv` (UTF-8 with BOM) and extract each employee’s Cristin person ID from the `NVA_profile` URL. Employees without an NVA profile are skipped and logged.
2. For each person, call `GET /persons/{id}/results` (paginated, up to 1000 items per page) to collect every declared result.
3. For each result, call `GET /results/{id}` to obtain detailed metadata (titles, summaries, category, venue, keywords, contributor preview, etc.).
4. Persist raw API responses under `data/cristin/raw/<person_id>/` (`results_listing.json` plus one JSON file per result) to aid auditing and incremental refresh.
5. Normalize the payloads into RAG-friendly records containing only topical expertise fields, deduplicate results with identical titles per employee (keeping the richest metadata), and emit them as newline-delimited JSON in `data/cristin/results.jsonl`.

## Running the Sync CLI

```bash
python -m app.index.cristin.sync \
  --staff-csv staff.csv \
  --output data/cristin/results.jsonl \
  --raw-dir data/cristin/raw \
  --cache-dir data/cache/cristin \
  --lang en
```

Flags of note:

- `--force-refresh` disables the disk cache (`data/cache/cristin`) and refetches everything.
- `--throttle-delay <seconds>` adds a sleep before each request if manual rate limiting is required.
- `--per-page` controls pagination size (Cristin allows up to 1000).

The CLI logs the total staff processed, results fetched, and normalized entries written. Raw directories and the JSONL file are recreated on each run.

## Language Choice

The ingestion defaults to `lang=en`, yielding the richest abstracts and categorical labels for the majority of records. Switching to Norwegian Bokmål is as simple as passing `--lang nb`, but note that summaries may be sparser.

## Relevant Fields Retained

Each JSONL row (and the `ResultRecord` dataclass) includes:

- `employee_id`, `employee_name`
- `cristin_result_id`, `source_url`
- `title`, `summary`, `venue`
- `category_code`, `category_name`, `year_published`
- `tags` (classification keywords plus category label)
- contributor preview (names and role codes if provided)
- supplemental context such as original language, media type, pages, and declared funding/project codes

These fields emphasize topical expertise while omitting operational metadata (identifiers, open-access flags, internal timestamps).

## Integrating with the Index Builder

Downstream index builders can stream `data/cristin/results.jsonl` and combine the records with other curated datasets (e.g., `data/staff.yaml`). The newline-delimited format enables incremental appends and plays nicely with Python generators or tools like `jq`.

## Failure Modes

- **Empty result sets:** logged at INFO; the employee simply has no Cristin entries.
- **API errors:** requests retry up to three times with exponential backoff. Persistent failures raise a `RuntimeError`.
- **Deprecated API:** if endpoints start returning 404/410, follow the guidance in `AGENTS.md` to notify maintainers and rebuild against the live NVA REST API.
