"""Lightweight lookup helpers for NVA publications (DOI/api URL fallback).

We read `data/nva/raw/**/<id>.json` once and cache a mapping from
`id -> {doi: str|None}`. Resolution rules:
1. If a DOI exists, return `https://doi.org/<doi>` (unless already a URL).
2. Otherwise return an API URL constructed from the publication id. We reuse
   the host from the original source URL when available; default host is
   https://api.test.nva.aws.unit.no.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

NVA_ID_RE = re.compile(r"(0[0-9a-f]{3}[0-9a-f\-]{28,})", re.IGNORECASE)
RAW_ROOT = Path(__file__).resolve().parent.parent / "data" / "nva" / "raw"
DEFAULT_API_HOST = "api.test.nva.aws.unit.no"


def _load_index() -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    if not RAW_ROOT.exists():
        return index
    for path in RAW_ROOT.glob("**/*.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, list):
            # Some exports may wrap the payload; pick the first dict if present
            obj = obj[0] if obj and isinstance(obj[0], dict) else {}
        pub_id_raw = obj.get("id") or ""
        # IDs in raw files may already be full URLs; extract trailing segment
        if isinstance(pub_id_raw, str) and pub_id_raw.startswith("http"):
            pub_id = pub_id_raw.rstrip("/").split("/")[-1].strip()
        else:
            pub_id = str(pub_id_raw).strip()
        if not pub_id:
            continue
        doi = (
            obj.get("entityDescription", {})
            .get("reference", {})
            .get("doi")
        )
        index[pub_id] = {"doi": str(doi or "").strip()}
    return index


# Preload at import to keep per-request latency low (startup cost preferred).
DOI_INDEX = _load_index()


def extract_pub_id(url: str | None) -> str | None:
    if not url:
        return None
    match = NVA_ID_RE.search(url)
    return match.group(1) if match else None


def _build_api_url(pub_id: str, source_url: str | None) -> str:
    host = DEFAULT_API_HOST
    if source_url:
        parsed = urlparse(source_url)
        if parsed.hostname and "api" in parsed.hostname:
            host = parsed.hostname
    return f"https://{host}/publication/{pub_id}"


def preferred_nva_url(pub_id: str | None, source_url: str | None = None) -> str | None:
    """Return DOI if present, else an API URL built from the publication id."""
    if not pub_id:
        return None
    data = DOI_INDEX.get(pub_id, {})
    doi = data.get("doi")
    if doi:
        return doi if doi.lower().startswith("http") else f"https://doi.org/{doi}"
    return _build_api_url(pub_id, source_url)
