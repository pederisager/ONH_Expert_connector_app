"""Utilities for loading precomputed staff summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_precomputed_summaries_by_language(path: str | Path) -> dict[str, dict[str, str]]:
    """Load precomputed summaries keyed by language then profile_url/name.

    Supported JSON formats:
    - Legacy: {"summaries": {"Name": "...", "url": "..."}}
    - New: {"summaries": {"no": {...}, "en": {...}}}
    """
    path = Path(path)
    if not path.exists():
        return {"no": {}, "en": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"no": {}, "en": {}}

    payload = data.get("summaries") if isinstance(data, dict) else data
    if payload is None:
        return {"no": {}, "en": {}}

    def _coerce_mapping(obj) -> dict[str, str]:
        summaries: dict[str, str] = {}
        if not isinstance(obj, dict):
            return summaries
        for key, value in obj.items():
            if isinstance(value, str):
                summaries[str(key)] = value.strip()
        return summaries

    def _coerce_list(items) -> dict[str, str]:
        summaries: dict[str, str] = {}
        if not isinstance(items, list):
            return summaries
        for item in items:
            if not isinstance(item, dict):
                continue
            summary = (item.get("summary") or "").strip()
            if not summary:
                continue
            key = item.get("profile_url") or item.get("name")
            if key:
                summaries[str(key)] = summary
        return summaries

    # New shape: summaries is a dict with language keys.
    if isinstance(payload, dict) and any(isinstance(v, dict) for v in payload.values()):
        by_lang: dict[str, dict[str, str]] = {"no": {}, "en": {}}
        for lang, value in payload.items():
            if not isinstance(lang, str):
                continue
            normalized = lang.lower()
            if isinstance(value, dict):
                by_lang[normalized] = _coerce_mapping(value)
            elif isinstance(value, list):
                by_lang[normalized] = _coerce_list(value)
        by_lang.setdefault("no", {})
        by_lang.setdefault("en", {})
        return by_lang

    # Legacy shape.
    if isinstance(payload, list):
        return {"no": _coerce_list(payload), "en": {}}
    return {"no": _coerce_mapping(payload), "en": {}}


def load_precomputed_summaries(path: str | Path) -> Dict[str, str]:
    """Legacy loader: returns Norwegian summaries as a flat mapping."""
    return load_precomputed_summaries_by_language(path).get("no", {})
