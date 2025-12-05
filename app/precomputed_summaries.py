"""Utilities for loading precomputed staff summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_precomputed_summaries(path: str | Path) -> Dict[str, str]:
    """Load precomputed summaries keyed by profile_url or name."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    payload = data.get("summaries") if isinstance(data, dict) else data
    summaries: Dict[str, str] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str):
                summaries[str(key)] = value.strip()
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            summary = (item.get("summary") or "").strip()
            if not summary:
                continue
            key = item.get("profile_url") or item.get("name")
            if key:
                summaries[str(key)] = summary
    return summaries
