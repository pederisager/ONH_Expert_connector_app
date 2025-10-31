"""Persistence helper for shortlist items."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import orjson


class ShortlistStore:
    """Stores shortlist items in a JSON file under `outputs/exports`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return orjson.loads(self.path.read_bytes())
        except orjson.JSONDecodeError:
            return []

    def save(self, items: List[dict[str, Any]]) -> None:
        payload = orjson.dumps(items, option=orjson.OPT_INDENT_2)
        self.path.write_bytes(payload)
