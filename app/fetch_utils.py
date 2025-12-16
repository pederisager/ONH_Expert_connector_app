"""Utilities for fetching and sanitizing remote content."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


class FetchNotAllowedError(RuntimeError):
    """Raised when a URL is outside the configured allowlist."""


class FetchUtils:
    """Fetch and normalize HTML content within an explicit allowlist."""

    def __init__(
        self,
        allowlist_domains: Sequence[str],
        max_kb_per_page: int = 100,
        offline_snapshots_dir: str | Path | None = None,
    ) -> None:
        self.allowlist_domains = set(allowlist_domains)
        self.max_bytes_per_page = max_kb_per_page * 1024
        self.offline_snapshots_dir = (
            Path(offline_snapshots_dir).resolve() if offline_snapshots_dir else None
        )

    def _is_allowed(self, url: str) -> bool:
        hostname = urlparse(url).hostname or ""
        return any(
            hostname == domain or hostname.endswith("." + domain)
            for domain in self.allowlist_domains
        )

    def _offline_snapshot_path(self, url: str) -> Path | None:
        if not self.offline_snapshots_dir:
            return None
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.offline_snapshots_dir / f"{digest}.json"

    def _load_offline_snapshot(self, url: str) -> Dict[str, Any] | None:
        path = self._offline_snapshot_path(url)
        if not path or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        text = (payload.get("text") or "")[: self.max_bytes_per_page]
        return {"url": url, "title": payload.get("title", ""), "text": text}

    async def fetch_page(self, client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
        if not self._is_allowed(url):
            offline = self._load_offline_snapshot(url)
            if offline:
                return offline
            raise FetchNotAllowedError(url)
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
        except httpx.HTTPError:
            offline = self._load_offline_snapshot(url)
            if offline:
                return offline
            raise
        content = response.content[: self.max_bytes_per_page]
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        title = soup.title.string if soup.title else ""
        text_content = " ".join(soup.stripped_strings)
        return {"url": url, "title": title, "text": text_content}

    async def fetch_many(self, urls: Sequence[str]) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = [self.fetch_page(client, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        pages: List[Dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            pages.append(result)
        return pages
