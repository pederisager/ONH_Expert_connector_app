from __future__ import annotations

import hashlib
import json

import httpx
import pytest
from app.fetch_utils import FetchUtils


@pytest.mark.asyncio
async def test_fetch_page_uses_offline_snapshot(monkeypatch, tmp_path) -> None:
    url = "https://oslonyehoyskole.no/om-oss/Ansatte/Test-Psykolog"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshots_dir / f"{digest}.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "url": url,
                "title": "Test Psykolog",
                "text": "Klinisk psykologi og traumebehandling i praksis.",
            }
        ),
        encoding="utf-8",
    )

    fetch_utils = FetchUtils(
        allowlist_domains=["oslonyehoyskole.no"],
        offline_snapshots_dir=snapshots_dir,
    )

    async def fake_get(self, url: str, timeout: float = 30.0):  # type: ignore[override]
        raise httpx.RequestError("offline", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get, raising=False)

    async with httpx.AsyncClient() as client:
        page = await fetch_utils.fetch_page(client, url)

    assert page["title"] == "Test Psykolog"
    assert "traumebehandling" in page["text"]


@pytest.mark.asyncio
async def test_fetch_page_returns_snapshot_for_disallowed_domain(tmp_path) -> None:
    url = "https://example.com/offline"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshots_dir / f"{digest}.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "url": url,
                "title": "Offline Example",
                "text": "Testrapport for maskinlæring.",
            }
        ),
        encoding="utf-8",
    )

    fetch_utils = FetchUtils(
        allowlist_domains=["oslonyehoyskole.no"],
        offline_snapshots_dir=snapshots_dir,
    )

    async with httpx.AsyncClient() as client:
        page = await fetch_utils.fetch_page(client, url)

    assert page["title"] == "Offline Example"
    assert "maskinlæring" in page["text"]
