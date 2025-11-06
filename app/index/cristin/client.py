"""HTTP client wrapper for the Cristin REST API."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urljoin

import httpx

from app.cache_manager import CacheManager

LOGGER = logging.getLogger("cristin.client")


def _canonicalize_params(params: dict[str, Any] | None) -> str:
    if not params:
        return ""
    items: list[tuple[str, Any]] = []
    for key, value in sorted(params.items()):
        if isinstance(value, (list, tuple)):
            for sub in value:
                items.append((key, sub))
        else:
            items.append((key, value))
    return urlencode(items, doseq=True)


@dataclass(slots=True)
class CristinClientConfig:
    base_url: str = "https://api.cristin.no/v2"
    lang: str = "en"
    per_page: int = 1000
    request_timeout: float = 30.0
    retry_attempts: int = 3
    retry_backoff: float = 1.5
    throttle_delay: float = 0.0


class CristinClient:
    """Lightweight wrapper around the Cristin REST API."""

    def __init__(
        self,
        *,
        config: CristinClientConfig | None = None,
        cache: CacheManager | None = None,
        session: httpx.Client | None = None,
    ) -> None:
        self.config = config or CristinClientConfig()
        self.cache = cache
        self._session = session
        self._owns_session = session is None

    @property
    def session(self) -> httpx.Client:
        if self._session is None:
            timeout = httpx.Timeout(self.config.request_timeout)
            self._session = httpx.Client(timeout=timeout)
        return self._session

    def close(self) -> None:
        if self._owns_session and self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "CristinClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    # -- Public API -----------------------------------------------------

    def get_person_results(self, person_id: str) -> list[dict[str, Any]]:
        """Retrieve all results for a person, transparently following pagination."""
        endpoint = f"/persons/{person_id}/results"
        params = {"lang": self.config.lang, "per_page": self.config.per_page}
        results: list[dict[str, Any]] = []
        next_url: str | None = self._build_url(endpoint)

        while next_url:
            payload = self._request_json(next_url, params if next_url.endswith(endpoint) else None)
            if isinstance(payload, list):
                results.extend(payload)
            else:
                LOGGER.warning("Unexpected payload for %s: %s", endpoint, type(payload))
            next_url = self._get_next_link()
            params = None  # subsequent pages include query params in link header
        return results

    def get_result(self, result_id: str) -> dict[str, Any]:
        """Fetch detail payload for a single result."""
        endpoint = f"/results/{result_id}"
        params = {"lang": self.config.lang}
        url = self._build_url(endpoint)
        payload = self._request_json(url, params)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict result for {result_id}, got {type(payload)}")
        return payload

    # -- Internal helpers ------------------------------------------------

    def _build_url(self, endpoint: str) -> str:
        base = self.config.base_url.rstrip("/") + "/"
        path = endpoint.lstrip("/")
        return urljoin(base, path)

    def _request_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        cache_key = None
        if self.cache:
            param_suffix = _canonicalize_params(params)
            cache_key = f"{url}?{param_suffix}" if param_suffix else url
            cached = self.cache.get(cache_key)
            if cached is not None:
                return json.loads(cached)

        payload = self._do_request(url, params)
        if self.cache and cache_key:
            self.cache.set(cache_key, json.dumps(payload, ensure_ascii=False))
        return payload

    def _do_request(self, url: str, params: dict[str, Any] | None) -> Any:
        attempt = 0
        last_exc: Exception | None = None
        while attempt < self.config.retry_attempts:
            try:
                self._last_link_header = None
                if self.config.throttle_delay:
                    time.sleep(self.config.throttle_delay)
                response = self.session.get(url, params=params, headers={"Accept": "application/json"})
                response.raise_for_status()
                self._last_link_header = response.headers.get("Link")
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_exc = exc
                attempt += 1
                sleep_for = self.config.retry_backoff**attempt
                LOGGER.warning(
                    "Request failure (%s), attempt %s/%s, sleeping %.2fs",
                    exc,
                    attempt,
                    self.config.retry_attempts,
                    sleep_for,
                )
                time.sleep(sleep_for)
        raise RuntimeError(f"Failed to fetch {url}") from last_exc

    def _get_next_link(self) -> str | None:
        header = getattr(self, "_last_link_header", None)
        if not header:
            return None
        parts = [part.strip() for part in header.split(",")]
        for part in parts:
            if 'rel="next"' in part:
                url_part = part.split(";")[0].strip()
                if url_part.startswith("<") and url_part.endswith(">"):
                    return url_part[1:-1]
        return None
