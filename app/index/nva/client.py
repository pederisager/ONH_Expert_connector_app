"""HTTP client wrapper for the NVA REST API."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin

import httpx

from app.cache_manager import CacheManager

LOGGER = logging.getLogger("nva.client")


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
class NvaApiKey:
    client_id: str
    client_secret: str
    token_url: str

    @classmethod
    def from_file(cls, path: Path) -> "NvaApiKey":
        payload = json.loads(path.read_text(encoding="utf-8"))
        client_id = payload.get("clientId") or payload.get("client_id")
        client_secret = payload.get("clientSecret") or payload.get("client_secret")
        token_url = payload.get("tokenUrl") or payload.get("token_url")
        if not client_id or not client_secret or not token_url:
            raise ValueError(f"NVA key file {path} is missing required fields.")
        return cls(client_id=str(client_id), client_secret=str(client_secret), token_url=str(token_url))


@dataclass(slots=True)
class NvaClientConfig:
    base_url: str = "https://api.nva.unit.no"
    results_per_page: int = 100
    request_timeout: float = 30.0
    retry_attempts: int = 3
    retry_backoff: float = 1.5
    throttle_delay: float = 0.0


class NvaClient:
    """Lightweight wrapper around the NVA REST API."""

    def __init__(
        self,
        *,
        api_key: NvaApiKey,
        config: NvaClientConfig | None = None,
        cache: CacheManager | None = None,
        session: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.config = config or NvaClientConfig()
        self.cache = cache
        self._session = session
        self._owns_session = session is None
        self._token_value: str | None = None
        self._token_expires_at: float = 0.0

    @property
    def session(self) -> httpx.Client:
        if self._session is None:
            timeout = httpx.Timeout(self.config.request_timeout)
            self._session = httpx.Client(timeout=timeout, base_url=self.config.base_url)
        return self._session

    def close(self) -> None:
        if self._owns_session and self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "NvaClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    # -- Public API -----------------------------------------------------

    def get_publications_for_person(self, person_id: str) -> list[dict[str, Any]]:
        """Retrieve all publications for a contributor, following pagination."""
        endpoint = "/search/resources"
        params = {
            "contributor": self._person_uri(person_id),
            "size": max(1, min(1000, self.config.results_per_page)),
        }
        results: list[dict[str, Any]] = []
        next_url: str | None = self._build_url(endpoint)

        while next_url:
            payload = self._request_json(next_url, params if next_url.endswith(endpoint) else None)
            hits = payload.get("hits") if isinstance(payload, dict) else None
            if isinstance(hits, list):
                results.extend(hit for hit in hits if isinstance(hit, dict))
            next_url = payload.get("nextResults") if isinstance(payload, dict) else None
            params = None
        return results

    # -- Internal helpers ------------------------------------------------

    def _build_url(self, endpoint: str) -> str:
        base = self.config.base_url.rstrip("/") + "/"
        path = endpoint.lstrip("/")
        return urljoin(base, path)

    def _person_uri(self, person_id: str) -> str:
        base = self.config.base_url.rstrip("/") + "/"
        return urljoin(base, f"cristin/person/{person_id}")

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
                if self.config.throttle_delay:
                    time.sleep(self.config.throttle_delay)
                headers = {"Accept": "application/json"}
                token = self._ensure_token()
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                response = self.session.get(url, params=params, headers=headers)
                response.raise_for_status()
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

    def _ensure_token(self) -> str:
        now = time.time()
        if self._token_value and now < self._token_expires_at:
            return self._token_value
        token, expires_in = self._fetch_token()
        buffer = 60
        self._token_value = token
        self._token_expires_at = now + max(60, expires_in - buffer)
        return token

    def _fetch_token(self) -> tuple[str, int]:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": self.api_key.client_id,
            "client_secret": self.api_key.client_secret,
        }
        response = self.session.post(self.api_key.token_url, headers=headers, data=data, auth=(self.api_key.client_id, self.api_key.client_secret))
        response.raise_for_status()
        payload = response.json()
        token = payload.get("access_token")
        expires_in = int(payload.get("expires_in") or 900)
        if not token:
            raise RuntimeError("Token response missing access_token")
        return str(token), expires_in
