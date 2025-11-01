"""CLI utility for refreshing `data/staff.yaml` from public sources."""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote_plus, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


BASE_URL = "https://oslonyehoyskole.no"
STAFF_LIST_ENDPOINT = f"{BASE_URL}/ansatte/ansatte_data"
ROLE_KEYWORDS = ("professor", "førsteamanuensis", "høyskolelektor")
NVA_BASE_URL = "https://nva.sikt.no"
LOGGER = logging.getLogger("refresh_staff")


@dataclass(slots=True)
class StaffListing:
    """Minimal information obtained from the public roster JSON."""

    name: str
    title: str
    department: str
    profile_url: str
    slug: str


@dataclass(slots=True)
class StaffProfile:
    """Detailed staff information gathered during refresh."""

    listing: StaffListing
    summary_text: str
    html_path: Path
    sources: list[str] = field(default_factory=list)
    nva_profile_url: str | None = None

    def add_source(self, url: str) -> None:
        if url not in self.sources:
            self.sources.append(url)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh staff.yaml using Oslo Nye Høyskole and NVA sources."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned changes without writing to staff.yaml.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Read from existing snapshots without performing network requests.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("data/offline_snapshots"),
        help="Directory where HTML/JSON snapshots are stored.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/staff.yaml"),
        help="Path to the staff.yaml file to update.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for troubleshooting.",
    )
    parser.add_argument(
        "--nva-api-base",
        type=str,
        default=os.environ.get("NVA_API_BASE", "https://api.nva.unit.no"),
        help="Base URL for the NVA search API (override with NVA_API_BASE).",
    )
    parser.add_argument(
        "--nva-api-key",
        type=str,
        default=os.environ.get("NVA_API_KEY"),
        help="Bearer token for the NVA search API (defaults to NVA_API_KEY env var).",
    )
    parser.add_argument(
        "--nva-max-results",
        type=int,
        default=8,
        help="Maximum number of NVA API search hits to inspect per staff member.",
    )
    return parser


class SnapshotFetcher:
    """Fetches remote content and persists snapshots for offline reuse."""

    def __init__(self, *, snapshot_dir: Path, offline: bool) -> None:
        self.snapshot_dir = snapshot_dir
        self.offline = offline
        self._client: httpx.Client | None = None

    def __enter__(self) -> "SnapshotFetcher":
        if not self.offline:
            self._client = httpx.Client(
                timeout=httpx.Timeout(30.0),
                headers={
                    "User-Agent": (
                        "ONH-Expert-Connector/refresh-staff (+https://oslonyehoyskole.no)"
                    )
                },
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if self._client is not None:
            self._client.close()
            self._client = None

    def _ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def fetch_text(
        self,
        url: str,
        snapshot_path: Path,
        *,
        binary: bool = False,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> str:
        if snapshot_path.exists():
            LOGGER.debug("Loading snapshot: %s", snapshot_path)
            data = snapshot_path.read_bytes() if binary else snapshot_path.read_text(encoding="utf-8")
            return data.decode("utf-8") if binary else data

        if self.offline:
            raise FileNotFoundError(
                f"Snapshot missing for offline mode: {snapshot_path}"
            )

        if self._client is None:
            raise RuntimeError("HTTP client is not initialized.")

        LOGGER.debug("Fetching %s", url)
        response = self._client.get(url, headers=headers, params=params)
        response.raise_for_status()
        content = response.content
        self._ensure_parent(snapshot_path)
        snapshot_path.write_bytes(content)
        return content.decode("utf-8")

    def fetch_json(
        self,
        url: str,
        snapshot_path: Path,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        text = self.fetch_text(
            url,
            snapshot_path,
            headers=headers,
            params=params,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - guardrail
            raise ValueError(f"Invalid JSON from {url}") from exc


def normalize_department(raw: str) -> str:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parts[0] if parts else raw.strip()


def normalize_profile_url(path: str) -> str:
    if path.startswith("http"):
        return path
    return urljoin(BASE_URL, path)


def slug_from_profile_url(profile_url: str) -> str:
    parsed = urlparse(profile_url)
    slug = Path(parsed.path).name
    if not slug:
        raise ValueError(f"Unable to derive slug from profile URL: {profile_url}")
    return slug


def should_include_role(title: str) -> bool:
    lowered = title.lower()
    return any(keyword in lowered for keyword in ROLE_KEYWORDS)


def normalize_person_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    filtered = "".join(ch for ch in decomposed if ch.isalnum())
    return filtered.casefold()


def tokenize_person_name(value: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", value)
    tokens: list[str] = []
    current: list[str] = []
    for char in decomposed:
        if char.isalpha():
            current.append(char.casefold())
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def name_similarity(candidate: str, target: str) -> float:
    candidate_norm = normalize_person_name(candidate)
    target_norm = normalize_person_name(target)
    if not candidate_norm or not target_norm:
        return 0.0
    target_tokens = tokenize_person_name(target)
    candidate_tokens = tokenize_person_name(candidate)
    if target_tokens and not all(token in candidate_tokens for token in target_tokens):
        return 0.0
    return difflib.SequenceMatcher(None, candidate_norm, target_norm).ratio()


def extract_profile_summary(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", class_="col-md-8 col-lg-8")
    if container is None:
        raise ValueError("Expected profile content container not found.")
    texts: list[str] = []
    for element in container.find_all(["h2", "h3", "h4", "p", "li"]):
        text = element.get_text(separator=" ", strip=True)
        if text:
            texts.append(text)
    summary = "\n".join(texts)
    return summary


class StaffCollector:
    """Pipeline for gathering Oslo Nye staff roster information."""

    def __init__(self, fetcher: SnapshotFetcher) -> None:
        self.fetcher = fetcher

    def collect_staff_profiles(self) -> list[StaffProfile]:
        listings = self._load_listings()
        profiles: list[StaffProfile] = []
        for listing in listings:
            profile = self._collect_profile(listing)
            if profile is not None:
                profiles.append(profile)
        return profiles

    def _load_listings(self) -> list[StaffListing]:
        snapshot_path = self.fetcher.snapshot_dir / "oslo_staff_index.json"
        payload = self.fetcher.fetch_json(STAFF_LIST_ENDPOINT, snapshot_path)
        if not isinstance(payload, list):
            raise ValueError("Unexpected staff index payload structure.")
        listings: list[StaffListing] = []
        for entry in payload:
            try:
                title = str(entry.get("field_ansatte_stilling", "")).strip()
                if not should_include_role(title):
                    continue
                name = str(entry.get("field_ansatte_navn") or entry.get("title") or "").strip()
                if not name:
                    continue
                department = normalize_department(str(entry.get("field_avdeling") or "").strip())
                profile_path = str(entry.get("view_node") or "").strip()
                if not profile_path:
                    continue
                profile_url = normalize_profile_url(profile_path)
                slug = slug_from_profile_url(profile_url)
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.warning("Skipping entry due to parsing error: %s (%s)", entry, exc)
                continue
            listing = StaffListing(
                name=name,
                title=title,
                department=department or "Ukjent",
                profile_url=profile_url,
                slug=slug,
            )
            listings.append(listing)
        LOGGER.info("Collected %d staff listings matching role filter.", len(listings))
        return listings

    def _collect_profile(self, listing: StaffListing) -> StaffProfile | None:
        snapshot_rel = Path("staff") / listing.slug / "profile.html"
        snapshot_path = self.fetcher.snapshot_dir / snapshot_rel
        try:
            html = self.fetcher.fetch_text(listing.profile_url, snapshot_path)
        except FileNotFoundError:
            LOGGER.error("Missing snapshot for %s in offline mode.", listing.name)
            return None
        except httpx.HTTPError as exc:  # pragma: no cover - network issue guard
            LOGGER.warning("Failed to fetch profile for %s (%s)", listing.name, exc)
            return None

        try:
            summary = extract_profile_summary(html)
        except ValueError as exc:
            LOGGER.warning("Profile parsing issue for %s: %s", listing.name, exc)
            summary = ""

        sources = [listing.profile_url]
        return StaffProfile(
            listing=listing,
            summary_text=summary,
            html_path=snapshot_path,
            sources=sources,
        )


class NvaResolver:
    """Resolves NVA research profile URLs for staff members."""

    SEARCH_PATH = "/search"
    API_SEARCH_PATH = "/search/resources"

    def __init__(
        self,
        fetcher: SnapshotFetcher,
        *,
        api_base: str = "https://api.nva.unit.no",
        api_key: str | None = None,
        max_results: int = 8,
    ) -> None:
        self.fetcher = fetcher
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.max_results = max(1, max_results)

    def attach_profile_urls(self, profiles: Sequence[StaffProfile]) -> tuple[int, int]:
        resolved = 0
        for profile in profiles:
            url = self._resolve_single(profile)
            if url:
                profile.nva_profile_url = url
                profile.add_source(url)
                resolved += 1
        unresolved = len(profiles) - resolved
        return resolved, unresolved

    def _resolve_single(self, profile: StaffProfile) -> str | None:
        listing = profile.listing
        api_url = self._resolve_via_api(profile)
        if api_url:
            return api_url
        return self._resolve_via_html(listing)

    def _resolve_via_api(self, profile: StaffProfile) -> str | None:
        listing = profile.listing
        if not self.fetcher.offline and not self.api_key:
            LOGGER.debug(
                "Skipping NVA API lookup for %s: missing API key.", listing.name
            )
            return None

        snapshot_rel = Path("staff") / listing.slug / "nva_search.json"
        snapshot_path = self.fetcher.snapshot_dir / snapshot_rel
        endpoint = f"{self.api_base}{self.API_SEARCH_PATH}"
        params = {
            "query": listing.name,
            "offset": "0",
            "limit": str(self.max_results),
            "type": "PERSON",
        }
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            payload = self.fetcher.fetch_json(
                endpoint,
                snapshot_path,
                headers=headers,
                params=params,
            )
        except FileNotFoundError:
            if self.fetcher.offline:
                LOGGER.warning(
                    "Missing NVA API snapshot for %s while running offline.",
                    listing.name,
                )
            else:
                LOGGER.debug(
                    "No cached NVA API snapshot for %s; continuing without API data.",
                    listing.name,
                )
            return None
        except httpx.HTTPError as exc:  # pragma: no cover - network guard
            LOGGER.warning("NVA API lookup failed for %s (%s)", listing.name, exc)
            return None
        except ValueError as exc:
            LOGGER.warning("Invalid JSON from NVA API for %s: %s", listing.name, exc)
            return None

        candidates = self._extract_api_candidates(payload, listing)
        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_url, best_name = candidates[0]
        LOGGER.debug(
            "Top NVA API candidate for %s: %s (score %.3f)",
            listing.name,
            best_name,
            best_score,
        )
        return best_url if best_score >= 0.6 else None

    def _resolve_via_html(self, listing: StaffListing) -> str | None:
        snapshot_rel = Path("staff") / listing.slug / "nva_search.html"
        snapshot_path = self.fetcher.snapshot_dir / snapshot_rel
        query = quote_plus(listing.name)
        search_url = f"{urljoin(NVA_BASE_URL, self.SEARCH_PATH)}?query={query}"
        try:
            html = self.fetcher.fetch_text(search_url, snapshot_path)
        except FileNotFoundError:
            LOGGER.warning(
                "Missing NVA search snapshot for %s while running offline.",
                listing.name,
            )
            return None
        except httpx.HTTPError as exc:  # pragma: no cover - network guardrail
            LOGGER.warning(
                "Failed to fetch NVA search results for %s (%s)", listing.name, exc
            )
            return None

        url = self._parse_html_search(html, listing)
        if url:
            return url

        LOGGER.info("Unable to locate NVA profile for %s.", listing.name)
        return None

    def _parse_html_search(self, html: str, listing: StaffListing) -> str | None:
        soup = BeautifulSoup(html, "html.parser")
        candidates: list[tuple[float, str, str]] = []
        for anchor in soup.select("a[href*='/research-profile/']"):
            href = anchor.get("href")
            if not href:
                continue
            candidate_url = urljoin(NVA_BASE_URL, href)
            candidate_name = anchor.get_text(" ", strip=True)
            if not candidate_name:
                continue
            score = name_similarity(candidate_name, listing.name)
            if score <= 0.0:
                continue
            candidates.append((score, candidate_url, candidate_name))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_url, best_name = candidates[0]
        LOGGER.debug(
            "Top NVA candidate for %s: %s (score %.3f)",
            listing.name,
            best_name,
            best_score,
        )
        return best_url if best_score >= 0.6 else None

    def _extract_api_candidates(
        self, payload: Any, listing: StaffListing
    ) -> list[tuple[float, str, str]]:
        seen: set[str] = set()
        candidates: list[tuple[float, str, str]] = []

        for url, name in self._iter_api_results(payload):
            if url in seen:
                continue
            seen.add(url)
            score = name_similarity(name, listing.name)
            if score <= 0.0:
                continue
            candidates.append((score, url, name))
        return candidates

    def _iter_api_results(self, payload: Any) -> Iterable[tuple[str, str]]:
        def walk(node: Any) -> Iterable[tuple[str, str]]:
            if isinstance(node, dict):
                url = self._extract_profile_url(node)
                if url:
                    name = self._extract_profile_name(node)
                    if name:
                        yield (url, name)
                for value in node.values():
                    yield from walk(value)
            elif isinstance(node, list):
                for item in node:
                    yield from walk(item)

        return walk(payload)

    def _extract_profile_url(self, node: dict[str, Any]) -> str | None:
        candidate_keys = ("uri", "url", "href", "externalId", "identifier", "id")
        for key in candidate_keys:
            value = node.get(key)
            if isinstance(value, str):
                trimmed = value.strip()
                if "/research-profile/" in trimmed:
                    return self._normalize_profile_url(trimmed)
                if trimmed.isdigit():
                    return urljoin(NVA_BASE_URL, f"/research-profile/{trimmed}")
                if trimmed.startswith("research-profile/"):
                    return urljoin(NVA_BASE_URL, trimmed)
            elif isinstance(value, int):
                return urljoin(NVA_BASE_URL, f"/research-profile/{value}")
        return None

    def _extract_profile_name(self, node: dict[str, Any]) -> str | None:
        name_keys = (
            "name",
            "displayName",
            "title",
            "mainTitle",
            "presentationName",
            "preferredName",
            "preferredFirstName",
            "preferredLastName",
        )
        for key in name_keys:
            if key in node:
                result = self._string_from_any(node[key])
                if result:
                    return result
        return None

    def _string_from_any(self, value: Any) -> str | None:
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        if isinstance(value, dict):
            for inner_value in value.values():
                result = self._string_from_any(inner_value)
                if result:
                    return result
        elif isinstance(value, list):
            for item in value:
                result = self._string_from_any(item)
                if result:
                    return result
        return None

    def _normalize_profile_url(self, url: str) -> str:
        if url.startswith("http"):
            return url
        return urljoin(NVA_BASE_URL, url if url.startswith("/") else f"/{url}")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    fetcher = SnapshotFetcher(snapshot_dir=args.snapshot_dir, offline=args.offline)
    if args.nva_api_key:
        LOGGER.info("NVA API key supplied; API lookup enabled.")
    else:
        LOGGER.info(
            "NVA API key not provided; relying on cached API snapshots or HTML search."
        )

    profiles: list[StaffProfile]
    with fetcher:
        collector = StaffCollector(fetcher)
        profiles = collector.collect_staff_profiles()
        resolver = NvaResolver(
            fetcher,
            api_base=args.nva_api_base,
            api_key=args.nva_api_key,
            max_results=args.nva_max_results,
        )
        resolved, unresolved = resolver.attach_profile_urls(profiles)

    LOGGER.info("Fetched details for %d staff profiles.", len(profiles))
    LOGGER.info(
        "Resolved NVA profiles for %d staff; %d unmatched.", resolved, unresolved
    )
    LOGGER.info("Dry run: %s", "yes" if args.dry_run else "no")

    # Further processing (NVA discovery, YAML update) is handled in subsequent steps.
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
