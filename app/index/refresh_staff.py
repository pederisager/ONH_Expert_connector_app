"""CLI utility for refreshing `data/staff.yaml` from public sources."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote_plus, urljoin, urlparse

import httpx
import yaml
from bs4 import BeautifulSoup
from bs4.element import Tag

from ..config_loader import AppConfig, ModelsConfig, load_app_config, load_models_config
from .builder import StaffIndexBuilder
from .chunking import Chunker
from .embedder_factory import create_embedding_backend
from .models import IndexPaths, SourceLink, StaffRecord
from .staff_info_loader import load_staff_info
from .vector_store import LocalVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_URL = "https://oslonyehoyskole.no"
STAFF_LIST_ENDPOINT = f"{BASE_URL}/ansatte/ansatte_data"
DEFAULT_ROLE_KEYWORDS = ("professor", "førsteamanuensis", "høyskolelektor")
NVA_BASE_URL = "https://nva.sikt.no"
LOGGER = logging.getLogger("refresh_staff")
CONTACT_EMAIL_PATTERN = re.compile(
    r"var\s+ansatte_e_post\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE
)
EXPERT_PAGE_URL = f"{BASE_URL}/Forskning/Finn-en-ekspert"


@dataclass(slots=True)
class StaffListing:
    """Minimal information obtained from the public roster JSON."""

    name: str
    title: str
    department: str
    profile_url: str
    slug: str
    image_url: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CsvStaffEntry:
    """Admin-curated staff record sourced from staff.csv."""

    name: str
    profile_url: str
    department: str
    nva_profile_url: str | None
    slug: str


@dataclass(slots=True)
class StaffProfile:
    """Detailed staff information gathered during refresh."""

    listing: StaffListing
    summary_text: str
    html_path: Path
    sources: list[str] = field(default_factory=list)
    nva_profile_url: str | None = None
    email: str | None = None
    phone: str | None = None

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
    parser.add_argument(
        "--academic-only",
        action="store_true",
        help="Restrict ingestion to common academic titles (Professor/Førsteamanuensis/Høyskolelektor).",
    )
    parser.add_argument(
        "--role-keyword",
        dest="role_keywords",
        action="append",
        help="Restrict staff to titles containing this case-insensitive keyword. Repeat the flag to add multiple keywords. Defaults to include all roles.",
    )
    parser.add_argument(
        "--records-output",
        type=Path,
        help="Optional path to write extracted staff records as JSONL for debugging.",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        help="Override the target directory for the vector index (defaults to rag.index-root).",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip embedding/index generation even after refreshing staff metadata.",
    )
    parser.add_argument(
        "--staff-csv",
        type=Path,
        default=Path("staff.csv"),
        help="Path to staff.csv containing the allowlisted staff entries.",
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
            data = (
                snapshot_path.read_bytes()
                if binary
                else snapshot_path.read_text(encoding="utf-8")
            )
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


def load_staff_allowlist(csv_path: Path) -> list[CsvStaffEntry]:
    if not csv_path.exists():
        raise FileNotFoundError(f"staff CSV not found: {csv_path}")
    entries: list[CsvStaffEntry] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"Name", "ONH_profile", "Department"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"staff CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            name = (row.get("Name") or "").strip()
            profile_url = (row.get("ONH_profile") or "").strip()
            department = (row.get("Department") or "").strip()
            nva_url = (row.get("NVA_profile") or "").strip() or None
            if not name or not profile_url:
                LOGGER.debug("Skipping CSV row with missing name/profile URL: %s", row)
                continue
            try:
                slug = slug_from_profile_url(profile_url)
            except ValueError as exc:
                LOGGER.warning(
                    "Skipping CSV row with invalid profile URL (%s): %s",
                    profile_url,
                    exc,
                )
                continue
            entry = CsvStaffEntry(
                name=name,
                profile_url=profile_url,
                department=department or "Ukjent",
                nva_profile_url=nva_url,
                slug=slug,
            )
            entries.append(entry)
    if not entries:
        raise ValueError(f"No valid staff entries found in {csv_path}")
    return entries


def should_include_role(title: str, keywords: Sequence[str]) -> bool:
    if not keywords:
        return True
    lowered = title.casefold()
    return any(keyword in lowered for keyword in keywords)


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


def _find_profile_container(soup: BeautifulSoup) -> Tag:
    containers = soup.find_all("div", class_="col-md-8 col-lg-8")
    if not containers:
        raise ValueError("Expected profile content container not found.")
    container = None
    for candidate in containers:
        if candidate.find(class_="sidebar"):
            continue
        if candidate.find_all(["p", "h2", "h3", "h4", "li"]):
            container = candidate
            break
    if container is None:
        container = containers[0]
    return container


def extract_profile_summary(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    container = _find_profile_container(soup)
    texts: list[str] = []
    for element in container.find_all(["h2", "h3", "h4", "p", "li"]):
        text = element.get_text(separator=" ", strip=True)
        if text:
            texts.append(text)
    summary = "\n".join(texts)
    return summary


def extract_profile_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    container = _find_profile_container(soup)
    title_element = container.find(class_="ansatte_stilling") or soup.find(
        class_="ansatte_stilling"
    )
    if title_element is None:
        heading = container.find(["h2", "h3"], class_="ansatte_stilling") or soup.find(
            ["h2", "h3"], class_="ansatte_stilling"
        )
        title_element = heading
    if title_element is None:
        heading = container.find(["h2", "h3"]) or soup.find(["h2", "h3"])
        title_element = heading
    if title_element is None:
        return None
    text = title_element.get_text(separator=" ", strip=True)
    return text or None


def parse_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        values = raw
    else:
        cleaned = str(raw or "").replace(";", ",")
        values = cleaned.split(",")
    return [item.strip() for item in values if item and item.strip()]


def extract_contact_details(html: str) -> tuple[str | None, str | None]:
    email: str | None = None
    phone: str | None = None

    email_match = CONTACT_EMAIL_PATTERN.search(html)
    if email_match:
        candidate = email_match.group(1).strip()
        if candidate:
            email = candidate

    soup = BeautifulSoup(html, "html.parser")
    phone_container = soup.select_one("div.telephone span")
    if phone_container:
        candidate = phone_container.get_text(strip=True)
        if candidate:
            phone = candidate
    return email, phone


def load_expert_topics(
    fetcher: SnapshotFetcher,
    csv_entries: Sequence[CsvStaffEntry],
) -> dict[str, list[str]]:
    """Return mapping of staff slug to expert topics from Finn-en-ekspert."""

    snapshot_path = fetcher.snapshot_dir / "experts" / "experts.html"
    try:
        html = fetcher.fetch_text(EXPERT_PAGE_URL, snapshot_path)
    except Exception as exc:  # pragma: no cover - network guard
        LOGGER.warning("Unable to fetch expert topic listing: %s", exc)
        return {}

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="table")
    if not tables:
        LOGGER.warning("Expert topic tables not found on Finn-en-ekspert page.")
        return {}

    name_index: dict[str, list[str]] = {}
    for entry in csv_entries:
        key = normalize_person_name(entry.name)
        name_index.setdefault(key, []).append(entry.slug)

    topics: dict[str, set[str]] = {}

    for table in tables:
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) != 2:
                continue
            topic_raw = cells[0].get_text(" ", strip=True)
            if not topic_raw:
                continue
            if cells[0].find("strong"):
                continue
            topic = " ".join(topic_raw.split())
            anchors = cells[1].find_all("a")
            if not anchors:
                continue
            for anchor in anchors:
                name = anchor.get_text(" ", strip=True)
                if not name:
                    continue
                key = normalize_person_name(name)
                matches = name_index.get(key)
                if not matches:
                    LOGGER.debug(
                        "Expert topic '%s' references unknown staff '%s'.", topic, name
                    )
                    continue
                if len(matches) > 1:
                    LOGGER.warning(
                        "Ambiguous expert match for name '%s'; skipping topic '%s'.",
                        name,
                        topic,
                    )
                    continue
                slug = matches[0]
                topics.setdefault(slug, set()).add(topic)

    enriched = {slug: sorted(values) for slug, values in topics.items() if values}
    LOGGER.info(
        "Enriched expert tags for %d staff from Finn-en-ekspert.", len(enriched)
    )
    return enriched


class StaffCollector:
    """Pipeline for gathering Oslo Nye staff roster information."""

    def __init__(
        self,
        fetcher: SnapshotFetcher,
        *,
        csv_entries_order: Sequence[CsvStaffEntry],
        csv_entries: Mapping[str, CsvStaffEntry],
        expert_topics: Mapping[str, Sequence[str]] | None = None,
        role_keywords: Sequence[str] | None = None,
    ) -> None:
        self.fetcher = fetcher
        self._role_keywords = tuple(
            keyword.casefold() for keyword in (role_keywords or ())
        )
        self._csv_entries_order = list(csv_entries_order)
        self._csv_entries = dict(csv_entries)
        self._allowed_slugs = {entry.slug for entry in self._csv_entries_order}
        self._expert_topics = {
            slug: list(topics) for slug, topics in (expert_topics or {}).items()
        }

    def collect_staff_profiles(self) -> list[StaffProfile]:
        listings = self._load_listings()
        listing_by_slug = {listing.slug: listing for listing in listings}
        profiles: list[StaffProfile] = []
        for entry in self._csv_entries_order:
            listing = listing_by_slug.get(entry.slug)
            if listing is None:
                listing = self._listing_from_csv_entry(entry)
            profile = self._collect_profile(listing)
            if profile is None:
                LOGGER.warning(
                    "Failed to collect profile for %s (%s)",
                    entry.name,
                    entry.profile_url,
                )
                continue
            self._apply_csv_overrides(profile, entry)
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
                if not should_include_role(title, self._role_keywords):
                    continue
                name = str(
                    entry.get("field_ansatte_navn") or entry.get("title") or ""
                ).strip()
                if not name:
                    continue
                department = normalize_department(
                    str(entry.get("field_avdeling") or "").strip()
                )
                profile_path = str(entry.get("view_node") or "").strip()
                if not profile_path:
                    continue
                profile_url = normalize_profile_url(profile_path)
                slug = slug_from_profile_url(profile_url)
                if self._allowed_slugs and slug not in self._allowed_slugs:
                    continue
                image_url = str(entry.get("field_top_bilde") or "").strip() or None
                tags = parse_tags(entry.get("field_tags"))
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.warning(
                    "Skipping entry due to parsing error: %s (%s)", entry, exc
                )
                continue
            listing = StaffListing(
                name=name,
                title=title,
                department=department or "Ukjent",
                profile_url=profile_url,
                slug=slug,
                image_url=image_url,
                tags=tags,
            )
            csv_entry = self._csv_entries.get(slug)
            if csv_entry:
                listing.name = csv_entry.name
                listing.department = csv_entry.department
                listing.profile_url = csv_entry.profile_url
            listings.append(listing)
        LOGGER.info("Collected %d staff listings matching role filter.", len(listings))
        return listings

    def _listing_from_csv_entry(self, entry: CsvStaffEntry) -> StaffListing:
        return StaffListing(
            name=entry.name,
            title="",
            department=entry.department,
            profile_url=entry.profile_url,
            slug=entry.slug,
        )

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
        title = extract_profile_title(html)
        if title:
            listing.title = title
        email, phone = extract_contact_details(html)
        sources = [listing.profile_url]
        return StaffProfile(
            listing=listing,
            summary_text=summary,
            html_path=snapshot_path,
            sources=sources,
            email=email,
            phone=phone,
        )

    def _apply_csv_overrides(self, profile: StaffProfile, entry: CsvStaffEntry) -> None:
        listing = profile.listing
        listing.name = entry.name
        listing.department = entry.department
        listing.profile_url = entry.profile_url
        expert_tags = self._expert_topics.get(entry.slug)
        if expert_tags:
            listing.tags = list(expert_tags)
        else:
            listing.tags = []
        if entry.nva_profile_url:
            profile.nva_profile_url = entry.nva_profile_url
            profile.add_source(entry.nva_profile_url)


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


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for item in items:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            results.append(item)
    return results


def prepare_staff_metadata(profiles: Sequence[StaffProfile]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def sort_key(profile: StaffProfile) -> tuple[str, str]:
        listing = profile.listing
        return (listing.department.casefold(), listing.name.casefold())

    for profile in sorted(profiles, key=sort_key):
        listing = profile.listing
        entry_items: list[tuple[str, Any]] = [("name", listing.name)]
        if listing.title:
            entry_items.append(("title", listing.title))
        entry_items.append(("department", listing.department))
        entry_items.append(("profile_url", listing.profile_url))
        if profile.email:
            entry_items.append(("email", profile.email))
        if profile.phone:
            entry_items.append(("phone", profile.phone))
        if listing.image_url:
            entry_items.append(("image_url", listing.image_url))
        entry_items.append(("sources", dedupe_preserve_order(profile.sources)))
        entry_items.append(("tags", list(listing.tags)))
        entry = {key: value for key, value in entry_items}
        entries.append(entry)
    return entries


def write_staff_yaml(
    entries: Sequence[dict[str, Any]],
    path: Path,
    *,
    dry_run: bool,
) -> None:
    serialized = yaml.safe_dump(
        list(entries),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    if dry_run:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        diff = list(
            difflib.unified_diff(
                existing.splitlines(),
                serialized.splitlines(),
                str(path),
                f"{path} (new)",
                lineterm="",
            )
        )
        if diff:
            for line in diff:
                print(line)
            LOGGER.info("Dry run complete. Diff printed above.")
        else:
            LOGGER.info("Dry run complete. No changes detected for %s.", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    LOGGER.info("Updated %s with %d staff entries.", path, len(entries))


def profiles_to_records(profiles: Sequence[StaffProfile]) -> list[StaffRecord]:
    records: list[StaffRecord] = []
    for profile in profiles:
        listing = profile.listing
        sources = [
            SourceLink(url=url) for url in dedupe_preserve_order(profile.sources)
        ]
        record = StaffRecord(
            slug=listing.slug,
            name=listing.name,
            title=listing.title,
            department=listing.department,
            profile_url=listing.profile_url,
            summary=profile.summary_text,
            sources=sources,
            tags=list(listing.tags),
        )
        records.append(record)
    return records


def write_records_jsonl(records: Sequence[StaffRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "slug": record.slug,
                "name": record.name,
                "title": record.title,
                "department": record.department,
                "profile_url": record.profile_url,
                "summary": record.summary,
                "sources": [link.url for link in record.sources],
                "tags": record.tags,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote %d staff records to %s.", len(records), path)


def build_index_from_records(
    records: Sequence[StaffRecord],
    *,
    app_config: AppConfig,
    models_config: ModelsConfig,
    index_root: Path,
) -> None:
    if not records:
        LOGGER.warning("No staff records with content available; skipping index build.")
        return

    chunker = Chunker(
        chunk_size=app_config.rag.chunk_size,
        chunk_overlap=app_config.rag.chunk_overlap,
        max_chunks=app_config.rag.max_chunks_per_profile,
    )
    embedder = create_embedding_backend(models_config, app_config=app_config)
    paths = IndexPaths(root=index_root)
    vector_store = LocalVectorStore(paths.vectors_dir)
    staff_info = load_staff_info(PROJECT_ROOT / "staff_info.json")
    builder = StaffIndexBuilder(
        paths=paths,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        staff_info=staff_info,
    )
    summary = builder.build(records)
    LOGGER.info(
        "Index build complete: %d staff processed, %d chunks generated, %d skipped.",
        summary.processed_staff,
        summary.total_chunks,
        len(summary.skipped_staff),
    )


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

    role_keywords: tuple[str, ...]
    if args.role_keywords:
        role_keywords = tuple(
            keyword.casefold() for keyword in args.role_keywords if keyword
        )
    elif args.academic_only:
        role_keywords = tuple(keyword.casefold() for keyword in DEFAULT_ROLE_KEYWORDS)
    else:
        role_keywords = tuple()

    csv_entries = load_staff_allowlist(args.staff_csv)
    csv_entries_by_slug = {entry.slug: entry for entry in csv_entries}

    fetcher = SnapshotFetcher(snapshot_dir=args.snapshot_dir, offline=args.offline)
    if args.nva_api_key:
        LOGGER.info("NVA API key supplied; API lookup enabled.")
    else:
        LOGGER.info(
            "NVA API key not provided; relying on cached API snapshots or HTML search."
        )

    with fetcher:
        expert_topics = load_expert_topics(fetcher, csv_entries)
        collector = StaffCollector(
            fetcher,
            csv_entries_order=csv_entries,
            csv_entries=csv_entries_by_slug,
            expert_topics=expert_topics,
            role_keywords=role_keywords,
        )
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

    metadata_entries = prepare_staff_metadata(profiles)
    LOGGER.info("Prepared metadata for %d staff entries.", len(metadata_entries))
    write_staff_yaml(metadata_entries, args.output, dry_run=args.dry_run)

    if args.dry_run:
        LOGGER.info("Dry run requested; skipping record export and index build.")
        return 0

    records = profiles_to_records(profiles)
    if args.records_output:
        write_records_jsonl(records, args.records_output)

    if args.skip_index:
        LOGGER.info("Index build skipped (--skip-index).")
        return 0

    app_config = load_app_config()
    models_config = load_models_config()
    index_root = args.index_root or Path(app_config.rag.index_root)
    LOGGER.info("Building index under %s", index_root)
    build_index_from_records(
        records,
        app_config=app_config,
        models_config=models_config,
        index_root=index_root,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
