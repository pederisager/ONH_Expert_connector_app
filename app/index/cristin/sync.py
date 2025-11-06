"""Command-line interface for ingesting Cristin research results."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from app.cache_manager import CacheManager

from .client import CristinClient, CristinClientConfig
from .models import ResultRecord
from .normalizer import normalize_result

LOGGER = logging.getLogger("cristin.sync")


@dataclass(slots=True)
class StaffMember:
    employee_id: str
    name: str
    department: str | None


@dataclass(slots=True)
class SyncStats:
    staff_processed: int = 0
    results_fetched: int = 0
    results_normalized: int = 0
    staff_skipped: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Cristin results for staff members and materialize them for RAG ingestion."
    )
    parser.add_argument(
        "--staff-csv",
        type=Path,
        default=Path("staff.csv"),
        help="Input CSV containing staff roster with NVA profile links.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cristin/results.jsonl"),
        help="Path to write normalized results JSONL.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/cristin/raw"),
        help="Directory where raw API responses are stored per employee.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/cristin"),
        help="Directory for HTTP cache. Disable with --force-refresh.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Two-letter language code to request from the API (default: en).",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=1000,
        help="Number of results fetched per page (max 1000).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass disk cache and force all requests.",
    )
    parser.add_argument(
        "--throttle-delay",
        type=float,
        default=0.0,
        help="Optional sleep (in seconds) before every API request.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def read_staff_csv(path: Path) -> tuple[list[StaffMember], int]:
    staff: list[StaffMember] = []
    skipped = 0
    with path.open(mode="r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            nva_profile = (row.get("NVA_profile") or "").strip()
            name = (row.get("Name") or "").strip()
            department = (row.get("Department") or "").strip() or None
            if not nva_profile:
                LOGGER.warning("Skipping %s: missing NVA profile link", name or "<unknown>")
                skipped += 1
                continue
            employee_id = extract_person_id(nva_profile)
            staff.append(StaffMember(employee_id=employee_id, name=name, department=department))
    return staff, skipped


def extract_person_id(url: str) -> str:
    parsed = urlparse(url.strip())
    path = parsed.path.rstrip("/")
    person_id = path.split("/")[-1]
    if not person_id:
        raise ValueError(f"Unable to parse person id from {url}")
    return person_id


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _information_score(record: ResultRecord) -> int:
    score = 0
    if record.summary:
        score += 1000 + min(len(record.summary), 2000)
    if record.venue:
        score += 200
    score += len(record.tags) * 50
    score += len(record.contributors) * 25
    score += sum(1 for val in record.extra_context.values() if val) * 10
    if record.category_name:
        score += 25
    return score


def _dedup_key(record: ResultRecord) -> str:
    if record.title:
        return record.title.strip().casefold()
    return record.cristin_result_id


def _choose_richer(existing: ResultRecord, candidate: ResultRecord) -> ResultRecord:
    existing_score = _information_score(existing)
    candidate_score = _information_score(candidate)
    if candidate_score > existing_score:
        return candidate
    if candidate_score < existing_score:
        return existing

    existing_year = existing.year_published or ""
    candidate_year = candidate.year_published or ""
    if candidate_year > existing_year:
        return candidate
    if candidate_year < existing_year:
        return existing

    if candidate.summary and not existing.summary:
        return candidate
    if existing.summary and not candidate.summary:
        return existing

    return existing


def sync_cristin_results(
    *,
    staff_members: Iterable[StaffMember],
    output_path: Path,
    raw_dir: Path,
    client_config: CristinClientConfig,
    cache: CacheManager | None,
    session=None,
) -> SyncStats:
    ensure_directory(output_path.parent)
    ensure_directory(raw_dir)
    stats = SyncStats()

    with CristinClient(config=client_config, cache=cache, session=session) as client:
        with output_path.open("w", encoding="utf-8") as sink:
            for member in staff_members:
                stats.staff_processed += 1
                person_raw_dir = raw_dir / member.employee_id
                ensure_directory(person_raw_dir)

                listings = client.get_person_results(member.employee_id)
                stats.results_fetched += len(listings)

                listing_path = person_raw_dir / "results_listing.json"
                listing_path.write_text(
                    json.dumps(listings, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                dedup_map: dict[str, ResultRecord] = {}
                for listing in listings:
                    result_id = listing.get("cristin_result_id")
                    if not result_id:
                        LOGGER.debug(
                            "Skipping listing without result id for %s",
                            member.employee_id,
                        )
                        continue
                    detail = client.get_result(str(result_id))

                    detail_path = person_raw_dir / f"{result_id}.json"
                    detail_path.write_text(
                        json.dumps(detail, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    record = normalize_result(
                        employee_id=member.employee_id,
                        employee_name=member.name,
                        listing_payload=listing,
                        detail_payload=detail,
                        lang=client_config.lang,
                    )

                    key = _dedup_key(record)
                    if key in dedup_map:
                        dedup_map[key] = _choose_richer(dedup_map[key], record)
                    else:
                        dedup_map[key] = record

                for record in dedup_map.values():
                    sink.write(json.dumps(record.to_json_dict(), ensure_ascii=False))
                    sink.write("\n")
                stats.results_normalized += len(dedup_map)
    return stats


def main(argv: list[str] | None = None) -> SyncStats:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    staff_members, skipped = read_staff_csv(args.staff_csv)
    if not staff_members:
        LOGGER.warning("No staff members found in %s", args.staff_csv)
        return SyncStats(staff_skipped=skipped)

    cache = None
    if not args.force_refresh:
        cache = CacheManager(directory=args.cache_dir)

    config = CristinClientConfig(
        lang=args.lang,
        per_page=args.per_page,
        throttle_delay=args.throttle_delay,
    )

    stats = sync_cristin_results(
        staff_members=staff_members,
        output_path=args.output,
        raw_dir=args.raw_dir,
        client_config=config,
        cache=cache,
    )
    stats.staff_skipped += skipped

    LOGGER.info(
        "Processed %s staff, fetched %s results, normalized %s entries",
        stats.staff_processed,
        stats.results_fetched,
        stats.results_normalized,
    )
    return stats


if __name__ == "__main__":
    main()
