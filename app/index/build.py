"""CLI for building the local vector index from curated staff data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.config_loader import load_app_config, load_models_config

from .records_loader import load_curated_records
from .refresh_staff import build_index_from_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the staff retrieval index from curated data/staff.yaml + staff_records.jsonl.",
    )
    parser.add_argument(
        "--staff-yaml",
        type=Path,
        default=Path("data/staff.yaml"),
        help="Path to the curated staff.yaml file (default: data/staff.yaml).",
    )
    parser.add_argument(
        "--records-jsonl",
        type=Path,
        default=Path("data/staff_records.jsonl"),
        help="Path to the JSONL file with staff summaries (default: data/staff_records.jsonl).",
    )
    parser.add_argument(
        "--nva-results",
        type=Path,
        default=Path("data/nva/results.jsonl"),
        help="Path to the NVA results JSONL file (default: data/nva/results.jsonl).",
    )
    parser.add_argument(
        "--app-config",
        type=Path,
        default=Path("data/app.config.yaml"),
        help="Path to app.config.yaml (default: data/app.config.yaml).",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=Path("data/models.yaml"),
        help="Path to models.yaml (default: data/models.yaml).",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=None,
        help="Override the target directory for the vector index (default: rag.index-root).",
    )
    parser.add_argument(
        "--max-nva-results",
        type=int,
        default=5,
        help="Maximum NVA results per staff member to embed (default: 5).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    app_config = load_app_config(args.app_config)
    models_config = load_models_config(args.models_config)
    index_root = args.index_root or Path(app_config.rag.index_root)

    logging.info("Loading curated staff records from %s and %s", args.staff_yaml, args.records_jsonl)
    records = load_curated_records(
        staff_yaml_path=args.staff_yaml,
        records_jsonl_path=args.records_jsonl,
        nva_results_path=args.nva_results,
        max_nva_results=max(1, args.max_nva_results),
    )
    logging.info("Loaded %d staff records. Building index in %s", len(records), index_root)

    build_index_from_records(
        records,
        app_config=app_config,
        models_config=models_config,
        index_root=index_root,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
