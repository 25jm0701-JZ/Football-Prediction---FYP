from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent  # world_cup/scripts/
WC_DIR = THIS_DIR.parent  # world_cup/
PROJECT_ROOT = WC_DIR.parent  # FYP/
sys.path.insert(0, str(WC_DIR))

from src.international_training_data import (
    DEFAULT_CUTOFF_DATE,
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_START_DATE,
    INTERNATIONAL_RESULTS_URL,
    PreparationConfig,
    download_international_results,
    prepare_training_data,
    write_preparation_outputs,
)


DEFAULT_EXTERNAL_PATH = WC_DIR / "data" / "raw" / "international_results.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "world_cup" / "poisson_training_matches.csv"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "outputs" / "world_cup" / "poisson_training_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare weighted senior international matches for a Poisson model."
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=DEFAULT_EXTERNAL_PATH,
        help="Cached martj42 international results CSV.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download or refresh the external CSV before preparing the dataset.",
    )
    parser.add_argument(
        "--start",
        type=pd.Timestamp,
        default=DEFAULT_START_DATE,
        help="First eligible match date (default: 2022-12-19).",
    )
    parser.add_argument(
        "--cutoff",
        type=pd.Timestamp,
        default=DEFAULT_CUTOFF_DATE,
        help="Last eligible match date (default: 2026-06-10).",
    )
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=DEFAULT_HALF_LIFE_DAYS,
        help="Time-decay half-life in days (default: 540).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_PATH)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.download or not args.external.exists():
        print(f"Downloading: {INTERNATIONAL_RESULTS_URL}")
        download_international_results(args.external)

    config = PreparationConfig(
        start_date=pd.Timestamp(args.start).normalize(),
        cutoff_date=pd.Timestamp(args.cutoff).normalize(),
        half_life_days=args.half_life_days,
    )
    frame, audit = prepare_training_data(
        qualifiers_path=WC_DIR / "data" / "raw" / "WorldCup2026.xlsx",
        rankings_path=WC_DIR / "world_cup_rankings_4_sheets_simple.xlsx",
        external_path=args.external,
        config=config,
    )
    write_preparation_outputs(frame, audit, args.output, args.audit_output)

    print("\nPreparation audit")
    for key, value in audit.items():
        print(f"{key}: {value}")
    print("\nCompetition summary")
    print(
        frame.groupby("CompetitionCategory")
        .agg(Matches=("Date", "size"), MeanWeight=("FinalWeight", "mean"))
        .sort_values("Matches", ascending=False)
        .round(3)
        .to_string()
    )
    print(f"\nSaved training data: {args.output}")
    print(f"Saved audit report: {args.audit_output}")


if __name__ == "__main__":
    main()
