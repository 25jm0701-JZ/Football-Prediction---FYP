import argparse
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent  # world_cup/scripts/
WC_DIR = THIS_DIR.parent  # world_cup/
PROJECT_ROOT = WC_DIR.parent  # FYP/
sys.path.insert(0, str(WC_DIR))

from src.world_cup_model import (
    format_probability_columns,
    load_model_bundle,
    predict_fixtures,
    set_intl_results_path,
)
from src.player_features import load_team_features


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Predict 2026 World Cup fixture probabilities.")
    parser.add_argument("fixtures", type=Path, help="CSV with Home,Away,Stage,HostAdvantage")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "world_cup" / "predictions_2026.csv",
    )
    parser.add_argument(
        "--rankings",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "world_cup" / "ratings_2026_live.csv",
        help="Current 2026 ratings CSV or the four-sheet rankings workbook.",
    )
    parser.add_argument(
        "--player-data",
        type=Path,
        default=Path("C:/Users/-jmmmm/Downloads/international-world-cup-players-2026-to-2026-stats.csv"),
        help="2026 World Cup player stats CSV for squad-quality features.",
    )
    args = parser.parse_args()

    # Enable rolling-form features from international results
    intl_path = WC_DIR / "data" / "raw" / "international_results.csv"
    if intl_path.exists():
        set_intl_results_path(intl_path)

    bundle = load_model_bundle(PROJECT_ROOT / "outputs" / "world_cup" / "world_cup_model.json")
    fixtures = pd.read_csv(args.fixtures)

    # Load 2026 player data and aggregate to team features
    team_features = None
    if args.player_data and args.player_data.exists():
        team_features = load_team_features({2026: args.player_data})

    predictions = predict_fixtures(
        fixtures,
        args.rankings,
        bundle,
        team_features=team_features,
    )
    predictions = format_probability_columns(predictions, ["P_H", "P_D", "P_A"])
    predictions[["FairOdds_H", "FairOdds_D", "FairOdds_A"]] = predictions[
        ["FairOdds_H", "FairOdds_D", "FairOdds_A"]
    ].round(3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(predictions.to_string(index=False))
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
