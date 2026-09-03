from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent  # world_cup/scripts/
WC_DIR = THIS_DIR.parent  # world_cup/
PROJECT_ROOT = WC_DIR.parent  # FYP/
sys.path.insert(0, str(WC_DIR))

from src.platform_prediction import build_platform_predictions
from src.poisson_model import WeightedPoissonModel
from src.world_cup_model import load_model_bundle, round_probability_columns


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Generate unified platform probabilities, odds and scores."
    )
    parser.add_argument("fixtures", type=Path)
    parser.add_argument(
        "--rankings",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "world_cup" / "ratings_2026_live.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "world_cup" / "platform_predictions_2026.csv",
    )
    parser.add_argument(
        "--xlsx-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "world_cup" / "platform_predictions_2026.xlsx",
    )
    args = parser.parse_args()

    main_bundle = load_model_bundle(PROJECT_ROOT / "outputs" / "world_cup" / "world_cup_model.json")
    poisson_model = WeightedPoissonModel.load(
        PROJECT_ROOT / "outputs" / "world_cup" / "poisson_model.json"
    )
    weight_payload = json.loads(
        (
            PROJECT_ROOT
            / "outputs"
            / "world_cup"
            / "historical_backtest"
            / "global_descriptive_weights.json"
        ).read_text(encoding="utf-8")
    )
    weights = (
        float(weight_payload["weight_result"]),
        float(weight_payload["weight_market_proxy"]),
        float(weight_payload["weight_poisson"]),
    )
    fixtures = pd.read_csv(args.fixtures)
    predictions = build_platform_predictions(
        fixtures,
        args.rankings,
        main_bundle,
        poisson_model,
        weights,
    )

    probability_groups = [
        ["ResultP_H", "ResultP_D", "ResultP_A"],
        ["MarketProxyP_H", "MarketProxyP_D", "MarketProxyP_A"],
        ["PoissonP_H", "PoissonP_D", "PoissonP_A"],
        ["OfficialP_H", "OfficialP_D", "OfficialP_A"],
        ["ExperimentalP_H", "ExperimentalP_D", "ExperimentalP_A"],
    ]
    for columns in probability_groups:
        predictions = round_probability_columns(predictions, columns)
    predictions["OfficialFairOdds_H"] = 1.0 / predictions["OfficialP_H"]
    predictions["OfficialFairOdds_D"] = 1.0 / predictions["OfficialP_D"]
    predictions["OfficialFairOdds_A"] = 1.0 / predictions["OfficialP_A"]
    predictions[
        [
            "Poisson_xG_H",
            "Poisson_xG_A",
            "OfficialFairOdds_H",
            "OfficialFairOdds_D",
            "OfficialFairOdds_A",
            "ModelDisagreement",
        ]
    ] = predictions[
        [
            "Poisson_xG_H",
            "Poisson_xG_A",
            "OfficialFairOdds_H",
            "OfficialFairOdds_D",
            "OfficialFairOdds_A",
            "ModelDisagreement",
        ]
    ].round(3)
    for column in ("MostLikelyScoreP", "Score1P", "Score2P", "Score3P"):
        predictions[column] = predictions[column].round(4)

    preferred_columns = [
        "MatchNumber",
        "DateUTC",
        "DateChina",
        "Group",
        "GroupRound",
        "Home",
        "Away",
        "Stage",
        "HostAdvantage",
        "Stadium",
        "City",
        "OfficialProbabilitySource",
        "OfficialOutcome",
        "OfficialP_H",
        "OfficialP_D",
        "OfficialP_A",
        "OfficialFairOdds_H",
        "OfficialFairOdds_D",
        "OfficialFairOdds_A",
        "Poisson_xG_H",
        "Poisson_xG_A",
        "MostLikelyScore",
        "MostLikelyScoreP",
        "Score1",
        "Score1P",
        "Score2",
        "Score2P",
        "Score3",
        "Score3P",
        "ModelConsistency",
        "ModelDisagreement",
        "DirectionAgreement",
        "PoissonOutcome",
        "ResultP_H",
        "ResultP_D",
        "ResultP_A",
        "MarketProxyP_H",
        "MarketProxyP_D",
        "MarketProxyP_A",
        "PoissonP_H",
        "PoissonP_D",
        "PoissonP_A",
        "ExperimentalP_H",
        "ExperimentalP_D",
        "ExperimentalP_A",
        "ExperimentalWeightResult",
        "ExperimentalWeightMarket",
        "ExperimentalWeightPoisson",
    ]
    predictions = predictions[preferred_columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False, encoding="utf-8-sig")

    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js is required to create the XLSX output.")
    subprocess.run(
        [
            node,
            str(THIS_DIR / "export_platform_predictions_xlsx.mjs"),
            str(args.output),
            str(args.xlsx_output),
        ],
        cwd=THIS_DIR,
        check=True,
    )
    display = predictions[
        [
            "MatchNumber",
            "Home",
            "Away",
            "OfficialP_H",
            "OfficialP_D",
            "OfficialP_A",
            "OfficialFairOdds_H",
            "OfficialFairOdds_D",
            "OfficialFairOdds_A",
            "MostLikelyScore",
            "MostLikelyScoreP",
            "ModelConsistency",
        ]
    ]
    print(display.to_string(index=False))
    print(f"\nSaved platform CSV: {args.output}")
    print(f"Saved platform workbook: {args.xlsx_output}")


if __name__ == "__main__":
    main()
