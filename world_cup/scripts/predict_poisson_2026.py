from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent  # world_cup/scripts/
WC_DIR = THIS_DIR.parent  # world_cup/
PROJECT_ROOT = WC_DIR.parent  # FYP/
sys.path.insert(0, str(WC_DIR))

from src.poisson_model import WeightedPoissonModel
from src.world_cup_model import format_probability_columns


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Predict World Cup scores and result probabilities with Poisson."
    )
    parser.add_argument(
        "fixtures",
        type=Path,
        help="CSV containing Home, Away and optional HostAdvantage.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "world_cup" / "poisson_model.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "world_cup" / "poisson_predictions_2026.csv",
    )
    parser.add_argument(
        "--xlsx-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "world_cup" / "poisson_predictions_2026.xlsx",
    )
    parser.add_argument("--max-goals", type=int, default=10)
    args = parser.parse_args()

    model = WeightedPoissonModel.load(args.model)
    fixtures = pd.read_csv(args.fixtures)
    predictions = model.predict_fixtures(
        fixtures,
        max_goals=args.max_goals,
        top_n=3,
    )
    predictions = format_probability_columns(
        predictions,
        ["PoissonP_H", "PoissonP_D", "PoissonP_A"],
    )
    predictions[["Poisson_xG_H", "Poisson_xG_A"]] = predictions[
        ["Poisson_xG_H", "Poisson_xG_A"]
    ].round(3)
    for column in (
        "MostLikelyScoreP",
        "Score1P",
        "Score2P",
        "Score3P",
    ):
        predictions[column] = predictions[column].round(4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False, encoding="utf-8-sig")
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js is required to create the formatted XLSX output.")
    subprocess.run(
        [
            node,
            str(THIS_DIR / "export_poisson_predictions_xlsx.mjs"),
            str(args.output),
            str(args.xlsx_output),
        ],
        cwd=THIS_DIR,
        check=True,
    )
    display_columns = [
        column
        for column in (
            "MatchNumber",
            "Home",
            "Away",
            "Poisson_xG_H",
            "Poisson_xG_A",
            "PoissonP_H",
            "PoissonP_D",
            "PoissonP_A",
            "Score1",
            "Score1P",
            "Score2",
            "Score2P",
            "Score3",
            "Score3P",
        )
        if column in predictions.columns
    ]
    print(predictions[display_columns].to_string(index=False))
    print(f"\nSaved predictions: {args.output}")
    print(f"Saved formatted workbook: {args.xlsx_output}")


if __name__ == "__main__":
    main()
