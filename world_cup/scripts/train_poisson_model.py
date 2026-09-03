from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent  # world_cup/scripts/
WC_DIR = THIS_DIR.parent  # world_cup/
PROJECT_ROOT = WC_DIR.parent  # FYP/
sys.path.insert(0, str(WC_DIR))

from src.poisson_model import WeightedPoissonModel


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Train the weighted international double-Poisson score model."
    )
    parser.add_argument(
        "--training-data",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "world_cup" / "poisson_training_matches.csv",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "world_cup" / "poisson_model.json",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "world_cup" / "poisson_training_metrics.json",
    )
    parser.add_argument("--l2", type=float, default=0.08)
    args = parser.parse_args()

    matches = pd.read_csv(args.training_data)
    model = WeightedPoissonModel(l2=args.l2).fit(matches)
    metrics = model.training_metrics(matches)
    model.save(args.model_output)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    parameters = pd.DataFrame(
        {
            "Team": model.teams,
            "Attack": model.attack,
            "DefenceWeakness": model.defence_weakness,
        }
    )
    print("Poisson model trained.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Baseline expected goals: {math_exp(model.intercept):.3f}")
    print(f"Home multiplier: {math_exp(model.home_advantage):.3f}")
    print("\nStrongest attacks")
    print(parameters.nlargest(10, "Attack").to_string(index=False))
    print("\nStrongest defences")
    print(parameters.nsmallest(10, "DefenceWeakness").to_string(index=False))
    print(f"\nSaved model: {args.model_output}")


def math_exp(value: float) -> float:
    import math
    return math.exp(value)


if __name__ == "__main__":
    main()
