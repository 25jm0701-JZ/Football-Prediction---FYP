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

from src.blend_backtest import (
    align_backtest_predictions,
    comparison_metrics,
    nested_blend_backtest,
    optimize_weights,
)
from src.historical_poisson import (
    WORLD_CUP_CYCLES,
    backtest_cycle,
    prepare_historical_cycle,
)
from src.world_cup_model import cross_validate, load_historical_matches


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "world_cup" / "historical_backtest"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Backtest historical Poisson and nested probability blends."
    )
    parser.add_argument("--half-life-days", type=float, default=540.0)
    parser.add_argument("--l2", type=float, default=0.08)
    parser.add_argument("--weight-step", type=float, default=0.05)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    poisson_predictions = []
    poisson_metrics = []
    audits = {}

    for year in sorted(WORLD_CUP_CYCLES):
        training, test_matches, audit = prepare_historical_cycle(
            external_path=WC_DIR / "data" / "raw" / "international_results.csv",
            games_path=WC_DIR / "data" / "raw" / "WorldCup2026.xlsx",
            year=year,
            half_life_days=args.half_life_days,
        )
        model, predictions, metrics = backtest_cycle(
            training,
            test_matches,
            l2=args.l2,
        )
        training.to_csv(
            OUTPUT_DIR / f"poisson_training_{year}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        predictions.to_csv(
            OUTPUT_DIR / f"poisson_predictions_{year}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        model.save(OUTPUT_DIR / f"poisson_model_{year}.json")
        poisson_predictions.append(predictions)
        poisson_metrics.append({"test_year": year, "model": "poisson_model", **metrics})
        audits[str(year)] = audit
        print(
            f"{year}: {len(training)} training matches, "
            f"log loss={metrics['log_loss']:.4f}, "
            f"Brier={metrics['brier_score']:.4f}"
        )

    poisson_frame = pd.concat(poisson_predictions, ignore_index=True)
    poisson_frame.to_csv(
        OUTPUT_DIR / "poisson_predictions_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(poisson_metrics).to_csv(
        OUTPUT_DIR / "poisson_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "poisson_data_audits.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    historical_matches, _ = load_historical_matches(
        WC_DIR / "data" / "raw" / "WorldCup2026.xlsx",
        WC_DIR / "world_cup_rankings_4_sheets_simple.xlsx",
    )
    original_predictions, _ = cross_validate(historical_matches)
    aligned = align_backtest_predictions(original_predictions, poisson_frame)
    blend_predictions, weights = nested_blend_backtest(
        aligned,
        step=args.weight_step,
    )
    metrics = comparison_metrics(aligned, blend_predictions)
    global_weights, global_selection_metrics = optimize_weights(
        aligned,
        step=args.weight_step,
    )

    aligned.to_csv(
        OUTPUT_DIR / "aligned_model_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    blend_predictions.to_csv(
        OUTPUT_DIR / "nested_blend_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    weights.to_csv(
        OUTPUT_DIR / "nested_blend_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        OUTPUT_DIR / "model_comparison_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall = metrics[metrics["test_year"].astype(str) == "ALL"].copy()
    best_row = overall.loc[overall["log_loss"].idxmin()]
    recommendation = {
        "selection_rule": "lowest out-of-sample ALL log loss",
        "recommended_probability_model": str(best_row["model"]),
        "recommended_log_loss": float(best_row["log_loss"]),
        "recommended_brier_score": float(best_row["brier_score"]),
        "recommended_accuracy": float(best_row["accuracy"]),
        "poisson_role": (
            "Use for expected goals and exact-score output. Do not include it "
            "in the production probability blend unless a later validation "
            "experiment improves out-of-sample log loss."
        ),
        "nested_blend_improved_over_recommendation": bool(
            overall.loc[
                overall["model"] == "nested_optimized_blend",
                "log_loss",
            ].iloc[0]
            < best_row["log_loss"]
        ),
    }
    (OUTPUT_DIR / "model_selection_recommendation.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "global_descriptive_weights.json").write_text(
        json.dumps(
            {
                "warning": (
                    "Descriptive only: selected on all 192 test matches and "
                    "not an unbiased performance estimate."
                ),
                "weight_result": global_weights[0],
                "weight_market_proxy": global_weights[1],
                "weight_poisson": global_weights[2],
                "selection_metrics": global_selection_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nNested blend weights")
    print(weights.to_string(index=False))
    print("\nModel comparison")
    print(metrics.to_string(index=False))
    print(
        "\nDescriptive all-data weights: "
        f"result={global_weights[0]:.2f}, "
        f"market={global_weights[1]:.2f}, "
        f"poisson={global_weights[2]:.2f}"
    )
    print(f"\nSaved historical backtest outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
