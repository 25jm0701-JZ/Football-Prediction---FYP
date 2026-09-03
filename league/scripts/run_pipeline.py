"""
League prediction pipeline: data → features → train → evaluate.

Usage:
    # Run single model
    python league/scripts/run_pipeline.py --model lr

    # Compare all models
    python league/scripts/run_pipeline.py --compare

    # Over/Under 2.5 prediction
    python league/scripts/run_pipeline.py --model lr --target over

    # Use multiple data files
    python league/scripts/run_pipeline.py --data league/data/raw/*.csv --compare
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add league to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_data, load_multiple
from src.train import compare_models, train_and_evaluate
from src.features import get_feature_columns
from src.models import list_available_models

TARGET_MAP = {
    "result": "target_result",
    "over": "target_over_2.5",
    "over25": "target_over_2.5",
}

TARGET_NAMES = {
    "target_result": "Match Result (H/D/A)",
    "target_over_2.5": "Over/Under 2.5 Goals",
}


def main():
    parser = argparse.ArgumentParser(
        description="League Match Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data", nargs="+",
        default=["league/data/raw/PL2526.csv"],
        help="CSV data file(s) from football-data.co.uk",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Model to train. One of: {list_available_models()}",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare all available models",
    )
    parser.add_argument(
        "--target", type=str, default="result",
        choices=["result", "over", "over25"],
        help="Prediction target: result (H/D/A) or over (O/U 2.5)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.2,
        help="Fraction of matches to hold out for testing (default: 0.2)",
    )
    parser.add_argument(
        "--windows", type=int, nargs="+", default=[5, 10],
        help="Rolling window sizes (default: 5 10)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/league",
        help="Directory for output files",
    )
    parser.add_argument(
        "--footystats", action="store_true",
        help="Enable FootyStats extra features (PPG, xG, possession) — PL only",
    )
    parser.add_argument(
        "--season-col", type=str, default="season",
        help="Column name for season labels (used with --footystats)",
    )

    args = parser.parse_args()
    target_col = TARGET_MAP[args.target]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────
    print("=" * 60)
    print(" League Match Prediction Pipeline")
    print("=" * 60)

    if len(args.data) == 1:
        print(f"\nLoading: {args.data[0]}")
        df = load_data(args.data[0])
    else:
        print(f"\nLoading {len(args.data)} files...")
        df = load_multiple(args.data)

    print(f"  Matches: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Target: {TARGET_NAMES[target_col]}")

    # Target distribution
    if target_col in df.columns:
        dist = df[target_col].value_counts().sort_index()
        print(f"  Distribution:")
        for k, v in dist.items():
            if target_col == "target_result":
                names = {0: "Away", 1: "Draw", 2: "Home"}
                label = names.get(k, k)
            else:
                label = "Over" if k == 1 else "Under"
            print(f"    {label}: {v} ({v/len(df)*100:.1f}%)")

    # ── Run ──────────────────────────────────────────────────────
    if args.compare:
        print(f"\n{'='*60}")
        print(" Comparing all models...")
        print(f"{'='*60}")

        results = compare_models(
            df,
            target=target_col,
            test_ratio=args.test_ratio,
            windows=args.windows,
            verbose=True,
            include_footystats=args.footystats,
        )

        # Save comparison
        comp_path = output_dir / "model_comparison.csv"
        results.to_csv(comp_path, index=False)
        print(f"\nSaved: {comp_path}")

    elif args.model:
        print(f"\n{'='*60}")
        print(f" Training model: {args.model}")
        print(f"{'='*60}")

        result = train_and_evaluate(
            df,
            model_name=args.model,
            target=target_col,
            test_ratio=args.test_ratio,
            windows=args.windows,
            verbose=True,
            return_model=False,
            include_footystats=args.footystats,
            footystats_season_col=args.season_col,
        )

        # Generate evaluation plots if probabilities available
        if result.get("y_prob") is not None:
            from src.evaluate import (
                classification_metrics,
                plot_confusion_matrix,
                plot_probability_distribution,
                print_metrics,
                save_predictions,
            )

            metrics = classification_metrics(
                result["y_test"], result["y_pred"], result["y_prob"]
            )
            print_metrics(metrics, f"Detailed: {args.model}")

            # Save predictions
            df_test = df.sort_values("date").iloc[-len(result["y_test"]):]
            pred_path = output_dir / f"predictions_{args.model}.csv"
            save_predictions(
                df_test, result["y_test"],
                result["y_pred"], result["y_prob"],
                save_path=pred_path,
            )

            # Plots
            plot_confusion_matrix(
                result["y_test"], result["y_pred"],
                save_path=output_dir / f"confusion_{args.model}.png",
            )
            plot_probability_distribution(
                result["y_test"], result["y_prob"],
                save_path=output_dir / f"prob_dist_{args.model}.png",
            )

    else:
        print("\nPlease specify --model or --compare")
        print("  python league/scripts/run_pipeline.py --model lr")
        print("  python league/scripts/run_pipeline.py --compare")
        parser.print_help()


if __name__ == "__main__":
    main()
