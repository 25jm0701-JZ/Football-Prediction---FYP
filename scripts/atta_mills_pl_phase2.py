#!/usr/bin/env python3
"""
Phase 2: Atta Mills-style features plus original clean pre-match extras.

Adds the project's original non-market, pre-match enhancements to Phase 1:
  - Rolling shots / shots-on-target / shot accuracy / conversion rates.
  - Head-to-head history.

Still excludes:
  - Bet365 odds from training.
  - Half-time variables.
  - O/U 2.5 target/features.
  - FootyStats PPG/xG/possession.
  - Current-match raw statistics that leak the result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "league"))

from scripts.atta_mills_pl_phase1 import (  # noqa: E402
    OUTPUT_DIR as PHASE1_OUTPUT_DIR,
    build_atta_mills_features,
    build_models,
    feature_mapping_table,
    load_pl_data,
    run_walk_forward,
)
from src.features import build_features, get_feature_columns  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "atta_mills_pl_walkforward" / "phase2_with_original_features"
WINDOWS = [5, 10]


def original_extra_feature_mapping(cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if "avg_shots" in col:
            family = "Original Extra: Rolling Shots"
            desc = "Rolling average total shots"
        elif "avg_sot" in col:
            family = "Original Extra: Rolling Shots on Target"
            desc = "Rolling average shots on target"
        elif "shot_accuracy" in col:
            family = "Original Extra: Shot Accuracy"
            desc = "Rolling shots-on-target divided by shots"
        elif "conversion_rate" in col:
            family = "Original Extra: Conversion Rate"
            desc = "Rolling goals divided by shots"
        elif col.startswith("h2h_"):
            family = "Original Extra: Head-to-Head"
            desc = "Previous meetings between the two teams"
        elif col.startswith(("home_avg_", "away_avg_", "home_win_rate", "away_win_rate", "home_draw_rate", "away_draw_rate", "home_loss_rate", "away_loss_rate")):
            family = "Original Overlap: Rolling Form"
            desc = "Existing pipeline rolling form feature overlapping Phase 1"
        else:
            family = "Original Extra: Other Clean Feature"
            desc = "Clean pre-match feature from the existing pipeline"
        rows.append({
            "feature": col,
            "feature_family": family,
            "interpretation": desc,
            "used_in_phase2": True,
        })
    return pd.DataFrame(rows)


def build_phase2_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    atta_df, atta_cols = build_atta_mills_features(df, windows=WINDOWS)

    original_df = build_features(
        df,
        windows=WINDOWS,
        include_shots=True,
        include_odds=False,
        include_h2h=True,
        include_footystats=False,
    )
    original_cols = get_feature_columns(original_df)

    # Keep only the original extras that are not already represented by Phase 1.
    keep_original = [
        col for col in original_cols
        if (
            col.startswith("h2h_")
            or "avg_shots" in col
            or "avg_sot" in col
            or "shot_accuracy" in col
            or "conversion_rate" in col
        )
    ]

    phase2 = atta_df.copy()
    for col in keep_original:
        phase2[col] = original_df[col].values

    feature_cols = atta_cols + keep_original

    atta_map = feature_mapping_table(atta_cols).rename(columns={"paper_family": "feature_family"})
    atta_map["phase"] = "Atta Mills-only"
    extra_map = original_extra_feature_mapping(keep_original)
    extra_map["phase"] = "Original clean extras"
    extra_map["home_or_away"] = extra_map["feature"].map(
        lambda c: "home" if c.startswith("home_") else "away" if c.startswith("away_") else "pair"
    )
    extra_map["window"] = extra_map["feature"].map(
        lambda c: 5 if c.endswith("_5") else 10 if c.endswith("_10") else ""
    )
    extra_map["used_in_phase1"] = False
    extra_map = extra_map[[
        "feature", "feature_family", "home_or_away", "window",
        "interpretation", "used_in_phase1", "used_in_phase2", "phase",
    ]]
    atta_map["used_in_phase2"] = True
    atta_map = atta_map[[
        "feature", "feature_family", "home_or_away", "window",
        "interpretation", "used_in_phase1", "used_in_phase2", "phase",
    ]]
    mapping = pd.concat([atta_map, extra_map], ignore_index=True)
    return phase2, feature_cols, mapping


def write_phase2_doc(df: pd.DataFrame, feature_cols: list[str], aggregate: pd.DataFrame) -> None:
    doc = PROJECT_ROOT / "docs" / "atta_mills_phase2_results.md"
    models = aggregate[aggregate["source"] == "model"].copy()
    market = aggregate[aggregate["source"] == "market"].iloc[0]
    best = models.sort_values(["accuracy", "f1_macro", "draw_f1"], ascending=False).iloc[0]

    phase1_path = PHASE1_OUTPUT_DIR / "model_comparison_walkforward.csv"
    phase1_note = "Phase 1 comparison file was not found."
    if phase1_path.exists():
        phase1 = pd.read_csv(phase1_path)
        phase1_best = phase1.sort_values(["accuracy", "f1_macro", "draw_f1"], ascending=False).iloc[0]
        phase1_note = (
            f"Phase 1 best model was `{phase1_best['model']}` at "
            f"{phase1_best['accuracy']:.2%} accuracy and {phase1_best['draw_f1']:.4f} Draw F1."
        )

    table = models[["model", "accuracy", "f1_macro", "f1_weighted", "draw_f1", "log_loss", "brier"]].copy()
    for col in ["accuracy", "f1_macro", "f1_weighted", "draw_f1", "log_loss", "brier"]:
        table[col] = table[col].map(lambda x: f"{x:.4f}")
    rows = "\n".join([
        f"| {r.model} | {r.accuracy} | {r.f1_macro} | {r.f1_weighted} | {r.draw_f1} | {r.log_loss} | {r.brier} |"
        for r in table.itertuples(index=False)
    ])

    content = f"""# Atta Mills Phase 2 Results

## Scope

- Data: Premier League football-data.co.uk files, {df['season_id'].nunique()} seasons, {len(df)} matches.
- Target: full-time H/D/A result only.
- Validation: season-by-season walk-forward.
- Training exclusions: Bet365 odds, half-time variables, O/U 2.5, FootyStats.
- Added after Phase 1: original clean pre-match H2H and rolling shot-efficiency features.
- Market benchmark: Bet365 closing odds, converted to de-vigged implied probabilities.

## Feature Set

Phase 2 uses {len(feature_cols)} features:

- 44 Atta Mills-style features from Phase 1.
- 21 original clean extras: 16 rolling shot-efficiency features and 5 H2H features.

Detailed feature mapping is saved in `outputs/atta_mills_pl_walkforward/phase2_with_original_features/feature_set.csv`.

## Aggregate Results

| Model | Accuracy | Macro F1 | Weighted F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
{rows}

## Comparison

{phase1_note}

Phase 2 best model is `{best['model']}` at {best['accuracy']:.2%} accuracy and {best['draw_f1']:.4f} Draw F1.
Bet365 closing market accuracy is {market['accuracy']:.2%}, with {market['draw_f1']:.4f} Draw F1.

## Interpretation

Phase 2 tests whether the project's original clean pre-match features add useful signal beyond the strict Atta Mills-style reproduction. Model choice remains data-driven: the final candidate should be selected from walk-forward accuracy, macro F1, Draw F1, log loss, and Brier score rather than from the reference paper alone.
"""
    doc.write_text(content, encoding="utf-8")


def main() -> None:
    # Patch the imported Phase 1 module's output directory so shared helpers write
    # confusion matrices to the Phase 2 folder.
    import scripts.atta_mills_pl_phase1 as phase1

    phase1.OUTPUT_DIR = OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Atta Mills Phase 2: Phase 1 + original clean PL features")
    print("=" * 78)

    df = load_pl_data()
    df_feat, feature_cols, mapping = build_phase2_features(df)
    print(f"[Data] Matches: {len(df)}")
    print(f"[Data] Seasons: {sorted(df['season_id'].unique())}")
    print(f"[Features] Total: {len(feature_cols)}")

    mapping.to_csv(OUTPUT_DIR / "feature_set.csv", index=False)
    df_feat[["date", "season_id", "home_team", "away_team", "target_result", *feature_cols]].to_csv(
        OUTPUT_DIR / "feature_matrix.csv",
        index=False,
    )

    fold_metrics, aggregate_metrics, predictions = run_walk_forward(df_feat, feature_cols, build_models)
    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "predictions_by_fold.csv", index=False)

    aggregate_metrics = aggregate_metrics.sort_values(
        ["source", "accuracy", "f1_macro", "draw_f1"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    model_summary = aggregate_metrics[aggregate_metrics["source"] == "model"].copy()
    market_summary = aggregate_metrics[aggregate_metrics["source"] == "market"].copy()
    model_summary.to_csv(OUTPUT_DIR / "model_comparison_walkforward.csv", index=False)
    market_summary.to_csv(OUTPUT_DIR / "market_comparison_bet365_closing.csv", index=False)
    aggregate_metrics.to_csv(OUTPUT_DIR / "aggregate_metrics_all.csv", index=False)
    write_phase2_doc(df, feature_cols, aggregate_metrics)

    print("\n=== Aggregate model comparison ===")
    print(model_summary[["model", "accuracy", "f1_macro", "draw_f1", "log_loss", "brier"]].to_string(index=False))
    print("\n=== Market benchmark ===")
    print(market_summary[["model", "accuracy", "f1_macro", "draw_f1", "log_loss", "brier"]].to_string(index=False))

    best = model_summary.sort_values(["accuracy", "f1_macro", "draw_f1"], ascending=False).iloc[0]
    market = market_summary.iloc[0]
    print(f"\nBest model: {best['model']} accuracy={best['accuracy']:.2%}, draw_f1={best['draw_f1']:.4f}")
    print(f"Bet365 closing: accuracy={market['accuracy']:.2%}, draw_f1={market['draw_f1']:.4f}")
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
