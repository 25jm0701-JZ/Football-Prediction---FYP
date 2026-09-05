#!/usr/bin/env python3
"""
Phase 3: Atta Mills-style reproduction with half-time information.

This is an in-play comparison experiment rather than a pre-match model.
It keeps the paper-style feature families from Phase 1, adds current-match
half-time goals/result, and deliberately excludes the project's original
extras such as H2H, rolling shots, FootyStats, player features, and O/U targets.

Two configurations are evaluated:
  - paper_ht_no_odds: paper-style team features + half-time variables.
  - paper_ht_with_b365_opening: the same features plus Bet365 opening odds
    converted to de-vigged probabilities.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "atta_mills_pl_walkforward" / "phase3_halftime_paper_replication"
WINDOWS = [5, 10]


def build_halftime_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Add current-match half-time score/result features."""
    features = pd.DataFrame(index=df.index)
    features["ht_home_goals"] = pd.to_numeric(df["home_goals_half"], errors="coerce")
    features["ht_away_goals"] = pd.to_numeric(df["away_goals_half"], errors="coerce")
    features["ht_goal_diff"] = features["ht_home_goals"] - features["ht_away_goals"]
    features["ht_total_goals"] = features["ht_home_goals"] + features["ht_away_goals"]
    features["ht_home_lead"] = (features["ht_goal_diff"] > 0).astype(float)
    features["ht_draw"] = (features["ht_goal_diff"] == 0).astype(float)
    features["ht_away_lead"] = (features["ht_goal_diff"] < 0).astype(float)

    cols = list(features.columns)
    mapping = pd.DataFrame([
        {
            "feature": col,
            "feature_family": "Half-Time Match State",
            "home_or_away": "match",
            "window": "",
            "interpretation": "Current-match half-time score/result information",
            "used_in_phase1": False,
            "used_in_phase3": True,
            "phase": "Paper half-time variables",
        }
        for col in cols
    ])
    return features, cols, mapping


def build_opening_odds_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Convert Bet365 opening odds to de-vigged probabilities for model input."""
    odds = df[["odds_home", "odds_draw", "odds_away"]].apply(pd.to_numeric, errors="coerce")
    valid = np.all(np.isfinite(odds.to_numpy(dtype=float)) & (odds.to_numpy(dtype=float) > 1.0), axis=1)

    features = pd.DataFrame(index=df.index)
    features["b365_open_home_prob"] = np.nan
    features["b365_open_draw_prob"] = np.nan
    features["b365_open_away_prob"] = np.nan
    features["b365_open_overround"] = np.nan

    implied = 1.0 / odds.loc[valid, ["odds_home", "odds_draw", "odds_away"]]
    totals = implied.sum(axis=1)
    normalized = implied.div(totals, axis=0)
    features.loc[valid, "b365_open_home_prob"] = normalized["odds_home"]
    features.loc[valid, "b365_open_draw_prob"] = normalized["odds_draw"]
    features.loc[valid, "b365_open_away_prob"] = normalized["odds_away"]
    features.loc[valid, "b365_open_overround"] = totals - 1.0

    cols = list(features.columns)
    mapping = pd.DataFrame([
        {
            "feature": col,
            "feature_family": "Bet Odds",
            "home_or_away": "market",
            "window": "",
            "interpretation": "Bet365 opening odds converted to implied probability/overround",
            "used_in_phase1": False,
            "used_in_phase3": True,
            "phase": "Paper betting-odds variables",
        }
        for col in cols
    ])
    return features, cols, mapping


def build_phase3_features(
    df: pd.DataFrame,
    include_opening_odds: bool,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    atta_df, atta_cols = build_atta_mills_features(df, windows=WINDOWS)
    phase3 = atta_df.copy()

    maps = []
    atta_map = feature_mapping_table(atta_cols).rename(columns={"paper_family": "feature_family"})
    atta_map["used_in_phase3"] = True
    atta_map["phase"] = "Atta Mills-only"
    atta_map = atta_map[[
        "feature", "feature_family", "home_or_away", "window",
        "interpretation", "used_in_phase1", "used_in_phase3", "phase",
    ]]
    maps.append(atta_map)

    ht_features, ht_cols, ht_map = build_halftime_features(phase3)
    for col in ht_cols:
        phase3[col] = ht_features[col].values
    maps.append(ht_map)

    feature_cols = atta_cols + ht_cols
    if include_opening_odds:
        odds_features, odds_cols, odds_map = build_opening_odds_features(phase3)
        for col in odds_cols:
            phase3[col] = odds_features[col].values
        feature_cols += odds_cols
        maps.append(odds_map)

    mapping = pd.concat(maps, ignore_index=True)
    return phase3, feature_cols, mapping


def _format_model_table(models: pd.DataFrame) -> str:
    table = models[["model", "accuracy", "f1_macro", "f1_weighted", "draw_f1", "log_loss", "brier"]].copy()
    for col in ["accuracy", "f1_macro", "f1_weighted", "draw_f1", "log_loss", "brier"]:
        table[col] = table[col].map(lambda x: f"{x:.4f}")
    return "\n".join([
        f"| {r.model} | {r.accuracy} | {r.f1_macro} | {r.f1_weighted} | {r.draw_f1} | {r.log_loss} | {r.brier} |"
        for r in table.itertuples(index=False)
    ])


def write_phase3_doc(df: pd.DataFrame, summaries: dict[str, dict[str, object]]) -> None:
    doc = PROJECT_ROOT / "docs" / "atta_mills_phase3_halftime_results.md"

    phase1_note = "Phase 1 comparison file was not found."
    phase1_path = PHASE1_OUTPUT_DIR / "model_comparison_walkforward.csv"
    if phase1_path.exists():
        phase1 = pd.read_csv(phase1_path)
        phase1_best = phase1.sort_values(["accuracy", "f1_macro", "draw_f1"], ascending=False).iloc[0]
        phase1_note = f"`{phase1_best['model']}` at {phase1_best['accuracy']:.2%} accuracy"

    sections = []
    comparison_rows = []
    for config_name, data in summaries.items():
        models = data["models"]
        market = data["market"]
        best = data["best"]
        feature_count = data["feature_count"]
        feature_note = data["feature_note"]
        comparison_rows.append(
            f"| {config_name} | {feature_count} | {best['model']} | {best['accuracy']:.4f} | "
            f"{best['f1_macro']:.4f} | {best['draw_f1']:.4f} | {best['log_loss']:.4f} | "
            f"{market['accuracy']:.4f} |"
        )
        sections.append(f"""### {config_name}

Feature set: {feature_count} features. {feature_note}

| Model | Accuracy | Macro F1 | Weighted F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
{_format_model_table(models)}

Best model: `{best['model']}` at {best['accuracy']:.2%} accuracy and {best['draw_f1']:.4f} Draw F1.
Bet365 closing market benchmark: {market['accuracy']:.2%} accuracy, {market['draw_f1']:.4f} Draw F1.
""")

    comparison = "\n".join(comparison_rows)
    details = "\n".join(sections)
    content = f"""# Atta Mills Phase 3 Half-Time Results

## Scope

- Data: Premier League football-data.co.uk files, {df['season_id'].nunique()} seasons, {len(df)} matches.
- Target: full-time H/D/A result.
- Validation: season-by-season walk-forward.
- Purpose: reproduce the paper-style in-play setting by adding half-time information.
- Excluded project extras: H2H, rolling shot-efficiency features, FootyStats, player features, and O/U 2.5 target/features.
- Market benchmark: Bet365 closing odds, converted to de-vigged implied probabilities.

## Aggregate Comparison

Phase 1 strict pre-match best: {phase1_note}.

| Config | Features | Best Model | Accuracy | Macro F1 | Draw F1 | Log Loss | Market Accuracy |
|---|---:|---|---:|---:|---:|---:|---:|
{comparison}

## Detailed Results

{details}

## Interpretation

This experiment is not a deployable pre-match predictor because current-match
half-time goals/result are only known after the first half. It is an explicit
methodology check: if accuracy jumps toward the Atta Mills et al. headline range
after adding half-time variables, the gap is likely explained by the paper's
in-play information rather than by the clean pre-match feature set.
"""
    doc.write_text(content, encoding="utf-8")


def run_config(name: str, include_opening_odds: bool, df: pd.DataFrame) -> dict[str, object]:
    import scripts.atta_mills_pl_phase1 as phase1

    out_dir = OUTPUT_ROOT / name
    phase1.OUTPUT_DIR = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_feat, feature_cols, mapping = build_phase3_features(df, include_opening_odds=include_opening_odds)
    mapping.to_csv(out_dir / "feature_set.csv", index=False)
    df_feat[["date", "season_id", "home_team", "away_team", "target_result", *feature_cols]].to_csv(
        out_dir / "feature_matrix.csv",
        index=False,
    )

    fold_metrics, aggregate_metrics, predictions = run_walk_forward(df_feat, feature_cols, build_models)
    fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(out_dir / "predictions_by_fold.csv", index=False)

    aggregate_metrics = aggregate_metrics.sort_values(
        ["source", "accuracy", "f1_macro", "draw_f1"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    models = aggregate_metrics[aggregate_metrics["source"] == "model"].copy()
    market = aggregate_metrics[aggregate_metrics["source"] == "market"].iloc[0]
    models.to_csv(out_dir / "model_comparison_walkforward.csv", index=False)
    aggregate_metrics[aggregate_metrics["source"] == "market"].to_csv(
        out_dir / "market_comparison_bet365_closing.csv",
        index=False,
    )
    aggregate_metrics.to_csv(out_dir / "aggregate_metrics_all.csv", index=False)

    best = models.sort_values(["accuracy", "f1_macro", "draw_f1"], ascending=False).iloc[0]
    feature_note = (
        "Includes 44 Atta Mills-style team-state features, 7 half-time features, and 4 Bet365 opening-odds features."
        if include_opening_odds
        else "Includes 44 Atta Mills-style team-state features and 7 half-time features."
    )
    return {
        "models": models,
        "market": market,
        "best": best,
        "feature_count": len(feature_cols),
        "feature_note": feature_note,
        "output_dir": out_dir,
    }


def main() -> None:
    print("=" * 78)
    print("Atta Mills Phase 3: paper-style half-time reproduction")
    print("=" * 78)

    df = load_pl_data()
    print(f"[Data] Matches: {len(df)}")
    print(f"[Data] Seasons: {sorted(df['season_id'].unique())}")

    configs = {
        "paper_ht_no_odds": False,
        "paper_ht_with_b365_opening": True,
    }
    summaries = {}
    for name, include_opening_odds in configs.items():
        print(f"\n=== Running {name} ===")
        summaries[name] = run_config(name, include_opening_odds, df)
        best = summaries[name]["best"]
        market = summaries[name]["market"]
        print(
            f"Best model: {best['model']} accuracy={best['accuracy']:.2%}, "
            f"draw_f1={best['draw_f1']:.4f}"
        )
        print(f"Bet365 closing: accuracy={market['accuracy']:.2%}, draw_f1={market['draw_f1']:.4f}")

    write_phase3_doc(df, summaries)

    print("\n=== Summary ===")
    rows = []
    for name, summary in summaries.items():
        best = summary["best"]
        market = summary["market"]
        rows.append({
            "config": name,
            "features": summary["feature_count"],
            "best_model": best["model"],
            "accuracy": best["accuracy"],
            "macro_f1": best["f1_macro"],
            "draw_f1": best["draw_f1"],
            "market_accuracy": market["accuracy"],
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_ROOT}")
    print(f"Saved doc to: {PROJECT_ROOT / 'docs' / 'atta_mills_phase3_halftime_results.md'}")


if __name__ == "__main__":
    main()
