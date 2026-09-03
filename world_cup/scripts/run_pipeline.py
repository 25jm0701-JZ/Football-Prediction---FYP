from pathlib import Path
import sys

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent  # world_cup/scripts/
WC_DIR = THIS_DIR.parent  # world_cup/
PROJECT_ROOT = WC_DIR.parent  # FYP/
sys.path.insert(0, str(WC_DIR))

from src.world_cup_model import (
    FORM_FEATURES,
    XG_FEATURES,
    cross_validate,
    fit_models,
    format_probability_columns,
    load_historical_matches,
    load_qualifier_matches,
    load_rankings,
    player_feature_names,
    save_model_bundle,
    set_intl_results_path,
    use_xgboost_result_model,
)


GAMES_PATH = WC_DIR / "data" / "raw" / "WorldCup2026.xlsx"
RANKINGS_PATH = WC_DIR / "world_cup_rankings_4_sheets_simple.xlsx"
LIVE_SNAPSHOT_PATH = WC_DIR / "data" / "raw" / "fifa_men_live_2026.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "world_cup"
BLEND_WEIGHT = 0.5

# Player data paths — adjust to your download location
PLAYER_DATA_DIR = Path("C:/Users/-jmmmm/Downloads")
PLAYER_DATA_PATHS = {
    2014: PLAYER_DATA_DIR / "international-fifa-world-cup-2014-brazil-players-2014-to-2014-stats.csv",
    2018: PLAYER_DATA_DIR / "international-fifa-world-cup-2018-russia-players-2018-to-2018-stats.csv",
    2022: PLAYER_DATA_DIR / "international-fifa-world-cup-2022-qatar-players-2022-to-2022-stats.csv",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Enable rolling-form features from the big international-results dataset
    intl_path = WC_DIR / "data" / "raw" / "international_results.csv"
    if intl_path.exists():
        set_intl_results_path(intl_path)
        print(f"[Form] Using international results: {intl_path}")

    # --- Load historical World Cup matches (2014 / 2018 / 2022) ---
    matches, historical_stats = load_historical_matches(
        GAMES_PATH, RANKINGS_PATH, player_data_paths=PLAYER_DATA_PATHS
    )

    # --- Append qualifier matches for extra training data -------------
    qualifiers = load_qualifier_matches(GAMES_PATH, RANKINGS_PATH)
    if qualifiers is not None and len(qualifiers):
        print(f"[Qualifiers] Loaded {len(qualifiers)} qualifier matches")
        # Zero-fill player-feature columns that qualifiers don't have
        for pf in player_feature_names():
            if pf not in qualifiers.columns:
                qualifiers[pf] = 0.0
        matches = pd.concat([matches, qualifiers], ignore_index=True)
    else:
        print("[Qualifiers] None found — training on WC finals only")

    # --- Optional: switch result model to XGBoost (uncomment to enable) --
    # try:
    #     use_xgboost_result_model()
    #     print("[Model] Using XGBoost for result model")
    # except RuntimeError as e:
    #     print(f"[Model] {e} -- falling back to Softmax")

    # --- Cross-validate & train final models --------------------------
    cv_predictions, metrics = cross_validate(matches, blend_weight=BLEND_WEIGHT)
    result_model, market_model = fit_models(matches)
    rankings_2026, stats_2026 = load_rankings(
        LIVE_SNAPSHOT_PATH if LIVE_SNAPSHOT_PATH.exists() else RANKINGS_PATH,
        2026,
    )
    rating_stats = {**historical_stats, "2026": stats_2026}

    model_path = OUTPUT_DIR / "world_cup_model.json"
    save_model_bundle(
        model_path,
        result_model,
        market_model,
        rating_stats,
        blend_weight=BLEND_WEIGHT,
    )

    processed_columns = [
        "Year",
        "Date",
        "Home",
        "Away",
        "HGFT",
        "AGFT",
        "Result",
        "HomePoints",
        "AwayPoints",
        "HomeRatingZ",
        "AwayRatingZ",
        "rating_diff_z",
        "host_advantage",
        "knockout",
        "H-Avg",
        "D-Avg",
        "A-Avg",
        "MarketP_H",
        "MarketP_D",
        "MarketP_A",
        *player_feature_names(),
        *FORM_FEATURES,
        *XG_FEATURES,
    ]
    processed_matches = format_probability_columns(
        matches[processed_columns],
        ["MarketP_H", "MarketP_D", "MarketP_A"],
    )
    rounded_cv_predictions = format_probability_columns(
        cv_predictions,
        ["P_H", "P_D", "P_A"],
    )
    processed_matches.to_csv(
        OUTPUT_DIR / "historical_matches_processed.csv", index=False, encoding="utf-8-sig"
    )
    rounded_cv_predictions.to_csv(
        OUTPUT_DIR / "cross_validation_predictions.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    rankings_2026[["Country", "Rank", "Points"]].to_csv(
        OUTPUT_DIR / "ratings_2026_live.csv", index=False, encoding="utf-8-sig"
    )

    print("Pipeline complete.")
    print(metrics.to_string(index=False))
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
