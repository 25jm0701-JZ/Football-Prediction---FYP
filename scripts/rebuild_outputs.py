"""
Regenerate all prediction output files with readable names and 3 decimal places.
Run whenever you want to refresh predictions.

Usage:
    python scripts/rebuild_outputs.py
"""

import sys
import json
from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJ = THIS_DIR.parent
WC_DIR = PROJ / "world_cup"
OUT = PROJ / "outputs" / "world_cup"
sys.path.insert(0, str(WC_DIR))

from src.world_cup_model import (
    predict_fixtures,
    load_model_bundle,
    set_intl_results_path,
)
from src.player_features import load_team_features
from src.poisson_model import WeightedPoissonModel
from src.platform_prediction import build_platform_predictions


def pick(r):
    return np.array(["H", "D", "A"])[np.argmax(r)]


def _write_csv(df, path):
    """Write CSV directly (overwrites even if file is open in another app)."""
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {path.name} ({len(df)} rows)")


def main():
    intl_path = WC_DIR / "data" / "raw" / "international_results.csv"
    if intl_path.exists():
        set_intl_results_path(intl_path)
        print(f"[Form] Using intl results: {intl_path.name}")

    RANKINGS = OUT / "ratings_2026_live.csv"
    BUNDLE = load_model_bundle(OUT / "world_cup_model.json")
    POISSON = WeightedPoissonModel.load(OUT / "poisson_model.json")
    WEIGHTS = json.loads(
        (OUT / "historical_backtest" / "global_descriptive_weights.json").read_text()
    )

    ALL = pd.read_csv(WC_DIR / "data" / "fixtures_2026_all.csv")
    pred = ALL[(ALL["Home"] != "TBD") & (ALL["Away"] != "TBD")].copy()

    # ------------------------------------------------------------------
    # 1. Softmax predictions
    # ------------------------------------------------------------------
    player_path = Path(
        "C:/Users/-jmmmm/Downloads"
        "/international-world-cup-players-2026-to-2026-stats.csv"
    )
    tf = load_team_features({2026: player_path}) if player_path.exists() else None
    sm = predict_fixtures(pred, RANKINGS, BUNDLE, team_features=tf)

    sm_out = pd.DataFrame({
        "Match_No": sm["MatchNumber"].astype(int),
        "Date_UTC": sm["DateUTC"],
        "Stage": sm["Stage"],
        "Home_Team": sm["Home"],
        "Away_Team": sm["Away"],
        "Host_Advantage": sm["host_advantage"].astype(int),
        "Prob_Home": sm["P_H"].round(3),
        "Prob_Draw": sm["P_D"].round(3),
        "Prob_Away": sm["P_A"].round(3),
        "Fair_Odds_Home": (1.0 / sm["P_H"]).round(3),
        "Fair_Odds_Draw": (1.0 / sm["P_D"]).round(3),
        "Fair_Odds_Away": (1.0 / sm["P_A"]).round(3),
    })
    sm_out["Predicted_Winner"] = sm_out[["Prob_Home", "Prob_Draw", "Prob_Away"]].apply(pick, axis=1)
    sm_out["Home_Goals"] = ""
    sm_out["Away_Goals"] = ""
    sm_out["Actual_Winner"] = ""
    sm_out["Correct"] = ""
    _write_csv(sm_out, OUT / "predictions_2026.csv")
    print(f"[1] predictions_2026.csv       -- {len(sm_out)} matches  (Softmax)")

    # ------------------------------------------------------------------
    # 2. Platform predictions (3-model blend)
    # ------------------------------------------------------------------
    for c in ["DateChina", "Group", "GroupRound", "Stadium", "City"]:
        if c not in pred.columns:
            pred[c] = ""

    plat = build_platform_predictions(
        pred, RANKINGS, BUNDLE, POISSON,
        (float(WEIGHTS["weight_result"]), float(WEIGHTS["weight_market_proxy"]), float(WEIGHTS["weight_poisson"])),
    )

    plat_out = pd.DataFrame({
        "Match_No": plat["MatchNumber"].astype(int),
        "Date_UTC": plat["DateUTC"],
        "Stage": plat["Stage"],
        "Home_Team": plat["Home"],
        "Away_Team": plat["Away"],
        "Host_Advantage": plat["HostAdvantage"].astype(int),
        "Prob_Home": plat["OfficialP_H"].round(3),
        "Prob_Draw": plat["OfficialP_D"].round(3),
        "Prob_Away": plat["OfficialP_A"].round(3),
        "Fair_Odds_Home": (1.0 / plat["OfficialP_H"]).round(3),
        "Fair_Odds_Draw": (1.0 / plat["OfficialP_D"]).round(3),
        "Fair_Odds_Away": (1.0 / plat["OfficialP_A"]).round(3),
        "Model_Consistency": plat["ModelConsistency"],
        "Model_Disagreement": plat["ModelDisagreement"].round(3),
        "SM_Prob_Home": plat["ResultP_H"].round(3),
        "SM_Prob_Draw": plat["ResultP_D"].round(3),
        "SM_Prob_Away": plat["ResultP_A"].round(3),
        "MP_Prob_Home": plat["MarketProxyP_H"].round(3),
        "MP_Prob_Draw": plat["MarketProxyP_D"].round(3),
        "MP_Prob_Away": plat["MarketProxyP_A"].round(3),
        "PS_Prob_Home": plat["PoissonP_H"].round(3),
        "PS_Prob_Draw": plat["PoissonP_D"].round(3),
        "PS_Prob_Away": plat["PoissonP_A"].round(3),
        "xG_Home": plat["Poisson_xG_H"].round(3),
        "xG_Away": plat["Poisson_xG_A"].round(3),
        "Most_Likely_Score": plat["MostLikelyScore"],
    })
    for col_set, name in [
        (["Prob_Home", "Prob_Draw", "Prob_Away"], "Predicted_Winner"),
        (["SM_Prob_Home", "SM_Prob_Draw", "SM_Prob_Away"], "SM_Prediction"),
        (["PS_Prob_Home", "PS_Prob_Draw", "PS_Prob_Away"], "PS_Prediction"),
    ]:
        plat_out[name] = plat_out[col_set].apply(pick, axis=1)

    plat_out["Home_Goals"] = ""
    plat_out["Away_Goals"] = ""
    plat_out["Actual_Winner"] = ""
    plat_out["Official_Correct"] = ""
    plat_out["Softmax_Correct"] = ""
    plat_out["Poisson_Correct"] = ""
    _write_csv(plat_out, OUT / "platform_predictions_2026.csv")
    print(f"[2] platform_predictions_2026.csv  -- {len(plat_out)} matches  (3-model blend)")

    # ------------------------------------------------------------------
    # 3. Backtest master
    # ------------------------------------------------------------------
    played = ALL[ALL["Played"] == True]
    played_map = {}
    for _, r in played.iterrows():
        mn = int(r["MatchNumber"])
        if pd.notna(r["HomeTeamScore"]) and pd.notna(r["AwayTeamScore"]):
            hg, ag = int(r["HomeTeamScore"]), int(r["AwayTeamScore"])
            actual = "H" if hg > ag else ("D" if hg == ag else "A")
            played_map[mn] = (hg, ag, actual)

    bt = pd.DataFrame({
        "Match": plat_out["Match_No"],
        "Date_UTC": plat_out["Date_UTC"],
        "Stage": plat_out["Stage"],
        "Home": plat_out["Home_Team"],
        "Away": plat_out["Away_Team"],
        "Host_Adv": plat_out["Host_Advantage"],
        "Prob_Home": plat_out["Prob_Home"],
        "Prob_Draw": plat_out["Prob_Draw"],
        "Prob_Away": plat_out["Prob_Away"],
        "Predicted": plat_out["Predicted_Winner"],
        "SM_Home": plat_out["SM_Prob_Home"],
        "SM_Draw": plat_out["SM_Prob_Draw"],
        "SM_Away": plat_out["SM_Prob_Away"],
        "SM_Pred": plat_out["SM_Prediction"],
        "PS_Home": plat_out["PS_Prob_Home"],
        "PS_Draw": plat_out["PS_Prob_Draw"],
        "PS_Away": plat_out["PS_Prob_Away"],
        "PS_Pred": plat_out["PS_Prediction"],
        "xG_Home": plat_out["xG_Home"],
        "xG_Away": plat_out["xG_Away"],
        "Likely_Score": plat_out["Most_Likely_Score"],
        "Consistency": plat_out["Model_Consistency"],
        "Disagreement": plat_out["Model_Disagreement"],
        "Home_Goals": "",
        "Away_Goals": "",
        "Actual": "",
        "Official_Correct": "",
        "SM_Correct": "",
        "PS_Correct": "",
    })
    for i, r in bt.iterrows():
        mn = r["Match"]
        if mn in played_map:
            hg, ag, actual = played_map[mn]
            bt.at[i, "Home_Goals"] = hg
            bt.at[i, "Away_Goals"] = ag
            bt.at[i, "Actual"] = actual
            bt.at[i, "Official_Correct"] = "Yes" if r["Predicted"] == actual else "No"
            bt.at[i, "SM_Correct"] = "Yes" if r["SM_Pred"] == actual else "No"
            bt.at[i, "PS_Correct"] = "Yes" if r["PS_Pred"] == actual else "No"

    _write_csv(bt, OUT / "backtest_master.csv")
    print(f"[3] backtest_master.csv       -- {len(bt)} matches")

    # Summary
    now = dt.datetime.now(dt.timezone.utc)
    played_count = len(played_map)
    print(f"\n=== Summary ({now.strftime('%Y-%m-%d %H:%M UTC')}) ===")
    print(f"  Total: 91 | Played: {played_count} | Future: {91 - played_count}")
    if played_count > 0:
        pdf = bt[bt["Actual"] != ""]
        for name, col in [("Official", "Official_Correct"), ("Softmax", "SM_Correct"), ("Poisson", "PS_Correct")]:
            acc = (pdf[col] == "Yes").mean()
            print(f"  {name:10s} accuracy: {acc:.1%} ({int((pdf[col]=='Yes').sum())}/{played_count})")


if __name__ == "__main__":
    main()
