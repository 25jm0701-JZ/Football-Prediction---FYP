"""Test if qualifier-derived features improve World Cup predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
WC_DIR = THIS_DIR.parent
PROJECT_ROOT = WC_DIR.parent
sys.path.insert(0, str(WC_DIR))
sys.stdout.reconfigure(encoding="utf-8")

from src.world_cup_model import HOSTS, canonical_team, load_rankings

RAW = WC_DIR / "data" / "raw"
rankings_path = WC_DIR / "world_cup_rankings_4_sheets_simple.xlsx"

# ===== 1. Load qualifier data =====
qual_files = sorted(RAW.glob("international-wc-qualification-*-matches-*.csv"))
print(f"Found {len(qual_files)} qualifier files")
qual_all = pd.concat([pd.read_csv(f) for f in qual_files], ignore_index=True)
print(f"Total qualifier matches: {len(qual_all)}")

qual_all["home_name"] = qual_all["home_team_name"].map(canonical_team)
qual_all["away_name"] = qual_all["away_team_name"].map(canonical_team)
qual_all["date"] = pd.to_datetime(qual_all["date_GMT"])

# ===== 2. Compute qualifier-derived features for each 2026 WC team =====
cutoff_26 = pd.Timestamp("2026-06-11")
fixtures = pd.read_csv(WC_DIR / "data" / "fixtures_2026_all.csv")
wc_teams = sorted(set(fixtures["Home"].unique()) | set(fixtures["Away"].unique()) - {"TBD"})

pre_features = {}
for team in wc_teams:
    as_home = qual_all[(qual_all["home_name"] == team) & (qual_all["date"] < cutoff_26)].tail(10)
    as_away = qual_all[(qual_all["away_name"] == team) & (qual_all["date"] < cutoff_26)].tail(10)
    nh, na = len(as_home), len(as_away)
    nt = nh + na

    if nt == 0:
        pre_features[team] = {"qual_matches": 0, "qual_ppg": 0, "qual_xg": 0, "qual_gd": 0, "qual_sot": 0}
        continue

    home_pts = sum(
        3 if r["home_team_goal_count"] > r["away_team_goal_count"]
        else (1 if r["home_team_goal_count"] == r["away_team_goal_count"] else 0)
        for _, r in as_home.iterrows()
    )
    away_pts = sum(
        3 if r["away_team_goal_count"] > r["home_team_goal_count"]
        else (1 if r["away_team_goal_count"] == r["home_team_goal_count"] else 0)
        for _, r in as_away.iterrows()
    )
    ppg = (home_pts + away_pts) / nt

    total_xg = as_home["Home Team Pre-Match xG"].sum() + as_away["Away Team Pre-Match xG"].sum()
    gf = as_home["home_team_goal_count"].sum() + as_away["away_team_goal_count"].sum()
    ga = as_home["away_team_goal_count"].sum() + as_away["home_team_goal_count"].sum()
    total_sot = as_home["home_team_shots_on_target"].sum() + as_away["away_team_shots_on_target"].sum()

    pre_features[team] = {
        "qual_matches": nt,
        "qual_ppg": ppg,
        "qual_xg": total_xg / nt,
        "qual_gd": (gf - ga) / nt,
        "qual_sot": total_sot / nt,
    }

feat_df = pd.DataFrame.from_dict(pre_features, orient="index")
print("\n=== Top 10 by Qualifier PPG ===")
print(feat_df.sort_values("qual_ppg", ascending=False).head(10).round(3).to_string())

print("\n=== Bottom 5 ===")
print(feat_df.sort_values("qual_ppg").head(5).to_string())

print(f"\n=== Teams with no qualifier data ===")
print(feat_df[feat_df["qual_matches"] == 0].to_string())

# ===== 3. Compare qualifier PPG vs FIFA ranking as predictors =====
# We can't do proper cross-validation since historical qualifier xG isn't available
# But we can check: does qualifier PPG rank teams similarly to FIFA ranking?

rankings_26, _ = load_rankings(rankings_path, 2026)
fifa_rank = rankings_26.set_index("Country")["Points"].to_dict()

print("\n=== Team comparison: FIFA rank vs Qualifier PPG ===")
comparison = []
for team in wc_teams:
    fpts = fifa_rank.get(team, 0)
    fz = (fpts - 1500) / 200  # approximate z-score
    qual = pre_features.get(team, {})
    comparison.append({
        "Team": team,
        "FIFA_Pts": fpts,
        "Qual_PPG": qual.get("qual_ppg", 0),
        "Qual_xG": qual.get("qual_xg", 0),
        "Qual_GD": qual.get("qual_gd", 0),
    })

comp_df = pd.DataFrame(comparison)
print(comp_df.sort_values("FIFA_Pts", ascending=False).head(10).to_string())
print()
print("Correlation between FIFA points and Qualifier PPG:",
      comp_df[comp_df["Qual_PPG"] > 0]["FIFA_Pts"].corr(comp_df[comp_df["Qual_PPG"] > 0]["Qual_PPG"]).round(3))
print("Correlation between FIFA points and Qualifier xG:",
      comp_df[comp_df["Qual_xG"] > 0]["FIFA_Pts"].corr(comp_df[comp_df["Qual_xG"] > 0]["Qual_xG"]).round(3))
print("Correlation between FIFA points and Qualifier GD:",
      comp_df[comp_df["Qual_GD"] > 0]["FIFA_Pts"].corr(comp_df[comp_df["Qual_GD"] > 0]["Qual_GD"]).round(3))
