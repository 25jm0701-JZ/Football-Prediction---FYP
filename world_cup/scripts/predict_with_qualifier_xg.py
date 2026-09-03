"""Predict Round of 32 using Softmax model enhanced with qualifier xG features."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
WC_DIR = THIS_DIR.parent
PROJECT_ROOT = WC_DIR.parent
sys.path.insert(0, str(WC_DIR))
sys.stdout.reconfigure(encoding="utf-8")

from src.world_cup_model import HOSTS, canonical_team, load_rankings, SoftmaxModel

RAW = WC_DIR / "data" / "raw"
rankings_path = WC_DIR / "world_cup_rankings_4_sheets_simple.xlsx"
bundle_path = PROJECT_ROOT / "outputs" / "world_cup" / "world_cup_model.json"

# ===== 1. Load qualifier data =====
print("Loading qualifier data...")
qual_files = sorted(RAW.glob("international-wc-qualification-*-matches-*.csv"))
qual_all = pd.concat([pd.read_csv(f) for f in qual_files], ignore_index=True)
qual_all["home_name"] = qual_all["home_team_name"].map(canonical_team)
qual_all["away_name"] = qual_all["away_team_name"].map(canonical_team)
qual_all["date"] = pd.to_datetime(qual_all["date_GMT"], errors="coerce")
cutoff_26 = pd.Timestamp("2026-06-11")

# ===== 2. Compute qualifier features for each 2026 WC team =====
fixtures = pd.read_csv(WC_DIR / "data" / "fixtures_2026_round32.csv")
fixtures["Home"] = fixtures["Home"].map(canonical_team)
fixtures["Away"] = fixtures["Away"].map(canonical_team)
wc_teams = sorted(set(fixtures["Home"]) | set(fixtures["Away"]))

team_stats = {}
for team in wc_teams:
    ah = qual_all[(qual_all["home_name"] == team) & (qual_all["date"] < cutoff_26)].tail(10)
    aa = qual_all[(qual_all["away_name"] == team) & (qual_all["date"] < cutoff_26)].tail(10)
    nh, na = len(ah), len(aa)
    nt = nh + na

    if nt == 0:
        # Host team - use international_results.csv as fallback
        ir = pd.read_csv(RAW / "international_results.csv")
        ir["date"] = pd.to_datetime(ir["date"])
        ir["home_team"] = ir["home_team"].map(canonical_team)
        ir["away_team"] = ir["away_team"].map(canonical_team)

        as_h = ir[(ir["home_team"] == team) & (ir["date"] < cutoff_26)].tail(10)
        as_a = ir[(ir["away_team"] == team) & (ir["date"] < cutoff_26)].tail(10)
        pts_h = sum(3 if r["home_score"] > r["away_score"] else (1 if r["home_score"] == r["away_score"] else 0) for _, r in as_h.iterrows())
        pts_a = sum(3 if r["away_score"] > r["home_score"] else (1 if r["away_score"] == r["home_score"] else 0) for _, r in as_a.iterrows())
        tg = len(as_h) + len(as_a)

        team_stats[team] = {
            "qual_matches": tg,
            "qual_ppg": (pts_h + pts_a) / max(tg, 1),
            "qual_xg": 0.0,
            "qual_gd": ((as_h["home_score"].sum() + as_a["away_score"].sum()) - (as_h["away_score"].sum() + as_a["home_score"].sum())) / max(tg, 1),
        }
        continue

    pts_h = sum(3 if r["home_team_goal_count"] > r["away_team_goal_count"]
                else (1 if r["home_team_goal_count"] == r["away_team_goal_count"] else 0) for _, r in ah.iterrows())
    pts_a = sum(3 if r["away_team_goal_count"] > r["home_team_goal_count"]
                else (1 if r["away_team_goal_count"] == r["home_team_goal_count"] else 0) for _, r in aa.iterrows())

    team_stats[team] = {
        "qual_matches": nt,
        "qual_ppg": (pts_h + pts_a) / nt,
        "qual_xg": (ah["Home Team Pre-Match xG"].sum() + aa["Away Team Pre-Match xG"].sum()) / nt,
        "qual_gd": (
            (ah["home_team_goal_count"].sum() + aa["away_team_goal_count"].sum()) -
            (ah["away_team_goal_count"].sum() + aa["home_team_goal_count"].sum())
        ) / nt,
    }

# ===== 3. Load existing Softmax model & compute baseline predictions =====
print("\nLoading Softmax model...")
bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
bundle["result_model"] = SoftmaxModel.from_dict(bundle["result_model"])
bundle["market_proxy_model"] = SoftmaxModel.from_dict(bundle["market_proxy_model"])
model = bundle["result_model"]

rankings_26, stats_26 = load_rankings(rankings_path, 2026)
fifa_26 = rankings_26.set_index("Country")["RatingZ"].to_dict()

# ---- Baseline: FIFA ranking only ----
def predict_softmax(home, away):
    home_z = fifa_26.get(home, 0)
    away_z = fifa_26.get(away, 0)
    feat = np.array([[home_z - away_z, int(home in HOSTS[2026]) - int(away in HOSTS[2026]), 1]])
    probs = model.predict_proba(feat)[0]
    return {"H": probs[0], "D": probs[1], "A": probs[2]}

# ---- Enhanced: FIFA ranking + qualifier xG ----
# Train a simple LR model on 2014+2018+2022 WC data with qualifier-style features
from sklearn.linear_model import LogisticRegression

IR = pd.read_csv(RAW / "international_results.csv")
IR["date"] = pd.to_datetime(IR["date"])
IR["home_team"] = IR["home_team"].map(canonical_team)
IR["away_team"] = IR["away_team"].map(canonical_team)

# Build historical training data with PPG features (as proxy for qualifier form)
wc_files = {
    2014: RAW / "international-fifa-world-cup-2014-brazil-matches-2014-to-2014-stats.csv",
    2018: RAW / "international-fifa-world-cup-2018-russia-matches-2018-to-2018-stats.csv",
    2022: RAW / "international-fifa-world-cup-2022-qatar-matches-2022-to-2022-stats.csv",
}
wc_starts = {2014: pd.Timestamp("2014-06-12"), 2018: pd.Timestamp("2018-06-14"), 2022: pd.Timestamp("2022-11-20")}

train_dfs = []
for year, path in wc_files.items():
    df = pd.read_csv(path)
    df["Home"] = df["home_team_name"].map(canonical_team)
    df["Away"] = df["away_team_name"].map(canonical_team)
    df["Target"] = np.where(df["home_team_goal_count"] > df["away_team_goal_count"], "H",
                            np.where(df["home_team_goal_count"] == df["away_team_goal_count"], "D", "A"))

    rankings_y, _ = load_rankings(rankings_path, year)
    fifa_y = rankings_y.set_index("Country")["RatingZ"].to_dict()
    df["rating_diff_z"] = df["Home"].map(fifa_y) - df["Away"].map(fifa_y)
    df["rating_diff_z"] = df["rating_diff_z"].fillna(0)
    df["host_advantage"] = (df["Home"].isin(HOSTS[year])).astype(int) - (df["Away"].isin(HOSTS[year])).astype(int)
    df["knockout"] = df["Game Week"].isna().astype(int)

    # Pre-tournament PPG from IR (proxy for qualifier form)
    cutoff = wc_starts[year]
    for team in set(df["Home"]) | set(df["Away"]):
        ah = IR[(IR["home_team"] == team) & (IR["date"] < cutoff)].tail(10)
        aa = IR[(IR["away_team"] == team) & (IR["date"] < cutoff)].tail(10)
        pts_h = sum(3 if r["home_score"] > r["away_score"] else (1 if r["home_score"] == r["away_score"] else 0) for _, r in ah.iterrows())
        pts_a = sum(3 if r["away_score"] > r["home_score"] else (1 if r["away_score"] == r["home_score"] else 0) for _, r in aa.iterrows())
        tg = len(ah) + len(aa)
        df.loc[df["Home"] == team, "form_ppg"] = (pts_h + pts_a) / max(tg, 1)
        df.loc[df["Away"] == team, "form_ppg_away"] = (pts_h + pts_a) / max(tg, 1)

    df["form_ppg_diff"] = df["form_ppg"].fillna(0) - df["form_ppg_away"].fillna(0)
    train_dfs.append(df)

train = pd.concat(train_dfs, ignore_index=True)

# Train enhanced LR model
feats_enhanced = ["rating_diff_z", "host_advantage", "knockout", "form_ppg_diff"]
X_enh = train[feats_enhanced].to_numpy()
y_enh = train["Target"].to_numpy()

# Symmetry augmentation
X_enh_aug = np.vstack([X_enh, -X_enh])
y_enh_aug = np.concatenate([y_enh, np.where(y_enh == "H", "A", np.where(y_enh == "A", "H", y_enh))])

lr_enh = LogisticRegression(max_iter=5000, C=0.5, multi_class="multinomial")
lr_enh.fit(X_enh_aug, y_enh_aug)

# Also train baseline LR (no form_ppg)
X_base = train[["rating_diff_z", "host_advantage", "knockout"]].to_numpy()
X_base_aug = np.vstack([X_base, -X_base])
lr_base = LogisticRegression(max_iter=5000, C=0.5, multi_class="multinomial")
lr_base.fit(X_base_aug, y_enh_aug)

import json

# ===== 4. Predict Round of 32 =====
print(f"\n{'='*120}")
print(f"{'2026 1/32决赛预测: Softmax vs 增强模型(含预选赛xG)':^120}")
print(f"{'='*120}\n")

rows = []
for _, r in fixtures.sort_index().iterrows():
    mn = int(r["MatchNumber"])
    home = r["Home"]
    away = r["Away"]
    ha_val = int(home in HOSTS[2026]) - int(away in HOSTS[2026])
    feats = np.array([[fifa_26.get(home, 0) - fifa_26.get(away, 0), ha_val, 1]])

    # Softmax
    sm_probs = model.predict_proba(feats)[0]
    sm_out = {"H": sm_probs[0], "D": sm_probs[1], "A": sm_probs[2]}

    # Baseline LR (same 3 feats)
    lr_probs = lr_base.predict_proba(feats)[0]
    lr_out = dict(zip(lr_base.classes_, lr_probs))

    # Enhanced LR (+ PPG)
    ppg_diff = team_stats.get(home, {}).get("qual_ppg", 0) - team_stats.get(away, {}).get("qual_ppg", 0)
    feats_enh = np.array([[fifa_26.get(home, 0) - fifa_26.get(away, 0), ha_val, 1, ppg_diff]])
    lr_enh_probs = lr_enh.predict_proba(feats_enh)[0]
    lr_enh_out = dict(zip(lr_enh.classes_, lr_enh_probs))

    # xG-based strength indicator
    home_xg = team_stats.get(home, {}).get("qual_xg", 0)
    away_xg = team_stats.get(away, {}).get("qual_xg", 0)
    xg_pred = "H" if home_xg > away_xg * 1.1 else ("A" if away_xg > home_xg * 1.1 else "D")

    rows.append({
        "Match": mn, "Home": home, "Away": away,
        "SM_H": sm_out["H"], "SM_D": sm_out["D"], "SM_A": sm_out["A"],
        "LR_H": lr_out.get("H", 0), "LR_D": lr_out.get("D", 0), "LR_A": lr_out.get("A", 0),
        "LRx_H": lr_enh_out.get("H", 0), "LRx_D": lr_enh_out.get("D", 0), "LRx_A": lr_enh_out.get("A", 0),
        "home_xg": home_xg, "away_xg": away_xg,
        "home_ppg": team_stats.get(home, {}).get("qual_ppg", 0),
        "away_ppg": team_stats.get(away, {}).get("qual_ppg", 0),
    })

comp = pd.DataFrame(rows)

# Print comparison
print(f"{'#':>4s} {'对阵':^55s} {'Softmax':^28s} {'LR(基础)':^28s} {'LR+PPG':^28s} {'xG':^12s}")
print("-" * 160)

agree_all = 0
for _, r in comp.iterrows():
    m = int(r["Match"])
    h, a = r["Home"], r["Away"]
    match_str = f"{h:22s} vs {a:22s}"

    def pred_str(row, prefix):
        hh, dd, aa = row[f"{prefix}_H"], row[f"{prefix}_D"], row[f"{prefix}_A"]
        p = "H" if hh == max(hh, dd, aa) else ("D" if dd == max(hh, dd, aa) else "A")
        return f"H{hh:.1%} D{dd:.1%} A{aa:.1%} [{p}]"

    sm = pred_str(r, "SM")
    lr = pred_str(r, "LR")
    lrx = pred_str(r, "LRx")

    sm_pred = sm.split("[")[1].rstrip("]")
    lr_pred = lr.split("[")[1].rstrip("]")
    lrx_pred = lrx.split("[")[1].rstrip("]")

    # Determine majority vote
    votes = [sm_pred, lr_pred, lrx_pred]
    from collections import Counter
    majority = Counter(votes).most_common(1)[0][0]
    agree = majority == sm_pred == lr_pred == lrx_pred
    if sm_pred == lr_pred == lrx_pred:
        agree_all += 1

    xg_str = f"{r['home_xg']:.2f}-{r['away_xg']:.2f}"
    ppg_str = f"{r['home_ppg']:.2f}-{r['away_ppg']:.2f}"

    print(f"{m:4d} {match_str}  {sm:>28s}  {lr:>28s}  {lrx:>28s}  {xg_str:>12s}")

print(f"\n一致(三模型同向): {agree_all}/16")

# Print key stats
print(f"\n=== 各球队预选赛xG ===")
stats_df = pd.DataFrame.from_dict(team_stats, orient="index")
stats_df = stats_df[stats_df["qual_matches"] > 0].sort_values("qual_xg", ascending=False)
print(f"{'球队':25s} {'场次':>5s} {'PPG':>6s} {'xG':>6s} {'净胜球':>8s}")
for team, s in stats_df.head(10).iterrows():
    print(f"{team:25s} {int(s['qual_matches']):5d} {s['qual_ppg']:6.2f} {s['qual_xg']:6.2f} {s['qual_gd']:8.2f}")

print(f"\n=== xG最低的5支球队 ===")
for team, s in stats_df.tail(5).iterrows():
    print(f"{team:25s} {int(s['qual_matches']):5d} {s['qual_ppg']:6.2f} {s['qual_xg']:6.2f} {s['qual_gd']:8.2f}")

# Save predictions
output_path = PROJECT_ROOT / "outputs" / "world_cup" / "round32_enhanced_predictions.csv"
comp.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\nSaved to: {output_path}")
