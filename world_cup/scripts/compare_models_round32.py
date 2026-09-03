"""Compare Softmax model vs LogisticRegression on 2026 Round of 32 predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
WC_DIR = THIS_DIR.parent
PROJECT_ROOT = WC_DIR.parent
sys.path.insert(0, str(WC_DIR))

from src.world_cup_model import HOSTS, canonical_team, load_rankings
from sklearn.linear_model import LogisticRegression


RAW = Path("world_cup/data/raw")


def get_ppg(team, cutoff, ir, n=20):
    ah = ir[(ir["home_team"] == team) & (ir["date"] < cutoff)].tail(n)
    aa = ir[(ir["away_team"] == team) & (ir["date"] < cutoff)].tail(n)
    pts = []
    for _, r in ah.iterrows():
        pts.append(3 if r["home_score"] > r["away_score"] else (1 if r["home_score"] == r["away_score"] else 0))
    for _, r in aa.iterrows():
        pts.append(3 if r["away_score"] > r["home_score"] else (1 if r["away_score"] == r["home_score"] else 0))
    if not pts:
        return 0.0
    w = np.exp(np.linspace(0, 1, len(pts)))
    return float(np.average(pts, weights=w / w.sum()))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # ---- Load international results ----
    ir = pd.read_csv(RAW / "international_results.csv")
    ir["date"] = pd.to_datetime(ir["date"])
    ir["home_team"] = ir["home_team"].map(canonical_team)
    ir["away_team"] = ir["away_team"].map(canonical_team)

    rankings_path = WC_DIR / "world_cup_rankings_4_sheets_simple.xlsx"
    wc_starts = {2014: pd.Timestamp("2014-06-12"), 2018: pd.Timestamp("2018-06-14"), 2022: pd.Timestamp("2022-11-20")}

    # ---- Build training data from 2014+2018+2022 ----
    train_dfs = []
    for year in [2014, 2018, 2022]:
        if year == 2014:
            path = RAW / "international-fifa-world-cup-2014-brazil-matches-2014-to-2014-stats.csv"
        elif year == 2018:
            path = RAW / "international-fifa-world-cup-2018-russia-matches-2018-to-2018-stats.csv"
        else:
            path = RAW / "international-fifa-world-cup-2022-qatar-matches-2022-to-2022-stats.csv"

        df = pd.read_csv(path)
        df["Home"] = df["home_team_name"].map(canonical_team)
        df["Away"] = df["away_team_name"].map(canonical_team)

        rankings, _ = load_rankings(rankings_path, year)
        fifa = rankings.set_index("Country")["RatingZ"].to_dict()
        df["rating_diff_z"] = df["Home"].map(fifa) - df["Away"].map(fifa)
        df["rating_diff_z"] = df["rating_diff_z"].fillna(0)

        hosts = HOSTS[year]
        df["host_advantage"] = (df["Home"].isin(hosts)).astype(int) - (df["Away"].isin(hosts)).astype(int)
        df["knockout"] = df["Game Week"].isna().astype(int)

        cutoff = wc_starts[year]
        teams = set(df["Home"]) | set(df["Away"])
        ppg_cache = {t: get_ppg(t, cutoff, ir) for t in teams}
        df["pretourn_ppg_diff"] = df["Home"].map(ppg_cache) - df["Away"].map(ppg_cache)

        df["Target"] = np.where(
            df["home_team_goal_count"] > df["away_team_goal_count"], "H",
            np.where(df["home_team_goal_count"] == df["away_team_goal_count"], "D", "A"),
        )
        train_dfs.append(df)

    train = pd.concat(train_dfs, ignore_index=True)

    # ---- Train LR model with symmetry aug ----
    features = ["rating_diff_z", "host_advantage", "knockout"]
    X = train[features].to_numpy()
    y = train["Target"].to_numpy()

    X_aug = np.vstack([X, -X])
    y_aug = np.concatenate([y, np.where(y == "H", "A", np.where(y == "A", "H", y))])

    lr = LogisticRegression(max_iter=5000, C=0.5, multi_class="multinomial")
    lr.fit(X_aug, y_aug)

    # ---- Predict 2026 Round of 32 ----
    f26 = pd.read_csv(WC_DIR / "data" / "fixtures_2026_round32.csv")
    f26["Home"] = f26["Home"].map(canonical_team)
    f26["Away"] = f26["Away"].map(canonical_team)

    rankings_26, _ = load_rankings(rankings_path, 2026)
    fifa_26 = rankings_26.set_index("Country")["RatingZ"].to_dict()
    f26["rating_diff_z"] = f26["Home"].map(fifa_26) - f26["Away"].map(fifa_26)
    f26["rating_diff_z"] = f26["rating_diff_z"].fillna(0)
    f26["knockout"] = 1

    hosts_26 = HOSTS[2026]
    f26["host_advantage"] = (f26["Home"].isin(hosts_26)).astype(int) - (f26["Away"].isin(hosts_26)).astype(int)

    cutoff_26 = pd.Timestamp("2026-06-11")
    teams_26 = set(f26["Home"]) | set(f26["Away"])
    ppg_26 = {t: get_ppg(t, cutoff_26, ir) for t in teams_26}
    f26["pretourn_ppg_diff"] = f26["Home"].map(ppg_26) - f26["Away"].map(ppg_26)

    X_pred = f26[features].to_numpy()
    probs = lr.predict_proba(X_pred)
    for i, cls in enumerate(lr.classes_):
        f26[f"LR_{cls}"] = probs[:, i]

    # ---- Load Softmax predictions ----
    sm = pd.read_csv(PROJECT_ROOT / "outputs" / "world_cup" / "softmax_round32_predictions.csv")

    # ---- Build comparison table ----
    rows = []
    for _, r in f26.sort_index().iterrows():
        mn = int(r["MatchNumber"])
        home = r["Home"]
        away = r["Away"]

        s = sm[sm["MatchNumber"] == mn].iloc[0]
        sm_h = float(s["P_H"])
        sm_d = float(s["P_D"])
        sm_a = float(s["P_A"])
        sm_pred = "H" if sm_h == max(sm_h, sm_d, sm_a) else ("D" if sm_d == max(sm_h, sm_d, sm_a) else "A")

        lr_h = float(r["LR_H"])
        lr_d = float(r["LR_D"])
        lr_a = float(r["LR_A"])
        lr_pred = "H" if lr_h == max(lr_h, lr_d, lr_a) else ("D" if lr_d == max(lr_h, lr_d, lr_a) else "A")

        rows.append({
            "Match": mn,
            "Home": home,
            "Away": away,
            "SM_H": sm_h, "SM_D": sm_d, "SM_A": sm_a, "SM_Pred": sm_pred,
            "LR_H": lr_h, "LR_D": lr_d, "LR_A": lr_a, "LR_Pred": lr_pred,
        })

    comp = pd.DataFrame(rows)

    # ---- Print ----
    print("=" * 130)
    title = "2026 1/32决赛预测对比: Softmax模型 vs LogisticRegression"
    print(f"{title:^130}")
    print("=" * 130)
    print()

    print(f"{'#':>4s}  {'对阵':^55s}  {'Softmax':^28s}  {'LogReg':^28s}  {'一致?':>5s}")
    print("-" * 130)

    agree_count = 0
    for _, r in comp.iterrows():
        match_str = f"{r['Home']:25s} vs {r['Away']:25s}"
        sm_str = f"H{r['SM_H']:.1%} D{r['SM_D']:.1%} A{r['SM_A']:.1%} [{r['SM_Pred']}]"
        lr_str = f"H{r['LR_H']:.1%} D{r['LR_D']:.1%} A{r['LR_A']:.1%} [{r['LR_Pred']}]"
        agree = "✅" if r["SM_Pred"] == r["LR_Pred"] else "⚠️"
        if r["SM_Pred"] == r["LR_Pred"]:
            agree_count += 1
        print(f"{r['Match']:4d}  {match_str}  {sm_str:>28s}  {lr_str:>28s}  {agree}")

    print()
    print(f"预测一致: {agree_count}/16")
    print()

    # ---- Also show absolute probability differences ----
    print("各模型预测概率对比（均值）:")
    print(f"  Softmax: H={comp['SM_H'].mean():.1%} D={comp['SM_D'].mean():.1%} A={comp['SM_A'].mean():.1%}")
    print(f"  LogReg:  H={comp['LR_H'].mean():.1%} D={comp['LR_D'].mean():.1%} A={comp['LR_A'].mean():.1%}")


if __name__ == "__main__":
    main()
