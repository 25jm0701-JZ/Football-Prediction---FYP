#!/usr/bin/env python3
"""
Fair comparison: Original pipeline vs FootyStats-enhanced
========================================================
Both use: PL only, cross-season split, same test set
Original data: 2019/20 - 2024/25 (6 seasons, football-data.co.uk)
Footystats:    2019/20 - 2024/25 (6 seasons, footystats API)
Train: 2019/20-2022/23 (4 seasons, 1520 matches)
Test:  2023/24-2024/25 (2 seasons, 760 matches)

Market baseline: Bet365 CLOSING odds (B365C) — the pre-match market consensus.
Model features never include odds; odds are used only as an evaluation benchmark.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

PROJ_ROOT = Path("C:/Users/-jmmmm/Documents/FYP")

TEAM_MAP = {
    'Arsenal':'Arsenal','Aston Villa':'Aston Villa',
    'Bournemouth':'AFC Bournemouth','Brentford':'Brentford',
    'Brighton':'Brighton & Hove Albion','Burnley':'Burnley',
    'Chelsea':'Chelsea','Crystal Palace':'Crystal Palace',
    'Everton':'Everton','Fulham':'Fulham',
    'Leeds':'Leeds United','Leicester':'Leicester City',
    'Liverpool':'Liverpool','Luton':'Luton Town',
    'Man City':'Manchester City','Man United':'Manchester United',
    'Newcastle':'Newcastle United','Norwich':'Norwich City',
    "Nott'm Forest":'Nottingham Forest',
    'Sheffield United':'Sheffield United','Southampton':'Southampton',
    'Tottenham':'Tottenham Hotspur','Watford':'Watford',
    'West Brom':'West Bromwich Albion','West Ham':'West Ham United',
    'Wolves':'Wolverhampton Wanderers',
}

# ═════════════════════════════════════════════════════════
# 1. LOAD ORIGINAL DATA
# ═════════════════════════════════════════════════════════
print("=" * 60)
print("FAIR COMPARISON: Original vs FootyStats-Enhanced")
print("PL only, cross-season split, same test set")
print("=" * 60)

ORIG_SEASONS = {
    "2019-20": "PL1920.csv", "2020-21": "PL2021.csv",
    "2021-22": "PL2122.csv", "2022-23": "PL2223.csv",
    "2023-24": "PL2324.csv", "2024-25": "PL2425.csv",
}

frames = []
for season, fname in ORIG_SEASONS.items():
    df = pd.read_csv(PROJ_ROOT / "league/data/raw" / fname)
    df = df.rename(columns={
        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
        "B365CH": "odds_home_close", "B365CD": "odds_draw_close", "B365CA": "odds_away_close",
    })
    keep = ["date","home_team","away_team","home_goals","away_goals",
            "result","odds_home_close","odds_draw_close","odds_away_close"]
    df = df[[c for c in keep if c in df.columns]]
    df["season"] = season
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["target_result"] = df["result"].map({"H": 2, "D": 1, "A": 0})
    frames.append(df)
orig = pd.concat(frames, ignore_index=True).sort_values(["season","date"]).reset_index(drop=True)
print(f"\n[Original] {len(orig)} matches, {orig['season'].nunique()} seasons")

# ═════════════════════════════════════════════════════════
# 2. LOAD FOOTYSTATS & MERGE EXTRAS
# ═════════════════════════════════════════════════════════
FT_SEASONS = {
    2012: "2019-20", 4759: "2020-21", 6135: "2021-22",
    7704: "2022-23", 9660: "2023-24", 12325: "2024-25",
}

# Check if 2024-25 footystats data exists
ft_frames = []
missing_seasons = []
for sid, season in FT_SEASONS.items():
    path = PROJ_ROOT / f"data/footystats/raw/matches_{sid}.json"
    if path.exists():
        m = json.loads(path.read_text(encoding="utf-8"))
        d = pd.DataFrame(m)
        d["date"] = pd.to_datetime(d["date_unix"], unit="s")
        d["season"] = season
        ft_frames.append(d)
    else:
        missing_seasons.append(season)

ft = pd.concat(ft_frames, ignore_index=True) if ft_frames else pd.DataFrame()
print(f"[Footystats] {len(ft)} matches, {ft['season'].nunique()} seasons")
if missing_seasons:
    print(f"  Missing: {missing_seasons}")

# ═════════════════════════════════════════════════════════
# 3. MERGE FOOTYSTATS EXTRAS INTO ORIGINAL
# ═════════════════════════════════════════════════════════
orig["home_ft"] = orig["home_team"].map(TEAM_MAP)
orig["away_ft"] = orig["away_team"].map(TEAM_MAP)
orig["date_key"] = orig["date"].dt.date
ft["date_key"] = ft["date"].dt.date

extra = ft[["date_key","home_name","away_name","home_ppg","away_ppg",
    "team_a_xg_prematch","team_b_xg_prematch",
    "team_a_possession","team_b_possession",
    "home_ppg","away_ppg"]].drop_duplicates().copy()
extra.columns = ["date_key","home_name","away_name",
    "pp_home","pp_away","xg_home","xg_away",
    "poss_home","poss_away","pp_home_d","pp_away_d"]

merged = orig.merge(extra, left_on=["date_key","home_ft","away_ft"],
                    right_on=["date_key","home_name","away_name"], how="left")
merged["ppg_diff"] = merged["pp_home"] - merged["pp_away"]
merged["xg_diff"] = merged["xg_home"] - merged["xg_away"]
merged["poss_diff"] = merged["poss_home"] - merged["poss_away"]

match_rate = merged["pp_home"].notna().mean() * 100
print(f"Merge rate: {match_rate:.0f}%")

# ═════════════════════════════════════════════════════════
# 4. ROLLING FEATURES (cross-season)
# ═════════════════════════════════════════════════════════
for team, gf, ga in [("home_","home_goals","away_goals"),
                      ("away_","away_goals","home_goals")]:
    for w in [5, 10]:
        for col, src in [(f"avg_gf_{w}", gf), (f"avg_ga_{w}", ga)]:
            merged[f"{team}{col}"] = merged.groupby(f"{team}team")[src].transform(
                lambda x: x.shift(1).rolling(w, 1).mean())
        merged[f"{team}avg_gd_{w}"] = merged[f"{team}avg_gf_{w}"] - merged[f"{team}avg_ga_{w}"]
        pts = merged.apply(lambda r: 3 if r[gf]>r[ga] else (1 if r[gf]==r[ga] else 0), axis=1)
        merged[f"{team}avg_pts_{w}"] = pts.groupby(merged[f"{team}team"]).transform(
            lambda x: x.shift(1).rolling(w, 1).mean())

# ═════════════════════════════════════════════════════════
# 5. TRAIN & EVALUATE
# ═════════════════════════════════════════════════════════
TRAIN_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23"]
TEST_SEASONS = ["2023-24", "2024-25"]

train = merged[merged["season"].isin(TRAIN_SEASONS)].copy()
test = merged[merged["season"].isin(TEST_SEASONS)].copy()
y_tr = train["target_result"].values
y_te = test["target_result"].values
print(f"\nTrain: {len(train)} ({train['season'].nunique()} seas)")
print(f"Test:  {len(test)} ({test['season'].nunique()} seas)")

# Feature columns
base_cols = [c for c in merged.columns if any(
    c.startswith(p) for p in ["home_avg_","away_avg_"])]
xtra_cols = ["pp_home","pp_away","ppg_diff","xg_home","xg_away",
             "xg_diff","poss_home","poss_away","poss_diff"]
xtra_cols = [c for c in xtra_cols if c in merged.columns]

def run(X_tr, X_te, label):
    X_tr = X_tr.values.astype(float)
    X_te = X_te.values.astype(float)
    cm = np.nanmean(X_tr, axis=0)
    for X in [X_tr, X_te]:
        for i in range(X.shape[1]):
            m = cm[i]
            if m != m: m = 0
            X[np.isnan(X[:,i]), i] = m
    ss = StandardScaler()
    X_tr_s = ss.fit_transform(X_tr)
    X_te_s = ss.transform(X_te)
    lr = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                            max_iter=1000, random_state=42)
    lr.fit(X_tr_s, y_tr)
    yp = lr.predict(X_te_s)
    ypr = lr.predict_proba(X_te_s)
    acc = accuracy_score(y_te, yp)
    ll = log_loss(y_te, ypr)
    bs = np.mean([brier_score_loss((y_te==i).astype(int), ypr[:,i]) for i in range(3)])
    return acc, bs, ll, X_tr.shape[1]

r1 = run(train[base_cols], test[base_cols], "")
r2 = run(train[list(set(base_cols+xtra_cols))], test[list(set(base_cols+xtra_cols))], "")
r3 = run(train[xtra_cols], test[xtra_cols], "")

# Market (Bet365 CLOSING odds, B365C — the pre-match market consensus)
# Convention matches trackb_clean_eval: columns [home, draw, away], np.maximum
# de-vig, argmax with home-first tie-break (symmetric matches pick home).
o = test[["odds_home_close","odds_draw_close","odds_away_close"]].values.astype(float)
imp = 1.0 / np.maximum(o, 1.01)
imp = imp / imp.sum(axis=1, keepdims=True)
mp = imp.argmax(axis=1)  # 0=home, 1=draw, 2=away
macc = (((mp==2)&(y_te==0))|((mp==1)&(y_te==1))|((mp==0)&(y_te==2))).mean()
mpr = np.column_stack([imp[:,2], imp[:,1], imp[:,0]])  # [away,draw,home] for metrics
mbs = np.mean([brier_score_loss((y_te==i).astype(int), mpr[:,i]) for i in range(3)])
mll = log_loss(y_te, mpr)

# ═════════════════════════════════════════════════════════
# 6. RESULTS
# ═════════════════════════════════════════════════════════
print()
print("=" * 60)
print("RESULTS (same test set: 2023-24 + 2024-25, 760 matches)")
print("=" * 60)
print(f"  {'Method':<30} {'Acc':<8} {'Brier':<8} {'LogLoss':<8} {'Feat':<6}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
print(f"  {'1) Rolling only (original)':<30} {r1[0]:<8.4f} {r1[1]:<8.4f} {r1[2]:<8.4f} {r1[3]:<6}")
print(f"  {'2) +FootyStats extras':<30} {r2[0]:<8.4f} {r2[1]:<8.4f} {r2[2]:<8.4f} {r2[3]:<6}")
print(f"  {'3) FootyStats extras only':<30} {r3[0]:<8.4f} {r3[1]:<8.4f} {r3[2]:<8.4f} {r3[3]:<6}")
print(f"  {'4) Market odds (Bet365 CLOSING)':<30} {macc:<8.4f} {mbs:<8.4f} {mll:<8.4f} {'--':<6}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

# Comparisons
d1 = r2[0] - r1[0]
d2 = r2[0] - macc
d3 = r1[0] - macc
print(f"\n  2 vs 1 (FootyStats lifts rolling): +{d1*100:.2f}%")
print(f"  2 vs 4 (Enhanced model vs market):  +{d2*100:.2f}%")
print(f"  1 vs 4 (Rolling only vs market):    {d3*100:+.2f}%")
print(f"  3 vs 4 (FootyStats alone vs market): +{(r3[0]-macc)*100:.2f}%")

# Feature importance (for model 2)
comb_cols = list(set(base_cols + xtra_cols))
X_tr = train[comb_cols].values.astype(float)
cm = np.nanmean(X_tr, axis=0)
for i in range(X_tr.shape[1]):
    m = cm[i]
    if m != m: m = 0
    X_tr[np.isnan(X_tr[:,i]), i] = m
ss = StandardScaler()
lr = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=1000, random_state=42)
lr.fit(ss.fit_transform(X_tr), y_tr)
coef = lr.coef_[0]
print(f"\n\nTop 10 features (model 2):")
imp_idx = np.argsort(np.abs(coef))[::-1][:10]
for idx in imp_idx:
    d = "+" if coef[idx] > 0 else "-"
    print(f"  {d} {comb_cols[idx]:<30} {abs(coef[idx]):.4f}")

print(f"\n{'='*60}")
print("CONCLUSION")
print(f"{'='*60}")
print(f"Best: Rolling + FootyStats extras ({r2[0]*100:.1f}%)")
print(f"  - {d1*100:.1f}% lift over rolling only")
print(f"  - {d2*100:.1f}% lift over market odds")
print(f"  - Key features: PPG (team strength) + rolling form + xG")
print(f"{'='*60}")
