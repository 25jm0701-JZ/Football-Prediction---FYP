#!/usr/bin/env python3
"""
Classic Poisson football model (Maher/Dixon-Coles style).
Uses team-specific attack/defense parameters + FootyStats PPG.

Train: 2019-20 to 2022-23 (4 seasons, 1520 matches)
Test:  2023-24 to 2024-25 (2 seasons, 760 matches)
Both have full FootyStats coverage.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

PROJ_ROOT = Path("C:/Users/-jmmmm/Documents/FYP")
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(PROJ_ROOT / "league"))

# ── Load data ──────────────────────────────────────────────────────
TEAM_MAP = {"Arsenal":"Arsenal","Aston Villa":"Aston Villa",
    "Bournemouth":"AFC Bournemouth","Brentford":"Brentford",
    "Brighton":"Brighton & Hove Albion","Burnley":"Burnley",
    "Chelsea":"Chelsea","Crystal Palace":"Crystal Palace","Everton":"Everton",
    "Fulham":"Fulham","Leeds":"Leeds United","Leicester":"Leicester City",
    "Liverpool":"Liverpool","Luton":"Luton Town","Man City":"Manchester City",
    "Man United":"Manchester United","Newcastle":"Newcastle United",
    "Norwich":"Norwich City","Nott'm Forest":"Nottingham Forest",
    "Sheffield United":"Sheffield United","Southampton":"Southampton",
    "Tottenham":"Tottenham Hotspur","Watford":"Watford",
    "West Brom":"West Bromwich Albion","West Ham":"West Ham United",
    "Wolves":"Wolverhampton Wanderers"}

ORIG = {"2019-20":"PL1920.csv","2020-21":"PL2021.csv","2021-22":"PL2122.csv",
        "2022-23":"PL2223.csv","2023-24":"PL2324.csv","2024-25":"PL2425.csv"}
FT_SEAS = {2012:"2019-20",4759:"2020-21",6135:"2021-22",7704:"2022-23",
           9660:"2023-24",12325:"2024-25"}

frames = []
for season, fname in ORIG.items():
    df = pd.read_csv(PROJ_ROOT / "league/data/raw" / fname)
    df = df.rename(columns={"Date":"date","HomeTeam":"home_team","AwayTeam":"away_team",
        "FTHG":"home_goals","FTAG":"away_goals","FTR":"result",
        "B365H":"odds_h","B365D":"odds_d","B365A":"odds_a"})
    keep = ["date","home_team","away_team","home_goals","away_goals","result",
            "odds_h","odds_d","odds_a"]
    df = df[[c for c in keep if c in df.columns]]
    df["season"] = season; df["target_result"] = df["result"].map({"H":2,"D":1,"A":0})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    frames.append(df)
all_df = pd.concat(frames, ignore_index=True).sort_values(["season","date"]).reset_index(drop=True)

# Merge footystats
ft_frames = []
for sid, season in FT_SEAS.items():
    path = PROJ_ROOT / f"data/footystats/raw/matches_{sid}.json"
    if path.exists():
        m = json.loads(path.read_text(encoding="utf-8"))
        d = pd.DataFrame(m); d["date"] = pd.to_datetime(d["date_unix"], unit="s"); d["season"]=season
        ft_frames.append(d)
ft = pd.concat(ft_frames, ignore_index=True) if ft_frames else pd.DataFrame()

all_df["_hf"] = all_df["home_team"].map(TEAM_MAP)
all_df["_af"] = all_df["away_team"].map(TEAM_MAP)
all_df["_dk"] = all_df["date"].dt.date
ft["_dk"] = ft["date"].dt.date
fte = ft[["_dk","home_name","away_name","home_ppg","away_ppg"]].drop_duplicates()
all_df = all_df.merge(fte, left_on=["_dk","_hf","_af"], right_on=["_dk","home_name","away_name"], how="left")

# ── Rolling features ──────────────────────────────────────────────
for team, gf, ga in [("home_","home_goals","away_goals"),("away_","away_goals","home_goals")]:
    for w in [5,10]:
        all_df[f"{team}avg_gf_{w}"] = all_df.groupby(f"{team}team")[gf].transform(
            lambda x: x.shift(1).rolling(w,1).mean())

# ── Split ──────────────────────────────────────────────────────────
train = all_df[all_df["season"].isin(["2019-20","2020-21","2021-22","2022-23"])].copy()
test = all_df[all_df["season"].isin(["2023-24","2024-25"])].copy()
y_tr = train["target_result"].values; y_te = test["target_result"].values

print("=" * 60)
print("CLASSIC POISSON FOOTBALL MODEL")
print("=" * 60)
print(f"Train: {len(train)} matches (4 seasons)")
print(f"Test:  {len(test)} matches ({test['season'].nunique()} seasons)")

# ══════════════════════════════════════════════════════════════════
# METHOD 1: WEIGHTED ATTACK/DEFENSE RATINGS (roll across matches)
# ══════════════════════════════════════════════════════════════════
# For each team-season, compute attack/defense strength relative to league avg
# attack = team_goals_per_game / league_goals_per_game
# defense = team_goals_conceded_per_game / league_goals_per_game

print("\n=== Method 1: Relative Attack/Defense Strength ===\n")

# Per-season attack/defense ratings
for season in train["season"].unique():
    mask_s = train["season"] == season
    # League average goals per match
    league_avg_gf = train.loc[mask_s, "home_goals"].mean()  # per team per game
    league_avg_ga = train.loc[mask_s, "away_goals"].mean()

    for team in train[mask_s]["home_team"].unique():
        mask_h = mask_s & (train["home_team"] == team)
        mask_a = mask_s & (train["away_team"] == team)
        team_gf = train.loc[mask_h, "home_goals"].sum() + train.loc[mask_a, "away_goals"].sum()
        team_ga = train.loc[mask_h, "away_goals"].sum() + train.loc[mask_a, "home_goals"].sum()
        n_h = mask_h.sum(); n_a = mask_a.sum(); n = n_h + n_a

        if n > 0 and league_avg_gf > 0:
            attack = (team_gf / n) / league_avg_gf  # >1 means better than avg
            defense = (team_ga / n) / league_avg_ga  # <1 means better than avg
        else:
            attack, defense = 1.0, 1.0

        train.loc[mask_h | mask_a, "attack"] = attack
        train.loc[mask_h | mask_a, "defense"] = defense

# For test, use last known ratings (from training)
# Map team -> last attack/defense in training
last_ratings = train.groupby("home_team").agg({"attack":"last","defense":"last"}).to_dict("index")
for team, ratings in last_ratings.items():
    mask_h = test["home_team"] == team
    mask_a = test["away_team"] == team
    test.loc[mask_h, "attack"] = ratings["attack"]
    test.loc[mask_a, "attack"] = ratings["attack"]
    test.loc[mask_h, "defense"] = ratings["defense"]
    test.loc[mask_a, "defense"] = ratings["defense"]

# Fill any missing with 1.0
for df in [train, test]:
    df["attack"] = df["attack"].fillna(1.0)
    df["defense"] = df["defense"].fillna(1.0)

# Poisson prediction using attack/defense
train_avg_h = train["home_goals"].mean()
train_avg_a = train["away_goals"].mean()

def poisson_ad_predict(df, avg_h, avg_a):
    """Predict using Attack/Defense ratings."""
    lambda_h = avg_h * df["attack"].values * df["defense"].values  # home atk * away def
    lambda_a = avg_a * df["attack"].values * df["defense"].values  # away atk * home def
    # Wait - home goals depend on home team's attack * away team's defense
    # but I stored attack/defense per team, need to get the right pairing
    return lambda_h, lambda_a

# Actually need to map attack/defense correctly
# For home goals: home_attack * away_defense
# For away goals: away_attack * home_defense

# Build team rating lookup
team_ratings = {}
for team in all_df["home_team"].unique():
    tm = train[train["home_team"] == team]
    if len(tm) > 0:
        atk = tm["attack"].iloc[-1] if "attack" in tm.columns else 1.0
        pdf = tm["defense"].iloc[-1] if "defense" in tm.columns else 1.0
    else:
        atk, pdf = 1.0, 1.0
    team_ratings[team] = {"attack": atk, "defense": pdf}

poi_ad_prob = np.zeros((len(test), 3))
poi_ad_pred = np.zeros(len(test), dtype=int)
lam_h_ad = np.zeros(len(test))
lam_a_ad = np.zeros(len(test))

for i, row in test.iterrows():
    idx = test.index.get_loc(i)
    h_team, a_team = row["home_team"], row["away_team"]

    lh = train_avg_h * team_ratings[h_team]["attack"] * team_ratings[a_team]["defense"]
    la = train_avg_a * team_ratings[a_team]["attack"] * team_ratings[h_team]["defense"]
    lam_h_ad[idx] = lh; lam_a_ad[idx] = la

    mg = 8
    pmf_h = np.array([poisson.pmf(g, lh) for g in range(mg+1)])
    pmf_a = np.array([poisson.pmf(g, la) for g in range(mg+1)])
    ph = sum(pmf_h[hi] * pmf_a[ai] for hi in range(1, mg+1) for ai in range(hi))
    pd_d = np.sum(pmf_h * pmf_a)
    pa = 1 - ph - pd_d
    poi_ad_prob[idx] = [pa, pd_d, ph]
    poi_ad_pred[idx] = np.argmax([pa, pd_d, ph])

poi_ad_acc = accuracy_score(y_te, poi_ad_pred)
poi_ad_ll = log_loss(y_te, poi_ad_prob)
poi_ad_bs = np.mean([brier_score_loss((y_te==i).astype(int), poi_ad_prob[:,i]) for i in range(3)])

print(f"Attack/Def Poisson:")
print(f"  Accuracy: {poi_ad_acc:.4f}  Brier: {poi_ad_bs:.4f}  LogLoss: {poi_ad_ll:.4f}")

# ══════════════════════════════════════════════════════════════════
# METHOD 2: POISSON FEATURES (avg_goals_for rolling)
# ══════════════════════════════════════════════════════════════════
print("\n=== Method 2: Poisson GLM with rolling features ===\n")

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

poi2_cols = ["home_avg_gf_5","home_avg_gf_10","away_avg_gf_5","away_avg_gf_10",
             "home_ppg","away_ppg"]
poi2_cols = [c for c in poi2_cols if c in train.columns]

def fit_poisson_glm(X, y):
    X = X.astype(float).copy()
    cm = np.nanmean(X, axis=0)
    for i in range(X.shape[1]):
        m = cm[i]; m = 0 if m != m else m
        X[np.isnan(X[:,i]), i] = m
    ss = StandardScaler(); X_s = ss.fit_transform(X)
    X_s = sm.add_constant(X_s)
    model = sm.GLM(y, X_s, family=sm.families.Poisson()).fit(disp=0)
    return model, ss

X_tr = train[poi2_cols]; X_te = test[poi2_cols]
pm_h, pss_h = fit_poisson_glm(X_tr.values, train["home_goals"].values)
pm_a, pss_a = fit_poisson_glm(X_tr.values, train["away_goals"].values)

X_te_f = X_te.values.astype(float).copy()
cm = np.nanmean(X_te_f, axis=0)
for i in range(X_te_f.shape[1]):
    m = cm[i]; m = 0 if m != m else m
    X_te_f[np.isnan(X_te_f[:,i]), i] = m
X_te_s = sm.add_constant(pss_h.transform(X_te_f))

lam_h2 = pm_h.predict(X_te_s)
lam_a2 = pm_a.predict(X_te_s)

poi2_prob = np.zeros((len(test), 3))
poi2_pred = np.zeros(len(test), dtype=int)
for i in range(len(test)):
    lh = lam_h2.iloc[i] if hasattr(lam_h2, 'iloc') else lam_h2[i]
    la = lam_a2.iloc[i] if hasattr(lam_a2, 'iloc') else lam_a2[i]
    mg = 8
    pmf_h = np.array([poisson.pmf(g, lh) for g in range(mg+1)])
    pmf_a = np.array([poisson.pmf(g, la) for g in range(mg+1)])
    ph = sum(pmf_h[hi] * pmf_a[ai] for hi in range(1, mg+1) for ai in range(hi))
    pd_d = np.sum(pmf_h * pmf_a)
    pa = 1 - ph - pd_d
    poi2_prob[i] = [pa, pd_d, ph]
    poi2_pred[i] = np.argmax([pa, pd_d, ph])

poi2_acc = accuracy_score(y_te, poi2_pred)
poi2_ll = log_loss(y_te, poi2_prob)
poi2_bs = np.mean([brier_score_loss((y_te==i).astype(int), poi2_prob[:,i]) for i in range(3)])
print(f"GLM Poisson (rolling):")
print(f"  Accuracy: {poi2_acc:.4f}  Brier: {poi2_bs:.4f}  LogLoss: {poi2_ll:.4f}")

# ══════════════════════════════════════════════════════════════════
# LR for comparison (same features)
# ══════════════════════════════════════════════════════════════════
print("\n=== LR for comparison ===\n")
lr_cols = list(set(poi2_cols + ["home_avg_gf_5","home_avg_gf_10"]))
X_tr_lr = train[lr_cols].values.astype(float); X_te_lr = test[lr_cols].values.astype(float)
cm = np.nanmean(X_tr_lr, axis=0)
for X in [X_tr_lr, X_te_lr]:
    for i in range(X.shape[1]):
        m = cm[i]; m = 0 if m != m else m
        X[np.isnan(X[:,i]), i] = m
ss = StandardScaler()
lr = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=2000, random_state=42)
lr.fit(ss.fit_transform(X_tr_lr), y_tr)
lr_prob = lr.predict_proba(ss.transform(X_te_lr))
lr_acc = accuracy_score(y_te, lr.predict(ss.transform(X_te_lr)))
lr_ll = log_loss(y_te, lr_prob)
lr_bs = np.mean([brier_score_loss((y_te==i).astype(int), lr_prob[:,i]) for i in range(3)])
print(f"LR (direct result):")
print(f"  Accuracy: {lr_acc:.4f}  Brier: {lr_bs:.4f}  LogLoss: {lr_ll:.4f}")

# Market
mpr = np.column_stack([1.0/test["odds_a"].replace(0,np.nan),1.0/test["odds_d"].replace(0,np.nan),1.0/test["odds_h"].replace(0,np.nan)])
mpr = mpr / mpr.sum(axis=1, keepdims=True)
macc = accuracy_score(y_te, np.argmax(mpr, axis=1)); mll = log_loss(y_te, mpr)
mbs = np.mean([brier_score_loss((y_te==i).astype(int), mpr[:,i]) for i in range(3)])

# O/U 2.5 for Poisson
te_ou = (test["home_goals"].values + test["away_goals"].values > 2.5).astype(int)
def poi_ou(lam_h, lam_a):
    probs = np.zeros(len(lam_h))
    for i in range(len(lam_h)):
        lh = lam_h[i] if not (hasattr(lam_h,'iloc') and isinstance(lam_h.iloc[i], float)) else lam_h
        la = lam_a[i]
        if hasattr(lh, 'iloc'): lh = lh.iloc[0]; la = la.iloc[0]
        p = 0
        for hi in range(9):
            for ai in range(9):
                if hi+ai > 2.5: p += poisson.pmf(hi,lh) * poisson.pmf(ai,la)
        probs[i] = p
    return probs

# ══════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print(f"Test: {len(test)} matches (2023-24 + 2024-25)")
print("=" * 60)
print(f"  {'Model':<30} {'Accuracy':<10} {'Brier':<10} {'LogLoss':<10}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
print(f"  {'Poisson A/D ratings':<30} {poi_ad_acc:<10.4f} {poi_ad_bs:<10.4f} {poi_ad_ll:<10.4f}")
print(f"  {'Poisson GLM (rolling)':<30} {poi2_acc:<10.4f} {poi2_bs:<10.4f} {poi2_ll:<10.4f}")
print(f"  {'LR (direct result)':<30} {lr_acc:<10.4f} {lr_bs:<10.4f} {lr_ll:<10.4f}")
print(f"  {'Market odds':<30} {macc:<10.4f} {mbs:<10.4f} {mll:<10.4f}")
print()

# Who wins
print(f"  Best model: ", end="")
scores = [("Poisson A/D", poi_ad_acc), ("Poisson GLM", poi2_acc), ("LR", lr_acc), ("Market", macc)]
scores.sort(key=lambda x: -x[1])
print(f"{scores[0][0]} ({scores[0][1]*100:.1f}%)")

# Feature importance from Poisson GLM
print(f"\n\nPoisson GLM coefficients (home goals):")
coef_names = ["const"] + poi2_cols
for i, (name, coef) in enumerate(zip(coef_names, pm_h.params)):
    print(f"  {name:<25} {coef:.4f}")

print(f"\nPoisson GLM coefficients (away goals):")
for i, (name, coef) in enumerate(zip(coef_names, pm_a.params)):
    print(f"  {name:<25} {coef:.4f}")
