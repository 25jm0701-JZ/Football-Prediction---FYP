"""
2026/27 PL GW1 Predictions - Fixed feature extraction
Each team's home/away stats computed directly from their last N matches
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "league")
import numpy as np
import pandas as pd
from pathlib import Path
from src.data_loader import load_data
from src.features import build_features, get_feature_columns
from src.models import fill_features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DIV_LEVEL = {"E0": 0, "E1": 1, "E2": 2, "E3": 3}

# ── 1. Load all English 4-league data ──
print("Loading English 4-league data...")
all_frames = []
for div, level in DIV_LEVEL.items():
    files = sorted(Path("league/data/english_raw").glob(f"{div}*.csv"))
    for f in files:
        df = load_data(f)
        df["league_level"] = level
        df["division"] = div
        all_frames.append(df)

hist = pd.concat(all_frames, ignore_index=True)
hist = hist.sort_values("date").reset_index(drop=True)
print(f"  Total: {len(hist)} matches")

# ── 2. Build features & train model ──
print("Training model...")
hist_feat = build_features(hist, windows=[5,10], include_odds=False, include_shots=True,
                           include_h2h=True, include_footystats=False)
feat_cols = get_feature_columns(hist_feat)
if "league_level" not in hist_feat.columns:
    hist_feat["league_level"] = hist["league_level"].values
feat_cols = list(set(feat_cols + ["league_level"]))

X = fill_features(hist_feat[feat_cols].values)
y = hist_feat["target_result"].values.astype(int)
lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=4000, C=1.0, random_state=42)
lr.fit(X, y)
print(f"  Accuracy: {accuracy_score(y, lr.predict(X))*100:.2f}%")

# ── 3. Compute PL averages for imputation ──
pl_data = hist[hist["division"] == "E0"]
print(f"  PL matches in training: {len(pl_data)}")

# Compute per-team PL averages for home/away stats
def team_avg_pl(team, perspective="home"):
    """Get a team's average stats from PL matches only"""
    if perspective == "home":
        m = pl_data[pl_data["home_team"] == team]
        if len(m) == 0: return {}
        return {
            "avg_goals_for": m["home_goals_full"].mean(),
            "avg_goals_against": m["away_goals_full"].mean(),
            "avg_goal_diff": (m["home_goals_full"] - m["away_goals_full"]).mean(),
            "avg_points": m.apply(lambda r: 3 if r["home_goals_full"]>r["away_goals_full"] else(1 if r["home_goals_full"]==r["away_goals_full"] else 0), axis=1).mean(),
            "win_rate": (m["home_goals_full"] > m["away_goals_full"]).mean(),
            "draw_rate": (m["home_goals_full"] == m["away_goals_full"]).mean(),
            "loss_rate": (m["home_goals_full"] < m["away_goals_full"]).mean(),
        }
    else:
        m = pl_data[pl_data["away_team"] == team]
        if len(m) == 0: return {}
        return {
            "avg_goals_for": m["away_goals_full"].mean(),
            "avg_goals_against": m["home_goals_full"].mean(),
            "avg_goal_diff": (m["away_goals_full"] - m["home_goals_full"]).mean(),
            "avg_points": m.apply(lambda r: 3 if r["away_goals_full"]>r["home_goals_full"] else(1 if r["away_goals_full"]==r["home_goals_full"] else 0), axis=1).mean(),
            "win_rate": (m["away_goals_full"] > m["home_goals_full"]).mean(),
            "draw_rate": (m["away_goals_full"] == m["home_goals_full"]).mean(),
            "loss_rate": (m["away_goals_full"] < m["home_goals_full"]).mean(),
        }

# ── 4. Compute current form for each team from 2025/26 PL data ──
print("\nComputing team form from 2025/26 PL...")
ls = load_data(Path("league/data/raw/PL2526.csv"))
ls = ls.sort_values("date").reset_index(drop=True)

def team_recent_form(team, n=5, perspective="home"):
    """Compute a team's recent stats from last N matches in given perspective"""
    if perspective == "home":
        m = ls[ls["home_team"] == team].tail(n)
        if len(m) == 0:
            return team_avg_pl(team, "home")
        return {
            "avg_goals_for": m["home_goals_full"].mean(),
            "avg_goals_against": m["away_goals_full"].mean(),
            "avg_goal_diff": (m["home_goals_full"] - m["away_goals_full"]).mean(),
            "avg_points": m.apply(lambda r: 3 if r["home_goals_full"]>r["away_goals_full"] else(1 if r["home_goals_full"]==r["away_goals_full"] else 0), axis=1).mean(),
            "win_rate": (m["home_goals_full"] > m["away_goals_full"]).mean(),
            "draw_rate": (m["home_goals_full"] == m["away_goals_full"]).mean(),
            "loss_rate": (m["home_goals_full"] < m["away_goals_full"]).mean(),
        }
    else:
        m = ls[ls["away_team"] == team].tail(n)
        if len(m) == 0:
            return team_avg_pl(team, "away")
        return {
            "avg_goals_for": m["away_goals_full"].mean(),
            "avg_goals_against": m["home_goals_full"].mean(),
            "avg_goal_diff": (m["away_goals_full"] - m["home_goals_full"]).mean(),
            "avg_points": m.apply(lambda r: 3 if r["away_goals_full"]>r["home_goals_full"] else(1 if r["away_goals_full"]==r["home_goals_full"] else 0), axis=1).mean(),
            "win_rate": (m["away_goals_full"] > m["home_goals_full"]).mean(),
            "draw_rate": (m["away_goals_full"] == m["home_goals_full"]).mean(),
            "loss_rate": (m["away_goals_full"] < m["home_goals_full"]).mean(),
        }

# ── 5. Build feature vector for any match ──
# feat_cols expected format: home_avg_goals_for_5, home_avg_goals_for_10,
#   away_avg_goals_for_5, away_avg_goals_for_10, etc.

# Map feature names to (perspective, metric, window)
FEAT_MAP = {}
for c in feat_cols:
    if c.startswith("home_") and c != "league_level":
        suffix = c[5:]  # remove "home_"
        FEAT_MAP[c] = ("home", suffix)
    elif c.startswith("away_") and c != "league_level":
        suffix = c[5:]  # remove "away_"
        FEAT_MAP[c] = ("away", suffix)

# Metric names mapping
METRIC_MAP = {
    "avg_goals_for_5": "avg_goals_for",
    "avg_goals_against_5": "avg_goals_against",
    "avg_goal_diff_5": "avg_goal_diff",
    "avg_points_5": "avg_points",
    "win_rate_5": "win_rate",
    "draw_rate_5": "draw_rate",
    "loss_rate_5": "loss_rate",
    "avg_goals_for_10": "avg_goals_for",
    "avg_goals_against_10": "avg_goals_against",
    "avg_goal_diff_10": "avg_goal_diff",
    "avg_points_10": "avg_points",
    "win_rate_10": "win_rate",
    "draw_rate_10": "draw_rate",
    "loss_rate_10": "loss_rate",
}

TEAM_MAP = {
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa",
    "AFC Bournemouth": "Bournemouth", "Brentford": "Brentford",
    "Brighton & Hove Albion": "Brighton", "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace", "Everton": "Everton",
    "Fulham": "Fulham", "Leeds United": "Leeds",
    "Liverpool": "Liverpool", "Manchester City": "Man City",
    "Manchester United": "Man United", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Sunderland": "Sunderland",
}

PROMOTED = {"Coventry City", "Hull City", "Ipswich Town"}

def build_feats(home_team, away_team):
    """Build feature vector for a match using team recent form"""
    h_key = TEAM_MAP.get(home_team, home_team)
    a_key = TEAM_MAP.get(away_team, away_team)

    vec = {}
    for c in feat_cols:
        if c == "league_level":
            vec[c] = 0.0  # PL match
            continue

        perspective, suffix = FEAT_MAP.get(c, (None, None))
        if perspective is None:
            vec[c] = 0.0
            continue

        # Determine window (5 or 10) from suffix
        window = 5 if "_5" in suffix else 10
        metric = METRIC_MAP.get(suffix, None)
        if metric is None:
            vec[c] = 0.0
            continue

        # Get team stats
        if perspective == "home":
            team = h_key
            if team in PROMOTED:
                stats = team_avg_pl(team, "home")
            else:
                stats = team_recent_form(team, window, "home")
        else:
            team = a_key
            if team in PROMOTED:
                stats = team_avg_pl(team, "away")
            else:
                stats = team_recent_form(team, window, "away")

        vec[c] = stats.get(metric, 0.0)

    return np.array([[vec[c] for c in feat_cols]], dtype=float)

# Also compute training means for any NaN imputation
train_means = {c: np.nanmean(X[:, i]) for i, c in enumerate(feat_cols)}

# ── 6. GW1 Fixtures ──
FIXTURES = [
    ("Arsenal", "Coventry City"),
    ("Hull City", "Manchester United"),
    ("Everton", "Crystal Palace"),
    ("Ipswich Town", "Sunderland"),
    ("Nottingham Forest", "Leeds United"),
    ("Brentford", "Tottenham Hotspur"),
    ("Brighton & Hove Albion", "Aston Villa"),
    ("Manchester City", "AFC Bournemouth"),
    ("Newcastle United", "Liverpool"),
    ("Fulham", "Chelsea"),
]

# Debug: check Man United's actual form
print("\nMan United 2025/26 end-of-season form:")
mu_home5 = team_recent_form("Man United", 5, "home")
mu_away5 = team_recent_form("Man United", 5, "away")
print(f"  Last 5 HOME: pts={mu_home5.get('avg_points',0):.2f}/g, gf={mu_home5.get('avg_goals_for',0):.2f}, ga={mu_home5.get('avg_goals_against',0):.2f}")
print(f"  Last 5 AWAY: pts={mu_away5.get('avg_points',0):.2f}/g, gf={mu_away5.get('avg_goals_for',0):.2f}, ga={mu_away5.get('avg_goals_against',0):.2f}")

# Debug: check PL averages
print("\nPL average home stats:")
pl_avg_h = team_avg_pl("Arsenal", "home")  # just use Arsenal as a template
for k, v in pl_avg_h.items():
    print(f"  {k}: {v:.3f}")

# ── 7. Predict ──
CLASS = ["Away", "Draw", "Home"]
print()
print("=" * 70)
print("  2026/27 PREMIER LEAGUE - GW1 PREDICTIONS (Fixed)")
print("  Model: LR on English 4 leagues (E0-E3, 2019-2026)")
print("  Promoted: PL-mean imputed")
print("=" * 70)

results = []
for home, away in FIXTURES:
    X_pred = build_feats(home, away)
    # Fill NaN with training means
    for j in range(X_pred.shape[1]):
        if np.isnan(X_pred[0, j]):
            X_pred[0, j] = train_means.get(feat_cols[j], 0.0)

    prob = lr.predict_proba(X_pred)[0]
    outcomes = [(prob[2], f"{home} win"), (prob[1], "Draw"), (prob[0], f"{away} win")]
    outcomes.sort(key=lambda x: -x[0])

    h_flag = " [P]" if home in PROMOTED else ""
    a_flag = " [P]" if away in PROMOTED else ""

    results.append({
        "Home": home, "Away": away,
        "H%": round(prob[2]*100, 1),
        "D%": round(prob[1]*100, 1),
        "A%": round(prob[0]*100, 1),
        "Prediction": CLASS[np.argmax(prob)],
    })

    print(f"\n  {home:25s}{h_flag} vs {away:25s}{a_flag}")
    print(f"  -> {outcomes[0][1]:30s}  ({outcomes[0][0]*100:.0f}%)")
    print(f"     H: {prob[2]*100:.0f}%  D: {prob[1]*100:.0f}%  A: {prob[0]*100:.0f}%")

# ── 8. Save ──
out_dir = Path("outputs/league")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "predictions_2627_gw1_fixed.csv"
pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n  Saved: {out_path}")
