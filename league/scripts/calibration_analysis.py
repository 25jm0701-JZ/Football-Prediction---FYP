"""
Calibration analysis for the 68.86% LR + FootyStats model.
"""
import sys
from pathlib import Path
sys.path.insert(0, "league")

import numpy as np
from src.data_loader import load_multiple
from src.models import temporal_split, fill_features
from src.features import build_features, get_feature_columns
from src.evaluate import classification_metrics, plot_calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _season(d):
    y = d.year
    return f"{y}-{str(y+1)[-2:]}" if d.month >= 8 else f"{y-1}-{str(y)[-2:]}"

# Load PL 6 seasons
files = [Path("league/data/raw") / f"PL{str(y)[-2:]}{str(y+1)[-2:]}.csv" for y in range(2019, 2025)]
df = load_multiple(files)
df["season"] = df["date"].apply(_season)
df = df.sort_values("date").reset_index(drop=True)

# Build features with FootyStats
df_feat = build_features(df, windows=[5, 10],
    include_odds=True, include_shots=True, include_h2h=True,
    include_footystats=True, footystats_season_col="season")
feat_cols = get_feature_columns(df_feat)

# Temporal split
train_df, test_df = temporal_split(df_feat, test_ratio=0.2)
X_train = fill_features(train_df[feat_cols].values.astype(float))
y_train = train_df["target_result"].values.astype(int)
X_test = fill_features(test_df[feat_cols].values.astype(float))
y_test = test_df["target_result"].values.astype(int)

# Train LR
lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=4000, C=1.0, random_state=42)
lr.fit(X_train, y_train)
y_prob = lr.predict_proba(X_test)
y_pred = np.argmax(y_prob, axis=1)

# Metrics
met = classification_metrics(y_test, y_pred, y_prob)
print(f"Accuracy:  {met['accuracy']:.4f}")
print(f"LogLoss:   {met['log_loss']:.4f}")
print(f"Brier:     {met['brier_mean']:.4f}")
print(f"Brier per class: {met['brier_per_class']}")
print(f"ROC AUC:   {met['roc_auc_macro']:.4f}")

# Plot calibration curve (saved to file)
output_dir = Path("outputs/league")
output_dir.mkdir(parents=True, exist_ok=True)
fig_path = output_dir / "calibration_lr_footystats.png"
fig = plot_calibration_curve(y_test, y_prob,
    class_names=["Away", "Draw", "Home"],
    save_path=fig_path)
plt.close(fig)
print(f"\nCalibration plot saved: {fig_path.resolve()}")

# Also compute per-class calibration stats (numeric)
print(f"\n{'='*60}")
print(f"  Per-class Calibration")
print(f"{'='*60}")
for i, name in enumerate(["Away", "Draw", "Home"]):
    y_bin = (y_test == i).astype(int)
    prob_pred, prob_true = calibration_curve(y_bin, y_prob[:, i], n_bins=5)
    print(f"\n  {name}:")
    print(f"    {'Predicted':>10} {'Actual':>8} {'Gap':>8}")
    for pp, pt in zip(prob_pred, prob_true):
        print(f"    {pp:>10.3f} {pt:>8.3f} {pp-pt:>+8.3f}")

# Compare with betting odds calibration (if available)
if all(c in df_feat.columns for c in ["impl_home_prob", "impl_draw_prob", "impl_away_prob"]):
    odds_probs = test_df[["impl_home_prob", "impl_draw_prob", "impl_away_prob"]].values.astype(float)
    from sklearn.metrics import brier_score_loss
    print(f"\n{'='*60}")
    print(f"  LR vs Market Odds: Brier Score")
    print(f"{'='*60}")
    for i, name in enumerate(["Away", "Draw", "Home"]):
        y_bin = (y_test == i).astype(int)
        lr_brier = brier_score_loss(y_bin, y_prob[:, i])
        odds_brier = brier_score_loss(y_bin, odds_probs[:, i])
        print(f"  {name:>6}:  LR={lr_brier:.4f}  Odds={odds_brier:.4f}  "
              f"Diff={lr_brier-odds_brier:+.4f} {'LR wins' if lr_brier < odds_brier else 'Odds wins'}")
    print(f"\n  LR mean Brier:  {met['brier_mean']:.4f}")
    odds_mean = np.mean([brier_score_loss((y_test==i).astype(int), odds_probs[:, i]) for i in range(3)])
    print(f"  Odds mean Brier: {odds_mean:.4f}")
