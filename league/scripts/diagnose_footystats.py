"""
Diagnose: reproduce the claimed 67.98% from Experiment 009.
Test multiple configurations to find the discrepancy.
"""
import sys
from pathlib import Path
sys.path.insert(0, "league")

import numpy as np
from src.data_loader import load_multiple
from src.train import train_and_evaluate
from src.features import build_features, get_feature_columns
from src.models import fill_features, temporal_split
from src.evaluate import classification_metrics
from sklearn.linear_model import LogisticRegression

def _season(d):
    y = d.year
    return f"{y}-{str(y+1)[-2:]}" if d.month >= 8 else f"{y-1}-{str(y)[-2:]}"

# Load only seasons that HAVE FootyStats data (2019-20 to 2024-25)
files = sorted(Path("league/data/raw").glob("PL1920.csv")) + \
        sorted(Path("league/data/raw").glob("PL2021.csv")) + \
        sorted(Path("league/data/raw").glob("PL2122.csv")) + \
        sorted(Path("league/data/raw").glob("PL2223.csv")) + \
        sorted(Path("league/data/raw").glob("PL2324.csv")) + \
        sorted(Path("league/data/raw").glob("PL2425.csv"))

print("=" * 60)
print("  Diagnosing LR + FootyStats accuracy gap")
print("=" * 60)

results = []

# Config A: 6 seasons (no 2025-26), 80/20 split
print("\n--- A: 6 seasons (2019-2025), 80/20 split ---")
df = load_multiple(files)
df["season"] = df["date"].apply(_season)
df = df.sort_values("date").reset_index(drop=True)
print(f"  {len(df)} matches, {df['date'].min().date()} -> {df['date'].max().date()}")

r = train_and_evaluate(
    df, model_name="lr", windows=[5, 10, 20],
    include_odds=True, include_shots=True, include_h2h=True,
    include_footystats=True, footystats_season_col="season",
    verbose=False,
)
print(f"  Acc: {r['accuracy']:.4f}")
results.append(("A) 6 seasons, [5,10,20]", r["accuracy"]))

# Config B: 6 seasons, [5,10] windows
r = train_and_evaluate(
    df, model_name="lr", windows=[5, 10],
    include_odds=True, include_shots=True, include_h2h=True,
    include_footystats=True, footystats_season_col="season",
    verbose=False,
)
print(f"  Acc: {r['accuracy']:.4f}")
results.append(("B) 6 seasons, [5,10]", r["accuracy"]))

# Config C: 6 seasons without FootyStats
r = train_and_evaluate(
    df, model_name="lr", windows=[5, 10, 20],
    include_odds=True, include_shots=True, include_h2h=True,
    include_footystats=False,
    verbose=False,
)
print(f"  Acc: {r['accuracy']:.4f}")
results.append(("C) 6 seasons, no FS", r["accuracy"]))

# Config D: 7 seasons, [5,10,20], with FootyStats
print("\n--- D: 7 seasons, [5,10,20] ---")
df7 = load_multiple(sorted(Path("league/data/raw").glob("PL*.csv")))
df7["season"] = df7["date"].apply(_season)
df7 = df7.sort_values("date").reset_index(drop=True)
r = train_and_evaluate(
    df7, model_name="lr", windows=[5, 10, 20],
    include_odds=True, include_shots=True, include_h2h=True,
    include_footystats=True, footystats_season_col="season",
    verbose=False,
)
print(f"  Acc: {r['accuracy']:.4f}")
results.append(("D) 7 seasons, [5,10,20]", r["accuracy"]))

# Config E: 7 seasons, no FootyStats
r = train_and_evaluate(
    df7, model_name="lr", windows=[5, 10, 20],
    include_odds=True, include_shots=True, include_h2h=True,
    include_footystats=False,
    verbose=False,
)
print(f"  Acc: {r['accuracy']:.4f}")
results.append(("E) 7 seasons, no FS", r["accuracy"]))

# Summary
print(f"\n{'='*60}")
print(f"  Summary")
print(f"{'='*60}")
for name, acc in results:
    print(f"  {name:<35} {acc:.4f}")
