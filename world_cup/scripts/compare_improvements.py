"""
Comprehensive comparison of all improvement approaches.

Compares:
  1. Feature-aggregation method  : unweighted mean  vs  minutes-weighted
  2. Classifier                  : Softmax  vs  XGBoost  vs  RandomForest  vs  GB
  3. Poisson enhancement         : baseline  vs  PlayerAwarePoisson
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
WC_DIR = THIS_DIR.parent
PROJECT_ROOT = WC_DIR.parent
sys.path.insert(0, str(WC_DIR))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from src.world_cup_model import (
    load_historical_matches,
    fit_models,
    cross_validate,
    SoftmaxModel,
    feature_matrix,
    target_matrix,
    market_target_matrix,
    feature_matrix,
    one_hot,
    result_labels,
    probability_metrics,
    augment_symmetry,
    CLASSES,
    BASE_FEATURES,
    FEATURES,
    player_feature_names,
)
from src.player_features import (
    load_team_features,
    load_team_features_weighted,
    add_player_features_to_matches,
    PLAYER_FEATURES,
    PLAYER_FEATURES_WEIGHTED,
    player_feature_names_weighted,
)
from src.ml_ensemble import XGBoostModel, RFModel, GBModel
from src.player_poisson import PlayerAwarePoisson

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GAMES = Path("world_cup/data/raw/WorldCup2026.xlsx")
RANKINGS = Path("world_cup/world_cup_rankings_4_sheets_simple.xlsx")
PLAYER_PATHS = {
    2014: Path(
        "C:/Users/-jmmmm/Downloads"
        "/international-fifa-world-cup-2014-brazil-players-2014-to-2014-stats.csv"
    ),
    2018: Path(
        "C:/Users/-jmmmm/Downloads"
        "/international-fifa-world-cup-2018-russia-players-2018-to-2018-stats.csv"
    ),
    2022: Path(
        "C:/Users/-jmmmm/Downloads"
        "/international-fifa-world-cup-2022-qatar-players-2022-to-2022-stats.csv"
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_matches(
    use_player: bool,
    weighted: bool = False,
    base_matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load matches; optionally attach player features.

    Returns a DataFrame whose columns match ``FEATURES`` — both unweighted
    and weighted player columns are present, but the set *not* selected is
    filled with zero so the comparison is clean.
    """
    if base_matches is not None:
        matches = base_matches.copy()
    else:
        matches, _ = load_historical_matches(GAMES, RANKINGS)

    # Zero-fill all player feature columns (required by FEATURES)
    for pf in player_feature_names() + player_feature_names_weighted():
        matches[pf] = 0.0

    if not use_player:
        return matches

    if weighted:
        tf = load_team_features_weighted(PLAYER_PATHS)
        matches = add_player_features_to_matches(
            matches, tf, feature_names=PLAYER_FEATURES_WEIGHTED
        )
        # unweighted cols stay zero
    else:
        tf = load_team_features(PLAYER_PATHS)
        matches = add_player_features_to_matches(
            matches, tf, feature_names=PLAYER_FEATURES
        )
        # weighted cols stay zero

    return matches


# ---------------------------------------------------------------------------
# CV for Softmax / XGBoost / RF / GB            (same feature interface)
# ---------------------------------------------------------------------------


def run_cv_classifier(
    matches: pd.DataFrame,
    model_factory,
    label: str,
) -> pd.DataFrame:
    """Leave-one-year-out CV for any model with ``fit`` / ``predict_proba``."""
    rows = []
    for held_out in (2014, 2018, 2022):
        train = matches[matches["Year"] != held_out]
        test = matches[matches["Year"] == held_out].copy()

        x_train = feature_matrix(train)
        y_train = target_matrix(train)
        x_test = feature_matrix(test)
        y_test = target_matrix(test)

        # Augment training
        x_aug, y_aug = augment_symmetry(x_train, y_train)
        model = model_factory()
        model.fit(x_aug, y_aug)
        proba = model.predict_proba(x_test)

        metrics = probability_metrics(y_test, proba)
        rows.append({
            "test_year": str(held_out),
            "model": label,
            **metrics,
        })
        # accuracy from argmax
        acc = float(np.mean(proba.argmax(axis=1) == y_test.argmax(axis=1)))
        rows[-1]["accuracy"] = acc

    # Aggregate across all years
    all_probas, all_targets = [], []
    for held_out in (2014, 2018, 2022):
        train = matches[matches["Year"] != held_out]
        test = matches[matches["Year"] == held_out]
        x_train = feature_matrix(train)
        y_train = target_matrix(train)
        x_test = feature_matrix(test)
        y_test = target_matrix(test)
        x_aug, y_aug = augment_symmetry(x_train, y_train)
        model = model_factory()
        model.fit(x_aug, y_aug)
        all_probas.append(model.predict_proba(x_test))
        all_targets.append(y_test)

    all_probas = np.vstack(all_probas)
    all_targets = np.vstack(all_targets)
    agg = probability_metrics(all_targets, all_probas)
    agg["accuracy"] = float(np.mean(all_probas.argmax(axis=1) == all_targets.argmax(axis=1)))
    rows.append({"test_year": "ALL", "model": label, **agg})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Poisson comparisons
# ---------------------------------------------------------------------------


def run_cv_poisson(
    matches: pd.DataFrame,
    use_player: bool,
    player_feat: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Leave-one-year-out CV for Poisson-based models.

    Uses the training-data matches (which have FinalWeight, TrueHome etc.)
    from the original pipeline training data, NOT the World Cup matches
    DataFrame used above.
    """
    from src.poisson_model import WeightedPoissonModel

    rows = []
    for held_out in (2014, 2018, 2022):
        train = matches[matches["Year"] != held_out]
        test = matches[matches["Year"] == held_out].copy()

        if use_player:
            model = PlayerAwarePoisson(player_features=tuple(player_feat or ()))
            model.fit(train)
        else:
            model = WeightedPoissonModel()
            model.fit(train)

        # Predict each test match
        home_outs, away_outs = [], []
        for _, row in test.iterrows():
            host_adv = int(row.get("HostAdvantage", 0) if "HostAdvantage" in row
                           else (row["Home"] in {"Canada", "Mexico", "USA"}).astype(int)
                           if hasattr(row, "Home") else 0)
            h, a = model.expected_goals(row["Home"], row["Away"], host_adv)
            home_outs.append(h)
            away_outs.append(a)

        # Convert to H/D/A probabilities via Poisson score matrix
        from src.poisson_model import score_summary

        probas = []
        for h, a in zip(home_outs, away_outs):
            s = score_summary(h, a, max_goals=10)
            probas.append([s["P_H"], s["P_D"], s["P_A"]])
        probas = np.asarray(probas)
        y_test = one_hot(result_labels(test["HomeGoals"] if "HomeGoals" in test
                                       else test["HGFT"],
                                       test["AwayGoals"] if "AwayGoals" in test
                                       else test["AGFT"]))

        metrics = probability_metrics(y_test, probas)
        acc = float(np.mean(probas.argmax(axis=1) == y_test.argmax(axis=1)))
        rows.append({
            "test_year": str(held_out),
            "model": "Poisson+Player" if use_player else "Poisson",
            **metrics,
            "accuracy": acc,
        })

    # Aggregate
    all_p, all_y = [], []
    for held_out in (2014, 2018, 2022):
        train = matches[matches["Year"] != held_out]
        test = matches[matches["Year"] == held_out]
        if use_player:
            model = PlayerAwarePoisson(player_features=tuple(player_feat or ()))
            model.fit(train)
        else:
            model = WeightedPoissonModel()
            model.fit(train)

        for _, row in test.iterrows():
            host_adv = 0
            h, a = model.expected_goals(row["Home"], row["Away"], host_adv)
            s = score_summary(h, a, max_goals=10)
            all_p.append([s["P_H"], s["P_D"], s["P_A"]])
        all_y.append(one_hot(
            result_labels(test["HomeGoals"] if "HomeGoals" in test else test["HGFT"],
                          test["AwayGoals"] if "AwayGoals" in test else test["AGFT"])
        ))
    all_p = np.vstack(all_p)
    all_y = np.vstack(all_y)
    agg = probability_metrics(all_y, all_p)
    agg["accuracy"] = float(np.mean(all_p.argmax(axis=1) == all_y.argmax(axis=1)))
    rows.append({"test_year": "ALL", "model": "Poisson+Player" if use_player else "Poisson",
                 **agg})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Poisson training data  (qualifier matches with FinalWeight etc.)
# ---------------------------------------------------------------------------

POISSON_TRAINING_PATH = (
    PROJECT_ROOT / "outputs" / "world_cup" / "poisson_training_matches.csv"
)


def _get_poisson_data() -> pd.DataFrame:
    """Load pre-prepared Poisson training data and attach player features."""
    pdf = pd.read_csv(POISSON_TRAINING_PATH, parse_dates=["Date"])
    pdf["Year"] = pd.to_datetime(pdf["Date"]).dt.year
    # Map to canonical teams
    from src.world_cup_model import canonical_team
    pdf["Home"] = pdf["Home"].map(canonical_team)
    pdf["Away"] = pdf["Away"].map(canonical_team)
    return pdf


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 90)
    print("  COMPREHENSIVE IMPROVEMENT COMPARISON  —  World Cup 2026")
    print("=" * 90)

    # ---- 1. Load base matches (no player features) -------------------- #
    print("\n[1/5] Loading match data …")
    base_matches, _ = load_historical_matches(GAMES, RANKINGS)

    # ---- 2. Softmax comparisons (base / unweighted / weighted) -------- #
    print("[2/5] Running Softmax models …")
    results = []

    # Softmax baseline (no player features)
    m_base = _make_matches(use_player=False, base_matches=base_matches)
    results.append(run_cv_classifier(m_base, SoftmaxModel, "Softmax (base)"))

    # Softmax + unweighted player features
    m_uw = _make_matches(use_player=True, weighted=False, base_matches=base_matches)
    results.append(run_cv_classifier(m_uw, SoftmaxModel, "Softmax + player (unweighted)"))

    # Softmax + weighted player features
    m_w = _make_matches(use_player=True, weighted=True, base_matches=base_matches)
    results.append(run_cv_classifier(m_w, SoftmaxModel, "Softmax + player (weighted)"))

    # ---- 3. XGBoost / RF / GB  (with unweighted player features) ----- #
    print("[3/5] Running tree-based models …")
    for name, factory in [
        ("XGBoost + player", XGBoostModel),
        ("RandomForest + player", RFModel),
        ("GradientBoost + player", GBModel),
    ]:
        results.append(run_cv_classifier(m_uw, factory, name))

    # ---- 4. Combine weighted player + XGBoost ------------------------ #
    results.append(run_cv_classifier(
        m_w, XGBoostModel, "XGBoost + player (weighted)"
    ))

    # ---- 5. Print classifier comparison table ------------------------- #
    all_metrics = pd.concat(results, ignore_index=True)
    print("\n" + "─" * 90)
    print("  CLASSIFIER COMPARISON  (log-loss ↓ better, accuracy ↑ better)")
    print("─" * 90)

    # Pivot for readability
    for metric_name, fmt in [("log_loss", ".4f"), ("accuracy", ".4f")]:
        print(f"\n  [{metric_name}]")
        print(f"  {'Model':45s} ", end="")
        for yr in ["2014", "2018", "2022", "ALL"]:
            print(f"{yr:>10s}", end="")
        print()
        print("  " + "-" * 85)
        for model_name in sorted(all_metrics["model"].unique()):
            sub = all_metrics[all_metrics["model"] == model_name]
            print(f"  {model_name:45s}", end="")
            for yr in ["2014", "2018", "2022", "ALL"]:
                val = sub[sub["test_year"] == yr][metric_name].values
                if len(val):
                    print(f"  {val[0]:{fmt}}", end="")
                else:
                    print(f"  {'N/A':>10s}", end="")
            print()

    # ---- 6. Best classifier vs Softmax comparison  ------------------- #
    print("\n" + "─" * 90)
    print("  BEST MODEL COMPARISON  (Δ vs Softmax baseline)")
    print("─" * 90)
    baseline = all_metrics[
        (all_metrics["model"] == "Softmax (base)") & (all_metrics["test_year"] == "ALL")
    ]
    if len(baseline):
        base_ll = baseline["log_loss"].iloc[0]
        base_acc = baseline["accuracy"].iloc[0]
        print(f"  Softmax baseline:         LL={base_ll:.4f}  Acc={base_acc:.4f}")
        for model_name in sorted(all_metrics["model"].unique()):
            if model_name == "Softmax (base)":
                continue
            sub = all_metrics[
                (all_metrics["model"] == model_name)
                & (all_metrics["test_year"] == "ALL")
            ]
            if len(sub):
                ll = sub["log_loss"].iloc[0]
                acc = sub["accuracy"].iloc[0]
                ll_d = (base_ll - ll) * 100
                acc_d = (acc - base_acc) * 100
                marker = " ★" if ll < base_ll else ""
                print(f"  {model_name:45s}  LL={ll:.4f} ({ll_d:+.2f}%)  Acc={acc:.4f} ({acc_d:+.2f}%){marker}")

    # ---- 7. Poisson comparison (needs separate training data) -------- #
    print("\n" + "─" * 90)
    print("  POISSON MODEL COMPARISON")
    print("─" * 90)

    if not POISSON_TRAINING_PATH.exists():
        print("  (Poisson training data not found)")
    else:
        poisson_data = _get_poisson_data()
        print(f"  Poisson training matches: {len(poisson_data)}")

        # Split: train on 2022-2025 data, test on 2026 (current cycle)
        train = poisson_data[poisson_data["Year"] < 2026].copy()
        test = poisson_data[poisson_data["Year"] == 2026].copy()

        def _eval_poisson(model, train_df, test_df):
            model.fit(train_df)
            probas = []
            for _, row in test_df.iterrows():
                h, a = model.expected_goals(row["Home"], row["Away"], host_advantage=0)
                from src.poisson_model import score_summary
                s = score_summary(h, a, max_goals=10)
                probas.append([s["P_H"], s["P_D"], s["P_A"]])
            probas = np.asarray(probas)
            y_true = one_hot(
                result_labels(test_df["HomeGoals"], test_df["AwayGoals"])
            )
            if probas.ndim == 1 or probas.shape[1] != 3:
                # degenerate fallback
                probas = np.tile([0.4, 0.2, 0.4], (len(probas) if probas.ndim == 1 else 1, 1))
            metrics = probability_metrics(y_true, probas)
            metrics["accuracy"] = float(np.mean(probas.argmax(axis=1) == y_true.argmax(axis=1)))
            return metrics

        # Baseline Poisson
        from src.poisson_model import WeightedPoissonModel
        base_metrics = _eval_poisson(WeightedPoissonModel(), train, test)

        # PlayerAwarePoisson — train with player features
        try:
            train_pf = train.copy()
            test_pf = test.copy()
            tf = load_team_features(PLAYER_PATHS)

            # Add Year needed for merge
            train_pf["Year"] = pd.to_datetime(train_pf["Date"]).dt.year
            test_pf["Year"] = pd.to_datetime(test_pf["Date"]).dt.year

            train_pf = add_player_features_to_matches(
                train_pf, tf, feature_names=PLAYER_FEATURES
            )
            test_pf = add_player_features_to_matches(
                test_pf, tf, feature_names=PLAYER_FEATURES
            )

            # Only keep matches that have player data (World Cup years)
            train_pf = train_pf.dropna(subset=PLAYER_FEATURES)
            test_pf = test_pf.dropna(subset=PLAYER_FEATURES)

            print(f"  (Player features available: {len(train_pf)} train, {len(test_pf)} test)")

            if len(train_pf) >= 20:
                pap_metrics = _eval_poisson(
                    PlayerAwarePoisson(player_features=tuple(PLAYER_FEATURES)),
                    train_pf, test_pf,
                )
                print(f"\n  {'Model':45s}  {'LogLoss':>10s}  {'Accuracy':>10s}")
                print("  " + "-" * 70)
                for name, m in [("Poisson (baseline)", base_metrics),
                                ("Poisson + Player features", pap_metrics)]:
                    print(f"  {name:45s}  {m['log_loss']:.4f}     {m['accuracy']:.4f}")
            else:
                print(f"  (Skipping — insufficient data)")
        except Exception as e:
            print(f"  (Poisson + player comparison failed: {e})")

    # ---- 8. Model coefficients (PlayerAwarePoisson β weights) -------- #
    print("\n" + "─" * 90)
    print("  PLAYER-AWARE POISSON  —  β coefficients")
    print("─" * 90)
    try:
        pap = PlayerAwarePoisson(player_features=tuple(PLAYER_FEATURES))
        pap.fit(train_pf)
        if pap.player_weights is not None:
            for name, beta in zip(PLAYER_FEATURES, pap.player_weights):
                print(f"  {name:45s}  β = {beta:+.6f}")
            print(f"  intercept                {pap.intercept:.4f}")
            print(f"  home_advantage           {pap.home_advantage:.4f}")
    except Exception as e:
        print(f"  (Could not fit PlayerAwarePoisson: {e})")

    print("\n" + "=" * 90)
    print("  DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
