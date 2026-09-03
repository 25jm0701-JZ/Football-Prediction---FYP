"""
Training and evaluation pipeline for league prediction models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.features import build_features, get_feature_columns
from src.models import (
    MODEL_REGISTRY,
    fill_features,
    get_model,
    list_available_models,
    temporal_split,
)


def train_and_evaluate(
    df: pd.DataFrame,
    model_name: str = "lr",
    target: str = "target_result",
    test_ratio: float = 0.2,
    windows: list[int] | None = None,
    model_kwargs: Optional[dict] = None,
    verbose: bool = True,
    return_model: bool = False,
    league_features: bool = False,
    league_normalize: bool = False,
    include_odds: bool = False,
    include_shots: bool = True,
    include_h2h: bool = True,
    include_pi_rating: bool = False,
    pi_rating_pi: float | None = None,
    include_footystats: bool = False,
    footystats_season_col: str = "season",
    poisson_model=None,
) -> dict:
    """Train a model and evaluate on temporal hold-out.

    Args:
        df: Raw DataFrame from data_loader.load_data()
        model_name: Model key (see models.list_available_models())
        target: Target column ('target_result' or 'target_over_2.5')
        test_ratio: Fraction of (latest) matches to hold out
        windows: Rolling window sizes for feature engineering
        model_kwargs: Extra kwargs passed to model constructor
        verbose: Print progress
        return_model: Also return the trained model
        include_footystats: Merge FootyStats extras (PPG, xG, possession)
        footystats_season_col: Column with season labels
        poisson_model: PoissonRatingModel instance to add pre-match
                       probabilities as extra features (~0.8pp lift).

    Returns:
        Dictionary with results
    """
    if model_kwargs is None:
        model_kwargs = {}

    # ── Prepare features ─────────────────────────────────────────
    df_feat = build_features(df, windows=windows,
                              league_features=league_features,
                              league_normalize=league_normalize,
                              include_odds=include_odds,
                              include_shots=include_shots,
                              include_h2h=include_h2h,
                              include_pi_rating=include_pi_rating,
                              pi_rating_pi=pi_rating_pi,
                              include_footystats=include_footystats,
                              footystats_season_col=footystats_season_col)
    feat_cols = get_feature_columns(df_feat)

    # Temporal split
    train_df, test_df = temporal_split(df_feat, test_ratio=test_ratio)

    # ── Poisson rating features (optional) ──────────────────────────
    if poisson_model is not None:
        if verbose:
            print("  Adding Poisson rating features...")

        pm = poisson_model
        pm.reset()
        pm.fit(train_df)

        # Training probabilities from the fit history
        train_poisson = np.array([
            [h["p_away"], h["p_draw"], h["p_home"]]
            for h in pm._history
        ])

        # Test probabilities (continue chronologically, online)
        test_poisson_list = []
        test_sorted = test_df.sort_values("date")
        for _, row in test_sorted.iterrows():
            pa, pd_, ph = pm.match_probabilities(
                row["home_team"], row["away_team"])
            test_poisson_list.append([pa, pd_, ph])
            pm.update(
                row["home_team"], row["away_team"],
                int(row["home_goals_full"]), int(row["away_goals_full"]),
            )
        test_poisson = np.array(test_poisson_list)

        # Add columns to dataframes
        poisson_cols = ["poisson_p_away", "poisson_p_draw", "poisson_p_home"]
        train_df[poisson_cols] = train_poisson
        test_df[poisson_cols] = test_poisson
        feat_cols.extend(poisson_cols)

        if verbose:
            print(f"  Poisson features added: {len(poisson_cols)}"
                  f" ({len(poisson_cols)} total features)")

    X_train = train_df[feat_cols].values.astype(float)
    y_train = train_df[target].values.astype(int)
    X_test = test_df[feat_cols].values.astype(float)
    y_test = test_df[target].values.astype(int)
    test_dates = test_df["date"]
    test_home = test_df["home_team"]
    test_away = test_df["away_team"]

    # Impute NaN
    X_train = fill_features(X_train)
    X_test = fill_features(X_test)

    is_poisson = model_name == "poisson"

    if verbose:
        print(f"Model: {model_name}")
        print(f"  Train: {len(X_train)} matches")
        print(f"  Test:  {len(X_test)} matches")
        print(f"  Features: {len(feat_cols)}")
        if not is_poisson:
            print(f"  Target distribution (train):")
            counts = pd.Series(y_train).value_counts().sort_index()
            for k, v in counts.items():
                print(f"    Class {k}: {v} ({v/len(y_train)*100:.1f}%)")

    # ── Train ─────────────────────────────────────────────────────
    model = get_model(model_name, **model_kwargs)

    if is_poisson:
        # Poisson predicts goals, not results directly
        # Need to extract goal columns from the split data
        goal_cols = ["home_goals_full", "away_goals_full"]
        if not all(c in train_df.columns for c in goal_cols):
            raise ValueError("Poisson model requires 'home_goals_full' and 'away_goals_full' columns")
        y_goals_train = train_df[goal_cols].values.astype(int)
        model.fit(X_train, y_goals_train)

        y_prob = model.predict_proba(X_test)
        y_pred = np.argmax(y_prob, axis=1)

        # Also predict O/U probs if available
        try:
            ou_prob = model.predict_ou_proba(X_test, threshold=2.5)
        except Exception:
            ou_prob = None
    else:
        # Standard classifier
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = None
        ou_prob = None

    # ── Evaluate ─────────────────────────────────────────────────
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    result = {
        "model": model_name,
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "test_dates": test_dates,
        "test_home": test_home,
        "test_away": test_away,
    }

    if verbose:
        n_classes = len(np.unique(y_test))
        if n_classes == 3:
            target_names = ["Away", "Draw", "Home"]
        elif n_classes == 2:
            target_names = ["Under", "Over"]
        else:
            target_names = None
        print(f"\n  Results on {len(y_test)} test matches:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        if target_names:
            print(f"\n  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=target_names))

    if return_model:
        result["model"] = model

    return result


def compare_models(
    df: pd.DataFrame,
    models: list[str] | None = None,
    target: str = "target_result",
    test_ratio: float = 0.2,
    windows: list[int] | None = None,
    verbose: bool = True,
    league_features: bool = False,
    league_normalize: bool = False,
    include_pi_rating: bool = False,
    include_footystats: bool = False,
    footystats_season_col: str = "season",
    poisson_model=None,
) -> pd.DataFrame:
    """Compare multiple models on the same data.

    Args:
        df: Raw DataFrame
        models: List of model keys (default: all available sklearn models)
        target: Target column
        test_ratio: Hold-out ratio
        windows: Rolling window sizes
        verbose: Print results
        include_footystats: Merge FootyStats extras

    Returns:
        DataFrame with model comparison results
    """
    if models is None:
        models = list(MODEL_REGISTRY.keys())

    results = []
    for model_name in models:
        try:
            if verbose:
                print(f"\n{'='*60}")
            r = train_and_evaluate(
                df, model_name=model_name, target=target,
                test_ratio=test_ratio, windows=windows,
                verbose=verbose, return_model=False,
                league_features=league_features,
                league_normalize=league_normalize,
                include_pi_rating=include_pi_rating,
                include_footystats=include_footystats,
                footystats_season_col=footystats_season_col,
                poisson_model=poisson_model,
            )
            results.append({
                "model": model_name,
                "accuracy": r["accuracy"],
                "f1_macro": r["f1_macro"],
                "f1_weighted": r["f1_weighted"],
                "train_size": r["train_size"],
                "test_size": r["test_size"],
            })
        except Exception as e:
            if verbose:
                print(f"  {model_name} FAILED: {e}")
            results.append({
                "model": model_name,
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "train_size": 0,
                "test_size": 0,
                "error": str(e),
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\n{'='*60}")
        print("Model Comparison (sorted by accuracy):")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))

    return results_df


def predict_matches(
    df: pd.DataFrame,
    model_name: str = "rf",
    target: str = "target_result",
    model_kwargs: Optional[dict] = None,
    windows: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train on ALL data and return predictions.

    Useful for final predictions when you don't need a test set.
    """
    if model_kwargs is None:
        model_kwargs = {}

    df_feat = build_features(df, windows=windows)
    feat_cols = get_feature_columns(df_feat)

    X = df_feat[feat_cols].values.astype(float)
    y = df_feat[target].values.astype(int)
    X = fill_features(X)

    model = get_model(model_name, **model_kwargs)
    model.fit(X, y)
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)
    else:
        y_prob = None

    return y_pred, y_prob
