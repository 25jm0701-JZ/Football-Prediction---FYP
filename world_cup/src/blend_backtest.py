from __future__ import annotations

import numpy as np
import pandas as pd

from src.world_cup_model import one_hot, probability_metrics


COMPONENTS = {
    "result": ["ResultP_H", "ResultP_D", "ResultP_A"],
    "market_proxy": ["MarketProxyP_H", "MarketProxyP_D", "MarketProxyP_A"],
    "poisson": ["PoissonP_H", "PoissonP_D", "PoissonP_A"],
}


def align_backtest_predictions(
    original_predictions: pd.DataFrame,
    poisson_predictions: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["Year", "Date", "Home", "Away", "Result"]
    result = original_predictions[
        original_predictions["Model"] == "result_model"
    ][keys + ["P_H", "P_D", "P_A"]].rename(
        columns={"P_H": "ResultP_H", "P_D": "ResultP_D", "P_A": "ResultP_A"}
    )
    market = original_predictions[
        original_predictions["Model"] == "historical_market_proxy"
    ][keys + ["P_H", "P_D", "P_A"]].rename(
        columns={
            "P_H": "MarketProxyP_H",
            "P_D": "MarketProxyP_D",
            "P_A": "MarketProxyP_A",
        }
    )
    poisson = poisson_predictions.rename(
        columns={
            "PoissonP_H": "PoissonP_H",
            "PoissonP_D": "PoissonP_D",
            "PoissonP_A": "PoissonP_A",
        }
    )
    poisson_columns = keys + [
        "PoissonP_H",
        "PoissonP_D",
        "PoissonP_A",
        "Poisson_xG_H",
        "Poisson_xG_A",
        "MostLikelyScore",
    ]
    merged = result.merge(market, on=keys, validate="one_to_one").merge(
        poisson[poisson_columns],
        on=keys,
        validate="one_to_one",
    )
    if len(merged) != len(poisson_predictions):
        raise ValueError(
            f"Prediction alignment lost rows: {len(poisson_predictions)} -> {len(merged)}"
        )
    return merged.sort_values(["Year", "Date", "Home", "Away"]).reset_index(drop=True)


def blend_probabilities(
    frame: pd.DataFrame,
    weights: tuple[float, float, float],
) -> np.ndarray:
    result_weight, market_weight, poisson_weight = weights
    return (
        result_weight * frame[COMPONENTS["result"]].to_numpy(dtype=float)
        + market_weight
        * frame[COMPONENTS["market_proxy"]].to_numpy(dtype=float)
        + poisson_weight * frame[COMPONENTS["poisson"]].to_numpy(dtype=float)
    )


def actual_matrix(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["Result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return one_hot(labels)


def optimize_weights(
    frame: pd.DataFrame,
    step: float = 0.05,
) -> tuple[tuple[float, float, float], dict]:
    if not 0 < step <= 1:
        raise ValueError("step must be in (0, 1].")
    actual = actual_matrix(frame)
    units = int(round(1.0 / step))
    best_weights = (1.0, 0.0, 0.0)
    best_metrics = {"log_loss": np.inf}
    for result_units in range(units + 1):
        for market_units in range(units - result_units + 1):
            poisson_units = units - result_units - market_units
            weights = (
                result_units / units,
                market_units / units,
                poisson_units / units,
            )
            probabilities = blend_probabilities(frame, weights)
            metrics = probability_metrics(actual, probabilities)
            if (
                metrics["log_loss"] < best_metrics["log_loss"] - 1e-12
                or (
                    abs(metrics["log_loss"] - best_metrics["log_loss"]) <= 1e-12
                    and metrics["brier_score"]
                    < best_metrics.get("brier_score", np.inf)
                )
            ):
                best_weights = weights
                best_metrics = metrics
    return best_weights, best_metrics


def nested_blend_backtest(
    frame: pd.DataFrame,
    step: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    weights_rows = []
    for held_out_year in sorted(frame["Year"].unique()):
        train = frame[frame["Year"] != held_out_year]
        test = frame[frame["Year"] == held_out_year].copy()
        weights, train_metrics = optimize_weights(train, step=step)
        probability = blend_probabilities(test, weights)
        test[["BlendP_H", "BlendP_D", "BlendP_A"]] = probability
        test["BlendWeightResult"] = weights[0]
        test["BlendWeightMarketProxy"] = weights[1]
        test["BlendWeightPoisson"] = weights[2]
        predictions.append(test)
        test_metrics = probability_metrics(actual_matrix(test), probability)
        weights_rows.append(
            {
                "test_year": int(held_out_year),
                "weight_result": weights[0],
                "weight_market_proxy": weights[1],
                "weight_poisson": weights[2],
                "selection_log_loss": train_metrics["log_loss"],
                "test_log_loss": test_metrics["log_loss"],
                "test_brier_score": test_metrics["brier_score"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(weights_rows)


def comparison_metrics(
    frame: pd.DataFrame,
    blend_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    model_columns = {
        "result_model": COMPONENTS["result"],
        "historical_market_proxy": COMPONENTS["market_proxy"],
        "poisson_model": COMPONENTS["poisson"],
    }
    for year in [*sorted(frame["Year"].unique()), "ALL"]:
        subset = frame if year == "ALL" else frame[frame["Year"] == year]
        for model, columns in model_columns.items():
            metrics = probability_metrics(
                actual_matrix(subset),
                subset[columns].to_numpy(dtype=float),
            )
            rows.append({"test_year": year, "model": model, **metrics})

        blend_subset = (
            blend_predictions
            if year == "ALL"
            else blend_predictions[blend_predictions["Year"] == year]
        )
        metrics = probability_metrics(
            actual_matrix(blend_subset),
            blend_subset[["BlendP_H", "BlendP_D", "BlendP_A"]].to_numpy(
                dtype=float
            ),
        )
        rows.append(
            {
                "test_year": year,
                "model": "nested_optimized_blend",
                **metrics,
            }
        )
    return pd.DataFrame(rows)
