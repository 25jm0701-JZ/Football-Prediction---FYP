from __future__ import annotations

import numpy as np
import pandas as pd

from src.poisson_model import WeightedPoissonModel
from src.world_cup_model import predict_fixture_components


OFFICIAL_PROBABILITY_SOURCE = "Historical Market Proxy"


def consistency_label(disagreement: float) -> str:
    if disagreement <= 0.10:
        return "High"
    if disagreement <= 0.20:
        return "Medium"
    return "Low"


def predicted_outcome(probabilities: np.ndarray) -> str:
    return ("Home", "Draw", "Away")[int(np.argmax(probabilities))]


def build_platform_predictions(
    fixtures: pd.DataFrame,
    rankings_path,
    main_bundle: dict,
    poisson_model: WeightedPoissonModel,
    experimental_weights: tuple[float, float, float],
) -> pd.DataFrame:
    frame, result_probability, market_probability = predict_fixture_components(
        fixtures,
        rankings_path,
        main_bundle,
    )
    poisson = poisson_model.predict_fixtures(
        fixtures,
        max_goals=10,
        top_n=3,
    )
    poisson_probability = poisson[
        ["PoissonP_H", "PoissonP_D", "PoissonP_A"]
    ].to_numpy(dtype=float)

    official_probability = market_probability
    result_weight, market_weight, poisson_weight = experimental_weights
    experimental_probability = (
        result_weight * result_probability
        + market_weight * market_probability
        + poisson_weight * poisson_probability
    )
    disagreement = 0.5 * np.abs(
        market_probability - poisson_probability
    ).sum(axis=1)

    frame[["ResultP_H", "ResultP_D", "ResultP_A"]] = result_probability
    frame[
        ["MarketProxyP_H", "MarketProxyP_D", "MarketProxyP_A"]
    ] = market_probability
    frame[
        ["PoissonP_H", "PoissonP_D", "PoissonP_A"]
    ] = poisson_probability
    frame[
        ["OfficialP_H", "OfficialP_D", "OfficialP_A"]
    ] = official_probability
    frame[
        ["ExperimentalP_H", "ExperimentalP_D", "ExperimentalP_A"]
    ] = experimental_probability

    frame["OfficialFairOdds_H"] = 1.0 / frame["OfficialP_H"]
    frame["OfficialFairOdds_D"] = 1.0 / frame["OfficialP_D"]
    frame["OfficialFairOdds_A"] = 1.0 / frame["OfficialP_A"]
    frame["OfficialOutcome"] = [
        predicted_outcome(row) for row in official_probability
    ]
    frame["PoissonOutcome"] = [
        predicted_outcome(row) for row in poisson_probability
    ]
    frame["ModelDisagreement"] = disagreement
    frame["ModelConsistency"] = [
        consistency_label(value) for value in disagreement
    ]
    frame["DirectionAgreement"] = (
        frame["OfficialOutcome"] == frame["PoissonOutcome"]
    )
    frame["OfficialProbabilitySource"] = OFFICIAL_PROBABILITY_SOURCE
    frame["ExperimentalWeightResult"] = result_weight
    frame["ExperimentalWeightMarket"] = market_weight
    frame["ExperimentalWeightPoisson"] = poisson_weight

    poisson_columns = [
        "Poisson_xG_H",
        "Poisson_xG_A",
        "MostLikelyScore",
        "MostLikelyScoreP",
        "Score1",
        "Score1P",
        "Score2",
        "Score2P",
        "Score3",
        "Score3P",
    ]
    for column in poisson_columns:
        frame[column] = poisson[column].to_numpy()
    return frame
