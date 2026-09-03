from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.international_training_data import (
    PreparationConfig,
    add_training_weights,
    filter_external_results,
    load_external_results,
    standardize_team,
)
from src.poisson_model import WeightedPoissonModel
from src.world_cup_model import HOSTS, probability_metrics, result_labels


@dataclass(frozen=True)
class WorldCupCycle:
    year: int
    start_date: pd.Timestamp
    cutoff_date: pd.Timestamp


WORLD_CUP_CYCLES = {
    2014: WorldCupCycle(
        2014,
        pd.Timestamp("2010-07-12"),
        pd.Timestamp("2014-06-11"),
    ),
    2018: WorldCupCycle(
        2018,
        pd.Timestamp("2014-07-14"),
        pd.Timestamp("2018-06-13"),
    ),
    2022: WorldCupCycle(
        2022,
        pd.Timestamp("2018-07-16"),
        pd.Timestamp("2022-11-19"),
    ),
}


def load_world_cup_matches(games_path: Path, year: int) -> pd.DataFrame:
    source = pd.read_excel(games_path, sheet_name=f"WorldCup{year}")
    frame = pd.DataFrame(
        {
            "Year": year,
            "Date": pd.to_datetime(source["Date"]).dt.normalize(),
            "Home": source["Home"].map(standardize_team),
            "Away": source["Away"].map(standardize_team),
            "HomeGoals": pd.to_numeric(source["HGFT"], errors="raise").astype(int),
            "AwayGoals": pd.to_numeric(source["AGFT"], errors="raise").astype(int),
        }
    )
    frame["Result"] = np.asarray(["H", "D", "A"])[
        result_labels(frame["HomeGoals"], frame["AwayGoals"])
    ]
    hosts = HOSTS[year]
    frame["HostAdvantage"] = (
        frame["Home"].isin(hosts).astype(int)
        - frame["Away"].isin(hosts).astype(int)
    )
    return frame


def historical_team_universe(
    external: pd.DataFrame,
    test_matches: pd.DataFrame,
) -> set[str]:
    qualifying = external[
        external["Tournament"]
        .str.lower()
        .str.contains("world cup qualification", na=False)
    ]
    return (
        set(qualifying["Home"])
        | set(qualifying["Away"])
        | set(test_matches["Home"])
        | set(test_matches["Away"])
    )


def prepare_historical_cycle(
    external_path: Path,
    games_path: Path,
    year: int,
    half_life_days: float = 540.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if year not in WORLD_CUP_CYCLES:
        raise ValueError(f"Unsupported World Cup year: {year}")
    cycle = WORLD_CUP_CYCLES[year]
    config = PreparationConfig(
        start_date=cycle.start_date,
        cutoff_date=cycle.cutoff_date,
        half_life_days=half_life_days,
    )
    test_matches = load_world_cup_matches(games_path, year)
    external = load_external_results(external_path)
    in_window = external[
        external["Date"].between(
            cycle.start_date,
            cycle.cutoff_date,
            inclusive="both",
        )
    ].copy()
    universe = historical_team_universe(in_window, test_matches)
    training, audit = filter_external_results(external, universe, config)
    training = add_training_weights(training, config)
    training["NeutralKnown"] = training["Neutral"].notna().astype(int)
    training["TrueHome"] = (~training["Neutral"]).astype("Int64")
    training = training.sort_values(["Date", "Home", "Away"]).reset_index(drop=True)

    missing_test_teams = (
        set(test_matches["Home"]) | set(test_matches["Away"])
    ) - (set(training["Home"]) | set(training["Away"]))
    if missing_test_teams:
        raise ValueError(
            f"{year} test teams missing from training data: {sorted(missing_test_teams)}"
        )

    audit.update(
        {
            "world_cup_year": year,
            "cycle_start": cycle.start_date.strftime("%Y-%m-%d"),
            "cycle_cutoff": cycle.cutoff_date.strftime("%Y-%m-%d"),
            "team_universe_size": len(universe),
            "training_rows": len(training),
            "training_team_count": len(
                set(training["Home"]) | set(training["Away"])
            ),
            "test_rows": len(test_matches),
        }
    )
    return training, test_matches, audit


def backtest_cycle(
    training: pd.DataFrame,
    test_matches: pd.DataFrame,
    l2: float = 0.08,
) -> tuple[WeightedPoissonModel, pd.DataFrame, dict]:
    model = WeightedPoissonModel(l2=l2).fit(training)
    predictions = model.predict_fixtures(test_matches, max_goals=10, top_n=3)
    output = predictions[
        [
            "Year",
            "Date",
            "Home",
            "Away",
            "HomeGoals",
            "AwayGoals",
            "Result",
            "HostAdvantage",
            "Poisson_xG_H",
            "Poisson_xG_A",
            "PoissonP_H",
            "PoissonP_D",
            "PoissonP_A",
            "MostLikelyScore",
            "MostLikelyScoreP",
            "Score1",
            "Score1P",
            "Score2",
            "Score2P",
            "Score3",
            "Score3P",
        ]
    ].copy()
    actual = np.zeros((len(output), 3), dtype=float)
    actual[
        np.arange(len(output)),
        output["Result"].map({"H": 0, "D": 1, "A": 2}).to_numpy(),
    ] = 1.0
    probabilities = output[
        ["PoissonP_H", "PoissonP_D", "PoissonP_A"]
    ].to_numpy(dtype=float)
    metrics = probability_metrics(actual, probabilities)
    metrics.update(
        {
            "goal_mae_home": float(
                np.mean(np.abs(output["HomeGoals"] - output["Poisson_xG_H"]))
            ),
            "goal_mae_away": float(
                np.mean(np.abs(output["AwayGoals"] - output["Poisson_xG_A"]))
            ),
            "exact_score_accuracy": float(
                np.mean(
                    output["MostLikelyScore"]
                    == (
                        output["HomeGoals"].astype(str)
                        + "-"
                        + output["AwayGoals"].astype(str)
                    )
                )
            ),
            "top3_score_accuracy": float(
                np.mean(
                    [
                        f"{row.HomeGoals}-{row.AwayGoals}"
                        in {row.Score1, row.Score2, row.Score3}
                        for row in output.itertuples()
                    ]
                )
            ),
        }
    )
    return model, output, metrics
