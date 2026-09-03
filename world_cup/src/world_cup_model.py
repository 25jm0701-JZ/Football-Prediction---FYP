from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


from src.team_names import canonical_team

from src.player_features import (
    PLAYER_FEATURES as _PF_NAMES,
    PLAYER_FEATURES_WEIGHTED as _PF_WEIGHTED,
    add_player_features_to_matches,
    load_team_features,
    load_team_features_weighted,
    player_feature_names,
)

from src.ml_ensemble import XGBoostModel, HAS_XGB

CLASSES = ["H", "D", "A"]

# Rolling form features computed from international_results.csv.
# Each is a (Home − Away) differential so augment_symmetry handles them.
FORM_FEATURES = ["form_ppg_10", "form_gd_10", "form_ppg_5"]

# Pre-match xG difference (from detailed match CSVs; zero-filled for future
# fixtures where pre-match xG is unavailable).
XG_FEATURES = ["prematch_xg_diff"]

BASE_FEATURES = ["rating_diff_z", "host_advantage", "knockout"] + FORM_FEATURES + XG_FEATURES
FEATURES = BASE_FEATURES + list(_PF_NAMES) + list(_PF_WEIGHTED)

# Path to the big international results CSV (used for rolling form).
_INTERNATIONAL_RESULTS_PATH = None  # set at runtime by set_intl_results_path()

# Model factory: set to XGBoostModel to use a tree-based classifier
# for the result model (market proxy stays Softmax).
_RESULT_MODEL_FACTORY = None  # resolved lazily in fit_models()
HOSTS = {
    2014: {"Brazil"},
    2018: {"Russia"},
    2022: {"Qatar"},
    2026: {"Canada", "Mexico", "USA"},
}
KNOCKOUT_START = {
    2014: pd.Timestamp("2014-06-28"),
    2018: pd.Timestamp("2018-06-30"),
    2022: pd.Timestamp("2022-12-03"),
}


# Mapping from year → detailed-match CSV filename (holds pre-match xG).
_DETAILED_CSV_FILES: dict[int, str] = {
    2014: "international-fifa-world-cup-2014-brazil-matches-2014-to-2014-stats.csv",
    2018: "international-fifa-world-cup-2018-russia-matches-2018-to-2018-stats.csv",
    2022: "international-fifa-world-cup-2022-qatar-matches-2022-to-2022-stats.csv",
}


def _attach_prematch_xg(
    matches: pd.DataFrame,
    raw_data_dir: Path,
) -> pd.DataFrame:
    """Merge ``prematch_xg_diff`` from the detailed match CSVs.

    Only years listed in ``_DETAILED_CSV_FILES`` are processed; other years
    (or future fixtures) get ``prematch_xg_diff = 0``.
    """
    result = matches.copy()
    result["prematch_xg_diff"] = 0.0

    for year, fname in _DETAILED_CSV_FILES.items():
        path = raw_data_dir / fname
        if not path.exists():
            continue
        detail = pd.read_csv(path)
        detail["Home"] = detail["home_team_name"].map(canonical_team)
        detail["Away"] = detail["away_team_name"].map(canonical_team)
        detail["_prematch_xg_h"] = pd.to_numeric(
            detail.get("Home Team Pre-Match xG", pd.Series(np.nan, index=detail.index)),
            errors="coerce",
        )
        detail["_prematch_xg_a"] = pd.to_numeric(
            detail.get("Away Team Pre-Match xG", pd.Series(np.nan, index=detail.index)),
            errors="coerce",
        )
        lookup = detail.set_index(["Home", "Away"])[["_prematch_xg_h", "_prematch_xg_a"]]

        mask = result["Year"] == year
        idx = result.index[mask]
        keys = list(zip(result.loc[mask, "Home"], result.loc[mask, "Away"]))
        vals = lookup.reindex(keys)
        h_vals = vals["_prematch_xg_h"].fillna(0.0).to_numpy()
        a_vals = vals["_prematch_xg_a"].fillna(0.0).to_numpy()
        result.loc[mask, "prematch_xg_diff"] = h_vals - a_vals

    return result


def remove_overround(home_odds, draw_odds, away_odds) -> np.ndarray:
    odds = np.column_stack([home_odds, draw_odds, away_odds]).astype(float)
    inverse = 1.0 / odds
    return inverse / inverse.sum(axis=1, keepdims=True)


def set_intl_results_path(path: Path) -> None:
    """Set the global path to international_results.csv so that
    ``compute_international_form`` can be called later without re-passing it."""
    global _INTERNATIONAL_RESULTS_PATH
    _INTERNATIONAL_RESULTS_PATH = Path(path)


def use_xgboost_result_model() -> None:
    """Switch the result-model factory from Softmax to XGBoost.
    Call before ``fit_models`` or ``cross_validate``."""
    global _RESULT_MODEL_FACTORY
    if not HAS_XGB:
        raise RuntimeError("XGBoost is not installed.")
    _RESULT_MODEL_FACTORY = XGBoostModel


def compute_international_form(
    matches: pd.DataFrame,
    intl_path: Path | None = None,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """For each match in *matches* compute rolling form features from
    international_results.csv.

    For each team, before the match date, look back up to *window*
    international matches and compute:

      form_ppg_{w}   — mean points per game over the last *w* matches
      form_gd_{w}    — mean goal differential  (goals_for − goals_against)

    All features are (Home − Away) differentials.

    Parameters
    ----------
    matches : pd.DataFrame
        Must have columns ``Home``, ``Away``, and a date column (``Date`` or
        ``DateUTC``).
    intl_path : Path or None
        Path to the international_results.csv.  Falls back to the module-level
        ``_INTERNATIONAL_RESULTS_PATH`` if not given.
    windows : tuple of int
        Rolling window sizes.

    Returns
    -------
    pd.DataFrame
        Copy of *matches* with new columns ``form_ppg_{w}`` and
        ``form_gd_{w}`` added (zero-filled for teams with insufficient history).
    """
    path = intl_path or _INTERNATIONAL_RESULTS_PATH
    if path is None:
        raise ValueError(
            "No international-results path. "
            "Pass ``intl_path`` or call ``set_intl_results_path()`` first."
        )

    intl = pd.read_csv(path)
    intl["date"] = pd.to_datetime(intl["date"], utc=False)
    intl["home_team"] = intl["home_team"].map(canonical_team)
    intl["away_team"] = intl["away_team"].map(canonical_team)

    result = matches.copy()

    # Identify date column and make it tz-naive for comparison with intl data
    date_col = "Date" if "Date" in result.columns else "DateUTC"
    dates = pd.to_datetime(result[date_col])
    if hasattr(dates.dtype, "tz") and dates.dtype.tz is not None:
        dates = dates.dt.tz_localize(None)

    # Initialise columns
    for w in windows:
        result[f"form_ppg_{w}"] = 0.0
        result[f"form_gd_{w}"] = 0.0

    # Pre-split by team for faster look‑ups
    all_teams = set(result["Home"]) | set(result["Away"])
    team_matches: dict[str, pd.DataFrame] = {}
    for team in all_teams:
        tm = intl[(intl["home_team"] == team) | (intl["away_team"] == team)].copy()
        # home_score / away_score may contain NaN in the 49K dataset
        gf_arr = np.where(tm["home_team"] == team, tm["home_score"], tm["away_score"])
        ga_arr = np.where(tm["home_team"] == team, tm["away_score"], tm["home_score"])
        tm["_gf"] = pd.Series(gf_arr, index=tm.index).fillna(0.0)
        tm["_ga"] = pd.Series(ga_arr, index=tm.index).fillna(0.0)
        tm["_pts"] = np.where(
            tm["_gf"] > tm["_ga"], 3,
            np.where(tm["_gf"] == tm["_ga"], 1, 0),
        )
        team_matches[team] = tm.sort_values("date")

    # Compute form per match
    for idx in result.index:
        home = result.at[idx, "Home"]
        away = result.at[idx, "Away"]
        cutoff = dates.loc[idx]

        for w in windows:
            h_hist = team_matches.get(home)
            a_hist = team_matches.get(away)

            h_df = h_hist[h_hist["date"] < cutoff].tail(w) if h_hist is not None else pd.DataFrame()
            a_df = a_hist[a_hist["date"] < cutoff].tail(w) if a_hist is not None else pd.DataFrame()

            h_ppg = float(h_df["_pts"].mean()) if len(h_df) > 0 else 0.0
            a_ppg = float(a_df["_pts"].mean()) if len(a_df) > 0 else 0.0
            h_gd = float((h_df["_gf"] - h_df["_ga"]).mean()) if len(h_df) > 0 else 0.0
            a_gd = float((a_df["_gf"] - a_df["_ga"]).mean()) if len(a_df) > 0 else 0.0

            result.at[idx, f"form_ppg_{w}"] = h_ppg - a_ppg
            result.at[idx, f"form_gd_{w}"] = h_gd - a_gd

    # Fill any remaining NaN (teams with zero history at all)
    for w in windows:
        result[f"form_ppg_{w}"] = result[f"form_ppg_{w}"].fillna(0.0)
        result[f"form_gd_{w}"] = result[f"form_gd_{w}"].fillna(0.0)

    return result


def round_probability_columns(
    frame: pd.DataFrame, columns: list[str], decimals: int = 3
) -> pd.DataFrame:
    """Round probability rows while keeping each row's total exactly equal to one."""
    output = frame.copy()
    scale = 10**decimals
    probabilities = output[columns].to_numpy(dtype=float)
    scaled = probabilities * scale
    rounded_units = np.floor(scaled).astype(int)
    units_missing = scale - rounded_units.sum(axis=1)
    fractions = scaled - rounded_units

    for row_index, missing in enumerate(units_missing):
        if missing > 0:
            order = np.argsort(-fractions[row_index])
            rounded_units[row_index, order[:missing]] += 1
        elif missing < 0:
            order = np.argsort(fractions[row_index])
            rounded_units[row_index, order[: -missing]] -= 1

    output[columns] = rounded_units / scale
    return output


def format_probability_columns(
    frame: pd.DataFrame, columns: list[str], decimals: int = 3
) -> pd.DataFrame:
    """Format already-rounded probabilities with a fixed number of decimal places."""
    output = round_probability_columns(frame, columns, decimals)
    for column in columns:
        output[column] = output[column].map(lambda value: f"{value:.{decimals}f}")
    return output


def result_labels(home_goals, away_goals) -> np.ndarray:
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    labels = np.full(len(home_goals), 1, dtype=int)
    labels[home_goals > away_goals] = 0
    labels[home_goals < away_goals] = 2
    return labels


def one_hot(labels: np.ndarray, n_classes: int = 3) -> np.ndarray:
    result = np.zeros((len(labels), n_classes), dtype=float)
    result[np.arange(len(labels)), labels] = 1.0
    return result


def add_intercept(features: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(features)), features])


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class SoftmaxModel:
    coefficients: np.ndarray | None = None
    l2: float = 0.02
    learning_rate: float = 0.03
    max_iter: int = 8000
    tolerance: float = 1e-9

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "SoftmaxModel":
        x = add_intercept(np.asarray(features, dtype=float))
        y = np.asarray(targets, dtype=float)
        weights = np.zeros((x.shape[1], y.shape[1]), dtype=float)
        first_moment = np.zeros_like(weights)
        second_moment = np.zeros_like(weights)
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        previous_loss = np.inf

        for iteration in range(1, self.max_iter + 1):
            probabilities = softmax(x @ weights)
            clipped = np.clip(probabilities, 1e-12, 1.0)
            penalty = 0.5 * self.l2 * np.sum(weights[1:] ** 2)
            loss = -np.sum(y * np.log(clipped)) / len(x) + penalty

            gradient = x.T @ (probabilities - y) / len(x)
            gradient[1:] += self.l2 * weights[1:]
            first_moment = beta1 * first_moment + (1 - beta1) * gradient
            second_moment = beta2 * second_moment + (1 - beta2) * (gradient**2)
            first_hat = first_moment / (1 - beta1**iteration)
            second_hat = second_moment / (1 - beta2**iteration)
            weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)

            if abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss

        self.coefficients = weights
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return softmax(add_intercept(np.asarray(features, dtype=float)) @ self.coefficients)

    def to_dict(self) -> dict:
        if self.coefficients is None:
            raise RuntimeError("Cannot serialize an unfitted model.")
        return {
            "model_type": "softmax",
            "classes": CLASSES,
            "features": FEATURES,
            "coefficients": self.coefficients.tolist(),
            "l2": self.l2,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SoftmaxModel":
        return cls(
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            l2=float(payload.get("l2", 0.02)),
        )


def augment_symmetry(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mirrored_features = np.asarray(features, dtype=float).copy()
    # Negate all differential (Home − Away) features; only "knockout" is
    # a non-differential binary flag shared equally by both teams.
    for col_idx, name in enumerate(FEATURES):
        if name != "knockout":
            mirrored_features[:, col_idx] *= -1
    mirrored_targets = np.asarray(targets, dtype=float)[:, [2, 1, 0]]
    return (
        np.vstack([features, mirrored_features]),
        np.vstack([targets, mirrored_targets]),
    )


def load_rankings(path: Path, year: int) -> tuple[pd.DataFrame, dict]:
    if path.suffix.lower() == ".csv":
        rankings = pd.read_csv(path)
    else:
        rankings = pd.read_excel(path, sheet_name=str(year))
    rankings["Country"] = rankings["Country"].map(canonical_team)
    rankings["Points"] = pd.to_numeric(rankings["Points"], errors="raise")
    mean = float(rankings["Points"].mean())
    std = float(rankings["Points"].std(ddof=0))
    rankings["RatingZ"] = (rankings["Points"] - mean) / std
    return rankings, {"mean": mean, "std": std}


def fifa_expected_score(team_points: float, opponent_points: float) -> float:
    return float(1.0 / (1.0 + 10.0 ** (-(team_points - opponent_points) / 600.0)))


def update_fifa_points(
    home_points: float,
    away_points: float,
    home_goals: int,
    away_goals: int,
    importance: float,
    penalty_winner: str | None = None,
    knockout_protection: bool = False,
) -> tuple[float, float]:
    expected_home = fifa_expected_score(home_points, away_points)
    expected_away = 1.0 - expected_home

    if penalty_winner:
        winner = penalty_winner.strip().upper()
        if winner not in {"H", "A"}:
            raise ValueError("penalty_winner must be 'H' or 'A'.")
        actual_home = 0.75 if winner == "H" else 0.50
        actual_away = 0.75 if winner == "A" else 0.50
    elif home_goals > away_goals:
        actual_home, actual_away = 1.0, 0.0
    elif home_goals < away_goals:
        actual_home, actual_away = 0.0, 1.0
    else:
        actual_home = actual_away = 0.5

    home_change = importance * (actual_home - expected_home)
    away_change = importance * (actual_away - expected_away)
    if knockout_protection:
        home_change = max(0.0, home_change)
        away_change = max(0.0, away_change)

    return home_points + home_change, away_points + away_change


def update_live_ratings(
    ratings: pd.DataFrame,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    importance: float,
    penalty_winner: str | None = None,
    knockout_protection: bool = False,
) -> pd.DataFrame:
    frame = ratings.copy()
    frame["Country"] = frame["Country"].map(canonical_team)
    home = canonical_team(home)
    away = canonical_team(away)
    countries = set(frame["Country"])
    missing = {home, away} - countries
    if missing:
        raise ValueError(f"Teams not found in live ratings: {sorted(missing)}")

    home_points = float(frame.loc[frame["Country"] == home, "Points"].iloc[0])
    away_points = float(frame.loc[frame["Country"] == away, "Points"].iloc[0])
    updated_home, updated_away = update_fifa_points(
        home_points,
        away_points,
        home_goals,
        away_goals,
        importance,
        penalty_winner=penalty_winner,
        knockout_protection=knockout_protection,
    )
    frame.loc[frame["Country"] == home, "Points"] = updated_home
    frame.loc[frame["Country"] == away, "Points"] = updated_away
    frame = frame.sort_values(["Points", "Country"], ascending=[False, True]).reset_index(drop=True)
    frame["Rank"] = np.arange(1, len(frame) + 1)
    return frame[["Country", "Rank", "Points"]]


def load_historical_matches(
    games_path: Path,
    rankings_path: Path,
    player_data_paths: dict[int, Path] | None = None,
) -> tuple[pd.DataFrame, dict]:
    frames = []
    year_stats = {}

    for year in (2014, 2018, 2022):
        games = pd.read_excel(games_path, sheet_name=f"WorldCup{year}")
        rankings, stats = load_rankings(rankings_path, year)
        year_stats[str(year)] = stats
        rating_lookup = rankings.set_index("Country")

        games["Year"] = year
        games["Home"] = games["Home"].map(canonical_team)
        games["Away"] = games["Away"].map(canonical_team)
        games["Date"] = pd.to_datetime(games["Date"])

        missing = (set(games["Home"]) | set(games["Away"])) - set(rating_lookup.index)
        if missing:
            raise ValueError(f"Missing {year} FIFA ratings for: {sorted(missing)}")

        games["HomePoints"] = games["Home"].map(rating_lookup["Points"])
        games["AwayPoints"] = games["Away"].map(rating_lookup["Points"])
        games["HomeRatingZ"] = games["Home"].map(rating_lookup["RatingZ"])
        games["AwayRatingZ"] = games["Away"].map(rating_lookup["RatingZ"])
        games["rating_diff_z"] = games["HomeRatingZ"] - games["AwayRatingZ"]
        games["host_advantage"] = (
            games["Home"].isin(HOSTS[year]).astype(int)
            - games["Away"].isin(HOSTS[year]).astype(int)
        )
        games["knockout"] = (games["Date"] >= KNOCKOUT_START[year]).astype(int)
        games["Result"] = np.asarray(CLASSES)[result_labels(games["HGFT"], games["AGFT"])]

        market = remove_overround(games["H-Avg"], games["D-Avg"], games["A-Avg"])
        games[["MarketP_H", "MarketP_D", "MarketP_A"]] = market
        frames.append(games)

    result = pd.concat(frames, ignore_index=True)

    # --- Attach rolling form features from international_results.csv -----
    if _INTERNATIONAL_RESULTS_PATH is not None:
        result = compute_international_form(result, _INTERNATIONAL_RESULTS_PATH)

    # --- Attach pre-match xG from detailed match CSVs --------------------
    raw_dir = games_path.parent
    result = _attach_prematch_xg(result, raw_dir)

    # --- Attach player-derived team features if available ---------------
    if player_data_paths is not None:
        # Unweighted features
        team_features = load_team_features(player_data_paths)
        result = add_player_features_to_matches(result, team_features,
                                                feature_names=list(_PF_NAMES))
        # Minutes-weighted features
        tf_w = load_team_features_weighted(player_data_paths)
        result = add_player_features_to_matches(result, tf_w,
                                                feature_names=list(_PF_WEIGHTED))

    return result, year_stats


def load_qualifier_matches(
    games_path: Path,
    rankings_path: Path,
) -> pd.DataFrame | None:
    """Load World Cup qualifier data from the xlsx and attach ranking + form features.

    Uses the ``WorldCup2026Qualifiers`` sheet.  Returns ``None`` if the sheet
    is missing or empty.
    """
    try:
        qf = pd.read_excel(games_path, sheet_name="WorldCup2026Qualifiers")
    except (ValueError, FileNotFoundError):
        return None

    if len(qf) == 0:
        return None

    qf = qf.rename(columns={
        "HG": "HGFT", "AG": "AGFT",
        "H_Avg": "H-Avg", "D_Avg": "D-Avg", "A_Avg": "A-Avg",
    })
    qf["Home"] = qf["Home"].map(canonical_team)
    qf["Away"] = qf["Away"].map(canonical_team)
    qf["Date"] = pd.to_datetime(qf["Date"])

    # Determine year from date, then snap to the nearest available
    # ranking sheet (2014 / 2018 / 2022 / 2026).
    raw_year = qf["Date"].dt.year
    qf["Year"] = raw_year.map(lambda y: 2026 if y >= 2026 else 2022)

    # FIFA rankings — use the snapped year so _attach_ranking_features
    # can find the correct sheet.
    qf = _attach_ranking_features(qf, rankings_path)

    qf["host_advantage"] = 0  # most qualifiers are neutral / not host
    qf["knockout"] = 0        # qualifiers are never knockout
    qf["Result"] = np.asarray(CLASSES)[result_labels(qf["HGFT"], qf["AGFT"])]

    # Pre-match xG from qualifier sheet (HxG / AxG, ~38 % coverage)
    qf["prematch_xg_diff"] = (
        pd.to_numeric(qf.get("HxG", pd.Series(np.nan, index=qf.index)), errors="coerce")
        - pd.to_numeric(qf.get("AxG", pd.Series(np.nan, index=qf.index)), errors="coerce")
    ).fillna(0.0)

    # Clean odds — replace '-' (missing) with NaN, then drop rows without odds
    for col in ["H-Avg", "D-Avg", "A-Avg"]:
        qf[col] = pd.to_numeric(qf[col], errors="coerce")
    qf = qf.dropna(subset=["H-Avg", "D-Avg", "A-Avg"]).copy()

    market = remove_overround(qf["H-Avg"], qf["D-Avg"], qf["A-Avg"])
    qf[["MarketP_H", "MarketP_D", "MarketP_A"]] = market

    # Rolling form features
    if _INTERNATIONAL_RESULTS_PATH is not None:
        qf = compute_international_form(qf, _INTERNATIONAL_RESULTS_PATH)

    return qf


def _attach_ranking_features(
    frame: pd.DataFrame,
    rankings_path: Path,
) -> pd.DataFrame:
    """Add ``rating_diff_z`` and related columns from FIFA rankings.

    Uses the ranking workbook with sheets keyed by year.
    """
    result = frame.copy()
    # Determine the year for rankings lookup — prefer ``Year`` column, fall
    # back to a heuristic based on the date range in the data.
    years_in_frame = sorted(result["Year"].unique()) if "Year" in result.columns else []

    # We usually know the year range; pick the first (or only) year present.
    target_years = years_in_frame if years_in_frame else [2026]

    # For each distinct year present, load that sheet, merge, then recombine.
    # (Simpler case: if only one year, load once.)
    if len(target_years) == 1:
        year = target_years[0]
        rankings, stats = load_rankings(rankings_path, year)
        lookup = rankings.set_index("Country")
        missing = (set(result["Home"]) | set(result["Away"])) - set(lookup.index)
        if missing:
            print(f"  [Rank] {len(missing)} teams missing from {year} sheet, "
                  f"filling with average rating")
        mean_pts = float(rankings["Points"].mean())
        result["HomePoints"] = result["Home"].map(
            lambda t: lookup.at[t, "Points"] if t in lookup.index else mean_pts
        )
        result["AwayPoints"] = result["Away"].map(
            lambda t: lookup.at[t, "Points"] if t in lookup.index else mean_pts
        )
        result["HomeRatingZ"] = result["Home"].map(
            lambda t: lookup.at[t, "RatingZ"] if t in lookup.index else 0.0
        )
        result["AwayRatingZ"] = result["Away"].map(
            lambda t: lookup.at[t, "RatingZ"] if t in lookup.index else 0.0
        )
        result["rating_diff_z"] = result["HomeRatingZ"] - result["AwayRatingZ"]
    else:
        # Multi-year case: load per year
        frames_by_year = []
        for year, group in result.groupby("Year", sort=False):
            rankings, _ = load_rankings(rankings_path, int(year))
            lookup = rankings.set_index("Country")
            missing = (set(group["Home"]) | set(group["Away"])) - set(lookup.index)
            if missing:
                print(f"  [Rank] {len(missing)} teams missing from {year} sheet, "
                      f"filling with average rating")
            mean_pts = float(rankings["Points"].mean())
            group = group.copy()
            group["HomePoints"] = group["Home"].map(
                lambda t: lookup.at[t, "Points"] if t in lookup.index else mean_pts
            )
            group["AwayPoints"] = group["Away"].map(
                lambda t: lookup.at[t, "Points"] if t in lookup.index else mean_pts
            )
            group["HomeRatingZ"] = group["Home"].map(
                lambda t: lookup.at[t, "RatingZ"] if t in lookup.index else 0.0
            )
            group["AwayRatingZ"] = group["Away"].map(
                lambda t: lookup.at[t, "RatingZ"] if t in lookup.index else 0.0
            )
            group["rating_diff_z"] = group["HomeRatingZ"] - group["AwayRatingZ"]
            frames_by_year.append(group)
        result = pd.concat(frames_by_year, ignore_index=True)

    return result


def feature_matrix(matches: pd.DataFrame) -> np.ndarray:
    raw = matches[FEATURES].to_numpy(dtype=float)
    return np.nan_to_num(raw, nan=0.0)


def target_matrix(matches: pd.DataFrame) -> np.ndarray:
    labels = matches["Result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return one_hot(labels)


def market_target_matrix(matches: pd.DataFrame) -> np.ndarray:
    return matches[["MarketP_H", "MarketP_D", "MarketP_A"]].to_numpy(dtype=float)


def fit_models(matches: pd.DataFrame):
    """Train result model + market proxy model.

    The result model uses ``_RESULT_MODEL_FACTORY`` (Softmax by default,
    switch to XGBoost via :func:`use_xgboost_result_model`).
    The market proxy always uses Softmax (linear in odds works well).
    """
    features = feature_matrix(matches)
    result_targets = target_matrix(matches)
    market_targets = market_target_matrix(matches)
    result_x, result_y = augment_symmetry(features, result_targets)
    market_x, market_y = augment_symmetry(features, market_targets)
    factory = _RESULT_MODEL_FACTORY or SoftmaxModel
    return factory().fit(result_x, result_y), SoftmaxModel().fit(market_x, market_y)


def probability_metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict:
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    labels = actual.argmax(axis=1)
    return {
        "accuracy": float(np.mean(probabilities.argmax(axis=1) == labels)),
        "log_loss": float(-np.mean(np.sum(actual * np.log(probabilities), axis=1))),
        "brier_score": float(np.mean(np.sum((probabilities - actual) ** 2, axis=1))),
    }


def cross_validate(matches: pd.DataFrame, blend_weight: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []

    for held_out_year in sorted(matches["Year"].unique()):
        train = matches[matches["Year"] != held_out_year]
        test = matches[matches["Year"] == held_out_year].copy()
        result_model, market_model = fit_models(train)
        x_test = feature_matrix(test)
        result_probability = result_model.predict_proba(x_test)
        market_proxy_probability = market_model.predict_proba(x_test)
        blend_probability = (
            blend_weight * result_probability
            + (1.0 - blend_weight) * market_proxy_probability
        )

        for name, probability in {
            "result_model": result_probability,
            "historical_market_proxy": market_proxy_probability,
            "combined_50_50": blend_probability,
            "actual_bookmaker_market": market_target_matrix(test),
        }.items():
            output = test[["Year", "Date", "Home", "Away", "Result"]].copy()
            output["Model"] = name
            output[["P_H", "P_D", "P_A"]] = probability
            predictions.append(output)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics = []
    for (year, model), group in prediction_frame.groupby(["Year", "Model"]):
        actual = one_hot(group["Result"].map({"H": 0, "D": 1, "A": 2}).to_numpy())
        values = probability_metrics(actual, group[["P_H", "P_D", "P_A"]].to_numpy())
        metrics.append({"test_year": int(year), "model": model, **values})

    for model, group in prediction_frame.groupby("Model"):
        actual = one_hot(group["Result"].map({"H": 0, "D": 1, "A": 2}).to_numpy())
        values = probability_metrics(actual, group[["P_H", "P_D", "P_A"]].to_numpy())
        metrics.append({"test_year": "ALL", "model": model, **values})

    return prediction_frame, pd.DataFrame(metrics)


def save_model_bundle(
    path: Path,
    result_model,
    market_model,
    rating_stats: dict,
    blend_weight: float = 0.5,
) -> None:
    payload = {
        "method": "within-tournament z-score + multinomial softmax",
        "blend_weight_result_model": blend_weight,
        "result_model": result_model.to_dict(),
        "market_proxy_model": market_model.to_dict(),
        "rating_stats": rating_stats,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _reconstruct_model(model_dict: dict):
    """Reconstruct a model from its serialized dict, handling different types."""
    model_type = model_dict.get("model_type", "softmax")
    if model_type == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("XGBoost model in bundle but xgboost is not installed.")
        return XGBoostModel.from_dict(model_dict)
    return SoftmaxModel.from_dict(model_dict)


def load_model_bundle(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result_model"] = _reconstruct_model(payload["result_model"])
    payload["market_proxy_model"] = _reconstruct_model(payload["market_proxy_model"])
    return payload


def predict_fixtures(
    fixtures: pd.DataFrame,
    rankings_path: Path,
    bundle: dict,
    year: int = 2026,
    team_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame, result_probability, market_probability = predict_fixture_components(
        fixtures,
        rankings_path,
        bundle,
        year=year,
        team_features=team_features,
    )
    weight = float(bundle["blend_weight_result_model"])
    probability = weight * result_probability + (1 - weight) * market_probability

    frame[["P_H", "P_D", "P_A"]] = probability
    frame["FairOdds_H"] = 1.0 / frame["P_H"]
    frame["FairOdds_D"] = 1.0 / frame["P_D"]
    frame["FairOdds_A"] = 1.0 / frame["P_A"]
    return frame


def predict_fixture_components(
    fixtures: pd.DataFrame,
    rankings_path: Path,
    bundle: dict,
    year: int = 2026,
    team_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rankings, stats = load_rankings(rankings_path, year)
    lookup = rankings.set_index("Country")
    frame = fixtures.copy()
    frame["Home"] = frame["Home"].map(canonical_team)
    frame["Away"] = frame["Away"].map(canonical_team)
    missing = (set(frame["Home"]) | set(frame["Away"])) - set(lookup.index)
    if missing:
        raise ValueError(f"Missing {year} FIFA ratings for: {sorted(missing)}")

    frame["HomePoints"] = frame["Home"].map(lookup["Points"])
    frame["AwayPoints"] = frame["Away"].map(lookup["Points"])
    frame["HomeRatingZ"] = (frame["HomePoints"] - stats["mean"]) / stats["std"]
    frame["AwayRatingZ"] = (frame["AwayPoints"] - stats["mean"]) / stats["std"]
    frame["rating_diff_z"] = frame["HomeRatingZ"] - frame["AwayRatingZ"]
    frame["host_advantage"] = pd.to_numeric(
        frame.get("HostAdvantage", pd.Series(0, index=frame.index)), errors="coerce"
    ).fillna(0)
    stage = frame.get("Stage", pd.Series("Group", index=frame.index)).astype(str)
    frame["knockout"] = (~stage.str.lower().eq("group")).astype(int)

    # --- Attach rolling form features from international_results.csv ---
    if _INTERNATIONAL_RESULTS_PATH is not None:
        # Ensure Date column exists (DateUTC → Date for compute_international_form)
        if "Date" not in frame.columns and "DateUTC" in frame.columns:
            frame["Date"] = pd.to_datetime(frame["DateUTC"])
            if hasattr(frame["Date"].dtype, "tz") and frame["Date"].dtype.tz is not None:
                frame["Date"] = frame["Date"].dt.tz_localize(None)
        frame = compute_international_form(frame, _INTERNATIONAL_RESULTS_PATH)

    # Pre-match xG is not available for future fixtures -- zero-fill
    frame["prematch_xg_diff"] = 0.0

    # Attach player-derived features — always zero-fill all, then compute
    # the ones supplied.
    from src.player_features import player_feature_names_weighted

    for pf in player_feature_names() + player_feature_names_weighted():
        frame[pf] = 0.0

    if team_features is not None:
        # Determine which feature set the supplied DataFrame contains by
        # checking a representative column name.
        has_weighted = any(c.endswith("_w") for c in team_features.columns)
        fnames = (
            list(_PF_NAMES) + list(_PF_WEIGHTED)
            if has_weighted
            else list(_PF_NAMES)
        )
        frame["Year"] = year
        frame = add_player_features_to_matches(
            frame, team_features, feature_names=fnames
        )
        frame = frame.drop(columns="Year")

    features = feature_matrix(frame)
    result_probability = bundle["result_model"].predict_proba(features)
    market_probability = bundle["market_proxy_model"].predict_proba(features)
    return frame, result_probability, market_probability
