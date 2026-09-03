from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from src.world_cup_model import canonical_team


INTERNATIONAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)
DEFAULT_START_DATE = pd.Timestamp("2022-12-19")
DEFAULT_CUTOFF_DATE = pd.Timestamp("2026-06-10")
DEFAULT_HALF_LIFE_DAYS = 540.0

TEAM_ALIASES = {
    "Brunei Darussalam": "Brunei",
    "Central African Republic": "Central Africa",
    "China PR": "China",
    "Côte d'Ivoire": "Ivory Coast",
    "Curaçao": "Curacao",
    "Democratic Republic of the Congo": "DR Congo",
    "Guinea-Bissau": "Guinea Bissau",
    "Korea DPR": "North Korea",
    "Korea Republic": "South Korea",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Republic of Ireland": "Ireland",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "Taiwan": "Chinese Taipei",
    "Trinidad and Tobago": "Trinidad & Tobago",
    "Türkiye": "Turkey",
    "United States": "USA",
    "Vietnam": "Vietnam",
}

EXCLUDED_TOURNAMENT_PATTERNS = (
    r"\bU[- ]?(?:15|16|17|18|19|20|21|22|23)\b",
    r"\bunder[- ]?(?:15|16|17|18|19|20|21|22|23)\b",
    r"\bOlympic\b",
    r"\bAfrican Nations Championship\b",
    r"\bCHAN\b",
    r"\bnon[- ]FIFA\b",
    r"\bunofficial\b",
)

# Official tables record these as administrative 3-0 results. Goal models need
# the score produced on the pitch; a match that was never played is excluded.
KNOWN_RESULT_ADJUSTMENTS = {
    ("2023-11-15", "Equatorial Guinea", "Namibia"): (
        1,
        0,
        "FIFA forfeit; restored played score",
    ),
    ("2023-11-20", "Liberia", "Equatorial Guinea"): (
        0,
        1,
        "FIFA forfeit; restored played score",
    ),
    ("2025-03-21", "South Africa", "Lesotho"): (
        2,
        0,
        "FIFA forfeit; restored played score",
    ),
    ("2025-10-09", "Malawi", "Equatorial Guinea"): (
        None,
        None,
        "Match not played; excluded from goal model",
    ),
}


@dataclass(frozen=True)
class PreparationConfig:
    start_date: pd.Timestamp = DEFAULT_START_DATE
    cutoff_date: pd.Timestamp = DEFAULT_CUTOFF_DATE
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS

    def validate(self) -> None:
        if self.start_date > self.cutoff_date:
            raise ValueError("start_date must not be after cutoff_date.")
        if self.half_life_days <= 0:
            raise ValueError("half_life_days must be positive.")


def standardize_team(name: str) -> str:
    cleaned = canonical_team(str(name).strip())
    return TEAM_ALIASES.get(cleaned, cleaned)


def download_international_results(
    destination: Path,
    url: str = INTERNATIONAL_RESULTS_URL,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "WorldCupFYP/1.0"})
    with urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())
    return destination


def competition_category(tournament: str) -> str:
    value = str(tournament).strip().lower()
    continental_tokens = (
        "afc asian cup",
        "asian cup",
        "africa cup of nations",
        "african cup of nations",
        "uefa euro",
        "gold cup",
        "oceania nations cup",
        "cafa nations cup",
        "copa américa",
        "copa america",
    )
    if "world cup" in value and ("qualification" in value or "qualifier" in value):
        return "world_cup_qualification"
    if "world cup" in value:
        return "world_cup_finals"
    if "nations league" in value:
        return "nations_league"
    if "qualification" in value or "qualifier" in value:
        if any(token in value for token in continental_tokens):
            return "continental_qualification"
    if any(token in value for token in continental_tokens):
        return "continental_finals"
    if "friendly" in value:
        return "friendly"
    return "other_senior"


def competition_weight(category: str) -> float:
    return {
        "world_cup_qualification": 1.00,
        "world_cup_finals": 1.00,
        "continental_finals": 0.95,
        "continental_qualification": 0.85,
        "nations_league": 0.75,
        "friendly": 0.35,
        "other_senior": 0.25,
    }[category]


def is_excluded_tournament(tournament: str) -> bool:
    value = str(tournament)
    return any(
        re.search(pattern, value, flags=re.IGNORECASE)
        for pattern in EXCLUDED_TOURNAMENT_PATTERNS
    )


def load_team_universe(
    qualifiers_path: Path,
    rankings_path: Path,
) -> set[str]:
    qualifiers = pd.read_excel(qualifiers_path, sheet_name="WorldCup2026Qualifiers")
    rankings = pd.read_excel(rankings_path, sheet_name="2026")
    names = (
        qualifiers["Home"].tolist()
        + qualifiers["Away"].tolist()
        + rankings["Country"].tolist()
    )
    return {standardize_team(name) for name in names}


def load_core_qualifiers(path: Path) -> pd.DataFrame:
    source = pd.read_excel(path, sheet_name="WorldCup2026Qualifiers")
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(source["Date"]).dt.normalize(),
            "Home": source["Home"].map(standardize_team),
            "Away": source["Away"].map(standardize_team),
            "HomeGoals": pd.to_numeric(source["HG"], errors="coerce"),
            "AwayGoals": pd.to_numeric(source["AG"], errors="coerce"),
            "Tournament": "FIFA World Cup qualification",
            "City": pd.NA,
            "Country": pd.NA,
            "Neutral": pd.NA,
            "Source": "core_2026_qualifiers",
            "SourcePriority": 0,
            "ScoreAdjusted": False,
            "AdjustmentReason": pd.NA,
            "ExcludeFromGoalModel": False,
        }
    )
    return frame


def load_external_results(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"External results file is missing columns: {sorted(missing)}")

    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(source["date"], errors="coerce").dt.normalize(),
            "Home": source["home_team"].map(standardize_team),
            "Away": source["away_team"].map(standardize_team),
            "HomeGoals": pd.to_numeric(source["home_score"], errors="coerce"),
            "AwayGoals": pd.to_numeric(source["away_score"], errors="coerce"),
            "Tournament": source["tournament"].astype(str),
            "City": source["city"],
            "Country": source["country"],
            "Neutral": source["neutral"].astype("boolean"),
            "Source": "international_results",
            "SourcePriority": 1,
            "ScoreAdjusted": False,
            "AdjustmentReason": pd.NA,
            "ExcludeFromGoalModel": False,
        }
    )
    return apply_known_result_adjustments(frame)


def apply_known_result_adjustments(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "ScoreAdjusted" not in frame:
        frame["ScoreAdjusted"] = False
    if "AdjustmentReason" not in frame:
        frame["AdjustmentReason"] = pd.NA
    if "ExcludeFromGoalModel" not in frame:
        frame["ExcludeFromGoalModel"] = False
    for index, row in frame.iterrows():
        if pd.isna(row["Date"]):
            continue
        key = (
            row["Date"].strftime("%Y-%m-%d"),
            row["Home"],
            row["Away"],
        )
        adjustment = KNOWN_RESULT_ADJUSTMENTS.get(key)
        if adjustment is None:
            continue
        home_goals, away_goals, reason = adjustment
        frame.at[index, "ScoreAdjusted"] = True
        frame.at[index, "AdjustmentReason"] = reason
        if home_goals is None:
            frame.at[index, "ExcludeFromGoalModel"] = True
        else:
            frame.at[index, "HomeGoals"] = home_goals
            frame.at[index, "AwayGoals"] = away_goals
    return frame


def match_key(frame: pd.DataFrame) -> pd.Series:
    first = frame[["Home", "Away"]].min(axis=1)
    second = frame[["Home", "Away"]].max(axis=1)
    return (
        frame["Date"].dt.strftime("%Y-%m-%d")
        + "|"
        + first.astype(str)
        + "|"
        + second.astype(str)
    )


def result_key(frame: pd.DataFrame) -> pd.Series:
    home_first = frame["Home"] <= frame["Away"]
    first_team = frame["Home"].where(home_first, frame["Away"])
    second_team = frame["Away"].where(home_first, frame["Home"])
    first_goals = pd.to_numeric(
        frame["HomeGoals"].where(home_first, frame["AwayGoals"]),
        errors="coerce",
    ).astype("Int64")
    second_goals = pd.to_numeric(
        frame["AwayGoals"].where(home_first, frame["HomeGoals"]),
        errors="coerce",
    ).astype("Int64")
    return (
        first_team.astype(str)
        + "|"
        + second_team.astype(str)
        + "|"
        + first_goals.astype(str)
        + "|"
        + second_goals.astype(str)
    )


def core_duplicate_indices(
    core: pd.DataFrame,
    external: pd.DataFrame,
    tolerance_days: int = 1,
) -> set[int]:
    core_keys = result_key(core)
    external_keys = result_key(external)
    external_groups: dict[str, list[int]] = {}
    for index, key in external_keys.items():
        external_groups.setdefault(key, []).append(index)

    duplicates: set[int] = set()
    for core_index, key in core_keys.items():
        candidates = [
            index
            for index in external_groups.get(key, [])
            if index not in duplicates
            and abs((external.at[index, "Date"] - core.at[core_index, "Date"]).days)
            <= tolerance_days
        ]
        if candidates:
            nearest = min(
                candidates,
                key=lambda index: abs(
                    (external.at[index, "Date"] - core.at[core_index, "Date"]).days
                ),
            )
            duplicates.add(nearest)
    return duplicates


def enrich_core_metadata(
    core: pd.DataFrame,
    external: pd.DataFrame,
) -> pd.DataFrame:
    core = core.copy()
    external = external.copy()
    core["_ResultKey"] = result_key(core)
    external["_ResultKey"] = result_key(external)
    external_groups = {
        key: group for key, group in external.groupby("_ResultKey", sort=False)
    }
    for core_index, row in core.iterrows():
        candidates = external_groups.get(row["_ResultKey"])
        if candidates is None:
            continue
        date_distance = (candidates["Date"] - row["Date"]).abs().dt.days
        candidates = candidates[date_distance <= 1]
        if candidates.empty:
            continue
        nearest_index = date_distance.loc[candidates.index].idxmin()
        for column in ("City", "Country", "Neutral"):
            if pd.isna(core.at[core_index, column]):
                core.at[core_index, column] = external.at[nearest_index, column]
    return core.drop(columns="_ResultKey")


def filter_external_results(
    frame: pd.DataFrame,
    team_universe: set[str],
    config: PreparationConfig,
) -> tuple[pd.DataFrame, dict]:
    config.validate()
    frame = frame.copy()
    if "ScoreAdjusted" not in frame:
        frame["ScoreAdjusted"] = False
    if "ExcludeFromGoalModel" not in frame:
        frame["ExcludeFromGoalModel"] = False
    audit = {"raw_external_rows": int(len(frame))}
    audit["known_score_adjustments"] = int(frame["ScoreAdjusted"].sum())
    audit["known_unplayed_matches"] = int(frame["ExcludeFromGoalModel"].sum())
    frame = frame[~frame["ExcludeFromGoalModel"]].copy()
    valid = frame.dropna(
        subset=["Date", "Home", "Away", "HomeGoals", "AwayGoals", "Tournament"]
    ).copy()
    audit["after_required_fields"] = int(len(valid))

    valid = valid[
        valid["Date"].between(config.start_date, config.cutoff_date, inclusive="both")
    ].copy()
    audit["after_date_window"] = int(len(valid))

    valid = valid[
        valid["Home"].isin(team_universe) & valid["Away"].isin(team_universe)
    ].copy()
    audit["after_current_team_universe"] = int(len(valid))

    valid = valid[~valid["Tournament"].map(is_excluded_tournament)].copy()
    audit["after_senior_competition_filter"] = int(len(valid))

    valid = valid[
        (valid["HomeGoals"] >= 0)
        & (valid["AwayGoals"] >= 0)
        & (valid["HomeGoals"] % 1 == 0)
        & (valid["AwayGoals"] % 1 == 0)
    ].copy()
    audit["after_score_validation"] = int(len(valid))
    return valid, audit


def add_training_weights(
    frame: pd.DataFrame,
    config: PreparationConfig,
) -> pd.DataFrame:
    output = frame.copy()
    output["CompetitionCategory"] = output["Tournament"].map(competition_category)
    output["CompetitionWeight"] = output["CompetitionCategory"].map(competition_weight)
    age_days = (config.cutoff_date - output["Date"]).dt.days.clip(lower=0)
    output["AgeDays"] = age_days.astype(int)
    output["TimeWeight"] = 0.5 ** (output["AgeDays"] / config.half_life_days)
    output["FinalWeight"] = output["CompetitionWeight"] * output["TimeWeight"]
    return output


def prepare_training_data(
    qualifiers_path: Path,
    rankings_path: Path,
    external_path: Path,
    config: PreparationConfig,
) -> tuple[pd.DataFrame, dict]:
    team_universe = load_team_universe(qualifiers_path, rankings_path)
    core = load_core_qualifiers(qualifiers_path)
    external_raw = load_external_results(external_path)
    external, audit = filter_external_results(external_raw, team_universe, config)

    core = core[
        core["Date"].between(config.start_date, config.cutoff_date, inclusive="both")
    ].copy()
    core = enrich_core_metadata(core, external)
    duplicate_external_indices = core_duplicate_indices(core, external)
    external = external.drop(index=duplicate_external_indices)
    combined = pd.concat([core, external], ignore_index=True)
    combined["_MatchKey"] = match_key(combined)
    before_deduplication = len(combined)
    combined = (
        combined.sort_values(["SourcePriority", "Date"])
        .drop_duplicates("_MatchKey", keep="first")
        .drop(columns="_MatchKey")
        .sort_values(["Date", "Home", "Away"])
        .reset_index(drop=True)
    )

    combined["HomeGoals"] = combined["HomeGoals"].astype(int)
    combined["AwayGoals"] = combined["AwayGoals"].astype(int)
    combined["ScoreAdjusted"] = combined["ScoreAdjusted"].astype(bool)
    combined["ExcludeFromGoalModel"] = combined["ExcludeFromGoalModel"].astype(bool)
    combined["Neutral"] = combined["Neutral"].astype("boolean")
    combined = add_training_weights(combined, config)
    combined["NeutralKnown"] = combined["Neutral"].notna().astype(int)
    combined["TrueHome"] = (~combined["Neutral"]).astype("Int64")

    audit.update(
        {
            "start_date": config.start_date.strftime("%Y-%m-%d"),
            "cutoff_date": config.cutoff_date.strftime("%Y-%m-%d"),
            "half_life_days": config.half_life_days,
            "team_universe_size": len(team_universe),
            "core_rows_in_window": int(len(core)),
            "cross_source_duplicates_removed": int(len(duplicate_external_indices)),
            "combined_rows_before_deduplication": int(before_deduplication),
            "duplicates_removed": int(before_deduplication - len(combined)),
            "final_training_rows": int(len(combined)),
            "final_team_count": int(
                len(set(combined["Home"]) | set(combined["Away"]))
            ),
            "neutral_known_rows": int(combined["NeutralKnown"].sum()),
            "competition_categories": {
                str(key): int(value)
                for key, value in combined["CompetitionCategory"]
                .value_counts()
                .sort_index()
                .items()
            },
            "source_rows": {
                str(key): int(value)
                for key, value in combined["Source"].value_counts().items()
            },
        }
    )
    return combined, audit


def write_preparation_outputs(
    frame: pd.DataFrame,
    audit: dict,
    output_path: Path,
    audit_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
