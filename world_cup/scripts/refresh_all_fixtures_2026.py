"""Fetch all 2026 World Cup fixtures (group + knockout) from the FIFA API.

Knockout PlaceHolder legend (48-team format):
  1A, 2A  = 1st/2nd from Group A
  3ABCD   = 3rd-place qualifier from groups A,B,C,D
  W73     = Winner of match #73
  RU101   = Runner-up of match #101
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent  # world_cup/scripts/
WC_DIR = THIS_DIR.parent  # world_cup/
PROJECT_ROOT = WC_DIR.parent  # FYP/
sys.path.insert(0, str(WC_DIR))

from src.world_cup_model import HOSTS, canonical_team

API_URL = (
    "https://api.fifa.com/api/v3/calendar/matches"
    "?idCompetition=17&idSeason=285023"
    "&from=2026-06-01&to=2026-07-20&language=en&count=200"
)
OUTPUT_PATH = WC_DIR / "data" / "fixtures_2026_all.csv"

# Stages that are already played and do not need predictions
STAGES_TO_SKIP = {"First Stage", "Round of 32"}
# For Round of 32, we still include unplayed matches but skip already-played ones


def description(values: list[dict] | None) -> str:
    return values[0]["Description"] if values else ""


def team_name(team: dict) -> str:
    return canonical_team(description(team.get("TeamName")))


def stage_label(stage_name: str, match_number: int) -> str:
    """Normalise FIFA stage names into our convention."""
    name = stage_name.strip()
    if name == "First Stage":
        return "Group"
    if name == "Round of 32":
        return "Round32"
    if name == "Round of 16":
        return "Round16"
    if name in ("Quarter-final", "Quarter-Final"):
        return "QuarterFinal"
    if name == "Semi-final":
        return "SemiFinal"
    if "third place" in name.lower() or "play-off" in name.lower():
        return "ThirdPlace"
    if name == "Final":
        return "Final"
    return name.replace(" ", "")


def is_match_played(match: dict) -> bool:
    hs = match.get("HomeTeamScore")
    aws = match.get("AwayTeamScore")
    status = match.get("MatchStatus")
    # Status 0 = finished, or scores are present
    if hs is not None and aws is not None:
        return True
    if status == 0:
        return True
    return False


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    request = Request(API_URL, headers={"User-Agent": "WorldCupFYP/1.0"})
    with urlopen(request, timeout=30) as response:
        matches = json.load(response)["Results"]

    hosts = HOSTS[2026]
    rows = []
    all_predictable = []  # Only matches we want to predict (upcoming, non-TBD)

    for match in matches:
        home = team_name(match["Home"]) if match.get("Home") else "TBD"
        away = team_name(match["Away"]) if match.get("Away") else "TBD"
        stage_desc = description(match.get("StageName"))
        stage = stage_label(stage_desc, int(match["MatchNumber"]))
        group = description(match.get("GroupName"))
        match_number = int(match["MatchNumber"])
        played = is_match_played(match)

        row = {
            "MatchNumber": match_number,
            "DateUTC": match.get("Date", ""),
            "Stage": stage,
            "StageDetail": stage_desc,
            "Group": group,
            "Home": home,
            "Away": away,
            "HostAdvantage": int(home in hosts) - int(away in hosts),
            "HomeTeamScore": match.get("HomeTeamScore"),
            "AwayTeamScore": match.get("AwayTeamScore"),
            "Played": played,
            "PlaceHolderA": match.get("PlaceHolderA", ""),
            "PlaceHolderB": match.get("PlaceHolderB", ""),
            "Winner": match.get("Winner", ""),
        }
        rows.append(row)

        # Collect predictable matches: upcoming (not played) with known teams
        if not played and "TBD" not in (home, away):
            all_predictable.append(row)

    fixtures = pd.DataFrame(rows).sort_values("MatchNumber").reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {len(fixtures)} total matches to: {OUTPUT_PATH}\n")

    # --- Summary ---
    for stage_name in ["Group", "Round32", "Round16", "QuarterFinal", "SemiFinal", "ThirdPlace", "Final"]:
        sub = fixtures[fixtures["Stage"] == stage_name]
        if len(sub) == 0:
            continue
        played = sub["Played"].sum()
        print(f"{stage_name}: {int(played)}/{len(sub)} played")

    print(f"\n--- Predictable upcoming matches ({len(all_predictable)}) ---")
    predict_df = pd.DataFrame(all_predictable)
    if len(predict_df):
        predict_df["DateChina"] = (
            pd.to_datetime(predict_df["DateUTC"], utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.strftime("%Y-%m-%d %H:%M")
        )
        for _, row in predict_df.iterrows():
            print(f"  #{row['MatchNumber']:3d} [{row['Stage']:12s}] {row['Home']:20s} vs {row['Away']:20s}  ({row['DateChina']})")

    # --- Export prediction-ready CSV ---
    # Only columns expected by predict_fixtures(): Home, Away, Stage, HostAdvantage
    pred_cols = ["MatchNumber", "DateUTC", "Home", "Away", "Stage", "HostAdvantage"]
    predict_out = WC_DIR / "data" / "fixtures_2026_predict.csv"
    predict_df[pred_cols].to_csv(predict_out, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(predict_df)} predictable fixtures to: {predict_out}")


if __name__ == "__main__":
    main()
