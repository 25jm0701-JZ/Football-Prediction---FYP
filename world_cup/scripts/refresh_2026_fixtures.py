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
OUTPUT_PATH = WC_DIR / "data" / "fixtures_2026_example.csv"


def description(values: list[dict] | None) -> str:
    return values[0]["Description"] if values else ""


def team_name(team: dict) -> str:
    return canonical_team(description(team.get("TeamName")))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    request = Request(API_URL, headers={"User-Agent": "WorldCupFYP/1.0"})
    with urlopen(request, timeout=30) as response:
        matches = json.load(response)["Results"]

    rows = []
    hosts = HOSTS[2026]
    for match in matches:
        if description(match.get("StageName")) != "First Stage":
            continue
        home = team_name(match["Home"])
        away = team_name(match["Away"])
        rows.append(
            {
                "MatchNumber": int(match["MatchNumber"]),
                "DateUTC": match["Date"],
                "Group": description(match.get("GroupName")),
                "Home": home,
                "Away": away,
                "Stage": "Group",
                "HostAdvantage": int(home in hosts) - int(away in hosts),
                "Stadium": description(match.get("Stadium", {}).get("Name")),
                "City": description(match.get("Stadium", {}).get("CityName")),
            }
        )

    fixtures = pd.DataFrame(rows).sort_values("MatchNumber").reset_index(drop=True)
    if len(fixtures) != 72:
        raise ValueError(f"Expected 72 group matches, received {len(fixtures)}.")
    if fixtures["Home"].eq("").any() or fixtures["Away"].eq("").any():
        raise ValueError("At least one group-stage team is missing.")

    fixtures["GroupRound"] = (
        fixtures.sort_values(["Group", "DateUTC", "MatchNumber"])
        .groupby("Group")
        .cumcount()
        .floordiv(2)
        .add(1)
        .sort_index()
    )
    fixtures["DateChina"] = (
        pd.to_datetime(fixtures["DateUTC"], utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.strftime("%Y-%m-%d %H:%M")
    )
    columns = [
        "MatchNumber",
        "DateUTC",
        "DateChina",
        "Group",
        "GroupRound",
        "Home",
        "Away",
        "Stage",
        "HostAdvantage",
        "Stadium",
        "City",
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixtures[columns].to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(fixtures[columns[:8]].to_string(index=False))
    print(f"\nSaved {len(fixtures)} group-stage fixtures to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
