from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent  # world_cup/scripts/
WC_DIR = THIS_DIR.parent  # world_cup/
PROJECT_ROOT = WC_DIR.parent  # FYP/
sys.path.insert(0, str(WC_DIR))

from src.world_cup_model import canonical_team


API_URL = (
    "https://api.fifa.com/api/v3/fifarankings/rankings/live"
    "?gender=1&sportType=0&count=300&language=en"
)
RANKINGS_WORKBOOK = WC_DIR / "world_cup_rankings_4_sheets_simple.xlsx"
SNAPSHOT_PATH = WC_DIR / "data" / "raw" / "fifa_men_live_2026.csv"
LIVE_PATH = PROJECT_ROOT / "outputs" / "world_cup" / "ratings_2026_live.csv"
METADATA_PATH = WC_DIR / "data" / "raw" / "fifa_men_live_2026_metadata.json"


def fetch_live_rankings() -> pd.DataFrame:
    request = Request(API_URL, headers={"User-Agent": "WorldCupFYP/1.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)

    rows = []
    for item in payload["Results"]:
        names = item.get("TeamName") or []
        name = names[0]["Description"] if names else item["IdCountry"]
        rows.append(
            {
                "Country": canonical_team(name),
                "Rank": int(item["Rank"]),
                "Points": float(item["TotalPoints"]),
                "PreviousRank": int(item["PrevRank"]),
                "PreviousPoints": float(item["PrevPoints"]),
                "CountryCode": item["IdCountry"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    tournament = pd.read_excel(RANKINGS_WORKBOOK, sheet_name="2026")
    tournament["Country"] = tournament["Country"].map(canonical_team)
    live = fetch_live_rankings()
    selected = tournament[["Country"]].merge(live, on="Country", how="left")

    missing = selected.loc[selected["Points"].isna(), "Country"].tolist()
    if missing:
        raise ValueError(f"FIFA live API did not match these tournament teams: {missing}")

    selected = selected.sort_values("Rank").reset_index(drop=True)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(SNAPSHOT_PATH, index=False, encoding="utf-8-sig", float_format="%.6f")
    selected[["Country", "Rank", "Points"]].to_csv(
        LIVE_PATH, index=False, encoding="utf-8-sig", float_format="%.6f"
    )

    metadata = {
        "source": API_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FIFA live/provisional data; not a published ranking schedule",
        "latest_published_official_ranking": "2026-04-01",
        "team_count": len(selected),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(selected[["Country", "Rank", "Points"]].to_string(index=False))
    print(f"\nSaved live snapshot: {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
