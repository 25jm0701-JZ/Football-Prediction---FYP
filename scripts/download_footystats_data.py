#!/usr/bin/env python3
"""
Download all 7 seasons of Premier League data from FootyStats API.
Stores raw JSON + processed CSV for quick loading.

Seasons: 2018/19 → 2024/25, ~2,660 matches + ~4,600 player records
"""

import json
import time
from pathlib import Path

import requests

API_KEY = "test85g57"
BASE_URL = "https://api.football-data-api.com"

# ── Config ─────────────────────────────────────────────────────────
SEASONS = {
    1625: "2018-19",
    2012: "2019-20",
    4759: "2020-21",
    6135: "2021-22",
    7704: "2022-23",
    9660: "2023-24",
    12325: "2024-25",
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "footystats"
RAW_DIR = OUTPUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

REQUESTS_REMAINING = 180  # track rate limit


def api_get(endpoint: str, **params) -> dict:
    """Make API request with rate limit tracking."""
    global REQUESTS_REMAINING
    params["key"] = API_KEY
    r = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=30)

    # Track remaining requests
    meta = r.json().get("metadata", {})
    if "request_remaining" in meta:
        REQUESTS_REMAINING = int(meta["request_remaining"])

    if r.status_code != 200:
        print(f"  WARNING: {endpoint} returned {r.status_code}: {r.text[:200]}")

    return r.json()


def download_season_matches(season_id: int, label: str):
    """Download all matches for a season."""
    print(f"\n[{label}] Downloading matches...")
    data = api_get("/league-matches", season_id=season_id)
    matches = data.get("data", [])

    if not matches:
        print(f"  No matches returned!")
        return None

    # Save raw JSON
    path = RAW_DIR / f"matches_{season_id}.json"
    path.write_text(json.dumps(matches, indent=2), encoding="utf-8")
    print(f"  {len(matches)} matches → {path.name}")
    print(f"  Remaining API calls: {REQUESTS_REMAINING}")
    return matches


def download_season_players(season_id: int, label: str):
    """Download all players for a season (handles pagination)."""
    print(f"[{label}] Downloading players...")
    all_players = []
    page = 1

    while True:
        data = api_get("/league-players", season_id=season_id, page=page)
        players = data.get("data", [])
        if not players:
            break
        all_players.extend(players)
        print(f"  Page {page}: {len(players)} players (total: {len(all_players)})")

        # Check if more pages
        pager = data.get("pager", {})
        if page >= pager.get("max_page", 1):
            break
        page += 1
        time.sleep(0.5)  # small delay between pages

    if all_players:
        path = RAW_DIR / f"players_{season_id}.json"
        path.write_text(json.dumps(all_players, indent=2), encoding="utf-8")
        print(f"  Total: {len(all_players)} players → {path.name}")
    else:
        print(f"  No players returned!")

    print(f"  Remaining API calls: {REQUESTS_REMAINING}")
    return all_players


def main():
    print("=" * 60)
    print(" FootyStats Data Downloader")
    print(f" {len(SEASONS)} seasons of Premier League data")
    print("=" * 60)

    summary = []

    for season_id, label in sorted(SEASONS.items(), key=lambda x: x[1]):
        matches = download_season_matches(season_id, label)
        players = download_season_players(season_id, label)

        summary.append({
            "season": label,
            "season_id": season_id,
            "matches": len(matches) if matches else 0,
            "players": len(players) if players else 0,
        })

        # Rate limit safety: if < 20 remaining, wait for refresh
        if REQUESTS_REMAINING < 20:
            print(f"\n  Rate limit low ({REQUESTS_REMAINING}), waiting 60s...")
            time.sleep(60)

    print("\n" + "=" * 60)
    print(" Download Summary:")
    print("=" * 60)
    total_m = sum(s["matches"] for s in summary)
    total_p = sum(s["players"] for s in summary)
    for s in summary:
        print(f"  {s['season']}: {s['matches']:>3} matches, {s['players']:>3} players")
    print(f"  {'TOTAL':<9}: {total_m} matches, {total_p} players")
    print(f"\n  Saved to: {RAW_DIR}")
    print(f"  API calls remaining: {REQUESTS_REMAINING}")


if __name__ == "__main__":
    main()
