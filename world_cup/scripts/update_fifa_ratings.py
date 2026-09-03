import argparse
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent  # world_cup/scripts/
WC_DIR = THIS_DIR.parent  # world_cup/
PROJECT_ROOT = WC_DIR.parent  # FYP/
sys.path.insert(0, str(WC_DIR))

from src.world_cup_model import canonical_team, update_live_ratings


def main() -> None:
    parser = argparse.ArgumentParser(description="Update live ratings after a real match.")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--home-goals", type=int, required=True)
    parser.add_argument("--away-goals", type=int, required=True)
    parser.add_argument(
        "--importance",
        type=float,
        default=50,
        help="FIFA match importance: World Cup before quarter-finals=50, quarter-finals onward=60.",
    )
    parser.add_argument(
        "--penalty-winner",
        choices=["H", "A"],
        help="Use only when the match is decided by a penalty shoot-out.",
    )
    parser.add_argument(
        "--knockout-protection",
        action="store_true",
        help="Apply FIFA final-competition protection against negative points.",
    )
    parser.add_argument(
        "--ratings",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "world_cup" / "ratings_2026_live.csv",
    )
    args = parser.parse_args()

    ratings = pd.read_csv(args.ratings)
    before = ratings.set_index("Country")["Points"].to_dict()
    updated = update_live_ratings(
        ratings,
        args.home,
        args.away,
        args.home_goals,
        args.away_goals,
        args.importance,
        penalty_winner=args.penalty_winner,
        knockout_protection=args.knockout_protection,
    )
    updated.to_csv(args.ratings, index=False, encoding="utf-8-sig")
    after = updated.set_index("Country")["Points"].to_dict()

    home = canonical_team(args.home)
    away = canonical_team(args.away)
    print(f"{home}: {before[home]:.2f} -> {after[home]:.2f}")
    print(f"{away}: {before[away]:.2f} -> {after[away]:.2f}")
    print(f"Updated: {args.ratings}")


if __name__ == "__main__":
    main()
