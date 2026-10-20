"""Draw every graphic for a block of rounds: one folder, clubs and players apart.

    python scripts/aggregate_rounds.py           # first: the tables
    python scripts/league_visuals.py             # then: Premier League, rounds 1-5
    python scripts/league_visuals.py --scope players --only goalkeepers xg
    python scripts/league_visuals.py --no-profiles

Reads output/aggregates/<league>_<season>_R<a>-<b>/ and rebuilds its ``visuals`` folder, which
holds exactly two folders, ``teams`` and ``players``. In each: a ranking for every measure
(``attack_``, ``defence_``, ``pressing_``, ``passing_``), the comparisons that set two measures
against each other, the league heat table and goalkeepers, and a ``profile_`` card for each club
and each player over the minutes floor. Every image is 16:9 (1600 x 900).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_analysis.paths import OUTPUT_DIR  # noqa: E402
from football_analysis.visuals import league_boards as boards  # noqa: E402
from football_analysis.visuals import social_cards  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--league", default="England_Premier_League")
    parser.add_argument("--season", default="2026-2027")
    parser.add_argument("--rounds", default="1-5")
    parser.add_argument("--scope", choices=["teams", "players"], help="only clubs or only players")
    parser.add_argument(
        "--only", nargs="*", help="draw cards whose file name contains any of these words"
    )
    parser.add_argument(
        "--no-profiles", action="store_true", help="skip the club and player profile cards"
    )
    args = parser.parse_args(argv)

    folder = OUTPUT_DIR / "aggregates" / f"{args.league}_{args.season}_R{args.rounds}"
    if not (folder / "teams_totals.csv").exists():
        print(f"No tables at {folder}; run scripts/aggregate_rounds.py first")
        return 1
    league = boards.League.load(folder, packages=OUTPUT_DIR / args.league / args.season)
    visuals = folder / "visuals"
    if not args.only and not args.scope and visuals.exists():
        shutil.rmtree(visuals)  # a full run replaces the folder, so nothing stale is left behind
    paths = social_cards.build_all(
        league, only=args.only, scope=args.scope, with_profiles=not args.no_profiles, report=print
    )
    teams = sum(1 for p in paths if p.parent.name == "teams")
    print(f"{teams} club graphics, {len(paths) - teams} player graphics -> {visuals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
