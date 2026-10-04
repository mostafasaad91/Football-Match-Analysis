"""Draw the league boards, every club's profile and every qualified player's profile.

    python scripts/aggregate_rounds.py           # first: the tables
    python scripts/league_visuals.py             # then: Premier League, rounds 1-5
    python scripts/league_visuals.py --no-players --league Italy_Serie_A --rounds 1-5

Reads output/aggregates/<league>_<season>_R<a>-<b>/ and writes into its ``visuals`` folder:
the league boards (numbered 01-10), ``teams/<club>.png`` and ``players/<club>/<name>.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_analysis.paths import OUTPUT_DIR  # noqa: E402
from football_analysis.visuals import league_boards as boards  # noqa: E402
from football_analysis.visuals import visual_redesign_preview as base  # noqa: E402

LEAGUE_BOARDS = (
    boards.xg_quadrant,
    boards.league_heat_table,
    boards.player_leaderboards,
    boards.style_map,
    boards.over_under,
    boards.chance_sources,
    boards.scoring_and_creating,
    boards.progressing_the_ball,
    boards.winning_the_ball,
    boards.goalkeepers,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--league", default="England_Premier_League")
    parser.add_argument("--season", default="2026-2027")
    parser.add_argument("--rounds", default="1-5", help="the block the tables cover, e.g. 1-5")
    parser.add_argument(
        "--min-minutes", type=float, default=270.0, help="floor for a player profile"
    )
    parser.add_argument(
        "--no-players", action="store_true", help="league boards and club profiles only"
    )
    args = parser.parse_args(argv)

    folder = OUTPUT_DIR / "aggregates" / f"{args.league}_{args.season}_R{args.rounds}"
    if not (folder / "teams_totals.csv").exists():
        print(f"No tables at {folder}; run scripts/aggregate_rounds.py first")
        return 1
    base.theme()
    league = boards.League.load(folder, packages=OUTPUT_DIR / args.league / args.season)

    for board in LEAGUE_BOARDS:
        print(board(league).name)
    teams = list(league.teams["team"])
    for team in teams:
        boards.team_profile(league, team)
    print(f"{len(teams)} club profiles")
    if not args.no_players:
        qualified = boards._qualified(league, args.min_minutes)
        made = 0
        for _, row in qualified.iterrows():
            if boards.player_profile(league, row["player"], row["team"], args.min_minutes):
                made += 1
        print(f"{made} player profiles")
    print(f"-> {folder / 'visuals'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
