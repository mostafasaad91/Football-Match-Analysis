"""Draw the timeline graphics for a block of rounds.

    python scripts/social_cards.py                         # every card, both frames
    python scripts/social_cards.py --only t05 p10          # just these
    python scripts/social_cards.py --format 16x9           # one frame only

Reads output/aggregates/<league>_<season>_R<a>-<b>/ (run scripts/aggregate_rounds.py first)
and writes PNGs into its ``visuals`` folder: ``twitter_4x5`` (1200 x 1500) and ``twitter_16x9``
(1600 x 900). t01-t15 are clubs and p01-p16 players, spread over attack, defence, pressing and
passing; the league boards and the club and player profiles are in the same ``visuals`` folder.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--only", nargs="*", help="card ids, e.g. t05 p10")
    parser.add_argument("--format", choices=["4x5", "16x9", "both"], default="both")
    args = parser.parse_args(argv)

    folder = OUTPUT_DIR / "aggregates" / f"{args.league}_{args.season}_R{args.rounds}"
    if not (folder / "teams_totals.csv").exists():
        print(f"No tables at {folder}; run scripts/aggregate_rounds.py first")
        return 1
    league = boards.League.load(folder, packages=OUTPUT_DIR / args.league / args.season)
    for path in social_cards.build_all(
        league,
        only=args.only,
        formats=("4x5", "16x9") if args.format == "both" else (args.format,),
    ):
        print(path.name)
    print(f"-> {folder / 'visuals'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
