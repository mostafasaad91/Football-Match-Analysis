"""
team_history.py
═════════════════════════════════════════════════════════════════════════════
Query the stored match history from the command line.

Every analyzer run appends to ``output/match_history.db``. This is the reader.

    python team_history.py matches
    python team_history.py matches --team Arsenal
    python team_history.py team Arsenal --last 6
    python team_history.py team Arsenal --last 6 --metrics field_tilt,box_entries,ppda
    python team_history.py player "Bukayo Saka" --last 5
    python team_history.py export Arsenal --last 10 --out arsenal_last10.csv
    python team_history.py replay
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from match_store import (
    list_matches,
    load_snapshot,
    save_match,
    stored_snapshots,
    player_match_log,
    team_match_log,
    team_totals,
)

# Columns that identify the row rather than describe the performance; shown
# first in the match log and never treated as a metric to average.
CONTEXT_COLUMNS = ["played_on", "competition", "opponent", "side", "score",
                   "goals_for", "goals_against"]


def _print(frame: pd.DataFrame, empty_message: str) -> int:
    if frame is None or frame.empty:
        print(empty_message)
        return 1
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame.to_string(index=False))
    return 0


def cmd_matches(args: argparse.Namespace) -> int:
    frame = list_matches(args.team)
    if not frame.empty:
        frame = frame[["match_id", "played_on", "competition", "home_team",
                       "away_team", "score"]]
    scope = f" for {args.team}" if args.team else ""
    return _print(frame, f"No matches stored{scope} yet. Analyse a fixture first.")


def cmd_team(args: argparse.Namespace) -> int:
    log = team_match_log(args.team, limit=args.last)
    if log.empty:
        print(f"No stored matches for {args.team!r}. Check the spelling against "
              f"'python team_history.py matches'.")
        return 1

    if args.summary:
        totals = team_totals(args.team, limit=args.last)
        if args.metrics:
            wanted = {name.strip() for name in args.metrics.split(",") if name.strip()}
            totals = totals[totals["metric"].isin(wanted)]
        print(f"{args.team} — {len(log)} match(es)\n")
        return _print(totals, "No numeric metrics stored.")

    columns = [c for c in CONTEXT_COLUMNS if c in log.columns]
    if args.metrics:
        wanted = [name.strip() for name in args.metrics.split(",") if name.strip()]
        missing = [name for name in wanted if name not in log.columns]
        if missing:
            print(f"Unknown metric(s): {', '.join(missing)}")
            print(f"Available: {', '.join(sorted(set(log.columns) - set(columns)))}")
            return 1
        columns += wanted
    else:
        # Without an explicit selection, show the headline set rather than a
        # forty-column wall.
        for name in ("possession_share", "field_tilt", "box_entries",
                     "final_third_entries", "high_regains", "sequence_xT"):
            if name in log.columns:
                columns.append(name)
    return _print(log[columns], "No rows.")


def cmd_player(args: argparse.Namespace) -> int:
    log = player_match_log(args.player, limit=args.last)
    if log.empty:
        print(f"No stored matches for {args.player!r}.")
        return 1
    columns = [c for c in ["played_on", "team", "home_team", "away_team", "score"]
               if c in log.columns]
    if args.metrics:
        wanted = [name.strip() for name in args.metrics.split(",") if name.strip()]
        columns += [name for name in wanted if name in log.columns]
    else:
        columns += [c for c in log.columns if c not in columns
                    and c not in {"match_id", "player", "metrics"}][:8]
    return _print(log[columns], "No rows.")


def cmd_export(args: argparse.Namespace) -> int:
    log = team_match_log(args.team, limit=args.last)
    if log.empty:
        print(f"No stored matches for {args.team!r}.")
        return 1
    log.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(log)} match rows to {args.out}")
    return 0



def cmd_replay(args: argparse.Namespace) -> int:
    """Recompute stored matches from their raw snapshots, without the network.

    This is what keeping the untouched payloads buys: a metric added today can
    be backfilled across every match already collected, instead of re-scraping
    a season to answer one new question.
    """
    import football_match_analysis as fma
    from match_metrics import advanced_metrics_frames

    ids = stored_snapshots()
    if args.match:
        ids = [m for m in ids if m == args.match]
    if not ids:
        print("No raw snapshots stored yet. Analyse a fixture first.")
        return 1

    for match_id in ids:
        payload = load_snapshot(match_id)
        if payload is None:
            print(f"{match_id}: snapshot missing")
            continue
        try:
            info, events, _players = fma.parse_all(payload)
            team_frame, player_frame = advanced_metrics_frames(events, info)
            save_match(info, team_frame, player_frame,
                       url=f"https://www.whoscored.com/matches/{match_id.removeprefix('ws-')}/live")
            print(f"{match_id}: recomputed {len(events)} events "
                  f"({info.get('home_name')} v {info.get('away_name')})")
        except Exception as error:
            print(f"{match_id}: failed — {error}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="team_history",
        description="Query the stored history of analysed matches.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    matches = sub.add_parser("matches", help="list stored matches")
    matches.add_argument("--team", help="only matches involving this team")
    matches.set_defaults(func=cmd_matches)

    team = sub.add_parser("team", help="one row per match for a team")
    team.add_argument("team")
    team.add_argument("--last", type=int, help="only the N most recent matches")
    team.add_argument("--metrics", help="comma-separated metric names")
    team.add_argument("--summary", action="store_true",
                      help="aggregate instead of listing each match")
    team.set_defaults(func=cmd_team)

    player = sub.add_parser("player", help="one row per match for a player")
    player.add_argument("player")
    player.add_argument("--last", type=int)
    player.add_argument("--metrics")
    player.set_defaults(func=cmd_player)

    export = sub.add_parser("export", help="write a team's match log to CSV")
    export.add_argument("team")
    export.add_argument("--last", type=int)
    export.add_argument("--out", default="team_history.csv")
    export.set_defaults(func=cmd_export)

    replay = sub.add_parser(
        "replay", help="recompute stored matches from their raw snapshots"
    )
    replay.add_argument("--match", help="one match id instead of all")
    replay.set_defaults(func=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
