"""CLI wrapper for the team season dashboard tool."""

from __future__ import annotations

import argparse

from team_dashboard import get_team_stats


TEAM_NAME = ""
SOFASCORE_TEAM_ID = None
SOFASCORE_TEAM_URL = ""
WHOSCORED_TEAM_URL = ""


def main() -> None:
    """Parse CLI arguments and run the dashboard builder."""
    parser = argparse.ArgumentParser(description="Build a team current-season statistical dashboard.")
    parser.add_argument("team", nargs="?", help="Team name, e.g. Arsenal")
    args = parser.parse_args()
    team = args.team or TEAM_NAME.strip() or input("Team name: ").strip()
    if not team:
        print("Team name is required.")
        raise SystemExit(1)
    get_team_stats(
        team,
        sofascore_team_id=SOFASCORE_TEAM_ID,
        sofascore_team_url=SOFASCORE_TEAM_URL,
        whoscored_team_url=WHOSCORED_TEAM_URL,
    )


if __name__ == "__main__":
    main()
