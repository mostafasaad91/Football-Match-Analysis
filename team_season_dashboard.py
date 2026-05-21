"""CLI wrapper for the team season dashboard tool."""

from __future__ import annotations

import argparse

from team_dashboard import get_team_stats


def main() -> None:
    """Parse CLI arguments and run the dashboard builder."""
    parser = argparse.ArgumentParser(description="Build a team current-season statistical dashboard.")
    parser.add_argument("team", nargs="?", help="Team name, e.g. Arsenal")
    args = parser.parse_args()
    team = args.team or input("Team name: ").strip()
    get_team_stats(team)


if __name__ == "__main__":
    main()
