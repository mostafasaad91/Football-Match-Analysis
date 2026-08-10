"""
fixtures.py
═════════════════════════════════════════════════════════════════════════════
Look up a fixture's WhoScored URL instead of hunting for it in a browser.

``data/fixtures/`` holds the season calendar for the leagues we cover, each
fixture already carrying its WhoScored match id. That turns the analyser's one
piece of manual input — the URL — into a lookup:

    python fixtures.py arsenal                    # every Arsenal fixture
    python fixtures.py arsenal --next             # the next one to be played
    python fixtures.py arsenal --played           # only fixtures already past
    python fixtures.py --on 2026-08-22            # everything that day
    python fixtures.py arsenal --next --url       # just the URL, for piping

Feed the result straight into the analyser:

    $env:MATCH_ANALYSIS_URL = python fixtures.py arsenal --next --url
    python football_match_analysis.py

The calendar is a fixed file, not a live feed. A postponement moves the real
match without moving this row, so treat the date as indicative and the match
id as authoritative.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "fixtures"
WHOSCORED_URL = "https://www.whoscored.com/matches/{id}/live"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_kickoff(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_fixtures(competition: str | None = None) -> list[dict]:
    """Every stored fixture, newest last, tagged with its competition."""
    if not FIXTURE_DIR.is_dir():
        return []
    rows: list[dict] = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = payload.get("competition", path.stem)
        if competition and competition.upper() != name.upper():
            continue
        for fixture in payload.get("fixtures", []):
            rows.append(
                {
                    **fixture,
                    "competition": name,
                    "season": payload.get("season", ""),
                    "url": WHOSCORED_URL.format(id=fixture["whoscored_id"]),
                }
            )
    rows.sort(key=lambda row: row["kickoff_utc"])
    return rows


def known_teams(rows: list[dict]) -> list[str]:
    return sorted({row["home"] for row in rows} | {row["away"] for row in rows})


def resolve_team(query: str, rows: list[dict]) -> str:
    """Turn a typed name into exactly one calendar slug, or refuse.

    Refusing beats guessing: "united" matching whichever United sorts first
    would send a run at the wrong match and the output would look completely
    normal.
    """
    slug = query.strip().lower().replace(" ", "-")
    teams = known_teams(rows)
    if slug in teams:
        return slug
    matches = [team for team in teams if slug in team]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise LookupError(
            f"No team matching {query!r}. Try one of: {', '.join(teams[:12])} …"
        )
    raise LookupError(
        f"{query!r} matches {len(matches)} teams: {', '.join(matches)}. Be more specific."
    )


def team_fixtures(team: str, rows: list[dict]) -> list[dict]:
    return [row for row in rows if team in (row["home"], row["away"])]


def _display(slug: str) -> str:
    return slug.replace("-", " ").title()


def _print_rows(rows: list[dict], url_only: bool) -> int:
    if not rows:
        print("No fixtures matched.")
        return 1
    if url_only:
        for row in rows:
            print(row["url"])
        return 0
    now = _now()
    for row in rows:
        kickoff = _parse_kickoff(row["kickoff_utc"])
        marker = "played" if kickoff < now else "     "
        print(
            f"{kickoff:%Y-%m-%d %H:%M}  {marker}  {row['competition']:<7} "
            f"{_display(row['home']):>22} v {_display(row['away']):<22}  {row['url']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fixtures",
        description="Find a fixture's WhoScored URL from the stored season calendar.",
    )
    parser.add_argument("team", nargs="?", help="team name, e.g. arsenal or 'aston villa'")
    parser.add_argument("--competition", help="restrict to one competition (EPL, LALIGA, SERIEA)")
    parser.add_argument("--on", help="only fixtures on this date (YYYY-MM-DD)")
    parser.add_argument("--next", action="store_true", help="only the next fixture still to be played")
    parser.add_argument("--last", action="store_true", help="only the most recent fixture already played")
    parser.add_argument("--played", action="store_true", help="only fixtures whose kickoff has passed")
    parser.add_argument("--url", action="store_true", help="print bare URLs, nothing else")
    args = parser.parse_args(argv)

    rows = load_fixtures(args.competition)
    if not rows:
        print(f"No fixture data in {FIXTURE_DIR}.", file=sys.stderr)
        return 1

    if args.team:
        try:
            team = resolve_team(args.team, rows)
        except LookupError as error:
            print(error, file=sys.stderr)
            return 1
        rows = team_fixtures(team, rows)

    if args.on:
        rows = [row for row in rows if row["kickoff_utc"].startswith(args.on)]

    now = _now()
    if args.played:
        rows = [row for row in rows if _parse_kickoff(row["kickoff_utc"]) < now]
    if args.last:
        past = [row for row in rows if _parse_kickoff(row["kickoff_utc"]) < now]
        rows = past[-1:]
    if args.next:
        rows = [row for row in rows if _parse_kickoff(row["kickoff_utc"]) >= now][:1]

    return _print_rows(rows, args.url)


if __name__ == "__main__":
    sys.exit(main())
