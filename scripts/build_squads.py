"""Write a starting squads.json from the fixtures already parsed.

The pipeline reads each team's squad out of the provider's own match feed, so a
fixture that lists a player for the wrong side is internally consistent: the
events agree with the squad, the squad agrees with the events, and every check
in match_sanity passes. Nothing inside the data can see it.

The only thing that can is a roster kept outside it. This writes a first draft
from every match in output/ — a starting point to correct by hand, not an
answer. Reviewing it once is the point: the entry that is wrong is exactly the
one the pipeline could not find on its own.

    python scripts/build_squads.py            # write squads.json
    python scripts/build_squads.py --check    # list disagreements, write nothing

A team that appears in more than one fixture gets the union of what was seen,
with the number of matches each name appeared in, so a one-off is easy to spot.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "squads.json"


def collect() -> tuple[dict[str, list[str]], dict[str, Counter]]:
    """(squad per team, how many fixtures each name appeared in)."""
    squads: dict[str, set[str]] = defaultdict(set)
    seen: dict[str, Counter] = defaultdict(Counter)
    for info_path in sorted((ROOT / "output").glob("*/match_info.json")):
        folder = info_path.parent
        players = folder / "players.csv"
        if not players.exists():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            frame = pd.read_csv(players)
        except Exception:
            continue
        for side in ("home", "away"):
            name = str(info.get(f"{side}_name") or "").strip()
            try:
                team_id = int(info[f"{side}_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if not name:
                continue
            listed = frame[frame["team_id"].eq(team_id)]["name"].dropna().astype(str)
            squads[name].update(listed)
            seen[name].update(set(listed))
    return {team: sorted(names) for team, names in squads.items()}, seen


def main(argv: list[str]) -> int:
    squads, seen = collect()
    if not squads:
        print("No parsed fixtures found under output/.")
        return 1

    if "--check" in argv:
        existing = {}
        if TARGET.exists():
            existing = {str(k).strip().lower(): set(v)
                        for k, v in json.loads(TARGET.read_text(encoding="utf-8")).items()}
        for team, names in squads.items():
            known = existing.get(team.lower())
            if known is None:
                print(f"{team}: no roster in {TARGET.name}")
                continue
            strangers = [n for n in names if n not in known]
            print(f"{team}: {len(strangers)} not in your roster"
                  + (f" — {', '.join(strangers)}" if strangers else ""))
        return 0

    if TARGET.exists():
        print(f"{TARGET.name} already exists; not overwriting a roster you have edited.")
        return 1

    TARGET.write_text(json.dumps(squads, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"Wrote {TARGET} for {len(squads)} team(s).")
    print("Review it: names seen in only one fixture are the ones worth checking.")
    for team, counter in sorted(seen.items()):
        once = [name for name, count in counter.items() if count == 1]
        if once and len(counter) > len(once):
            print(f"  {team}: seen once — {', '.join(sorted(once))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
