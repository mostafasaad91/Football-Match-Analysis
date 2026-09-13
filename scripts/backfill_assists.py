"""Put the assist back on every shot in the packages already on disk.

The parse read the assisting player from a qualifier value that the feed never
fills, so every exported ``events.csv`` carries an empty ``assist_player``. The
fix in ``football_match_analysis.assist_provider`` covers fixtures parsed from
now on; this repairs the ones already published, from the raw match-centre
snapshot each was parsed from, with the same rule. Nothing is re-scraped and
nothing is guessed from the pass that happened to come before the shot.

    python scripts/backfill_assists.py             # every package
    python scripts/backfill_assists.py --dry-run   # report, write nothing

Both copies of a package are repaired, dark and light, because each reads its
own ``events.csv``. This is data only: pictures and PDFs already drawn keep
what they show, and only a redraw puts the assists on them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "output"
SNAPSHOTS = OUTPUT / "raw_snapshots"


def _load(path: Path) -> dict | None:
    try:
        return json.loads(gzip.open(path, "rt", encoding="utf-8").read())
    except Exception:
        return None


def snapshot_index() -> dict:
    """Snapshot paths keyed by (home id, away id, date), the fields a package keeps.

    Paths, not parsed payloads: a hundred match centres held at once ran a
    machine out of memory while renders were running beside it. Each is parsed
    again when its package comes up, one at a time.
    """
    found = {}
    for path in SNAPSHOTS.glob("ws-*.json.gz"):
        data = _load(path)
        if data is None:
            continue
        key = ((data.get("home") or {}).get("teamId"),
               (data.get("away") or {}).get("teamId"),
               str(data.get("startDate") or data.get("startTime") or "")[:10])
        found[key] = path
        del data
    return found


def assists_by_event(data: dict) -> dict:
    """{provider event id: (assist player name, assist kind)} for one match."""
    from football_match_analysis import _events_by_team_and_id, assist_provider

    names = {int(k): v for k, v in (data.get("playerIdNameDictionary") or {}).items()}
    events = data.get("events") or []
    by_event = _events_by_team_and_id(events)
    found = {}
    for event in events:
        if not event.get("isShot"):
            continue
        provider, kind = assist_provider(event, by_event)
        if provider is not None and event.get("id") is not None:
            found[int(event["id"])] = (names.get(provider, ""), kind)
    return found


def repair(events_path: Path, assists: dict, write: bool) -> tuple[int, int]:
    """Fill one events.csv. Returns (goals assisted, shots assisted)."""
    frame = pd.read_csv(events_path, low_memory=False)
    if "event_id" not in frame or "assist_player" not in frame:
        return 0, 0
    ids = pd.to_numeric(frame["event_id"], errors="coerce")
    player = frame["assist_player"].astype(object)
    kind = frame["assist_type"].astype(object) if "assist_type" in frame else None
    shots = goals = 0
    for index, event_id in ids.items():
        if pd.isna(event_id) or int(event_id) not in assists:
            continue
        name, how = assists[int(event_id)]
        if not name:
            continue
        player.at[index] = name
        if kind is not None and how:
            kind.at[index] = how
        shots += 1
        if bool(frame.at[index, "is_goal"]) if "is_goal" in frame else False:
            goals += 1
    if write and shots:
        frame["assist_player"] = player
        if kind is not None:
            frame["assist_type"] = kind
        frame.to_csv(events_path, index=False, encoding="utf-8-sig")
    return goals, shots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    snapshots = snapshot_index()
    packages = [p.parent for p in sorted(OUTPUT.rglob("match_info.json"))
                if p.parent.name != "light"
                and not any(x.startswith(".") for x in p.parent.parts)
                and (p.parent / "events.csv").exists()]
    total_goals = total_shots = 0
    missing = []
    for folder in packages:
        info = json.loads((folder / "match_info.json").read_text(encoding="utf-8-sig"))
        key = (info.get("home_id"), info.get("away_id"), str(info.get("date"))[:10])
        path = snapshots.get(key)
        data = _load(path) if path else None
        if data is None:
            missing.append(folder.name)
            continue
        assists = assists_by_event(data)
        del data
        goals, shots = repair(folder / "events.csv", assists, write=not args.dry_run)
        light = folder / "light" / "events.csv"
        if light.exists():
            repair(light, assists, write=not args.dry_run)
        total_goals += goals
        total_shots += shots
        print(f"{folder.name:<46} goals {goals:>2}   assisted shots {shots:>3}")

    print(f"\n{len(packages) - len(missing)}/{len(packages)} packages"
          f"{' (dry run)' if args.dry_run else ''}: "
          f"{total_goals} assisted goals, {total_shots} assisted shots")
    if missing:
        print("no snapshot for:", ", ".join(missing))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
