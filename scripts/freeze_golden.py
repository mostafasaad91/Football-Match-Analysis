#!/usr/bin/env python3
"""Re-freeze the reference output that ``tests/test_metrics_golden.py`` compares against.

Run this only when a metric was *meant* to change. The golden test exists to
make an unintended change loud, so re-freezing without reading the diff turns
the safety net off. The intended loop is:

    python -m pytest tests/test_metrics_golden.py     # see exactly what moved
    python scripts/freeze_golden.py                   # accept it
    git diff tests/golden/                            # review the new numbers

Usage
-----
    python scripts/freeze_golden.py                   # re-freeze in place
    python scripts/freeze_golden.py --check           # report drift, write nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from match_metrics import advanced_metrics_frames  # noqa: E402

# Kept in step with tests/test_metrics_golden.py. A fixture is the events file
# plus the match identity it was played under; the metrics need both.
FIXTURES = {
    "france_vs_england": {
        "events": REPO_ROOT / "sample_data" / "France_vs_England_4-6" / "events.csv",
        "info": {
            "home_id": 341,
            "away_id": 345,
            "home_name": "France",
            "away_name": "England",
            "score": "4-6",
        },
    },
}

EXPORTS = {
    "team_advanced_metrics.csv": 0,
    "player_sequence_metrics.csv": 1,
}


def freeze(name: str, check_only: bool) -> int:
    fixture = FIXTURES[name]
    events_path: Path = fixture["events"]
    if not events_path.exists():
        print(f"error: {events_path} is missing", file=sys.stderr)
        return 1

    target = REPO_ROOT / "tests" / "golden" / name
    target.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(events_path, encoding="utf-8-sig")
    frames = advanced_metrics_frames(events, fixture["info"])

    changed = 0
    for filename, index in EXPORTS.items():
        frame = frames[index]
        destination = target / filename
        # Round to the precision the reference is published at, so a float
        # representation difference never reads as a metric change.
        rounded = frame.copy()
        for column in rounded.select_dtypes(include="number").columns:
            rounded[column] = rounded[column].round(4)

        if destination.exists():
            previous = pd.read_csv(destination, encoding="utf-8-sig")
            same = previous.shape == rounded.shape and previous.to_csv(
                index=False
            ) == rounded.to_csv(index=False)
            if same:
                print(f"  {filename}: unchanged")
                continue

        changed += 1
        if check_only:
            print(f"  {filename}: WOULD CHANGE")
            continue
        rounded.to_csv(destination, index=False, encoding="utf-8-sig")
        print(f"  {filename}: rewritten ({len(rounded)} rows, {len(rounded.columns)} cols)")

    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        default="france_vs_england",
        choices=sorted(FIXTURES),
        help="which frozen reference to rewrite",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the reference is stale without rewriting it",
    )
    args = parser.parse_args()

    print(f"fixture: {args.fixture}")
    changed = freeze(args.fixture, args.check)

    if args.check:
        if changed:
            print(f"\n{changed} export(s) differ from the frozen reference.")
            return 1
        print("\nReference is current.")
        return 0

    if changed:
        print(f"\nRewrote {changed} export(s). Review with: git diff tests/golden/")
    else:
        print("\nNothing to do — the reference already matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
