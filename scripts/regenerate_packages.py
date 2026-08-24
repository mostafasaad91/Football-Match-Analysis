"""Rebuild every rendered package from the frames already on disk.

The parse is the expensive half and it does not change: events.csv, xg.csv and
the two metric exports are what the provider gave us, and they are written once
per fixture. What changes is everything downstream — the visuals, the report,
the article and the posters — so a fix to the prose or the cover means every
package on disk is a version behind until it is rebuilt.

Re-running the pipeline would re-scrape all of it. This walks the output tree
instead, hands each fixture's stored frames back to generate_match_package, and
writes the packages again. Nothing touches the network and nothing is reparsed.

    python scripts/regenerate_packages.py             # every match
    python scripts/regenerate_packages.py Hull PSG    # ones whose folder matches

The theme is fixed at import time by MATCH_ANALYSIS_THEME, so the light package
is built the way the pipeline builds it: in a child process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "output"
FRAMES = ("events.csv", "players.csv", "xg.csv",
          "team_advanced_metrics.csv", "player_sequence_metrics.csv")


def fixtures(patterns: list[str]) -> list[Path]:
    """Every match folder holding a full set of frames, dark package only."""
    found = []
    for info in sorted(OUTPUT.rglob("match_info.json")):
        out = info.parent
        if out.name == "light":
            continue
        if not all((out / name).exists() for name in FRAMES):
            continue
        if patterns and not any(p.lower() in out.name.lower() for p in patterns):
            continue
        found.append(out)
    return found


def rebuild(out: Path) -> dict:
    """One fixture, rebuilt in place from its own exports."""
    from visual_redesign_full import generate_match_package

    info = json.loads((out / "match_info.json").read_text(encoding="utf-8"))
    frames = {name: pd.read_csv(out / name) for name in FRAMES}
    return generate_match_package(
        frames["events.csv"],
        frames["players.csv"],
        frames["xg.csv"],
        frames["team_advanced_metrics.csv"],
        frames["player_sequence_metrics.csv"],
        info,
        out,
    )


def main() -> int:
    patterns = [a for a in sys.argv[1:] if not a.startswith("-")]
    targets = fixtures(patterns)
    if not targets:
        print("No rendered fixtures matched.")
        return 1

    # One theme per process: the renderer reads MATCH_ANALYSIS_THEME at import
    # and colours every module constant from it, so a second theme in the same
    # interpreter would draw the first one's palette.
    theme = os.environ.get("MATCH_ANALYSIS_THEME", "dark")
    print(f"Rebuilding {len(targets)} package(s) [{theme}]\n")

    failed = []
    for index, out in enumerate(targets, 1):
        label = f"[{index}/{len(targets)}] {out.name}"
        started = time.time()
        try:
            rebuild(out)
            print(f"{label}  ok  ({time.time() - started:.0f}s)")
        except Exception as error:
            failed.append((out.name, f"{type(error).__name__}: {error}"))
            print(f"{label}  FAILED  {type(error).__name__}: {error}")
            traceback.print_exc()

    if failed:
        print(f"\n{len(failed)} failed:")
        for name, reason in failed:
            print(f"  {name}: {reason}")
        return 1
    print(f"\nAll {len(targets)} rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
