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

The theme is fixed at import time by MATCH_ANALYSIS_THEME, so each theme needs
its own interpreter. Both of them are children here and this process only
waits: a coordinator holding a finished dark render is what starved the light
one of memory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
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
    from football_match_analysis import choose_matchup_colors
    from visual_redesign_full import generate_match_package

    info_path = out / "match_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    # Resolve again instead of trusting colours stored by an older version.
    # In the default kit mode this selects a real alternate kit on a clash; in
    # explicit role mode choose_matchup_colors returns the fixed pair.
    home_color, away_color = choose_matchup_colors(
        str(info.get("home_name") or "Home"),
        str(info.get("away_name") or "Away"),
        info.get("home_kit_type"),
        info.get("away_kit_type"),
    )
    if (info.get("home_color"), info.get("away_color")) != (home_color, away_color):
        info["home_color"], info["away_color"] = home_color, away_color
        info_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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


def _run(arguments: list[str], theme: str, light_copy: str) -> subprocess.CompletedProcess:
    child = {**os.environ, "MATCH_ANALYSIS_THEME": theme,
             "MATCH_ANALYSIS_LIGHT_COPY": light_copy, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(arguments, cwd=ROOT, env=child, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def _reason(done: subprocess.CompletedProcess) -> str:
    tail = (done.stderr or done.stdout or "").strip().splitlines()[-3:]
    return " | ".join(line.strip() for line in tail) or f"exit {done.returncode}"


def main() -> int:
    if "--child" in sys.argv:
        # One package, in this process, for the coordinator below. Rebuilding
        # in the coordinator is what this mode exists to avoid.
        rebuild(Path(sys.argv[sys.argv.index("--child") + 1]).resolve())
        return 0

    patterns = [a for a in sys.argv[1:] if not a.startswith("-")]
    targets = fixtures(patterns)
    if not targets:
        print("No rendered fixtures matched.")
        return 1

    # One theme per process: the renderer reads MATCH_ANALYSIS_THEME at import
    # and colours every module constant from it, so a second theme in the same
    # interpreter would draw the first one's palette.
    theme = os.environ.get("MATCH_ANALYSIS_THEME", "dark")
    print(f"Rebuilding {len(targets)} package(s) [{theme}]")
    print()

    # Both renders run in their own process and this one holds nothing.
    #
    # generate_match_package already drops every figure and collects before it
    # launches the light child, and on this machine that was still not enough:
    # CPython does not hand the freed pages back, so the child inherited a
    # parent holding gigabytes and matplotlib died allocating its first canvas
    # with "MemoryError: bad allocation". The package was written, the light
    # copy silently was not, and a half-package looks finished from the outside.
    # A coordinator that only waits has nothing to hand over.
    failed = []
    for index, out in enumerate(targets, 1):
        label = f"[{index}/{len(targets)}] {out.name}"
        started = time.time()
        dark = _run([sys.executable, str(Path(__file__).resolve()), "--child", str(out)],
                    theme, "0")
        if dark.returncode != 0:
            failed.append((out.name, _reason(dark)))
            print(f"{label}  FAILED  {_reason(dark)}")
            continue
        if theme != "light":
            light = _run([sys.executable, str(ROOT / "render_light.py"), str(out), "--child"],
                         "light", "0")
            if light.returncode != 0:
                failed.append((f"{out.name} (light)", _reason(light)))
                print(f"{label}  ok, LIGHT FAILED  {_reason(light)}")
                continue
        print(f"{label}  ok  ({time.time() - started:.0f}s)")

    if failed:
        print()
        print(f"{len(failed)} failed:")
        for name, reason in failed:
            print(f"  {name}: {reason}")
        return 1
    print()
    print(f"All {len(targets)} rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
