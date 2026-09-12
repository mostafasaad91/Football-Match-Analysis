"""Redraw one visual in packages already on disk, without rebuilding them.

A full rebuild is seven minutes a fixture because it redraws eighty figures,
writes the article, the posters and two PDFs. When the fix is to a single
chart -- a label that overlapped, a note placed over a marker -- that is the
wrong price, and it also rewrites files that were correct.

This loads each package's own stored frames, calls one draw function, and
drops the result over the file already published under that name. Nothing is
reparsed and nothing else in the package is touched.

    python scripts/refresh_visual.py finishing_quality
    python scripts/refresh_visual.py finishing_quality Napoli Arsenal

Both themes are done: the dark copy in this process and the light copy in a
child, because visualization_components reads MATCH_ANALYSIS_THEME once when
it is first imported.

The PDF and the posters embed their own copy of every chart, so a package
refreshed this way carries the corrected PNG beside a PDF still showing the
old one. That is the trade this script exists to make; rebuild the package if
the PDF has to agree.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "output"


def packages(patterns: list[str]) -> list[Path]:
    """Every published package holding the frames a redraw needs."""
    found = []
    for info in sorted(OUTPUT.rglob("match_info.json")):
        out = info.parent
        if out.name == "light" or any(p.startswith(".") for p in out.parts):
            continue
        if not (out / "events.csv").exists():
            continue
        if patterns and not any(p.lower() in out.name.lower() for p in patterns):
            continue
        found.append(out)
    return found


def redraw(package: Path, theme_dir: Path, function: str, token: str) -> str:
    """Draw the one figure and put it where the published copy already sits.

    The draw function names its own output by the number it had while the
    package was being written, and publication renumbers. The published name
    is whatever is on disk, so the new figure is moved onto it rather than
    saved under the name the renderer would choose.
    """
    import json
    import visual_redesign_full as full

    existing = sorted(theme_dir.glob(f"*{token}.png"))
    if not existing:
        return "no published copy"

    info = json.loads((package / "match_info.json").read_text(encoding="utf-8"))
    events = pd.read_csv(package / "events.csv")

    staging = Path(tempfile.mkdtemp(prefix="refresh-"))
    try:
        full.configure_match(info, staging)
        drawn = getattr(full, function)(events)
        if drawn is None or not Path(drawn).exists():
            return "draw produced nothing"
        for target in existing:
            shutil.copyfile(drawn, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return "ok"


def main() -> int:
    arguments = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not arguments:
        print(__doc__)
        return 2
    function, patterns = arguments[0], arguments[1:]
    token = function

    theme = os.environ.get("MATCH_ANALYSIS_THEME", "dark")
    light = theme == "light"
    targets = packages(patterns)
    if not targets:
        print("No published packages matched.")
        return 1

    print(f"Refreshing {function} in {len(targets)} package(s) [{theme}]", flush=True)
    started = time.time()
    problems = []
    for index, package in enumerate(targets, 1):
        theme_dir = package / "light" if light else package
        if not theme_dir.is_dir():
            problems.append((package.name, "no light copy"))
            continue
        try:
            result = redraw(package, theme_dir, function, token)
        except Exception as error:                      # noqa: BLE001
            result = f"{type(error).__name__}: {error}"
        if result != "ok":
            problems.append((package.name, result))
        print(f"  [{index}/{len(targets)}] {package.name}: {result}", flush=True)

    print(f"\n{len(targets) - len(problems)}/{len(targets)} refreshed [{theme}] "
          f"in {(time.time() - started) / 60:.1f} min")
    for name, reason in problems:
        print(f"  {name}: {reason}")

    # The light theme is a property of the interpreter, so it needs its own.
    if not light and os.environ.get("REFRESH_VISUAL_LIGHT", "1") != "0":
        print("\n--- light copies ---", flush=True)
        child = {**os.environ, "MATCH_ANALYSIS_THEME": "light",
                 "PYTHONIOENCODING": "utf-8"}
        done = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                               function, *patterns],
                              cwd=ROOT, env=child)
        return done.returncode or (1 if problems else 0)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
