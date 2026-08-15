"""Render a second, light-page copy of a fixture's package.

The project's identity is AMOLED black. A light "Ink & Petrol" palette has
lived in ``visualization_components`` alongside it, but it is selected by
``MATCH_ANALYSIS_THEME`` when that module is first imported, and every renderer
copies its colours into module constants at import time. One process therefore
renders one theme, and no amount of reassigning globals afterwards changes the
values already captured.

So the light copy is rendered by a child process with the environment variable
set, reading the frames the dark run has already written next to the fixture.
It lands in ``<match>/light/`` so the two sets can be compared without either
overwriting the other, and a failure here never costs the dark package:
publishing the black set is the point, the light set is for choosing.

Run directly on any finished match folder:

    python render_light.py output/PSG_vs_Aston_Villa_2-1
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Piped stdout on Windows defaults to cp1252, and this module prints arrows and
# accented player names. Without this the render succeeds and the *reporting*
# of it raises.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

LIGHT_DIRNAME = "light"
_REQUIRED = (
    "events.csv",
    "players.csv",
    "xg.csv",
    "team_advanced_metrics.csv",
    "player_sequence_metrics.csv",
    "match_info.json",
)


def light_dir(match_dir: Path | str) -> Path:
    return Path(match_dir) / LIGHT_DIRNAME


def missing_inputs(match_dir: Path | str) -> list[str]:
    """Which of the frames the light pass needs are not on disk."""
    base = Path(match_dir)
    return [name for name in _REQUIRED if not (base / name).exists()]


def render_light_package(match_dir: Path | str, *, timeout: int = 1800) -> Path | None:
    """Re-render one fixture on the light page. Returns the folder, or None.

    Never raises: the caller has a finished dark package in hand and a light
    copy that failed to build is not a reason to lose it.
    """
    base = Path(match_dir).resolve()
    absent = missing_inputs(base)
    if absent:
        print(f"  ! light copy skipped, missing: {', '.join(absent)}")
        return None

    environment = {**os.environ, "MATCH_ANALYSIS_THEME": "light"}
    # Guard against a light run spawning its own light run.
    environment["MATCH_ANALYSIS_LIGHT_COPY"] = "0"
    # The child's stdout is a pipe, so Windows gives it cp1252 and every arrow
    # or accented player name in its output raises UnicodeEncodeError -- which
    # surfaced as the light copy "failing" after it had rendered.
    environment["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(base), "--child"],
            env=environment,
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"  ! light copy failed: {type(error).__name__}: {error}")
        return None
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-4:]
        print(f"  ! light copy failed (exit {completed.returncode})")
        for line in tail:
            print(f"    {line}")
        return None
    return light_dir(base)


def _render_here(match_dir: Path) -> Path:
    """The actual render. Only correct in a process started with the light theme."""
    import pandas as pd

    from visualization_components import IS_LIGHT_THEME

    if not IS_LIGHT_THEME:
        raise RuntimeError(
            "render_light must run with MATCH_ANALYSIS_THEME=light; the theme is "
            "read when visualization_components is first imported"
        )

    from visual_redesign_full import generate_match_package

    match_info = json.loads((match_dir / "match_info.json").read_text(encoding="utf-8"))
    package = generate_match_package(
        pd.read_csv(match_dir / "events.csv"),
        pd.read_csv(match_dir / "players.csv"),
        pd.read_csv(match_dir / "xg.csv"),
        pd.read_csv(match_dir / "team_advanced_metrics.csv"),
        pd.read_csv(match_dir / "player_sequence_metrics.csv"),
        match_info,
        light_dir(match_dir),
        clean=True,
    )
    return package["output_dir"]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    match_dir = Path(argv[0]).resolve()
    if not match_dir.is_dir():
        print(f"not a directory: {match_dir}")
        return 2

    if "--child" in argv:
        out = _render_here(match_dir)
        print(f"Light package → {out}")
        return 0

    out = render_light_package(match_dir)
    if out is None:
        return 1
    print(f"Light package → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
