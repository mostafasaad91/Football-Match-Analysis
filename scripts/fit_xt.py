"""Fit the expected-threat surface on every match stored under output/.

    python scripts/fit_xt.py

Writes data/models/xt_grid.json. The light copies are skipped: they hold the
same events as the dark package beside them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from football_analysis.metrics.expected_threat import fit_grid, save_grid  # noqa: E402

COLUMNS = ["x", "y", "end_x", "end_y", "is_pass", "is_shot", "is_goal", "is_penalty",
           "is_penalty_shootout", "outcome", "qualifier_names"]


def frames():
    for path in sorted(ROOT.glob("output/*/*/Matchweek_*/*/events.csv")):
        if "light" in path.parts:
            continue
        try:
            yield pd.read_csv(path, usecols=lambda c: c in COLUMNS, low_memory=False)
        except Exception as error:  # a half-written package
            print(f"  skipped {path.parent.name}: {error}")


def main() -> int:
    count = [0]

    def counted():
        for frame in frames():
            count[0] += 1
            yield frame

    grid = fit_grid(counted())
    path = save_grid(grid, count[0])
    print(f"xT surface fitted on {count[0]} matches -> {path}")
    print(f"  max cell {grid.max():.3f}, own box {grid[0].max():.4f}, edge of opp box {grid[13].max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
