"""Fit the VAEP models on every match stored under output/.

    python scripts/fit_vaep.py

Writes data/models/vaep_model.joblib and prints held-out AUC and Brier score
(every tenth match is held out before the final fit on all of them).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from football_analysis.metrics.vaep import MODEL_PATH, fit  # noqa: E402

COLUMNS = ["period_code", "minute", "second", "team_id", "player", "type", "outcome", "x", "y",
           "end_x", "end_y", "is_header", "is_own_goal", "is_penalty_shootout", "qualifier_names"]


def frames():
    for path in sorted(ROOT.glob("output/*/*/Matchweek_*/*/events.csv")):
        if "light" in path.parts:
            continue
        try:
            yield pd.read_csv(path, usecols=lambda c: c in COLUMNS, low_memory=False)
        except Exception as error:
            print(f"  skipped {path.parent.name}: {error}")


def main() -> int:
    bundle = fit(frames())
    print(f"VAEP fitted on {bundle['matches']} matches, {bundle['actions']:,} actions -> {MODEL_PATH}")
    print(json.dumps(bundle["validation"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
