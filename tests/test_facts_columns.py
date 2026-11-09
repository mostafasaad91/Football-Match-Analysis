"""Every team metric the article reads must exist in the table it reads from.

``match_facts`` maps attribute names to column names by string. A spelling
that does not match leaves the attribute at its zero default without a word,
and the article prints it: "xT_per_possession" was looked up as
"xt_per_possession" and every article said "0.00 per possession against 0.00".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_analysis.metrics.match_facts import _TEAM_METRIC_FIELDS
from football_analysis.metrics.match_metrics import team_advanced_metrics

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "sample_data" / "France_vs_England_4-6" / "events.csv"
INFO = {"home_id": 341, "away_id": 345, "home_name": "France", "away_name": "England"}


def test_every_mapped_column_is_written():
    events = pd.read_csv(EVENTS, encoding="utf-8-sig", low_memory=False)
    metrics = team_advanced_metrics(events, INFO)
    written = set(metrics["home"])
    missing = sorted(col for col in _TEAM_METRIC_FIELDS.values() if col not in written)
    assert not missing, f"match_facts reads columns the metrics never write: {missing}"
