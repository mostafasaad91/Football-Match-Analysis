import pandas as pd

from match_store import rolling_summary


def test_recent_form_compares_latest_window_with_previous_window():
    frame = pd.DataFrame({"match_id": range(1, 7), "box_entries": [10, 9, 8, 6, 5, 4]})
    result = rolling_summary(frame, window=3, id_columns=("match_id",))
    row = result[result.metric.eq("box_entries")].iloc[0]
    assert row.matches == 3
    assert row.recent_average == 9.0
    assert row.previous_average == 5.0
    assert row.trend == 4.0


def test_form_keeps_zero_and_does_not_turn_missing_into_zero():
    frame = pd.DataFrame({"metric": [0, 1], "other": [None, None]})
    result = rolling_summary(frame, window=2)
    assert result[result.metric.eq("metric")].iloc[0].recent_average == .5
    assert "other" not in result.metric.tolist()
