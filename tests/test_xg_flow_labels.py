"""The xG flow page when the two curves run close together.

Two sides that finish on nearly the same xG used to print one total on top of the
other, and goals scored within a few minutes of each other had their names stacked.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from football_analysis.visuals import visual_redesign_preview as base


def _shot(minute, team, xg, goal=False, player="Someone"):
    return {
        "minute": minute,
        "second": 0,
        "team_id": team,
        "xG": xg,
        "is_shot": True,
        "is_goal": goal,
        "player": player,
        "period": 1 if minute < 45 else 2,
        "type": "Goal" if goal else "MissedShots",
    }


@pytest.fixture
def close_match(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "OUT_DIR", tmp_path)
    monkeypatch.setattr(base, "COMPARE_DIR", tmp_path)
    home, away = base.HOME_ID, base.AWAY_ID
    rows = [
        _shot(1, home, 0.05, True, "Early One"),
        _shot(2, home, 0.20, True, "Early Two"),
        _shot(30, away, 0.30),
        _shot(50, away, 0.90, True, "Middle Man"),
        _shot(66, home, 0.80, True, "Late One"),
        _shot(69, home, 0.05, True, "Late Two"),
        _shot(88, away, 0.02),
    ]
    return pd.DataFrame(rows)


def _boxes(fig, wanted):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return [
        text.get_window_extent(renderer)
        for ax in fig.axes
        for text in ax.texts
        if wanted(text.get_text())
    ]


def _overlapping(boxes):
    return [
        (a, b)
        for i, a in enumerate(boxes)
        for b in boxes[i + 1 :]
        if a.x0 < b.x1 and a.x1 > b.x0 and a.y0 < b.y1 and a.y1 > b.y0
    ]


def test_close_totals_and_goal_labels_do_not_overlap(close_match, monkeypatch):
    kept = {}
    monkeypatch.setattr(base, "save", lambda fig, name: kept.setdefault("fig", fig))
    base.theme()
    base.xg_flow(close_match)
    fig = kept["fig"]
    try:
        # home finishes on 1.10 and away on 1.22: the two totals and their captions
        totals = _boxes(fig, lambda s: s in {"1.10", "1.22"} or s.startswith("xG  ·"))
        assert len(totals) == 4
        assert not _overlapping(totals)
        names = _boxes(fig, lambda s: s.split("  ")[0] in {"One", "Two", "Man"})
        assert len(names) == 5
        assert not _overlapping(names)
        # no goal label runs off the left edge over the axis figures
        plot = next(ax for ax in fig.axes if ax.get_xlabel() == "MATCH MINUTE")
        left = plot.get_window_extent(fig.canvas.get_renderer()).x0
        assert all(box.x0 >= left - 1 for box in names)
    finally:
        plt.close(fig)
