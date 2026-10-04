"""First pass of the visual refresh: things that were wrong, not merely plain.

A shot map that called nine blocks "Saved", two-team charts that drew each side
on its own scale, "1 entries" printed on a chart, and no post-shot xG or
line-breaking passes in the match statistics, on the cover or on the posters.
"""

import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest
from conftest import match_dir

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_data" / "France_vs_England_4-6" / "events.csv"
SAMPLE_INFO = {
    "home_id": 341,
    "away_id": 345,
    "home_name": "France",
    "away_name": "England",
    "score": "4-6",
}


# ── the shot map names what happened to each shot ───────────────────────────
@pytest.fixture(scope="module")
def coventry():
    out = match_dir("Arsenal_vs_Coventry_3-0")
    if not (out / "match_info.json").exists():
        pytest.skip("Arsenal_vs_Coventry_3-0 has not been rendered")
    return out


def test_the_shot_map_legend_counts_blocks_apart_from_saves(coventry, tmp_path, monkeypatch):
    from football_analysis.visuals import visual_redesign_full as v

    info = json.loads((coventry / "match_info.json").read_text(encoding="utf-8-sig"))
    events = pd.read_csv(coventry / "events.csv", low_memory=False)
    xg = pd.read_csv(coventry / "xg.csv")
    v.configure_match(info, tmp_path)
    v.base.theme()

    captured = []
    monkeypatch.setattr(v, "save", lambda fig, name: captured.append(fig) or Path(name))
    v.shot_map(events, xg, info["home_id"], 2)

    legend = next(ax.get_legend() for ax in captured[0].axes if ax.get_legend() is not None)
    shown = {
        t.get_text().split(" (")[0]: int(t.get_text().split("(")[1].rstrip(")"))
        for t in legend.get_texts()
    }
    shots = events[
        (events["team_id"] == info["home_id"])
        & events["is_shot"].astype(str).str.lower().eq("true")
        & ~events["is_own_goal"].astype(str).str.lower().eq("true")
    ]
    kinds = shots["shot_whoscored_type"].value_counts()
    assert shown.get("Blocked", 0) == int(kinds.get("BlockedShot", 0))
    assert shown.get("Saved", 0) == int(kinds.get("SavedShot", 0))
    assert shown.get("Blocked", 0) > 0, "the fixture should have blocked shots to prove the split"


# ── two-team charts share their value axis ──────────────────────────────────
def test_share_limits_gives_both_panels_the_same_range():
    import matplotlib.pyplot as plt

    from football_analysis.visuals.insight_visuals import share_limits

    fig, (a, b) = plt.subplots(1, 2)
    a.set_xlim(0, 40)
    b.set_xlim(0, 60)
    a.set_ylim(0, 2.5)
    b.set_ylim(0, 0.8)
    share_limits([a, b], "x")
    share_limits([a, b], "y")
    assert a.get_xlim() == b.get_xlim() == (0, 60)
    assert a.get_ylim() == b.get_ylim() == (0, 2.5)
    plt.close(fig)


def test_share_limits_leaves_an_inverted_category_axis_alone():
    import matplotlib.pyplot as plt

    from football_analysis.visuals.insight_visuals import share_limits

    fig, (a, b) = plt.subplots(1, 2)
    a.set_ylim(3.6, -0.6)
    b.set_ylim(3.6, -0.6)
    a.set_xlim(0, 10)
    b.set_xlim(0, 20)
    share_limits([a, b], "x")
    assert a.get_ylim() == (3.6, -0.6)
    plt.close(fig)


@pytest.mark.parametrize(
    ("n", "text"),
    [(0, "0 entries"), (1, "1 entry"), (2, "2 entries")],
)
def test_counted_agrees_with_the_number(n, text):
    from football_analysis.visuals.insight_visuals import counted

    assert counted(n, "entry", "entries") == text


def test_counted_regular_plural():
    from football_analysis.visuals.insight_visuals import counted

    assert counted(1, "shot") == "1 shot"
    assert counted(3, "shot") == "3 shots"


# ── post-shot xG and line-breaking passes are published ─────────────────────
@pytest.fixture(scope="module")
def sample_events():
    if not SAMPLE.exists():
        pytest.skip("sample events not available")
    return pd.read_csv(SAMPLE, encoding="utf-8-sig")


def test_team_metrics_carry_post_shot_xg_and_line_breaking_passes(sample_events):
    from football_analysis.metrics.match_metrics import advanced_metrics_frames

    frame, _players = advanced_metrics_frames(sample_events, SAMPLE_INFO)
    for column in ("xGoT", "line_breaking_passes", "line_breaking_completed"):
        assert column in frame.columns, column
    assert (frame["line_breaking_completed"] <= frame["line_breaking_passes"]).all()
    assert (frame["xGoT"] >= 0).all()


def test_post_shot_xg_ignores_own_goals_and_shootout_kicks():
    from football_analysis.metrics.match_metrics import team_post_shot_xg

    row = {
        "team_id": 1,
        "is_shot": True,
        "shot_whoscored_type": "Goal",
        "xG": 0.4,
        "goal_mouth_y": 52.0,
        "goal_mouth_z": 10.0,
        "body_part": "RightFoot",
    }
    honest = pd.DataFrame([row])
    padded = pd.DataFrame([row, {**row, "is_own_goal": True}, {**row, "is_penalty_shootout": True}])
    assert team_post_shot_xg(padded, 1) == team_post_shot_xg(honest, 1) > 0


def test_the_statistics_page_lists_post_shot_xg_and_line_breaking_passes(sample_events):
    from football_analysis.visuals import tactical_visualizations as tv

    ppda = {"home": {"ppda": 8.0}, "away": {"ppda": 9.0}}
    fig = tv.make_match_stats_v2(sample_events, SAMPLE_INFO, ppda)
    words = {t.get_text() for ax in fig.axes for t in ax.texts}
    assert "xGOT" in words
    assert "Line-breaking passes" in words


def test_the_cover_rows_include_both_new_measures():
    from football_analysis.reports.tactical_pdf_report import TacticalPDF

    keys = {key for _label, key, _shape in TacticalPDF.COVER_ROWS}
    assert {"xGoT", "line_breaking_completed"} <= keys
    shown = [row for _heading, rows in TacticalPDF.COVER_GROUPS for row in rows]
    assert len(shown) == len(TacticalPDF.COVER_ROWS), "every cover row sits in a group"


def test_a_snapshot_fills_in_metrics_added_after_it_was_written(sample_events):
    from football_analysis.metrics.match_metrics import advanced_metrics_frames
    from football_analysis.render.render_snapshot import _backfill_team_metrics

    fresh, _ = advanced_metrics_frames(sample_events, SAMPLE_INFO)
    old = fresh.drop(columns=["xGoT", "line_breaking_passes", "line_breaking_completed"])
    filled = _backfill_team_metrics(sample_events, SAMPLE_INFO, old)
    assert filled["xGoT"].tolist() == fresh["xGoT"].tolist()
    assert filled["line_breaking_completed"].tolist() == fresh["line_breaking_completed"].tolist()
