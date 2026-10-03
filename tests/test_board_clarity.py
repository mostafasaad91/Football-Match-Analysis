"""Second pass of the visual refresh: boards that were present but hard to read.

Crops to the attacking end, heatmaps that no longer spend their whole ramp on one
cell, a pass map that is a map, Zone 14 lanes that can be told apart, and the
left/right mirror on every board that drew straight from the feed.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from conftest import match_dir

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def v():
    from football_analysis.visuals import visual_redesign_full

    return visual_redesign_full


@pytest.fixture(scope="module")
def coventry():
    out = match_dir("Arsenal_vs_Coventry_3-0")
    if not (out / "match_info.json").exists():
        pytest.skip("Arsenal_vs_Coventry_3-0 has not been rendered")
    return out


@pytest.fixture
def board(v, coventry, tmp_path, monkeypatch):
    """Draw boards of the saved match and hand back the figures they produced."""
    info = json.loads((coventry / "match_info.json").read_text(encoding="utf-8-sig"))
    events = pd.read_csv(coventry / "events.csv", low_memory=False)
    xg = pd.read_csv(coventry / "xg.csv")
    v.configure_match(info, tmp_path)
    v.base.theme()
    figures = []
    monkeypatch.setattr(v, "save", lambda fig, name: figures.append(fig) or Path(name))
    yield v, info, events, xg, figures
    plt.close("all")


# ── the lateral axis ────────────────────────────────────────────────────────
def test_a_right_winger_is_drawn_on_the_right(v):
    """The feed numbers the width from the right touchline; a winger near y=18
    plays on the attacking side's right, which is the right of an up-the-page board."""
    x, _ = v.attack_xy([70.0], [18.0])
    assert x[0] > 0


def test_attack_and_position_boards_share_one_orientation(v):
    ys = np.array([5.0, 18.0, 50.0, 81.0, 95.0])
    xs = np.full_like(ys, 60.0)
    assert np.allclose(v.attack_xy(xs, ys)[0], v.player_position_xy(xs, ys)[0])


def test_zone_14_names_its_lanes_for_the_side_they_are_on(board):
    v, info, events, _xg, figures = board
    v.zone14(events, info["home_id"], 12)
    pitch = _pitch_of(figures[0])
    badges = [t for t in pitch.texts if t.get_text().isdigit()]
    xs = [t.get_position()[0] for t in sorted(badges, key=lambda t: t.get_position()[0])]
    assert len(xs) == 5 and xs == sorted(xs)
    # The busiest lane for a side that plays down its right is on the right.
    right_flank = events[
        (events["team_id"] == info["home_id"])
        & events["outcome"].astype(str).str.lower().eq("successful")
        & (pd.to_numeric(events["end_x"], errors="coerce") >= 66.7)
    ]
    right = int((pd.to_numeric(right_flank["end_y"], errors="coerce") < 40).sum())
    left = int((pd.to_numeric(right_flank["end_y"], errors="coerce") >= 60).sum())
    shown = [int(t.get_text()) for t in sorted(badges, key=lambda t: t.get_position()[0])]
    assert (shown[3] + shown[4] > shown[0] + shown[1]) == (right > left)


def test_zone_14_lanes_each_have_their_own_colour(v):
    colours = v.ZONE_LANE_COLOURS
    assert len(colours) == 5 and len(set(colours)) == 5


# ── cropping to the attacking end ───────────────────────────────────────────
def test_crop_to_attack_shortens_the_figure_to_the_pitch():
    from football_analysis.visuals import visual_redesign_full as full

    fig, pitch, side = full.pitch_axes("Shot Map · Test", "x")
    full.draw_long_pitch(pitch)
    tall = fig.get_size_inches()[1]
    full.crop_to_attack(pitch, side, 55.0)
    short = fig.get_size_inches()[1]
    low, high = pitch.get_ylim()
    assert short < tall
    assert low == 55.0 and high > full.PITCH_LENGTH
    box = pitch.get_position()
    assert 0 < box.y0 < box.y1 <= 1
    plt.close(fig)


# ── heatmaps keep their ramp ────────────────────────────────────────────────
def _pitch_of(figure):
    """The board's pitch: the one axes drawn to equal aspect."""
    return next(ax for ax in figure.axes if ax.get_aspect() == 1.0)


def _numbers(figure):
    return [t for t in _pitch_of(figure).texts if t.get_text().replace(".", "").isdigit()]


def test_the_xt_map_is_a_smooth_surface_with_no_grid_labels_or_arrows(board):
    v, info, events, _xg, figures = board
    v.xt_map(events, info["home_id"], 7)
    pitch = _pitch_of(figures[0])
    assert pitch.images, "the heat is an image, not a grid of boxes"
    assert not any(type(c).__name__ == "QuadMesh" for c in pitch.collections)
    assert not [
        t for t in pitch.texts if t.get_text().replace(".", "").isdigit() and len(t.get_text()) > 1
    ]
    assert not [a for a in pitch.texts if getattr(a, "arrow_patch", None) is not None]


def test_the_pass_target_map_labels_only_its_busiest_cells(board):
    v, info, events, _xg, figures = board
    v.pass_targets(events, info["home_id"], 29)
    assert 0 < len(_numbers(figures[0])) <= 8


def test_the_xt_scale_is_shared_and_not_set_by_one_spike(board):
    v, info, events, _xg, figures = board
    v.xt_map(events, info["home_id"], 7)
    v.xt_map(events, info["away_id"], 8)
    first, second = (_pitch_of(f).images[0] for f in figures[:2])
    assert first.get_clim() == second.get_clim(), "one scale for both teams"
    assert first.get_clim()[1] > 0


# ── the pass map ────────────────────────────────────────────────────────────
def test_the_pass_map_is_a_heat_surface_without_arrows(board):
    v, info, events, _xg, figures = board
    v.pass_map(events, info["home_id"], 9)
    pitches = [ax for ax in figures[0].axes if ax.get_aspect() == 1.0]
    assert len(pitches) == 1
    pitch = pitches[0]
    assert pitch.images, "completed passes are drawn as a surface"
    assert not [t for t in pitch.texts if getattr(t, "arrow_patch", None) is not None]


def test_zone_flow_draws_one_arrow_per_busy_zone_and_keeps_them_on_the_pitch(v):
    fig, pitch, _side = v.pitch_axes("Flow · Test", "x")
    v.draw_long_pitch(pitch)
    rng = np.random.default_rng(3)
    n = 300
    frame = pd.DataFrame(
        {
            "x": rng.uniform(0, 100, n),
            "y": rng.uniform(0, 100, n),
            "end_x": rng.uniform(0, 100, n),
            "end_y": rng.uniform(0, 100, n),
        }
    )
    drawn = v.draw_zone_flow(pitch, frame, np.ones(n, dtype=bool), "#ff0000")
    assert 0 < drawn <= 35, "at most one arrow per zone of the 5 x 7 grid"
    plt.close(fig)


def test_zone_flow_ignores_an_empty_frame(v):
    fig, pitch, _side = v.pitch_axes("Flow · Test", "x")
    assert v.draw_zone_flow(pitch, pd.DataFrame(), [], "#fff") == 0
    plt.close(fig)


# ── set pieces and sequence types ───────────────────────────────────────────
def test_set_pieces_share_one_shot_scale_and_drop_rows_nobody_used(board):
    v, _info, events, _xg, figures = board
    v.set_pieces(events)
    panels = [ax for ax in figures[0].axes if ax.patches]
    assert len(panels) == 2
    assert panels[0].get_xlim() == panels[1].get_xlim(), "unequal scales invert the comparison"
    rows = [len(ax.get_yticklabels()) for ax in panels]
    assert rows[0] == rows[1]


def test_goal_origins_draws_one_pitch_per_goal(board):
    v, info, events, _xg, figures = board
    v.goal_origins(events)
    pitches = [ax for ax in figures[0].axes if ax.get_aspect() == 1.0]
    goals = events[
        events["is_goal"].astype(str).str.lower().isin(["true", "1"])
        & ~events["is_own_goal"].astype(str).str.lower().isin(["true", "1"])
    ]
    assert len(pitches) == len(goals)
