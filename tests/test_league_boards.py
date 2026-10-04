"""League boards: the pure parts, and the boards themselves on the saved block."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_analysis.paths import OUTPUT_DIR
from football_analysis.visuals import league_boards as lb
from football_analysis.visuals import visual_redesign_preview as base

BLOCK = OUTPUT_DIR / "aggregates" / "England_Premier_League_2026-2027_R1-5"
PACKAGES = OUTPUT_DIR / "England_Premier_League" / "2026-2027"


# ── pure helpers ─────────────────────────────────────────────────────────────
def test_the_best_in_the_league_gets_a_full_slice_and_the_worst_none():
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("abcd"))
    ranks = lb.percentile_rank(series, lower_is_better=False)
    assert ranks["a"] == 0.0 and ranks["d"] == 1.0
    inverted = lb.percentile_rank(series, lower_is_better=True)
    assert inverted["a"] == 1.0 and inverted["d"] == 0.0


def test_tied_values_share_a_rank():
    ranks = lb.percentile_rank(pd.Series([1.0, 2.0, 2.0, 3.0]), lower_is_better=False)
    assert ranks.iloc[1] == ranks.iloc[2]


def test_a_dark_club_colour_is_lifted_until_it_can_be_read():
    lifted = lb.readable("#0A1030")
    r, g, b = (int(lifted[i : i + 2], 16) / 255 for i in (1, 3, 5))
    assert 0.2126 * r + 0.7152 * g + 0.0722 * b >= 0.30
    assert lb.readable("#FFFFFF") == "#ffffff"


def test_goalkeepers_are_recognised_by_the_role_word():
    frame = pd.DataFrame({"role": ["Goalkeeper", "Defender", "GK", "Unknown"]})
    assert list(lb.is_goalkeeper(frame)) == [True, False, True, False]


def test_a_surname_is_the_last_word():
    assert lb.surname("Bukayo Saka") == "Bukayo Saka".split()[-1]
    assert lb.surname("Raya") == "Raya"


def test_the_ordinal_suffix():
    assert [lb._ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22)] == [
        "st",
        "nd",
        "rd",
        "th",
        "th",
        "th",
        "th",
        "st",
        "nd",
    ]


# ── the boards on the saved block ───────────────────────────────────────────────
@pytest.fixture(scope="module")
def league():
    if not (BLOCK / "teams_totals.csv").exists():
        pytest.skip("the Premier League block has not been aggregated")
    base.theme()
    return lb.League.load(BLOCK, packages=PACKAGES)


def test_every_club_has_a_colour_and_a_crest_id(league):
    teams = set(league.teams["team"])
    assert teams <= set(league.colours) and teams <= set(league.ids)


def test_every_league_board_draws_a_png(league, tmp_path, monkeypatch):
    monkeypatch.setattr(league, "folder", tmp_path)
    for board in (
        lb.xg_quadrant,
        lb.league_heat_table,
        lb.player_leaderboards,
        lb.style_map,
        lb.over_under,
        lb.chance_sources,
        lb.scoring_and_creating,
        lb.progressing_the_ball,
        lb.winning_the_ball,
        lb.goalkeepers,
    ):
        path = board(league)
        assert path.exists() and path.stat().st_size > 20_000, board.__name__


def test_the_outfield_leaderboards_leave_goalkeepers_out(league):
    pool = lb._qualified(league, 270.0)
    assert not lb.is_goalkeeper(pool).any()


def test_a_club_profile_and_a_player_profile_are_drawn(league, tmp_path, monkeypatch):
    monkeypatch.setattr(league, "folder", tmp_path)
    team = league.teams["team"].iloc[0]
    assert lb.team_profile(league, team).exists()
    pool = lb._qualified(league, 270.0)
    row = pool.iloc[0]
    assert lb.player_profile(league, row["player"], row["team"]).exists()


def test_a_player_below_the_minutes_floor_gets_no_profile(league):
    short = league.per90[league.per90["minutes"] < 270.0]
    if short.empty:
        pytest.skip("everyone qualifies")
    row = short.iloc[0]
    assert lb.player_profile(league, row["player"], row["team"], 270.0) is None


def test_labels_beside_badges_do_not_overlap_each_other(league):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 9))
    t = league.per_match.set_index("team")
    ax.set_xlim(t["xG"].min() - 0.2, t["xG"].max() + 0.2)
    ax.set_ylim(t["xG_against"].min() - 0.2, t["xG_against"].max() + 0.2)
    lb.place_labels(ax, list(zip(t["xG"], t["xG_against"])), list(t.index))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [text.get_window_extent(renderer) for text in ax.texts]
    assert len(boxes) == len(t)
    clashes = sum(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    plt.close(fig)
    assert clashes <= 1, "at most one unavoidable clash between two labels"
