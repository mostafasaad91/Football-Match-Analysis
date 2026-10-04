"""The season graphics: one folder split into clubs and players, all 16:9, none overlapping."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from football_analysis.paths import OUTPUT_DIR
from football_analysis.visuals import league_boards as lb
from football_analysis.visuals import social_cards as sc

BLOCK = OUTPUT_DIR / "aggregates" / "England_Premier_League_2026-2027_R1-5"


@pytest.fixture(scope="module")
def league():
    if not (BLOCK / "teams_totals.csv").exists():
        pytest.skip("the Premier League block has not been aggregated")
    return lb.League.load(BLOCK, packages=OUTPUT_DIR / "England_Premier_League" / "2026-2027")


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


# ── overlap ──────────────────────────────────────────────────────────────────
def test_overlapping_marks_are_spread_until_they_do_not_touch():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    xs = np.array([5.0, 5.05, 5.1, 5.02, 2.0])
    ys = np.array([5.0, 5.0, 5.05, 4.95, 8.0])
    moved = lb.spread_apart(ax, xs, ys, radius_px=20)
    pixels = ax.transData.transform(moved)
    distances = [np.hypot(*(pixels[i] - pixels[j])) for i in range(4) for j in range(i + 1, 4)]
    plt.close(fig)
    assert min(distances) >= 38, "four stacked marks end up a mark's width apart"
    assert np.allclose(moved[4], [2.0, 8.0]), "a mark that overlaps nothing does not move"


def test_labels_do_not_sit_on_each_other_or_on_a_badge():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    rng = np.random.default_rng(1)
    pts = rng.uniform(2, 8, size=(14, 2))
    names = [f"Club number {i}" for i in range(14)]
    texts = lb.place_labels(
        ax, [tuple(p) for p in pts], names, radius_px=14, fontsize=8, badges=True
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in texts]
    clashes = sum(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    plt.close(fig)
    assert clashes <= 1


def test_the_panel_title_shrinks_to_stay_clear_of_the_chart(league):
    fig = sc.frame(league, "Attack", "A very long title that has to fit in the panel", "Subtitle")
    renderer = fig.canvas.get_renderer()
    widest = max(t.get_window_extent(renderer).x1 for t in fig.texts)
    plt_width = sc.SIZE[0] * fig.dpi
    assert widest <= sc.PANEL_RIGHT * plt_width + 2
    import matplotlib.pyplot as plt

    plt.close(fig)


# ── structure ────────────────────────────────────────────────────────────────
def test_there_is_a_ranking_for_every_listed_measure_and_ids_are_unique(league):
    cards = sc.cards(league)
    keys = {(c.scope, c.name) for c in cards}
    assert len(keys) == len(cards)
    assert {c.scope for c in cards} == {"teams", "players"}
    for group in ("Attack", "Defence", "Pressing", "Passing"):
        for scope in ("teams", "players"):
            assert any(c.group == group and c.scope == scope for c in cards), (scope, group)


def test_clubs_and_players_each_get_a_profile_per_entry(league):
    profile_cards = sc.profiles(league)
    clubs = [c for c in profile_cards if c.scope == "teams"]
    assert len(clubs) == len(league.teams)
    assert len([c for c in profile_cards if c.scope == "players"]) == len(
        lb._qualified(league, sc.MIN_MINUTES)
    )


def test_a_sample_draws_into_two_folders_at_sixteen_by_nine(league, tmp_path, monkeypatch):
    monkeypatch.setattr(league, "folder", tmp_path)
    paths = sc.build_all(
        league,
        only=[
            "overview_01",
            "overview_02",
            "attack_01",
            "goalkeepers",
            "finishers_and",
            "who_wins",
            "profile_Arsenal",
        ],
        report=print,
    )
    assert {p.parent.name for p in paths} == {"teams", "players"} or {
        p.parent.name for p in paths
    } == {"teams"}
    assert {p.parent.parent.name for p in paths} == {"visuals"}
    assert len(paths) >= 6
    for path in paths:
        with Image.open(path) as image:
            assert image.size == (1600, 900), path.name
        assert path.stat().st_size > 30_000, path.name


def test_a_full_run_leaves_only_teams_and_players_folders(league, tmp_path, monkeypatch):
    monkeypatch.setattr(league, "folder", tmp_path)
    sc.build_all(league, only=["attack_01_", "defence_00_goalkeepers"], with_profiles=False)
    assert sorted(p.name for p in (tmp_path / "visuals").iterdir()) == ["players", "teams"]
