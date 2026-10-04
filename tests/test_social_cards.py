"""The timeline graphics: every card draws in both frames, into one folder, from the saved block."""

import pytest
from PIL import Image

from football_analysis.paths import OUTPUT_DIR
from football_analysis.visuals import league_boards as lb
from football_analysis.visuals import social_cards as sc

BLOCK = OUTPUT_DIR / "aggregates" / "England_Premier_League_2026-2027_R1-5"
GROUPS = {"Attack", "Defence", "Pressing", "Passing"}


@pytest.fixture(scope="module")
def league():
    if not (BLOCK / "teams_totals.csv").exists():
        pytest.skip("the Premier League block has not been aggregated")
    return lb.League.load(BLOCK, packages=OUTPUT_DIR / "England_Premier_League" / "2026-2027")


def test_clubs_and_players_each_cover_all_four_groups():
    for kind in "tp":
        groups = {card.group for card in sc.CARDS if card.name.startswith(kind)}
        assert groups == GROUPS


def test_card_ids_are_unique_and_numbered():
    names = [card.name for card in sc.CARDS]
    assert len(names) == len(set(names))
    assert all(n[0] in "tp" and n[1:].isdigit() for n in names)


def test_each_frame_is_the_size_it_says(league, tmp_path, monkeypatch):
    monkeypatch.setattr(league, "folder", tmp_path)
    for name, size in (("4x5", (1200, 1500)), ("16x9", (1600, 900))):
        paths = sc.build_all(league, only=["t05", "p03"], formats=(name,))
        assert len(paths) == 2
        for path in paths:
            with Image.open(path) as image:
                assert image.size == size, path.name
            assert path.parent.name == f"twitter_{name}"
            assert path.parent.parent.name == "visuals"


def test_every_card_draws_in_both_frames(league, tmp_path, monkeypatch):
    monkeypatch.setattr(league, "folder", tmp_path)
    paths = sc.build_all(league)
    assert len(paths) == 2 * len(sc.CARDS)
    assert all(p.stat().st_size > 30_000 for p in paths)


def test_the_frame_is_reset_after_drawing(league, tmp_path, monkeypatch):
    monkeypatch.setattr(league, "folder", tmp_path)
    sc.build_all(league, only=["t05"], formats=("16x9",))
    assert sc.GEO is sc.FOUR_BY_FIVE


def test_a_card_filename_carries_its_id(league, tmp_path, monkeypatch):
    monkeypatch.setattr(league, "folder", tmp_path)
    (path,) = sc.build_all(league, only=["t05"], formats=("4x5",))
    assert path.name.startswith("t05_")
