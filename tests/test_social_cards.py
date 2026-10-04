"""The timeline graphics: every card draws, at 1200 x 1500, from the saved block."""

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


def test_there_are_two_cards_a_group_for_clubs_and_for_players():
    names = {card.name for card in sc.CARDS}
    assert names == {f"{kind}{i}" for kind in "tp" for i in range(1, 9)}
    for kind in "tp":
        groups = [card.group for card in sc.CARDS if card.name.startswith(kind)]
        assert sorted(groups) == [
            "Attack",
            "Attack",
            "Defence",
            "Defence",
            "Passing",
            "Passing",
            "Pressing",
            "Pressing",
        ]


def test_every_card_draws_a_timeline_sized_png(league, tmp_path, monkeypatch):
    monkeypatch.setattr(league, "folder", tmp_path)
    for path in sc.build_all(league):
        with Image.open(path) as image:
            assert image.size == (1200, 1500), path.name
        assert path.stat().st_size > 30_000, path.name


def test_only_draws_the_cards_asked_for(league, tmp_path, monkeypatch):
    monkeypatch.setattr(league, "folder", tmp_path)
    paths = sc.build_all(league, only=["t3", "p5"])
    assert sorted(p.name[:2] for p in paths) == ["p5", "t3"]
