"""A higher xG total is not the same claim as a better performance.

The article opened "PSG won the match Aston Villa played better in" and then,
two paragraphs later, printed the figure that contradicts it: 1.99 of Villa's
2.15 expected goals arrived while they were behind. Ninety-three per cent of
the supposed performance came after the game had gone against them, against a
side that had dropped its line and stopped committing bodies. Level, they
managed 0.16.

The evidence was in the frames the whole time. The prose simply formed its
verdict from the total and never looked at the split underneath it — which is
how a team that was poor gets published as the better side.

Nothing here reaches outside the data: the judgement is built from the
game-state splits the pipeline already computes, so a reader can check it.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from match_verdict import CHASING_SHARE, LEVEL_XG_FLOOR, read_match
from conftest import match_dir

ROOT = Path(__file__).resolve().parent.parent


def _fixture(match):
    out = match_dir(match)
    if not (out / "match_info.json").exists():
        pytest.skip(f"{match} has not been rendered")
    return (
        pd.read_csv(out / "team_advanced_metrics.csv"),
        pd.read_csv(out / "xg.csv"),
        json.loads((out / "match_info.json").read_text(encoding="utf-8")),
    )


def _frames(*, home_xg, away_xg, home_goals, away_goals,
            home_split, away_split):
    """Two sides, with their xG split by the state it was created in."""
    team_metrics = pd.DataFrame([
        {"side": "home", "team": "Home",
         "game_state_leading_xG": home_split[0],
         "game_state_drawing_xG": home_split[1],
         "game_state_trailing_xG": home_split[2],
         "box_entries": 10, "final_third_entries": 30},
        {"side": "away", "team": "Away",
         "game_state_leading_xG": away_split[0],
         "game_state_drawing_xG": away_split[1],
         "game_state_trailing_xG": away_split[2],
         "box_entries": 10, "final_third_entries": 30},
    ])
    xg = pd.DataFrame([
        {"team": "Home", "xG": home_xg, "goals": home_goals,
         "xG_per_shot": 0.12, "big_chances": 2},
        {"team": "Away", "xG": away_xg, "goals": away_goals,
         "xG_per_shot": 0.12, "big_chances": 2},
    ])
    info = {"home_id": 1, "away_id": 2, "home_name": "Home", "away_name": "Away"}
    return team_metrics, xg, info


def test_a_loser_who_only_created_while_behind_is_named_as_chasing():
    """The shape the prose used to publish as the better performance."""
    verdict = read_match(*_frames(
        home_xg=1.10, away_xg=2.15, home_goals=2, away_goals=1,
        home_split=(0.56, 0.54, 0.0), away_split=(0.0, 0.16, 1.99)))
    assert verdict.loser == "Away"
    assert verdict.loser_out_created_winner
    assert verdict.loser_was_only_chasing
    assert verdict.summary() == "chasing"


def test_a_loser_who_created_while_level_deserved_more():
    """Out-created the winner in a fair contest: the old reading was right."""
    verdict = read_match(*_frames(
        home_xg=1.10, away_xg=2.15, home_goals=2, away_goals=1,
        home_split=(0.56, 0.54, 0.0), away_split=(0.0, 1.35, 0.80)))
    assert verdict.loser_out_created_winner
    assert not verdict.loser_was_only_chasing
    assert verdict.summary() == "deserved more"


def test_a_loser_who_created_less_is_not_flattered_into_a_claim():
    verdict = read_match(*_frames(
        home_xg=2.00, away_xg=0.60, home_goals=2, away_goals=0,
        home_split=(1.40, 0.60, 0.0), away_split=(0.0, 0.05, 0.55)))
    assert not verdict.loser_out_created_winner
    assert verdict.summary() == "matched the result"


def test_a_draw_has_no_loser_to_judge():
    verdict = read_match(*_frames(
        home_xg=1.20, away_xg=1.10, home_goals=1, away_goals=1,
        home_split=(0.4, 0.5, 0.3), away_split=(0.3, 0.5, 0.3)))
    assert verdict.winner is None and verdict.loser is None
    assert verdict.summary() == "level"


def test_both_conditions_are_needed_before_the_total_is_called_a_chase():
    """A high chasing share alone is ordinary — every trailing side creates."""
    high_share_but_real = read_match(*_frames(
        home_xg=1.10, away_xg=2.60, home_goals=2, away_goals=1,
        home_split=(0.56, 0.54, 0.0),
        away_split=(0.0, LEVEL_XG_FLOOR + 0.2, 1.85)))
    beaten = high_share_but_real.of("Away")
    assert beaten.chasing_share >= CHASING_SHARE
    assert not beaten.flattered, "a side that created while level is not chasing"


@pytest.mark.parametrize("match", ["PSG_vs_Aston_Villa_2-1",
                                   "Arsenal_vs_Man_City_3-0",
                                   "Arsenal_vs_Coventry_3-0"])
def test_the_real_fixtures_are_judged_from_their_own_frames(match):
    verdict = read_match(*_fixture(match))
    for side in (verdict.home, verdict.away):
        assert 0.0 <= side.chasing_share <= 1.0, side
        assert abs((side.level_xg + side.chasing_xg) - side.xg) < 0.02, side


def test_villa_is_the_case_this_exists_for():
    verdict = read_match(*_fixture("PSG_vs_Aston_Villa_2-1"))
    villa = verdict.of("Aston Villa")
    assert villa.chasing_share > 0.9, villa.chasing_share
    assert villa.level_xg < 0.2, villa.level_xg
    assert verdict.loser_was_only_chasing


def test_the_article_and_the_report_reach_the_same_verdict():
    """Two documents, one judgement — the drift this session kept finding."""
    from match_article import build_article
    from tactical_pdf_report import _section_copy, build_context

    out = match_dir("PSG_vs_Aston_Villa_2-1")
    if not (out / "match_info.json").exists():
        pytest.skip("fixture not rendered")
    info = json.loads((out / "match_info.json").read_text(encoding="utf-8"))
    frames = [pd.read_csv(out / n) for n in
              ("events.csv", "xg.csv", "team_advanced_metrics.csv",
               "player_sequence_metrics.csv")]

    article = build_article(*frames, info, out)
    context = build_context(*frames, info)
    cards = dict(_section_copy(context)["Match Story"]["data"])

    assert "chase" in article.sections[0].heading.lower()
    assert any("chase" in heading.lower() for heading in cards)
    assert "played better in" not in " ".join(
        p for s in article.sections for p in s.paragraphs)
