"""Sentences in the report must agree with the numbers printed beside them.

Several readings named a fixed side rather than the leading one. Arsenal 3-0
Manchester City printed "Man City's curve finished above Arsenal's" on the xG
flow page against 1.88 xG to 1.08, and "Arsenal attempted 9 shots; Man City
also attempted 12" on the match story. Both render perfectly and are simply
wrong, which is the only kind of defect this file is looking for.

The check is run over a fixture and over its mirror image — the same match with
the two sides swapped — because naming the away side is indistinguishable from
naming the leader until the leader is the home side.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from conftest import match_dir
from tactical_pdf_report import (
    _lead,
    _legacy_visual_explanation,
    _section_copy,
    build_context,
    visual_explanation,
)

ROOT = Path(__file__).resolve().parent.parent


def _context(match="Arsenal_vs_Man_City_3-0"):
    out = match_dir(match)
    if not (out / "match_info.json").exists():
        pytest.skip(f"{match} has not been rendered")
    info = json.loads((out / "match_info.json").read_text(encoding="utf-8"))
    return build_context(
        pd.read_csv(out / "events.csv"),
        pd.read_csv(out / "xg.csv"),
        pd.read_csv(out / "team_advanced_metrics.csv"),
        pd.read_csv(out / "player_sequence_metrics.csv"),
        info,
    )


def _mirrored(context):
    """The same match with the sides swapped, so 'home' and 'leader' differ."""
    swapped = dict(context)
    for key in list(context):
        if key.startswith("home_"):
            twin = "away_" + key[5:]
            if twin in context:
                swapped[key], swapped[twin] = context[twin], context[key]
    swapped["home"], swapped["away"] = context["away"], context["home"]
    return swapped


# --------------------------------------------------------------------------
# the helper every such sentence now asks
# --------------------------------------------------------------------------

def test_the_leader_is_whichever_side_leads():
    assert _lead("H", "A", 2.0, 1.0) == ("H", "A", False)
    assert _lead("H", "A", 1.0, 2.0) == ("A", "H", False)


def test_a_tie_inside_the_tolerance_reports_level():
    _leader, _trailer, level = _lead("H", "A", 1.00, 1.02, tolerance=0.05)
    assert level


def test_an_unusable_value_does_not_raise():
    assert _lead("H", "A", None, 1.0) == ("H", "A", False)
    assert _lead("H", "A", "n/a", "n/a") == ("H", "A", False)


# --------------------------------------------------------------------------
# the sentences
# --------------------------------------------------------------------------

def _xg_flow_claim(context):
    text = visual_explanation(Path("01_xg_flow.png"), context)
    return next(part for part in text.split(". ") if "curve" in part)


@pytest.mark.parametrize("mirror", [False, True])
def test_the_xg_curve_names_the_side_that_created_more(mirror):
    context = _context()
    if mirror:
        context = _mirrored(context)
    leader, trailer, _level = _lead(
        context["home"], context["away"], context["home_xG"], context["away_xG"])
    claim = _xg_flow_claim(context)
    assert claim.startswith(f"{leader}'s curve finished above {trailer}'s"), claim


@pytest.mark.parametrize("mirror", [False, True])
def test_the_shot_volume_card_never_says_two_numbers_are_equal(mirror):
    context = _context()
    if mirror:
        context = _mirrored(context)
    head, body = _section_copy(context)["Match Story"]["data"][0]
    level = int(context["home_shots"]) == int(context["away_shots"])
    assert ("level" in head.lower()) == level, (head, body)
    assert " also attempted" not in body, body


@pytest.mark.parametrize("mirror", [False, True])
def test_chance_quality_names_the_side_with_the_better_xg_per_shot(mirror):
    context = _context()
    if mirror:
        context = _mirrored(context)
    leader, _trailer, level = _lead(
        context["home"], context["away"],
        context["home_xG_per_shot"], context["away_xG_per_shot"], tolerance=0.005)
    text = _legacy_visual_explanation(Path("xg_summary.png"), context)
    if not level:
        assert f"favoured {leader}" in text, text


@pytest.mark.parametrize("mirror", [False, True])
def test_the_overview_names_the_territory_and_the_xg_sides_correctly(mirror):
    context = _context()
    if mirror:
        context = _mirrored(context)
    tilt_leader, _t, _l = _lead(context["home"], context["away"],
                                context["home_field_tilt"], context["away_field_tilt"])
    xg_leader, _x, xg_level = _lead(context["home"], context["away"],
                                    context["home_xG"], context["away_xG"],
                                    tolerance=0.05)
    text = _legacy_visual_explanation(Path("match_stats.png"), context)
    assert tilt_leader in text, text
    if not xg_level and xg_leader != tilt_leader:
        assert f"while {xg_leader} produced the stronger xG return" in text, text


def test_the_contradiction_is_only_claimed_when_there_is_one():
    """One side leading both is not a contradiction and must not be called one."""
    context = _context()
    context["home_field_tilt"], context["away_field_tilt"] = 70.0, 30.0
    context["home_xG"], context["away_xG"] = 2.0, 1.0
    text = _legacy_visual_explanation(Path("match_stats.png"), context)
    assert "central contradiction" not in text, text
    assert "the xG return followed the territory" in text, text
