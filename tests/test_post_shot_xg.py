"""xGoT must be post-shot expected goals, and the same number everywhere.

Two separate things were published under the name. The team card summed the
*pre-shot* xG of on-target attempts, which contains no information about where
the ball went — a shot rolled at the keeper and one in the top corner scored
identically — and the report described that as "post-shot xG". The player radar
bucketed the placement qualifier into four hand-picked multipliers, so every
corner priced the same and the exact coordinates the provider records were
rounded into a zone.

`match_metrics.post_shot_xg` already did the job properly, reading the point
where the ball crossed the line. It was written, tested, and called by nothing.

These tests hold the wiring in place: the metric responds to placement, the
radar and the team card agree, and a fixture exported before the fix still
produces a self-consistent package.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from match_metrics import post_shot_xg
from player_radar import player_metrics
from visual_redesign_full import _corrected_xgot

ROOT = Path(__file__).resolve().parent.parent
MATCHES = ["Arsenal_vs_Man_City_3-0", "PSG_vs_Aston_Villa_2-1"]


def _fixture(match):
    out = ROOT / "output" / match
    if not (out / "match_info.json").exists():
        pytest.skip(f"{match} has not been rendered")
    return (
        pd.read_csv(out / "events.csv"),
        pd.read_csv(out / "xg.csv"),
        json.loads((out / "match_info.json").read_text(encoding="utf-8")),
    )


def _shot(xg, gy, gz):
    return pd.DataFrame([{
        "xG": xg, "goal_mouth_y": gy, "goal_mouth_z": gz,
        "shot_whoscored_type": "Goal", "is_shot": True,
    }])


def test_placement_changes_the_value():
    """The whole point: the same chance, struck in two places, is not equal."""
    corner = float(post_shot_xg(_shot(0.30, 55.5, 2.0)).iloc[0])
    at_keeper = float(post_shot_xg(_shot(0.30, 50.0, 2.0)).iloc[0])
    assert corner > at_keeper * 1.5, (corner, at_keeper)


def test_a_well_placed_shot_can_exceed_its_pre_shot_value():
    """The old cap at team xG made this impossible, which was the tell."""
    assert float(post_shot_xg(_shot(0.30, 55.5, 2.0)).iloc[0]) > 0.30


def test_an_off_target_shot_is_worth_nothing_post_shot():
    off = pd.DataFrame([{
        "xG": 0.5, "goal_mouth_y": 50.0, "goal_mouth_z": 2.0,
        "shot_whoscored_type": "MissedShots", "is_shot": True,
    }])
    assert float(post_shot_xg(off).sum()) == 0.0


@pytest.mark.parametrize("match", MATCHES)
def test_the_radar_and_the_team_card_agree(match):
    """A player's xGoT must sum to the team's, in the same document."""
    # Every player who acted, not the percentile pool: that pool drops anyone
    # under a touch floor, and Haaland's ten touches took his 0.18 out of the
    # sum while leaving it in the team's.
    events, xg, info = _fixture(match)
    corrected = _corrected_xgot(events, xg, info)

    for side in ("home", "away"):
        team_id, name = int(info[f"{side}_id"]), str(info[f"{side}_name"])
        squad = sorted(set(events[events["team_id"].eq(team_id)]["player"]
                           .dropna().astype(str)))
        summed = sum(player_metrics(events, p)["xGOT"] for p in squad)
        stated = float(corrected[corrected["team"].astype(str).eq(name)].iloc[0]["xGoT"])
        # Rounding per player against rounding once for the team.
        assert abs(summed - stated) <= 0.05, (name, summed, stated)


@pytest.mark.parametrize("match", MATCHES)
def test_an_old_export_still_produces_a_consistent_package(match):
    """The correction runs on the frames, not on when they were written."""
    events, xg, info = _fixture(match)
    corrected = _corrected_xgot(events, xg, info)
    for _, row in corrected.iterrows():
        team_id = (int(info["home_id"]) if str(row["team"]) == str(info["home_name"])
                   else int(info["away_id"]))
        expected = round(float(post_shot_xg(events[events["team_id"].eq(team_id)]).sum()), 2)
        assert abs(float(row["xGoT"]) - expected) < 1e-9, (row["team"], row["xGoT"], expected)


def test_the_correction_leaves_an_export_without_placement_alone():
    """Nothing to price from means nothing to change."""
    events, xg, info = _fixture(MATCHES[0])
    stripped = events.drop(columns=["goal_mouth_y", "goal_mouth_z"])
    assert _corrected_xgot(stripped, xg, info)["xGoT"].tolist() == xg["xGoT"].tolist()
