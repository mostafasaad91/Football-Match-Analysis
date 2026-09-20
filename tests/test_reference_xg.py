"""Opta's shot map teaches the model by default, and publishes only when asked.

Every test supplies its shot map, so nothing here touches the network.
"""

import pandas as pd
import pytest

import reference_xg as RX
import xg_alignment as XA


@pytest.fixture(autouse=True)
def publishing(monkeypatch, request):
    """Print Opta's values, which is what most of these tests are about.

    The shipped default collects the map and publishes our own model, so a
    test that wants the other mode says so by asking for this fixture's
    absence -- see the two that carry the ``collecting`` marker.
    """
    if "collecting" not in request.keywords:
        monkeypatch.setenv("MATCH_ANALYSIS_REFERENCE_XG", "publish")


INFO = {"home_id": 211, "away_id": 13, "home_name": "Brighton", "away_name": "Arsenal",
        "date": "2026-09-19", "score": "3 : 0", "competition": "Premier League"}


def _event(event_id, team, minute, player, x, y, xg, period="SecondHalf", **flags):
    row = {"event_id": event_id, "team_id": team, "minute": minute, "period": period,
           "player": player, "x": x, "y": y, "is_shot": True, "is_own_goal": False,
           "is_penalty": False, "is_penalty_shootout": False, "xG": xg,
           "xg_source": "engine"}
    row.update(flags)
    return row


def _shot(team, minute, player, x, y, xg, added=None, period="SecondHalf", **extra):
    return {"teamId": team, "min": minute, "minAdded": added, "period": period,
            "playerName": player, "x": x, "y": y, "expectedGoals": xg, **extra}


PAYLOAD = {
    "fixture": {"id": "5795457", "url": "/matches/arsenal-vs-brighton-hove-albion/3bfk5g",
                "home": "Brighton & Hove Albion", "away": "Arsenal",
                "utc": "2026-09-19", "score": "3 - 0"},
    "teams": [{"id": 10204, "name": "Brighton & Hove Albion"}, {"id": 9825, "name": "Arsenal"}],
    "shots": [
        _shot(9825, 78, "Kai Havertz", 92.3, 32.0, 0.632076,
              isOnTarget=True, isBlocked=False, expectedGoalsOnTarget=0.816809),
        _shot(10204, 90, "Matthew O'Riley", 97.2, 33.3, 0.301452, added=0,
              isOnTarget=True, isBlocked=True, expectedGoalsOnTarget=0.4),
        # WhoScored writes this one as minute 95; FotMob as 90+5.
        _shot(9825, 90, "Martín Zubimendi", 93.9, 30.2, 0.028984, added=5),
        # Not in the event feed at all.
        _shot(9825, 90, "Declan Rice", 74.0, 25.6, 0.015632, added=3),
        _shot(10204, 30, "Pascal Groß", 90.0, 34.0, 0.79, situation="Penalty"),
    ],
}


def _events():
    return pd.DataFrame([
        _event(1, 13, 77, "Kai Havertz", 86.8, 47.4, 0.2779),
        _event(2, 211, 89, "Matt O'Riley", 91.8, 49.1, 0.4492),
        _event(3, 13, 95, "Martín Zubimendi", 88.4, 44.2, 0.0181),
        _event(4, 211, 30, "Pascal Groß", 88.5, 50.0, 0.79, is_penalty=True),
        _event(5, 13, 60, "Ezri Konsa", 94.6, 49.5, 0.0661, is_own_goal=True),
        {"event_id": 6, "team_id": 13, "minute": 10, "is_shot": False, "type": "Pass"},
    ])


def test_paired_shots_take_opta_and_say_so():
    info = dict(INFO)
    out, note = RX.apply_reference_xg(_events(), info, payload=PAYLOAD)
    by_id = out.set_index("event_id")
    assert by_id.loc[1, "xG"] == pytest.approx(0.6321)
    assert by_id.loc[2, "xG"] == pytest.approx(0.3015)
    assert (by_id.loc[[1, 2, 3], "xg_source"] == RX.SOURCE).all()
    assert "3 of 3 shots" in note and "1 FotMob shot(s) not in the event feed" in note
    assert info["xg_reference_source"].startswith("Opta via FotMob for 3 of 3")


def test_post_shot_value_rides_along_only_for_shots_that_reached_the_keeper():
    out, _ = RX.apply_reference_xg(_events(), dict(INFO), payload=PAYLOAD)
    by_id = out.set_index("event_id")
    assert by_id.loc[1, "xgot_reference"] == pytest.approx(0.8168)
    # A blocked attempt never reached the keeper, whatever FotMob files it under.
    assert pd.isna(by_id.loc[2, "xgot_reference"])
    assert pd.isna(by_id.loc[3, "xgot_reference"])


def test_post_shot_xg_prefers_opta_and_falls_back_to_placement():
    from match_metrics import post_shot_xg

    shots = pd.DataFrame([
        {"xG": 0.63, "shot_whoscored_type": "SavedShot", "goal_mouth_y": 50.0,
         "goal_mouth_z": 5.0, "xgot_reference": 0.8168},
        {"xG": 0.63, "shot_whoscored_type": "SavedShot", "goal_mouth_y": 50.0,
         "goal_mouth_z": 5.0, "xgot_reference": float("nan")},
        {"xG": 0.10, "shot_whoscored_type": "MissedShots", "xgot_reference": 0.5},
    ])
    values = post_shot_xg(shots)
    assert values.iloc[0] == pytest.approx(0.8168)
    assert values.iloc[1] != pytest.approx(0.8168) and 0 < values.iloc[1] < 0.97
    # Off target is zero post-shot value, even if something upstream filed one.
    assert values.iloc[2] == 0.0


def test_added_time_is_read_as_one_running_minute():
    out, _ = RX.apply_reference_xg(_events(), dict(INFO), payload=PAYLOAD)
    assert out.set_index("event_id").loc[3, "xG"] == pytest.approx(0.029)


def test_penalties_and_own_goals_keep_the_engine_value():
    out, _ = RX.apply_reference_xg(_events(), dict(INFO), payload=PAYLOAD)
    by_id = out.set_index("event_id")
    assert by_id.loc[4, "xG"] == 0.79 and by_id.loc[4, "xg_source"] == "engine"
    assert by_id.loc[5, "xG"] == 0.0661 and by_id.loc[5, "xg_source"] == "engine"


def test_a_shot_by_the_other_side_in_the_same_minute_is_not_taken():
    events = pd.DataFrame([_event(1, 211, 78, "Kai Havertz", 86.8, 47.4, 0.2779)])
    out, note = RX.apply_reference_xg(events, dict(INFO), payload=PAYLOAD)
    assert out.loc[0, "xG"] == 0.2779
    assert "values kept" in note


def test_switched_off_leaves_everything_alone(monkeypatch):
    monkeypatch.setenv("MATCH_ANALYSIS_REFERENCE_XG", "0")
    events = _events()
    out, note = RX.apply_reference_xg(events, dict(INFO), payload=PAYLOAD)
    assert out is events and "switched off" in note


@pytest.mark.collecting
def test_by_default_the_map_is_collected_and_our_numbers_published(monkeypatch):
    monkeypatch.delenv("MATCH_ANALYSIS_REFERENCE_XG", raising=False)
    events = _events()
    info = dict(INFO)
    out, note = RX.apply_reference_xg(events, info, payload=PAYLOAD)
    assert out is events
    assert "stored for training" in note and "our model's values published" in note
    assert "xg_reference_source" not in info


@pytest.mark.collecting
def test_collecting_is_also_what_an_unknown_setting_means(monkeypatch):
    monkeypatch.setenv("MATCH_ANALYSIS_REFERENCE_XG", "learn")
    out, note = RX.apply_reference_xg(_events(), dict(INFO), payload=PAYLOAD)
    assert "stored for training" in note


def test_a_broken_shot_map_never_raises():
    out, note = RX.apply_reference_xg(_events(), dict(INFO), payload={"teams": None, "shots": [{"min": "x"}]})
    assert out["xG"].tolist()[:3] == [0.2779, 0.4492, 0.0181]
    assert "values kept" in note


@pytest.mark.parametrize("ours, theirs", [
    (("Man Utd", "Ipswich"), ("Manchester United", "Ipswich Town")),
    (("PSG", "Monaco"), ("Paris Saint-Germain", "AS Monaco")),
    (("RBL", "Borussia M.Gladbach"), ("RB Leipzig", "Borussia Mönchengladbach")),
    (("Bayern", "Bodoe/Glimt"), ("Bayern München", "Bodø/Glimt")),
])
def test_short_names_find_their_fixture(ours, theirs):
    fixture = {"id": "1", "url": "/m", "home": theirs[0], "away": theirs[1],
               "utc": "2026-09-13", "score": "2 - 1"}
    assert RX.match_fixture({"date": "2026-09-13", "home": ours[0], "away": ours[1],
                             "score": "2 : 1"}, [fixture]) is fixture


def test_the_two_manchester_clubs_are_not_one_club():
    fixtures = [{"id": "city", "url": "/c", "home": "Manchester City", "away": "Fulham",
                 "utc": "2026-09-13", "score": "1 - 0"}]
    assert RX.match_fixture({"date": "2026-09-13", "home": "Man Utd", "away": "Fulham",
                             "score": ""}, fixtures) is None


def test_a_different_score_is_a_different_fixture():
    fixture = {"id": "1", "url": "/m", "home": "Brighton & Hove Albion", "away": "Arsenal",
               "utc": "2026-09-19", "score": "0 - 1"}
    assert RX.match_fixture({"date": "2026-09-19", "home": "Brighton", "away": "Arsenal",
                             "score": "3 : 0"}, [fixture]) is None


def test_the_alignment_reader_accepts_the_reference_fit():
    assert XA.REFERENCE_METHOD in XA.METHODS
    stored = {"method": XA.REFERENCE_METHOD, "weights": {"const": 0.0, "logit": 1.0, "bc_dist": 1.0}}
    geometry = {"distance": 20.0, "angle": 0.4, "central": 0.5, "dy": 3.0}
    plain = XA.align(0.1, geometry, {}, {"body_part": "RightFoot"}, stored=stored)
    flagged = XA.align(0.1, geometry, {"is_big": True}, {"body_part": "RightFoot"}, stored=stored)
    assert plain == pytest.approx(0.1)
    assert flagged > plain


def test_interactions_stay_out_of_the_goal_fit():
    assert not XA.INTERACTIONS & set(XA.LEVEL_TERMS)
