"""Goals prevented and handling: one keeper measure shared by board, card and prose."""

import pandas as pd

from football_analysis.metrics.match_metrics import (
    goalkeeper_name,
    goalkeeper_shot_stopping,
    post_shot_xg,
)

KEEPER_TEAM, SHOOTERS = 1, 2


def _shot(kind, gy=52.0, gz=12.0, xg=0.2, **extra):
    return {
        "team_id": SHOOTERS,
        "player": "Striker",
        "type": kind,
        "is_shot": True,
        "shot_whoscored_type": kind,
        "xG": xg,
        "goal_mouth_y": gy,
        "goal_mouth_z": gz,
        "body_part": "RightFoot",
        **extra,
    }


def _keeper(kind, *qualifiers, player="Keeper"):
    return {
        "team_id": KEEPER_TEAM,
        "player": player,
        "type": kind,
        "qualifier_names": str(list(qualifiers)),
    }


def _match():
    return pd.DataFrame(
        [
            _shot("Goal"),
            _shot("SavedShot"),
            _shot("SavedShot"),
            _shot("Goal", xg=0.79, is_penalty=True),  # left out of goals prevented
            _shot("Goal", is_own_goal=True),  # not a shot at him
            _shot("MissedShots"),
            _keeper("Save", "Collected", "StandingSave"),
            _keeper("Save", "ParriedDanger", "DivingSave"),
            _keeper("Save", "Blocked", player="Defender"),  # an outfield block
            _keeper("Claim", "HighClaim"),
            _keeper("KeeperPickup"),
            _keeper("Error", "LeadingToAttempt", "LeadingToGoal"),
        ]
    )


def test_goals_prevented_is_post_shot_xg_faced_minus_goals_conceded():
    events = _match()
    result = goalkeeper_shot_stopping(events, KEEPER_TEAM, "Keeper")
    faced = events[events["shot_whoscored_type"].isin(["Goal", "SavedShot"])]
    faced = faced[faced.get("is_penalty").ne(True) & faced.get("is_own_goal").ne(True)]
    assert result["on_target_faced"] == 3
    assert result["goals_conceded"] == 1
    assert result["psxg_faced"] == round(float(post_shot_xg(faced).sum()), 2)
    assert result["goals_prevented"] == round(result["psxg_faced"] - 1, 2)


def test_penalties_are_reported_on_their_own():
    result = goalkeeper_shot_stopping(_match(), KEEPER_TEAM, "Keeper")
    assert (result["penalties_faced"], result["penalties_saved"]) == (1, 0)


def test_handling_reads_where_each_save_sent_the_ball():
    result = goalkeeper_shot_stopping(_match(), KEEPER_TEAM, "Keeper")
    assert result["saves"] == 2  # the defender's block is not his
    assert (result["caught"], result["parried_safe"], result["parried_danger"]) == (1, 0, 1)
    assert result["handling_pct"] == 50.0
    assert result["diving_saves"] == 1
    assert result["high_claims"] == 1


def test_an_error_is_counted_once_whatever_it_led_to():
    result = goalkeeper_shot_stopping(_match(), KEEPER_TEAM, "Keeper")
    assert result["errors_to_shot"] == 1
    assert result["errors_to_goal"] == 1


def test_without_a_name_only_keeper_saves_count():
    result = goalkeeper_shot_stopping(_match(), KEEPER_TEAM)
    assert result["saves"] == 2
    assert result["high_claims"] == 0  # not guessed without a keeper


def test_ids_from_a_csv_still_match():
    events = _match().astype({"team_id": float})
    assert goalkeeper_shot_stopping(events, str(KEEPER_TEAM), "Keeper")["saves"] == 2


def test_the_keeper_is_found_from_keeper_only_actions():
    assert goalkeeper_name(_match(), KEEPER_TEAM) == "Keeper"
    assert goalkeeper_name(_match(), SHOOTERS) == ""
