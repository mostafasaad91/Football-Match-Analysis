"""The assist is read off the shot, where the provider records it.

Every goal on disk was published without an assist because the parse looked
for the player in the value of an ``IntentionalAssist`` qualifier, which the
feed leaves empty. These pin the rule that replaced it.
"""

from football_match_analysis import _events_by_team_and_id, assist_provider


def _q(name, value=None):
    qualifier = {"type": {"displayName": name}}
    if value is not None:
        qualifier["value"] = value
    return qualifier


def _pass(team, event_id, *kinds):
    return {"teamId": team, "eventId": event_id, "type": {"displayName": "Pass"},
            "qualifiers": [_q(kind) for kind in kinds]}


def _shot(team, *flags, provider=None, related=None):
    shot = {"teamId": team, "eventId": 900, "isShot": True,
            "qualifiers": [_q(flag) for flag in flags]}
    if provider is not None:
        shot["relatedPlayerId"] = provider
    if related is not None:
        shot["relatedEventId"] = related
    return shot


def test_the_provider_comes_from_the_shot_not_the_qualifier_value():
    # IntentionalAssist carries no value in the feed; the old parse found none.
    shot = _shot(13, "Assisted", "IntentionalAssist", provider=361862, related=41)
    events = [_pass(13, 41, "Cross"), shot]
    assert assist_provider(shot, _events_by_team_and_id(events)) == (361862, "Cross")


def test_the_pass_is_found_under_the_shooting_team():
    # eventId counts per team, so the other side has a 41 of its own.
    shot = _shot(13, "Assisted", provider=7, related=41)
    events = [_pass(96, 41, "ThroughBall"), _pass(13, 41, "LayOff"), shot]
    assert assist_provider(shot, _events_by_team_and_id(events)) == (7, "LayOff")


def test_a_shot_with_no_assist_flag_has_no_provider():
    # A related player without the flag is a rebound or a deflection, not an assist.
    shot = _shot(13, "RightFoot", provider=7, related=41)
    assert assist_provider(shot, _events_by_team_and_id([_pass(13, 41), shot])) == (None, None)


def test_an_own_goal_is_nobody_s_assist():
    shot = _shot(13, "Assisted", "OwnGoal", provider=7, related=41)
    assert assist_provider(shot, _events_by_team_and_id([_pass(13, 41), shot])) == (None, None)


def test_the_provider_stands_when_the_pass_is_missing():
    shot = _shot(13, "Assisted", provider="7", related=41)
    assert assist_provider(shot, {}) == (7, None)
