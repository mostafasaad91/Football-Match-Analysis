"""The one thing the data cannot check about itself: who plays for whom.

Every other check in match_sanity reads the frames against each other. A squad
cannot be checked that way, because the pipeline takes it from the provider's
own match feed: the events agree with the squad and the squad agrees with the
events, so a fixture that lists a player for the wrong side is perfectly
coherent and passes everything.

Two ways in, both added because of that. A roster the user keeps, which is the
only outside opinion the project has and is therefore opt-in; and the project's
own match history, which already records who played for whom and needs no
configuration at all.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import match_sanity as ms

ROOT = Path(__file__).resolve().parent.parent
MATCH = "Arsenal_vs_Man_City_3-0"


def _fixture():
    out = ROOT / "output" / MATCH
    if not (out / "players.csv").exists():
        pytest.skip(f"{MATCH} has not been rendered")
    return (
        pd.read_csv(out / "events.csv"),
        pd.read_csv(out / "players.csv"),
        json.loads((out / "match_info.json").read_text(encoding="utf-8")),
    )


def _with_roster(monkeypatch, tmp_path, roster):
    # Capture the real reader before patching: a lambda that calls the patched
    # name calls itself.
    real = ms._known_squads
    (tmp_path / ms.SQUADS_FILE).write_text(
        json.dumps(roster, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(ms, "_known_squads", lambda root=None: real(tmp_path))


def test_a_player_missing_from_your_roster_is_named(monkeypatch, tmp_path):
    events, players, info = _fixture()
    home = players[players["team_id"].eq(int(info["home_id"]))]["name"].astype(str).tolist()
    dropped = home[:2]
    _with_roster(monkeypatch, tmp_path, {info["home_name"]: home[2:]})

    problems = ms.check_players_belong_to_the_squad_you_named(events, players, info)
    assert problems, "a squad missing two of its listed players was not flagged"
    detail = str(problems[0])
    for name in dropped:
        assert name in detail, (name, detail)


def test_a_complete_roster_is_silent(monkeypatch, tmp_path):
    events, players, info = _fixture()
    roster = {
        str(info[f"{side}_name"]): players[players["team_id"].eq(int(info[f"{side}_id"]))]
        ["name"].dropna().astype(str).tolist()
        for side in ("home", "away")
    }
    _with_roster(monkeypatch, tmp_path, roster)
    assert ms.check_players_belong_to_the_squad_you_named(events, players, info) == []


def test_no_roster_means_no_opinion(monkeypatch, tmp_path):
    """Opt-in: the project will not invent a squad it was never given."""
    events, players, info = _fixture()
    monkeypatch.setattr(ms, "_known_squads", lambda root=None: {})
    assert ms.check_players_belong_to_the_squad_you_named(events, players, info) == []


def test_a_team_you_did_not_list_is_left_alone(monkeypatch, tmp_path):
    """A partial roster checks the teams it covers and nothing else."""
    events, players, info = _fixture()
    _with_roster(monkeypatch, tmp_path, {"Some Other Club": ["A Player"]})
    assert ms.check_players_belong_to_the_squad_you_named(events, players, info) == []


def test_an_accent_or_spelling_difference_is_not_a_stranger(monkeypatch, tmp_path):
    """A roster typed by hand will not carry the provider's diacritics."""
    events, players, info = _fixture()
    home_id = int(info["home_id"])
    home = players[players["team_id"].eq(home_id)]["name"].dropna().astype(str).tolist()
    plain = [ms._fold(n).upper() for n in home]  # no accents, no spaces, no case
    _with_roster(monkeypatch, tmp_path, {info["home_name"]: plain})
    assert ms.check_players_belong_to_the_squad_you_named(events, players, info) == []


def test_the_history_check_is_quiet_without_a_database(monkeypatch):
    """It reads the project's own history; absent, it has nothing to say."""
    events, players, info = _fixture()
    monkeypatch.setattr(ms, "__file__", str(ROOT / "nowhere" / "match_sanity.py"))
    assert ms.check_no_player_changed_team_since_a_stored_match(events, players, info) == []


def test_folding_ignores_accents_case_and_punctuation():
    assert ms._fold("Martin Ødegaard") == ms._fold("martin odegaard")
    assert ms._fold("Rúben Dias") == ms._fold("Ruben  Dias")
    assert ms._fold("Nott'm Forest") == ms._fold("nottm forest")
    assert ms._fold("A") != ms._fold("B")


def test_the_real_fixture_passes_every_check():
    """No regression: the shipped fixture stays coherent."""
    events, players, info = _fixture()
    xg = pd.read_csv(ROOT / "output" / MATCH / "xg.csv")
    assert ms.inspect(events, players, xg, info) == []


# --------------------------------------------------------------------------
# a row with no player is not a player
# --------------------------------------------------------------------------

def test_a_blank_player_name_is_not_a_stranger():
    """The shape that refused Casa Pia vs Benfica an article.

    dropna removes a missing cell but not an empty string, so one event with
    ``player = ""`` counted as a player who is not in the squad — and the
    message named him as nothing at all. It only bites before the frames are
    written: a blank field comes back from CSV as NaN, so the same fixture
    passed when re-read from disk and failed while it was being built.
    """
    players = pd.DataFrame({
        "name": ["Ana", "Beto", "Caio", "Dinis"],
        "team_id": [1, 1, 2, 2],
    })
    events = pd.DataFrame({
        "player": ["Ana", "", "Beto", None, "Caio", "   ", "Dinis"],
        "team_id": [1, 1, 1, 1, 2, 2, 2],
    })
    info = {"home_id": 1, "away_id": 2, "home_name": "Home", "away_name": "Away"}
    assert ms.check_event_players_belong_to_their_team(events, players, info) == []


def test_a_real_stranger_is_still_caught():
    """The blank-name fix must not blind the check to a genuine one."""
    players = pd.DataFrame({"name": ["Ana", "Beto"], "team_id": [1, 2]})
    events = pd.DataFrame({
        "player": ["Ana", "", "Someone Else", "Beto"],
        "team_id": [1, 1, 1, 2],
    })
    info = {"home_id": 1, "away_id": 2, "home_name": "Home", "away_name": "Away"}
    problems = ms.check_event_players_belong_to_their_team(events, players, info)
    assert len(problems) == 1
    assert "Someone Else" in str(problems[0])


def test_a_blank_name_does_not_read_as_playing_for_both_sides():
    players = pd.DataFrame({
        "name": ["Ana", "", None, "Beto"],
        "team_id": [1, 1, 2, 2],
    })
    info = {"home_id": 1, "away_id": 2, "home_name": "Home", "away_name": "Away"}
    assert ms.check_no_player_appears_for_both_sides(None, players, info) == []


def test_surrounding_whitespace_is_not_a_different_player():
    players = pd.DataFrame({"name": ["Ana ", "Beto"], "team_id": [1, 2]})
    events = pd.DataFrame({"player": ["Ana", "Beto "], "team_id": [1, 2]})
    info = {"home_id": 1, "away_id": 2, "home_name": "Home", "away_name": "Away"}
    assert ms.check_event_players_belong_to_their_team(events, players, info) == []
