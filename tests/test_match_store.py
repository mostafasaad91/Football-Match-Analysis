import pandas as pd
import pytest

from match_store import (
    list_matches,
    match_identity,
    metric_percentile,
    player_match_log,
    save_match,
    team_match_log,
    team_totals,
)


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "history.db"


def _info(score="2-1", home="Arsenal", away="Chelsea"):
    return {"home_name": home, "away_name": away, "score": score, "date": "2026-03-01"}


def _team_frame(home="Arsenal", away="Chelsea", tilt=60.0):
    return pd.DataFrame(
        [
            {"team": home, "side": "home", "team_id": 1, "field_tilt": tilt, "box_entries": 12},
            {"team": away, "side": "away", "team_id": 2, "field_tilt": 100 - tilt, "box_entries": 5},
        ]
    )


def _player_frame(team="Arsenal"):
    return pd.DataFrame([{"player": "Bukayo Saka", "team": team, "xGChain": 0.8}])


def test_match_id_prefers_the_provider_id_from_the_url():
    url = "https://www.whoscored.com/matches/1903428/live/england-premier-league"
    assert match_identity(_info(), url) == "ws-1903428"
    # Without a URL the key is the two teams — deliberately not the date, so a
    # postponement does not split one fixture into two rows.
    assert match_identity(_info()) == "arsenal-vs-chelsea"
    assert "2026-03-01" not in match_identity(_info())


def test_saving_the_same_fixture_twice_replaces_rather_than_duplicates(db):
    """Re-analysing a match must not double-count it in the history."""
    url = "https://www.whoscored.com/matches/999/live/x"
    save_match(_info(), _team_frame(tilt=60.0), _player_frame(), url=url, db_path=db)
    save_match(_info(), _team_frame(tilt=71.5), _player_frame(), url=url, db_path=db)

    assert len(list_matches(db_path=db)) == 1
    log = team_match_log("Arsenal", db_path=db)
    assert len(log) == 1
    # The second write wins.
    assert float(log.iloc[0]["field_tilt"]) == 71.5


def test_team_match_log_records_opponent_and_goals(db):
    save_match(_info(score="3-1"), _team_frame(), _player_frame(), db_path=db)
    log = team_match_log("Arsenal", db_path=db)
    assert log.iloc[0]["opponent"] == "Chelsea"
    assert int(log.iloc[0]["goals_for"]) == 3
    assert int(log.iloc[0]["goals_against"]) == 1


def test_team_totals_separates_the_sum_from_the_per_match_average(db):
    for index in range(3):
        save_match(
            _info(),
            _team_frame(tilt=50.0 + index * 10),
            None,
            url=f"https://www.whoscored.com/matches/{index}/live/x",
            db_path=db,
        )
    totals = team_totals("Arsenal", db_path=db)
    tilt = totals[totals["metric"] == "field_tilt"].iloc[0]
    assert int(tilt["matches"]) == 3
    assert tilt["total"] == pytest.approx(180.0)
    assert tilt["per_match"] == pytest.approx(60.0)


def test_last_n_filter_takes_the_most_recent(db):
    for index in range(5):
        info = _info()
        info["date"] = f"2026-03-0{index + 1}"
        save_match(
            info,
            _team_frame(tilt=index * 10.0),
            None,
            url=f"https://www.whoscored.com/matches/{index}/live/x",
            db_path=db,
        )
    assert len(team_match_log("Arsenal", limit=2, db_path=db)) == 2


def test_player_log_spans_matches(db):
    for index in range(2):
        save_match(
            _info(),
            _team_frame(),
            _player_frame(),
            url=f"https://www.whoscored.com/matches/{index}/live/x",
            db_path=db,
        )
    assert len(player_match_log("Bukayo Saka", db_path=db)) == 2


def test_percentile_stays_silent_until_there_is_enough_history(db):
    """A percentile from a handful of rows is noise dressed as a number."""
    for index in range(4):
        save_match(
            _info(),
            _team_frame(tilt=index * 10.0),
            None,
            url=f"https://www.whoscored.com/matches/{index}/live/x",
            db_path=db,
        )
    assert metric_percentile("field_tilt", 15.0, db_path=db) is None

    for index in range(4, 12):
        save_match(
            _info(),
            _team_frame(tilt=index * 10.0),
            None,
            url=f"https://www.whoscored.com/matches/{index}/live/x",
            db_path=db,
        )
    value = metric_percentile("field_tilt", 100.0, db_path=db)
    assert value is not None and 0 <= value <= 100


def test_match_key_survives_a_reschedule(db):
    """Two sources dating the same postponed fixture differently must not
    become two rows — the events attach to one and the score to the other."""
    base = {
        "home_name": "Aston Villa",
        "away_name": "Arsenal",
        "competition": "EPL",
        "season": "2026-2027",
    }
    early = dict(base, date="2026-08-29")
    late = dict(base, date="2026-08-31")
    assert match_identity(early) == match_identity(late)

    save_match(early, _team_frame("Aston Villa", "Arsenal"), None, db_path=db)
    save_match(late, _team_frame("Aston Villa", "Arsenal"), None, db_path=db)
    assert len(list_matches(db_path=db)) == 1


def test_raw_snapshot_round_trips_and_is_replayable(db, tmp_path):
    from match_store import load_snapshot, save_snapshot, stored_snapshots

    payload = {"events": [{"id": 1, "type": "Pass"}], "home": {"name": "Arsenal"}}
    save_match(_info(), _team_frame(), None, url="https://www.whoscored.com/matches/7/live/x", db_path=db)
    path = save_snapshot("ws-7", payload, db_path=db)

    assert path.exists() and path.suffix == ".gz"
    assert load_snapshot("ws-7", db_path=db) == payload
    assert "ws-7" in stored_snapshots(db_path=db)
    # A match that was never snapshotted is absent, not an error.
    assert load_snapshot("ws-does-not-exist", db_path=db) is None


def test_schema_migration_adds_columns_to_an_existing_database(db):
    """CREATE TABLE IF NOT EXISTS leaves an older file on its original columns."""
    import sqlite3

    db.parent.mkdir(parents=True, exist_ok=True)
    legacy = sqlite3.connect(str(db))
    legacy.execute(
        "CREATE TABLE matches (match_id TEXT PRIMARY KEY, played_on TEXT, "
        "competition TEXT, home_team TEXT NOT NULL, away_team TEXT NOT NULL, "
        "home_goals INTEGER, away_goals INTEGER, score TEXT, url TEXT, "
        "stored_at TEXT NOT NULL)"
    )
    legacy.commit()
    legacy.close()

    # Opening through the store must upgrade it rather than fail on insert.
    save_match(_info(), _team_frame(), None, db_path=db)
    assert len(list_matches(db_path=db)) == 1
