import pytest

from fixtures import (
    known_teams,
    load_fixtures,
    resolve_team,
    team_fixtures,
)


@pytest.fixture(scope="module")
def rows():
    stored = load_fixtures()
    if not stored:
        pytest.skip("no fixture calendar in data/fixtures/")
    return stored


def test_every_fixture_carries_a_usable_whoscored_url(rows):
    """A row without an id is worse than a missing row: it looks like an
    answer and sends the analyser nowhere."""
    for row in rows:
        assert row["whoscored_id"], row
        assert row["url"].startswith("https://www.whoscored.com/matches/")
        assert row["whoscored_id"] in row["url"]


def test_a_team_never_plays_itself(rows):
    for row in rows:
        assert row["home"] != row["away"], row


def test_the_calendar_is_sorted_by_kickoff(rows):
    kickoffs = [row["kickoff_utc"] for row in rows]
    assert kickoffs == sorted(kickoffs)


def test_match_ids_are_unique(rows):
    ids = [row["whoscored_id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_each_league_is_a_complete_double_round_robin(rows):
    """20 teams home and away is 380 fixtures; a short league means the
    calendar was truncated somewhere and lookups will silently miss matches."""
    by_competition: dict[str, list[dict]] = {}
    for row in rows:
        by_competition.setdefault(row["competition"], []).append(row)
    for competition, fixtures in by_competition.items():
        teams = known_teams(fixtures)
        assert len(fixtures) == len(teams) * (len(teams) - 1), competition


def test_exact_name_wins_over_substring(rows):
    """"leeds" is also inside no other slug, but the principle matters where
    one team's name contains another's."""
    assert resolve_team("arsenal", rows) == "arsenal"
    assert resolve_team("Aston Villa", rows) == "aston-villa"


def test_an_ambiguous_name_is_refused_rather_than_guessed(rows):
    with pytest.raises(LookupError, match="matches"):
        resolve_team("real", rows)


def test_an_unknown_name_says_so(rows):
    with pytest.raises(LookupError, match="No team matching"):
        resolve_team("newport pagnell town", rows)


def test_a_team_plays_every_other_team_twice(rows):
    arsenal = team_fixtures("arsenal", rows)
    opponents = [
        row["away"] if row["home"] == "arsenal" else row["home"] for row in arsenal
    ]
    assert len(arsenal) == 38
    assert sorted(set(opponents)) == sorted(set(opponents))
    assert all(opponents.count(name) == 2 for name in set(opponents))


def test_competition_filter_narrows_the_calendar(rows):
    epl = load_fixtures("EPL")
    assert epl
    assert {row["competition"] for row in epl} == {"EPL"}
    assert len(epl) < len(rows)
