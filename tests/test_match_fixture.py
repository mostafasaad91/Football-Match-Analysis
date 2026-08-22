"""A match belongs on a shelf, not in a heap.

Every fixture landed directly in ``output/``, so six competitions and two
seasons sat in one flat directory sorted by nothing. The competition and the
season are both in the URL WhoScored serves; the matchweek is in neither the
URL nor the match feed, so it is grouped by the week the match was played in
and named for what that is.
"""

import pytest

from match_fixture import (
    Fixture,
    describe,
    from_url,
    round_from_date,
    shelf,
)

PREMIER = ("https://www.whoscored.com/matches/1/live/"
           "england-premier-league-2026-2027-arsenal-coventry")


def test_the_competition_and_season_come_out_of_the_url():
    fixture = from_url(PREMIER)
    assert fixture.region == "England"
    assert fixture.competition == "Premier League"
    assert fixture.season == "2026-2027"


@pytest.mark.parametrize("url, folder", [
    (PREMIER, "England_Premier_League"),
    ("https://www.whoscored.com/matches/2/live/spain-laliga-2026-2027-sevilla-rayo",
     "Spain_LaLiga"),
    ("https://www.whoscored.com/matches/3/live/"
     "europe-uefa-super-cup-2025-2026-paris-saint-germain-aston-villa",
     "Europe_UEFA_Super_Cup"),
    ("https://www.whoscored.com/matches/4/live/"
     "portugal-liga-portugal-2026-2027-casa-pia-ac-benfica",
     "Portugal_Liga_Portugal"),
    ("https://www.whoscored.com/matches/5/live/"
     "england-league-one-2026-2027-notts-co-leicester",
     "England_League_One"),
])
def test_real_urls_land_in_the_folder_they_should(url, folder):
    assert from_url(url).competition_folder == folder


def test_a_two_word_region_is_not_split_into_the_competition():
    """"united-states-mls" is not a region called United."""
    fixture = from_url("https://www.whoscored.com/matches/6/live/"
                       "united-states-mls-2026-cincinnati-miami")
    assert fixture.region == "United States"
    assert fixture.competition == "MLS"


def test_a_url_it_cannot_read_is_filed_as_unsorted_rather_than_guessed():
    for bad in (None, "", "https://example.com/nothing/useful",
                "https://www.whoscored.com/matches/7/live/no-season-here"):
        assert from_url(bad).competition_folder == "Unsorted"


def test_an_explicit_round_wins():
    assert shelf(PREMIER, "Matchweek 03") == (
        "England_Premier_League", "2026-2027", "Matchweek_03")


def test_the_date_groups_the_round_when_none_is_given():
    """Nothing in the data carries a matchweek, so the week is the round."""
    assert shelf(PREMIER, "", "2026-08-17T20:15:00") == (
        "England_Premier_League", "2026-2027", "Week_of_2026-08-17")


def test_every_day_of_one_week_groups_together():
    """A round is a weekend, and a weekend spills either side of it."""
    week = {round_from_date(f"2026-08-{day:02d}") for day in range(17, 24)}
    assert week == {"Week_of_2026-08-17"}
    assert round_from_date("2026-08-24") == "Week_of_2026-08-24"


def test_a_date_it_cannot_read_produces_no_round_rather_than_a_wrong_one():
    for bad in (None, "", "not a date", "17 August"):
        assert round_from_date(bad) == ""


def test_without_a_round_the_season_is_the_last_shelf():
    assert shelf(PREMIER) == ("England_Premier_League", "2026-2027")


def test_a_folder_name_survives_a_filesystem():
    """Slashes, colons and accents all reach a directory name."""
    fixture = Fixture("Côte d'Ivoire", "Ligue 1 / Pro", "2026", "Round 1:2")
    for part in fixture.parts:
        assert not set(part) & set('\\/:*?"<>|')
        assert part == part.strip("._")


def test_the_description_reads_as_a_line_not_a_path():
    fixture = Fixture("England", "Premier League", "2026-2027", "Matchweek 03")
    assert describe(fixture) == "England Premier League 2026-2027 · Matchweek 03"


def test_the_pipeline_builds_the_shelved_path():
    import football_match_analysis as fa

    info = {
        "home_name": "Arsenal", "away_name": "Coventry", "score": "3 : 0",
        "date": "2026-08-17T20:15:00", "url": PREMIER,
    }
    built = fa._match_output_folder(info, "output").replace("\\", "/")
    assert built.endswith(
        "output/England_Premier_League/2026-2027/Week_of_2026-08-17/"
        "Arsenal_vs_Coventry_3-0")
