"""What a bar on a radar is allowed to claim.

Three defects, all of them wrong statements rather than ugly ones.

A rate was ranked without a denominator. Lucas Herrington came on for
thirty-three minutes, played three passes, completed all three, and his 100%
put him in the 95th percentile for passing accuracy — above Bruno Fernandes,
who played ninety-two at 80% and scored 53. The radar said the substitute was
the better passer, from a sample of three.

The subtitle printed a participation status where a reader expects a position.
The goalkeeper read "Player" and Semi Ajayi, who scored, read "sub_out".
players.csv has carried the real position all along.

Every bar was a percentile against all twenty-nine players on the pitch, so a
centre-back was ranked on expected goals against forwards and a forward on
clearances against centre-backs. The median player had fifteen of thirty slices
at zero, and the shape of a radar described the position rather than the
performance.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import player_radar as pr
from conftest import match_dir

MATCH = "Hull_vs_Man_Utd_2-0"


@pytest.fixture(scope="module")
def fixture():
    out = match_dir(MATCH)
    if not (out / "players.csv").exists():
        pytest.skip(f"{MATCH} has not been rendered")
    events = pd.read_csv(out / "events.csv")
    squad = pd.read_csv(out / "players.csv")
    allm, elig = pr.compute_metrics_pool(events)
    return events, squad, allm, elig


# --------------------------------------------------------------------------
# a rate needs a denominator
# --------------------------------------------------------------------------

def test_a_rate_on_too_few_attempts_is_not_ranked():
    """The case this exists for, stated as the numbers that produced it."""
    three_passes = {"Passes": 3, "Pass %": 100}
    ninety_two = {"Passes": 92, "Pass %": 80}
    assert not pr.rate_is_measured(three_passes, "Pass %")
    assert pr.rate_is_measured(ninety_two, "Pass %")


def test_a_metric_that_is_not_a_rate_is_always_ranked():
    """Counts carry their own denominator: five tackles is five tackles."""
    for metric in ("Goals", "Clearances", "xG", "Recov\neries"):
        assert pr.rate_is_measured({}, metric), metric


@pytest.mark.parametrize("metric,key,floor", list(
    (m, k, f) for m, (k, f) in pr.RATE_FLOORS.items()))
def test_every_floor_sits_just_above_its_own_boundary(metric, key, floor):
    assert not pr.rate_is_measured({key: floor - 1}, metric)
    assert pr.rate_is_measured({key: floor}, metric)


def test_a_missing_denominator_is_not_treated_as_enough(fixture):
    """An export without the attempt count must not rank the rate anyway."""
    assert not pr.rate_is_measured({}, "Pass %")
    assert not pr.rate_is_measured({"Passes": None}, "Pass %")
    assert not pr.rate_is_measured({"Passes": "n/a"}, "Pass %")


def test_the_substitute_no_longer_outranks_the_playmaker(fixture):
    """End to end, on the fixture that produced the complaint."""
    events, squad, allm, elig = fixture
    sub = pr.player_metrics(events, "Lucas Herrington")
    playmaker = pr.player_metrics(events, "Bruno Fernandes")
    if not sub or not playmaker:
        pytest.skip("this fixture no longer carries those players")
    assert not pr.rate_is_measured(sub, "Pass %"), sub.get("Passes")
    assert pr.rate_is_measured(playmaker, "Pass %"), playmaker.get("Passes")


# --------------------------------------------------------------------------
# the subtitle says a position
# --------------------------------------------------------------------------

def test_the_position_comes_from_the_squad_not_from_participation(fixture):
    _events, squad, _allm, _elig = fixture
    assert pr.player_position(squad, "Konstantinos Tzolakis") == "GK"
    assert pr.describe_position("GK") == "Goalkeeper"


def test_every_position_code_in_the_export_has_a_name(fixture):
    """A code with no name would print the code, which is not English."""
    _events, squad, _allm, _elig = fixture
    for code in sorted(set(squad["position"].dropna().astype(str))):
        assert code in pr.POSITION_NAMES, code


def test_an_unknown_position_falls_back_rather_than_printing_nothing():
    assert pr.describe_position("", fallback="Player") == "Player"
    assert pr.describe_position("XYZ") == "XYZ"


def test_a_missing_squad_export_does_not_break_the_lookup():
    assert pr.player_position(None, "anyone") == ""
    assert pr.player_position(pd.DataFrame(), "anyone") == ""
    assert pr.player_position(pd.DataFrame({"name": ["a"]}), "a") == ""


# --------------------------------------------------------------------------
# who a bar measures against
# --------------------------------------------------------------------------

def test_a_defender_is_ranked_among_defenders(fixture):
    _events, squad, _allm, elig = fixture
    pool = pr.line_pool(squad, elig, pr.position_line("DC"))
    assert 0 < len(pool) < len(elig), (len(pool), len(elig))
    for player in pool:
        assert pr.position_line(pr.player_position(squad, player)) == "defence"


def test_a_line_too_thin_to_rank_falls_back_to_the_whole_pitch(fixture):
    """A pool of two says more about who else was picked than about the player."""
    _events, squad, _allm, elig = fixture
    assert pr.line_pool(squad, elig, "nonsense") == list(elig)
    assert pr.line_pool(None, elig, "defence") == list(elig)


def test_the_keeper_is_never_pooled_with_outfielders(fixture):
    """There is one a side, and ranking him against ten who are not keepers is
    what produced a radar of thirty empty slices."""
    _events, squad, _allm, elig = fixture
    assert pr.line_pool(squad, elig, "keeper") == list(elig)


def test_every_outfield_position_belongs_to_a_line():
    outfield = [c for c in pr.POSITION_NAMES if c not in ("Sub", "GK")]
    for code in outfield:
        assert pr.position_line(code) in {"defence", "midfield", "attack"}, code


def test_the_three_lines_are_all_used():
    lines = {pr.position_line(c) for c in pr.POSITION_NAMES}
    assert {"defence", "midfield", "attack"} <= lines


@pytest.mark.parametrize("code", ["DC", "MC", "FW"])
def test_ranking_within_a_line_changes_what_a_bar_says(fixture, code):
    """The point of the change, measured rather than asserted.

    A centre-back's expected goals against forwards is near the floor whatever
    he did. Against the other defenders it is a comparison he can win.
    """
    events, squad, allm, elig = fixture
    line = pr.position_line(code)
    pool = pr.line_pool(squad, elig, line)
    if len(pool) >= len(elig):
        pytest.skip(f"{line} is not its own pool in this fixture")

    someone = next(p for p in pool)
    metrics = pr.player_metrics(events, someone)
    moved = False
    for metric in ("xG", "Clearances", "Passes", "Recov\neries"):
        value = metrics.get(metric, 0)
        if not value:
            continue
        whole = pr._percentile(allm, elig, metric, value)
        within = pr._percentile(allm, pool, metric, value)
        if abs(whole - within) > 1e-9:
            moved = True
    assert moved, f"{someone}: ranking within {line} changed nothing"


# --------------------------------------------------------------------------
# the goalkeeper's own radar
# --------------------------------------------------------------------------

def test_the_keeper_gets_goalkeeping_metrics(fixture):
    """None of this existed. He was drawn on the outfield layout — goals,
    dribbles, expected goals, aerial duels — and twenty-two of his thirty
    slices were structurally zero."""
    events, _squad, _allm, _elig = fixture
    keeper = pr.goalkeeper_metrics(events, "Konstantinos Tzolakis")
    assert keeper, "no goalkeeping metrics were produced"
    for metric in ("Saves", "Save %", "Shots\nfaced", "Claims", "Sweeps",
                   "Pickups", "Passes", "Pass %"):
        assert metric in keeper, metric


def test_the_keeper_metrics_agree_with_each_other(fixture):
    events, _squad, _allm, _elig = fixture
    keeper = pr.goalkeeper_metrics(events, "Konstantinos Tzolakis")
    saves, conceded = keeper["Saves"], keeper["Goals\nconceded"]
    assert keeper["Shots\nfaced"] == saves + conceded
    if keeper["Shots\nfaced"]:
        assert keeper["Save %"] == round(100 * saves / (saves + conceded))


def test_a_percentage_is_printed_without_a_decimal_it_does_not_have(fixture):
    """"47.0" is not a percentage, it is a percentage with a stray digit."""
    events, _squad, _allm, _elig = fixture
    keeper = pr.goalkeeper_metrics(events, "Konstantinos Tzolakis")
    for metric in ("Save %", "Pass %", "Long ball %"):
        assert isinstance(keeper[metric], int), (metric, keeper[metric])


def test_the_long_ball_slice_holds_attempts_not_completions(fixture):
    """_RATIO_DISPLAY reads the metric itself as the denominator, so holding
    the completed count in both places printed "9 / 9" beside 22% accuracy."""
    events, _squad, _allm, _elig = fixture
    keeper = pr.goalkeeper_metrics(events, "Konstantinos Tzolakis")
    assert keeper["Long\nballs"] >= keeper["Longballs_comp"]
    if keeper["Long\nballs"]:
        assert keeper["Long ball %"] == round(
            100 * keeper["Longballs_comp"] / keeper["Long\nballs"])


def test_post_shot_expected_goals_is_not_on_the_keeper_radar():
    """The model totals 30.0 against 60 goals actually scored — a ratio of
    0.50 — so "goals prevented" off that baseline would put every keeper in
    every report eight tenths of a goal below expectation."""
    slices = [m for _g, _c, ms in pr.GK_GROUPS for m in ms]
    assert "PSxG\nfaced" not in slices
    assert "Goals\nprevented" not in slices


def test_more_is_worse_draws_no_bar():
    """A long wedge beside GOALS CONCEDED reads as an achievement."""
    for metric in pr.GK_NO_BAR:
        assert pr.gk_bar(metric, 5) == 0.0, metric


def test_every_keeper_slice_that_draws_a_bar_has_a_scale():
    for _group, _colour, metrics in pr.GK_GROUPS:
        for metric in metrics:
            assert metric in pr.GK_FULL_BAR or metric in pr.GK_NO_BAR, metric


def test_the_keeper_bar_is_a_share_and_stays_inside_it():
    assert pr.gk_bar("Saves", 0) == 0
    assert pr.gk_bar("Saves", pr.GK_FULL_BAR["Saves"]) == 100
    assert pr.gk_bar("Saves", 999) == 100
    assert 0 < pr.gk_bar("Saves", 2) < 100


# --------------------------------------------------------------------------
# no slice says the same thing twice
# --------------------------------------------------------------------------

def test_the_threat_group_no_longer_asks_one_shot_sample_six_ways():
    threat = next(ms for name, _c, ms in pr.GROUPS if name == "THREAT")
    for dropped in ("npxG", "xGOT", "xG/\nShot", "xG\nBuildup"):
        assert dropped not in threat, dropped
    assert "xG" in threat and "xA" in threat


def test_no_metric_appears_in_two_groups():
    seen = [m for _g, _c, ms in pr.GROUPS for m in ms]
    assert len(seen) == len(set(seen)), sorted(
        m for m in seen if seen.count(m) > 1)
    keeper = [m for _g, _c, ms in pr.GK_GROUPS for m in ms]
    assert len(keeper) == len(set(keeper))


def test_the_outfield_radar_lost_the_duplicates_and_kept_the_rest():
    slices = [m for _g, _c, ms in pr.GROUPS for m in ms]
    assert len(slices) == 26, len(slices)
