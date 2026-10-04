"""Adding matches up without summing what cannot be summed."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_analysis.metrics.aggregate import (
    aggregate_players,
    aggregate_teams,
    is_rate,
    partial_columns,
    per90,
)

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "aggregate_rounds.py"


def _script():
    spec = importlib.util.spec_from_file_location("aggregate_rounds_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _player(match, name, minutes, **values):
    row = {
        "match_id": match,
        "player": name,
        "team": "Arsenal",
        "role_group": "Midfield",
        "minutes": minutes,
        "passes": 0,
        "completed_passes": 0,
        "shots": 0,
        "xG": 0.0,
        "xA": 0.0,
        "touches": 0,
        "defensive_actions": 0,
        "defensive_height": np.nan,
        "padj_defensive_actions": np.nan,
    }
    row.update(values)
    return row


def test_player_ratios_are_recomputed_from_the_sums_not_averaged():
    rows = pd.DataFrame(
        [
            _player("m1", "Rice", 90, passes=100, completed_passes=90),
            _player("m2", "Rice", 90, passes=10, completed_passes=5),
        ]
    )
    out = aggregate_players(rows).iloc[0]
    # 95 of 110, not the mean of 90% and 50%
    assert out["pass_pct"] == pytest.approx(100 * 95 / 110)
    assert out["passes"] == 110 and out["matches"] == 2 and out["minutes"] == 180


def test_defensive_height_is_weighted_by_the_actions_behind_it():
    rows = pd.DataFrame(
        [
            _player("m1", "Rice", 90, defensive_actions=9, defensive_height=30.0),
            _player("m2", "Rice", 90, defensive_actions=1, defensive_height=80.0),
        ]
    )
    assert aggregate_players(rows).iloc[0]["defensive_height"] == pytest.approx(35.0)


def test_players_with_the_same_name_in_two_teams_stay_apart():
    rows = pd.DataFrame(
        [
            _player("m1", "Thomas", 90, shots=1),
            {**_player("m1", "Thomas", 90, shots=4), "team": "Coventry"},
        ]
    )
    assert len(aggregate_players(rows)) == 2


def test_per90_uses_total_minutes_and_drops_the_players_below_the_floor():
    rows = pd.DataFrame(
        [
            _player("m1", "Rice", 90, shots=2, xG=0.4),
            _player("m2", "Rice", 90, shots=4, xG=0.8),
            _player("m1", "Saka", 20, shots=3, xG=0.9),
        ]
    )
    rates = per90(aggregate_players(rows), minimum_minutes=90)
    assert list(rates["player"]) == ["Rice"]
    assert rates.iloc[0]["shots_p90"] == pytest.approx(3.0)
    assert rates.iloc[0]["xG_p90"] == pytest.approx(0.6)


def _team(match, name, goals_for, goals_against, **values):
    return {
        "match_id": match,
        "team": name,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "xG": 1.0,
        "shots": 10,
        "field_tilt": 60.0,
        "team_id": 1,
        **values,
    }


def test_team_results_points_and_totals():
    rows = pd.DataFrame(
        [
            _team("m1", "Arsenal", 3, 0),
            _team("m2", "Arsenal", 1, 1),
            _team("m3", "Arsenal", 0, 2),
        ]
    )
    totals, per_match = aggregate_teams(rows)
    row = totals.iloc[0]
    assert (row["won"], row["drawn"], row["lost"], row["points"]) == (1, 1, 1, 4)
    assert row["goal_difference"] == 1 and row["xG"] == pytest.approx(3.0)
    assert per_match.iloc[0]["goals_for"] == pytest.approx(4 / 3)


def test_rates_are_averaged_and_left_out_of_the_totals():
    rows = pd.DataFrame(
        [
            _team("m1", "Arsenal", 1, 0, field_tilt=70.0),
            _team("m2", "Arsenal", 1, 0, field_tilt=50.0),
        ]
    )
    totals, per_match = aggregate_teams(rows)
    assert np.isnan(totals.iloc[0]["field_tilt"])
    assert per_match.iloc[0]["field_tilt"] == pytest.approx(60.0)


def test_xg_per_shot_is_recomputed_from_the_totals():
    rows = pd.DataFrame(
        [
            _team("m1", "Arsenal", 1, 0, xG=2.0, shots=10),
            _team("m2", "Arsenal", 1, 0, xG=1.0, shots=30),
        ]
    )
    totals, _ = aggregate_teams(rows)
    assert totals.iloc[0]["xG_per_shot"] == pytest.approx(3.0 / 40)


def test_the_table_is_ordered_like_a_league_table():
    rows = pd.DataFrame(
        [
            _team("m1", "Arsenal", 0, 1),
            _team("m1", "Coventry", 1, 0),
        ]
    )
    totals, per_match = aggregate_teams(rows)
    assert list(totals["team"]) == ["Coventry", "Arsenal"]
    assert list(per_match["team"]) == ["Coventry", "Arsenal"]


@pytest.mark.parametrize("name", ["field_tilt", "ppda", "transition_shot_rate", "touch_att_pct"])
def test_rate_like_names_are_recognised(name):
    assert is_rate(name)


@pytest.mark.parametrize("name", ["shots", "box_entries", "high_regains", "xG"])
def test_counts_are_not_mistaken_for_rates(name):
    assert not is_rate(name)


def test_rounds_parse_as_a_range_or_a_single_round():
    module = _script()
    assert list(module.parse_rounds("1-5")) == [1, 2, 3, 4, 5]
    assert list(module.parse_rounds("3")) == [3]
    with pytest.raises(Exception):
        module.parse_rounds("5-1")


def test_the_round_number_is_not_added_up_as_if_it_were_a_statistic():
    rows = pd.DataFrame(
        [_team("m1", "Arsenal", 1, 0, round=1), _team("m2", "Arsenal", 1, 0, round=2)]
    )
    totals, per_match = aggregate_teams(rows)
    assert "round" not in totals.columns and "round" not in per_match.columns


# ── the added statistics ─────────────────────────────────────────────────────
def test_ppda_is_the_summed_passes_over_the_summed_actions():
    rows = pd.DataFrame(
        [
            _team("m1", "Arsenal", 1, 0, ppda_passes_allowed=100, ppda_defensive_actions=10),
            _team("m2", "Arsenal", 1, 0, ppda_passes_allowed=100, ppda_defensive_actions=40),
        ]
    )
    totals, per_match = aggregate_teams(rows)
    # 200 / 50 = 4.0, not the mean of 10.0 and 2.5
    assert totals.iloc[0]["ppda"] == pytest.approx(4.0)
    assert per_match.iloc[0]["ppda"] == pytest.approx(4.0)
    assert totals.iloc[0]["ppda_passes_allowed"] == 200, (
        "the two halves are counts, so they are summed"
    )


def test_progressive_passes_and_build_up_successes_are_counts_not_rates():
    assert not is_rate("progressive_passes")
    assert not is_rate("build_up_successes")
    assert is_rate("directness") and is_rate("rest_defence_vulnerability")


def test_for_and_against_figures_become_differences():
    rows = pd.DataFrame(
        [
            _team("m1", "Arsenal", 3, 1, xG=2.0, xG_against=0.5, xGoT_against=1.2, shots_against=5),
            _team("m2", "Arsenal", 0, 0, xG=1.0, xG_against=1.5, xGoT_against=0.3, shots_against=9),
        ]
    )
    totals, _ = aggregate_teams(rows)
    row = totals.iloc[0]
    assert row["xG_difference"] == pytest.approx(1.0)
    assert row["goals_minus_xG"] == pytest.approx(0.0)
    assert row["goals_prevented"] == pytest.approx(1.5 - 1.0)


def test_pass_completion_is_recomputed_from_the_counts():
    rows = pd.DataFrame(
        [
            _team("m1", "Arsenal", 1, 0, passes=100, passes_completed=90),
            _team("m2", "Arsenal", 1, 0, passes=300, passes_completed=240),
        ]
    )
    totals, _ = aggregate_teams(rows)
    assert totals.iloc[0]["pass_pct"] == pytest.approx(100 * 330 / 400)


def test_players_assists_and_key_passes_are_summed_and_given_per_90():
    rows = pd.DataFrame(
        [
            _player("m1", "Saka", 90, assists=1, key_passes=3, shots=2, shots_on_target=1, goals=1),
            _player("m2", "Saka", 90, assists=2, key_passes=5, shots=2, shots_on_target=2, goals=0),
        ]
    )
    totals = aggregate_players(rows).iloc[0]
    assert (totals["assists"], totals["key_passes"], totals["goal_contributions"]) == (3, 8, 4)
    assert totals["shot_accuracy"] == pytest.approx(75.0)
    rates = per90(aggregate_players(rows)).iloc[0]
    assert rates["assists_p90"] == pytest.approx(1.5)
    assert rates["key_passes_p90"] == pytest.approx(4.0)


def test_the_plain_counts_agree_with_the_statistics_page():
    from conftest import match_dir
    import json

    package = match_dir("Arsenal_vs_Coventry_3-0")
    if not (package / "match_info.json").exists():
        pytest.skip("Arsenal_vs_Coventry_3-0 has not been rendered")
    from football_analysis.metrics.extra_stats import team_extra

    info = json.loads((package / "match_info.json").read_text(encoding="utf-8-sig"))
    events = pd.read_csv(package / "events.csv", low_memory=False)
    side = team_extra(events, info).set_index("team")
    arsenal, coventry = side.loc["Arsenal"], side.loc["Coventry"]
    # the figures the match statistics page prints for this fixture
    assert (arsenal["passes"], arsenal["key_passes"], coventry["key_passes"]) == (653, 17, 3)
    assert (arsenal["tackles"], arsenal["interceptions"], arsenal["blocks"]) == (9, 5, 1)
    assert (arsenal["aerials_won"], arsenal["aerials"], coventry["aerials_won"]) == (11, 28, 17)
    assert (arsenal["ground_duels_won"], arsenal["ground_duels"], coventry["ground_duels_won"]) == (
        11,
        19,
        8,
    )
    assert arsenal["ppda_passes_allowed"] / arsenal["ppda_defensive_actions"] == pytest.approx(
        5.55, abs=0.01
    )


def test_a_column_present_for_only_some_matches_is_reported():
    rows = pd.DataFrame(
        [
            _team("m1", "Arsenal", 1, 0, line_breaking_completed=20.0),
            _team("m2", "Arsenal", 1, 0, line_breaking_completed=np.nan),
            _team("m3", "Arsenal", 1, 0, line_breaking_completed=24.0),
        ]
    )
    assert partial_columns(rows) == ["line_breaking_completed"]
    assert partial_columns(rows.drop(columns="line_breaking_completed")) == []


def test_there_is_no_rating_column_in_the_player_totals():
    rows = pd.DataFrame([_player("m1", "Rice", 90, rating=8.0)])
    assert "avg_rating" not in aggregate_players(rows).columns


def test_the_player_frame_cache_does_not_hand_one_match_to_another():
    """Two matches of the same length with no id or url used to share a cache entry."""
    import json
    from conftest import match_dir

    from football_analysis.render.render_snapshot import load_snapshot
    from football_analysis.visuals.advanced_profiles import _people_for

    pairs = []
    for name in ("Arsenal_vs_Coventry_3-0",):
        package = match_dir(name)
        if (package / "match_info.json").exists():
            pairs.append(package)
    if not pairs:
        pytest.skip("no rendered package to test with")
    events, players, _xg, _teams, _pm, info = load_snapshot(pairs[0])
    first = _people_for(events, players, info)
    other_info = {**info, "home_id": 99991, "away_id": 99992}
    second = _people_for(events, players, other_info)
    assert second is not first, "a different pair of sides must not reuse the cached frame"
