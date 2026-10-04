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
