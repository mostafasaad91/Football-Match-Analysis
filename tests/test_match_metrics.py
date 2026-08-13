import unittest

import pandas as pd

from match_metrics import (
    advanced_metrics_frames,
    blocked_shot_mask,
    build_possessions,
    cross_mask,
    defensive_block_events,
    defensive_blocks_count,
    fouls_committed_count,
    fouls_committed_mask,
    high_regain_events,
    player_sequence_metrics,
    progressive_pass_mask,
    team_advanced_metrics,
    touch_mask,
)


def event(team_id, event_type, minute, second, x, end_x=None, **overrides):
    row = {
        "event_id": f"{minute}-{second}-{team_id}-{event_type}",
        "period_code": "1H",
        "minute": minute,
        "second": second,
        "team_id": team_id,
        "player": f"Player {team_id}",
        "type": event_type,
        "outcome": "Successful",
        "x": x,
        "y": 50.0,
        "end_x": x if end_x is None else end_x,
        "end_y": 50.0,
        "is_pass": event_type == "Pass",
        "is_shot": event_type in {"Goal", "SavedShot", "MissedShots"},
        "is_goal": event_type == "Goal",
        "is_cross": False,
        "is_penalty_shootout": False,
        "qualifier_names": [],
        "xG": 0.0,
        "xT": 0.0,
    }
    row.update(overrides)
    return row


class MatchMetricTests(unittest.TestCase):
    def setUp(self):
        self.info = {"home_id": 1, "away_id": 2}

    def test_provider_recoveries_are_separate_from_possession_regains(self):
        events = pd.DataFrame(
            [
                event(1, "Pass", 0, 1, 20, 30),
                event(2, "BallRecovery", 0, 10, 40),
                event(2, "Pass", 0, 12, 40, 65, xT=0.12),
                event(2, "SavedShot", 0, 18, 78, xG=0.25),
                event(1, "Pass", 0, 30, 25, 40),
            ]
        )
        metrics = team_advanced_metrics(events, self.info)

        self.assertEqual(metrics["home"]["provider_recoveries"], 0)
        self.assertEqual(metrics["away"]["provider_recoveries"], 1)
        self.assertEqual(metrics["home"]["possession_regains"], 1)
        self.assertEqual(metrics["away"]["possession_regains"], 1)
        self.assertEqual(metrics["away"]["transitions"], 1)
        self.assertEqual(metrics["away"]["transition_shots"], 1)

    def test_paired_foul_rows_count_only_the_offending_team(self):
        events = pd.DataFrame(
            [
                event(1, "Foul", 2, 1, 50, outcome="Unsuccessful"),
                event(2, "Foul", 2, 1, 50, outcome="Successful"),
                event(1, "Foul", 4, 1, 55, outcome="Unsuccessful"),
                event(2, "Foul", 4, 1, 55, outcome="Successful"),
                event(2, "Foul", 8, 1, 60, outcome="Unsuccessful"),
                event(1, "Foul", 8, 1, 60, outcome="Successful"),
            ]
        )

        committed = fouls_committed_mask(events)
        self.assertEqual(int(committed.sum()), 3)
        self.assertEqual(fouls_committed_count(events, 1), 2)
        self.assertEqual(fouls_committed_count(events, 2), 1)

    def test_blocks_use_original_shot_classification_and_defending_team(self):
        events = pd.DataFrame(
            [
                event(
                    2,
                    "SavedShot",
                    12,
                    1,
                    88,
                    y=35,
                    shot_category="Blocked",
                    shot_whoscored_type="BlockedShot",
                    qualifier_names=["Blocked", "BlockedX", "BlockedY"],
                ),
                event(
                    2,
                    "SavedShot",
                    20,
                    1,
                    91,
                    y=62,
                    shot_category="On Target",
                    shot_whoscored_type="SavedShot",
                    qualifier_names=["BlockedX", "BlockedY"],
                ),
            ]
        )

        self.assertEqual(int(blocked_shot_mask(events).sum()), 1)
        self.assertEqual(defensive_blocks_count(events, 1, 2), 1)
        self.assertEqual(defensive_blocks_count(events, 2, 1), 0)
        blocks = defensive_block_events(events, 1, 2)
        self.assertEqual(blocks.iloc[0]["team_id"], 1)
        self.assertEqual(blocks.iloc[0]["type"], "BlockedShot")
        self.assertAlmostEqual(float(blocks.iloc[0]["x"]), 12.0)
        self.assertAlmostEqual(float(blocks.iloc[0]["y"]), 65.0)

    def test_restart_is_not_an_attacking_transition(self):
        events = pd.DataFrame(
            [
                event(1, "Pass", 1, 0, 25, 40),
                event(
                    2,
                    "Pass",
                    1,
                    10,
                    20,
                    75,
                    qualifier_names=["ThrowIn", "FastBreak"],
                ),
                event(2, "SavedShot", 1, 15, 85, xG=0.3),
            ]
        )
        _, possessions = build_possessions(events)
        away = possessions[possessions["team_id"] == 2].iloc[0]

        self.assertEqual(away["start_reason"], "restart")
        self.assertFalse(bool(away["is_transition"]))

    def test_high_regain_uses_possession_start_and_same_possession_outcome(self):
        events = pd.DataFrame(
            [
                event(1, "Pass", 2, 0, 35, 45),
                event(2, "Interception", 2, 5, 70),
                event(2, "Pass", 2, 8, 70, 88),
                event(2, "Goal", 2, 12, 90, xG=0.4),
            ]
        )
        high = high_regain_events(events, 2)
        metrics = team_advanced_metrics(events, self.info)["away"]

        self.assertEqual(len(high), 1)
        self.assertEqual(high.iloc[0]["type"], "Interception")
        self.assertEqual(metrics["high_regains"], 1)
        self.assertEqual(metrics["transition_goals"], 1)

    def test_transition_outcomes_are_limited_to_twelve_seconds(self):
        events = pd.DataFrame(
            [
                event(1, "Pass", 2, 30, 30, 45),
                event(2, "BallRecovery", 2, 35, 40),
                event(2, "Pass", 2, 38, 40, 70, xT=0.1),
                event(2, "SavedShot", 2, 55, 85, xG=0.5),
            ]
        )
        metrics = team_advanced_metrics(events, self.info)["away"]

        self.assertEqual(metrics["transitions"], 1)
        self.assertEqual(metrics["transition_shots"], 0)
        self.assertEqual(metrics["transition_xG"], 0.0)

    def test_canonical_action_masks(self):
        events = pd.DataFrame(
            [
                event(1, "Pass", 3, 0, 20, 50),
                event(1, "Pass", 3, 2, 55, 66),
                event(1, "Pass", 3, 4, 70, 90, is_cross=True, end_y=50),
                event(1, "Clearance", 3, 6, 15, 40),
            ]
        )

        self.assertEqual(int(progressive_pass_mask(events).sum()), 3)
        self.assertEqual(int(cross_mask(events).sum()), 1)
        self.assertEqual(int(touch_mask(events).sum()), 3)

    def test_metric_cache_does_not_pollute_dataframe_attrs(self):
        events = pd.DataFrame(
            [
                event(1, "Pass", 4, 0, 20, 35),
                event(2, "Interception", 4, 5, 65),
            ]
        )
        team_advanced_metrics(events, self.info)

        self.assertNotIn("_canonical_possessions", events.attrs)
        self.assertNotIn("_canonical_team_metrics", events.attrs)
        combined = pd.concat([events.iloc[:1], events.iloc[1:]], ignore_index=True)
        self.assertEqual(len(combined), 2)

    def test_progression_efficiency_and_sequence_metrics(self):
        events = pd.DataFrame(
            [
                event(1, "Pass", 5, 0, 20, 70, player="Alice", xT=0.1),
                event(1, "Pass", 5, 4, 70, 85, player="Bob", xT=0.2),
                event(1, "SavedShot", 5, 8, 88, player="Carol", xG=0.3),
                event(2, "Pass", 5, 20, 20, 30, player="Opponent"),
            ]
        )
        metrics = team_advanced_metrics(events, self.info)["home"]

        self.assertEqual(metrics["deep_completions"], 1)
        self.assertEqual(metrics["build_up_success_rate"], 100.0)
        self.assertEqual(metrics["final_third_entry_efficiency"], 100.0)
        self.assertEqual(metrics["box_entry_to_shot_rate"], 100.0)
        self.assertEqual(metrics["sequence_xT"], 0.3)
        self.assertGreater(metrics["directness"], 0)

    def test_xgchain_and_xgbuildup_player_attribution(self):
        events = pd.DataFrame(
            [
                event(1, "Pass", 6, 0, 30, 55, player="Alice"),
                event(
                    1,
                    "Pass",
                    6,
                    4,
                    55,
                    75,
                    player="Bob",
                    is_key_pass=True,
                ),
                event(1, "SavedShot", 6, 8, 82, player="Carol", xG=0.4),
            ]
        )
        sequence = player_sequence_metrics(events)

        self.assertEqual(sequence["Alice"]["xGChain"], 0.4)
        self.assertEqual(sequence["Bob"]["xGChain"], 0.4)
        self.assertEqual(sequence["Carol"]["xGChain"], 0.4)
        self.assertEqual(sequence["Alice"]["xGBuildup"], 0.4)
        self.assertEqual(sequence["Bob"]["xGBuildup"], 0.0)
        self.assertEqual(sequence["Carol"]["xGBuildup"], 0.0)

    def test_counterpress_and_rest_defence_rates(self):
        events = pd.DataFrame(
            [
                event(1, "Pass", 7, 0, 40, 70, player="Home A"),
                event(2, "Interception", 7, 5, 60, player="Away A"),
                event(2, "Pass", 7, 7, 60, 85, player="Away B"),
                event(1, "BallRecovery", 7, 9, 70, player="Home B"),
            ]
        )
        metrics = team_advanced_metrics(events, self.info)["home"]

        self.assertEqual(metrics["counterpress_attempts"], 1)
        self.assertEqual(metrics["counterpress_success_rate"], 100.0)
        self.assertEqual(metrics["rest_defence_exposures"], 1)
        self.assertEqual(metrics["rest_defence_vulnerability"], 100.0)

    def test_progressive_break_into_final_third_is_a_dangerous_counter(self):
        events = pd.DataFrame(
            [
                event(1, "Pass", 7, 0, 40, 75, player="Home A"),
                event(2, "Interception", 7, 5, 15, player="Away A"),
                event(2, "Pass", 7, 7, 15, 70, player="Away B"),
                event(1, "BallRecovery", 7, 10, 68, player="Home B"),
            ]
        )
        metrics = team_advanced_metrics(events, self.info)["home"]

        self.assertEqual(metrics["rest_defence_exposures"], 1)
        self.assertEqual(metrics["rest_defence_dangerous_counters"], 1)
        self.assertEqual(metrics["rest_defence_vulnerability"], 100.0)

    def test_game_state_is_assigned_before_each_possession(self):
        events = pd.DataFrame(
            [
                event(1, "Pass", 8, 0, 30, 60, player="Home A"),
                event(1, "Goal", 8, 5, 85, player="Home B", xG=0.2),
                event(2, "Pass", 8, 15, 30, 50, player="Away A"),
                event(2, "SavedShot", 8, 20, 80, player="Away B", xG=0.3),
            ]
        )
        metrics = team_advanced_metrics(events, self.info)

        self.assertEqual(metrics["home"]["game_state_splits"]["drawing"]["shots"], 1)
        self.assertEqual(metrics["away"]["game_state_splits"]["trailing"]["shots"], 1)

    def test_advanced_export_frames_include_requested_fields(self):
        events = pd.DataFrame(
            [
                event(1, "Pass", 9, 0, 20, 70, player="Alice", xT=0.1),
                event(1, "SavedShot", 9, 5, 85, player="Carol", xG=0.2),
                event(2, "Pass", 9, 15, 30, 45, player="Opponent"),
            ]
        )
        info = {
            **self.info,
            "home_name": "Home",
            "away_name": "Away",
        }
        team_frame, player_frame = advanced_metrics_frames(events, info)

        expected_team_columns = {
            "field_tilt",
            "deep_completions",
            "build_up_success_rate",
            "final_third_entry_efficiency",
            "box_entry_to_shot_rate",
            "sequence_xT",
            "directness",
            "counterpress_success_rate",
            "rest_defence_vulnerability",
            "game_state_drawing_xG",
        }
        self.assertTrue(expected_team_columns.issubset(team_frame.columns))
        self.assertTrue(
            {"xGChain", "xGBuildup", "sequence_xT"}.issubset(player_frame.columns)
        )


if __name__ == "__main__":
    unittest.main()


# ── Advanced metrics added for the extended report ───────────────────────
def _sample_events():
    import pandas as pd
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    events = pd.read_csv(root / "sample_data" / "France_vs_England_4-6" / "events.csv", encoding="utf-8-sig")
    for column in ("minute", "second", "x", "y", "end_x", "end_y", "xG", "xT", "team_id"):
        if column in events.columns:
            events[column] = pd.to_numeric(events[column], errors="coerce")
    return events


def test_post_shot_xg_only_counts_shots_that_reached_the_target():
    from match_metrics import post_shot_xg

    events = _sample_events()
    shots = events[events["is_shot"] == True]  # noqa: E712
    psxg = post_shot_xg(shots)

    off_target = ~shots["shot_whoscored_type"].isin(["Goal", "SavedShot"])
    assert float(psxg[off_target].sum()) == 0.0
    # Placement adds value over the raw chance for shots that were on target,
    # while off-target shots drop out entirely.
    on_target_xg = float(shots.loc[~off_target, "xG"].fillna(0).sum())
    assert on_target_xg < float(psxg.sum()) < float(shots["xG"].fillna(0).sum())


def test_post_shot_xg_does_not_hard_code_goals_to_one():
    """A goal must keep its modelled value, otherwise 'goals prevented' is
    positive for every keeper in every match and measures nothing."""
    from match_metrics import post_shot_xg

    events = _sample_events()
    goals = events[(events["is_shot"] == True) & (events["is_goal"] == True)]  # noqa: E712
    assert not goals.empty
    assert float(post_shot_xg(goals).max()) < 1.0


def test_placement_difficulty_rises_from_the_keeper_to_the_corner():
    from match_metrics import placement_difficulty

    at_keeper = placement_difficulty(0.0, 0.15)
    low_post = placement_difficulty(0.9, 0.15)
    top_corner = placement_difficulty(0.95, 0.9)
    assert at_keeper < low_post < top_corner
    assert 0.0 <= at_keeper and top_corner <= 1.0


def test_set_piece_breakdown_reads_the_source_off_the_delivery():
    """The feed tags the corner/free kick on the delivery, not on the shot, so
    a shot-row-only reading reports every set piece as open play."""
    from match_metrics import set_piece_breakdown

    events = _sample_events()
    england = set_piece_breakdown(events, 345)
    assert england["corner"]["shots"] >= 1
    assert england["penalty"]["shots"] == 1
    total = sum(bucket["shots"] for bucket in england.values())
    assert total == int(((events["is_shot"] == True) & (events["team_id"] == 345)).sum())  # noqa: E712


def test_windowed_metrics_cover_the_match():
    from match_metrics import defensive_line_height, team_compactness, xg_momentum

    events = _sample_events()
    import numpy as np

    momentum = xg_momentum(events, 341, 345)
    assert not momentum.empty
    # The printed differential must equal the two printed components, within
    # the rounding the table itself applies.
    assert np.allclose(
        momentum["differential"],
        momentum["home_xG"] - momentum["away_xG"],
        atol=5e-4,
    )

    height = defensive_line_height(events, 341)
    assert not height.empty
    assert height["height"].between(0, 100).all()

    compact = team_compactness(events, 341)
    assert not compact.empty
    assert (compact["vertical_spread"] >= 0).all()


def test_network_centrality_names_the_connectors():
    from match_metrics import network_centrality

    centrality = network_centrality(_sample_events(), 341)
    assert not centrality.empty
    # Betweenness is normalised, sorted best-first, and rewards position in the
    # network rather than raw pass volume.
    assert centrality["betweenness"].between(0, 1).all()
    assert centrality["betweenness"].is_monotonic_decreasing
    assert centrality.loc[0, "player"] != centrality.sort_values("passes_made").iloc[0]["player"]


def test_turnovers_separate_losses_from_punished_losses():
    from match_metrics import turnover_events

    turnovers = turnover_events(_sample_events(), 341)
    assert not turnovers.empty
    punished = turnovers[turnovers["punished"]]
    assert len(punished) < len(turnovers)
    # Only punished losses carry a conceded xG, and only they record a delay.
    assert (turnovers.loc[~turnovers["punished"], "conceded_xG"] == 0).all()
    assert punished["seconds_to_shot"].notna().all()


def test_duel_map_keeps_location_and_kind():
    from match_metrics import duel_map

    duels = duel_map(_sample_events(), 341)
    assert not duels.empty
    assert set(duels["kind"].unique()) <= {"aerial", "ground"}
    assert duels["x"].between(0, 100).all() and duels["y"].between(0, 100).all()


def test_shot_placement_zones_only_count_on_target_shots():
    from match_metrics import shot_placement_zones

    events = _sample_events()
    zones = shot_placement_zones(events, 345)
    on_target = int(
        (
            (events["is_shot"] == True)  # noqa: E712
            & (events["team_id"] == 345)
            & events["shot_whoscored_type"].isin(["Goal", "SavedShot"])
        ).sum()
    )
    assert 0 < sum(zones.values()) <= on_target
    assert len(zones) == 9


def test_pass_geometry_falls_back_to_coordinates():
    """The feed keeps qualifier names but drops their values, so length and
    angle have to be reconstructed from the coordinates."""
    from match_metrics import pass_geometry

    events = _sample_events()
    passes = events[events["is_pass"] == True].dropna(subset=["x", "y", "end_x", "end_y"])  # noqa: E712
    geometry = pass_geometry(passes)

    assert len(geometry) == len(passes)
    assert (geometry["length_m"] >= 0).all()
    # No pass can be longer than the pitch diagonal.
    assert geometry["length_m"].max() < 130
    assert geometry["direction_deg"].between(-180, 180).all()


def test_pass_length_profile_separates_long_balls():
    from match_metrics import pass_length_profile

    profile = pass_length_profile(_sample_events(), 341)
    assert profile["passes"] > 0
    assert 0 <= profile["long_ball_share"] <= 100
    assert 0 <= profile["completion"] <= 100
    # Long balls are harder than the average pass.
    assert profile["long_ball_completion"] < profile["completion"]


def test_goalkeeper_distribution_reports_launch_behaviour():
    from match_metrics import goalkeeper_distribution

    henderson = goalkeeper_distribution(_sample_events(), 345, "Dean Henderson")
    assert henderson["distributions"] > 0
    assert 0 <= henderson["launch_share"] <= 100
    # Launching is lower percentage than distributing short.
    assert henderson["launch_completion"] <= henderson["completion"]


def test_press_resistance_separates_pressed_from_free_passing():
    from match_metrics import press_resistance

    resistance = press_resistance(_sample_events(), 341)
    assert resistance["passes_under_pressure"] > 0
    assert 0 < resistance["pressed_share"] < 100
    # Passing under pressure is harder than passing unopposed, so the gap is
    # negative; the size of it is the actual finding.
    assert resistance["pressed_completion"] < resistance["free_completion"]
    assert resistance["resistance_gap"] < 0


def test_line_breaking_passes_start_behind_and_end_beyond_the_line():
    from match_metrics import line_breaking_passes

    breaks = line_breaking_passes(_sample_events(), 341, 345)
    assert not breaks.empty
    assert (breaks["x"] < breaks["line_height"]).all()
    assert (breaks["end_x"] > breaks["line_height"]).all()


def test_win_probability_is_a_distribution_that_hardens_over_time():
    from match_metrics import win_probability

    curve = win_probability(_sample_events(), 341, 345)
    assert not curve.empty
    totals = curve["home_win"] + curve["draw"] + curve["away_win"]
    assert ((totals - 1.0).abs() < 0.01).all()
    # Kick-off is symmetric between two teams with no goals yet.
    first = curve.iloc[0]
    assert abs(first["home_win"] - first["away_win"]) < 0.01
    # The same lead is worth more late than early.
    early = curve[(curve["goal_difference"] == -1) & (curve["minute"] <= 45)]
    late = curve[(curve["goal_difference"] == -1) & (curve["minute"] >= 70)]
    if not early.empty and not late.empty:
        assert late["away_win"].max() > early["away_win"].min()


def test_zone_value_rises_toward_goal_and_favours_the_centre():
    from match_metrics import ZONE_VALUE_MAX, zone_value

    assert zone_value(5, 50) < zone_value(50, 50) < zone_value(75, 50) < zone_value(89, 50)
    # Same depth, pinned wide, is worth less than the same depth centrally.
    assert zone_value(89, 95) < zone_value(89, 50)
    assert zone_value(89, 50) <= ZONE_VALUE_MAX


def test_action_values_price_gains_losses_and_defensive_work():
    from match_metrics import action_values

    events = _sample_events()
    values = action_values(events)
    assert len(values) == len(events)
    # A model that only ever adds value is not measuring risk.
    assert float(values.min()) < 0 < float(values.max())


def test_defensive_actions_do_not_bank_the_whole_threat():
    """Crediting a clearance with the full mirrored zone value put centre-backs
    at the top of every ranking, which described the model, not the match."""
    from match_metrics import CLEARANCE_CREDIT, REGAIN_CREDIT

    assert 0 < CLEARANCE_CREDIT < REGAIN_CREDIT < 1


def test_player_action_value_ranks_the_decisive_players_first():
    from match_metrics import player_action_value

    ranked = player_action_value(_sample_events())
    assert not ranked.empty
    assert ranked["total_value"].is_monotonic_decreasing
    # Offensive and defensive contributions add to the total, so a centre-back
    # and a winger can appear in the same ranking.
    combined = ranked["offensive_value"] + ranked["defensive_value"]
    assert ((combined - ranked["total_value"]).abs() < 0.01).all()


def test_sequence_typology_splits_xg_by_how_it_was_built():
    from match_metrics import sequence_typology

    typology = sequence_typology(_sample_events(), 345)
    assert not typology.empty
    assert abs(typology["share_of_xG"].sum() - 100.0) < 0.5
    assert typology["xG"].is_monotonic_decreasing
    assert set(typology["type"]) <= {
        "build_up", "sustained", "direct", "counter", "set_piece", "other"
    }


def test_receptions_between_lines_land_in_the_pocket():
    from match_metrics import receptions_between_lines

    pockets = receptions_between_lines(_sample_events(), 341, 345)
    assert not pockets.empty
    # Every reception sits in the band immediately in front of the line.
    assert ((pockets["x"] <= pockets["line_height"]) & (pockets["x"] >= pockets["line_height"] - 12.0)).all()


def test_switches_cross_the_pitch():
    from match_metrics import SWITCH_WIDTH, switches_of_play

    switches = switches_of_play(_sample_events(), 341)
    assert not switches.empty
    assert (switches["width"] >= SWITCH_WIDTH).all()


def test_goal_origin_chains_describe_every_goal():
    from match_metrics import goal_origin_chains

    events = _sample_events()
    chains = goal_origin_chains(events, 341, 345)
    scored = int(
        ((events["is_goal"] == True) & (events["is_own_goal"] != True)).sum()  # noqa: E712
    )
    assert len(chains) == scored
    assert chains["minute"].is_monotonic_increasing
    assert (chains["players"] >= 1).all()


def test_substitution_impact_pairs_each_arrival_with_its_own_departure():
    """A quadruple change at half time used to report the same player leaving
    four times, because the departure was matched by proximity, not in order."""
    from match_metrics import substitution_impact

    impact = substitution_impact(_sample_events(), 341, 345)
    assert not impact.empty
    named = impact[impact["player_off"] != ""]
    for (_team, _minute), group in named.groupby(["team_id", "minute"]):
        assert group["player_off"].nunique() == len(group)


def test_field_tilt_timeline_shares_sum_to_one_hundred():
    from match_metrics import field_tilt_timeline

    timeline = field_tilt_timeline(_sample_events(), 341, 345)
    assert not timeline.empty
    assert ((timeline["home_tilt"] + timeline["away_tilt"] - 100.0).abs() < 0.2).all()


def test_deep_metrics_all_return_something_on_real_data():
    """Smoke cover for the explanatory batch so a schema change cannot quietly
    empty a page."""
    import match_metrics as m

    events = _sample_events()
    assert not m.pressing_triggers(events, 341).empty
    assert m.rest_defence_structure(events, 341)["losses"] > 0
    assert m.time_to_progress(events, 341)["possessions"] > 0
    assert m.second_ball_recovery(events, 341)["contests"] > 0
    assert not m.third_man_combinations(events, 341).empty
    assert not m.substitution_impact(events, 341, 345).empty
