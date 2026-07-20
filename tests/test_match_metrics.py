import unittest

import pandas as pd

from match_metrics import (
    advanced_metrics_frames,
    build_possessions,
    cross_mask,
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
