"""Post-shot xG from the fitted model: where it applies, and what it must respect."""

import pandas as pd

from football_analysis.metrics.match_metrics import post_shot_xg
from football_analysis.xg import psxg_model as pm

MODEL = pm.load()


def _shot(gy, gz, xg=0.10, kind="SavedShot", **extra):
    return {
        "is_shot": True,
        "shot_whoscored_type": kind,
        "xG": xg,
        "goal_mouth_y": gy,
        "goal_mouth_z": gz,
        "body_part": "RightFoot",
        **extra,
    }


def test_the_shipped_fit_is_readable():
    assert MODEL is not None, "data/models/psxg_model.json is missing or foreign"
    assert set(MODEL["weights"]) == set(pm.TERMS)


def test_a_corner_is_harder_to_save_than_a_shot_at_the_keeper():
    at_keeper = pm.predict(0.10, 0.0, 0.4, False, model=MODEL)
    low_post = pm.predict(0.10, 0.9, 0.05, False, model=MODEL)
    top_corner = pm.predict(0.10, 0.9, 0.9, False, model=MODEL)
    assert at_keeper < low_post < top_corner


def test_a_better_chance_stays_worth_more_in_the_same_place():
    assert pm.predict(0.40, 0.5, 0.5, False, model=MODEL) > pm.predict(
        0.05, 0.5, 0.5, False, model=MODEL
    )


def test_no_fit_means_no_prediction():
    assert pm.predict(0.10, 0.5, 0.5, False, model={}) is None


def test_the_render_prefers_opta_then_the_fit():
    events = pd.DataFrame(
        [
            _shot(50.0, 10.0, xgot_reference=0.42),  # Opta priced it: kept as is
            _shot(54.0, 34.0),  # top corner: the fit
            _shot(50.0, 10.0, kind="MissedShots"),  # off target: nothing
        ]
    )
    values = post_shot_xg(events)
    assert values.iloc[0] == 0.42
    assert 0.10 < values.iloc[1] <= 0.97
    assert values.iloc[2] == 0.0


def test_penalties_and_own_goals_are_left_to_the_heuristic():
    # The map the fit learns from carries neither, so the fit never prices them.
    fitted = post_shot_xg(pd.DataFrame([_shot(52.0, 12.0, xg=0.30)]))
    penalty = post_shot_xg(pd.DataFrame([_shot(52.0, 12.0, xg=0.30, is_penalty=True)]))
    assert fitted.iloc[0] != penalty.iloc[0]
