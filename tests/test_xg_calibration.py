"""A model may only correct itself when the data it has says it should.

The shipped xG model is a logistic whose coefficients were reasoned from
published research rather than fitted. Fitting it properly takes tens of
thousands of shots — roughly ten goals per coefficient is the usual floor, and
there are fifteen. The archive currently holds 358 non-penalty shots and 41
goals, which produced 37.4 predicted against 41 scored: six tenths of a
standard deviation, and no bias to correct.

So the gate matters more than the fit. These tests hold it shut on data that
cannot support a correction, open it on data that can, and check that the two
parameters it does fit can only stretch the existing curve.
"""

import json

import numpy as np
import pytest

import xg_calibration as xc


def _synthetic(n=4000, bias=1.0, seed=0):
    """Shots whose true probability is `bias` times the model's log odds."""
    rng = np.random.default_rng(seed)
    predicted = rng.beta(1.4, 9.0, size=n)          # a realistic xG spread
    logits = np.array([xc._logit(p) for p in predicted])
    truth = 1.0 / (1.0 + np.exp(-(bias * logits)))
    outcomes = (rng.random(n) < truth).astype(float)
    return predicted, outcomes


def test_a_small_archive_earns_no_correction():
    predicted, outcomes = _synthetic(n=300, bias=1.6)
    assert xc.fit(predicted, outcomes) is None


def test_enough_shots_but_too_few_goals_earns_none():
    """Goals are the scarce quantity, not shots."""
    predicted, outcomes = _synthetic(n=2000, bias=1.0, seed=3)
    outcomes = np.zeros_like(outcomes)
    outcomes[:20] = 1.0
    assert xc.fit(predicted, outcomes) is None


def test_a_well_calibrated_model_is_left_alone():
    """No improvement on held-out folds means no correction."""
    predicted, outcomes = _synthetic(n=6000, bias=1.0, seed=1)
    found = xc.fit(predicted, outcomes)
    assert found is None or abs(found.slope - 1.0) < 0.2


def test_a_biased_model_is_corrected_towards_the_truth():
    predicted, outcomes = _synthetic(n=8000, bias=1.7, seed=2)
    found = xc.fit(predicted, outcomes)
    assert found is not None, "a clear, large bias was not corrected"
    assert found.log_loss_after < found.log_loss_before
    assert found.slope > 1.1, found.slope


def test_a_correction_only_bends_the_curve_it_was_given():
    """Platt scaling is monotone: it cannot reorder two shots."""
    calibration = xc.Calibration(1.4, -0.3, 2000, 200, 0.30, 0.29)
    values = [0.02, 0.05, 0.12, 0.3, 0.6, 0.9]
    corrected = [calibration.apply(v) for v in values]
    assert corrected == sorted(corrected)
    assert all(0.0 < c < 1.0 for c in corrected)


def test_the_identity_when_nothing_was_earned(tmp_path):
    assert xc.load(tmp_path) is None
    for value in (0.01, 0.25, 0.94):
        assert xc.calibrated(value, None) == pytest.approx(value)


def test_a_saved_correction_round_trips(tmp_path):
    original = xc.Calibration(1.23, -0.45, 2100, 260, 0.31, 0.30)
    xc.save(original, tmp_path)
    loaded = xc.load(tmp_path)
    assert loaded is not None
    assert loaded.slope == pytest.approx(original.slope)
    assert loaded.intercept == pytest.approx(original.intercept)
    assert json.loads((tmp_path / xc.CALIBRATION_FILE).read_text())["shots"] == 2100


def test_a_damaged_file_is_ignored_rather_than_raised(tmp_path):
    (tmp_path / xc.CALIBRATION_FILE).write_text("{not json", encoding="utf-8")
    assert xc.load(tmp_path) is None


def test_evaluate_reports_the_distance_in_standard_deviations():
    """The number that decides whether a gap is real or the sample."""
    predicted = np.full(400, 0.10)
    outcomes = np.zeros(400)
    outcomes[:40] = 1.0                      # exactly as predicted
    report = xc.evaluate(predicted, outcomes)
    assert report["goals"] == 40
    assert abs(report["sigma"]) < 0.01
    assert report["ratio"] == pytest.approx(1.0)


def test_the_shipped_model_is_not_being_corrected_today():
    """The archive is too small, so the pipeline must run uncorrected."""
    import football_match_analysis as fa

    fa._XG_CALIBRATION = "unread"
    assert fa._apply_xg_calibration(0.25) == pytest.approx(0.25)


# --------------------------------------------------------------------------
# the model's own shape
# --------------------------------------------------------------------------

def _xg(x, y=50.0, **extra):
    import football_match_analysis as fa

    row = {"x": x, "y": y, "qualifier_names": extra.pop("q", ""), "is_shot": True}
    row.update(extra)
    return fa._opta_like_local_xg_from_row(row)


def test_a_shot_never_gets_better_by_moving_further_away():
    """The foot model's distance² term is positive, so its curve is a parabola.

    It fell to a minimum around twenty-six metres and then climbed: a strike
    from seventy-four metres, inside the taker's own half, was priced at 0.786,
    and seventeen per cent of the shots in the archive sat past that turn.
    """
    values = [_xg(x) for x in range(99, 24, -1)]
    for near, far in zip(values, values[1:]):
        assert far <= near + 1e-9, "xG rises with distance"


def test_the_far_tail_lands_where_published_models_put_it():
    import football_match_analysis as fa

    def at(metres):
        # Walk back along the centre line until the geometry agrees.
        for x in range(99, 0, -1):
            if fa._shot_geometry_features({"x": x, "y": 50})["distance"] >= metres:
                return _xg(x)
        return 0.0

    assert 0.010 <= at(30) <= 0.05, at(30)
    assert 0.003 <= at(40) <= 0.02, at(40)
    assert at(60) <= 0.01, at(60)
    assert at(70) <= 0.005, at(70)


def test_the_near_field_is_untouched_by_the_far_tail_fix():
    """Everything inside the turn is where the model was tuned."""
    assert 0.25 <= _xg(94) <= 0.45
    assert 0.08 <= _xg(88) <= 0.18
    assert 0.03 <= _xg(83) <= 0.09


def test_a_wider_angle_is_worth_less_than_a_central_one():
    straight = _xg(88, 50)
    wide = _xg(88, 85)
    assert wide < straight, (wide, straight)


def test_headers_and_free_kicks_were_already_monotone():
    """Only the foot submodel carried the positive quadratic."""
    header = [_xg(x, is_header=True, q="Head") for x in range(99, 40, -1)]
    assert header == sorted(header, reverse=True)
