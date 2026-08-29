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
    """Passing None means "do not correct".

    It used to mean "go and find one", which is the same thing a caller writes
    when it wants the identity — so there was no way to ask for an uncorrected
    value at all, and this test passed only while no correction existed.
    """
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


def test_the_shipped_model_is_not_being_corrected():
    """No correction has been earned, and the last one was an artifact.

    It was fitted against scripts/xg_report, which rebuilt each shot from the
    stored snapshot and joined its qualifier names with a comma while the model
    splits that field on a pipe. BigChance, Cross and Head never reached it, so
    every shot was priced as an ordinary foot shot from open play — 22% under
    what the pipeline gives the same shots — and the 16% shortfall the
    correction was written to close belonged to a model the package does not
    ship. Measured properly: 109.6 xG against 108 goals.
    """
    import football_match_analysis as fa
    import xg_calibration as xc

    fa._XG_CALIBRATION = "unread"
    assert xc.load() is None, "a correction was written; check it was earned"
    for value in (0.03, 0.20, 0.60):
        assert fa._apply_xg_calibration(value) == pytest.approx(value)


def test_the_correction_the_archive_would_fit_today_is_refused():
    """Slope 0.752 is the right shape and still not good enough.

    The model is over-confident at the top, so a slope below one is what a fit
    reaches for — and pulling the top down drags the four hundred shots below
    0.05 up with it, from a mean of 0.035 against an observed 0.035 to 0.049.
    Log-loss barely notices, because those shots are already cheap. The banded
    gate does.
    """
    rng = np.random.default_rng(11)
    # A model whose error runs the way this one's does: too high where it is
    # confident, too low in the middle, correct at the bottom.
    predicted = rng.beta(1.4, 9.0, size=6000)
    truth = np.where(predicted > 0.40, predicted * 0.61,
                     np.where(predicted > 0.10, predicted * 1.35, predicted))
    outcomes = (rng.random(6000) < truth).astype(float)

    found = xc.fit(predicted, outcomes)
    if found is not None:                       # only if it left the bands better
        corrected = np.array([found.apply(v) for v in predicted])
        assert (xc.banded_error(corrected, outcomes)
                < xc.banded_error(predicted, outcomes))


def test_the_banded_gate_sees_what_log_loss_cannot():
    """Most of the shots are cheap, so they decide the mean log-loss.

    A correction that halves the error on nine hundred shots priced near zero
    and doubles it on the fifty that carry a shot map still improves log-loss.
    banded_error counts each band on its own, so it does not.
    """
    predicted = np.concatenate([np.full(900, 0.04), np.full(50, 0.60)])
    outcomes = np.concatenate([np.zeros(900), np.zeros(50)])
    outcomes[:36] = 1.0                          # 0.04 is exactly right
    outcomes[900:920] = 1.0                      # 0.60 should have been 0.40

    # Thirty predicted against twenty scored in the top band, nothing in the
    # bottom one — which is what a shot map shows and log-loss shrugs at.
    assert xc.banded_error(predicted, outcomes) == pytest.approx(10.0, abs=0.5)
    # Move the top band onto the truth and the error is gone, though the shots
    # that decide log-loss never moved at all.
    fixed = np.concatenate([np.full(900, 0.04), np.full(50, 0.40)])
    assert xc.banded_error(fixed, outcomes) < 1.0


# --------------------------------------------------------------------------
# the measurement has to price shots the way the package does
# --------------------------------------------------------------------------

def test_the_report_prices_a_shot_the_way_the_pipeline_does():
    """A calibration fitted on one scoring path and applied to another is worse
    than no calibration at all, and this is how that happened.

    xg_report rebuilds each shot from the stored snapshot. It joined the
    qualifier names with a comma; _qnames splits that field on a pipe. So the
    whole list came back as one nonsense qualifier, BigChance and Cross and Head
    went missing, and every shot was priced as an ordinary foot shot from open
    play — 22% below what the pipeline gives the same thirty shots. Nothing in
    the report looked wrong: it was internally consistent, and consistently
    measuring a model that is not shipped.
    """
    import football_match_analysis as fa
    from scripts.xg_report import shots_from_snapshots

    event = {
        "isShot": True, "isGoal": False, "x": 91.0, "y": 47.0,
        "qualifiers": [{"type": {"displayName": "BigChance"}},
                       {"type": {"displayName": "RegularPlay"}}],
    }
    names = [q["type"]["displayName"] for q in event["qualifiers"]]
    # However the report chooses to carry them, the model has to see them.
    rebuilt = {"x": event["x"], "y": event["y"], "is_shot": True,
               "qualifier_names": "|".join(names),
               "big_chance": "BigChance" in names}
    assert "BigChance" in fa._qnames(rebuilt)
    assert fa._opta_like_local_xg_from_row(rebuilt) > fa._opta_like_local_xg_from_row(
        {"x": event["x"], "y": event["y"], "is_shot": True, "qualifier_names": ""})

    # And the real path, if there are snapshots to read.
    shots = shots_from_snapshots()
    if shots.empty:
        pytest.skip("no snapshots stored")
    assert shots["big_chance"].any(), "no shot reached the model as a big chance"
    # Exactly, not as a substring: BigChanceCreated rides the same event and
    # belongs to whoever made the pass.
    named = shots["qualifier_names"].map(lambda v: "BigChance" in str(v).split("|"))
    assert int(named.sum()) == int(shots["big_chance"].sum())
    for _, row in shots[named].head(20).iterrows():
        assert "BigChance" in fa._qnames(row)


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


def test_the_error_that_is_left_runs_with_distance_not_with_probability():
    """What is actually wrong with this model, recorded so it is not mistaken
    for the thing a calibration can fix.

    Across the archive it is 2.5 standard deviations too high inside eleven
    metres and 2.0 too low outside, and the two cancel in a total that reads
    109.6 against 108 goals. A slope and an intercept on the model's own answer
    move both ends together, so no correction shaped like Platt can reach it.
    The missing information is what the shot faced: Opta's big-chance flag
    converts near 36% wherever it is taken, while a seven-metre shot Opta
    declined to flag converts at 3% and this model, which sees geometry and not
    goalkeepers, prices it at 10%.
    """
    close_unflagged = _xg(93)                 # about seven metres, no flag
    far_flagged = _xg(80, q="BigChance", big_chance=True)   # about eighteen
    assert close_unflagged > 0.09, close_unflagged
    assert far_flagged < close_unflagged * 3.0
    # The archive says the second of these converts more often than the first.
    # Until the model can see why, it says the opposite, and that is the gap.


def test_a_wider_angle_is_worth_less_than_a_central_one():
    straight = _xg(88, 50)
    wide = _xg(88, 85)
    assert wide < straight, (wide, straight)


def test_headers_and_free_kicks_were_already_monotone():
    """Only the foot submodel carried the positive quadratic."""
    header = [_xg(x, is_header=True, q="Head") for x in range(99, 40, -1)]
    assert header == sorted(header, reverse=True)


# --------------------------------------------------------------------------
# the floor the correction had to clear
# --------------------------------------------------------------------------

def test_the_size_floor_was_not_lowered_to_admit_a_fit():
    """It was, once, and the fit it admitted was measuring a broken model.

    600 and 75 were chosen because 955 shots and 108 goals were reading a 16%
    shortfall that did not exist. With the measurement repaired there is no
    shortfall, so the reason for lowering the floor is gone with it.
    """
    assert xc.MIN_SHOTS >= 1500
    assert xc.MIN_GOALS >= 150


def test_the_cross_validated_test_still_gates_it():
    """Lowering the size floor must not lower the evidence bar."""
    import numpy as np

    import xg_calibration as xc

    # A model that is already perfectly calibrated has nothing to gain, so the
    # held-out folds refuse the correction however many shots are supplied.
    rng = np.random.default_rng(7)
    p = rng.uniform(0.02, 0.6, 4000)
    y = (rng.uniform(size=4000) < p).astype(float)
    assert xc.fit(p, y) is None


def test_a_correction_is_refused_below_the_floor_however_good_it_looks():
    import numpy as np

    import xg_calibration as xc

    rng = np.random.default_rng(3)
    p = rng.uniform(0.02, 0.6, 100)
    y = (rng.uniform(size=100) < p * 2).astype(float)
    assert xc.fit(p, y) is None, "100 shots is not evidence"


def test_nothing_is_bolted_onto_platt():
    """Platt is monotone; the ramp that used to sit on top of it was not.

    That ramp held the correction off the far tail and needed a clamp to stop
    it pushing a better chance under a worse one — shape written to make one
    fit survive a shot map, and that fit is gone. What is left has to be Platt
    exactly, or the gates above are testing something other than what ships.
    """
    for slope, intercept in ((1.4, -0.3), (0.7, -0.35), (2.2, -1.1), (0.5, 0.2)):
        correction = xc.Calibration(slope, intercept, 2000, 200, 0.30, 0.29)
        values = np.linspace(0.001, 0.99, 400)
        out = [correction.apply(v) for v in values]
        assert out == sorted(out), (slope, intercept)
        assert all(0.0 < v < 1.0 for v in out), (slope, intercept)
        for value in values[::37]:
            expected = xc._sigmoid(slope * xc._logit(value) + intercept)
            assert out[list(values).index(value)] == pytest.approx(expected)
