"""The local xG correction must be distance-shaped and match-held-out."""
import json

import numpy as np
import pytest

import xg_calibration as xc


def _biased_archive(n=1200, seed=2):
    rng = np.random.default_rng(seed)
    p = rng.beta(1.4, 9.0, n)
    p[::5] = rng.uniform(.40, .85, len(p[::5]))
    d = rng.uniform(2.5, 32.0, n)
    truth = []
    for probability, distance in zip(p, d):
        value = probability
        if value > xc.HIGH_KNEE:
            value = xc.HIGH_KNEE + (value-xc.HIGH_KNEE)*.20
        truth.append(xc._sigmoid(xc._logit(value)+.45*xc._range_basis(distance)))
    y = (rng.random(n) < np.asarray(truth)).astype(float)
    groups = np.asarray([f"match-{i % 30}" for i in range(n)])
    return p, y, d, groups


def test_small_or_ungrouped_archives_earn_no_correction():
    p, y, d, groups = _biased_archive(300)
    assert xc.fit(p, y, d, groups) is None
    p, y, d, groups = _biased_archive(1200)
    assert xc.fit(p, y, d, None) is None


def test_clear_distance_bias_is_corrected_on_complete_match_folds():
    p, y, d, groups = _biased_archive()
    found = xc.fit(p, y, d, groups)
    assert found is not None
    assert found.high_gain < .9
    assert found.range_logit > 0.1
    assert found.log_loss_after < found.log_loss_before
    folds = xc._group_folds(groups)
    assert set().union(*folds) == set(groups)
    assert sum(len(fold) for fold in folds) == len(set(groups))


def test_shape_lowers_high_chances_and_lifts_range_without_exploding():
    correction = xc.Calibration(.15, .4, 955, 108, 37, .311, .299)
    assert correction.apply(.60, 4.0) < .60
    assert correction.apply(.10, 16.0) > .10
    assert .04 < correction.apply(.04, 31.0) < .07
    assert correction.apply(.25, None) == pytest.approx(.25)


def test_correction_does_not_reorder_equal_distance_shots():
    correction = xc.Calibration(.15, .4, 955, 108, 37, .311, .299)
    values = np.linspace(.001, .99, 300)
    for distance in (3.0, 6.0, 12.0, 18.0, 35.0):
        out = [correction.apply(value, distance) for value in values]
        assert out == sorted(out)
        assert all(0.0 < value < 1.0 for value in out)


def test_identity_and_round_trip(tmp_path):
    assert xc.calibrated(.25, 5.0, None) == pytest.approx(.25)
    original = xc.Calibration(.20, .45, 2100, 260, 60, .31, .30)
    xc.save(original, tmp_path)
    loaded = xc.load(tmp_path)
    assert loaded is not None
    assert loaded.high_gain == pytest.approx(original.high_gain)
    assert loaded.range_logit == pytest.approx(original.range_logit)
    raw = json.loads((tmp_path / xc.CALIBRATION_FILE).read_text())
    assert raw["method"] == xc.METHOD
    assert raw["matches"] == 60


def test_old_or_damaged_files_are_ignored(tmp_path):
    path = tmp_path / xc.CALIBRATION_FILE
    path.write_text('{"knee": 0.37, "gain": 0.15}', encoding="utf-8")
    assert xc.load(tmp_path) is None
    path.write_text("{not json", encoding="utf-8")
    assert xc.load(tmp_path) is None


def test_evaluate_reports_an_exact_total_as_zero_sigma():
    predicted = np.full(400, .10)
    outcomes = np.zeros(400); outcomes[:40] = 1.0
    report = xc.evaluate(predicted, outcomes)
    assert report["goals"] == 40
    assert abs(report["sigma"]) < .01


def test_shipped_calibration_has_the_evidence_and_direction_claimed():
    found = xc.load()
    assert found is not None
    assert found.shots >= xc.MIN_SHOTS
    assert found.goals >= xc.MIN_GOALS
    assert found.matches >= xc.MIN_MATCHES
    assert 0 < found.high_gain < 1
    assert found.range_logit > 0
    assert found.log_loss_after < found.log_loss_before


def test_main_pipeline_passes_distance_but_leaves_penalties_alone():
    import football_match_analysis as fa

    fa._XG_CALIBRATION = xc.Calibration(.15, .4, 955, 108, 37, .311, .299)
    close = {"x": 97.0, "y": 50.0, "is_shot": True,
             "qualifier_names": "BigChance", "big_chance": True}
    raw = fa._xg_foot_shot(fa._shot_geometry_features(close),
                           fa._shot_context_features(close))
    assert fa._opta_like_local_xg_from_row(close) < raw
    penalty = dict(close, is_penalty=True, qualifier_names="Penalty")
    assert fa._opta_like_local_xg_from_row(penalty) == fa.XG_PENALTY_VALUE


def test_foot_model_still_falls_with_distance_after_calibration():
    import football_match_analysis as fa

    fa._XG_CALIBRATION = "unread"
    values = [fa._opta_like_local_xg_from_row(
        {"x": x, "y": 50.0, "is_shot": True, "qualifier_names": ""})
        for x in range(99, 24, -1)]
    for near, far in zip(values, values[1:]):
        assert far <= near + 1e-9


def test_provider_values_and_fixed_penalties_bypass_local_calibration(monkeypatch):
    import football_match_analysis as fa

    fa._XG_CALIBRATION = xc.Calibration(.05, 1.0, 955, 108, 37, .311, .299)
    monkeypatch.setattr(fa, "XG_USE_PROVIDER_SHOT_XG", True)
    assert fa._opta_like_local_xg_from_row({"xG": .23}) == pytest.approx(.23)
    assert fa._opta_like_local_xg_from_row(
        {"x": 97, "y": 50, "is_penalty": True}) == fa.XG_PENALTY_VALUE


def test_far_tail_remains_in_published_ranges():
    import football_match_analysis as fa

    fa._XG_CALIBRATION = "unread"

    def at(metres):
        for x in range(99, 0, -1):
            row = {"x": x, "y": 50.0, "is_shot": True, "qualifier_names": ""}
            if fa._shot_geometry_features(row)["distance"] >= metres:
                return fa._opta_like_local_xg_from_row(row)
        return 0.0

    assert .010 <= at(30) <= .05
    assert .003 <= at(40) <= .02
    assert at(60) <= .01
    assert at(70) <= .005


def test_wider_angle_and_headers_keep_their_expected_order():
    import football_match_analysis as fa

    fa._XG_CALIBRATION = "unread"
    central = fa._opta_like_local_xg_from_row(
        {"x": 88, "y": 50, "is_shot": True, "qualifier_names": ""})
    wide = fa._opta_like_local_xg_from_row(
        {"x": 88, "y": 85, "is_shot": True, "qualifier_names": ""})
    assert wide < central
    headers = [fa._opta_like_local_xg_from_row(
        {"x": x, "y": 50, "is_shot": True, "is_header": True,
         "qualifier_names": "Head"}) for x in range(99, 40, -1)]
    assert headers == sorted(headers, reverse=True)
