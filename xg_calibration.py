"""Distance calibration for the local (non-provider) xG model.

The archive exposes two cancelling errors: its highest probabilities (almost
all very close chances) are over-priced while normal shooting-range attempts
are under-priced. The first term is a monotone high-probability bend; the second
is a smooth distance gate. Validation always holds out complete matches.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

CALIBRATION_FILE = "xg_calibration.json"
METHOD = "monotone_close_and_distance_range_v1"
MIN_SHOTS, MIN_GOALS, MIN_MATCHES = 800, 90, 20
MIN_HIGH_SHOTS, MIN_RANGE_SHOTS = 40, 250
FOLDS = 5
MIN_LOG_LOSS_GAIN = 0.004
MIN_DISTANCE_ERROR_GAIN = 0.10
DISTANCE_BANDS = ((0.0, 4.0), (4.0, 7.0), (7.0, 11.0),
                  (11.0, 16.0), (16.0, 22.0), (22.0, 999.0))
PROBABILITY_BANDS = ((0, .05), (.05, .10), (.10, .20), (.20, .40), (.40, 1.01))
BANDS = PROBABILITY_BANDS  # public diagnostic name retained for xg_report
HIGH_KNEE = 0.37


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 40.0)))
    exp = math.exp(max(z, -40.0))
    return exp / (1.0 + exp)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _range_basis(distance: float) -> float:
    """Lift grows from 7m to full strength at 25m, then stays bounded."""
    d = max(float(distance), 0.0)
    return max(0.0, min(1.0, (d - 7.0) / 18.0))


@dataclass(frozen=True)
class Calibration:
    high_gain: float
    range_logit: float
    shots: int
    goals: int
    matches: int
    log_loss_before: float
    log_loss_after: float
    method: str = METHOD

    def apply(self, probability: float, distance: float | None = None) -> float:
        """Correct a local-model probability; missing distance is identity."""
        value = float(probability)
        if distance is None:
            return value
        # This bend is monotone in the original probability, so a farther shot
        # cannot become better merely because it crossed a distance threshold.
        if value > HIGH_KNEE:
            value = HIGH_KNEE + (value - HIGH_KNEE) * self.high_gain
        shooting_range = _range_basis(distance)
        if shooting_range:
            value = _sigmoid(_logit(value) + self.range_logit * shooting_range)
        return value

    def as_dict(self) -> dict:
        return {"method": self.method, "high_knee": HIGH_KNEE,
                "high_gain": self.high_gain,
                "range_logit": self.range_logit, "shots": self.shots,
                "goals": self.goals, "matches": self.matches,
                "log_loss_before": self.log_loss_before,
                "log_loss_after": self.log_loss_after}


def _log_loss(probabilities, outcomes) -> float:
    values = list(probabilities)
    total = 0.0
    for p, y in zip(values, outcomes):
        p = min(max(float(p), 1e-6), 1.0 - 1e-6)
        y = float(y)
        total -= y * math.log(p) + (1.0 - y) * math.log(1.0 - p)
    return total / max(len(values), 1)


def distance_banded_error(probabilities, outcomes, distances,
                          bands=DISTANCE_BANDS) -> float:
    p, y, d = list(probabilities), list(outcomes), list(distances)
    return float(sum(abs(sum(float(p[i]) for i in range(len(p)) if lo <= float(d[i]) < hi)
                         - sum(float(y[i]) for i in range(len(y)) if lo <= float(d[i]) < hi))
                     for lo, hi in bands))


def banded_error(probabilities, outcomes, bands=None) -> float:
    p, y = list(probabilities), list(outcomes)
    bands = bands or PROBABILITY_BANDS
    return float(sum(abs(sum(float(p[i]) for i in range(len(p)) if lo <= float(p[i]) < hi)
                         - sum(float(y[i]) for i in range(len(y)) if lo <= float(p[i]) < hi))
                     for lo, hi in bands))


def evaluate(probabilities, outcomes) -> dict:
    p, y = [float(v) for v in probabilities], [float(v) for v in outcomes]
    if not p:
        return {}
    goals, predicted = sum(y), sum(p)
    base = goals / len(y)
    return {"shots": len(p), "goals": int(goals), "predicted": predicted,
            "ratio": predicted / max(goals, 1.0), "log_loss": _log_loss(p, y),
            "baseline_log_loss": _log_loss([base] * len(p), y),
            "brier": sum((a-b)**2 for a, b in zip(p, y)) / len(p),
            "sigma": (goals-predicted) / math.sqrt(max(predicted, 1e-9))}


def _fit_coefficients(p, y, d) -> tuple[float, float]:
    """Small deterministic search over the two constrained shape parameters."""
    best = None
    for high_gain in (step / 20.0 for step in range(1, 21)):
        for range_logit in (step / 20.0 for step in range(0, 21)):
            candidate = Calibration(high_gain, range_logit, 0, 0, 0, 0.0, 0.0)
            corrected = [candidate.apply(probability, distance)
                         for probability, distance in zip(p, d)]
            score = _log_loss(corrected, y)
            if best is None or score < best[0]:
                best = (score, high_gain, range_logit)
    return best[1], best[2]


def _apply_many(p, d, coefficients):
    calibration = Calibration(coefficients[0], coefficients[1], 0, 0, 0, 0.0, 0.0)
    return [calibration.apply(probability, distance)
            for probability, distance in zip(p, d)]


def _group_folds(groups: Sequence[str], folds: int = FOLDS) -> list[set[str]]:
    counts = {}
    for group in groups:
        counts[str(group)] = counts.get(str(group), 0) + 1
    buckets = [(set(), 0) for _ in range(min(folds, len(counts)))]
    for group, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        target = min(range(len(buckets)), key=lambda i: buckets[i][1])
        buckets[target][0].add(group)
        buckets[target] = (buckets[target][0], buckets[target][1]+count)
    return [bucket[0] for bucket in buckets]


def fit(probabilities, outcomes, distances=None, groups=None, *,
        min_shots: int = MIN_SHOTS, min_goals: int = MIN_GOALS) -> Calibration | None:
    """Fit only if the correction wins on complete held-out matches."""
    p, y = [float(v) for v in probabilities], [float(v) for v in outcomes]
    if distances is None or groups is None:
        return None
    d, g = [float(v) for v in distances], [str(v) for v in groups]
    if not (len(p) == len(y) == len(d) == len(g)):
        raise ValueError("probabilities, outcomes, distances and groups must align")
    if len(p) < min_shots or sum(y) < min_goals or len(set(g)) < MIN_MATCHES:
        return None
    if sum(value > HIGH_KNEE for value in p) < MIN_HIGH_SHOTS:
        return None
    if sum(7.0 <= v < 30.0 for v in d) < MIN_RANGE_SHOTS:
        return None

    corrected = [None] * len(p)
    for held_out in _group_folds(g):
        train = [i for i, group in enumerate(g) if group not in held_out]
        test = [i for i, group in enumerate(g) if group in held_out]
        coefficients = _fit_coefficients([p[i] for i in train], [y[i] for i in train],
                                         [d[i] for i in train])
        for index, value in zip(test, _apply_many([p[i] for i in test],
                                                  [d[i] for i in test], coefficients)):
            corrected[index] = value
    if any(value is None for value in corrected):
        return None
    before, after = _log_loss(p, y), _log_loss(corrected, y)
    distance_before = distance_banded_error(p, y, d)
    distance_after = distance_banded_error(corrected, y, d)
    if before-after < MIN_LOG_LOSS_GAIN:
        return None
    if distance_after > distance_before*(1.0-MIN_DISTANCE_ERROR_GAIN):
        return None
    coefficients = _fit_coefficients(p, y, d)
    return Calibration(coefficients[0], coefficients[1], len(p), int(sum(y)),
                       len(set(g)), before, after)


def load(root=None) -> Calibration | None:
    path = (Path(root) if root else Path(__file__).resolve().parent) / CALIBRATION_FILE
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("method") != METHOD:
            return None
        if abs(float(raw.get("high_knee", HIGH_KNEE)) - HIGH_KNEE) > 1e-9:
            return None
        return Calibration(float(raw["high_gain"]), float(raw["range_logit"]),
                           int(raw.get("shots", 0)), int(raw.get("goals", 0)),
                           int(raw.get("matches", 0)), float(raw.get("log_loss_before", 0.0)),
                           float(raw.get("log_loss_after", 0.0)), METHOD)
    except Exception:
        return None


def save(calibration: Calibration, root=None) -> Path:
    path = (Path(root) if root else Path(__file__).resolve().parent) / CALIBRATION_FILE
    path.write_text(json.dumps(calibration.as_dict(), indent=2)+"\n", encoding="utf-8")
    return path


_LOOK_IT_UP = object()


def calibrated(probability: float, distance: float | None = None,
               calibration=_LOOK_IT_UP) -> float:
    if calibration is _LOOK_IT_UP:
        calibration = load()
    if calibration is None:
        return float(probability)
    return calibration.apply(float(probability), distance)
