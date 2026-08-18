"""Let the xG model correct itself, but only once the data says it should.

The shipped model is a logistic on distance, angle and context whose
coefficients were reasoned from published research rather than fitted. On the
matches collected so far it holds up: 358 non-penalty shots produced 41 goals
against 37.4 predicted, which is 0.6 of a standard deviation and no evidence of
bias at all.

That number is the whole reason this file is careful. Fitting a full xG model
takes tens of thousands of shots — around ten goals per coefficient is the
usual floor, and the model has fifteen. Fitting fifteen on forty-one goals does
not learn the game, it memorises fourteen matches and gets worse everywhere
else. So nothing here fits the model.

What it fits is two numbers: a slope and an intercept on the model's own log
odds, which is Platt scaling. Two parameters is a defensible ask of a few
hundred shots, and it can only stretch or shift the existing curve, never
invent a new shape.

Even that is gated. The correction is accepted only when there are enough
shots to see it, and only when it beats the uncorrected model on data it was
not fitted to. Otherwise the file it would have written is not written, and
the model ships exactly as it is — which is the expected outcome today and for
a while yet.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CALIBRATION_FILE = "xg_calibration.json"

# Below this there is nothing to learn: the sampling noise on the total is
# larger than any bias worth correcting. 1500 shots is roughly 60 matches and
# puts the standard error on the goal total near 4%.
MIN_SHOTS = 1500
MIN_GOALS = 150

# A correction has to earn its place on data it never saw.
FOLDS = 5
MIN_IMPROVEMENT = 0.002   # in log-loss, per shot


@dataclass(frozen=True)
class Calibration:
    slope: float
    intercept: float
    shots: int
    goals: int
    log_loss_before: float
    log_loss_after: float

    def apply(self, probability: float) -> float:
        return _sigmoid(self.slope * _logit(probability) + self.intercept)

    def as_dict(self) -> dict:
        return {
            "slope": self.slope, "intercept": self.intercept,
            "shots": self.shots, "goals": self.goals,
            "log_loss_before": self.log_loss_before,
            "log_loss_after": self.log_loss_after,
        }


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp = math.exp(z)
    return exp / (1.0 + exp)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _fit_platt(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and intercept on the model's log odds."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits.reshape(-1, 1), y)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def evaluate(probabilities, outcomes) -> dict:
    """What the current model is worth on these shots. No fitting."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if not len(p):
        return {}
    base = float(y.mean()) or 1e-6
    return {
        "shots": int(len(p)),
        "goals": int(y.sum()),
        "predicted": float(p.sum()),
        "ratio": float(p.sum() / max(y.sum(), 1)),
        "log_loss": _log_loss(p, y),
        "baseline_log_loss": _log_loss(np.full_like(p, base), y),
        "brier": float(((p - y) ** 2).mean()),
        # How far the total sits from the goals scored, in standard deviations
        # of a Poisson count. Under about two, the difference is the sample.
        "sigma": float((y.sum() - p.sum()) / math.sqrt(max(p.sum(), 1e-9))),
    }


def fit(probabilities, outcomes, *, min_shots: int = MIN_SHOTS,
        min_goals: int = MIN_GOALS) -> Calibration | None:
    """A correction, or None when the data does not support one.

    Returns None for three separate reasons, and they are all good ones: too
    few shots, too few goals, or no improvement on held-out folds.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) < min_shots or y.sum() < min_goals:
        return None
    if len(np.unique(y)) < 2:
        return None

    logits = np.array([_logit(v) for v in p])

    # Cross-validated first: a correction that only helps the shots it was
    # fitted on is the overfitting this file exists to avoid.
    order = np.random.default_rng(0).permutation(len(p))
    folds = np.array_split(order, FOLDS)
    before, after = [], []
    for index in range(FOLDS):
        test = folds[index]
        train = np.concatenate([folds[j] for j in range(FOLDS) if j != index])
        if len(np.unique(y[train])) < 2:
            return None
        slope, intercept = _fit_platt(logits[train], y[train])
        corrected = 1.0 / (1.0 + np.exp(-(slope * logits[test] + intercept)))
        before.append(_log_loss(p[test], y[test]))
        after.append(_log_loss(corrected, y[test]))

    improvement = float(np.mean(before) - np.mean(after))
    if improvement < MIN_IMPROVEMENT:
        return None

    slope, intercept = _fit_platt(logits, y)
    corrected = 1.0 / (1.0 + np.exp(-(slope * logits + intercept)))
    return Calibration(slope, intercept, int(len(p)), int(y.sum()),
                       float(np.mean(before)), float(np.mean(after)))


def load(root=None) -> Calibration | None:
    """The stored correction, if one was ever earned."""
    base = Path(root) if root else Path(__file__).resolve().parent
    path = base / CALIBRATION_FILE
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Calibration(
            float(raw["slope"]), float(raw["intercept"]),
            int(raw.get("shots", 0)), int(raw.get("goals", 0)),
            float(raw.get("log_loss_before", 0.0)),
            float(raw.get("log_loss_after", 0.0)),
        )
    except Exception:
        return None


def save(calibration: Calibration, root=None) -> Path:
    base = Path(root) if root else Path(__file__).resolve().parent
    path = base / CALIBRATION_FILE
    path.write_text(json.dumps(calibration.as_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def calibrated(probability: float, calibration: Calibration | None = None) -> float:
    """Apply the stored correction if there is one. Identity if not."""
    if calibration is None:
        calibration = load()
    if calibration is None:
        return float(probability)
    return calibration.apply(float(probability))
