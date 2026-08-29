"""Let the xG model correct itself, but only once the data says it should.

The shipped model is a logistic on distance, angle and context whose
coefficients were reasoned from published research rather than fitted. On the
matches collected so far its level holds up: 955 non-penalty shots produced 108
goals against 109.6 predicted, two tenths of a standard deviation.

Its shape does not, and no correction here can repair that. The model is 2.5
standard deviations too high inside eleven metres and 2.0 too low outside, the
two errors cancelling in the total that looks so good. What it is missing is
what the shot faced — Opta's own big-chance flag converts at 36% wherever it is
taken, while a shot from seven metres that Opta declined to flag converts at 3%,
and the model, which sees geometry and not goalkeepers, prices that one at 10%.
That is a coefficient the archive is nowhere near large enough to fit, and it is
not a slope and an intercept on the answer the model already gave.

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
# larger than any bias worth correcting. Roughly sixty matches, which puts the
# standard error on the goal total near 4%.
#
# This floor was briefly 600 and 75, lowered to admit a correction that read a
# 16% shortfall across 955 shots. That shortfall was not real. scripts/xg_report
# rebuilt each shot from the stored snapshot and joined its qualifier names with
# a comma, while the model splits that field on a pipe — so BigChance, Cross and
# Head never reached it and every shot was priced as an ordinary foot shot from
# open play, 22% under what the pipeline gives the same shots. The correction was
# fitted against a model the package does not ship. Measured properly the total
# is 109.6 xG against 108 goals, two tenths of a standard deviation, and there
# was never a level to correct.
MIN_SHOTS = 1500
MIN_GOALS = 150

# A correction has to earn its place on data it never saw.
FOLDS = 5
SEEDS = 8                 # one shuffle is a coin flip; see fit()
MIN_IMPROVEMENT = 0.002   # in log-loss, per shot

# Bands to hold the correction to, and how much of the mispricing it has to
# clear. Log-loss alone cannot police this model: seven hundred of its nine
# hundred shots sit below 0.10 and are priced correctly, so they decide the
# mean, and the fifty shots above 0.40 that carry the visible error cannot move
# it by MIN_IMPROVEMENT however wrong they are. A correction that halves the
# error where nobody looks and doubles it on every shot map passes on log-loss.
# So calibration is measured directly, per band, on pooled out-of-fold
# predictions — pooled because the absolute error of eleven shots in one fold is
# mostly the shuffle.
BANDS = ((0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.01))
MIN_BAND_GAIN = 0.05      # share of the banded mispricing the fit has to remove


@dataclass(frozen=True)
class Calibration:
    slope: float
    intercept: float
    shots: int
    goals: int
    log_loss_before: float
    log_loss_after: float

    def apply(self, probability: float) -> float:
        """The corrected probability.

        Platt and nothing else. There was a ramp here that held the correction
        off the far tail, and a clamp to stop the ramp reordering two shots,
        both of them written to make one particular fit survive contact with a
        shot map — the fit that turned out to be measuring a crippled model.
        With the measurement repaired, the fit it existed for is not the fit
        this file would produce, and a piece of shape nobody can derive from the
        data is worse than no correction at all.
        """
        return float(_sigmoid(self.slope * _logit(probability) + self.intercept))

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


def banded_error(probabilities, outcomes, bands=BANDS) -> float:
    """Goals mispriced, summed over the bands. Zero is perfect calibration.

    Errors are added band by band rather than over the whole sample, because a
    model that prices its clear chances high and its half-chances low reads as
    flawless on the total. This one does: 109.6 xG against 108 goals, and 2.5
    standard deviations too high inside eleven metres.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(sum(
        abs(p[(p >= low) & (p < high)].sum() - y[(p >= low) & (p < high)].sum())
        for low, high in bands
    ))


def _out_of_fold(logits: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray | None:
    """One corrected probability per shot, each from a fit that never saw it."""
    order = np.random.default_rng(seed).permutation(len(logits))
    folds = np.array_split(order, FOLDS)
    corrected = np.empty(len(logits), dtype=float)
    for index in range(FOLDS):
        test = folds[index]
        train = np.concatenate([folds[j] for j in range(FOLDS) if j != index])
        if len(np.unique(y[train])) < 2:
            return None
        slope, intercept = _fit_platt(logits[train], y[train])
        corrected[test] = 1.0 / (1.0 + np.exp(-(slope * logits[test] + intercept)))
    return corrected


def fit(probabilities, outcomes, *, min_shots: int = MIN_SHOTS,
        min_goals: int = MIN_GOALS) -> Calibration | None:
    """A correction, or None when the data does not support one.

    Returns None for four separate reasons, and they are all good ones: too few
    shots, too few goals, no improvement in log-loss on held-out folds, or a
    correction that leaves the bands worse calibrated than it found them.

    The last gate is the one with teeth. On the archive as it stands the fit
    comes out at slope 0.752 — the right shape for a model that is too
    confident at the top — and it still fails, because pulling the top down
    drags the four hundred shots below 0.05 up with it, from a mean of 0.035
    against an observed 0.035 to 0.049. Two parameters move the whole curve
    together, and this model's error does not run with its own probability: it
    runs with distance, too high inside eleven metres and too low outside, the
    two cancelling in the total. Nothing shaped like Platt can find that.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) < min_shots or y.sum() < min_goals:
        return None
    if len(np.unique(y)) < 2:
        return None

    logits = np.array([_logit(v) for v in p])

    # Cross-validated, and over several shuffles: with one, whether a
    # correction is accepted comes down to which shots landed in which fold.
    # Across eight seeds this model's fit ranged from -0.0008 to +0.0036
    # against a floor of 0.002 — three seeds would have written a file and five
    # would not.
    before, after, banded = [], [], []
    baseline_bands = banded_error(p, y)
    for seed in range(SEEDS):
        corrected = _out_of_fold(logits, y, seed)
        if corrected is None:
            return None
        before.append(_log_loss(p, y))
        after.append(_log_loss(corrected, y))
        banded.append(banded_error(corrected, y))

    if float(np.mean(before) - np.mean(after)) < MIN_IMPROVEMENT:
        return None
    if max(banded) > baseline_bands * (1.0 - MIN_BAND_GAIN):
        return None

    slope, intercept = _fit_platt(logits, y)
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


_LOOK_IT_UP = object()


def calibrated(probability: float, calibration=_LOOK_IT_UP) -> float:
    """Apply the stored correction if there is one. Identity if not.

    Passing None means "do not correct". It used to mean "go and find one",
    which is the same thing a caller writes when it wants the identity, so
    there was no way to ask for an uncorrected value at all.
    """
    if calibration is _LOOK_IT_UP:
        calibration = load()
    if calibration is None:
        return float(probability)
    return calibration.apply(float(probability))
