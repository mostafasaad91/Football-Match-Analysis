"""Post-shot xG: how likely a shot on target was to beat the keeper, as struck.

The report judges goalkeepers on post-shot xG faced against goals conceded, and
for a season it could not publish that difference. The placement heuristic in
``match_metrics.post_shot_xg`` multiplies the pre-shot xG by how far the ball
was placed from the keeper, and across the 2434 on-target shots Opta has
priced it came to 436.5 for 754 goals: every keeper in every report would have
"prevented" minus six tenths of a goal a match, which is a statement about the
heuristic rather than the keepers.

This is the same answer fitted instead of guessed. The target is Opta's own
post-shot value (xGOT) for every on-target shot FotMob lists, paired to our
events by ``reference_xg``; the inputs are what our events carry -- the
pre-shot xG, where the ball crossed the line, and whether it was headed. On
2434 shots over 269 matches it is 0.16 from Opta per shot out of fold, sums to
770.1 against Opta's 770.4, and cuts the per-match total error from 1.25 to
0.36.

Coefficients live in ``data/models/psxg_model.json``, written by
``scripts/fit_psxg.py`` and refitted after every round. Without the file every
value falls back to the heuristic, so a fresh clone renders as before.
Penalties and own goals never reach here: Opta's shot map carries neither, so
there is nothing to fit them on.
"""

from __future__ import annotations

import json
import math

from football_analysis.paths import MODELS_DIR

MODEL_FILE = "psxg_model.json"
METHOD = "psxg_opta_fitted_v1"

TERMS = (
    "const",
    "logit",  # the pre-shot chance, on the log-odds scale
    "wide",  # |x| across the goal, 0 at the centre and 1 at a post
    "wide2",
    "high",  # y up the goal, 0 on the ground and 1 at the bar
    "high2",
    "corner",  # wide and high together: the placement no keeper reaches
    "low",  # along the ground
    "header",
)

_LOADED: object = "unread"


def features(xg: float, px: float, py: float, header: bool) -> dict[str, float]:
    """The design row for one on-target shot, named rather than positional."""
    probability = min(max(float(xg), 1e-4), 1 - 1e-4)
    wide = min(abs(float(px)), 1.0)
    high = min(max(float(py), 0.0), 1.0)
    return {
        "const": 1.0,
        "logit": math.log(probability / (1 - probability)),
        "wide": wide,
        "wide2": wide * wide,
        "high": high,
        "high2": high * high,
        "corner": wide * high,
        "low": 1.0 if high < 0.08 else 0.0,
        "header": 1.0 if header else 0.0,
    }


def load(root=None) -> dict | None:
    """The stored coefficients, or None when the file is absent or foreign."""
    from pathlib import Path

    path = (Path(root) if root else MODELS_DIR) / MODEL_FILE
    if not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if stored.get("method") != METHOD or not isinstance(stored.get("weights"), dict):
        return None
    return stored


def stored() -> dict | None:
    """The shipped fit, read once per process."""
    global _LOADED
    if _LOADED == "unread":
        _LOADED = load()
    return _LOADED or None


def predict(xg: float, px: float, py: float, header: bool, model: dict | None = None):
    """PSxG for one on-target shot, or None when no fit is stored.

    Never raises: a row the model cannot read is handed back to the caller's
    fallback rather than failing the render.
    """
    model = model if model is not None else stored()
    if not model:
        return None
    try:
        weights = model["weights"]
        row = features(xg, px, py, header)
        total = sum(float(weights.get(name, 0.0)) * row[name] for name in TERMS)
        return 1.0 / (1.0 + math.exp(-total))
    except Exception:
        return None
