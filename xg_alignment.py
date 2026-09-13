"""Align the local xG engine with the provider model the audience compares against.

The engine is calibrated against outcomes and, in aggregate, it is right: 2064
non-penalty shots on disk carry 226 expected goals for 209 scored. Read shot by
shot against Opta's own numbers, though, two systematic differences show up.

Ordinary shots agree closely -- root mean square difference 0.036 over 1556 of
them. Chances the provider flags "big chance" do not: 0.177 over 349, because
the engine gives the flag a broadly fixed lift while Opta prices each chance on
what it can see and spreads them from 0.03 to 0.99 where the engine spans 0.02
to 0.55. Sunderland 0-2 Arsenal is the shape of the problem: every ordinary
shot in it agreed, and the four flagged chances carried the whole disagreement.

So this fits a correction on top of the engine, separately for flagged and
unflagged shots, using what the event carries and the engine did not read: how
far a blocked shot travelled before the block (a defender two metres away is a
different chance from one eight metres away), the shot zone, the body part, and
whether it was a first-time strike.

What it cannot recover is where the defenders stood. Opta has that and the
event feed does not publish it, so the flagged-chance error floors near 0.16
however the correction is written -- interactions, ridge, dropping the engine
term and refitting from geometry alone were all tried and none moved it. Only
29 of the 349 flagged chances on disk were blocked, which is the sample the
hardest case has to learn from; it grows with the archive, and the fit is re-run.

Coefficients live in ``xg_alignment.json`` beside this file and are produced by
``scripts/fit_xg_alignment.py``. Without that file every value passes through
unchanged, so a fresh clone renders the engine's own numbers.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

ALIGNMENT_FILE = "xg_alignment.json"
METHOD = "outcome_fitted_big_chance_v1"

# Every term the fit may use, in one place, so the stored file and the reader
# cannot disagree about which name means what.
TERMS = (
    "const", "logit", "big_chance",
    "dist", "dist2", "inv_dist", "angle", "log_angle", "central", "dy",
    "header", "right_foot", "first_touch", "six_yard", "box_centre",
    "box_wide", "out_of_box", "individual", "intentional_assist",
    "fast_break", "through_ball", "cross", "one_on_one", "from_corner",
    # Read after the ball was struck, so they may describe the chance but can
    # never help predict whether it was scored: a blocked shot never is. The
    # fit that targets goals leaves them at zero; only a fit that targets
    # another model's published numbers may use them.
    "block_dist", "block_dist2", "block_known", "blocked",
)

# Anything the outcome fit is allowed to weigh. A shot cannot be priced on what
# happened to it after it left the boot.
PRE_SHOT = tuple(t for t in TERMS
                 if t not in {"block_dist", "block_dist2", "block_known", "blocked"})

# Where the shot was taken from is the engine's answer to give, and the distance
# calibration already corrects it on held-out matches. Letting this layer weigh
# distance as well put a positive coefficient on it -- the level correction was
# pulling close shots down and needed something to hold the far ones up -- and
# the published xG stopped falling with distance between 20 and 30 metres. Two
# corrections arguing over one variable is worse than either alone, so this one
# is allowed the level and the flags and nothing geometric.
GEOMETRIC = {"dist", "dist2", "inv_dist", "angle", "log_angle", "central", "dy",
             "six_yard", "box_centre", "box_wide", "out_of_box"}
LEVEL_TERMS = tuple(t for t in PRE_SHOT if t not in GEOMETRIC and t != "logit")

_LOADED: object = "unread"


def qualifier_names(row) -> set[str]:
    """The qualifier display names on one shot, from a live event or a CSV row.

    ``football_match_analysis._qnames`` reads the live payload and a pipe-joined
    string; the exported frames hold the repr of a Python list, which that
    function returns as one unusable token. Both spellings are handled here
    rather than guessing which side is calling.
    """
    if not hasattr(row, "get"):
        return set()
    quals = row.get("qualifiers") or []
    if isinstance(quals, (list, tuple)) and quals and isinstance(quals[0], dict):
        found = {str((q.get("type") or {}).get("displayName", "")) for q in quals
                 if isinstance(q, dict)}
        found.discard("")
        if found:
            return found
    raw = row.get("qualifier_names")
    if isinstance(raw, (list, tuple, set)):
        return {str(x) for x in raw if x and str(x).lower() != "nan"}
    text = str(raw or "").strip()
    if not text or text.lower() == "nan":
        return set()
    if text.startswith(("[", "(")):
        try:
            return {str(x) for x in ast.literal_eval(text)}
        except Exception:
            pass
    separator = "|" if "|" in text else ","
    return {part.strip().strip("'\" ") for part in text.split(separator) if part.strip()}


def _number(value, default=float("nan")) -> float:
    try:
        found = float(value)
        return default if math.isnan(found) else found
    except (TypeError, ValueError):
        return default


def block_distance(row, names: set[str]) -> tuple[float, bool]:
    """How far the ball travelled before a defender stopped it, in metres.

    Only meaningful where the shot was actually blocked: on a save the same
    qualifier pair records the goal line, which is a different fact and would
    read as a defender standing on it.
    """
    if "Blocked" not in names:
        return 0.0, False
    x, y = _number(row.get("x")), _number(row.get("y"))
    bx, by = _number(row.get("blocked_x")), _number(row.get("blocked_y"))
    if any(math.isnan(v) for v in (x, y, bx, by)):
        return 0.0, False
    return math.hypot((bx - x) * 1.05, (by - y) * 0.68), True


def features(value: float, geometry: dict, context: dict, row) -> dict:
    """The design row for one shot, named rather than positional."""
    names = qualifier_names(row)
    probability = min(max(float(value), 1e-6), 1 - 1e-6)
    distance = min(float(geometry["distance"]), 40.0)
    block, known = block_distance(row, names)
    block = min(block, 15.0)
    body = str(row.get("body_part") or "")
    shot_type = str(row.get("shot_whoscored_type") or "")
    return {
        "const": 1.0,
        "logit": math.log(probability / (1 - probability)),
        "big_chance": 1.0 if is_big_chance(row, context) else 0.0,
        "dist": distance / 40.0,
        "dist2": (distance / 40.0) ** 2,
        "inv_dist": 1.0 / (1.0 + distance),
        "angle": float(geometry["angle"]) / math.pi,
        "log_angle": math.log(max(float(geometry["angle"]), 1e-3)),
        "central": float(geometry["central"]),
        "dy": float(geometry["dy"]) / 34.0,
        "block_dist": block / 15.0,
        "block_dist2": (block / 15.0) ** 2,
        "block_known": 1.0 if known else 0.0,
        "blocked": 1.0 if shot_type == "BlockedShot" else 0.0,
        "header": 1.0 if body == "Head" or context.get("is_header") else 0.0,
        "right_foot": 1.0 if body == "RightFoot" else 0.0,
        "first_touch": 1.0 if "FirstTouch" in names else 0.0,
        "six_yard": 1.0 if "SmallBoxCentre" in names else 0.0,
        "box_centre": 1.0 if "BoxCentre" in names else 0.0,
        "box_wide": 1.0 if {"BoxLeft", "BoxRight"} & names else 0.0,
        "out_of_box": 1.0 if "OutOfBoxCentre" in names else 0.0,
        "individual": 1.0 if "IndividualPlay" in names else 0.0,
        "intentional_assist": 1.0 if "IntentionalAssist" in names else 0.0,
        "fast_break": 1.0 if context.get("is_fast") else 0.0,
        "through_ball": 1.0 if context.get("is_through") else 0.0,
        "cross": 1.0 if context.get("is_cross") else 0.0,
        "one_on_one": 1.0 if context.get("is_one_on_one") else 0.0,
        "from_corner": 1.0 if "FromCorner" in names else 0.0,
    }


def is_big_chance(row, context: dict) -> bool:
    if context.get("is_big"):
        return True
    return "BigChance" in qualifier_names(row)


def load(root=None):
    """The stored coefficients, or None when the file is absent or foreign."""
    base = Path(root) if root else Path(__file__).resolve().parent
    path = base / ALIGNMENT_FILE
    if not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if stored.get("method") != METHOD:
        return None
    if not isinstance(stored.get("weights"), dict):
        return None
    return stored


def align(value: float, geometry: dict, context: dict, row, stored=None) -> float:
    """The engine's number, moved onto the provider's scale.

    Returns the value untouched when no fit is stored or when anything in the
    row is unreadable: a correction that raises is worse than one that declines.
    Penalties never reach here - the engine returns those before it calibrates.
    """
    global _LOADED
    if stored is None:
        if _LOADED == "unread":
            _LOADED = load()
        stored = _LOADED
    if not stored:
        return float(value)
    try:
        weights = stored["weights"]
        row_features = features(value, geometry, context, row)
        total = sum(float(weights.get(name, 0.0)) * row_features[name] for name in TERMS)
        return 1.0 / (1.0 + math.exp(-total))
    except Exception:
        return float(value)
