"""Reading a value out of a data frame without printing its absence.

Every module here pulls names and numbers out of pandas objects and drops them
straight into a sentence or a label. The idiom that spread through the codebase
was ``str(row.get("player") or "")``, and it is wrong twice:

- ``float("nan") or ""`` returns the NaN, because a NaN is truthy. So a missing
  scorer became the string "nan" in a published paragraph.
- ``row.get("player", "Goal")`` only uses the default when the *key* is absent.
  A column that exists with an empty cell returns the NaN instead.

Then ``str(...).split()[-1]`` raises IndexError on the empty string, which took
a chart down rather than a sentence.

The same defect has now been found in four separate files, which is the point
at which it stops being a bug and starts being a missing helper.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["text", "number", "whole", "surname", "ratio"]


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    # pandas NA / NaT and numpy scalars, without importing pandas here.
    try:
        import pandas as pd

        if pd.isna(value):
            return True
    except (TypeError, ValueError, ImportError):
        pass
    return str(value).strip().lower() in ("", "nan", "none", "nat", "<na>")


def text(value: Any, default: str = "") -> str:
    """A trimmed string, or ``default`` when the cell holds nothing usable."""
    if _missing(value):
        return default
    return str(value).strip()


def number(value: Any, default: float = 0.0) -> float:
    """A float, or ``default`` when the cell is empty or not a number.

    Infinities are treated as missing: they come from a division the caller did
    not guard, and printing "inf xG" is worse than printing the default.
    """
    if _missing(value):
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(result) or math.isinf(result):
        return float(default)
    return result


def whole(value: Any, default: int = 0) -> int:
    """An int, or ``default``. int(nan) raises; this does not."""
    return int(number(value, default))


_SURNAME_PARTICLES = {
    "van", "von", "de", "der", "den", "di", "da", "dos", "das", "du", "la", "le",
    "mac", "mc", "ter", "ten", "el", "al", "del", "della", "st", "st.",
}


def surname(value: Any, default: str = "") -> str:
    """The surname of a name, particles included, safely.

    The last word alone printed "Dijk" for van Dijk and "Allister" for Mac
    Allister on every board that names players. A particle before the last
    word is kept with it. ``"".split()[-1]`` raises IndexError, and a chart that
    cannot label a goal should still draw the goal.
    """
    parts = text(value).split()
    if not parts:
        return default
    start = len(parts) - 1
    while start > 1 and parts[start - 1].lower() in _SURNAME_PARTICLES:
        start -= 1
    return " ".join(parts[start:])


def short_name(value: Any) -> str:
    """Everything after the first name: "Di Lorenzo", "De Bruyne", "Kolo Muani".

    The last word alone printed "Lorenzo", "Bruyne" and "Muani".
    """
    parts = text(value).split()
    return " ".join(parts[1:]) if len(parts) > 1 else (parts[0] if parts else "")


def ratio(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    """numerator / denominator, or ``default`` when that is not a number."""
    bottom = number(denominator, 0.0)
    if not bottom:
        return float(default)
    return number(numerator, 0.0) / bottom
