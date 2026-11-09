"""Expected threat (xT) learned from the matches we have stored.

The package used to read xT from a 12x8 grid written by hand, whose six-yard
cells were worth 0.76 -- roughly double any published xT surface -- and summed
only the positive steps of a possession. A team then "generated 34.59 xT" in a
match, a number no reader could place against anything.

This module fits Karun Singh's xT surface by value iteration on every stored
open-play pass and shot, and values a completed pass as the change in threat
between where it started and where it ended. The grid is fitted once
(``scripts/fit_xt.py``) and read from ``data/models/xt_grid.json``.

    xT(z) = s(z) * g(z) + m(z) * sum_z' T(z -> z') * xT(z')

s: share of actions in z that are shots, g: their goal rate, m: share that are
moves (passes), T: where completed moves from z end, with failed moves counted
in the denominator so losing the ball costs threat.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from football_analysis.paths import PROJECT_ROOT

GRID_PATH = PROJECT_ROOT / "data" / "models" / "xt_grid.json"
NX, NY = 16, 12  # cells along the pitch length and across it

# Set-piece deliveries are restarts, not open-play moves the surface describes.
_RESTART_QUALIFIERS = ("CornerTaken", "FreekickTaken", "ThrowIn", "GoalKick", "IndirectFreekickTaken")


def _cells(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    cx = np.clip((pd.to_numeric(x, errors="coerce").fillna(0).to_numpy() / 100.0 * NX).astype(int), 0, NX - 1)
    cy = np.clip((pd.to_numeric(y, errors="coerce").fillna(0).to_numpy() / 100.0 * NY).astype(int), 0, NY - 1)
    return cx, cy


def _truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _open_play_passes(events: pd.DataFrame) -> pd.DataFrame:
    passes = events[_truthy(events.get("is_pass", pd.Series(False, index=events.index)))]
    passes = passes.dropna(subset=["x", "y", "end_x", "end_y"])
    if "qualifier_names" in passes.columns:
        q = passes["qualifier_names"].astype(str)
        restart = np.zeros(len(passes), dtype=bool)
        for name in _RESTART_QUALIFIERS:
            restart |= q.str.contains(name, regex=False).to_numpy()
        passes = passes[~restart]
    return passes


def _shots(events: pd.DataFrame) -> pd.DataFrame:
    shots = events[_truthy(events.get("is_shot", pd.Series(False, index=events.index)))]
    if "is_penalty" in shots.columns:
        shots = shots[~_truthy(shots["is_penalty"])]
    if "is_penalty_shootout" in shots.columns:
        shots = shots[~_truthy(shots["is_penalty_shootout"])]
    return shots.dropna(subset=["x", "y"])


def fit_grid(frames: Iterable[pd.DataFrame], iterations: int = 200) -> np.ndarray:
    """Fit the xT surface on many matches' events. Returns an (NX, NY) array."""
    shots_n = np.zeros((NX, NY))
    goals_n = np.zeros((NX, NY))
    moves_n = np.zeros((NX, NY))
    trans = np.zeros((NX * NY, NX * NY))
    for events in frames:
        if events is None or events.empty:
            continue
        shots = _shots(events)
        sx, sy = _cells(shots["x"], shots["y"])
        np.add.at(shots_n, (sx, sy), 1)
        np.add.at(goals_n, (sx, sy), _truthy(shots.get("is_goal", pd.Series(False, index=shots.index))).to_numpy())
        passes = _open_play_passes(events)
        px, py = _cells(passes["x"], passes["y"])
        np.add.at(moves_n, (px, py), 1)
        ok = passes.get("outcome", pd.Series("", index=passes.index)).astype(str).str.lower().eq("successful").to_numpy()
        ex, ey = _cells(passes["end_x"], passes["end_y"])
        np.add.at(trans, (px[ok] * NY + py[ok], ex[ok] * NY + ey[ok]), 1)

    actions = shots_n + moves_n
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(actions > 0, shots_n / actions, 0.0)
        m = np.where(actions > 0, moves_n / actions, 0.0)
        g = np.where(shots_n > 0, goals_n / shots_n, 0.0)
        t = np.where(moves_n.reshape(-1, 1) > 0, trans / moves_n.reshape(-1, 1), 0.0)
    xt = np.zeros((NX, NY))
    for _ in range(iterations):
        new = s * g + m * (t @ xt.reshape(-1)).reshape(NX, NY)
        if np.abs(new - xt).max() < 1e-7:
            xt = new
            break
        xt = new
    return xt


def save_grid(grid: np.ndarray, matches: int, path: Path = GRID_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model": "xT value iteration (Singh), open-play passes and non-penalty shots",
        "shape": [NX, NY],
        "matches": int(matches),
        "grid": [[round(float(v), 6) for v in row] for row in grid],
    }, indent=1), encoding="utf-8")
    return path


_GRID_MEMO: list = []


def load_grid(path: Path = GRID_PATH) -> np.ndarray | None:
    if not _GRID_MEMO:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            _GRID_MEMO.append(np.asarray(data["grid"], dtype=float))
        except Exception:
            _GRID_MEMO.append(None)
    return _GRID_MEMO[0]


def action_xt(events: pd.DataFrame, grid: np.ndarray | None = None) -> pd.Series:
    """Threat added by each event: end cell minus start cell for completed passes.

    Failed passes and every other event carry 0. Values can be negative: a
    completed pass backwards gives threat away, and a possession's total is
    the net of its steps, not only the ones that went forward.
    """
    grid = load_grid() if grid is None else grid
    out = pd.Series(0.0, index=events.index)
    if grid is None or events.empty or not {"x", "y", "end_x", "end_y"}.issubset(events.columns):
        return out
    is_pass = _truthy(events.get("is_pass", pd.Series(False, index=events.index)))
    ok = events.get("outcome", pd.Series("", index=events.index)).astype(str).str.lower().eq("successful")
    mask = is_pass & ok & events[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
    if not mask.any():
        return out
    sub = events[mask]
    sx, sy = _cells(sub["x"], sub["y"])
    ex, ey = _cells(sub["end_x"], sub["end_y"])
    out.loc[mask] = grid[ex, ey] - grid[sx, sy]
    return out


def rescore(events: pd.DataFrame) -> pd.DataFrame:
    """Replace the events' xT column with the learned surface, when one is fitted."""
    if load_grid() is None:
        return events
    events = events.copy()
    events["xT"] = action_xt(events).round(5)
    events.attrs["xt_model"] = "learned"
    return events
