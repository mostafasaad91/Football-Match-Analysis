"""VAEP drawn on the pitch rather than listed.

A ranking table says who added value; it does not say where, or how. Here each
player is a disc at the average spot of their on-ball actions, sized by the
value they added, so the shape of a side's contribution reads like a
formation: a full-back whose disc dwarfs the winger's, a goalkeeper who was
busy. Beside it, the single actions that moved the match most, and a split
bar of attack against defence for the leading players.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_analysis.metrics.frame_values import surname
from football_analysis.visuals.visualization_components import (
    BG_DARK, GRID_COL, TEXT_DIM, TEXT_MAIN, label_outline, themed_pitch,
)

_ACTION_WORDS = {
    "Pass": "pass", "Shot": "shot", "Goal": "goal", "SavedShot": "shot", "MissedShots": "shot",
    "TakeOn": "dribble", "Tackle": "tackle", "Interception": "interception", "Clearance": "clearance",
    "BallRecovery": "recovery", "Save": "save", "Claim": "claim", "Punch": "punch",
    "KeeperSweeper": "sweep", "Aerial": "header duel", "BlockedPass": "block", "Challenge": "duel",
}


def player_spots(values: pd.DataFrame, team_id) -> pd.DataFrame:
    """One row per player of a side: mean action location and VAEP totals."""
    own = values[values["team_id"] == team_id].dropna(subset=["player"])
    if own.empty:
        return pd.DataFrame(columns=["player", "x", "y", "vaep", "offensive", "defensive", "actions"])
    return (own.groupby("player")
            .agg(x=("x", "mean"), y=("y", "mean"), vaep=("vaep", "sum"),
                 offensive=("offensive", "sum"), defensive=("defensive", "sum"),
                 actions=("vaep", "size"))
            .reset_index()
            .query("actions >= 8")
            .sort_values("vaep", ascending=False))


def draw_value_map(ax, values: pd.DataFrame, team_id, colour: str, *, label_top: int = 6,
                   size_scale: float = 2600.0, fontsize: float = 9.0, flip: bool = False,
                   draw_pitch: bool = True, placed: list | None = None) -> pd.DataFrame:
    """Draw one side's value map on ``ax``, attacking left to right (or right to left with ``flip``).

    Both sides go on one pitch -- home attacking right, away flipped to attack
    left -- so the map gets the full width instead of two cramped squares.
    """
    if draw_pitch:
        themed_pitch(ax)
        # 105 x 68 proportions rather than themed_pitch's square.
        ax.set_aspect(68 / 105)
    spots = player_spots(values, team_id)
    if spots.empty:
        return spots
    if flip:
        spots = spots.assign(x=100 - spots["x"], y=100 - spots["y"])
    placed = [] if placed is None else placed
    positive = spots["vaep"].clip(lower=0)
    sizes = 120 + size_scale * positive / max(positive.max(), 0.25)
    for (_, r), s in zip(spots.iterrows(), sizes):
        if r["vaep"] >= 0:
            ax.scatter(r["x"], r["y"], s=s, color=colour, alpha=0.88, edgecolor="white", lw=1.4, zorder=4)
        else:  # gave value away: a hollow ring, kept small
            ax.scatter(r["x"], r["y"], s=150, facecolor="none", edgecolor=colour, lw=1.8, zorder=4)
    # Greedy placement: below the disc, else above, else right, else left --
    # whichever keeps the label clear of the labels already placed (by either side).
    for rank, (_, r) in enumerate(spots.iterrows()):
        if rank >= label_top:
            break
        radius = np.sqrt(sizes.iloc[rank]) * 0.55
        options = [((0, -8 - radius), "center", "top", (0, -9)),
                   ((0, 8 + radius), "center", "bottom", (0, 9)),
                   ((8 + radius, 0), "left", "center", (13, 0)),
                   ((-8 - radius, 0), "right", "center", (-13, 0)),
                   ((0, -30 - radius), "center", "top", (0, -20)),
                   ((0, 30 + radius), "center", "bottom", (0, 20))]
        chosen = options[0]
        for option in options:
            cx, cy = r["x"] + option[3][0], r["y"] + option[3][1]
            if all(abs(cx - px) > 17 or abs(cy - py) > 12 for px, py in placed):
                chosen = option
                break
        placed.append((r["x"] + chosen[3][0], r["y"] + chosen[3][1]))
        ax.annotate(f"{surname(str(r['player']))}\n{r['vaep']:+.2f}", (r["x"], r["y"]),
                    xytext=chosen[0], textcoords="offset points", ha=chosen[1], va=chosen[2],
                    color=TEXT_MAIN, fontsize=fontsize, fontweight="bold", zorder=6,
                    path_effects=label_outline(BG_DARK))
    # The greedy placement only knows label anchors; the drawn boxes of the two
    # sides' labels could still touch (Pote over Bremer). Settle them on screen.
    from football_analysis.visuals.visualization_components import resolve_label_overlaps

    resolve_label_overlaps(ax)
    return spots


def top_actions(values: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    return values.dropna(subset=["player"]).sort_values("vaep", ascending=False).head(n)


def action_label(row) -> str:
    word = _ACTION_WORDS.get(str(row["type"]), str(row["type"]).lower())
    return f"{int(row['minute'])}'  {surname(str(row['player']))}  ·  {word}"


def draw_split_bars(ax, spots_by_team: list[tuple[pd.DataFrame, str]], n: int = 8,
                    fontsize: float = 9.0) -> None:
    """Attack (solid) and defence (faded) for the leading players of both sides."""
    rows = []
    for spots, colour in spots_by_team:
        for _, r in spots.iterrows():
            rows.append((r["vaep"], r["player"], r["offensive"], r["defensive"], colour))
    rows = sorted(rows, key=lambda t: t[0], reverse=True)[:n][::-1]
    for i, (total, name, off, dfn, colour) in enumerate(rows):
        ax.barh(i, off, color=colour, height=0.62)
        ax.barh(i, dfn, left=max(off, 0) if dfn >= 0 else 0, color=colour, alpha=0.4, height=0.62)
        ax.text(max(off, 0) + max(dfn, 0) + 0.02, i, f"{total:+.2f}", va="center", color=TEXT_MAIN,
                fontsize=fontsize, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([surname(str(r[1])) for r in rows], color=TEXT_MAIN, fontsize=fontsize)
    ax.axvline(0, color=TEXT_DIM, lw=0.8)
    ax.set_facecolor((0, 0, 0, 0))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COL)
    ax.tick_params(colors=TEXT_DIM, labelsize=fontsize - 1)
    reach = max([max(r[2], 0) + max(r[3], 0) for r in rows] + [0.1])
    ax.set_xlim(min([min(r[2], 0) for r in rows] + [0]) * 1.2, reach * 1.25)
