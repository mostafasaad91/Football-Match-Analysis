"""The report cover's hero image, drawn from the match it opens.

The cover carried type and two rules and nothing else, on a page that is
mostly white. This puts the match itself on the front of the report: the
control surface both sides held, with every shot of the match on top of it.

Rendered rather than drawn in the PDF because both layers already exist as
computed surfaces in this project, and reportlab has no way to draw a
smoothly interpolated field. The image is written next to the report and
embedded from there.

It stays dark on both pages. A dark plate on the light page reads as a
photograph laid on paper, which is what a cover image should look like; a
light one would dissolve into the sheet, and the whole point of the artwork is
that it is the loudest thing on the front.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Circle, Rectangle

from match_metrics import pitch_control

# The image's own ground. Not pure black: the control surface has to darken
# toward the contested middle and needs somewhere below itself to go.
GROUND = "#0E1218"
LINE = "#FFFFFF"

# 1008x614pt on the page, rendered at 2x for print.
WIDTH_PX, HEIGHT_PX = 2016, 1228
DPI = 144

# The artwork is dark on both pages, so the kit colours reaching it have been
# fitted to the wrong ground: on the light page the report hands over PSG's
# raw #004170, which is correct against paper and nearly invisible here. They
# are lifted again, against this image's own ground.
MARK_FLOOR = 4.2


def _luminance(rgb) -> float:
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(rgb, other) -> float:
    bright, dark = sorted((_luminance(rgb), _luminance(other)))
    return (dark + 0.05) / (bright + 0.05)


def lift_for_artwork(colour: str, floor: float = MARK_FLOOR) -> str:
    """Brighten a kit colour until it carries against the artwork's ground."""
    import colorsys

    try:
        rgb = mcolors.to_rgb(colour)
    except ValueError:
        return colour
    ground = mcolors.to_rgb(GROUND)
    if _contrast(rgb, ground) >= floor:
        return colour
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    low, high = lightness, 1.0
    for _ in range(24):
        mid = (low + high) / 2
        if _contrast(colorsys.hls_to_rgb(hue, mid, saturation), ground) >= floor:
            high = mid
        else:
            low = mid
    return mcolors.to_hex(colorsys.hls_to_rgb(hue, high, saturation))


def _bool(series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _pitch(ax, lw: float, alpha: float):
    """Markings for a horizontal pitch on 0-100 by 0-100 axes."""
    ax.add_patch(Rectangle((0, 0), 100, 100, fill=False, ec=LINE, lw=lw, alpha=alpha))
    ax.plot([50, 50], [0, 100], color=LINE, lw=lw, alpha=alpha)
    ax.add_patch(Circle((50, 50), 9.15, fill=False, ec=LINE, lw=lw, alpha=alpha))
    for goal_x, direction in ((0.0, 1), (100.0, -1)):
        ax.add_patch(Rectangle((goal_x if direction > 0 else goal_x - 16.5, 21.1),
                               16.5, 57.8, fill=False, ec=LINE, lw=lw, alpha=alpha))
        ax.add_patch(Rectangle((goal_x if direction > 0 else goal_x - 5.8, 36.8),
                               5.8, 26.4, fill=False, ec=LINE, lw=lw, alpha=alpha))
        ax.scatter([goal_x + direction * 11], [50], s=7, color=LINE, alpha=alpha)
        ax.add_patch(Arc((goal_x + direction * 11, 50), 18.3, 18.3,
                         theta1=308 if direction > 0 else 128,
                         theta2=52 if direction > 0 else 232,
                         ec=LINE, lw=lw, alpha=alpha))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")


def build_cover_art(
    events: pd.DataFrame,
    out_path: Path | str,
    *,
    home_id: int,
    away_id: int,
    home_colour: str,
    away_colour: str,
) -> Path | None:
    """Render the cover image for one fixture. Returns None if it cannot.

    A cover that fails to draw its artwork must still produce a report, so
    every failure here is swallowed and the caller falls back to the plain
    typographic cover.
    """
    try:
        out_path = Path(out_path)
        home_id, away_id = int(home_id), int(away_id)
        home_colour = lift_for_artwork(home_colour)
        away_colour = lift_for_artwork(away_colour)

        fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI,
                         facecolor=GROUND)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(GROUND)

        # Layer one: the space each side held, as a smooth field. Kept muted so
        # the shots drawn in the same two colours still separate from it.
        surface, _shares = pitch_control(events, home_id, away_id)
        ramp = LinearSegmentedColormap.from_list(
            "control", [away_colour, GROUND, home_colour])
        ax.imshow(np.asarray(surface), origin="lower", extent=[0, 100, 0, 100],
                  cmap=ramp, vmin=0.18, vmax=0.82, aspect="auto",
                  interpolation="bicubic", alpha=0.55, zorder=1)

        _pitch(ax, lw=1.4, alpha=0.34)

        # Layer two: every shot, at full strength. Each side attacks its own
        # end, in the same frame the control surface is built in.
        pso = _bool(events.get("is_penalty_shootout"))
        if pso.empty:
            pso = pd.Series(False, index=events.index)
        shots = events[_bool(events["is_shot"]) & ~pso].dropna(subset=["x", "y"]).copy()
        shots["xG"] = pd.to_numeric(shots["xG"], errors="coerce").fillna(0).clip(lower=0)
        shots["_x"] = pd.to_numeric(shots["x"], errors="coerce")
        shots["_y"] = pd.to_numeric(shots["y"], errors="coerce")

        for team_id, colour, mirror in ((home_id, home_colour, False),
                                        (away_id, away_colour, True)):
            side = shots[shots["team_id"].eq(team_id)]
            if side.empty:
                continue
            px = 100.0 - side["_x"] if mirror else side["_x"]
            py = 100.0 - side["_y"] if mirror else side["_y"]
            sizes = 130 + side["xG"].to_numpy() * 3200
            goals = _bool(side["is_goal"])
            ax.scatter(px[~goals], py[~goals], s=sizes[~goals.to_numpy()],
                       facecolor=colour, edgecolor=colour, linewidth=1.8,
                       alpha=0.82, zorder=4)
            ax.scatter(px[goals], py[goals], s=sizes[goals.to_numpy()] * 1.7,
                       marker="*", facecolor=colour, edgecolor=LINE,
                       linewidth=1.8, alpha=0.98, zorder=6)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, facecolor=GROUND, dpi=DPI)
        plt.close(fig)
        return out_path
    except Exception:
        plt.close("all")
        return None
