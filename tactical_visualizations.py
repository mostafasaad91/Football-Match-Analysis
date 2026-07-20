"""
tactical_visualizations.py — production v2 chart renderers + DataFrame adapters.

Each visual exposes two functions:
    • render_<name>_v2(...)  — pure renderer; takes simple data structures,
                              returns a matplotlib Figure (no I/O).
    • make_<name>_v2(events, info, ...)  — DataFrame adapter; converts the
                              project's events DataFrame to the renderer's
                              expected shape and calls render_*.

The renderers compose the design-system primitives from visualization_components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import textwrap as _tw
import re
from matplotlib.colors import LinearSegmentedColormap, to_rgba

from visualization_components import (
    chrome,
    panel_card,
    metric_strip,
    key_insight,
    themed_pitch,
    raised_panel_backdrop,
    BG_DARK,
    BG_MID,
    BG_PANEL,
    BG_HEADER,
    BG_PITCH,
    GRID_COL,
    GRID_SOFT,
    TEXT_BR,
    TEXT_MAIN,
    TEXT_DIM,
    TEXT_FAD,
    C_HOME,
    C_AWAY,
    C_GOLD,
    shadow,
    readable_on,
    readable_team_text,
    network_link_palette,
    ACCENT_TEXT,
    FONT_SANS,
    FONT_MONO,
    panel_header_geom,
)
from visualization_components import _panel_rect
from match_metrics import (
    cross_mask,
    defensive_block_events,
    defensive_blocks_count,
    fouls_committed_count,
    high_regain_events,
    player_sequence_metrics,
    progressive_pass_mask,
    team_advanced_metrics,
)

IS_LIGHT_THEME = BG_DARK.upper() in {"#FFFFFF", "WHITE"}
ROW_BG = "#FFFFFF" if IS_LIGHT_THEME else "#101010"
MID_BG = "#FFFFFF" if IS_LIGHT_THEME else "#0a0a0a"
PASS_ARROW = "#111827" if IS_LIGHT_THEME else "#F8FAFC"
PASS_NEG = "#7F1D1D" if IS_LIGHT_THEME else "#FCA5A5"
GOAL_ROW_HOME = "#FFFFFF" if IS_LIGHT_THEME else "#0d0d0d"
GOAL_ROW_AWAY = "#F1F5F9" if IS_LIGHT_THEME else "#0d0d0d"
C_GREEN = "#3DDC84"


# ─────────────────────────────────────────────────────────────────────────────
# FINAL VISUAL PATCH: safe vertical pitch helpers for v2 pitch visuals only
# ─────────────────────────────────────────────────────────────────────────────
VP_W = 54.0
VP_L = 105.0
XT_ARROW = C_HOME if not IS_LIGHT_THEME else "#62617A"
XT_NEG_ARROW = C_AWAY if not IS_LIGHT_THEME else "#7F1D1D"  # negative-xT accent
TEAM_COLOR_FALLBACK = C_HOME


def _clean_dark_navy(color: str | None) -> str:
    """Preserve the supplied team identity on AMOLED backgrounds.

    Match colours have already been clash-checked by the main pipeline. Older
    versions replaced dark blues with a generic light blue, which made clubs
    and national teams lose their real identity. Only near-black colours are
    lifted, and the lift keeps the original hue instead of substituting cyan.
    """
    if not color:
        return color or TEAM_COLOR_FALLBACK
    try:
        r, g, b, _a = to_rgba(color)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum < 0.055 and max(r, g, b) < 0.28:
            import colorsys

            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            r, g, b = colorsys.hsv_to_rgb(h, max(s, 0.68), max(v, 0.62))
            return "#{:02X}{:02X}{:02X}".format(
                round(r * 255), round(g * 255), round(b * 255)
            )
    except Exception:
        pass
    return str(color)


def _vp_xy(x, y):
    """Map WhoScored 0-100 coords to vertical 68x105 pitch coords.
    Old x = attack progress left→right, old y = lateral width.
    New y = attack progress bottom→top, new x = lateral width.

    The width axis is MIRRORED (100 - y): in this feed y=0 is the RIGHT flank
    and y=100 the LEFT flank, so a right-sided player (e.g. a right winger,
    low y) must render on the RIGHT of an attack-up pitch. This matches the
    pass-network / average-positions convention (both already mirrored); the
    shot map, xT map, danger creation, Zone 14 and cross maps all share this
    helper and were previously drawn left-right reversed.
    """
    return (
        (100.0 - np.asarray(y, dtype=float)) * VP_W / 100.0,
        np.asarray(x, dtype=float) * VP_L / 100.0,
    )


def _vp_point(x, y):
    nx, ny = _vp_xy(float(x), float(y))
    return float(nx), float(ny)


def _is_pitch_coord(x, y) -> bool:
    try:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        return bool(np.all((xa >= 0) & (xa <= 100)) and np.all((ya >= 0) & (ya <= 100)))
    except Exception:
        return False


def _vp_extent(extent):
    xmin, xmax, ymin, ymax = [float(v) for v in extent]
    return [
        ymin * VP_W / 100.0,
        ymax * VP_W / 100.0,
        xmin * VP_L / 100.0,
        xmax * VP_L / 100.0,
    ]


def _draw_vertical_pitch(
    ax,
    *,
    attacking_only: bool = False,
    line_color: str = "#3A3A3A",
    line_alpha: float = 0.56,
):
    """Draw a narrow vertical pitch without touching visualization_components.themed_pitch."""
    ax.set_facecolor(BG_PITCH)
    ax.set_aspect("equal")
    ax.set_xlim(-2, VP_W + 2)
    ax.set_ylim((VP_L * 0.50 - 2) if attacking_only else -2, VP_L + 2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL)
        s.set_linewidth(1.0)
        s.set_alpha(1.0)
    lc = dict(color=line_color, lw=1.05, alpha=line_alpha, zorder=2)
    y0 = VP_L * 0.50 if attacking_only else 0.0
    # Boundary + halfway
    ax.plot([0, VP_W, VP_W, 0, 0], [y0, y0, VP_L, VP_L, y0], **lc)
    if not attacking_only:
        ax.plot([0, VP_W], [VP_L / 2, VP_L / 2], **lc)
        ax.add_patch(mpatches.Circle((VP_W / 2, VP_L / 2), 9.15, fill=False, **lc))
        ax.scatter(
            [VP_W / 2], [VP_L / 2], s=6, color=line_color, alpha=line_alpha, zorder=2
        )
    # Boxes + spots + goal lines — widths kept proportional to the pitch width
    # (real ratios on a 68 m pitch) so a narrower VP_W still looks correct.
    pa_w, ga_w, goal_w = 0.593 * VP_W, 0.269 * VP_W, 0.108 * VP_W
    for gy, sign in [(0.0, 1), (VP_L, -1)]:
        if attacking_only and gy == 0.0:
            continue
        ax.plot(
            [
                (VP_W - pa_w) / 2,
                (VP_W - pa_w) / 2,
                (VP_W + pa_w) / 2,
                (VP_W + pa_w) / 2,
            ],
            [gy, gy + sign * 16.5, gy + sign * 16.5, gy],
            **lc,
        )
        ax.plot(
            [
                (VP_W - ga_w) / 2,
                (VP_W - ga_w) / 2,
                (VP_W + ga_w) / 2,
                (VP_W + ga_w) / 2,
            ],
            [gy, gy + sign * 5.5, gy + sign * 5.5, gy],
            **lc,
        )
        ax.scatter(
            [VP_W / 2],
            [gy + sign * 11],
            color=line_color,
            s=5,
            alpha=line_alpha,
            zorder=2,
        )
        ax.plot(
            [(VP_W - goal_w) / 2, (VP_W + goal_w) / 2],
            [gy, gy],
            color=line_color,
            lw=1.45,
            alpha=line_alpha,
            zorder=2,
        )
    if not attacking_only:
        ax.add_patch(
            mpatches.Arc(
                (VP_W / 2, 16.5),
                18,
                18,
                angle=0,
                theta1=180,
                theta2=360,
                color=line_color,
                lw=0.95,
                alpha=line_alpha * 0.70,
                zorder=2,
            )
        )
    ax.add_patch(
        mpatches.Arc(
            (VP_W / 2, VP_L - 16.5),
            18,
            18,
            angle=0,
            theta1=0,
            theta2=180,
            color=line_color,
            lw=0.95,
            alpha=line_alpha * 0.70,
            zorder=2,
        )
    )


def _draw_vertical_attack_arrow(ax, *, x=-1.3, y0=3.0, y1=16.0):
    # Soft backdrop strip so the arrow + label stay legible even when a
    # heatmap or dense overlay sits directly behind this corner of the
    # pitch (e.g. Pass Target Zones' imshow covers the full 0-100 extent).
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x - 2.0, y0 - 2.2),
            15.5,
            (y1 - y0) + 4.4,
            boxstyle="round,pad=0.0,rounding_size=1.6",
            facecolor=BG_DARK,
            edgecolor="none",
            alpha=0.62,
            zorder=29,
        )
    )
    ar = mpatches.FancyArrowPatch(
        (x, y0),
        (x, y1),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.6,
        color=XT_ARROW,
        alpha=0.95,
        zorder=30,
    )
    ax.add_patch(ar)
    ax.text(
        x + 2.6,
        (y0 + y1) / 2,
        "ATTACK",
        ha="left",
        va="center",
        color=XT_ARROW,
        fontsize=7.5,
        fontweight="bold",
        alpha=1.0,
        family=FONT_MONO,
        zorder=31,
    )


def _draw_vertical_arrow(
    ax,
    start,
    end,
    *,
    color=XT_ARROW,
    lw=2.0,
    alpha=0.9,
    mutation_scale=12.5,
    rad=0.04,
    zorder=10,
):
    sx, sy = _vp_point(start[0], start[1])
    ex, ey = _vp_point(end[0], end[1])
    ar = mpatches.FancyArrowPatch(
        (sx, sy),
        (ex, ey),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        alpha=alpha,
        zorder=zorder,
        shrinkA=0,
        shrinkB=0,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(ar)
    return ar


class _VerticalPitchProxy:
    """Proxy for existing overlay callbacks that still use 0-100 coordinates."""

    def __init__(self, ax):
        object.__setattr__(self, "_ax", ax)

    def __getattr__(self, name):
        return getattr(self._ax, name)

    @property
    def transAxes(self):
        return self._ax.transAxes

    @property
    def transData(self):
        return self._ax.transData

    def plot(self, x, y, *args, **kwargs):
        if _is_pitch_coord(x, y):
            x, y = _vp_xy(x, y)
        return self._ax.plot(x, y, *args, **kwargs)

    def scatter(self, x, y, *args, **kwargs):
        if _is_pitch_coord(x, y):
            x, y = _vp_xy(x, y)
        return self._ax.scatter(x, y, *args, **kwargs)

    def text(self, x, y, s, *args, **kwargs):
        if kwargs.get("transform") is None and _is_pitch_coord(x, y):
            x, y = _vp_point(x, y)
        return self._ax.text(x, y, s, *args, **kwargs)

    def annotate(self, text, xy, xytext=None, *args, **kwargs):
        if _is_pitch_coord(xy[0], xy[1]):
            xy = _vp_point(xy[0], xy[1])
        if xytext is not None and _is_pitch_coord(xytext[0], xytext[1]):
            xytext = _vp_point(xytext[0], xytext[1])
        return self._ax.annotate(text, xy=xy, xytext=xytext, *args, **kwargs)

    def imshow(self, X, *args, **kwargs):
        extent = kwargs.get("extent")
        if (
            extent is not None
            and len(extent) == 4
            and _is_pitch_coord([extent[0], extent[1]], [extent[2], extent[3]])
        ):
            # Transpose (length→vertical, width→horizontal) then mirror the
            # width axis so the heatmap lands on the same flanks as the mirrored
            # scatter overlays. A 2D `alpha` array is flipped identically.
            X = np.asarray(X).T[:, ::-1]
            al = kwargs.get("alpha")
            if isinstance(al, np.ndarray) and al.ndim == 2:
                kwargs["alpha"] = np.asarray(al).T[:, ::-1]
            kwargs["extent"] = _vp_extent(extent)
        return self._ax.imshow(X, *args, **kwargs)

    def contour(self, Z, *args, **kwargs):
        extent = kwargs.get("extent")
        if (
            extent is not None
            and len(extent) == 4
            and _is_pitch_coord([extent[0], extent[1]], [extent[2], extent[3]])
        ):
            Z = np.asarray(Z).T[:, ::-1]
            kwargs["extent"] = _vp_extent(extent)
        return self._ax.contour(Z, *args, **kwargs)

    def axvline(self, x=0, *args, **kwargs):
        if 0 <= float(x) <= 100:
            yy = float(x) * VP_L / 100.0
            return self._ax.plot([0, VP_W], [yy, yy], *args, **kwargs)
        return self._ax.axvline(x, *args, **kwargs)

    def axhline(self, y=0, *args, **kwargs):
        # y is a WIDTH value (0-100) → a vertical line on the pitch. Mirror it
        # (100 - y) to match _vp_xy's width convention (y=0 = right flank).
        if 0 <= float(y) <= 100:
            xx = (100.0 - float(y)) * VP_W / 100.0
            return self._ax.plot([xx, xx], [0, VP_L], *args, **kwargs)
        return self._ax.axhline(y, *args, **kwargs)

    def axvspan(self, xmin, xmax, *args, **kwargs):
        if 0 <= float(xmin) <= 100 and 0 <= float(xmax) <= 100:
            y0 = float(xmin) * VP_L / 100.0
            y1 = float(xmax) * VP_L / 100.0
            return self._ax.axhspan(y0, y1, *args, **kwargs)
        return self._ax.axvspan(xmin, xmax, *args, **kwargs)

    def axhspan(self, ymin, ymax, *args, **kwargs):
        # Width band → vertical band on the pitch; mirror both edges.
        if 0 <= float(ymin) <= 100 and 0 <= float(ymax) <= 100:
            x0 = (100.0 - float(ymax)) * VP_W / 100.0
            x1 = (100.0 - float(ymin)) * VP_W / 100.0
            return self._ax.axvspan(x0, x1, *args, **kwargs)
        return self._ax.axhspan(ymin, ymax, *args, **kwargs)

    def add_patch(self, patch):
        try:
            if (
                isinstance(patch, mpatches.Rectangle)
                and patch.get_transform() == self._ax.transData
            ):
                x, y = patch.get_xy()
                w = patch.get_width()
                h = patch.get_height()
                if _is_pitch_coord([x, x + w], [y, y + h]):
                    # Width axis (horizontal) mirrored: the band [y, y+h] maps
                    # to [100-(y+h), 100-y] so it lands on the same flank as the
                    # mirrored scatter overlays.
                    new_patch = mpatches.Rectangle(
                        ((100.0 - (y + h)) * VP_W / 100.0, x * VP_L / 100.0),
                        h * VP_W / 100.0,
                        w * VP_L / 100.0,
                        facecolor=patch.get_facecolor(),
                        edgecolor=patch.get_edgecolor(),
                        linewidth=patch.get_linewidth(),
                        alpha=patch.get_alpha(),
                        zorder=patch.get_zorder(),
                        hatch=patch.get_hatch(),
                        fill=patch.get_fill(),
                    )
                    return self._ax.add_patch(new_patch)
            if (
                isinstance(patch, mpatches.Circle)
                and patch.get_transform() == self._ax.transData
            ):
                x, y = patch.center
                if _is_pitch_coord(x, y):
                    nx, ny = _vp_point(x, y)
                    new_patch = mpatches.Ellipse(
                        (nx, ny),
                        width=2 * patch.radius * VP_W / 100.0,
                        height=2 * patch.radius * VP_L / 100.0,
                        facecolor=patch.get_facecolor(),
                        edgecolor=patch.get_edgecolor(),
                        linewidth=patch.get_linewidth(),
                        alpha=patch.get_alpha(),
                        zorder=patch.get_zorder(),
                        fill=patch.get_fill(),
                    )
                    return self._ax.add_patch(new_patch)
        except Exception:
            pass
        return self._ax.add_patch(patch)


# Legacy national-team palette data retained for compatibility helpers only.
# Production rendering uses the fixed first-team/second-team role colours.
# First colour in each list is the primary DISPLAY colour on dark charts.
NATIONAL_TEAM_COLOR_FALLBACKS = {
    "mexico": ["#006847", "#CE1126", "#FFFFFF"],
    "south africa": ["#FFB81C", "#007A4D", "#7DD3FC"],
    "egypt": ["#CE1126", "#FFFFFF", "#000000"],
    "argentina": ["#75AADB", "#FFFFFF", "#F6B40E"],
    "brazil": ["#FFDF00", "#009C3B", "#7DD3FC"],
    "france": ["#0055A4", "#FFFFFF", "#EF4135"],
    "england": ["#C8102E", "#FFFFFF", "#1D4ED8"],
    "spain": ["#AA151B", "#F1BF00", "#7DD3FC"],
    "germany": ["#DD0000", "#FFFFFF", "#000000"],
    "italy": ["#0066B3", "#FFFFFF", "#008C45"],
    "portugal": ["#006600", "#FF0000", "#FFCC00"],
    "netherlands": ["#F36C21", "#21468B", "#FFFFFF"],
    "belgium": ["#ED2939", "#FAE042", "#000000"],
    "morocco": ["#C1272D", "#006233", "#FFFFFF"],
    "united states": ["#3C3B6E", "#B22234", "#FFFFFF"],
    "usa": ["#3C3B6E", "#B22234", "#FFFFFF"],
    "canada": ["#FF0000", "#FFFFFF", "#111111"],
    "japan": ["#BC002D", "#FFFFFF", "#1D4ED8"],
    "saudi arabia": ["#006C35", "#FFFFFF", "#111111"],
    "qatar": ["#8A1538", "#FFFFFF", "#111111"],
    "ghana": ["#FCD116", "#CE1126", "#006B3F"],
    "senegal": ["#00853F", "#FDEF42", "#E31B23"],
    "uruguay": ["#6CABDD", "#FFFFFF", "#FCD116"],
    "colombia": ["#FCD116", "#7DD3FC", "#CE1126"],
    "croatia": ["#FF0000", "#FFFFFF", "#7DD3FC"],
    "switzerland": ["#D52B1E", "#FFFFFF", "#111111"],
    "australia": ["#FFCD00", "#00843D", "#7DD3FC"],
    "south korea": ["#C60C30", "#7DD3FC", "#FFFFFF"],
    "korea republic": ["#C60C30", "#7DD3FC", "#FFFFFF"],
    "ivory coast": ["#F77F00", "#009E60", "#FFFFFF"],
    "cote d'ivoire": ["#F77F00", "#009E60", "#FFFFFF"],
    "côte d'ivoire": ["#F77F00", "#009E60", "#FFFFFF"],
    "tunisia": ["#E70013", "#FFFFFF", "#111111"],
    "tun": ["#E70013", "#FFFFFF", "#111111"],
    "sweden": ["#FECB00", "#006AA7", "#7DD3FC"],
    "swe": ["#FECB00", "#006AA7", "#7DD3FC"],
    "algeria": ["#006233", "#FFFFFF", "#D21034"],
    "turkey": ["#E30A17", "#FFFFFF", "#111111"],
    "scotland": ["#005EB8", "#FFFFFF", "#111111"],
    "norway": ["#BA0C2F", "#7DD3FC", "#FFFFFF"],
    "dr congo": ["#007FFF", "#F7D618", "#CE1021"],
    "d r congo": ["#007FFF", "#F7D618", "#CE1021"],
    "congo dr": ["#007FFF", "#F7D618", "#CE1021"],
    "democratic republic of congo": ["#007FFF", "#F7D618", "#CE1021"],
    "democratic republic of the congo": ["#007FFF", "#F7D618", "#CE1021"],
    "congo": ["#009543", "#FBDE4A", "#DC241F"],
    "angola": ["#CC092F", "#F7D618", "#111111"],
    "cape verde": ["#7DD3FC", "#FFFFFF", "#CF2027"],
    "mali": ["#14B53A", "#FCD116", "#CE1126"],
    "burkina faso": ["#EF2B2D", "#009E49", "#FCD116"],
    "jamaica": ["#009B3A", "#FED100", "#111111"],
}


def _team_color_fallback(team_name: str, fallback: str) -> str:
    """Use national kit colour when v2 chart is rendered without injected match colours."""
    key = str(team_name or "").strip().lower()
    palette = NATIONAL_TEAM_COLOR_FALLBACKS.get(key)
    if not palette:
        # Loose matching for provider names / abbreviations.
        for k, vals in NATIONAL_TEAM_COLOR_FALLBACKS.items():
            if key and (key in k or k in key):
                palette = vals
                break
    if not palette:
        return fallback
    for col in palette:
        # Never use near-white or near-background colours as data marks.
        try:
            rgb = np.array(to_rgba(col)[:3])
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            bg = np.array(to_rgba(BG_DARK)[:3])
            dist = float(np.linalg.norm(rgb - bg))
            if 0.035 < lum < 0.86 and dist > 0.22:
                return _clean_dark_navy(col)
        except Exception:
            pass
    return _clean_dark_navy(fallback)


# Strong text-readability helpers for AMOLED pure-black visuals.
def _safe_ui_text(color: str, bg: str = BG_PANEL, *, min_ratio: float = 5.0) -> str:
    """Return white when a UI/accent/team colour would be too dark on the panel."""
    try:
        return readable_on(color, bg, min_ratio=min_ratio, fallback=TEXT_BR)
    except Exception:
        return TEXT_BR


def _safe_team_text(color: str, bg: str = BG_PANEL) -> str:
    """Use team colour in text only when legible; otherwise use white."""
    try:
        return readable_team_text(color, bg)
    except Exception:
        return TEXT_BR


def _strong_number_text(color: str | None = None, bg: str = BG_MID) -> str:
    """Numbers above bars/cards should never appear dark on AMOLED backgrounds."""
    if color:
        return _safe_ui_text(color, bg, min_ratio=5.2)
    return TEXT_BR


def _type_badge_style(raw_type: str, team_color: str):
    """Readable dark pill for goal type labels, matching the reference
    HTML's mono-coloured badge style (text/border share the accent colour,
    background is a faint tint of the same colour)."""
    text = str(raw_type or "OPEN PLAY").replace("_", " ").upper()
    text = re.sub(r"\s+", " ", text).strip()
    if text in {"OP"}:
        text = "OPEN PLAY"
    if text in {"PK"}:
        text = "PENALTY"
    if text in {"SP"}:
        text = "SET PIECE"

    if text.startswith("OPEN PLAY"):
        accent = C_GREEN
    elif "PENALTY" in text:
        accent = "#F97316"
    elif "SET PIECE" in text:
        accent = _safe_team_text(team_color)
    else:
        accent = _safe_team_text(team_color)

    return text, accent


def _display_player_name(name: str, max_len: int = 13) -> str:
    parts = [p for p in str(name or "").replace("_", " ").split() if p]
    if not parts:
        return "—"
    label = parts[-1]
    if len(label) <= 2 and len(parts) > 1:
        label = parts[-2]
    if len(label) > max_len:
        label = label[: max_len - 1] + "."
    return label.upper()


def _label_offset(x: float, y: float, idx: int) -> tuple[float, float, str, str]:
    """Place labels just outside nodes, biased away from pitch edges."""
    if x < 18:
        dx, ha = 3.6, "left"
    elif x > 82:
        dx, ha = -3.6, "right"
    else:
        dx = -2.8 if idx % 2 else 2.8
        ha = "right" if dx < 0 else "left"

    if y < 20:
        dy, va = 3.3, "bottom"
    elif y > 80:
        dy, va = -3.3, "top"
    else:
        dy = -3.0 if idx % 3 == 0 else 3.0
        va = "top" if dy < 0 else "bottom"
    return dx, dy, ha, va


def _label_candidates(
    x: float, y: float, idx: int
) -> list[tuple[float, float, str, str]]:
    primary = _label_offset(x, y, idx)
    candidates = [
        primary,
        (-3.2, 4.4, "right", "bottom"),
        (3.2, 4.4, "left", "bottom"),
        (-3.2, -4.4, "right", "top"),
        (3.2, -4.4, "left", "top"),
        (0.0, 5.8, "center", "bottom"),
        (0.0, -5.8, "center", "top"),
        (-6.2, 0.0, "right", "center"),
        (6.2, 0.0, "left", "center"),
    ]
    if x < 18:
        candidates.sort(key=lambda c: 0 if c[0] > 0 else 1)
    elif x > 82:
        candidates.sort(key=lambda c: 0 if c[0] < 0 else 1)
    if y < 14:
        candidates.sort(key=lambda c: 0 if c[1] > 0 else 1)
    elif y > 86:
        candidates.sort(key=lambda c: 0 if c[1] < 0 else 1)
    deduped = []
    seen = set()
    for c in candidates:
        key = (round(c[0], 1), round(c[1], 1), c[2], c[3])
        if key not in seen:
            deduped.append(c)
            seen.add(key)
    return deduped


def _rough_label_box(
    x: float, y: float, label: str, ha: str, va: str, fontsize: float
) -> tuple[float, float, float, float]:
    w = max(4.8, len(label) * fontsize * 0.19)
    h = max(2.8, fontsize * 0.46)
    if ha == "right":
        x0, x1 = x - w, x
    elif ha == "center":
        x0, x1 = x - w / 2, x + w / 2
    else:
        x0, x1 = x, x + w
    if va == "top":
        y0, y1 = y - h, y
    elif va == "center":
        y0, y1 = y - h / 2, y + h / 2
    else:
        y0, y1 = y, y + h
    return x0, y0, x1, y1


def _boxes_overlap(a, b, pad: float = 0.7) -> bool:
    return not (
        a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]
    )


def _draw_player_label(
    ax,
    p: dict,
    idx: int,
    *,
    fontsize: float = 6.0,
    zorder: int = 9,
    taken: list | None = None,
) -> None:
    taken = taken if taken is not None else []
    label = _display_player_name(p.get("name"))
    chosen = None
    for dx, dy, ha, va in _label_candidates(float(p["x"]), float(p["y"]), idx):
        lx, ly = float(p["x"]) + dx, float(p["y"]) + dy
        box = _rough_label_box(lx, ly, label, ha, va, fontsize)
        inside = box[0] >= 0 and box[2] <= 100 and box[1] >= 0 and box[3] <= 100
        if inside and not any(_boxes_overlap(box, old) for old in taken):
            chosen = (dx, dy, ha, va, box)
            break
    if chosen is None:
        dx, dy, ha, va = _label_offset(float(p["x"]), float(p["y"]), idx)
        lx, ly = float(p["x"]) + dx, float(p["y"]) + dy
        box = _rough_label_box(lx, ly, label, ha, va, fontsize)
    else:
        dx, dy, ha, va, box = chosen
        lx, ly = float(p["x"]) + dx, float(p["y"]) + dy
    taken.append(box)
    ax.text(
        lx,
        ly,
        label,
        ha=ha,
        va=va,
        color=TEXT_BR,
        fontsize=fontsize,
        fontweight="bold",
        bbox=dict(
            facecolor=BG_DARK, edgecolor="none", alpha=0.64, boxstyle="round,pad=0.12"
        ),
        path_effects=[pe.withStroke(linewidth=1.2, foreground=BG_DARK)],
        zorder=zorder,
        clip_on=True,
    )


def _wrap_axis_label(label: str, width: int = 12) -> str:
    return "\n".join(_tw.wrap(str(label), width=width, break_long_words=False)) or str(
        label
    )


def _blend_hex(c1: str, c2: str, amount: float = 0.5) -> str:
    a = np.array(to_rgba(c1)[:3], dtype=float)
    b = np.array(to_rgba(c2)[:3], dtype=float)
    rgb = np.clip(a * (1 - amount) + b * amount, 0, 1)
    return "#{:02x}{:02x}{:02x}".format(*(int(v * 255) for v in rgb))


def _compute_duels(events, team_id):
    """Per-team duel counts, Opta-style.

    Aerial duel: a `Aerial` event is logged for BOTH contesting players — the
    winner as Successful, the loser as Unsuccessful — so a team's aerials-won =
    its Successful `Aerial` rows and aerials-contested = all its `Aerial` rows.

    Ground duel: modelled as the dribble contests (`TakeOn`) so both teams share
    the SAME contested total, exactly like aerials. Each TakeOn is one on-ground
    50/50 between a dribbler and a defender: the dribbler's team wins it on a
    Successful TakeOn, the defending team wins it on an Unsuccessful one. Hence
    ground_total = every TakeOn in the match (same for both sides) and a team's
    ground_won = its own successful dribbles + the opponent's failed dribbles.
    Returns (aerial_won, aerial_total, ground_won, ground_total).
    """
    if events is None or events.empty or "type" not in events.columns:
        return 0, 0, 0, 0
    ty = events["type"].astype(str)
    has_out = "outcome" in events.columns
    out = events["outcome"].astype(str) if has_out else None
    tmask = events["team_id"] == team_id
    # Opponent = the other team that appears most in the feed.
    _others = [t for t in events["team_id"].dropna().unique() if t != team_id]
    opp_id = (
        max(_others, key=lambda t: int((events["team_id"] == t).sum()))
        if _others
        else None
    )
    omask = events["team_id"] == opp_id if opp_id is not None else (tmask & False)

    def _c(mask, type_name, success=None):
        m = mask & (ty == type_name)
        if success is not None and has_out:
            m = m & (out == ("Successful" if success else "Unsuccessful"))
        return int(m.sum())

    aerial_total = _c(tmask, "Aerial")
    aerial_won = _c(tmask, "Aerial", True) if has_out else 0

    to_self = _c(tmask, "TakeOn")
    to_opp = _c(omask, "TakeOn")
    ground_total = to_self + to_opp  # same for both sides
    if has_out:
        # own dribbles beaten + opponent dribbles stopped
        ground_won = _c(tmask, "TakeOn", True) + (to_opp - _c(omask, "TakeOn", True))
    else:
        ground_won = 0
    return aerial_won, aerial_total, ground_won, ground_total


# ═════════════════════════════════════════════════════════════════════════
#  1. xG FLOW v2
# ═════════════════════════════════════════════════════════════════════════
def _match_extra_time_pens(events, info):
    """Detect extra time and a penalty-shootout score from event periods.
    Local twin of match_report._extra_time_and_pens (kept separate to
    avoid a cross-module import) — same logic, used by chart-level visuals."""
    if events is None or events.empty or "period_code" not in events.columns:
        return False, None
    periods_seen = set(events["period_code"].dropna().astype(str).str.lower().unique())
    went_to_et = bool(periods_seen & {"et1", "etht", "et2"}) or any(
        "extratime" in pc or "extra time" in pc for pc in periods_seen
    )
    has_pso = ("pso" in periods_seen) or ("penaltyshootout" in periods_seen)
    if not has_pso or "is_penalty_shootout" not in events.columns:
        return went_to_et, None
    pso = events[events["is_penalty_shootout"].fillna(False)]
    if pso.empty:
        return went_to_et, None
    # A scored shootout kick is a row of type == "Goal" (WhoScored does not set
    # is_goal on shootout kicks). Fall back to is_goal only if type is missing.
    if "type" in pso.columns:
        scored = pso[pso["type"].astype(str).str.lower() == "goal"]
    else:
        scored = pso[pso.get("is_goal", False).fillna(False)]
    if scored.empty:
        return went_to_et, None
    hid, aid = info.get("home_id"), info.get("away_id")
    side_col = "team_id" if "team_id" in scored.columns else "scoring_team"
    h_pens = int((scored[side_col] == hid).sum())
    a_pens = int((scored[side_col] == aid).sum())
    return True, (h_pens, a_pens)


def render_xg_flow_v2(
    hn,
    an,
    score,
    hc,
    ac,
    shots_h,
    shots_a,
    went_to_et=False,
    pens=None,
    own_home=None,
    own_away=None,
):
    own_home = own_home or []
    own_away = own_away or []
    fig = plt.figure(figsize=(15, 9.5), facecolor=BG_DARK)
    chrome(
        fig,
        section="XG FLOW · MATCH ANALYSIS",
        title=f"{hn} vs {an} — xG Flow",
        subtitle="Cumulative Expected Goals minute by minute · "
        "stars mark goals · shaded territory = chance creation",
        hn=hn,
        an=an,
        score=score,
        footer_note="Step height = shot xG · steeper curve = better chances",
    )

    # Main chart panel — flat panel, hairline border only (no raised glow),
    # with a header strip matching panel_card()'s `.panel-head` styling.
    PX, PY, PW, PH = 0.05, 0.22, 0.62, 0.66
    header_h = 0.040
    header = mpatches.FancyBboxPatch(
        (PX, PY + PH - header_h),
        PW,
        header_h,
        boxstyle="round,pad=0.0,rounding_size=0.006",
        transform=fig.transFigure,
        facecolor=BG_HEADER,
        edgecolor=GRID_COL,
        linewidth=1.0,
        zorder=1,
    )
    fig.add_artist(header)
    dot_y = PY + PH - header_h / 2
    fig.add_artist(
        mpatches.Circle(
            (PX + 0.018, dot_y),
            0.0035,
            transform=fig.transFigure,
            facecolor=C_GOLD,
            edgecolor="none",
            zorder=2,
        )
    )
    fig.text(
        PX + 0.030,
        dot_y,
        "XG FLOW",
        ha="left",
        va="center",
        color=TEXT_BR,
        fontsize=10.5,
        fontweight="bold",
        family=FONT_MONO,
        zorder=2,
    )

    ax = fig.add_axes([PX, PY, PW, PH - header_h])
    ax.set_facecolor(BG_MID)
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL)
        s.set_linewidth(1.0)
        s.set_alpha(1.0)

    # A match that went to extra time has real shots past minute 90 — extend
    # the plotted duration (and every 90-only assumption below) to cover them,
    # instead of silently clipping/flattening the curve at the normal-time mark.
    # Duration comes ONLY from whether the match actually reached extra time
    # (from the period codes). A 90-minute game with a 94' stoppage shot is
    # still 90 minutes — don't stretch the axis to 120.
    max_shot_minute = max([s["minute"] for s in shots_h + shots_a], default=90)
    went_to_et = bool(went_to_et)
    duration = 120 if went_to_et else 90
    curve_end = (
        max(duration + 5, max_shot_minute + 3)
        if went_to_et
        else min(max(95, max_shot_minute + 2), 99)
    )

    def _cum(shots):
        ms = sorted(shots, key=lambda s: s["minute"])
        xs, ys = [0], [0]
        cum = 0
        for s in ms:
            xs += [s["minute"], s["minute"]]
            ys += [cum, cum + s["xG"]]
            cum += s["xG"]
        xs.append(curve_end)
        ys.append(cum)
        return xs, ys, cum

    hx, hy, h_total = _cum(shots_h)
    ax_, ay, a_total = _cum(shots_a)
    ax.fill_between(hx, 0, hy, color=hc, alpha=0.16, zorder=2, lw=0)
    ax.fill_between(ax_, 0, ay, color=ac, alpha=0.16, zorder=2, lw=0)
    ax.plot(hx, hy, color=hc, lw=2.6, solid_capstyle="round", zorder=3)
    ax.plot(ax_, ay, color=ac, lw=2.6, solid_capstyle="round", zorder=3)

    # Goal stars
    for shots, col in [(shots_h, hc), (shots_a, ac)]:
        ms = sorted(shots, key=lambda s: s["minute"])
        cum = 0
        for s in ms:
            cum += s["xG"]
            if s["is_goal"]:
                ax.scatter(
                    [s["minute"]],
                    [cum],
                    s=170,
                    marker="*",
                    color=C_GOLD,
                    edgecolor=BG_DARK,
                    lw=1.3,
                    zorder=5,
                )
                ax.annotate(
                    s["player"].split()[-1],
                    xy=(s["minute"], cum),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    color=TEXT_BR,
                    fontsize=8.5,
                    fontweight="bold",
                    family=FONT_SANS,
                    bbox=dict(
                        facecolor=BG_HEADER,
                        edgecolor=GRID_COL,
                        alpha=1.0,
                        boxstyle="round,pad=0.28",
                        linewidth=1.0,
                    ),
                )

    y_max = max(h_total, a_total, 0.5) * 1.15
    markers = [(45, "HT"), (90, "FT")]
    if went_to_et:
        aet_label = f"AET\n{pens[0]}-{pens[1]} PENS" if pens is not None else "AET"
        markers += [(105, "ET-HT"), (120, aet_label)]
    for xv, lb in markers:
        ax.axvline(xv, color=TEXT_FAD, lw=1.0, ls=(0, (1, 3)), alpha=0.7, zorder=1)
        ax.text(
            xv,
            y_max * 0.99,
            lb,
            ha="center",
            va="top",
            color=C_GOLD,
            fontsize=9,
            fontweight="bold",
            family=FONT_MONO,
            linespacing=1.4,
        )

    # Hottest 10-min window
    def _best_window(shots, w=10):
        if not shots:
            return None
        best = (0, 0, 0)
        for start in range(0, duration - w + 1):
            x = sum(s["xG"] for s in shots if start <= s["minute"] < start + w)
            if x > best[0]:
                best = (x, start, start + w)
        return best

    bw_h = _best_window(shots_h)
    bw_a = _best_window(shots_a)
    best = bw_h if (bw_h and (not bw_a or bw_h[0] > bw_a[0])) else bw_a
    momentum_team = hn if best is bw_h else an
    if best and best[0] > 0:
        band_y = y_max * 0.012
        ax.plot(
            [best[1], best[2]],
            [band_y, band_y],
            color=C_GOLD,
            lw=4.5,
            solid_capstyle="round",
            alpha=0.95,
            zorder=4,
        )
        ax.text(
            (best[1] + best[2]) / 2,
            y_max * 0.045,
            f"hottest 10-min · {momentum_team}",
            ha="center",
            color=C_GOLD,
            fontsize=9,
            fontweight="bold",
            family=FONT_MONO,
        )

    ax.set_xlim(0, curve_end)
    ax.set_ylim(0, y_max)
    ax.set_xticks(
        [0, 15, 30, 45, 60, 75, 90, 105, 120]
        if went_to_et
        else [0, 15, 30, 45, 60, 75, 90]
    )
    ax.tick_params(colors=TEXT_FAD, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(FONT_MONO)
    for sp in ["top", "right", "left", "bottom"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID_SOFT, lw=0.8, alpha=1.0)
    ax.set_xlabel(
        "MINUTE", color=TEXT_DIM, fontsize=9, fontweight="bold", family=FONT_MONO
    )
    ax.set_ylabel(
        "CUMULATIVE xG", color=TEXT_DIM, fontsize=9, fontweight="bold", family=FONT_MONO
    )
    # Inline legend chips
    ax.text(0.015, 0.965, "●", color=hc, fontsize=15, transform=ax.transAxes, va="top")
    ax.text(
        0.038,
        0.96,
        hn.upper(),
        color=hc,
        fontsize=10,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax.transAxes,
        va="top",
    )
    ax.text(0.015, 0.895, "●", color=ac, fontsize=15, transform=ax.transAxes, va="top")
    ax.text(
        0.038,
        0.89,
        an.upper(),
        color=ac,
        fontsize=10,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax.transAxes,
        va="top",
    )

    # Goals (by minute) panel. Each entry: (minute, scorer_label, xg_text,
    # team_col). Own goals are listed too (tagged OG, no xG) so both teams'
    # goals appear and the panel matches the scoreline even when a side only
    # scored via an own goal.
    goals = []
    for s in shots_h:
        if s["is_goal"]:
            sc = s["player"].split()[-1] if s["player"] else "—"
            goals.append((s["minute"], sc, f"{s['xG']:.2f}", hc))
    for s in shots_a:
        if s["is_goal"]:
            sc = s["player"].split()[-1] if s["player"] else "—"
            goals.append((s["minute"], sc, f"{s['xG']:.2f}", ac))
    for m, p in own_home:
        goals.append((m, (p.split()[-1] if p else "—") + " (OG)", "OG", hc))
    for m, p in own_away:
        goals.append((m, (p.split()[-1] if p else "—") + " (OG)", "OG", ac))
    goals.sort(key=lambda g: g[0])

    ax2 = panel_card(
        fig, 0.70, 0.54, 0.27, 0.34, title="Goals (by minute)", accent=C_GOLD
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.text(
        0.05,
        0.90,
        "MIN",
        color=TEXT_DIM,
        fontsize=8.5,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax2.transAxes,
        va="center",
    )
    ax2.text(
        0.20,
        0.90,
        "SCORER",
        color=TEXT_DIM,
        fontsize=8.5,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax2.transAxes,
        va="center",
    )
    ax2.text(
        0.93,
        0.90,
        "xG",
        ha="right",
        color=TEXT_DIM,
        fontsize=8.5,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax2.transAxes,
        va="center",
    )
    ax2.plot(
        [0.04, 0.96], [0.84, 0.84], color=GRID_COL, lw=1.0, transform=ax2.transAxes
    )
    if goals:
        n = max(len(goals), 1)
        rh = 0.74 / n
        for i, (gmin, scorer, xg_text, team_col) in enumerate(goals):
            cy = 0.78 - (i + 0.5) * rh
            ax2.plot(
                [0.04, 0.96],
                [cy - rh * 0.46, cy - rh * 0.46],
                color=GRID_SOFT,
                lw=0.8,
                transform=ax2.transAxes,
                zorder=1,
            )
            ax2.add_patch(
                mpatches.FancyBboxPatch(
                    (0.05, cy - 0.045),
                    0.075,
                    0.09,
                    boxstyle="round,pad=0.0,rounding_size=0.012",
                    facecolor=C_GOLD,
                    alpha=0.12,
                    lw=0,
                    transform=ax2.transAxes,
                    zorder=2,
                )
            )
            ax2.text(
                0.087,
                cy,
                f"{gmin}'",
                ha="center",
                va="center",
                color=C_GOLD,
                fontsize=10,
                fontweight="bold",
                family=FONT_MONO,
                transform=ax2.transAxes,
                zorder=3,
            )
            # A small team-colour dot ties each scorer to their side.
            ax2.scatter(
                [0.175],
                [cy],
                s=42,
                color=team_col,
                edgecolor=BG_DARK,
                lw=0.6,
                transform=ax2.transAxes,
                zorder=3,
            )
            ax2.text(
                0.21,
                cy,
                scorer,
                ha="left",
                va="center",
                color=TEXT_BR,
                fontsize=10.0,
                fontweight="bold",
                family=FONT_SANS,
                transform=ax2.transAxes,
                zorder=2,
            )
            ax2.text(
                0.93,
                cy,
                xg_text,
                ha="right",
                va="center",
                color=(TEXT_DIM if xg_text == "OG" else TEXT_BR),
                fontsize=10.5,
                fontweight="bold",
                family=FONT_MONO,
                transform=ax2.transAxes,
                zorder=2,
            )
    else:
        ax2.text(
            0.5,
            0.55,
            "No goals scored",
            ha="center",
            va="center",
            color=TEXT_FAD,
            fontsize=10,
            style="italic",
            family=FONT_SANS,
            transform=ax2.transAxes,
        )

    diff = h_total - a_total
    leader = hn if diff > 0 else an
    duration_txt = "120 minutes (AET)" if went_to_et else "90 minutes"
    if best and best[0] > 0:
        insight = (
            f"{leader} produced {abs(diff):.2f} more xG over the {duration_txt}. "
            f"The hottest 10-minute spell came from {momentum_team} "
            f"({best[1]:02d}'–{best[2]:02d}') with {best[0]:.2f} xG packed "
            f"into that window."
        )
    else:
        insight = (
            f"{leader} produced {abs(diff):.2f} more xG over the " f"{duration_txt}."
        )
    if pens is not None:
        insight += (
            f" The tie was ultimately settled on penalties, " f"{pens[0]}-{pens[1]}."
        )
    key_insight(fig, 0.70, 0.22, 0.27, 0.30, text=insight, wrap=34)

    # Goal counts include own goals credited to each side, so the cards match
    # the real scoreline (a team that only scored via an OG still shows 1).
    h_goals = sum(1 for s in shots_h if s["is_goal"]) + len(own_home)
    a_goals = sum(1 for s in shots_a if s["is_goal"]) + len(own_away)
    cards = [
        (f"{hn[:14]} xG", f"{h_total:.2f}", hc),
        (f"{hn[:14]} Goals", str(h_goals), hc),
        ("xG Diff", f"{'+' if diff >= 0 else ''}{diff:.2f}", TEXT_BR),
        (f"{an[:14]} Goals", str(a_goals), ac),
        (f"{an[:14]} xG", f"{a_total:.2f}", ac),
    ]
    metric_strip(fig, cards=cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  2. SHOT MAP v2
# ═════════════════════════════════════════════════════════════════════════
def render_shot_map_v2(team_name, opp_name, score, team_color, shots):
    fig = plt.figure(figsize=(15, 10), facecolor=BG_DARK)
    chrome(
        fig,
        section="SHOT MAP",
        title=f"{team_name} — Shot Map",
        subtitle="Each shot is an arrow to goal · colour = outcome "
        "(goal · saved · blocked · off) · width & label = xG · "
        "blue ring = penalty",
        hn=team_name,
        an=opp_name,
        score=score,
        footer_note="Direction of attack →",
    )

    team_color = _clean_dark_navy(team_color)

    PX, PY, PW, PH = 0.05, 0.22, 0.44, 0.64
    header_h, body_h = panel_header_geom(PH)
    header = mpatches.FancyBboxPatch(
        (PX, PY + PH - header_h),
        PW,
        header_h,
        boxstyle="round,pad=0.0,rounding_size=0.006",
        transform=fig.transFigure,
        facecolor=BG_HEADER,
        edgecolor=GRID_COL,
        linewidth=1.0,
        zorder=1,
    )
    fig.add_artist(header)
    dot_y = PY + PH - header_h / 2
    fig.add_artist(
        mpatches.Circle(
            (PX + 0.018, dot_y),
            0.0035,
            transform=fig.transFigure,
            facecolor=team_color,
            edgecolor="none",
            zorder=2,
        )
    )
    fig.text(
        PX + 0.030,
        dot_y,
        "SHOT MAP",
        ha="left",
        va="center",
        color=TEXT_BR,
        fontsize=10.5,
        fontweight="bold",
        family=FONT_MONO,
        zorder=2,
    )
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (PX, PY),
            PW,
            body_h,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor=BG_MID,
            edgecolor=GRID_COL,
            linewidth=1.0,
            zorder=-2,
        )
    )

    ax = fig.add_axes([PX + 0.03, PY + 0.02, PW - 0.06, body_h - 0.04])
    # Attacking half only — zooms the shot region so dots spread out instead
    # of piling up in the top third of a full-length pitch. Brighter, thicker
    # markings than the shared default so it reads as a real pitch (crisp
    # penalty box, 6-yard box, spot and D-arc) rather than a faint outline.
    _draw_vertical_pitch(ax, attacking_only=True, line_color="#6B7280", line_alpha=0.85)
    pax = _VerticalPitchProxy(ax)

    goals = [s for s in shots if s["is_goal"]]
    on_t = [s for s in shots if (not s["is_goal"]) and s["is_on_target"]]
    blocked = [
        s
        for s in shots
        if (not s["is_goal"]) and (not s["is_on_target"]) and s.get("is_blocked")
    ]
    off = [
        s
        for s in shots
        if (not s["is_goal"]) and (not s["is_on_target"]) and (not s.get("is_blocked"))
    ]
    GOAL_RING = "#FFC23C"
    SAVE_RING = "#3DDC84"
    BLOCK_COL = "#6B7280"
    OFF_RING = C_AWAY
    PEN_RING = "#38BDF8"

    # Penalty-area depth shading + average shot-distance line (under the shots).
    pax.axvspan(83.0, 100.0, color=team_color, alpha=0.05, zorder=1)
    if shots:
        avg_x = float(np.mean([s["x"] for s in shots]))
        pax.axvline(avg_x, color=TEXT_DIM, lw=0.9, ls=(0, (5, 4)), alpha=0.5, zorder=2)
        ax.text(
            VP_W - 1.0,
            avg_x * VP_L / 100.0 + 0.6,
            "AVG SHOT",
            ha="right",
            va="bottom",
            color=TEXT_DIM,
            fontsize=6.0,
            fontweight="bold",
            family=FONT_MONO,
            alpha=0.7,
            zorder=2,
        )

    def _ring(s):
        if s["is_goal"]:
            return GOAL_RING
        if s["is_on_target"]:
            return SAVE_RING
        if s.get("is_blocked"):
            return BLOCK_COL
        return OFF_RING

    # Each shot is an arrow from where it was taken toward the goal mouth.
    # Arrow colour encodes the outcome, width scales with xG, and every shot
    # carries its xG value at the origin so individual shots stay readable.
    def _shot_target(s):
        # Aim each shot at the goal line (x = 100) near its own lateral
        # position, so arrows point toward goal and fan into the goal mouth
        # instead of all piling onto a single point. Coords are (x=length,
        # y=width); the goal mouth sits at x=100, width ≈ 44–56.
        return 100.0, float(np.clip(s["y"], 42.0, 58.0))

    for s in sorted(shots, key=lambda d: d["xG"]):
        # Defense-in-depth: even if a caller other than _shots_for_team hands
        # us a shot with an out-of-pitch coordinate, never let it plot (or its
        # label export the whole figure's saved bounding box) outside the
        # 0-100 pitch — skip it instead of drawing a stray point + label.
        if not (0 <= s["x"] <= 100 and 0 <= s["y"] <= 100):
            continue
        ring = _ring(s)
        xgn = min(s["xG"] / 0.40, 1.0)
        tx, ty = _shot_target(s)
        _draw_vertical_arrow(
            ax,
            (s["x"], s["y"]),
            (tx, ty),
            color=ring,
            lw=1.3 + 3.0 * xgn,
            alpha=0.85,
            mutation_scale=8 + 7 * xgn,
            rad=0.04,
            zorder=5,
        )
        if s.get("is_penalty"):
            pax.scatter(
                [s["x"]],
                [s["y"]],
                s=180,
                marker="o",
                facecolor="none",
                edgecolor=PEN_RING,
                linewidth=1.3,
                alpha=0.9,
                zorder=6,
            )
        pax.scatter(
            [s["x"]],
            [s["y"]],
            s=46 + s["xG"] * 240,
            marker="o",
            facecolor=ring,
            edgecolor=BG_DARK,
            linewidth=0.8,
            alpha=0.95,
            zorder=6,
        )
        pax.text(
            s["x"] - 1.3,
            s["y"] - 1.4,
            f"{s['xG']:.2f}",
            ha="right",
            va="top",
            color=ring,
            fontsize=6.0,
            fontweight="bold",
            family=FONT_MONO,
            zorder=7,
            clip_on=True,
        )

    if goals:
        gsorted = sorted(goals, key=lambda s: s["y"])
        n_g = len(gsorted)
        anchor_x = VP_W + 1.4
        y_top, y_bot = 102, 58
        for i, s in enumerate(gsorted):
            label_y = y_top - i * (y_top - y_bot) / max(n_g - 1, 1)
            surname = s["player"].split()[-1] if s["player"] else "—"
            ax.annotate(
                f"{surname} {s['minute']}'",
                xy=_vp_point(s["x"], s["y"]),
                xytext=(anchor_x, label_y),
                ha="left",
                va="center",
                color=C_GOLD,
                fontsize=8,
                fontweight="bold",
                family=FONT_SANS,
                arrowprops=dict(
                    arrowstyle="-",
                    color=GRID_COL,
                    lw=0.9,
                    alpha=0.75,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=7,
            )

    _draw_vertical_attack_arrow(ax, x=-1.3, y0=58.0, y1=70.0)

    chips = [
        ("Goal", GOAL_RING, "→"),
        ("Saved", SAVE_RING, "→"),
        ("Blocked", BLOCK_COL, "→"),
        ("Off", OFF_RING, "→"),
        ("Penalty", PEN_RING, "◌"),
    ]
    cx = PX + 0.010
    for lbl, col, mk in chips:
        fig.text(
            cx,
            PY - 0.018,
            mk,
            ha="left",
            va="center",
            color=col,
            fontsize=10.5,
            family=FONT_SANS,
        )
        fig.text(
            cx + 0.013,
            PY - 0.018,
            lbl.upper(),
            ha="left",
            va="center",
            color=TEXT_DIM,
            fontsize=7.3,
            fontweight="bold",
            family=FONT_MONO,
        )
        cx += 0.013 + 0.0058 * len(lbl) + 0.016

    # Top scorers
    by_player = {}
    for s in shots:
        by_player.setdefault(s["player"], {"xG": 0, "shots": 0, "goals": 0})
        by_player[s["player"]]["xG"] += s["xG"]
        by_player[s["player"]]["shots"] += 1
        by_player[s["player"]]["goals"] += int(s["is_goal"])
    top = sorted(by_player.items(), key=lambda kv: -kv[1]["xG"])[:5]

    ax2 = panel_card(
        fig, 0.50, 0.50, 0.46, 0.38, title="Top Shot Sources (by xG)", accent=team_color
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.text(
        0.04,
        0.90,
        "PLAYER",
        color=TEXT_DIM,
        fontsize=8.7,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax2.transAxes,
        va="center",
    )
    ax2.text(
        0.62,
        0.90,
        "SH",
        ha="center",
        color=TEXT_DIM,
        fontsize=8.7,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax2.transAxes,
        va="center",
    )
    ax2.text(
        0.78,
        0.90,
        "G",
        ha="center",
        color=TEXT_DIM,
        fontsize=8.7,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax2.transAxes,
        va="center",
    )
    ax2.text(
        0.95,
        0.90,
        "xG",
        ha="right",
        color=TEXT_DIM,
        fontsize=8.7,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax2.transAxes,
        va="center",
    )
    ax2.plot(
        [0.03, 0.97], [0.84, 0.84], color=GRID_COL, lw=1.0, transform=ax2.transAxes
    )
    if top:
        n = len(top)
        rh = 0.74 / n
        for i, (player, d) in enumerate(top):
            cy = 0.78 - (i + 0.5) * rh
            if i > 0:
                ax2.plot(
                    [0.03, 0.97],
                    [cy + rh / 2, cy + rh / 2],
                    color=GRID_SOFT,
                    lw=0.8,
                    transform=ax2.transAxes,
                    zorder=1,
                )
            label = (player or "—").split()[-1] if player else "—"
            ax2.text(
                0.04,
                cy,
                label,
                ha="left",
                va="center",
                color=TEXT_BR,
                fontsize=10.5,
                fontweight="bold",
                family=FONT_SANS,
                transform=ax2.transAxes,
                zorder=2,
            )
            ax2.text(
                0.62,
                cy,
                str(d["shots"]),
                ha="center",
                va="center",
                color=TEXT_DIM,
                fontsize=10,
                family=FONT_MONO,
                transform=ax2.transAxes,
                zorder=2,
            )
            ax2.text(
                0.78,
                cy,
                str(d["goals"]),
                ha="center",
                va="center",
                color=C_GOLD if d["goals"] else TEXT_DIM,
                fontsize=10,
                fontweight="bold",
                family=FONT_MONO,
                transform=ax2.transAxes,
                zorder=2,
            )
            ax2.text(
                0.95,
                cy,
                f"{d['xG']:.2f}",
                ha="right",
                va="center",
                color=team_color,
                fontsize=10.5,
                fontweight="bold",
                family=FONT_MONO,
                transform=ax2.transAxes,
                zorder=2,
            )
    else:
        ax2.text(
            0.5,
            0.4,
            "No shots recorded",
            ha="center",
            va="center",
            color=TEXT_FAD,
            fontsize=10,
            style="italic",
            family=FONT_SANS,
            transform=ax2.transAxes,
        )

    n_shots = len(shots)
    n_goals = len(goals)
    n_ot = len(on_t) + n_goals
    total_xg = sum(s["xG"] for s in shots)
    big_chances = sum(1 for s in shots if s["xG"] >= 0.30)
    insight = (
        f"{team_name} took {n_shots} shots worth {total_xg:.2f} xG. "
        f"{n_ot}/{n_shots} forced the keeper into action. "
        f"{big_chances} big chance(s); finishing was "
        f"{'over' if n_goals > total_xg else 'under'}-performing with "
        f"{n_goals} goals."
    )
    key_insight(fig, 0.50, 0.20, 0.46, 0.26, text=insight, wrap=58)

    cards = [
        ("Total xG", f"{total_xg:.2f}", team_color),
        ("Goals", str(n_goals), C_GOLD),
        ("Shots", str(n_shots), TEXT_BR),
        ("On-Target %", f"{round(100*n_ot/n_shots) if n_shots else 0}%", team_color),
        ("Big Chances", str(big_chances), C_GOLD),
        ("xG / Shot", f"{(total_xg/n_shots) if n_shots else 0:.2f}", TEXT_BR),
    ]
    metric_strip(fig, cards=cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  3. SHOT BREAKDOWN v2
# ═════════════════════════════════════════════════════════════════════════
def render_shot_breakdown_v2(hn, an, score, home, away, goals, hc=None, ac=None):
    hc = hc or C_HOME
    ac = ac or C_AWAY
    fig = plt.figure(figsize=(16, 10), facecolor=BG_DARK)
    chrome(
        fig,
        section="SHOT BREAKDOWN",
        title=f"{hn} vs {an} — Shot Breakdown & Goals",
        subtitle="Volume · placement · finishing — and how every goal "
        "actually arrived",
        hn=hn,
        an=an,
        score=score,
        footer_note="Bars = shot volume · table = every goal scored",
    )

    ax1 = panel_card(
        fig,
        0.04,
        0.49,
        0.56,
        0.32,
        title="Shot Volume by Outcome",
        accent=C_GOLD,
        body=False,
    )
    # Legend chips inside the panel body (top-left), matching the reference
    # HTML's `.legend-row` placed just below the panel header.
    ax1.text(
        0.012, 0.965, "●", color=hc, fontsize=13, transform=ax1.transAxes, va="top"
    )
    ax1.text(
        0.034,
        0.965,
        hn.upper(),
        color=hc,
        fontsize=9.5,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax1.transAxes,
        va="top",
    )
    ax1.text(
        0.012, 0.895, "●", color=ac, fontsize=13, transform=ax1.transAxes, va="top"
    )
    ax1.text(
        0.034,
        0.895,
        an.upper(),
        color=ac,
        fontsize=9.5,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax1.transAxes,
        va="top",
    )

    keys = ["shots", "on_target", "off_target", "blocked", "post"]
    labels = ["Total Shots", "On Target", "Off Target", "Blocked", "Woodwork"]
    n = len(keys)
    pos = np.arange(n)
    w = 0.34
    h_vals = [home.get(k, 0) for k in keys]
    a_vals = [away.get(k, 0) for k in keys]
    ax1.set_xlim(-0.6, n - 0.4)
    bar_ymax = max(h_vals + a_vals + [1]) * 2.3
    ax1.set_ylim(-bar_ymax * 0.10, bar_ymax)
    for y in np.arange(0, bar_ymax, 5):
        ax1.axhline(y, color=GRID_SOFT, lw=0.8, alpha=1.0, zorder=0)
    ax1.bar(pos - w / 2, h_vals, w, color=hc, lw=0, zorder=2)
    ax1.bar(pos + w / 2, a_vals, w, color=ac, lw=0, zorder=2)
    for i, (hv, av) in enumerate(zip(h_vals, a_vals)):
        ax1.text(
            i - w / 2,
            hv + bar_ymax * 0.018,
            str(hv),
            ha="center",
            va="bottom",
            color=TEXT_BR,
            fontsize=10.5,
            fontweight="bold",
            family=FONT_MONO,
        )
        ax1.text(
            i + w / 2,
            av + bar_ymax * 0.018,
            str(av),
            ha="center",
            va="bottom",
            color=TEXT_BR,
            fontsize=10.5,
            fontweight="bold",
            family=FONT_MONO,
        )
    ax1.set_xticks(pos)
    ax1.set_xticklabels(
        labels, color=TEXT_DIM, fontsize=9.5, family=FONT_SANS, fontweight="bold"
    )
    ax1.tick_params(axis="x", length=0, pad=10)
    ax1.set_yticks([])
    ax1.axhline(0, color=GRID_COL, lw=1.0, alpha=1.0, zorder=1)
    for sp in ["top", "right", "left", "bottom"]:
        ax1.spines[sp].set_visible(False)

    diff_xg = home.get("xG", 0) - away.get("xG", 0)
    leader = hn if diff_xg > 0 else an
    insight = (
        f"{leader} produced the stronger chance profile "
        f"(xG {'+' if diff_xg >= 0 else ''}{diff_xg:.2f}). "
        f"{home.get('on_target', 0)} of {hn}'s {home.get('shots', 0)} shots "
        f"forced saves vs. only {away.get('on_target', 0)} from "
        f"{an}'s {away.get('shots', 0)}."
    )
    key_insight(fig, 0.62, 0.49, 0.34, 0.32, text=insight)

    ax2 = panel_card(
        fig, 0.04, 0.16, 0.92, 0.27, title="Goals & Assists", accent=C_GOLD
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    cols = [
        ("MIN", 0.04),
        ("SCORER", 0.14),
        ("TEAM", 0.34),
        ("TYPE", 0.49),
        ("ASSIST", 0.64),
        ("xG", 0.92),
    ]
    for c, x in cols:
        ax2.text(
            x,
            0.87,
            c,
            ha="left" if c != "xG" else "right",
            va="center",
            color=TEXT_DIM,
            fontsize=9,
            fontweight="bold",
            family=FONT_MONO,
            transform=ax2.transAxes,
        )
    ax2.plot(
        [0.03, 0.97], [0.81, 0.81], color=GRID_COL, lw=1.0, transform=ax2.transAxes
    )
    if goals:
        n_g = len(goals)
        row_h = 0.68 / n_g
        for i, g in enumerate(goals):
            mn, scorer, gtype, assist, xg, side = g
            cy = 0.74 - (i + 0.5) * row_h
            team_col = hc if side == "home" else ac
            team_nm = hn if side == "home" else an
            if i > 0:
                ax2.plot(
                    [0.03, 0.97],
                    [cy + row_h / 2, cy + row_h / 2],
                    color=GRID_SOFT,
                    lw=0.8,
                    transform=ax2.transAxes,
                    zorder=1,
                )
            # Minute chip, matching the reference `.row .min` styling.
            ax2.add_patch(
                mpatches.FancyBboxPatch(
                    (0.035, cy - 0.038),
                    0.06,
                    0.076,
                    boxstyle="round,pad=0.0,rounding_size=0.010",
                    facecolor=C_GOLD,
                    alpha=0.10,
                    lw=0,
                    transform=ax2.transAxes,
                    zorder=2,
                )
            )
            ax2.text(
                0.065,
                cy,
                mn,
                ha="center",
                va="center",
                color=C_GOLD,
                fontsize=10.5,
                fontweight="bold",
                family=FONT_MONO,
                transform=ax2.transAxes,
                zorder=3,
            )
            ax2.text(
                0.14,
                cy,
                scorer,
                ha="left",
                va="center",
                color=TEXT_BR,
                fontsize=11,
                fontweight="bold",
                family=FONT_SANS,
                transform=ax2.transAxes,
                zorder=2,
            )
            ax2.text(
                0.34,
                cy,
                team_nm,
                ha="left",
                va="center",
                color=team_col,
                fontsize=10,
                fontweight="bold",
                family=FONT_MONO,
                transform=ax2.transAxes,
                zorder=2,
            )
            type_text, accent = _type_badge_style(gtype, team_col)
            ax2.text(
                0.49,
                cy,
                type_text[:24],
                ha="left",
                va="center",
                color=accent,
                fontsize=8.4,
                fontweight="bold",
                family=FONT_MONO,
                transform=ax2.transAxes,
                zorder=3,
                bbox=dict(
                    boxstyle="round,pad=0.32",
                    facecolor=to_rgba(accent, 0.08),
                    edgecolor=to_rgba(accent, 0.35),
                    linewidth=1.0,
                ),
            )
            ax2.text(
                0.64,
                cy,
                assist,
                ha="left",
                va="center",
                color=TEXT_DIM,
                fontsize=10,
                family=FONT_SANS,
                transform=ax2.transAxes,
                zorder=2,
            )
            ax2.text(
                0.92,
                cy,
                f"{xg:.2f}",
                ha="right",
                va="center",
                color=TEXT_BR,
                fontsize=11,
                fontweight="bold",
                family=FONT_MONO,
                transform=ax2.transAxes,
                zorder=2,
            )
    else:
        ax2.text(
            0.5,
            0.45,
            "No goals scored",
            ha="center",
            va="center",
            color=TEXT_FAD,
            fontsize=11,
            style="italic",
            family=FONT_SANS,
            transform=ax2.transAxes,
        )

    home_g = sum(1 for g in goals if g[5] == "home")
    away_g = sum(1 for g in goals if g[5] == "away")
    cards = [
        ("Final Score", f"{score}", TEXT_BR),
        (f"{hn[:14]} xG", f"{home.get('xG', 0):.2f}", hc),
        ("xG Diff", f"{'+' if diff_xg >= 0 else ''}{diff_xg:.2f}", TEXT_BR),
        (f"{an[:14]} xG", f"{away.get('xG', 0):.2f}", ac),
        ("Goals — H/A", f"{home_g} / {away_g}", TEXT_BR),
    ]
    metric_strip(fig, cards=cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  4. PASS NETWORK v2
# ═════════════════════════════════════════════════════════════════════════
PASS_ROLE_COLORS = {
    "gk": "#FFC23C",  # goalkeeper — gold
    "def": C_HOME,  # defender — canonical ultraviolet
    "mid": "#3DDC84",  # midfielder — green
    "att": C_AWAY,  # forward — canonical chartreuse
}
# Substituted players get one distinct fill colour, kept separate from the
# four positional units so the eye reads "this player came on/off" instantly.
PASS_SUB_COLOR = "#A78BFA"  # violet


def _detect_depth_axis(players: list) -> str:
    """Determine which axis ('x' or 'y') represents pitch depth (own goal →
    opponent goal) for a given group of players, by looking at which axis
    has the larger spread across the squad. A football team spreads more
    along the longitudinal axis (defender-to-forward, ~80+ units of range)
    than along the lateral axis (touchline-to-touchline, ~60-70 units),
    so the axis with the bigger range is the depth axis. This makes the
    role-bucket classifier robust regardless of whether the data source
    stores depth in `x` or in `y` — different providers, and even
    different match feeds from the same provider, can swap them, and the
    spread test is reliable across both conventions."""
    if not players:
        return "x"
    xs = [p.get("x", 50) for p in players if p.get("x") is not None]
    ys = [p.get("y", 50) for p in players if p.get("y") is not None]
    if len(xs) < 2 or len(ys) < 2:
        return "x"
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    # If both ranges are comparable (within ~10%), prefer x as a tie-breaker
    # since it's the documented convention for most providers.
    if x_range >= y_range * 0.9:
        return "x"
    return "y"


def _detect_attack_direction(players: list, depth_axis: str) -> int:
    """Return +1 if higher depth values mean 'closer to opponent's goal'
    (the conventional 0→100 layout where the keeper sits at the low end),
    or -1 if the squad is laid out the other way (keeper at the high end).
    We pick whichever extreme is occupied by a single outlier — almost
    always the goalkeeper — as the 'own goal' end."""
    if not players:
        return 1
    vals = sorted(p.get(depth_axis, 50) for p in players)
    if len(vals) < 4:
        return 1
    low_gap = vals[1] - vals[0]
    high_gap = vals[-1] - vals[-2]
    # The keeper sits noticeably further from the next-deepest teammate
    # than any outfield gap. If that outlier gap is at the LOW end, the
    # keeper is at low depth, which is the conventional layout: depth
    # grows from own goal (0) to opp goal (100), direction = +1. If the
    # gap is at the HIGH end instead, the layout is flipped and we need
    # to invert via direction = -1 so the bucket thresholds still apply.
    if low_gap >= high_gap:
        return 1
    return -1


def _infer_position_bucket(p: dict, depth_axis: str = "x", direction: int = 1) -> str:
    """Infer a broad positional bucket (gk/def/mid/att) for colouring nodes
    by the player's depth (0 = own goal line, 100 = opponent's goal line).

    `depth_axis` and `direction` are produced by `_detect_depth_axis` and
    `_detect_attack_direction` for the whole squad once, then passed in.
    This decouples the classifier from any assumption about which raw key
    holds depth or which direction the team is attacking — the spread and
    keeper-gap tests recover both from the data itself, so the same logic
    works whether the feed uses x-as-depth or y-as-depth, and whether the
    side is laid out 0→opp-goal or 100→opp-goal.

    An explicit position string on the player dict always wins over the
    geometric heuristic."""
    explicit = str(p.get("position") or p.get("pos") or "").strip().upper()
    if explicit and explicit != "SUB":
        if explicit in {"GK", "GOALKEEPER"}:
            return "gk"
        # Midfield codes must be tested before the generic "D" defender prefix,
        # otherwise "DMC"/"DML"/"DMR" (defensive midfielders) wrongly match "D".
        if explicit.startswith(("DM", "CM", "AM", "M")):
            return "mid"
        if explicit.startswith(("WB", "D", "CB", "LB", "RB")):
            return "def"
        if explicit.startswith(("F", "ST", "CF", "LW", "RW", "W")):
            return "att"
    raw = p.get(depth_axis, 50)
    # Normalise so that 'depth' always means 0 = own goal, 100 = opp goal,
    # regardless of which way the raw data is oriented.
    depth = raw if direction >= 0 else (100 - raw)
    if depth <= 8:
        return "gk"
    if depth <= 38:
        return "def"
    if depth <= 68:
        return "mid"
    return "att"


def _role_color(role, team_color):
    """Node fill colour. The reference identity colours every node by
    positional bucket (gk/def/mid/att) rather than by team, so substitution
    states (sub_in/sub_out/etc.) are now drawn as a ring accent instead of
    overriding the fill — see _draw_pass_network_half."""
    team_color = _clean_dark_navy(team_color)
    return PASS_ROLE_COLORS.get(role or "", team_color)


def _substitution_ring_color(role):
    """Optional outer-ring accent for substitution states, drawn alongside
    the positional fill colour rather than replacing it."""
    return {
        "sub_in": "#22C55E",
        "sub_out": "#F59E0B",
        "both_sub": "#A78BFA",
        "red_card": "#F87171",
    }.get(role or "", None)


def _role_badge(role):
    return {
        "sub_in": "↑",
        "sub_out": "↓",
        "both_sub": "↕",
        "red_card": "RC",
    }.get(role or "", "")


def render_pass_network_v2(team_name, opp_name, score, team_color, players, edges):
    fig = plt.figure(figsize=(15, 10), facecolor=BG_DARK)
    chrome(
        fig,
        section="PASS NETWORK",
        title=f"{team_name} — Pass Network",
        subtitle="All passing links shown · colour and width reveal connection strength",
        hn=team_name,
        an=opp_name,
        score=score,
        footer_note="Direction of attack →",
    )

    team_color = _clean_dark_navy(team_color)

    PX, PY, PW, PH = 0.06, 0.16, 0.40, 0.72
    header_h, body_h = panel_header_geom(PH)
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (PX, PY + PH - header_h),
            PW,
            header_h,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor=BG_HEADER,
            edgecolor=GRID_COL,
            linewidth=1.0,
            zorder=1,
        )
    )
    dot_y = PY + PH - header_h / 2
    fig.add_artist(
        mpatches.Circle(
            (PX + 0.018, dot_y),
            0.0035,
            transform=fig.transFigure,
            facecolor=team_color,
            edgecolor="none",
            zorder=2,
        )
    )
    fig.text(
        PX + 0.030,
        dot_y,
        "PASS NETWORK",
        ha="left",
        va="center",
        color=TEXT_BR,
        fontsize=10.5,
        fontweight="bold",
        family=FONT_MONO,
        zorder=2,
    )
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (PX, PY),
            PW,
            body_h,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor=BG_MID,
            edgecolor=GRID_COL,
            linewidth=1.0,
            zorder=-2,
        )
    )

    ax = fig.add_axes([PX + 0.045, PY + 0.020, PW - 0.09, body_h - 0.045])
    _draw_vertical_pitch(ax, line_alpha=0.50)
    pax = _VerticalPitchProxy(ax)
    by_name = {p["name"]: p for p in players}

    max_e = max((e["count"] for e in edges), default=1)
    counts = [e["count"] for e in edges]
    medium_cut = np.percentile(counts, 50) if counts else 0
    strong_cut = np.percentile(counts, 78) if counts else 0
    low_col, mid_col, strong_col = network_link_palette(team_color)
    drawable_edges = [
        e
        for e in sorted(edges, key=lambda d: d["count"])
        if e["from"] in by_name and e["to"] in by_name
    ]

    drawn_edges = []
    for e in drawable_edges:
        p1 = by_name[e["from"]]
        p2 = by_name[e["to"]]
        ratio = e["count"] / max_e
        if e["count"] >= strong_cut:
            line_col = strong_col
            lw = 1.25 + 3.80 * ratio
            alpha = 0.55 + 0.35 * ratio
            z = 4
        elif e["count"] >= medium_cut:
            line_col = mid_col
            lw = 0.80 + 2.30 * ratio
            alpha = 0.30 + 0.28 * ratio
            z = 3
        else:
            line_col = low_col
            lw = 0.34 + 1.15 * ratio
            alpha = 0.10 + 0.14 * ratio
            z = 2
        pax.plot(
            [p1["x"], p2["x"]],
            [p1["y"], p2["y"]],
            color=line_col,
            lw=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=z,
        )
        drawn_edges.append((p1, p2, e["count"], lw))

    top_for_labels = sorted(drawn_edges, key=lambda t: -t[2])[:4]
    for p1, p2, count, _lw in top_for_labels:
        mx, my = (p1["x"] + p2["x"]) / 2, (p1["y"] + p2["y"]) / 2
        pax.text(
            mx,
            my,
            str(count),
            ha="center",
            va="center",
            color=TEXT_BR,
            fontsize=7.2,
            fontweight="bold",
            family=FONT_MONO,
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor=BG_DARK,
                edgecolor=GRID_COL,
                lw=0.8,
                alpha=0.92,
            ),
            zorder=4,
        )

    max_p = max((p["passes"] for p in players), default=1)
    players_sorted = sorted(players, key=lambda d: d.get("passes", 0), reverse=True)
    _depth_axis = _detect_depth_axis(players)
    _direction = _detect_attack_direction(players, _depth_axis)
    for rank, p in enumerate(players_sorted):
        size = 300 + 1200 * (p["passes"] / max_p)
        bucket = _infer_position_bucket(p, _depth_axis, _direction)
        node_color = _role_color(bucket, team_color)
        sub_ring = _substitution_ring_color(p.get("role"))
        badge = _role_badge(p.get("role"))
        ring_color = sub_ring or node_color
        pax.scatter(
            [p["x"]], [p["y"]], s=size + 260, color=BG_DARK, alpha=0.92, zorder=5
        )
        pax.scatter(
            [p["x"]],
            [p["y"]],
            s=size + 110,
            color=ring_color,
            alpha=0.40 if sub_ring else 0.28,
            zorder=5,
        )
        pax.scatter(
            [p["x"]],
            [p["y"]],
            s=size,
            color=node_color,
            edgecolor=TEXT_BR,
            lw=1.35,
            alpha=0.98,
            zorder=6,
        )
        short = (p["name"] or "").split()[-1][:9]
        if badge:
            short = f"{short} {badge}"
        label_offset = -7.5 if p["y"] > 54 else 7.5
        va = "top" if label_offset < 0 else "bottom"
        name_fs = 8.1 if rank < 8 else 7.4
        pax.text(
            p["x"],
            min(100, max(0, p["y"] + label_offset)),
            short,
            ha="center",
            va=va,
            color=TEXT_BR,
            fontsize=name_fs,
            fontweight="bold",
            family=FONT_SANS,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor=BG_DARK,
                edgecolor="none",
                alpha=0.88,
            ),
            zorder=7,
        )
        pax.text(
            p["x"],
            p["y"],
            str(p["passes"]),
            ha="center",
            va="center",
            color=BG_DARK,
            fontsize=7.4,
            fontweight="bold",
            family=FONT_MONO,
            zorder=8,
        )

    _draw_vertical_attack_arrow(ax, x=-1.3, y0=3.0, y1=16.0)

    ax2 = panel_card(
        fig,
        0.50,
        0.50,
        0.46,
        0.38,
        title="Top Partnerships (passes)",
        accent=team_color,
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.text(
        0.04,
        0.90,
        "PAIR",
        color=TEXT_DIM,
        fontsize=8.7,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax2.transAxes,
        va="center",
    )
    ax2.text(
        0.95,
        0.90,
        "PASSES",
        ha="right",
        color=TEXT_DIM,
        fontsize=8.7,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax2.transAxes,
        va="center",
    )
    ax2.plot(
        [0.03, 0.97], [0.84, 0.84], color=GRID_COL, lw=1.0, transform=ax2.transAxes
    )
    top_edges = sorted(edges, key=lambda e: -e["count"])[:8]
    if top_edges:
        n = len(top_edges)
        rh = 0.74 / n
        for i, e in enumerate(top_edges):
            cy = 0.78 - (i + 0.5) * rh
            if i > 0:
                ax2.plot(
                    [0.03, 0.97],
                    [cy + rh / 2, cy + rh / 2],
                    color=GRID_SOFT,
                    lw=0.8,
                    transform=ax2.transAxes,
                    zorder=1,
                )
            from_short = (e["from"] or "").split()[-1]
            to_short = (e["to"] or "").split()[-1]
            pair = f"{from_short} ↔ {to_short}"
            ax2.text(
                0.04,
                cy,
                pair[:23],
                ha="left",
                va="center",
                color=TEXT_BR,
                fontsize=10.5,
                fontweight="bold",
                family=FONT_SANS,
                transform=ax2.transAxes,
                zorder=2,
            )
            ax2.text(
                0.95,
                cy,
                str(e["count"]),
                ha="right",
                va="center",
                color=team_color,
                fontsize=11,
                fontweight="bold",
                family=FONT_MONO,
                transform=ax2.transAxes,
                zorder=2,
            )
    else:
        ax2.text(
            0.5,
            0.4,
            "No links recorded",
            ha="center",
            va="center",
            color=TEXT_FAD,
            fontsize=10,
            style="italic",
            family=FONT_SANS,
            transform=ax2.transAxes,
        )

    total_passes = sum(p["passes"] for p in players)
    if players:
        top_player = max(players, key=lambda p: p["passes"])
        most_active = (top_player["name"] or "").split()[-1]
        avg_x = int(np.mean([p["x"] for p in players]))
        insight = (
            f"{team_name}'s network keeps every player link on the pitch, "
            f"using colour and line weight to separate low, medium and strong "
            f"connections. {most_active} was the busiest hub with "
            f"{top_player['passes']} passes; the centre of gravity sat at x≈{avg_x}."
        )
    else:
        insight = f"{team_name} pass network — insufficient pass data."
    key_insight(fig, 0.50, 0.16, 0.46, 0.30, text=insight, wrap=58)

    if players:
        avg_x = int(np.mean([p["x"] for p in players]))
        y_spread = int(max(p["y"] for p in players) - min(p["y"] for p in players))
    else:
        avg_x, y_spread = 0, 0
    cards = [
        ("Total Passes", str(total_passes), TEXT_BR),
        ("Players", str(len(players)), team_color),
        ("Top Partner.", str(top_edges[0]["count"]) if top_edges else "0", TEXT_BR),
        ("Links Shown", f"{len(drawable_edges)}/{len(edges)}", team_color),
        ("Y Spread", f"{y_spread}", TEXT_BR),
    ]
    metric_strip(fig, cards=cards)
    return fig


def _pass_half_accent(team_color: str, half: int) -> str:
    return _clean_dark_navy(team_color)


PASS_PITCH_W = 54.0
PASS_PITCH_L = 100.0


def _pass_vxy(x: float, y: float) -> tuple[float, float]:
    # y=0 is the right flank and y=100 is the left flank in this feed, so the
    # width axis must be mirrored (100 - y) — otherwise right-sided players
    # (e.g. a right-back or right winger) are drawn on the left of the
    # attack-up pitch instead of the right.
    return (100.0 - float(y)) * (PASS_PITCH_W / 100.0), float(x)


def _pass_vplayer(p: dict, depth_axis: str = "x", direction: int = 1) -> dict:
    # Infer the positional bucket from the ORIGINAL (pre-rotation) raw
    # coordinates, using the auto-detected depth axis and attack direction
    # for this specific team so the classification is correct even when
    # the underlying feed swaps x and y or orients the side the other way.
    # Doing this before _pass_vxy swaps the axes is critical — after
    # rotation, neither p["x"] nor p["y"] still means "depth", so the
    # bucket must be computed first.
    bucket = _infer_position_bucket(p, depth_axis, direction)
    orig_x = p.get("x", 50)
    vx, vy = _pass_vxy(p.get("x", 50), p.get("y", 50))
    out = dict(p)
    out["x"] = vx
    out["y"] = vy
    out["_bucket"] = bucket
    out["_orig_x"] = orig_x
    return out


def _themed_pass_pitch_vertical(ax, *, line_alpha: float = 0.52):
    ax.set_facecolor(BG_PITCH)
    ax.set_aspect("equal")
    ax.set_xlim(-2, PASS_PITCH_W + 4)
    ax.set_ylim(-2, PASS_PITCH_L + 2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL)
        s.set_linewidth(1.0)
        s.set_alpha(1.0)

    line_grey = "#3A3A3A"
    lc = dict(color=line_grey, lw=1.00, alpha=line_alpha * 0.86, zorder=2)
    w, l = PASS_PITCH_W, PASS_PITCH_L
    ax.plot([0, w, w, 0, 0], [0, 0, l, l, 0], **lc)
    ax.plot([0, w], [50, 50], **lc)
    ax.add_patch(plt.Circle((w / 2, 50), 9.15, fill=False, **lc))
    ax.scatter([w / 2], [50], color=line_grey, s=8, alpha=line_alpha, zorder=2)

    pa_w = 0.593 * w
    ga_w = 0.269 * w
    for y0, sign in [(0, 1), (l, -1)]:
        ax.plot(
            [(w - pa_w) / 2, (w - pa_w) / 2, (w + pa_w) / 2, (w + pa_w) / 2],
            [y0, y0 + sign * 16.5, y0 + sign * 16.5, y0],
            **lc,
        )
        ax.plot(
            [(w - ga_w) / 2, (w - ga_w) / 2, (w + ga_w) / 2, (w + ga_w) / 2],
            [y0, y0 + sign * 5.5, y0 + sign * 5.5, y0],
            **lc,
        )
        ax.scatter(
            [w / 2], [y0 + sign * 11], color=line_grey, s=6, alpha=line_alpha, zorder=2
        )


def _draw_player_label_vertical(
    ax,
    p: dict,
    idx: int,
    *,
    fontsize: float,
    zorder: int,
    taken: list | None = None,
    all_nodes: list | None = None,
    pitch_w: float = PASS_PITCH_W,
    pitch_l: float = PASS_PITCH_L,
    node_radius: float = 3.4,
    node_radii: dict | None = None,
) -> None:
    taken = taken if taken is not None else []
    all_nodes = all_nodes if all_nodes is not None else []
    node_radii = node_radii if node_radii is not None else {}
    label = _display_player_name(p.get("name"))
    # Wider base offsets so labels clear neighbouring node circles in dense
    # formations (e.g. crowded attacking lines in the 2nd half). The first
    # few candidates hug the node closely; later ones push further out for
    # very dense clusters where the close-in spots are all taken.
    own_r = node_radii.get(id(p), node_radius)
    candidates = [
        (own_r + 0.5, own_r + 0.8, "left", "bottom"),
        (-(own_r + 0.5), own_r + 0.8, "right", "bottom"),
        (own_r + 0.5, -(own_r + 0.8), "left", "top"),
        (-(own_r + 0.5), -(own_r + 0.8), "right", "top"),
        (0.0, own_r + 3.2, "center", "bottom"),
        (0.0, -(own_r + 3.2), "center", "top"),
        (own_r + 3.2, 0.0, "left", "center"),
        (-(own_r + 3.2), 0.0, "right", "center"),
        (own_r + 1.9, own_r + 2.3, "left", "bottom"),
        (-(own_r + 1.9), own_r + 2.3, "right", "bottom"),
        (own_r + 1.9, -(own_r + 2.3), "left", "top"),
        (-(own_r + 1.9), -(own_r + 2.3), "right", "top"),
        (0.0, own_r + 5.8, "center", "bottom"),
        (0.0, -(own_r + 5.8), "center", "top"),
    ]
    if p["x"] > pitch_w - 10:
        candidates.sort(key=lambda c: 0 if c[0] < 0 else 1)
    elif p["x"] < 10:
        candidates.sort(key=lambda c: 0 if c[0] > 0 else 1)
    if p["y"] > pitch_l - 10:
        candidates.sort(key=lambda c: 0 if c[1] < 0 else 1)
    elif p["y"] < 10:
        candidates.sort(key=lambda c: 0 if c[1] > 0 else 1)

    # Treat every other player's node position as a small "no-go" box so
    # labels never get drawn across a neighbouring circle. Each neighbour
    # uses its own actual rendered radius rather than a fixed constant, so
    # bigger (higher-pass-count) nodes correctly claim more space.
    node_boxes = [
        (
            n["x"] - node_radii.get(id(n), node_radius),
            n["y"] - node_radii.get(id(n), node_radius),
            n["x"] + node_radii.get(id(n), node_radius),
            n["y"] + node_radii.get(id(n), node_radius),
        )
        for n in all_nodes
        if n is not p
    ]

    chosen = None
    for dx, dy, ha, va in candidates:
        lx, ly = p["x"] + dx, p["y"] + dy
        box = _rough_label_box(lx, ly, label, ha, va, fontsize)
        inside = (
            box[0] >= -1
            and box[2] <= pitch_w + 1
            and box[1] >= -1
            and box[3] <= pitch_l + 1
        )
        clashes_label = any(_boxes_overlap(box, old, pad=0.4) for old in taken)
        clashes_node = any(_boxes_overlap(box, nb, pad=0.6) for nb in node_boxes)
        if inside and not clashes_label and not clashes_node:
            chosen = (lx, ly, ha, va, box)
            break
    if chosen is None:
        # Fall back to the candidate with the least overlap rather than a
        # fixed index, to degrade gracefully in very dense clusters. Ties on
        # overlap count break toward the candidate closest to the node, so
        # the label still reads as "belonging" to its player visually.
        def _overlap_score(cand):
            dx, dy, ha, va = cand
            lx, ly = p["x"] + dx, p["y"] + dy
            box = _rough_label_box(lx, ly, label, ha, va, fontsize)
            score = sum(1 for old in taken if _boxes_overlap(box, old))
            score += sum(1 for nb in node_boxes if _boxes_overlap(box, nb, pad=0.3))
            dist = (dx * dx + dy * dy) ** 0.5
            return (score, dist)

        dx, dy, ha, va = min(candidates, key=_overlap_score)
        lx, ly = p["x"] + dx, p["y"] + dy
        box = _rough_label_box(lx, ly, label, ha, va, fontsize)
    else:
        lx, ly, ha, va, box = chosen
    taken.append(box)
    # Leader line: when a label had to be pushed away from its node to avoid
    # an overlap, draw a faint connector so it stays unambiguous which player
    # the name belongs to.
    if ((lx - p["x"]) ** 2 + (ly - p["y"]) ** 2) ** 0.5 > own_r + 2.4:
        ax.plot(
            [p["x"], lx],
            [p["y"], ly],
            color="#6A6A6A",
            lw=0.4,
            alpha=0.5,
            zorder=zorder - 1,
            solid_capstyle="round",
        )
    ax.text(
        lx,
        ly,
        label,
        ha=ha,
        va=va,
        color=TEXT_BR,
        fontsize=fontsize,
        fontweight="bold",
        family=FONT_SANS,
        bbox=dict(
            facecolor=BG_DARK,
            edgecolor="#363D49",
            lw=0.7,
            alpha=1.0,
            boxstyle="round,pad=0.18",
        ),
        path_effects=[pe.withStroke(linewidth=1.6, foreground=BG_DARK)],
        zorder=zorder,
        clip_on=True,
    )


def _draw_pass_label_above(
    ax,
    p: dict,
    *,
    fontsize: float,
    zorder: int,
    taken: list,
    node_radii: dict | None = None,
    all_nodes: list | None = None,
    node_radius: float = 3.4,
    pitch_w: float = PASS_PITCH_W,
    pitch_l: float = PASS_PITCH_L,
) -> None:
    """Pass-network-only label placement: name sits fixed directly above its
    node (number already lives inside the node). Collisions are resolved by
    stacking the label further up, with a small sideways nudge tried at each
    stack level, so names stay visually locked to "their" player even in a
    dense 15-man half where a sub's node sits almost directly above the
    starter it replaced."""
    node_radii = node_radii if node_radii is not None else {}
    all_nodes = all_nodes if all_nodes is not None else []
    own_r = node_radii.get(id(p), node_radius)
    label = _display_player_name(p.get("name"))
    gap = own_r + 1.0
    node_boxes = [
        (
            n["x"] - node_radii.get(id(n), node_radius),
            n["y"] - node_radii.get(id(n), node_radius),
            n["x"] + node_radii.get(id(n), node_radius),
            n["y"] + node_radii.get(id(n), node_radius),
        )
        for n in all_nodes
        if n is not p
    ]
    chosen = None
    for step in range(7):
        base_ly = p["y"] + gap + step * (fontsize * 0.22 + 1.0)
        for dx, ha in (
            (0.0, "center"),
            (own_r * 1.2, "left"),
            (-own_r * 1.2, "right"),
            (own_r * 2.4, "left"),
            (-own_r * 2.4, "right"),
        ):
            lx = p["x"] + dx
            box = _rough_label_box(lx, base_ly, label, ha, "bottom", fontsize)
            clashes = any(_boxes_overlap(box, old, pad=0.3) for old in taken) or any(
                _boxes_overlap(box, nb, pad=0.3) for nb in node_boxes
            )
            if not clashes:
                chosen = (lx, base_ly, ha, box)
                break
        if chosen:
            break
    if chosen is None:
        # Nothing fully clear — settle on the highest stacked, centered spot
        # rather than looping forever; a rare, tightly-packed cluster.
        ly = p["y"] + gap + 6 * (fontsize * 0.22 + 1.0)
        lx = p["x"]
        ha = "center"
        box = _rough_label_box(lx, ly, label, ha, "bottom", fontsize)
        chosen = (lx, ly, ha, box)
    lx, ly, ha, box = chosen
    # Keep the label from drifting past the pitch's top edge in tall stacks.
    if box[3] > pitch_l + 1:
        overflow = box[3] - (pitch_l + 1)
        ly -= overflow
        box = _rough_label_box(lx, ly, label, ha, "bottom", fontsize)
    taken.append(box)
    if ((lx - p["x"]) ** 2 + (ly - p["y"]) ** 2) ** 0.5 > own_r + 2.0:
        ax.plot(
            [p["x"], lx],
            [p["y"] + own_r, ly],
            color="#6A6A6A",
            lw=0.4,
            alpha=0.5,
            zorder=zorder - 1,
            solid_capstyle="round",
        )
    ax.text(
        lx,
        ly,
        label,
        ha=ha,
        va="bottom",
        color=TEXT_BR,
        fontsize=fontsize,
        fontweight="bold",
        family=FONT_SANS,
        bbox=dict(
            facecolor=BG_DARK,
            edgecolor="#363D49",
            lw=0.7,
            alpha=1.0,
            boxstyle="round,pad=0.18",
        ),
        path_effects=[pe.withStroke(linewidth=1.6, foreground=BG_DARK)],
        zorder=zorder,
        clip_on=True,
    )


def _convex_hull(points):
    """Andrew's monotone-chain convex hull (no SciPy dependency).
    `points` = list of (x, y). Returns the hull vertices in order, or the
    input unchanged if there are fewer than 3 points."""
    pts = sorted(set((round(x, 4), round(y, 4)) for x, y in points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _draw_pass_network_half(ax, title, players, edges, accent):
    accent = _clean_dark_navy(accent)
    _themed_pass_pitch_vertical(ax, line_alpha=0.50)
    ax.text(
        0.02,
        0.985,
        title.upper(),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=accent,
        fontsize=10.5,
        fontweight="bold",
        family=FONT_MONO,
        zorder=20,
    )
    for idx, p in enumerate(
        sorted(players, key=lambda d: d.get("passes", 0), reverse=True), start=1
    ):
        p["display_id"] = idx
    _depth_axis = _detect_depth_axis(players)
    _direction = _detect_attack_direction(players, _depth_axis)
    v_players = [_pass_vplayer(p, _depth_axis, _direction) for p in players]
    by_name = {p["name"]: p for p in v_players}

    # Node sizing (computed early, before edges/repulsion, so both use the
    # final radii). Dense halves (subs on) get smaller markers so 14-15
    # nodes don't turn the pitch into a wall of overlapping circles.
    max_p = max((p["passes"] for p in v_players), default=1)
    BASE_SIZE = 110 + 430 * 0.5  # size at a "typical" half-max passer
    BASE_RADIUS = 3.0
    if len(v_players) >= 15:
        SIZE_SCALE = 0.60
    elif len(v_players) >= 13:
        SIZE_SCALE = 0.72
    else:
        SIZE_SCALE = 1.0
    sizes_by_id = {}
    for p in v_players:
        sizes_by_id[id(p)] = SIZE_SCALE * (110 + 430 * (p["passes"] / max_p))
    node_radii = {
        pid: BASE_RADIUS * (sz / BASE_SIZE) ** 0.5 for pid, sz in sizes_by_id.items()
    }

    # Declutter: when two players' true average positions sit almost on top
    # of each other (a sub warming up right next to the player they replaced,
    # e.g. Gordon by Saka), the markers/shirt-numbers themselves overlap — no
    # amount of label routing fixes that. Nudge nodes apart just enough for
    # markers to separate; edges and labels below read off these same nudged
    # coordinates so the whole map stays internally consistent. A thin
    # connector line is drawn later for any node that actually moved.
    orig_xy = {id(p): (p["x"], p["y"]) for p in v_players}
    for _ in range(40):
        moved = False
        for a in v_players:
            for b in v_players:
                if a is b:
                    continue
                dx = b["x"] - a["x"]
                dy = b["y"] - a["y"]
                dist = (dx * dx + dy * dy) ** 0.5
                min_dist = (node_radii[id(a)] + node_radii[id(b)]) * 0.95
                if dist < min_dist:
                    moved = True
                    if dist < 0.05:
                        ang = (hash((id(a), id(b))) % 360) * np.pi / 180.0
                        dx, dy, dist = np.cos(ang), np.sin(ang), 1.0
                    push = (min_dist - dist) / 2.0
                    ux, uy = dx / dist, dy / dist
                    a["x"] -= ux * push
                    a["y"] -= uy * push
                    b["x"] += ux * push
                    b["y"] += uy * push
        if not moved:
            break
    for p in v_players:
        p["x"] = float(np.clip(p["x"], 1, PASS_PITCH_W - 1))
        p["y"] = float(np.clip(p["y"], 1, PASS_PITCH_L - 1))

    max_e = max((e["count"] for e in edges), default=1)
    counts = [e["count"] for e in edges]
    medium_cut = np.percentile(counts, 50) if counts else 0
    strong_cut = np.percentile(counts, 78) if counts else 0
    # The relationship palette is computed independently of the node fill.
    # Grey/white teams automatically switch to indigo links, so a connection
    # can never share the same colour as a player circle.
    low_col, mid_col, strong_col = network_link_palette(accent)
    n_players = len(v_players)
    drawable_edges = [
        e
        for e in sorted(edges, key=lambda d: d["count"])
        if e["from"] in by_name and e["to"] in by_name
    ]
    # Drop ghost links (a single stray pass between a pair). On a dense half
    # (subs on, ~13+ players) raise the floor so the map stays readable; on a
    # very sparse half keep count==1 so the network doesn't vanish.
    if n_players >= 13:
        _min_draw = 3
    elif sum(1 for e in drawable_edges if e["count"] >= 2) >= 8:
        _min_draw = 2
    else:
        _min_draw = 1
    drawable_edges = [e for e in drawable_edges if e["count"] >= _min_draw]

    drawn_edges = []
    for i, e in enumerate(drawable_edges):
        p1 = by_name[e["from"]]
        p2 = by_name[e["to"]]
        ratio = e["count"] / max_e
        is_strong = False
        if e["count"] >= strong_cut:
            line_col = strong_col
            lw = 1.05 + 3.10 * ratio
            alpha = 0.55 + 0.32 * ratio
            glow_alpha = 0.08
            z = 4
            is_strong = True
        elif e["count"] >= medium_cut:
            line_col = mid_col
            lw = 0.60 + 1.45 * ratio
            alpha = 0.22 + 0.20 * ratio
            glow_alpha = 0.0
            z = 3
        else:
            # On a dense half (13+ players) the weakest tier is pure noise —
            # skip drawing it entirely instead of a faint line, matching the
            # "LOW" legend chip which simply won't appear if nothing uses it.
            if n_players >= 13:
                continue
            line_col = low_col
            lw = 0.28 + 0.50 * ratio
            alpha = 0.06 + 0.07 * ratio
            glow_alpha = 0.0
            z = 2
        if glow_alpha:
            ax.plot(
                [p1["x"], p2["x"]],
                [p1["y"], p2["y"]],
                color=line_col,
                lw=lw + 4.0,
                alpha=glow_alpha,
                solid_capstyle="round",
                zorder=z,
            )
        ax.plot(
            [p1["x"], p2["x"]],
            [p1["y"], p2["y"]],
            color=line_col,
            lw=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=z,
        )
        drawn_edges.append((p1, p2, e["count"]))

    # Auto-shrink label fontsize as the squad on this half gets denser, so a
    # 15-man second half (with subs on) doesn't suffer the same overlap rate
    # as an 11-man first half. (node sizes/radii were already computed above,
    # before the repulsion pass, so edges/nodes/labels all agree.)
    if n_players <= 12:
        label_fontsize = 6.8
    elif n_players <= 14:
        label_fontsize = 6.4
    else:
        label_fontsize = 6.0

    # A substitute = a player who started on the bench (not in the XI). Guard
    # against a feed that mislabels everyone: if every node looks like a sub,
    # the starter flag is unreliable, so draw them all as circles.
    _all_sub = v_players and all(pp.get("is_sub") for pp in v_players)

    # ── Team shape (convex hull) + average defensive line ──────────────
    # Drawn under the edges/nodes to give the "block" a readable footprint,
    # matching the StatsBomb-style team-shape overlay. Built from on-pitch
    # outfield starters so a wandering sub doesn't distort the shape.
    core = [p for p in v_players if not (p.get("is_sub") and not _all_sub)]
    if len(core) >= 3:
        hull = _convex_hull([(p["x"], p["y"]) for p in core])
        if len(hull) >= 3:
            ax.add_patch(
                mpatches.Polygon(
                    hull,
                    closed=True,
                    facecolor=accent,
                    edgecolor=mid_col,
                    alpha=0.06,
                    lw=1.0,
                    joinstyle="round",
                    zorder=1,
                )
            )
            ax.add_patch(
                mpatches.Polygon(
                    hull,
                    closed=True,
                    facecolor="none",
                    edgecolor=mid_col,
                    alpha=0.28,
                    lw=1.0,
                    joinstyle="round",
                    zorder=1,
                )
            )
    # Average defensive-line height: skip the keeper (deepest node), then take
    # the next few deepest as the back line and draw their mean height.
    by_depth = sorted(core, key=lambda p: p["y"])
    if len(by_depth) >= 5:
        back_line = by_depth[1:5]
        dline = float(np.mean([p["y"] for p in back_line]))
        ax.plot(
            [2, PASS_PITCH_W - 2],
            [dline, dline],
            ls=(0, (6, 4)),
            color=strong_col,
            lw=1.0,
            alpha=0.45,
            zorder=1,
        )
        ax.text(
            PASS_PITCH_W - 2,
            dline + 0.6,
            "DEF LINE",
            ha="right",
            va="bottom",
            color=strong_col,
            fontsize=5.6,
            fontweight="bold",
            family=FONT_MONO,
            alpha=0.7,
            zorder=1,
        )

    # Identify the goalkeeper (positional GK, else the deepest node) so the
    # node can carry a distinct gold edge.
    gk_node = None
    for p in v_players:
        if (
            _infer_position_bucket(
                {"x": p.get("_orig_x", p["y"]), "position": p.get("position")}
            )
            == "gk"
        ):
            gk_node = p
            break
    if gk_node is None and by_depth:
        gk_node = by_depth[0]

    taken_labels = []
    for rank, p in enumerate(
        sorted(v_players, key=lambda d: d.get("passes", 0), reverse=True)
    ):
        size = sizes_by_id[id(p)]
        role = p.get("role") or ""
        # Every node uses the single team colour. Bench players (not in the
        # starting XI) are set apart by SHAPE (square) instead of colour.
        is_sub = bool(p.get("is_sub")) and not _all_sub
        is_gk = gk_node is not None and p is gk_node
        marker = "s" if is_sub else "o"
        node_color = accent
        # A sent-off player keeps the team fill but gains a red outer ring.
        rc_ring = "#F87171" if role == "red_card" else None
        is_hub = rank == 0
        # If this node got nudged apart by the repulsion pass, draw a thin
        # connector back to its true average position so it stays traceable.
        ox, oy = orig_xy[id(p)]
        if (p["x"] - ox) ** 2 + (p["y"] - oy) ** 2 > 0.7**2:
            ax.plot(
                [ox, p["x"]],
                [oy, p["y"]],
                color="#6A6A6A",
                lw=0.5,
                alpha=0.55,
                zorder=3,
                solid_capstyle="round",
            )
        # Thin dark separator so overlapping nodes still read as distinct,
        # kept tight for an airier look.
        ax.scatter(
            [p["x"]],
            [p["y"]],
            s=size + 80 * SIZE_SCALE,
            color=BG_DARK,
            marker=marker,
            alpha=0.90,
            zorder=5,
        )
        if is_hub:
            # Gold halo marks the team's busiest passing hub at a glance.
            ax.scatter(
                [p["x"]],
                [p["y"]],
                s=size + 210 * SIZE_SCALE,
                facecolor="none",
                marker=marker,
                edgecolor=C_GOLD,
                lw=1.4,
                alpha=0.75,
                zorder=5,
            )
        # Outer ring only where it carries meaning (sent-off / substitute);
        # ordinary starters get no extra halo for a cleaner look.
        if rc_ring:
            ax.scatter(
                [p["x"]],
                [p["y"]],
                s=size + 95 * SIZE_SCALE,
                color=rc_ring,
                marker=marker,
                alpha=0.50,
                zorder=5,
            )
        elif is_sub:
            ax.scatter(
                [p["x"]],
                [p["y"]],
                s=size + 95 * SIZE_SCALE,
                color="#FFFFFF",
                marker=marker,
                alpha=0.45,
                zorder=5,
            )
        # Node fill is slightly translucent so dense clusters feel lighter;
        # GK → gold edge, sub → white edge, outfield starter → soft team-tint edge.
        edge_c = (
            C_GOLD
            if is_gk
            else (TEXT_BR if is_sub else _blend_hex(accent, "#ffffff", 0.45))
        )
        edge_w = 2.2 if is_gk else (1.5 if is_sub else 1.1)
        ax.scatter(
            [p["x"]],
            [p["y"]],
            s=size,
            color=node_color,
            marker=marker,
            edgecolor=edge_c,
            lw=edge_w,
            alpha=0.90,
            zorder=6,
        )
        # Number inside the node: shirt number when known, else passing rank.
        _shirt = p.get("shirt")
        if _shirt is not None and str(_shirt).strip() not in {"", "None", "nan"}:
            _sv = str(_shirt)
            node_num = (
                str(int(float(_sv))) if _sv.replace(".", "", 1).isdigit() else _sv
            )
        else:
            node_num = str(p.get("display_id", rank + 1))
        ax.text(
            p["x"],
            p["y"],
            node_num,
            ha="center",
            va="center",
            color="#ffffff",
            fontsize=6.8 * (0.88 if SIZE_SCALE < 1.0 else 1.0),
            fontweight="bold",
            family=FONT_MONO,
            zorder=8,
            path_effects=[pe.withStroke(linewidth=1.6, foreground=BG_DARK)],
        )
        # Every participant keeps a direct name label. The collision-aware
        # stacker moves dense labels and draws a leader line when necessary.
        label_cap = n_players
        if rank < label_cap:
            _draw_pass_label_above(
                ax,
                p,
                fontsize=label_fontsize,
                zorder=9,
                taken=taken_labels,
                node_radii=node_radii,
                all_nodes=v_players,
            )

    arrow_x = PASS_PITCH_W + 0.4
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (arrow_x - 1.2, 75.5),
            2.6,
            22.0,
            boxstyle="round,pad=0.0,rounding_size=1.0",
            facecolor=BG_DARK,
            edgecolor="none",
            alpha=0.55,
            zorder=18,
        )
    )
    ax.annotate(
        "",
        xy=(arrow_x, 95),
        xytext=(arrow_x, 79),
        arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.3, alpha=0.95),
        zorder=19,
        annotation_clip=False,
    )
    ax.text(
        arrow_x,
        76.5,
        "ATTACK",
        ha="center",
        va="top",
        color=C_GOLD,
        fontsize=6.3,
        fontweight="bold",
        family=FONT_MONO,
        alpha=0.95,
        zorder=19,
        clip_on=False,
        rotation=90,
    )
    _legend_x = 0.02
    if n_players < 13:
        # "LOW" only appears in the legend when the tier is actually drawn.
        ax.text(
            _legend_x,
            0.035,
            "LOW",
            transform=ax.transAxes,
            color=low_col,
            fontsize=7.0,
            fontweight="bold",
            family=FONT_MONO,
            ha="left",
            va="bottom",
            alpha=0.85,
        )
        _legend_x = 0.10
    ax.text(
        _legend_x,
        0.035,
        "MEDIUM",
        transform=ax.transAxes,
        color=mid_col,
        fontsize=7.0,
        fontweight="bold",
        family=FONT_MONO,
        ha="left",
        va="bottom",
        alpha=0.9,
    )
    ax.text(
        _legend_x + 0.135,
        0.035,
        "STRONG",
        transform=ax.transAxes,
        color=strong_col,
        fontsize=7.0,
        fontweight="bold",
        family=FONT_MONO,
        ha="left",
        va="bottom",
        alpha=0.95,
    )
    return len(drawable_edges)


def render_pass_network_halves_v2(
    team_name, opp_name, score, team_color, first_half, second_half
):
    team_color = _clean_dark_navy(team_color)
    fig = plt.figure(figsize=(16, 10), facecolor=BG_DARK)
    chrome(
        fig,
        section="PASS NETWORK",
        title=f"{team_name} — Pass Network by Half",
        subtitle="Two-map view: first half and second half shown separately for cleaner structure",
        hn=team_name,
        an=opp_name,
        score=score,
        footer_note="Direction of attack →",
    )

    # Legend: nodes use the single team colour; a square marks substitutes.
    rx = 0.030
    fig.add_artist(
        mpatches.Circle(
            (rx + 0.006, 0.866),
            0.0050,
            transform=fig.transFigure,
            facecolor=team_color,
            edgecolor=TEXT_BR,
            linewidth=0.8,
            zorder=5,
        )
    )
    fig.text(
        rx + 0.018,
        0.866,
        f"{team_name.upper()} (STARTER)",
        ha="left",
        va="center",
        color=TEXT_BR,
        fontsize=8.7,
        fontweight="bold",
        family=FONT_MONO,
        zorder=5,
    )
    rx += 0.018 + 0.0072 * len(team_name + " (STARTER)") + 0.030
    sq = 0.0092
    fig.add_artist(
        mpatches.Rectangle(
            (rx, 0.866 - sq / 2),
            sq,
            sq,
            transform=fig.transFigure,
            facecolor=team_color,
            edgecolor=TEXT_BR,
            linewidth=0.8,
            zorder=5,
        )
    )
    fig.text(
        rx + 0.016,
        0.866,
        "SUBSTITUTE",
        ha="left",
        va="center",
        color=TEXT_BR,
        fontsize=8.7,
        fontweight="bold",
        family=FONT_MONO,
        zorder=5,
    )

    h1_col = _pass_half_accent(team_color, 1)
    h2_col = _pass_half_accent(team_color, 2)

    # Card frames behind each pitch, matching `.pitch-panel` (hairline
    # border + dark header strip carrying the "1ST HALF" / "2ND HALF" tag).
    panel_y, panel_h = 0.155, 0.685
    panel_specs = [(0.04, 0.46, "1st Half", h1_col), (0.52, 0.46, "2nd Half", h2_col)]
    for px, pw, ptitle, pcol in panel_specs:
        _panel_rect(fig, px, panel_y, pw, panel_h, zorder=-2)
        header_h = 0.040
        header = mpatches.FancyBboxPatch(
            (px, panel_y + panel_h - header_h),
            pw,
            header_h,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor=BG_HEADER,
            edgecolor=GRID_COL,
            linewidth=1.0,
            zorder=-1,
        )
        fig.add_artist(header)
        dot_y = panel_y + panel_h - header_h / 2
        fig.text(
            px + 0.018,
            dot_y,
            ptitle.upper(),
            ha="left",
            va="center",
            color=pcol,
            fontsize=10,
            fontweight="bold",
            family=FONT_MONO,
            zorder=1,
        )
        fig.text(
            px + pw - 0.018,
            dot_y,
            "ATTACK →",
            ha="right",
            va="center",
            color=TEXT_FAD,
            fontsize=8.5,
            fontweight="bold",
            family=FONT_MONO,
            zorder=1,
        )

    ax1 = fig.add_axes([0.145, 0.205, 0.255, 0.595])
    ax2 = fig.add_axes([0.600, 0.205, 0.255, 0.595])
    h1_shown = _draw_pass_network_half(
        ax1, "", first_half["players"], first_half["edges"], h1_col
    )
    h2_shown = _draw_pass_network_half(
        ax2, "", second_half["players"], second_half["edges"], h2_col
    )

    def _top_pair(edges):
        if not edges:
            return "0"
        return str(max(e["count"] for e in edges))

    h1_passes = sum(p["passes"] for p in first_half["players"])
    h2_passes = sum(p["passes"] for p in second_half["players"])
    h1_players = len(first_half["players"])
    h2_players = len(second_half["players"])
    fig.text(
        0.105,
        0.172,
        f"1H · {h1_passes} passes · {h1_players} players · "
        f"{h1_shown}/{len(first_half['edges'])} links · number = shirt number",
        color=TEXT_DIM,
        fontsize=9,
        fontweight="bold",
        family=FONT_MONO,
    )
    fig.text(
        0.560,
        0.172,
        f"2H · {h2_passes} passes · {h2_players} players · "
        f"{h2_shown}/{len(second_half['edges'])} links · number = shirt number",
        color=TEXT_DIM,
        fontsize=9,
        fontweight="bold",
        family=FONT_MONO,
    )

    cards = [
        ("1H Passes", str(h1_passes), TEXT_BR),
        ("1H Top Link", _top_pair(first_half["edges"]), TEXT_BR),
        ("2H Passes", str(h2_passes), TEXT_BR),
        ("2H Top Link", _top_pair(second_half["edges"]), TEXT_BR),
        (
            "Total Links",
            f"{len(first_half['edges']) + len(second_half['edges'])}",
            C_GOLD,
        ),
    ]
    metric_strip(fig, cards=cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  5. xT MAP v2
# ═════════════════════════════════════════════════════════════════════════
def render_xt_map_v2(team_name, opp_name, score, team_color, passes):
    team_color = _clean_dark_navy(team_color)
    fig = plt.figure(figsize=(16, 10), facecolor=BG_DARK)
    chrome(
        fig,
        section="XT MAP",
        title=f"{team_name} — Expected Threat (xT)",
        subtitle="Heatmap = pitch xT value (cell numbers) · indigo arrows = the top-10 threat-creating passes",
        hn=team_name,
        an=opp_name,
        score=score,
        footer_note="Direction of attack ↑",
    )

    PX, PY, PW, PH = 0.05, 0.16, 0.46, 0.72
    header_h, body_h = panel_header_geom(PH)
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (PX, PY + PH - header_h),
            PW,
            header_h,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor=BG_HEADER,
            edgecolor=GRID_COL,
            linewidth=1.0,
            zorder=1,
        )
    )
    dot_y = PY + PH - header_h / 2
    fig.add_artist(
        mpatches.Circle(
            (PX + 0.018, dot_y),
            0.0035,
            transform=fig.transFigure,
            facecolor=team_color,
            edgecolor="none",
            zorder=2,
        )
    )
    fig.text(
        PX + 0.030,
        dot_y,
        "XT MAP",
        ha="left",
        va="center",
        color=TEXT_BR,
        fontsize=10.5,
        fontweight="bold",
        family=FONT_MONO,
        zorder=2,
    )
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (PX, PY),
            PW,
            body_h,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor=BG_MID,
            edgecolor=GRID_COL,
            linewidth=1.0,
            zorder=-2,
        )
    )

    ax = fig.add_axes([PX + 0.035, PY + 0.020, PW - 0.07, body_h - 0.045])
    _draw_vertical_pitch(ax, line_alpha=0.42)
    pax = _VerticalPitchProxy(ax)

    rows_n, cols_n = 8, 12
    cell_w, cell_h = 100 / cols_n, 100 / rows_n
    grid = np.zeros((rows_n, cols_n))
    for r in range(rows_n):
        for c in range(cols_n):
            grid[r, c] = ((c / cols_n) ** 1.6) * 0.6 + (
                1 - abs(r - rows_n / 2 + 0.5) / (rows_n / 2)
            ) * 0.18

    # Perceptual navy -> teal -> green -> gold ramp with a VISIBLE floor, so
    # even low-xT cells (own half) stay coloured — no black voids in the grid.
    cmap = LinearSegmentedColormap.from_list(
        "xt", ["#14233f", "#175a6b", "#2f9e7d", "#d9a400", "#ffe08a"]
    )
    pax.imshow(
        grid,
        extent=[0, 100, 0, 100],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=0.7,
        alpha=0.88,
        zorder=1,
    )

    for r in range(rows_n):
        for c in range(cols_n):
            v = grid[r, c]
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            # Dark ink on hot (light gold) cells, light ink on cold cells.
            num_col = "#0a0a0a" if v >= 0.42 else TEXT_BR
            pax.text(
                cx,
                cy,
                f"{v:.3f}",
                ha="center",
                va="center",
                color=num_col,
                fontsize=5.8,
                fontweight="bold",
                family=FONT_MONO,
                alpha=0.95,
                zorder=3,
            )

    pos = [p for p in passes if p.get("successful") and p.get("xT", 0) > 0]
    # Only the highest-threat passes are drawn as arrows — the mass of small
    # positive/negative passes is left to the heatmap, which keeps the map clean.

    # Top 10 threat passes: outlined indigo arrows.
    top_passes = sorted(pos, key=lambda p: -p["xT"])[:10]
    for p in top_passes:
        _draw_vertical_arrow(
            ax,
            (p["x"], p["y"]),
            (p["end_x"], p["end_y"]),
            color=XT_ARROW,
            lw=2.2,
            alpha=0.95,
            mutation_scale=12.5,
            rad=0.055,
            zorder=8,
        )

    _draw_vertical_attack_arrow(ax, x=-1.3, y0=3.0, y1=16.0)

    by_player = {}
    for p in pos:
        nm = p.get("player") or "—"
        by_player.setdefault(nm, {"xT": 0, "n": 0})
        by_player[nm]["xT"] += p["xT"]
        by_player[nm]["n"] += 1
    top_creators = sorted(by_player.items(), key=lambda kv: -kv[1]["xT"])[:6]

    ax2 = panel_card(
        fig, 0.55, 0.50, 0.41, 0.38, title="Top xT Creators", accent=team_color
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.text(
        0.04,
        0.90,
        "PLAYER",
        color=TEXT_DIM,
        fontsize=8.7,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax2.transAxes,
        va="center",
    )
    ax2.text(
        0.65,
        0.90,
        "PASS",
        color=TEXT_DIM,
        fontsize=8.7,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax2.transAxes,
        va="center",
        ha="center",
    )
    ax2.text(
        0.95,
        0.90,
        "xT",
        ha="right",
        color=TEXT_DIM,
        fontsize=8.7,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax2.transAxes,
        va="center",
    )
    ax2.plot(
        [0.03, 0.97], [0.84, 0.84], color=GRID_COL, lw=1.0, transform=ax2.transAxes
    )
    if top_creators:
        n = max(len(top_creators), 1)
        rh = 0.74 / n
        for i, (nm, d) in enumerate(top_creators):
            cy = 0.78 - (i + 0.5) * rh
            if i > 0:
                ax2.plot(
                    [0.03, 0.97],
                    [cy + rh / 2, cy + rh / 2],
                    color=GRID_SOFT,
                    lw=0.8,
                    transform=ax2.transAxes,
                    zorder=1,
                )
            ax2.text(
                0.04,
                cy,
                (nm or "—").split()[-1],
                ha="left",
                va="center",
                color="#FFFFFF",
                fontsize=11.0,
                fontweight="bold",
                family=FONT_SANS,
                transform=ax2.transAxes,
                zorder=2,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=BG_DARK)],
            )
            ax2.text(
                0.65,
                cy,
                str(d["n"]),
                ha="center",
                va="center",
                color=TEXT_DIM,
                fontsize=10,
                family=FONT_MONO,
                transform=ax2.transAxes,
                zorder=2,
            )
            ax2.text(
                0.95,
                cy,
                f"{d['xT']:.3f}",
                ha="right",
                va="center",
                color=C_GOLD,
                fontsize=10.5,
                fontweight="bold",
                family=FONT_MONO,
                transform=ax2.transAxes,
                zorder=2,
            )
    else:
        ax2.text(
            0.5,
            0.4,
            "No positive-xT passes",
            ha="center",
            va="center",
            color=TEXT_FAD,
            fontsize=10,
            style="italic",
            family=FONT_SANS,
            transform=ax2.transAxes,
        )

    total_xt = sum(p["xT"] for p in pos)
    n_pass = len(pos)
    if top_creators and total_xt > 0:
        leader_name = (top_creators[0][0] or "—").split()[-1]
        leader_xt = top_creators[0][1]["xT"]
        leader_share = leader_xt / total_xt * 100
        insight = (
            f"{team_name} created {total_xt:.2f} xT across {n_pass} "
            f"positive passes. {leader_name} alone delivered "
            f"{leader_share:.0f}% of the team's threat through the lines."
        )
    else:
        insight = (
            f"{team_name} created {total_xt:.2f} xT across {n_pass} positive passes."
        )
    key_insight(fig, 0.55, 0.16, 0.41, 0.30, text=insight, wrap=52)

    cards = [
        ("Total xT", f"{total_xt:.2f}", C_GOLD),
        ("Pos. Passes", str(n_pass), team_color),
        ("Avg xT/Pass", f"{(total_xt/n_pass if n_pass else 0):.3f}", C_GOLD),
        ("Top Pass xT", f"{max((p['xT'] for p in pos), default=0):.3f}", team_color),
        (
            "Top Creator",
            (top_creators[0][0].split()[-1] if top_creators else "—"),
            TEXT_BR,
        ),
    ]
    metric_strip(fig, cards=cards)
    return fig


# Shot types counted as on-target in WhoScored event data
ON_TARGET_TYPES = {"Goal", "SavedShot"}


def _safe(v, default=0):
    try:
        if v is None:
            return default
        import math

        if isinstance(v, float) and math.isnan(v):
            return default
        return v
    except Exception:
        return default


def _match_colors(info):
    """Return the canonical first-listed and second-listed team colours."""
    return C_HOME, C_AWAY


def _shots_for_team(events, team_id):
    out = []
    sub = events[(events["is_shot"] == True) & (events["team_id"] == team_id)]
    for _, row in sub.iterrows():
        # A shot with no real x/y in the feed must be DROPPED, not defaulted
        # to (0, 0) — (0, 0) is a real point (the far side of the pitch), so
        # silently plotting a coordinate-less shot there sends its arrow/xG
        # label way outside the shot-map's normal zone. On export with
        # bbox_inches='tight' that stray label drags the whole saved image
        # taller, showing up as a big blank gap with an orphaned "0.00".
        raw_x, raw_y = row.get("x"), row.get("y")
        if raw_x is None or raw_y is None:
            continue
        try:
            if pd.isna(raw_x) or pd.isna(raw_y):
                continue
        except Exception:
            pass
        # Drop own goals: an own goal is logged on the scorer's own team_id but
        # credited (scoring_team) to the opponent, and its coordinates sit at
        # the scorer's OWN goal line — so drawing it on this team's attack-up
        # shot map puts a stray arrow at the wrong (defensive) end. It isn't a
        # real shot at the opponent's goal, so it doesn't belong here.
        if bool(row.get("is_goal", False)):
            st = row.get("scoring_team")
            try:
                if st is not None and not pd.isna(st) and int(st) != int(team_id):
                    continue
            except Exception:
                pass
        shot_type = row.get("shot_whoscored_type") or row.get("type") or ""
        _q = (
            str(_safe(row.get("qualifier_names"), ""))
            + " "
            + str(_safe(row.get("situation"), ""))
            + " "
            + str(_safe(row.get("play_type"), ""))
        ).lower()
        is_pen = bool(row.get("is_penalty", False)) or "penalty" in _q
        out.append(
            {
                "x": float(raw_x),
                "y": float(raw_y),
                "xG": float(_safe(row.get("xG"), 0) or 0),
                "is_goal": bool(row.get("is_goal", False)),
                "is_on_target": shot_type in ON_TARGET_TYPES,
                "is_blocked": shot_type == "BlockedShot",
                "is_penalty": is_pen,
                "player": str(_safe(row.get("player"), "")),
                "minute": int(_safe(row.get("minute"), 0) or 0),
                "end_x": _safe(row.get("end_x"), None),
                "end_y": _safe(row.get("end_y"), None),
            }
        )
    return out


def _own_goals(events, info):
    """Own goals as (minute, scorer, beneficiary_team_id) — these are real
    goals in the scoreline but are NOT shots the beneficiary took, so the
    shot-based xG flow drops them; the goals panel still lists them (tagged
    OG) so both teams' goals show and the goal counts match the scoreline."""
    out = []
    if events is None or events.empty or "is_goal" not in events.columns:
        return out
    g = events[events["is_goal"].fillna(False)]
    if "is_penalty_shootout" in g.columns:
        g = g[~g["is_penalty_shootout"].fillna(False)]
    if "scoring_team" not in g.columns:
        return out
    for _, r in g.iterrows():
        st = r.get("scoring_team")
        tid = r.get("team_id")
        try:
            if st is not None and not pd.isna(st) and int(st) != int(tid):
                out.append(
                    (
                        int(_safe(r.get("minute"), 0) or 0),
                        str(_safe(r.get("player"), "") or ""),
                        int(st),
                    )
                )
        except Exception:
            continue
    return out


def make_xg_flow_v2(events, info, xg_data=None):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    hid = info.get("home_id")
    aid = info.get("away_id")
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)
    went_to_et, pens = _match_extra_time_pens(events, info)
    ogs = _own_goals(events, info)
    own_home = [(m, p) for (m, p, t) in ogs if t == hid]
    own_away = [(m, p) for (m, p, t) in ogs if t == aid]
    return render_xg_flow_v2(
        hn,
        an,
        str(score),
        hc,
        ac,
        _shots_for_team(events, hid),
        _shots_for_team(events, aid),
        went_to_et=went_to_et,
        pens=pens,
        own_home=own_home,
        own_away=own_away,
    )


def make_shot_map_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name = an if is_home else hn
    score = info.get("score") or "—"
    return render_shot_map_v2(
        team_name, opp_name, str(score), team_color, _shots_for_team(events, team_id)
    )


def make_shot_breakdown_v2(events, info, xg_data):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)
    h = xg_data.get(hn, {}) if xg_data else {}
    a = xg_data.get(an, {}) if xg_data else {}
    home = {
        k: int(_safe(h.get(k), 0))
        for k in ("shots", "on_target", "off_target", "blocked", "post")
    }
    away = {
        k: int(_safe(a.get(k), 0))
        for k in ("shots", "on_target", "off_target", "blocked", "post")
    }
    home["xG"] = float(_safe(h.get("xG"), 0) or 0)
    away["xG"] = float(_safe(a.get("xG"), 0) or 0)

    def _assist_from_context(goal_row):
        """Fallback assist lookup from the same-team action immediately before a goal."""
        explicit = goal_row.get("assist_player")
        if explicit and str(explicit).lower() != "nan":
            return str(explicit), goal_row.get("assist_type")
        if events is None or events.empty:
            return "", ""
        scoring_team = goal_row.get("scoring_team") or goal_row.get("team_id")
        minute = _safe(goal_row.get("minute"), 0) or 0
        second = _safe(goal_row.get("second"), 0) or 0
        goal_time = float(minute) * 60 + float(second)
        cand = events[events.get("team_id") == scoring_team].copy()
        if cand.empty:
            return "", ""
        cand["__t"] = cand.get("minute", 0).fillna(0).astype(float) * 60 + cand.get(
            "second", 0
        ).fillna(0).astype(float)
        cand = cand[(cand["__t"] < goal_time) & (cand["__t"] >= goal_time - 25)]
        if "is_pass" in cand.columns:
            cand = cand[cand["is_pass"] == True]
        if "outcome" in cand.columns:
            successful = cand[
                cand["outcome"].fillna("").astype(str).str.lower().eq("successful")
            ]
            if not successful.empty:
                cand = successful
        if cand.empty:
            return "", ""
        if "is_key_pass" in cand.columns and cand["is_key_pass"].any():
            cand = cand[cand["is_key_pass"] == True]
        last = cand.sort_values("__t").iloc[-1]
        name = str(last.get("player") or "")
        q = str(last.get("qualifier_names") or "")
        if "Cross" in q:
            at = "Cross"
        elif "ThroughBall" in q:
            at = "ThroughBall"
        elif "Chipped" in q:
            at = "Chipped"
        elif "FastBreak" in q:
            at = "FastBreak"
        else:
            at = last.get("assist_type") or "Assist"
        return name, at

    def _goal_qualifiers(row):
        q = row.get("qualifier_names") or []
        if isinstance(q, str):
            return q.lower()
        if isinstance(q, (list, tuple, set)):
            return " ".join(str(x) for x in q).lower()
        return ""

    def _body_label(row):
        q = _goal_qualifiers(row)
        body = str(row.get("body_part") or "").lower()
        if bool(row.get("is_header")) or "head" in q or "head" in body:
            return "HEADER"
        if "right" in body and "foot" in body:
            return "RIGHT FOOT"
        if "left" in body and "foot" in body:
            return "LEFT FOOT"
        if "foot" in body:
            return "FOOT"
        return ""

    def _goal_type_from_context(goal_row):
        if bool(goal_row.get("is_own_goal")):
            return "OWN GOAL"
        if bool(goal_row.get("is_penalty")) or "penalty" in _goal_qualifiers(goal_row):
            base = "PENALTY"
        else:
            q = _goal_qualifiers(goal_row)
            if "fromcorner" in q or "cornertaken" in q or "corner" in q:
                base = "CORNER"
            elif "throwin" in q or "throw in" in q:
                base = "THROW-IN"
            elif (
                "freekick" in q
                or "free kick" in q
                or bool(goal_row.get("is_direct_fk"))
            ):
                base = "FREE KICK"
            else:
                base = "OPEN PLAY"
                if events is not None and not events.empty:
                    scoring_team = goal_row.get("scoring_team") or goal_row.get(
                        "team_id"
                    )
                    minute = _safe(goal_row.get("minute"), 0) or 0
                    second = _safe(goal_row.get("second"), 0) or 0
                    goal_time = float(minute) * 60 + float(second)
                    cand = events[events.get("team_id") == scoring_team].copy()
                    if not cand.empty:
                        cand["__t"] = cand.get("minute", 0).fillna(0).astype(
                            float
                        ) * 60 + cand.get("second", 0).fillna(0).astype(float)
                        cand = cand[
                            (cand["__t"] < goal_time) & (cand["__t"] >= goal_time - 70)
                        ]
                        for _, prev in (
                            cand.sort_values("__t", ascending=False).head(8).iterrows()
                        ):
                            pq = _goal_qualifiers(prev)
                            if (
                                "cornertaken" in pq
                                or "fromcorner" in pq
                                or "corner" in pq
                            ):
                                base = "CORNER"
                                break
                            if "throwin" in pq or "throw in" in pq:
                                base = "THROW-IN"
                                break
                            if "freekick" in pq or "free kick" in pq:
                                base = "FREE KICK"
                                break
        body = _body_label(goal_row)
        return f"{base} - {body}" if body and base not in {"OWN GOAL"} else base

    # Goals list — same shape the renderer expects
    goals_list = []
    gdf = events[events["is_goal"] == True].sort_values("minute")
    for _, row in gdf.iterrows():
        side = "home" if row.get("scoring_team") == info.get("home_id") else "away"
        gtype = _goal_type_from_context(row)
        ap, at = _assist_from_context(row)
        if ap and str(ap).lower() != "nan":
            assist = f"{ap}" + (f" ({at})" if at and str(at).lower() != "nan" else "")
        else:
            assist = "—"
        goals_list.append(
            (
                f"{int(_safe(row.get('minute'), 0))}'",
                (str(row.get("player") or "")).split()[-1] or "—",
                gtype,
                assist,
                float(_safe(row.get("xG"), 0) or 0),
                side,
            )
        )
    return render_shot_breakdown_v2(hn, an, str(score), home, away, goals_list, hc, ac)


def make_pass_network_v2(events, info, team_id, team_color):
    """
    Build players + edges by inferring receivers from the next same-team
    event (same approach as the legacy `build_pass_network`).
    """
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name = an if is_home else hn
    score = info.get("score") or "—"

    team_evts = (
        events[events["team_id"] == team_id]
        .sort_values(["minute", "second"])
        .reset_index(drop=True)
    )

    sub_in = set(info.get("sub_in") or [])
    sub_out = set(info.get("sub_out") or [])
    red_cards = set(info.get("red_cards") or [])
    meta = info.get("player_meta") or {}

    def _pid_role(pid):
        if pid in red_cards:
            return "red_card"
        if pid in sub_in and pid in sub_out:
            return "both_sub"
        if pid in sub_in:
            return "sub_in"
        if pid in sub_out:
            return "sub_out"
        return ""

    def _build_network(src_evts):
        passes = src_evts[src_evts["is_pass"] == True]
        nodes = {}
        if not passes.empty:
            for pid, grp in passes.groupby("player_id"):
                try:
                    nodes[pid] = {
                        "name": str(grp["player"].iloc[0]),
                        "avg_x": float(grp["x"].mean()),
                        "avg_y": float(grp["y"].mean()),
                        "passes": int(len(grp)),
                    }
                except Exception:
                    continue

        # RECV_MAX_DIST: a genuine reception starts where the ball arrived, so
        # the next same-team event should begin near the pass-end coordinate.
        # If it starts far away the "next event" is not the receiver (the pass
        # was cut out, cleared, or the feed skipped an event) — drop that link.
        RECV_MAX_DIST = 14.0
        edges_count = {}  # undirected pair total (drives width/tier)
        dir_count = {}  # directed (passer -> receiver) for arrow direction
        succ = src_evts[
            (src_evts["is_pass"] == True) & (src_evts["outcome"] == "Successful")
        ]
        for i in range(len(succ)):
            curr_idx = succ.index[i]
            row = succ.iloc[i]
            passer_id = row.get("player_id")
            if passer_id is None:
                continue
            later = src_evts[
                (src_evts.index > curr_idx) & src_evts["player_id"].notna()
            ]
            if later.empty:
                continue
            nxt = later.iloc[0]
            recv_id = nxt["player_id"]
            if recv_id == passer_id:
                continue
            # Validate the receiver by pass-end proximity (Opta end coords).
            try:
                pe_x, pe_y = row.get("end_x"), row.get("end_y")
                nx_, ny_ = nxt.get("x"), nxt.get("y")
                vals = [pe_x, pe_y, nx_, ny_]
                if all(
                    v is not None and not (isinstance(v, float) and np.isnan(v))
                    for v in vals
                ):
                    if (
                        (float(pe_x) - float(nx_)) ** 2
                        + (float(pe_y) - float(ny_)) ** 2
                    ) ** 0.5 > RECV_MAX_DIST:
                        continue
            except Exception:
                pass
            dir_count[(passer_id, recv_id)] = dir_count.get((passer_id, recv_id), 0) + 1
            if recv_id not in nodes:
                rr = src_evts[src_evts["player_id"] == recv_id]
                if not rr.empty:
                    nodes[recv_id] = {
                        "name": str(rr["player"].iloc[0]),
                        "avg_x": float(rr["x"].mean()),
                        "avg_y": float(rr["y"].mean()),
                        "passes": 0,
                    }
            key = tuple(sorted([passer_id, recv_id]))
            edges_count[key] = edges_count.get(key, 0) + 1

        # Prune the network to its connected core (StatsBomb/Opta-style) so a
        # player who only made a stray pass or two — typically a late
        # substitute — does not float at the edge of the pitch with no links.
        # A node is kept only if it is connected by a real link (a pair it
        # exchanged >= MIN_EDGE passes with) OR it is a high-volume passer.
        # A half with many candidate nodes (e.g. 5 subs came on) is where the
        # map gets genuinely crowded, so tighten both thresholds there — pass
        # volume is the best available proxy for "played a meaningful chunk
        # of the half" without per-player on/off minute timestamps.
        if len(nodes) >= 15:
            MIN_EDGE, MIN_NODE_PASSES = 3, 9
        elif len(nodes) >= 13:
            MIN_EDGE, MIN_NODE_PASSES = 3, 7
        else:
            MIN_EDGE, MIN_NODE_PASSES = 2, 6
        connected = set()
        for (a, b), c in edges_count.items():
            if c >= MIN_EDGE:
                connected.add(a)
                connected.add(b)
        keep = {
            pid
            for pid in nodes
            if pid in connected or nodes[pid]["passes"] >= MIN_NODE_PASSES
        }
        # Always retain substitutes who genuinely played part of the half. Subs
        # come on late and naturally make fewer passes, so the volume-based
        # pruning above wrongly drops them and the SUBSTITUTE nodes vanish from
        # the map. Keep any sub (by explicit sub role or meta flag) that made a
        # handful of passes so they appear, coloured as a substitute.
        SUB_MIN_PASSES = 1
        for pid in nodes:
            is_sub = bool((meta.get(pid) or {}).get("is_sub")) or (pid in sub_in)
            if is_sub and nodes[pid]["passes"] >= SUB_MIN_PASSES:
                keep.add(pid)
        # Never let the map collapse on sparse halves: if pruning is too
        # aggressive, fall back to every node that touched the ball.
        if len(keep) < 7:
            keep = set(nodes.keys())

        players_list = [
            {
                "name": nodes[pid]["name"],
                "x": nodes[pid]["avg_x"],
                "y": nodes[pid]["avg_y"],
                "passes": nodes[pid]["passes"],
                "role": _pid_role(pid),
                "position": (meta.get(pid) or {}).get("position"),
                "shirt": (meta.get(pid) or {}).get("shirt"),
                "is_sub": bool((meta.get(pid) or {}).get("is_sub")),
            }
            for pid in keep
            if pid in nodes
            and not (np.isnan(nodes[pid]["avg_x"]) or np.isnan(nodes[pid]["avg_y"]))
        ]

        edges_list = []
        for (a, b), count in edges_count.items():
            if a not in keep or b not in keep:
                continue
            if a not in nodes or b not in nodes:
                continue
            # Orient the link toward the dominant passing direction so the
            # renderer can draw the arrowhead from the main passer to the main
            # receiver of the pair.
            ab = dir_count.get((a, b), 0)
            ba = dir_count.get((b, a), 0)
            f_id, t_id = (a, b) if ab >= ba else (b, a)
            edges_list.append(
                {
                    "from": nodes[f_id]["name"],
                    "to": nodes[t_id]["name"],
                    "count": int(count),
                }
            )
        edges_list.sort(key=lambda e: -e["count"])
        return {"players": players_list, "edges": edges_list}

    first_half_evts = team_evts[team_evts["minute"].fillna(0) < 46].copy()
    second_half_evts = team_evts[team_evts["minute"].fillna(0) >= 46].copy()
    if first_half_evts.empty and not team_evts.empty:
        first_half_evts = team_evts.iloc[: len(team_evts) // 2].copy()
    if second_half_evts.empty and not team_evts.empty:
        second_half_evts = team_evts.iloc[len(team_evts) // 2 :].copy()

    return render_pass_network_halves_v2(
        team_name,
        opp_name,
        str(score),
        team_color,
        _build_network(first_half_evts),
        _build_network(second_half_evts),
    )


def make_xt_map_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name = an if is_home else hn
    score = info.get("score") or "—"

    passes_list = []
    if "xT" in events.columns:
        sub = events[
            (events["is_pass"] == True)
            & (events["team_id"] == team_id)
            & (events["xT"].notna())
        ]
        for _, row in sub.iterrows():
            passes_list.append(
                {
                    "x": float(_safe(row.get("x"), 0)),
                    "y": float(_safe(row.get("y"), 0)),
                    "end_x": float(_safe(row.get("end_x"), 0)),
                    "end_y": float(_safe(row.get("end_y"), 50)),
                    "xT": float(_safe(row.get("xT"), 0) or 0),
                    "player": str(_safe(row.get("player"), "")),
                    "successful": row.get("outcome") == "Successful",
                }
            )
    return render_xt_map_v2(team_name, opp_name, str(score), team_color, passes_list)


# ═════════════════════════════════════════════════════════════════════════
#  GENERIC PITCH-OVERLAY v2 — used by all simple "pitch + dots/arrows"
#  visuals (defensive heatmap, average positions, box entries, crosses,
#  high turnovers, etc.). Same chrome + sidebar + metric strip shape.
# ═════════════════════════════════════════════════════════════════════════
def render_pitch_overlay_v2(
    *,
    section,
    title,
    subtitle,
    hn,
    an,
    score,
    footer_note,
    team_color,
    draw_overlay,
    draw_legend=None,
    sidebar_title,
    sidebar_headers,
    sidebar_rows,
    sidebar_value_cols=None,
    insight_text,
    metric_cards,
):
    """
    sidebar_rows: list of tuples; each tuple aligns with sidebar_headers.
    sidebar_value_cols: optional list of column-x positions for each header.
                       Defaults to evenly spaced.
    draw_legend: optional callback(fig, panel_x, panel_y) for visuals that
                 need an extra legend row below the pitch, drawn directly
                 in figure coordinates so its position never depends on
                 the inner axes' own coordinate system.
    """
    team_color = _clean_dark_navy(team_color)
    fig = plt.figure(figsize=(16, 10), facecolor=BG_DARK)
    chrome(
        fig,
        section=section,
        title=title,
        subtitle=subtitle,
        hn=hn,
        an=an,
        score=score,
        footer_note=footer_note,
    )

    PX, PY, PW, PH = 0.05, 0.16, 0.46, 0.72
    header_h, body_h = panel_header_geom(PH)
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (PX, PY + PH - header_h),
            PW,
            header_h,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor=BG_HEADER,
            edgecolor=GRID_COL,
            linewidth=1.0,
            zorder=1,
        )
    )
    dot_y = PY + PH - header_h / 2
    fig.add_artist(
        mpatches.Circle(
            (PX + 0.018, dot_y),
            0.0035,
            transform=fig.transFigure,
            facecolor=team_color,
            edgecolor="none",
            zorder=2,
        )
    )
    fig.text(
        PX + 0.030,
        dot_y,
        section,
        ha="left",
        va="center",
        color=TEXT_BR,
        fontsize=10.5,
        fontweight="bold",
        family=FONT_MONO,
        zorder=2,
    )
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (PX, PY),
            PW,
            body_h,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor=BG_MID,
            edgecolor=GRID_COL,
            linewidth=1.0,
            zorder=-2,
        )
    )

    legend_reserve = 0.105 if callable(draw_legend) else 0.020
    ax = fig.add_axes(
        [PX + 0.035, PY + legend_reserve, PW - 0.07, body_h - legend_reserve - 0.030]
    )
    _draw_vertical_pitch(ax, line_alpha=0.46)
    pax = _VerticalPitchProxy(ax)
    if callable(draw_overlay):
        draw_overlay(pax)
    _draw_vertical_attack_arrow(ax, x=-1.3, y0=3.0, y1=16.0)
    if callable(draw_legend):
        draw_legend(fig, PX, PY)

    # Sidebar table
    ax2 = panel_card(
        fig, 0.55, 0.50, 0.41, 0.38, title=sidebar_title, accent=team_color
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    n_cols = len(sidebar_headers)
    if sidebar_value_cols is None:
        if n_cols == 1:
            xs = [0.04]
        elif n_cols == 2:
            xs = [0.04, 0.95]
        else:
            xs = [0.04] + [0.04 + (i * 0.91 / (n_cols - 1)) for i in range(1, n_cols)]
    else:
        xs = sidebar_value_cols
    for i, (lbl, x) in enumerate(zip(sidebar_headers, xs)):
        ha = "left" if i == 0 else ("right" if i == n_cols - 1 else "center")
        ax2.text(
            x,
            0.90,
            lbl,
            ha=ha,
            va="center",
            color=TEXT_DIM,
            fontsize=8.7,
            fontweight="bold",
            family=FONT_MONO,
            transform=ax2.transAxes,
        )
    ax2.plot(
        [0.03, 0.97], [0.84, 0.84], color=GRID_COL, lw=1.0, transform=ax2.transAxes
    )
    if sidebar_rows:
        n = max(len(sidebar_rows), 1)
        rh = 0.74 / n
        for i, row in enumerate(sidebar_rows):
            cy = 0.78 - (i + 0.5) * rh
            if i > 0:
                ax2.plot(
                    [0.03, 0.97],
                    [cy + rh / 2, cy + rh / 2],
                    color=GRID_SOFT,
                    lw=0.8,
                    transform=ax2.transAxes,
                    zorder=1,
                )
            for j, (val, x) in enumerate(zip(row, xs)):
                ha = "left" if j == 0 else ("right" if j == n_cols - 1 else "center")
                is_last = j == n_cols - 1
                # Keep identifiers and player names bright. Team colour is
                # reserved for marks; the final value only uses it when WCAG
                # contrast is sufficient, otherwise it falls back to white.
                col = _safe_team_text(team_color) if is_last else TEXT_BR
                fs = 10.5 if j < n_cols - 1 else 11
                fw = "bold"
                fam = FONT_SANS if j < n_cols - 1 else FONT_MONO
                ax2.text(
                    x,
                    cy,
                    str(val),
                    ha=ha,
                    va="center",
                    color=col,
                    fontsize=fs,
                    fontweight=fw,
                    family=fam,
                    transform=ax2.transAxes,
                    zorder=2,
                )
    else:
        ax2.text(
            0.5,
            0.4,
            "No data recorded",
            ha="center",
            va="center",
            color=TEXT_FAD,
            fontsize=10,
            style="italic",
            family=FONT_SANS,
            transform=ax2.transAxes,
        )

    key_insight(fig, 0.55, 0.16, 0.41, 0.30, text=insight_text, wrap=52)
    metric_strip(fig, cards=metric_cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  DEFENSIVE HEATMAP v2  (figs 26 / 27)
# ═════════════════════════════════════════════════════════════════════════
DEF_TYPE_COLORS = {
    "Tackle": "#22c55e",
    "Interception": "#f59e0b",
    "Clearance": "#F5C542",
    "BlockedShot": "#a855f7",
    "BallRecovery": "#22C55E",
    "Foul": "#ef4444",
    "Aerial": "#facc15",
}


DEF_HEAT_CMAP = LinearSegmentedColormap.from_list(
    "defensive_density",
    [
        (0.00, to_rgba("#000000", 0.00)),
        (0.18, to_rgba("#166534", 0.12)),
        (0.42, to_rgba("#22c55e", 0.42)),
        (0.68, to_rgba("#facc15", 0.68)),
        (0.88, to_rgba("#fb7185", 0.82)),
        (1.00, to_rgba("#ffffff", 0.95)),
    ],
)


def _gaussian_kernel1d(radius: int = 4, sigma: float = 1.55) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x**2) / (2 * sigma**2))
    return k / max(k.sum(), 1e-9)


def _smooth_density_grid(
    grid: np.ndarray, radius: int = 4, sigma: float = 1.55
) -> np.ndarray:
    if grid.size == 0 or not np.any(grid):
        return grid
    kernel = _gaussian_kernel1d(radius=radius, sigma=sigma)
    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=grid
    )
    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=smoothed
    )
    return smoothed


def _blocked_shots_for_team(events, info, team_id) -> int:
    """Count blocked shots as defensive blocks by the opponent of the shooter."""
    hid = info.get("home_id")
    aid = info.get("away_id")
    opp_id = aid if team_id == hid else (hid if team_id == aid else None)
    if opp_id is None:
        return 0
    opponent_blocked_shots = defensive_blocks_count(events, team_id, opp_id)
    if opponent_blocked_shots == 0:
        opp_side = "away" if team_id == hid else "home"
        mc = (info.get("matchcentre_stats", {}) or {}).get(opp_side, {}) or {}
        opponent_blocked_shots = int(_safe(mc.get("blocked"), 0) or 0)
    return opponent_blocked_shots


def _defensive_events_for_team(events, info, team_id):
    """Return defensive events, mapping opponent BlockedShot events to this team."""
    own_types = [t for t in DEF_TYPE_COLORS.keys() if t != "BlockedShot"]
    own = events[(events["team_id"] == team_id) & events["type"].isin(own_types)].copy()
    hid = info.get("home_id")
    aid = info.get("away_id")
    opp_id = aid if team_id == hid else (hid if team_id == aid else None)

    blocks = (
        defensive_block_events(events, team_id, opp_id)
        if opp_id is not None
        else events.iloc[0:0].copy()
    )
    if own.empty:
        return blocks
    if blocks.empty:
        return own
    return pd.concat([own, blocks], ignore_index=True, sort=False)


def make_defensive_heatmap_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name = an if is_home else hn
    score = info.get("score") or "—"

    sub = _defensive_events_for_team(events, info, team_id)
    by_type = {}
    by_player = {}
    points = []
    for _, row in sub.iterrows():
        t = row.get("type")
        x = float(_safe(row.get("x"), 50))
        y = float(_safe(row.get("y"), 50))
        p = str(_safe(row.get("player"), "—"))
        points.append((x, y, t))
        by_type[t] = by_type.get(t, 0) + 1
        by_player[p] = by_player.get(p, 0) + 1

    def draw_overlay(ax):
        if not points:
            return
        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)
        grid, _, _ = np.histogram2d(ys, xs, bins=(34, 48), range=((0, 100), (0, 100)))
        density = _smooth_density_grid(grid, radius=4, sigma=1.65)
        if np.any(density):
            norm = density / max(float(density.max()), 1e-9)
            alpha = np.clip(norm**0.70, 0, 1) * 0.86
            alpha[norm < 0.055] = 0
            ax.imshow(
                norm,
                extent=[0, 100, 0, 100],
                origin="lower",
                cmap=DEF_HEAT_CMAP,
                alpha=alpha,
                interpolation="bilinear",
                zorder=2,
                aspect="auto",
            )
            levels = [0.28, 0.48, 0.68]
            ax.contour(
                norm,
                levels=levels,
                extent=[0, 100, 0, 100],
                origin="lower",
                colors=[team_color, "#facc15", "#ffffff"],
                linewidths=[0.65, 0.85, 1.05],
                alpha=0.34,
                zorder=3,
            )
            hot_idx = np.argwhere(norm >= max(0.68, float(norm.max()) * 0.72))
            if len(hot_idx):
                for r, c in hot_idx[:: max(1, len(hot_idx) // 7)][:7]:
                    hx = (c + 0.5) * (100 / norm.shape[1])
                    hy = (r + 0.5) * (100 / norm.shape[0])
                    ax.scatter(
                        [hx],
                        [hy],
                        s=70,
                        facecolor="none",
                        edgecolor="#ffffff",
                        lw=0.9,
                        alpha=0.50,
                        zorder=4,
                    )

        for t, col in DEF_TYPE_COLORS.items():
            pts = [(x, y) for x, y, kind in points if kind == t]
            if not pts:
                continue
            px = [p[0] for p in pts]
            py = [p[1] for p in pts]
            ax.scatter(
                px,
                py,
                s=34,
                facecolor=col,
                edgecolor="#ffffff",
                lw=0.35,
                alpha=0.66,
                zorder=5,
            )

    # Legend drawn directly in figure coordinates (not pitch or axes
    # coordinates) so its position is independent of the pitch's own
    # aspect ratio and never overlaps the metric strip below it.
    def draw_legend(fig, panel_x, panel_y):
        density_items = [
            ("LOW DENSITY", "#22c55e"),
            ("MEDIUM", "#facc15"),
            ("HIGH", "#fb7185"),
        ]
        ly = panel_y + 0.072
        lx = panel_x + 0.045
        for lbl, col in density_items:
            fig.add_artist(
                plt.Circle(
                    (lx, ly),
                    0.0035,
                    transform=fig.transFigure,
                    facecolor=col,
                    edgecolor=TEXT_BR,
                    lw=0.7,
                    zorder=10,
                )
            )
            fig.text(
                lx + 0.014,
                ly,
                lbl,
                ha="left",
                va="center",
                color=TEXT_DIM,
                fontsize=8,
                fontweight="bold",
                family=FONT_MONO,
                zorder=10,
            )
            lx += 0.014 + 0.0072 * len(lbl) + 0.022

        action_types = [t for t in DEF_TYPE_COLORS if by_type.get(t, 0) > 0][:6]
        ax_x = panel_x + 0.045
        ay = panel_y + 0.034
        for t in action_types:
            col = DEF_TYPE_COLORS[t]
            lbl = t.replace("BlockedShot", "Block").replace("BallRecovery", "Recovery")
            fig.add_artist(
                plt.Circle(
                    (ax_x, ay),
                    0.0028,
                    transform=fig.transFigure,
                    facecolor=col,
                    edgecolor=TEXT_BR,
                    lw=0.5,
                    zorder=10,
                )
            )
            fig.text(
                ax_x + 0.013,
                ay,
                lbl.upper(),
                ha="left",
                va="center",
                color=TEXT_DIM,
                fontsize=7.5,
                fontweight="bold",
                family=FONT_MONO,
                zorder=10,
            )
            ax_x += 0.013 + 0.0068 * len(lbl) + 0.020

    top_players = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    sidebar_rows = [
        (name.split()[-1] if name else "—", str(cnt)) for name, cnt in top_players
    ]

    total = len(points)
    leader_name = top_players[0][0].split()[-1] if top_players else "—"
    insight = (
        (
            f"{team_name} completed {total} defensive actions. {leader_name} "
            f"led the workload with {top_players[0][1]} actions. "
            f"Tackles: {by_type.get('Tackle', 0)} · "
            f"Interceptions: {by_type.get('Interception', 0)} · "
            f"Clearances: {by_type.get('Clearance', 0)} · "
            f"Blocks: {by_type.get('BlockedShot', 0)}."
        )
        if top_players
        else f"{team_name} — no defensive data."
    )

    cards = [
        ("Total Actions", str(total), team_color),
        ("Tackles", str(by_type.get("Tackle", 0)), TEXT_BR),
        ("Interceptions", str(by_type.get("Interception", 0)), TEXT_BR),
        ("Clearances", str(by_type.get("Clearance", 0)), TEXT_BR),
        ("Blocks", str(by_type.get("BlockedShot", 0)), TEXT_BR),
    ]
    return render_pitch_overlay_v2(
        section="DEFENSIVE ACTIONS",
        title=f"{team_name} — Defensive Heatmap",
        subtitle="Heatmap intensity = concentration of defensive actions · "
        "brighter zones show repeated defensive activity",
        hn=team_name,
        an=opp_name,
        score=str(score),
        footer_note="Heat = defensive density · dots = individual action types",
        team_color=team_color,
        draw_overlay=draw_overlay,
        draw_legend=draw_legend,
        sidebar_title="Top Defenders (actions)",
        sidebar_headers=["PLAYER", "ACT"],
        sidebar_rows=sidebar_rows,
        insight_text=insight,
        metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  AVERAGE POSITIONS v2  (figs 29 / 30)
# ═════════════════════════════════════════════════════════════════════════
def make_avg_positions_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name = an if is_home else hn
    score = info.get("score") or "—"

    sub_in = set(info.get("sub_in") or [])
    sub_out = set(info.get("sub_out") or [])
    red_cards = set(info.get("red_cards") or [])
    meta = info.get("player_meta") or {}

    def _pid_role(pid):
        if pid in red_cards:
            return "red_card"
        if pid in sub_in and pid in sub_out:
            return "both_sub"
        if pid in sub_in:
            return "sub_in"
        if pid in sub_out:
            return "sub_out"
        return ""

    sub = events[
        (events["team_id"] == team_id)
        & events["player_id"].notna()
        & events["x"].notna()
        & events["y"].notna()
    ]
    grp_all = (
        sub.groupby(["player_id", "player"], dropna=True)
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            touches=("event_id", "count"),
        )
        .reset_index()
        .sort_values("touches", ascending=False)
    )
    keep_ids = set(grp_all.head(11)["player_id"].tolist())
    keep_ids.update(
        pid
        for pid in grp_all["player_id"].tolist()
        if pid in sub_in or pid in sub_out or pid in red_cards
    )
    grp = grp_all[grp_all["player_id"].isin(keep_ids)]

    players = []
    for idx, (_, r) in enumerate(grp.iterrows(), start=1):
        pid = r["player_id"]
        players.append(
            {
                "name": str(r["player"]),
                "x": float(r["x"]),
                # Raw width — the width mirror (y=0 = right flank) is now
                # done centrally in _vp_xy, so pre-mirroring here too
                # would double-flip and reverse the players again.
                "y": float(r["y"]),
                "touches": int(r["touches"]),
                "role": _pid_role(pid),
                "is_sub": bool((meta.get(pid) or {}).get("is_sub")),
                "position": (meta.get(pid) or {}).get("position"),
                "display_id": idx,
            }
        )
    max_t = max((p["touches"] for p in players), default=1)
    # Guard: if the starter flag is unreliable (everyone flagged a sub), draw
    # all as circles rather than all squares.
    _all_sub = bool(players) and all(p.get("is_sub") for p in players)
    n_players = len(players)
    label_fontsize = 7.0 if n_players <= 12 else (6.6 if n_players <= 14 else 6.2)

    # Identify the goalkeeper (positional GK, else the deepest node) so it can
    # carry the same distinct gold edge used on the pass network.
    gk_player = None
    for p in players:
        if _infer_position_bucket({"x": p["x"], "position": p.get("position")}) == "gk":
            gk_player = p
            break
    if gk_player is None and players:
        gk_player = min(players, key=lambda p: p["x"])
    hub_player = max(players, key=lambda p: p["touches"]) if players else None

    def draw_overlay(ax):
        _link_low, shape_color, _link_strong = network_link_palette(team_color)
        # Team shape: a convex hull over the outfield starters (matching the
        # pass network's "block footprint" treatment) instead of a plain
        # centroid spider-web, which reads far better at a glance.
        core = [p for p in players if not (p.get("is_sub") and not _all_sub)]
        if len(core) >= 3:
            hull = _convex_hull([(p["x"], p["y"]) for p in core])
            if len(hull) >= 3:
                # add_patch does NOT go through the proxy's coordinate
                # conversion (only scatter/text/plot do) — every hull vertex
                # must be converted through _vp_xy by hand, or the polygon
                # is drawn using raw 0-100 values on axes whose native units
                # are the narrower VP_W/VP_L scale, producing a skewed shape.
                hull_vp = [_vp_xy(hx, hy) for hx, hy in hull]
                ax.add_patch(
                    mpatches.Polygon(
                        hull_vp,
                        closed=True,
                        facecolor=team_color,
                        edgecolor=shape_color,
                        alpha=0.07,
                        lw=1.0,
                        joinstyle="round",
                        zorder=1,
                    )
                )
                ax.add_patch(
                    mpatches.Polygon(
                        hull_vp,
                        closed=True,
                        facecolor="none",
                        edgecolor=shape_color,
                        alpha=0.30,
                        lw=1.0,
                        joinstyle="round",
                        zorder=1,
                    )
                )
        if players:
            cx = float(np.mean([p["x"] for p in players]))
            cy = float(np.mean([p["y"] for p in players]))
            ax.scatter(
                [cx],
                [cy],
                s=45,
                marker="+",
                color=C_GOLD,
                linewidth=1.3,
                alpha=0.70,
                zorder=3,
            )

        # Smaller nodes + per-node label no-go radii (in 0-100 pitch units) so
        # names never get drawn across a circle — the main overlap fix.
        sizes = {id(p): 150 + 520 * (p["touches"] / max_t) for p in players}
        base_sz = 150 + 520 * 0.5
        node_radii = {id(p): 3.6 * (sizes[id(p)] / base_sz) ** 0.5 for p in players}

        # Declutter: when two players' TRUE average positions sit almost on
        # top of each other (e.g. several substitutes occupying a similar
        # tactical slot in a short cameo), the markers themselves overlap —
        # no amount of label routing fixes that. Nudge the display position
        # apart just enough for the markers to separate; the analytical
        # figures (avg depth, spreads, insight text) still use the real,
        # un-nudged p["x"]/p["y"]. A thin connector line is drawn when a node
        # actually moved, so the true spot stays traceable.
        disp = {id(p): [p["x"], p["y"]] for p in players}
        for _ in range(40):
            moved = False
            for a in players:
                for b in players:
                    if a is b:
                        continue
                    ax_, ay_ = disp[id(a)]
                    bx_, by_ = disp[id(b)]
                    dx, dy = bx_ - ax_, by_ - ay_
                    dist = (dx * dx + dy * dy) ** 0.5
                    min_dist = (node_radii[id(a)] + node_radii[id(b)]) * 0.92
                    if dist < min_dist:
                        moved = True
                        if dist < 0.05:
                            ang = (hash((id(a), id(b))) % 360) * np.pi / 180.0
                            dx, dy, dist = np.cos(ang), np.sin(ang), 1.0
                        push = (min_dist - dist) / 2.0
                        ux, uy = dx / dist, dy / dist
                        disp[id(a)][0] -= ux * push
                        disp[id(a)][1] -= uy * push
                        disp[id(b)][0] += ux * push
                        disp[id(b)][1] += uy * push
            if not moved:
                break
        for pid, (dx_, dy_) in disp.items():
            disp[pid] = [float(np.clip(dx_, 2, 98)), float(np.clip(dy_, 2, 98))]
        # Nodes at their (possibly nudged) DISPLAY position — used for every
        # marker/label/collision check below, so labels route against where
        # the dots actually are, not the stale true position.
        disp_nodes = [
            {"x": disp[id(p)][0], "y": disp[id(p)][1], "name": p["name"]}
            for p in players
        ]
        disp_radii = {id(dn): node_radii[id(p)] for p, dn in zip(players, disp_nodes)}

        taken_labels = []
        for idx, p in enumerate(players):
            sz = sizes[id(p)]
            dpx, dpy = disp[id(p)]
            if (dpx - p["x"]) ** 2 + (dpy - p["y"]) ** 2 > 0.7**2:
                ax.plot(
                    [p["x"], dpx],
                    [p["y"], dpy],
                    color="#6A6A6A",
                    lw=0.5,
                    alpha=0.55,
                    zorder=3,
                    solid_capstyle="round",
                )
            role = p.get("role") or ""
            # Single team colour for every node; bench players (not in the
            # starting XI) set apart by a square marker, not a colour.
            is_sub = bool(p.get("is_sub")) and not _all_sub
            is_gk = gk_player is not None and p is gk_player
            is_hub = hub_player is not None and p is hub_player
            marker = "s" if is_sub else "o"
            node_color = team_color
            rc_ring = "#F87171" if role == "red_card" else None
            ax.scatter(
                [dpx],
                [dpy],
                s=sz + 90,
                color=BG_DARK,
                marker=marker,
                alpha=0.88,
                zorder=4,
            )
            if is_hub:
                # Gold halo marks the team's most-involved player at a glance.
                ax.scatter(
                    [dpx],
                    [dpy],
                    s=sz + 220,
                    facecolor="none",
                    marker=marker,
                    edgecolor=C_GOLD,
                    lw=1.4,
                    alpha=0.75,
                    zorder=4,
                )
            if rc_ring:
                ax.scatter(
                    [dpx],
                    [dpy],
                    s=sz + 90,
                    color=rc_ring,
                    marker=marker,
                    alpha=0.45,
                    zorder=4,
                )
            # GK → thick gold edge; sub → white edge; outfield starter →
            # soft team-tint edge, matching the pass network's node identity.
            edge_c = (
                C_GOLD
                if is_gk
                else (TEXT_BR if is_sub else _blend_hex(team_color, "#ffffff", 0.45))
            )
            edge_w = 2.2 if is_gk else (1.5 if is_sub else 1.1)
            ax.scatter(
                [dpx],
                [dpy],
                s=sz,
                color=node_color,
                marker=marker,
                edgecolor=edge_c,
                lw=edge_w,
                alpha=0.92,
                zorder=5,
            )
            ax.text(
                dpx,
                dpy,
                str(p["display_id"]),
                ha="center",
                va="center",
                color="#ffffff",
                fontsize=6.6,
                fontweight="bold",
                family=FONT_MONO,
                zorder=6,
                path_effects=[pe.withStroke(linewidth=1.4, foreground=BG_DARK)],
            )
            # Labels route against the DISPLAY positions of every node (incl.
            # itself), sharing the same anti-overlap logic as the pass network.
            _draw_player_label_vertical(
                ax,
                disp_nodes[idx],
                idx,
                fontsize=label_fontsize,
                zorder=7,
                taken=taken_labels,
                all_nodes=disp_nodes,
                pitch_w=100.0,
                pitch_l=100.0,
                node_radii=disp_radii,
            )

    rows = [
        (
            f"#{p['display_id']}",
            p["name"].split()[-1] if p["name"] else "—",
            str(p["touches"]),
        )
        for p in players[:9]
    ]

    if players:
        avg_x = int(np.mean([p["x"] for p in players]))
        width_spread = int(max(p["y"] for p in players) - min(p["y"] for p in players))
        length_spread = int(max(p["x"] for p in players) - min(p["x"] for p in players))
        total_touches = sum(p["touches"] for p in players)
        insight = (
            f"{team_name}'s average shape sat at x≈{avg_x} (depth) with "
            f"a length spread of {length_spread} (defence-to-attack) and "
            f"a width spread of {width_spread} (touchline-to-touchline). "
            f"{players[0]['name'].split()[-1]} held the most touches "
            f"({players[0]['touches']})."
        )
    else:
        avg_x = width_spread = length_spread = total_touches = 0
        insight = f"{team_name} — no positional data."

    cards = [
        ("Avg X (Depth)", str(avg_x), team_color),
        ("Length Spread", str(length_spread), TEXT_BR),
        ("Width Spread", str(width_spread), TEXT_BR),
        ("Total Touches", str(total_touches), team_color),
        ("Players", str(len(players)), team_color),
    ]
    return render_pitch_overlay_v2(
        section="AVERAGE POSITIONS",
        title=f"{team_name} — Average Positions",
        subtitle="Each node at the player's average touch position · size "
        "= touches · square = substitute",
        hn=team_name,
        an=opp_name,
        score=str(score),
        footer_note="Shape signals defensive line height + width",
        team_color=team_color,
        draw_overlay=draw_overlay,
        sidebar_title="Most Active (touches)",
        sidebar_headers=["#", "PLAYER", "TOUCHES"],
        sidebar_rows=rows,
        sidebar_value_cols=[0.06, 0.26, 0.93],
        insight_text=insight,
        metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  BOX ENTRIES v2  (figs 32 / 33)
# ═════════════════════════════════════════════════════════════════════════
def make_box_entries_v2(events, info, team_id, team_color):
    """
    Box entry = pass or carry that ends inside the opponent box
    (end_x ≥ 83, 21 ≤ end_y ≤ 79) but starts outside.
    """
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name = an if is_home else hn
    score = info.get("score") or "—"

    BOX_X, BOX_Y_LO, BOX_Y_HI = 83, 21, 79
    sub = events[
        (events["team_id"] == team_id)
        & events["x"].notna()
        & events["y"].notna()
        & events["end_x"].notna()
        & events["end_y"].notna()
        & (events["outcome"] == "Successful")
    ]

    entries = []
    for _, r in sub.iterrows():
        sx, sy = float(r["x"]), float(r["y"])
        ex, ey = float(r["end_x"]), float(r["end_y"])
        in_box = ex >= BOX_X and BOX_Y_LO <= ey <= BOX_Y_HI
        from_out = not (sx >= BOX_X and BOX_Y_LO <= sy <= BOX_Y_HI)
        if in_box and from_out:
            kind = "pass" if bool(r.get("is_pass")) else "carry"
            channel = "left" if sy < 38 else ("centre" if sy <= 62 else "right")
            entries.append(
                {
                    "sx": sx,
                    "sy": sy,
                    "ex": ex,
                    "ey": ey,
                    "kind": kind,
                    "player": str(r.get("player") or "—"),
                    "channel": channel,
                }
            )

    def draw_overlay(ax):
        for e in entries:
            col = ACCENT_TEXT if e["kind"] == "pass" else "#22c55e"
            ax.annotate(
                "",
                xy=(e["ex"], e["ey"]),
                xytext=(e["sx"], e["sy"]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=col,
                    lw=1.2,
                    alpha=0.78,
                    connectionstyle="arc3,rad=0.10",
                ),
                zorder=4,
            )
        # legend chips bottom
        for x0, lbl, col in [(2, "Pass entry", C_GOLD), (32, "Carry entry", "#22c55e")]:
            ax.annotate(
                "",
                xy=(x0 + 8, -3),
                xytext=(x0, -3),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.4, alpha=0.9),
                clip_on=False,
            )
            ax.text(
                x0 + 9.5,
                -3,
                lbl,
                ha="left",
                va="center",
                color=TEXT_MAIN,
                fontsize=8,
                fontweight="bold",
                clip_on=False,
            )

    by_player = {}
    for e in entries:
        by_player[e["player"]] = by_player.get(e["player"], 0) + 1
    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    rows = [(p.split()[-1] if p else "—", str(c)) for p, c in top]

    n_pass = sum(1 for e in entries if e["kind"] == "pass")
    n_carry = sum(1 for e in entries if e["kind"] == "carry")
    n_left = sum(1 for e in entries if e["channel"] == "left")
    n_centre = sum(1 for e in entries if e["channel"] == "centre")
    n_right = sum(1 for e in entries if e["channel"] == "right")
    dom = max(
        [("left", n_left), ("centre", n_centre), ("right", n_right)],
        key=lambda kv: kv[1],
    )
    insight = (
        (
            f"{team_name} entered the box {len(entries)} times "
            f"({n_pass} via pass, {n_carry} via carry). The dominant channel "
            f"was the {dom[0]} ({dom[1]} entries)."
        )
        if entries
        else f"{team_name} — no successful box entries recorded."
    )

    cards = [
        ("Total Entries", str(len(entries)), C_GOLD),
        ("Pass", str(n_pass), team_color),
        ("Carry", str(n_carry), C_GOLD),
        ("Left / Centre", f"{n_left} / {n_centre}", team_color),
        ("Right", str(n_right), C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="BOX ENTRIES",
        title=f"{team_name} — Box Entries",
        subtitle="Arrows show every successful entry into the opposition box "
        "· gold = pass entry · green = carry entry",
        hn=team_name,
        an=opp_name,
        score=str(score),
        footer_note="Box = the 18-yard area",
        team_color=team_color,
        draw_overlay=draw_overlay,
        sidebar_title="Top Entry-Makers",
        sidebar_headers=["PLAYER", "ENTRIES"],
        sidebar_rows=rows,
        insight_text=insight,
        metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  HIGH TURNOVERS v2  (figs 36 / 37)
# ═════════════════════════════════════════════════════════════════════════
def make_high_turnovers_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name = an if is_home else hn
    score = info.get("score") or "—"

    sub = high_regain_events(events, team_id)

    points = []
    by_player = {}
    for _, r in sub.iterrows():
        x = float(_safe(r.get("x"), 70))
        y = float(_safe(r.get("y"), 50))
        p = str(_safe(r.get("player"), "—"))
        points.append((x, y, str(r.get("type"))))
        by_player[p] = by_player.get(p, 0) + 1

    def draw_overlay(ax):
        # Shade the high-zone (x >= 60) with a visible tint + dashed
        # boundary line, matching the Zone 14 treatment.
        ax.axvspan(60, 100, color=C_GOLD, alpha=0.10, zorder=0)
        ax.axvline(60, color=C_GOLD, lw=1.3, ls=(0, (3, 2)), alpha=0.7, zorder=1)
        ax.text(
            80,
            97,
            "HIGH ZONE",
            ha="center",
            va="top",
            color=C_GOLD,
            fontsize=7.5,
            fontweight="bold",
            family=FONT_MONO,
            alpha=0.95,
            zorder=2,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=BG_DARK,
                edgecolor="none",
                alpha=0.65,
            ),
        )
        for x, y, t in points:
            col = DEF_TYPE_COLORS.get(t, team_color)
            ax.scatter(
                [x],
                [y],
                s=150,
                facecolor=col,
                edgecolor=TEXT_BR,
                lw=1.0,
                alpha=0.92,
                zorder=4,
            )

    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    rows = [(p.split()[-1] if p else "—", str(c)) for p, c in top]

    leader = top[0][0].split()[-1] if top else "—"
    insight = (
        (
            f"{team_name} regained possession {len(points)} times in the "
            f"final 40 metres — the press's tangible reward. {leader} led "
            f"with {top[0][1]} high regains."
        )
        if top
        else f"{team_name} — no high regains recorded."
    )

    side = "home" if is_home else "away"
    advanced = team_advanced_metrics(events, info)[side]
    cards = [
        ("High Regains", str(len(points)), team_color),
        ("Transition Shots", str(advanced["transition_shots"]), TEXT_BR),
        ("Regain→Shot", f'{advanced["regain_to_shot_rate"]:.0f}%', TEXT_BR),
        ("Counterpress", str(advanced["counterpress_regains"]), TEXT_BR),
        ("Top Player", leader, team_color),
    ]
    return render_pitch_overlay_v2(
        section="HIGH REGAINS",
        title=f"{team_name} — High Regains",
        subtitle="Each dot = a possession regain inside the final 40m · "
        "colour = action type · gold band = high zone",
        hn=team_name,
        an=opp_name,
        score=str(score),
        footer_note="High regain = new open-play possession starting at x ≥ 60",
        team_color=team_color,
        draw_overlay=draw_overlay,
        sidebar_title="Top High-Pressers",
        sidebar_headers=["PLAYER", "REGAINS"],
        sidebar_rows=rows,
        insight_text=insight,
        metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  DANGER CREATION v2  (figs 10 / 11)
# ═════════════════════════════════════════════════════════════════════════
def make_danger_creation_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name = an if is_home else hn
    score = info.get("score") or "—"

    # Fixed, semantic colours for the three action types so they never blend
    # (a gold team's shots used to clash with the gold key-pass diamonds).
    KP_COL, ENTRY_COL, BC_RING, GOAL_COL = "#38BDF8", "#94A3B8", "#FFFFFF", "#22C55E"

    sub = events[events["team_id"] == team_id].reset_index(drop=True)
    shots, kps, entries = [], [], []
    by_player = {}
    big_chances = 0
    for i, r in sub.iterrows():
        x = float(_safe(r.get("x"), 50))
        y = float(_safe(r.get("y"), 50))
        p = str(_safe(r.get("player"), "—"))
        if bool(r.get("is_shot")):
            # Drop own goals: logged on the scorer's own team_id but not a
            # shot AT the opponent's goal (would sit at the wrong end).
            if (
                bool(r.get("is_goal"))
                and int(_safe(r.get("scoring_team"), team_id)) != team_id
            ):
                continue
            xg = float(_safe(r.get("xG"), 0) or 0)
            bc = bool(r.get("big_chance"))
            goal = (
                bool(r.get("is_goal"))
                and int(_safe(r.get("scoring_team"), team_id)) == team_id
            )
            shots.append({"i": i, "x": x, "y": y, "xg": xg, "bc": bc, "goal": goal})
            if bc:
                big_chances += 1
            by_player[p] = by_player.get(p, 0) + 1
        if bool(r.get("is_key_pass")):
            kps.append({"i": i, "x": x, "y": y})
            by_player[p] = by_player.get(p, 0) + 1
        # Box entry approximation
        if bool(r.get("is_pass")) and r.get("outcome") == "Successful":
            ex = float(_safe(r.get("end_x"), x))
            ey = float(_safe(r.get("end_y"), y))
            if ex >= 83 and 21 <= ey <= 79 and not (x >= 83 and 21 <= y <= 79):
                entries.append((x, y, ex, ey))

    # Chance-build links: pair each key pass with the next same-team shot
    # within a few events — the pass that actually led to the attempt.
    links = []
    for kp in kps:
        nxt = [s for s in shots if 0 < s["i"] - kp["i"] <= 6]
        if nxt:
            links.append((kp, min(nxt, key=lambda s: s["i"] - kp["i"])))

    def draw_overlay(ax):
        # Box entries — faint slate arrows in the background.
        for sx, sy, ex, ey in entries:
            ax.annotate(
                "",
                xy=(ex, ey),
                xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ENTRY_COL,
                    lw=0.7,
                    alpha=0.18,
                    connectionstyle="arc3,rad=0.12",
                ),
                zorder=2,
            )
        # Chance-build links — light gold curves from key pass to its shot.
        for kp, s in links:
            ax.annotate(
                "",
                xy=(s["x"], s["y"]),
                xytext=(kp["x"], kp["y"]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=C_GOLD,
                    lw=1.1,
                    alpha=0.45,
                    connectionstyle="arc3,rad=0.16",
                ),
                zorder=4,
            )
        # Key passes — blue diamonds.
        for kp in kps:
            ax.scatter(
                [kp["x"]],
                [kp["y"]],
                s=85,
                marker="D",
                facecolor=KP_COL,
                edgecolor="white",
                lw=1.0,
                alpha=0.9,
                zorder=5,
            )
        # Shots — team-colour circles sized by xG; big chance = white halo,
        # goal = green ring + gold star.
        for s in shots:
            sz = 70 + 300 * min(s["xg"] / 0.4, 1.0)
            if s["bc"]:
                ax.scatter(
                    [s["x"]],
                    [s["y"]],
                    s=sz + 150,
                    marker="o",
                    facecolor="none",
                    edgecolor=BC_RING,
                    lw=1.6,
                    alpha=0.9,
                    zorder=6,
                )
            ax.scatter(
                [s["x"]],
                [s["y"]],
                s=sz,
                marker="o",
                facecolor=team_color,
                edgecolor="white",
                lw=1.2,
                alpha=0.95,
                zorder=7,
            )
            if s["goal"]:
                ax.scatter(
                    [s["x"]],
                    [s["y"]],
                    s=sz + 220,
                    marker="o",
                    facecolor="none",
                    edgecolor=GOAL_COL,
                    lw=2.0,
                    zorder=7,
                )
                ax.scatter(
                    [s["x"]],
                    [s["y"]],
                    s=70,
                    marker="*",
                    facecolor=C_GOLD,
                    edgecolor=BG_DARK,
                    lw=0.8,
                    zorder=8,
                )

    def draw_legend(fig, px, py):
        # Legend sits inside the pitch's empty own-half (lower area) so it never
        # collides with the bottom metric strip. Faint backing keeps it legible
        # over the pitch lines.
        lax = fig.add_axes([0, 0, 1, 1])
        lax.set_axis_off()
        lax.set_xlim(0, 1)
        lax.set_ylim(0, 1)
        # Sit the legend low, along the pitch's own-goal strip (clear of the
        # attacking-half markers), on a solid backing so it never reads as
        # tangled up in the pitch lines.
        x = px + 0.028
        y = py + 0.045
        lax.add_patch(
            mpatches.FancyBboxPatch(
                (px + 0.015, y - 0.026),
                0.435,
                0.052,
                boxstyle="round,pad=0.0,rounding_size=0.006",
                transform=fig.transFigure,
                facecolor=BG_DARK,
                edgecolor=GRID_COL,
                lw=0.9,
                alpha=0.94,
                zorder=25,
            )
        )
        items = [
            ("o", team_color, "SHOT (xG)"),
            ("D", KP_COL, "KEY PASS"),
            (">", C_GOLD, "LED TO SHOT"),
            ("bc", None, "BIG CHANCE"),
            ("*", C_GOLD, "GOAL"),
        ]
        for mk, col, lbl in items:
            if mk == ">":
                lax.annotate(
                    "",
                    xy=(x + 0.016, y),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, alpha=0.75),
                    zorder=27,
                )
                x += 0.020
            elif mk == "bc":
                lax.scatter(
                    [x + 0.004],
                    [y],
                    s=120,
                    marker="o",
                    facecolor="none",
                    edgecolor=BC_RING,
                    lw=1.5,
                    zorder=27,
                )
                x += 0.014
            elif mk == "*":
                lax.scatter(
                    [x + 0.004],
                    [y],
                    s=130,
                    marker="*",
                    facecolor=col,
                    edgecolor=BG_DARK,
                    lw=0.6,
                    zorder=27,
                )
                x += 0.014
            else:
                lax.scatter(
                    [x + 0.004],
                    [y],
                    s=80,
                    marker=mk,
                    facecolor=col,
                    edgecolor="white",
                    lw=0.8,
                    zorder=27,
                )
                x += 0.014
            lax.text(
                x + 0.004,
                y,
                lbl,
                color=TEXT_DIM,
                fontsize=7.6,
                fontweight="bold",
                family=FONT_MONO,
                ha="left",
                va="center",
                zorder=27,
            )
            x += 0.010 + 0.0072 * len(lbl)

    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    rows = [(p.split()[-1] if p else "—", str(c)) for p, c in top]
    leader = top[0][0].split()[-1] if top else "—"
    n_goals = sum(1 for s in shots if s["goal"])
    insight = (
        (
            f"{team_name} produced {len(shots)} shots ({big_chances} big "
            f"chance{'s' if big_chances != 1 else ''}), {len(kps)} key passes and "
            f"{len(entries)} box entries. {leader} led danger creation; "
            f"{len(links)} shots came directly off a key pass."
        )
        if top
        else f"{team_name} — no danger actions recorded."
    )

    cards = [
        ("Shots", str(len(shots)), team_color),
        ("Big Chances", str(big_chances), BC_RING),
        ("Key Passes", str(len(kps)), KP_COL),
        ("Box Entries", str(len(entries)), ENTRY_COL),
        ("Top Creator", leader, C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="DANGER CREATION",
        title=f"{team_name} — Danger Creation",
        subtitle="Gold arrow = key pass that led to a shot · circle size = xG "
        "· white ring = big chance · star = goal",
        hn=team_name,
        an=opp_name,
        score=str(score),
        footer_note="Follow the gold arrows: how each dangerous shot was built",
        team_color=team_color,
        draw_overlay=draw_overlay,
        draw_legend=draw_legend,
        sidebar_title="Top Creators",
        sidebar_headers=["PLAYER", "ACT"],
        sidebar_rows=rows,
        insight_text=insight,
        metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  ZONE 14 + HALF-SPACES v2  (figs 14 / 15)
# ═════════════════════════════════════════════════════════════════════════
def make_zone14_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name = an if is_home else hn
    score = info.get("score") or "—"

    # Zone 14: x ∈ [70, 83], y ∈ [37, 63]
    # Half-spaces:  x ∈ [60, 95], y ∈ [22, 37] OR [63, 78]
    Z14 = lambda x, y: 70 <= x <= 83 and 37 <= y <= 63
    HSL = lambda x, y: 60 <= x <= 95 and 22 <= y < 37
    HSR = lambda x, y: 60 <= x <= 95 and 63 < y <= 78

    sub = events[
        (events["team_id"] == team_id) & events["x"].notna() & events["y"].notna()
    ]
    z14_pts, hs_pts = [], []
    by_player = {}
    for _, r in sub.iterrows():
        x = float(_safe(r.get("x"), 50))
        y = float(_safe(r.get("y"), 50))
        p = str(_safe(r.get("player"), "—"))
        if Z14(x, y):
            z14_pts.append((x, y, p))
            by_player[p] = by_player.get(p, 0) + 1
        elif HSL(x, y) or HSR(x, y):
            hs_pts.append((x, y, p))
            by_player[p] = by_player.get(p, 0) + 0.5

    # Grid-cell heatmap over the Zone14/half-space corridor: every cell
    # value is a REAL touch count from the event data (not a smoothed/
    # interpolated density estimate), matching the xT map's grid identity.
    # The bin edges are the ZONE BOUNDARIES THEMSELVES (not an arbitrary
    # column/row count), so the Zone14/half-space dashed outlines always run
    # exactly along a cell border — they can never slice through a number.
    xedges = np.array([60.0, 70.0, 83.0, 95.0])  # depth: HS-only | Zone14 | HS-only
    yedges = np.array([22.0, 37.0, 63.0, 78.0])  # width: HS-L | Zone14 band | HS-R
    GRID_COLS, GRID_ROWS = len(xedges) - 1, len(yedges) - 1
    corridor = sub[
        (sub["x"] >= xedges[0])
        & (sub["x"] <= xedges[-1])
        & (sub["y"] >= yedges[0])
        & (sub["y"] <= yedges[-1])
    ]
    cell_counts = np.zeros((GRID_COLS, GRID_ROWS))
    if not corridor.empty:
        cell_counts, _, _ = np.histogram2d(
            corridor["x"].to_numpy(dtype=float),
            corridor["y"].to_numpy(dtype=float),
            bins=[xedges, yedges],
        )
    grid_vmax = cell_counts.max() if cell_counts.size else 0

    def draw_overlay(ax):
        # `ax` here is the vertical-pitch proxy: scatter/text/annotate auto-
        # convert raw 0-100 pitch coordinates, but add_patch does NOT (it
        # forwards straight to the real axes, whose native units are
        # VP_W-wide / VP_L-tall, not 0-100). Every Rectangle must therefore
        # have its corners converted through _vp_xy by hand, or it lands
        # off-canvas — this was already silently true for the old Zone14/
        # half-space outline boxes below, just unnoticed since the scatter
        # dots and labels (which ARE converted) still read fine on their own.
        def _grid_rect(x0, y0, x1, y1, **kw):
            px0, py0 = _vp_xy(x0, y0)
            px1, py1 = _vp_xy(x1, y1)
            return mpatches.Rectangle((px0, py0), px1 - px0, py1 - py0, **kw)

        cmap = LinearSegmentedColormap.from_list(
            "z14grid",
            ["#0a0a0a", "#123524", "#1f6b3a", "#3bb35f", "#a8e063", "#FFC23C"],
        )
        for i in range(GRID_COLS):
            for j in range(GRID_ROWS):
                v = cell_counts[i, j]
                ratio = (v / grid_vmax) if grid_vmax else 0.0
                ax.add_patch(
                    _grid_rect(
                        xedges[i],
                        yedges[j],
                        xedges[i + 1],
                        yedges[j + 1],
                        facecolor=cmap(ratio),
                        edgecolor=BG_DARK,
                        lw=0.9,
                        alpha=0.92,
                        zorder=1,
                    )
                )
                if v > 0:
                    ccx = (xedges[i] + xedges[i + 1]) / 2
                    ccy = (yedges[j] + yedges[j + 1]) / 2
                    txt_col = "#0a0a0a" if ratio > 0.55 else "#e8e8e8"
                    ax.text(
                        ccx,
                        ccy,
                        str(int(v)),
                        ha="center",
                        va="center",
                        color=txt_col,
                        fontsize=9.5,
                        fontweight="bold",
                        family=FONT_MONO,
                        zorder=2,
                    )

        # Zone 14 outline == exactly the middle cell's own border (never
        # crosses a number). Label sits past the grid's depth range (outside
        # x:[60,95]) so it never lands on a numbered cell either.
        ax.add_patch(
            _grid_rect(
                70,
                37,
                83,
                63,
                facecolor="none",
                lw=2.0,
                edgecolor=C_GOLD,
                linestyle=(0, (3, 2)),
                zorder=4,
            )
        )
        ax.text(
            50.0,
            97.0,
            "ZONE 14",
            ha="center",
            color=C_GOLD,
            fontsize=8,
            fontweight="bold",
            alpha=1.0,
            family=FONT_MONO,
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=BG_DARK,
                edgecolor=C_GOLD,
                lw=0.6,
                alpha=0.85,
            ),
        )
        # Half-spaces == exactly the left/right column borders. Labels sit
        # past the grid's width range (outside y:[22,78], i.e. near the
        # touchlines) so they never land on a numbered cell either.
        ax.add_patch(
            _grid_rect(
                60,
                22,
                95,
                37,
                facecolor="none",
                lw=1.5,
                edgecolor="#22c55e",
                linestyle=(0, (3, 2)),
                zorder=4,
            )
        )
        ax.add_patch(
            _grid_rect(
                60,
                63,
                95,
                78,
                facecolor="none",
                lw=1.5,
                edgecolor="#22c55e",
                linestyle=(0, (3, 2)),
                zorder=4,
            )
        )
        ax.text(
            77.5,
            12.0,
            "HALF-SPACE (L)",
            ha="center",
            color="#22c55e",
            fontsize=7,
            fontweight="bold",
            alpha=1.0,
            family=FONT_MONO,
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=BG_DARK,
                edgecolor="none",
                alpha=0.75,
            ),
        )
        ax.text(
            77.5,
            88.0,
            "HALF-SPACE (R)",
            ha="center",
            color="#22c55e",
            fontsize=7,
            fontweight="bold",
            alpha=1.0,
            family=FONT_MONO,
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=BG_DARK,
                edgecolor="none",
                alpha=0.75,
            ),
        )

    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    rows = [(p.split()[-1] if p else "—", str(int(c))) for p, c in top]
    insight = (
        f"{team_name} touched the ball {len(z14_pts)} times in Zone 14 "
        f"and {len(hs_pts)} times in the half-spaces. Central-pocket "
        f"access is a leading indicator of chance creation."
    )
    cards = [
        ("Zone 14", str(len(z14_pts)), C_GOLD),
        ("Half-Spaces", str(len(hs_pts)), "#22c55e"),
        ("Total", str(len(z14_pts) + len(hs_pts)), TEXT_BR),
        ("Top Z14", (top[0][0].split()[-1] if top else "—"), team_color),
        ("Top Count", str(int(top[0][1])) if top else "0", team_color),
    ]
    return render_pitch_overlay_v2(
        section="ZONE 14 + HALF-SPACES",
        title=f"{team_name} — Zone 14 & Half-Spaces",
        subtitle="Grid cells show real touch counts · gold rectangle = "
        "Zone 14 (central pocket) · green = the flanking half-spaces",
        hn=team_name,
        an=opp_name,
        score=str(score),
        footer_note="More central-pocket touches → more chance creation",
        team_color=team_color,
        draw_overlay=draw_overlay,
        sidebar_title="Top Operators",
        sidebar_headers=["PLAYER", "ACT"],
        sidebar_rows=rows,
        insight_text=insight,
        metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  CROSSES v2  (figs 24 / 25)
# ═════════════════════════════════════════════════════════════════════════
def make_crosses_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name = an if is_home else hn
    score = info.get("score") or "—"

    sub = events[(events["team_id"] == team_id) & cross_mask(events)]

    crosses = []
    by_player = {}
    for _, r in sub.iterrows():
        ok = r.get("outcome") == "Successful"
        from_left = float(r["y"]) >= 78
        crosses.append(
            {
                "sx": float(r["x"]),
                "sy": float(r["y"]),
                "ex": float(r["end_x"]),
                "ey": float(r["end_y"]),
                "ok": ok,
                "from_left": from_left,
                "player": str(r.get("player") or "—"),
            }
        )
        if ok:
            p = str(r.get("player") or "—")
            by_player[p] = by_player.get(p, 0) + 1

    def draw_overlay(ax):
        for c in crosses:
            col = team_color if c["ok"] else "#64748b"
            alpha = 0.85 if c["ok"] else 0.35
            ax.annotate(
                "",
                xy=(c["ex"], c["ey"]),
                xytext=(c["sx"], c["sy"]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=col,
                    lw=1.3,
                    alpha=alpha,
                    connectionstyle="arc3,rad=0.10",
                ),
                zorder=4,
            )
        for x0, lbl, col in [
            (2, "Successful", team_color),
            (32, "Unsuccessful", "#64748b"),
        ]:
            ax.annotate(
                "",
                xy=(x0 + 8, -3),
                xytext=(x0, -3),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.4, alpha=0.9),
                clip_on=False,
            )
            ax.text(
                x0 + 9.5,
                -3,
                lbl,
                ha="left",
                va="center",
                color=TEXT_MAIN,
                fontsize=8,
                fontweight="bold",
                clip_on=False,
            )

    n = len(crosses)
    n_ok = sum(1 for c in crosses if c["ok"])
    n_left = sum(1 for c in crosses if c["from_left"])
    n_right = n - n_left
    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    rows = [(p.split()[-1] if p else "—", str(c)) for p, c in top]
    insight = (
        f"{team_name} attempted {n} crosses and completed {n_ok} "
        f"({(n_ok/n*100 if n else 0):.0f}%). Left flank: {n_left} · "
        f"right flank: {n_right}."
    )
    cards = [
        ("Total Crosses", str(n), C_GOLD),
        ("Successful", str(n_ok), team_color),
        ("Accuracy", f"{(n_ok/n*100 if n else 0):.0f}%", C_GOLD),
        ("Left", str(n_left), team_color),
        ("Right", str(n_right), C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="CROSSES",
        title=f"{team_name} — Crosses",
        subtitle="Solid arrows = successful crosses · faded = unsuccessful "
        "· flank reveals the side's wide-attack channel",
        hn=team_name,
        an=opp_name,
        score=str(score),
        footer_note="Cross = provider cross flag or qualifier · geometry only as fallback",
        team_color=team_color,
        draw_overlay=draw_overlay,
        sidebar_title="Top Crossers",
        sidebar_headers=["PLAYER", "OK"],
        sidebar_rows=rows,
        insight_text=insight,
        metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  PROGRESSIVE PASSES v2  (figs 22 / 23)
# ═════════════════════════════════════════════════════════════════════════
def make_progressive_passes_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name = an if is_home else hn
    score = info.get("score") or "—"

    sub = events[(events["team_id"] == team_id) & progressive_pass_mask(events)]
    progressives = []
    by_player = {}
    for _, r in sub.iterrows():
        sx, sy = float(r["x"]), float(r.get("y") or 50)
        ex, ey = float(r["end_x"]), float(r.get("end_y") or 50)
        progressives.append(
            {
                "sx": sx,
                "sy": sy,
                "ex": ex,
                "ey": ey,
                "player": str(r.get("player") or "—"),
            }
        )
        p = str(r.get("player") or "—")
        by_player[p] = by_player.get(p, 0) + 1

    def draw_overlay(ax):
        for p in progressives:
            ax.annotate(
                "",
                xy=(p["ex"], p["ey"]),
                xytext=(p["sx"], p["sy"]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=team_color,
                    lw=1.0,
                    alpha=0.55,
                    connectionstyle="arc3,rad=0.06",
                ),
                zorder=3,
            )
        # Highlight top 10 by gain
        top10 = sorted(progressives, key=lambda p: (p["ex"] - p["sx"]), reverse=True)[
            :10
        ]
        for p in top10:
            ax.annotate(
                "",
                xy=(p["ex"], p["ey"]),
                xytext=(p["sx"], p["sy"]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=C_GOLD,
                    lw=1.8,
                    alpha=0.95,
                    connectionstyle="arc3,rad=0.08",
                ),
                zorder=5,
            )

    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    rows = [(p.split()[-1] if p else "—", str(c)) for p, c in top]
    avg_gain = np.mean([p["ex"] - p["sx"] for p in progressives]) if progressives else 0
    insight = (
        f"{team_name} played {len(progressives)} progressive passes — the "
        f"forward-progress engine. Top 10 by raw gain are highlighted in "
        f"gold. Average gain per pass: {avg_gain:.1f} m of pitch."
    )
    cards = [
        ("Progressives", str(len(progressives)), C_GOLD),
        ("Avg Gain", f"{avg_gain:.1f}", team_color),
        ("Top 10 Pass", "10" if progressives else "0", C_GOLD),
        ("Top Player", (top[0][0].split()[-1] if top else "—"), team_color),
        ("Top Count", str(top[0][1]) if top else "0", C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="PROGRESSIVE PASSES",
        title=f"{team_name} — Progressive Passes",
        subtitle="Every pass that closed ≥25% of the distance to goal · "
        "gold arrows = top-10 by raw forward gain",
        hn=team_name,
        an=opp_name,
        score=str(score),
        footer_note="Progressive = forward pass past the threshold",
        team_color=team_color,
        draw_overlay=draw_overlay,
        sidebar_title="Top Progressors",
        sidebar_headers=["PLAYER", "PROG"],
        sidebar_rows=rows,
        insight_text=insight,
        metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  PASS MAP BY THIRD v2  (figs 19 / 20)
# ═════════════════════════════════════════════════════════════════════════
def make_pass_thirds_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name = an if is_home else hn
    score = info.get("score") or "—"

    sub = events[
        (events["team_id"] == team_id)
        & (events["is_pass"] == True)
        & events["x"].notna()
        & events["end_x"].notna()
    ]
    def_p, mid_p, att_p = [], [], []
    n_def_ok = n_mid_ok = n_att_ok = 0
    by_player = {}
    for _, r in sub.iterrows():
        sx = float(r["x"])
        sy = float(r.get("y") or 50)
        ex = float(r["end_x"])
        ey = float(r.get("end_y") or 50)
        ok = r.get("outcome") == "Successful"
        rec = (sx, sy, ex, ey, ok)
        if sx < 33:
            def_p.append(rec)
            n_def_ok += int(ok)
        elif sx < 67:
            mid_p.append(rec)
            n_mid_ok += int(ok)
        else:
            att_p.append(rec)
            n_att_ok += int(ok)
        if ok:
            p = str(r.get("player") or "—")
            by_player[p] = by_player.get(p, 0) + 1

    def draw_overlay(ax):
        def_col = "#475569" if IS_LIGHT_THEME else "#5A5A5A"
        att_col = "#B45309" if IS_LIGHT_THEME else C_GOLD
        n_all = len(def_p) + len(mid_p) + len(att_p)
        # Scale opacity down as volume grows, so a busy match (40+ passes)
        # doesn't collapse into a solid mass of overlapping arrows. Each
        # individual arrow stays readable; the *density* still communicates
        # volume through layering instead of through each line being loud.
        density_factor = max(0.35, min(1.0, 18 / max(n_all, 1)))
        for grp, col in [(def_p, def_col), (mid_p, team_color), (att_p, att_col)]:
            for sx, sy, ex, ey, ok in grp:
                a = (0.50 if ok else 0.16) * density_factor + (0.0 if ok else 0.0)
                a = max(a, 0.08)
                lw = 0.85 if ok else 0.55
                ax.annotate(
                    "",
                    xy=(ex, ey),
                    xytext=(sx, sy),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=col,
                        lw=lw,
                        alpha=a,
                        mutation_scale=5.5 if ok else 4.5,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=3,
                )
        # Third dividers — drawn with a backing strip so they stay visible
        # even when arrows are densely packed against them.
        div_col = "#3A3A3A" if IS_LIGHT_THEME else TEXT_DIM
        ax.axvline(33, color=div_col, lw=1.1, ls=(0, (4, 3)), alpha=0.75, zorder=2)
        ax.axvline(67, color=div_col, lw=1.1, ls=(0, (4, 3)), alpha=0.75, zorder=2)
        for tx, lbl in [(16.5, "DEF"), (50, "MID"), (83, "ATT")]:
            ax.text(
                tx,
                96.5,
                lbl,
                color=TEXT_BR,
                fontsize=8,
                fontweight="bold",
                family=FONT_MONO,
                ha="center",
                va="top",
                zorder=5,
                bbox=dict(
                    boxstyle="round,pad=0.22",
                    facecolor=BG_DARK,
                    edgecolor=GRID_COL,
                    linewidth=0.8,
                    alpha=0.88,
                ),
            )

    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    rows = [(p.split()[-1] if p else "—", str(c)) for p, c in top]
    n_total = len(def_p) + len(mid_p) + len(att_p)
    insight = (
        f"{team_name} played {n_total} passes — {len(def_p)} from the "
        f"defensive third, {len(mid_p)} from the middle, "
        f"{len(att_p)} from the attacking third. Final-third volume is "
        f"the cleanest read of break-down activity."
    )
    cards = [
        ("Total Passes", str(n_total), TEXT_BR),
        ("Defensive 3rd", str(len(def_p)), TEXT_BR),
        ("Middle 3rd", str(len(mid_p)), team_color),
        ("Attacking 3rd", str(len(att_p)), C_GOLD),
        ("Att 3rd Acc.", f"{(n_att_ok/len(att_p)*100 if att_p else 0):.0f}%", C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="PASS MAP BY THIRD",
        title=f"{team_name} — Pass Map by Third",
        subtitle="Grey = defensive-third passes · team colour = middle "
        "third · gold = attacking third",
        hn=team_name,
        an=opp_name,
        score=str(score),
        footer_note="Final-third volume signals break-down efficiency",
        team_color=team_color,
        draw_overlay=draw_overlay,
        sidebar_title="Top Passers",
        sidebar_headers=["PLAYER", "OK"],
        sidebar_rows=rows,
        insight_text=insight,
        metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  PASS TARGET ZONES v2  (figs 38 / 39)
# ═════════════════════════════════════════════════════════════════════════
def make_pass_target_zones_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name = an if is_home else hn
    score = info.get("score") or "—"

    sub = events[
        (events["team_id"] == team_id)
        & (events["is_pass"] == True)
        & (events["outcome"] == "Successful")
        & events["end_x"].notna()
        & events["end_y"].notna()
    ]

    rows_n, cols_n = 6, 8
    cell_w = 100 / cols_n
    cell_h = 100 / rows_n
    grid = np.zeros((rows_n, cols_n))
    for _, r in sub.iterrows():
        ex = float(r["end_x"])
        ey = float(r["end_y"])
        c = min(int(ex // cell_w), cols_n - 1)
        rr = min(int(ey // cell_h), rows_n - 1)
        grid[rr, c] += 1

    def draw_overlay(ax):
        from matplotlib.colors import LinearSegmentedColormap as _LCM

        cmap = _LCM.from_list("pt", ["#050505", team_color])
        ax.imshow(
            grid,
            extent=[0, 100, 0, 100],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            alpha=0.78,
            zorder=1,
        )
        # Top-3 cells get numeric label
        flat = [(grid[r, c], r, c) for r in range(rows_n) for c in range(cols_n)]
        for v, r, c in sorted(flat, reverse=True)[:5]:
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            ax.text(
                cx,
                cy,
                f"{int(v)}",
                ha="center",
                va="center",
                color=TEXT_BR,
                fontsize=10,
                fontweight="bold",
                path_effects=shadow(2),
                zorder=4,
            )

    total = int(grid.sum())
    flat = [(grid[r, c], r, c) for r in range(rows_n) for c in range(cols_n)]
    flat.sort(reverse=True)
    hot = flat[0]
    hot_zone = (
        "attacking"
        if hot[2] >= cols_n * 2 / 3
        else ("middle" if hot[2] >= cols_n / 3 else "defensive")
    ) + " third"
    insight = (
        f"{team_name} found targets {total} times. The hottest receiving "
        f"zone was in the {hot_zone} ({int(hot[0])} passes landed there)."
    )
    cards = [
        ("Targets Found", str(total), C_GOLD),
        ("Hottest Zone", hot_zone.title()[:10], team_color),
        ("Top Cell", str(int(hot[0])), C_GOLD),
        (
            "Att 3rd %",
            f"{(int(grid[:, cols_n*2//3:].sum())/total*100 if total else 0):.0f}%",
            team_color,
        ),
        ("Cells > 0", str(int((grid > 0).sum())), C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="PASS TARGET ZONES",
        title=f"{team_name} — Pass Target Zones",
        subtitle="Heatmap of where successful passes landed · top-5 cells "
        "show the raw count of passes received",
        hn=team_name,
        an=opp_name,
        score=str(score),
        footer_note="Where the team wanted the ball to arrive",
        team_color=team_color,
        draw_overlay=draw_overlay,
        sidebar_title="Receiving-Zone Notes",
        sidebar_headers=["WHERE", "PASSES"],
        sidebar_rows=[
            ("Att third", str(int(grid[:, cols_n * 2 // 3 :].sum()))),
            ("Mid third", str(int(grid[:, cols_n // 3 : cols_n * 2 // 3].sum()))),
            ("Def third", str(int(grid[:, : cols_n // 3].sum()))),
            ("Top half (Y)", str(int(grid[rows_n // 2 :, :].sum()))),
            ("Bottom half", str(int(grid[: rows_n // 2, :].sum()))),
        ],
        insight_text=insight,
        metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  BALL TOUCHES v2  (fig 18 — shared, uses both teams)
# ═════════════════════════════════════════════════════════════════════════
def make_ball_touches_v2(
    events,
    info,
    *,
    section="BALL TOUCHES",
    title_label="Ball Touches",
    insight_intro=None,
):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    hid = info.get("home_id")
    aid = info.get("away_id")
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)

    rows_n, cols_n = 6, 10
    cell_w = 100 / cols_n
    cell_h = 100 / rows_n
    grid_h = np.zeros((rows_n, cols_n))
    grid_a = np.zeros((rows_n, cols_n))
    sub = events[events["x"].notna() & events["y"].notna()]
    for _, r in sub.iterrows():
        x = float(r["x"])
        y = float(r["y"])
        c = min(int(x // cell_w), cols_n - 1)
        rr = min(int(y // cell_h), rows_n - 1)
        if r.get("team_id") == hid:
            grid_h[rr, c] += 1
        elif r.get("team_id") == aid:
            grid_a[rr, c] += 1

    def draw_overlay(ax):
        # Diff grid: positive = home, negative = away
        diff = grid_h - grid_a
        from matplotlib.colors import LinearSegmentedColormap as _LCM

        cmap = _LCM.from_list("dom", [ac, MID_BG, hc])
        vmax = max(abs(diff).max(), 1)
        ax.imshow(
            diff,
            extent=[0, 100, 0, 100],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            alpha=0.65,
            zorder=1,
        )
        # Number on EVERY cell that has at least one touch — top 5 get a
        # bigger, brighter label, the rest stay smaller and dimmer so the
        # eye still picks the hot cells first.
        flat = [
            (grid_h[r, c] + grid_a[r, c], r, c)
            for r in range(rows_n)
            for c in range(cols_n)
        ]
        top5 = {(r, c) for _, r, c in sorted(flat, reverse=True)[:5]}
        for v, r, c in flat:
            if v <= 0:
                continue
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            is_top = (r, c) in top5
            ax.text(
                cx,
                cy,
                f"{int(v)}",
                ha="center",
                va="center",
                color=TEXT_BR if is_top else TEXT_DIM,
                fontsize=9 if is_top else 7,
                fontweight="bold",
                family=FONT_MONO,
                zorder=4,
            )

    n_h = int(grid_h.sum())
    n_a = int(grid_a.sum())
    diff = n_h - n_a
    leader = hn if diff > 0 else an
    insight = (
        f"{leader} touched the ball {abs(diff)} more times overall. "
        f"Heatmap shows where each side dominated possession — the "
        f"colour at each cell points to the team with more touches there."
    )
    cards = [
        (f"{hn[:14]} Touches", str(n_h), hc),
        ("Total", str(n_h + n_a), TEXT_BR),
        (f"{an[:14]} Touches", str(n_a), ac),
        ("Diff", f"{'+' if diff >= 0 else ''}{diff}", TEXT_BR),
        ("Leader", leader[:10], hc if diff > 0 else ac),
    ]
    return render_pitch_overlay_v2(
        section=section,
        title=f"{hn} vs {an} — {title_label}",
        subtitle="Each cell coloured by which team had more touches there "
        "· top-5 cells show the raw combined touch count",
        hn=hn,
        an=an,
        score=str(score),
        footer_note="Where the game actually got played",
        team_color=hc,
        draw_overlay=draw_overlay,
        sidebar_title="Touch Distribution",
        sidebar_headers=["WHERE", "DIFF"],
        sidebar_rows=[
            (
                "Att 3rd (H–A)",
                f"{int(grid_h[:, cols_n*2//3:].sum() - grid_a[:, cols_n*2//3:].sum()):+d}",
            ),
            (
                "Mid 3rd (H–A)",
                f"{int(grid_h[:, cols_n//3:cols_n*2//3].sum() - grid_a[:, cols_n//3:cols_n*2//3].sum()):+d}",
            ),
            (
                "Def 3rd (H–A)",
                f"{int(grid_h[:, :cols_n//3].sum() - grid_a[:, :cols_n//3].sum()):+d}",
            ),
            ("Total (H)", str(n_h)),
            ("Total (A)", str(n_a)),
        ],
        insight_text=insight,
        metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  TERRITORIAL CONTROL v2 + DOMINATING ZONE v2 (figs 17 / 33)
# ═════════════════════════════════════════════════════════════════════════
def make_territorial_v2(events, info):
    """Same engine as ball touches but framed as territorial control."""
    return make_ball_touches_v2(
        events,
        info,
        section="TERRITORIAL CONTROL",
        title_label="Territorial Control",
    )


def make_dominating_zone_v2(events, info):
    """Same engine but framed as 'who dominated each zone'."""
    return make_ball_touches_v2(
        events,
        info,
        section="DOMINATING ZONE",
        title_label="Dominating Zone",
    )


# ═════════════════════════════════════════════════════════════════════════
#  GENERIC BAR-COMPARISON v2 — used by Shot Comparison / xG Summary /
#  Defensive Summary (figs 9, 13, 30).
# ═════════════════════════════════════════════════════════════════════════
def render_bar_compare_v2(
    *,
    section,
    title,
    subtitle,
    hn,
    an,
    score,
    footer_note,
    hc,
    ac,
    rows,
    insight_text,
    metric_cards,
):
    """
    rows: list of (metric_label, home_value, away_value)
    """
    fig = plt.figure(figsize=(15, 9.5), facecolor=BG_DARK)
    chrome(
        fig,
        section=section,
        title=title,
        subtitle=subtitle,
        hn=hn,
        an=an,
        score=score,
        footer_note=footer_note,
    )

    ax = panel_card(
        fig, 0.04, 0.20, 0.62, 0.62, title="Side-by-side", accent=C_GOLD, body=False
    )
    ax.set_facecolor(BG_MID)
    ax.text(0.012, 0.965, "●", color=hc, fontsize=13, transform=ax.transAxes, va="top")
    ax.text(
        0.034,
        0.965,
        hn.upper(),
        color=hc,
        fontsize=9.5,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax.transAxes,
        va="top",
    )
    ax.text(0.012, 0.895, "●", color=ac, fontsize=13, transform=ax.transAxes, va="top")
    ax.text(
        0.034,
        0.895,
        an.upper(),
        color=ac,
        fontsize=9.5,
        fontweight="bold",
        family=FONT_MONO,
        transform=ax.transAxes,
        va="top",
    )

    # A value may be a plain number, or a (bar_height, display_label) tuple so
    # a row like a duel can scale its bar by "won" while printing "won/total".
    def _bar_val(v):
        return float(v[0]) if isinstance(v, (tuple, list)) else float(v)

    def _bar_lbl(v):
        return str(v[1]) if isinstance(v, (tuple, list)) else _fmt_num(v)

    n = len(rows)
    pos = np.arange(n)
    h_vals = [_bar_val(r[1]) for r in rows]
    a_vals = [_bar_val(r[2]) for r in rows]
    h_lbls = [_bar_lbl(r[1]) for r in rows]
    a_lbls = [_bar_lbl(r[2]) for r in rows]
    labels = [r[0] for r in rows]
    w = 0.38
    ymax = max(h_vals + a_vals + [1]) * 1.9
    ax.set_xlim(-0.6, n - 0.4)
    # Trim the empty band under the baseline so the category names sit right
    # beneath their own columns (easier to read which label owns which bar).
    ax.set_ylim(-ymax * 0.02, ymax)
    for y in np.linspace(0, ymax, 6):
        ax.axhline(y, color=GRID_SOFT, lw=0.8, alpha=1.0, zorder=0)
    ax.bar(pos - w / 2, h_vals, w, color=hc, lw=0, zorder=2)
    ax.bar(pos + w / 2, a_vals, w, color=ac, lw=0, zorder=2)
    for i in range(n):
        ax.text(
            i - w / 2,
            h_vals[i] + ymax * 0.018,
            h_lbls[i],
            ha="center",
            va="bottom",
            color=TEXT_BR,
            fontsize=9.5,
            fontweight="bold",
            family=FONT_MONO,
        )
        ax.text(
            i + w / 2,
            a_vals[i] + ymax * 0.018,
            a_lbls[i],
            ha="center",
            va="bottom",
            color=TEXT_BR,
            fontsize=9.5,
            fontweight="bold",
            family=FONT_MONO,
        )
    ax.set_xticks(pos)
    ax.set_xticklabels(
        [_wrap_axis_label(x, 12) for x in labels],
        color=TEXT_DIM,
        fontsize=9.0,
        fontweight="bold",
        family=FONT_SANS,
        linespacing=1.2,
    )
    # Tuck the category names close under the baseline (raised up) rather than
    # leaving a wide gap between the bars and their labels.
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks([])
    ax.axhline(0, color=GRID_COL, lw=1.0, alpha=1.0, zorder=1)
    for sp in ["top", "right", "left", "bottom"]:
        ax.spines[sp].set_visible(False)

    key_insight(fig, 0.69, 0.30, 0.27, 0.50, text=insight_text, wrap=34)
    metric_strip(fig, cards=metric_cards)
    return fig


def _fmt_num(v):
    try:
        f = float(v)
        if f.is_integer():
            return str(int(f))
        return f"{f:.2f}"
    except Exception:
        return str(v)


def make_shot_comparison_v2(events, info, xg_data):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)
    h = (xg_data or {}).get(hn, {})
    a = (xg_data or {}).get(an, {})
    rows = [
        ("Total Shots", h.get("shots", 0), a.get("shots", 0)),
        ("On Target", h.get("on_target", 0), a.get("on_target", 0)),
        ("Big Chances", h.get("big_chances", 0), a.get("big_chances", 0)),
        ("xG", float(h.get("xG", 0) or 0), float(a.get("xG", 0) or 0)),
        ("xGoT", float(h.get("xGoT", 0) or 0), float(a.get("xGoT", 0) or 0)),
    ]
    diff = float(h.get("xG", 0) or 0) - float(a.get("xG", 0) or 0)
    leader = hn if diff > 0 else an
    insight = (
        f"{leader} produced the stronger shooting profile — leading on "
        f"{sum(1 for r in rows if r[1] > r[2])} of the {len(rows)} "
        f"metrics. Total xG difference: "
        f"{'+' if diff >= 0 else ''}{diff:.2f}."
    )
    cards = [
        (f"{hn[:14]} xG", _fmt_num(h.get("xG", 0)), hc),
        ("xG Diff", f"{'+' if diff >= 0 else ''}{diff:.2f}", TEXT_BR),
        (f"{an[:14]} xG", _fmt_num(a.get("xG", 0)), ac),
        ("Total Shots", str(h.get("shots", 0) + a.get("shots", 0)), C_GOLD),
        ("Total OT", str(h.get("on_target", 0) + a.get("on_target", 0)), C_GOLD),
    ]
    return render_bar_compare_v2(
        section="SHOT COMPARISON",
        title=f"{hn} vs {an} — Shot Comparison",
        subtitle="Five headline shooting metrics side-by-side · gold "
        "label = the metric leader",
        hn=hn,
        an=an,
        score=str(score),
        footer_note="Read top-to-bottom: who created the better profile?",
        hc=hc,
        ac=ac,
        rows=rows,
        insight_text=insight,
        metric_cards=cards,
    )


def make_xg_summary_v2(events, info, xg_data):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)
    h = (xg_data or {}).get(hn, {})
    a = (xg_data or {}).get(an, {})
    rows = [
        ("xG", float(h.get("xG", 0) or 0), float(a.get("xG", 0) or 0)),
        ("xGoT", float(h.get("xGoT", 0) or 0), float(a.get("xGoT", 0) or 0)),
        ("Goals", h.get("goals", 0), a.get("goals", 0)),
        ("Big Chances", h.get("big_chances", 0), a.get("big_chances", 0)),
    ]
    h_xg = float(h.get("xG", 0) or 0)
    a_xg = float(a.get("xG", 0) or 0)
    h_g = h.get("goals", 0)
    a_g = a.get("goals", 0)
    over_h = h_g - h_xg
    over_a = a_g - a_xg
    insight = (
        f"xG: {hn} {h_xg:.2f} vs {an} {a_xg:.2f}. Finishing performance: "
        f"{hn} {'+' if over_h >= 0 else ''}{over_h:.2f} vs xG, "
        f"{an} {'+' if over_a >= 0 else ''}{over_a:.2f}. "
        f"xGoT shows post-shot finishing quality."
    )
    cards = [
        (f"{hn[:14]} xG", f"{h_xg:.2f}", hc),
        (f"{hn[:14]} Goals", str(h_g), C_GOLD),
        ("Goals - xG (H/A)", f"{over_h:+.1f}/{over_a:+.1f}", C_GOLD),
        (f"{an[:14]} Goals", str(a_g), C_GOLD),
        (f"{an[:14]} xG", f"{a_xg:.2f}", ac),
    ]
    return render_bar_compare_v2(
        section="xG / xGoT SUMMARY",
        title=f"{hn} vs {an} — xG and xGoT",
        subtitle="xG measures pre-shot quality · xGoT measures post-shot "
        "placement & power · gap to goals = finishing variance",
        hn=hn,
        an=an,
        score=str(score),
        footer_note="Below xG = wasteful · above xG = clinical",
        hc=hc,
        ac=ac,
        rows=rows,
        insight_text=insight,
        metric_cards=cards,
    )


def make_defensive_summary_v2(events, info):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    hid = info.get("home_id")
    aid = info.get("away_id")
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)
    advanced = team_advanced_metrics(events, info)
    home_advanced = advanced["home"]
    away_advanced = advanced["away"]

    def _count(team_id, type_name):
        return int(
            ((events["team_id"] == team_id) & (events["type"] == type_name)).sum()
        )

    rows = [
        ("Tackles", _count(hid, "Tackle"), _count(aid, "Tackle")),
        ("Interceptions", _count(hid, "Interception"), _count(aid, "Interception")),
        ("Clearances", _count(hid, "Clearance"), _count(aid, "Clearance")),
        (
            "Blocks",
            _blocked_shots_for_team(events, info, hid),
            _blocked_shots_for_team(events, info, aid),
        ),
        (
            "Provider recoveries",
            home_advanced["provider_recoveries"],
            away_advanced["provider_recoveries"],
        ),
        ("Fouls", fouls_committed_count(events, hid), fouls_committed_count(events, aid)),
    ]
    # Totals & "top type" use only the six core actions above — duels are shown
    # as extra bars but kept OUT of the total, since a duel win overlaps with a
    # tackle already counted (double-counting would inflate the total).
    h_total = sum(r[1] for r in rows)
    a_total = sum(r[2] for r in rows)

    # Duels: bar height = won, printed as "won/total". Aerial and ground.
    h_aw, h_at, h_gw, h_gt = _compute_duels(events, hid)
    a_aw, a_at, a_gw, a_gt = _compute_duels(events, aid)
    duel_rows = [
        (
            "Possession regains",
            home_advanced["possession_regains"],
            away_advanced["possession_regains"],
        ),
        (
            "High regains",
            home_advanced["high_regains"],
            away_advanced["high_regains"],
        ),
        ("Aerial duels", (h_aw, f"{h_aw}/{h_at}"), (a_aw, f"{a_aw}/{a_at}")),
        ("Ground duels", (h_gw, f"{h_gw}/{h_gt}"), (a_gw, f"{a_gw}/{a_gt}")),
    ]

    leader = hn if h_total > a_total else an
    insight = (
        f"{leader} did more defensive work overall ({max(h_total, a_total)} "
        f"vs {min(h_total, a_total)}). Tackles + interceptions describe duels; "
        f"provider recoveries are feed events, while possession regains are "
        f"inferred from control changes. Duel "
        f"bars show won/contested (aerial + ground)."
    )
    cards = [
        (f"{hn[:14]} Total", str(h_total), hc),
        ("Diff", f"{'+' if h_total - a_total >= 0 else ''}{h_total - a_total}", C_GOLD),
        (f"{an[:14]} Total", str(a_total), ac),
        ("Aerials W (H/A)", f"{h_aw}/{a_aw}", C_GOLD),
        ("Ground W (H/A)", f"{h_gw}/{a_gw}", C_GOLD),
    ]
    return render_bar_compare_v2(
        section="DEFENSIVE SUMMARY",
        title=f"{hn} vs {an} — Defensive Summary",
        subtitle="Provider actions + inferred possession regains + duels "
        "(won/contested)",
        hn=hn,
        an=an,
        score=str(score),
        footer_note="Provider recovery = feed event · possession regain = "
        "inferred control change",
        hc=hc,
        ac=ac,
        rows=rows + duel_rows,
        insight_text=insight,
        metric_cards=cards,
    )


def make_transition_summary_v2(events, info):
    """Compare canonical attacking-transition and pressing outcomes."""
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)
    advanced = team_advanced_metrics(events, info)
    home = advanced["home"]
    away = advanced["away"]

    rows = [
        ("Attacking transitions", home["transitions"], away["transitions"]),
        ("Transition shots", home["transition_shots"], away["transition_shots"]),
        ("Transition goals", home["transition_goals"], away["transition_goals"]),
        (
            "Transition box entries",
            home["transition_box_entries"],
            away["transition_box_entries"],
        ),
        ("High regains", home["high_regains"], away["high_regains"]),
        (
            "Counterpress regains",
            home["counterpress_regains"],
            away["counterpress_regains"],
        ),
    ]
    leader = hn if home["transition_shots"] >= away["transition_shots"] else an
    insight = (
        f"{leader} produced more shots from transition. A transition starts "
        "after an open-play regain and must progress at least 20 pitch units, "
        "reach the final third or box, or create a shot within 12 seconds."
    )
    cards = [
        (f"{hn[:12]} xG", f'{home["transition_xG"]:.2f}', hc),
        (f"{an[:12]} xG", f'{away["transition_xG"]:.2f}', ac),
        (
            "Shot Rate H/A",
            f'{home["transition_shot_rate"]:.0f}/{away["transition_shot_rate"]:.0f}%',
            C_GOLD,
        ),
        (
            "Avg Progress H/A",
            f'{home["avg_transition_progress"]:.0f}/{away["avg_transition_progress"]:.0f}',
            C_GOLD,
        ),
        (
            "Transition xT H/A",
            f'{home["transition_xT"]:.2f}/{away["transition_xT"]:.2f}',
            C_GOLD,
        ),
    ]
    return render_bar_compare_v2(
        section="TRANSITIONS",
        title=f"{hn} vs {an} — Transition Performance",
        subtitle="Open-play regains converted into fast territorial or shooting outcomes",
        hn=hn,
        an=an,
        score=str(score),
        footer_note="Window: first 12 seconds of the same possession · restarts excluded",
        hc=hc,
        ac=ac,
        rows=rows,
        insight_text=insight,
        metric_cards=cards,
    )


def make_advanced_metrics_summary_v2(events, info):
    """Compare the complete canonical advanced-metric set."""
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)
    advanced = team_advanced_metrics(events, info)
    home = advanced["home"]
    away = advanced["away"]

    rows = [
        ("Field tilt %", home["field_tilt"], away["field_tilt"]),
        ("Deep completions", home["deep_completions"], away["deep_completions"]),
        (
            "Build-up success %",
            home["build_up_success_rate"],
            away["build_up_success_rate"],
        ),
        (
            "Final-third entry efficiency %",
            home["final_third_entry_efficiency"],
            away["final_third_entry_efficiency"],
        ),
        (
            "Box entry → shot %",
            home["box_entry_to_shot_rate"],
            away["box_entry_to_shot_rate"],
        ),
        ("Sequence xT", home["sequence_xT"], away["sequence_xT"]),
        ("Directness %", home["directness"], away["directness"]),
        (
            "Counterpress success %",
            home["counterpress_success_rate"],
            away["counterpress_success_rate"],
        ),
        (
            "Rest-defence vulnerability %",
            home["rest_defence_vulnerability"],
            away["rest_defence_vulnerability"],
        ),
    ]

    sequence = player_sequence_metrics(events)

    def _top_sequence(team_id, metric):
        candidates = []
        for player, values in sequence.items():
            player_events = events[events["player"].astype(str) == str(player)]
            if player_events.empty or player_events["team_id"].dropna().empty:
                continue
            if player_events["team_id"].dropna().mode().iloc[0] == team_id:
                candidates.append((player, float(values.get(metric, 0.0))))
        return max(candidates, key=lambda item: item[1]) if candidates else ("—", 0.0)

    home_chain = _top_sequence(info.get("home_id"), "xGChain")
    away_chain = _top_sequence(info.get("away_id"), "xGChain")
    home_buildup = _top_sequence(info.get("home_id"), "xGBuildup")
    away_buildup = _top_sequence(info.get("away_id"), "xGBuildup")
    cards = [
        ("Home xGChain", f"{home_chain[0].split()[-1]} {home_chain[1]:.2f}", hc),
        ("Away xGChain", f"{away_chain[0].split()[-1]} {away_chain[1]:.2f}", ac),
        (
            "Home xGBuildup",
            f"{home_buildup[0].split()[-1]} {home_buildup[1]:.2f}",
            hc,
        ),
        (
            "Away xGBuildup",
            f"{away_buildup[0].split()[-1]} {away_buildup[1]:.2f}",
            ac,
        ),
        (
            "Seq xT / Poss H-A",
            f'{home["sequence_xT_per_possession"]:.2f}-{away["sequence_xT_per_possession"]:.2f}',
            C_GOLD,
        ),
    ]
    insight = (
        "These metrics separate territory, progression efficiency, sequence value, "
        "counterpressing and protection after advanced attacks. Rest-defence "
        "vulnerability is the only row where lower is better."
    )
    return render_bar_compare_v2(
        section="ADVANCED METRICS",
        title=f"{hn} vs {an} — Advanced Team Metrics",
        subtitle="Canonical possession-sequence metrics · lower rest-defence vulnerability is better",
        hn=hn,
        an=an,
        score=str(score),
        footer_note="xGChain/xGBuildup are non-penalty player sequence credits",
        hc=hc,
        ac=ac,
        rows=rows,
        insight_text=insight,
        metric_cards=cards,
    )


def make_game_state_summary_v2(events, info):
    """Compare team output while leading, drawing, and trailing."""
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)
    advanced = team_advanced_metrics(events, info)
    home = advanced["home"]["game_state_splits"]
    away = advanced["away"]["game_state_splits"]

    rows = []
    state_labels = {"leading": "Leading", "drawing": "Drawing", "trailing": "Trailing"}
    for state in ("leading", "drawing", "trailing"):
        label = state_labels[state]
        rows.extend(
            [
                (f"{label} shots", home[state]["shots"], away[state]["shots"]),
                (f"{label} xG", home[state]["xG"], away[state]["xG"]),
                (
                    f"{label} transitions",
                    home[state]["transitions"],
                    away[state]["transitions"],
                ),
            ]
        )
    cards = [
        (
            "Leading Poss H-A",
            f'{home["leading"]["possessions"]}-{away["leading"]["possessions"]}',
            C_GOLD,
        ),
        (
            "Drawing Poss H-A",
            f'{home["drawing"]["possessions"]}-{away["drawing"]["possessions"]}',
            C_GOLD,
        ),
        (
            "Trailing Poss H-A",
            f'{home["trailing"]["possessions"]}-{away["trailing"]["possessions"]}',
            C_GOLD,
        ),
        (
            "Leading xT H-A",
            f'{home["leading"]["sequence_xT"]:.2f}-{away["leading"]["sequence_xT"]:.2f}',
            C_GOLD,
        ),
        (
            "Trailing xT H-A",
            f'{home["trailing"]["sequence_xT"]:.2f}-{away["trailing"]["sequence_xT"]:.2f}',
            C_GOLD,
        ),
    ]
    return render_bar_compare_v2(
        section="GAME STATE",
        title=f"{hn} vs {an} — Game-State Splits",
        subtitle="Possession output measured against the score before each sequence began",
        hn=hn,
        an=an,
        score=str(score),
        footer_note="Shootout goals excluded · own goals credited to the benefiting team",
        hc=hc,
        ac=ac,
        rows=rows,
        insight_text=(
            "Game-state splits show whether volume came from controlling a lead, "
            "breaking a draw, or chasing the score. Compare rates only where each "
            "team had enough possessions in that state."
        ),
        metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  xT PER MINUTE v2  (fig 21 — diverging bars)
# ═════════════════════════════════════════════════════════════════════════
def make_xt_per_minute_v2(events, info):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    hid = info.get("home_id")
    aid = info.get("away_id")
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)

    if "xT" not in events.columns:
        # Fallback empty visual
        fig = plt.figure(figsize=(15, 9.5), facecolor=BG_DARK)
        chrome(
            fig,
            section="XT PER MINUTE",
            title=f"{hn} vs {an} — xT per Minute",
            subtitle="No xT data in this dataset",
            hn=hn,
            an=an,
            score=str(score),
            footer_note="—",
        )
        ax = fig.add_axes([0.05, 0.30, 0.92, 0.5])
        ax.set_facecolor(BG_MID)
        for s in ax.spines.values():
            s.set_edgecolor(GRID_COL)
            s.set_linewidth(1.0)
        ax.text(
            0.5,
            0.5,
            "No xT data",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=14,
            family=FONT_SANS,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    xt = events[
        events["xT"].notna() & (events["xT"] > 0) & (events["outcome"] == "Successful")
    ].copy()
    h_min = xt[xt["team_id"] == hid].groupby("minute")["xT"].sum()
    a_min = xt[xt["team_id"] == aid].groupby("minute")["xT"].sum()
    # A match that went to extra time has real xT past minute 90 — the old
    # hardcoded range(1, 95) silently dropped it from the bars/rolling curve/
    # hottest-window search even though the Total xT card already summed it
    # (h_min.sum() isn't range-limited), a confusing mismatch. Extend the
    # plotted range to cover whatever minutes actually occurred.
    went_to_et, pens = _match_extra_time_pens(events, info)
    max_evt_minute = (
        int(events["minute"].max())
        if "minute" in events.columns and not events.empty
        else 90
    )
    # Duration from period codes only (a 90-min game with stoppage stays 90).
    went_to_et = bool(went_to_et)
    duration = 120 if went_to_et else 90
    _end = (
        max(duration + 5, max_evt_minute + 2)
        if went_to_et
        else min(max(95, max_evt_minute + 2), 99)
    )
    mins = list(range(1, _end))
    h_vals = [float(h_min.get(m, 0)) for m in mins]
    a_vals = [-float(a_min.get(m, 0)) for m in mins]

    fig = plt.figure(figsize=(15, 9.5), facecolor=BG_DARK)
    chrome(
        fig,
        section="XT PER MINUTE",
        title=f"{hn} vs {an} — xT per Minute",
        subtitle="Diverging bars: home xT rises above zero · away xT "
        "drops below · curves are 5-min rolling averages",
        hn=hn,
        an=an,
        score=str(score),
        footer_note="Spikes = momentum windows",
    )

    PX, PY, PW, PH = 0.05, 0.22, 0.62, 0.66
    header_h = 0.040
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (PX, PY + PH - header_h),
            PW,
            header_h,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor=BG_HEADER,
            edgecolor=GRID_COL,
            linewidth=1.0,
            zorder=1,
        )
    )
    dot_y = PY + PH - header_h / 2
    fig.add_artist(
        mpatches.Circle(
            (PX + 0.018, dot_y),
            0.0035,
            transform=fig.transFigure,
            facecolor=C_GOLD,
            edgecolor="none",
            zorder=2,
        )
    )
    fig.text(
        PX + 0.030,
        dot_y,
        "XT PER MINUTE",
        ha="left",
        va="center",
        color=TEXT_BR,
        fontsize=10.5,
        fontweight="bold",
        family=FONT_MONO,
        zorder=2,
    )

    ax = fig.add_axes([PX, PY, PW, PH - header_h])
    ax.set_facecolor(BG_MID)
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL)
        s.set_linewidth(1.0)
        s.set_alpha(1.0)
    ax.bar(mins, h_vals, color=hc, width=0.85, zorder=3)
    ax.bar(mins, a_vals, color=ac, width=0.85, zorder=3)
    ax.axhline(0, color=GRID_COL, lw=1.0, zorder=4)

    import pandas as _pd

    _hv = _pd.Series(h_vals).rolling(5, center=True, min_periods=1).mean()
    _av = _pd.Series(a_vals).rolling(5, center=True, min_periods=1).mean()
    ax.plot(mins, _hv, color=hc, lw=2.0, alpha=0.95, zorder=5, solid_capstyle="round")
    ax.plot(mins, _av, color=ac, lw=2.0, alpha=0.95, zorder=5, solid_capstyle="round")

    ymax = max(max(h_vals + [0.001]), abs(min(a_vals + [-0.001]))) * 1.15
    markers = [(45, "HT"), (90, "FT")]
    if went_to_et:
        aet_label = f"AET\n{pens[0]}-{pens[1]} PENS" if pens is not None else "AET"
        markers += [(105, "ET-HT"), (120, aet_label)]
    for xv, lb in markers:
        ax.axvline(xv, color=TEXT_FAD, lw=1.0, ls=(0, (1, 3)), alpha=0.7, zorder=2)
        ax.text(
            xv,
            ymax * 0.97,
            lb,
            ha="center",
            va="top",
            color=C_GOLD,
            fontsize=9,
            fontweight="bold",
            family=FONT_MONO,
            linespacing=1.4,
        )
    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(0, mins[-1] + 1)
    ax.set_xlabel(
        "MINUTE", color=TEXT_DIM, fontsize=9, fontweight="bold", family=FONT_MONO
    )
    ax.set_ylabel(
        "xT  (▲ HOME · ▼ AWAY)",
        color=TEXT_DIM,
        fontsize=9,
        fontweight="bold",
        family=FONT_MONO,
    )
    ax.tick_params(colors=TEXT_FAD, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(FONT_MONO)
    for sp in ["top", "right", "left", "bottom"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID_SOFT, lw=0.8, alpha=1.0)

    ht = float(h_min.sum())
    at = float(a_min.sum())
    leader = hn if ht > at else an
    diff = abs(ht - at)

    # Hottest 5-min for each side
    def _best(values, w=5):
        if not any(values):
            return (0.0, 0, 0)
        best = (0.0, 0, 0)
        for s in range(0, len(values) - w + 1):
            x = sum(values[s : s + w])
            if x > best[0]:
                best = (x, s, s + w)
        return best

    bw_h = _best(h_vals)
    bw_a = _best([-v for v in a_vals])
    duration_txt = "120 minutes (AET)" if went_to_et else "90 minutes"
    insight = (
        f"{leader} created {diff:.2f} more xT over the {duration_txt}. "
        f"{hn}'s hottest 5-min: {bw_h[1]:02d}'–{bw_h[2]:02d}' "
        f"({bw_h[0]:.2f} xT). {an}'s hottest 5-min: "
        f"{bw_a[1]:02d}'–{bw_a[2]:02d}' ({bw_a[0]:.2f} xT)."
    )
    if pens is not None:
        insight += f" The tie was settled on penalties, {pens[0]}-{pens[1]}."
    key_insight(fig, 0.70, 0.22, 0.27, 0.66, text=insight, wrap=34)

    cards = [
        (f"{hn[:14]} xT", f"{ht:.2f}", hc),
        ("Diff", f"{'+' if ht-at >= 0 else ''}{ht-at:.2f}", TEXT_BR),
        (f"{an[:14]} xT", f"{at:.2f}", ac),
        ("Hottest H 5'", f"{bw_h[0]:.2f}", hc),
        ("Hottest A 5'", f"{bw_a[0]:.2f}", ac),
    ]
    metric_strip(fig, cards=cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  GOALKEEPER SAVES v2  (fig 12)
# ═════════════════════════════════════════════════════════════════════════
def make_gk_saves_v2(events, info):
    """Plots every shot each keeper faced + outcome."""
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    hid = info.get("home_id")
    aid = info.get("away_id")
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)

    fig = plt.figure(figsize=(16, 10), facecolor=BG_DARK)
    chrome(
        fig,
        section="GOALKEEPER SAVES",
        title=f"{hn} vs {an} — Keeper Saves",
        subtitle="Each dot is a shot the keeper faced · size = xG · "
        "filled = goal · ringed = saved",
        hn=hn,
        an=an,
        score=str(score),
        footer_note="Save quality scales with the xG of shots faced",
    )

    # Two pitches side-by-side, each in its own flat hairline-bordered card
    # with a header strip — matching panel_card()'s `.panel-head`.
    for i, (team_id, opp_id, team_name, team_color, x0, w) in enumerate(
        [
            (aid, hid, hn + "'s Keeper", hc, 0.04, 0.45),
            (hid, aid, an + "'s Keeper", ac, 0.51, 0.45),
        ]
    ):
        PX, PY, PW, PH = x0, 0.16, w, 0.72
        header_h = 0.040
        fig.add_artist(
            mpatches.FancyBboxPatch(
                (PX, PY + PH - header_h),
                PW,
                header_h,
                boxstyle="round,pad=0.0,rounding_size=0.006",
                transform=fig.transFigure,
                facecolor=BG_HEADER,
                edgecolor=GRID_COL,
                linewidth=1.0,
                zorder=1,
            )
        )
        dot_y = PY + PH - header_h / 2
        fig.add_artist(
            mpatches.Circle(
                (PX + 0.018, dot_y),
                0.0035,
                transform=fig.transFigure,
                facecolor=team_color,
                edgecolor="none",
                zorder=2,
            )
        )
        fig.text(
            PX + 0.030,
            dot_y,
            team_name.upper(),
            ha="left",
            va="center",
            color=TEXT_BR,
            fontsize=10.5,
            fontweight="bold",
            family=FONT_MONO,
            zorder=2,
        )
        fig.add_artist(
            mpatches.FancyBboxPatch(
                (PX, PY),
                PW,
                PH - header_h,
                boxstyle="round,pad=0.0,rounding_size=0.006",
                transform=fig.transFigure,
                facecolor=BG_MID,
                edgecolor=GRID_COL,
                linewidth=1.0,
                zorder=-2,
            )
        )

        ax = fig.add_axes(
            [x0 + 0.075, 0.205, min(w * 0.62, 0.28), PH - header_h - 0.07]
        )
        _draw_vertical_pitch(ax, attacking_only=False)
        pax = _VerticalPitchProxy(ax)
        # Shots THE OPPONENT took (= shots THIS keeper faced)
        sub = events[(events["team_id"] == opp_id) & (events["is_shot"] == True)]
        n_total = len(sub)
        n_goal = 0
        n_save = 0
        xg_faced = 0.0
        for _, r in sub.iterrows():
            x = float(_safe(r.get("x"), 80))
            y = float(_safe(r.get("y"), 50))
            xg = float(_safe(r.get("xG"), 0) or 0)
            xg_faced += xg
            stype = r.get("shot_whoscored_type") or r.get("type") or ""
            if bool(r.get("is_goal")):
                pax.scatter(
                    [x],
                    [y],
                    s=80 + xg * 1500,
                    marker="*",
                    color=C_GOLD,
                    edgecolor=team_color,
                    lw=1.5,
                    alpha=0.98,
                    zorder=5,
                )
                n_goal += 1
            elif stype == "SavedShot":
                pax.scatter(
                    [x],
                    [y],
                    s=60 + xg * 1300,
                    facecolor="none",
                    edgecolor=team_color,
                    lw=2.0,
                    alpha=0.92,
                    zorder=4,
                )
                n_save += 1
            else:
                pax.scatter(
                    [x],
                    [y],
                    s=40 + xg * 1100,
                    facecolor="none",
                    edgecolor=TEXT_FAD,
                    lw=1.0,
                    alpha=0.55,
                    zorder=3,
                )
        # Per-keeper summary line, just below the pitch header.
        fig.text(
            x0 + w / 2,
            PY + PH - header_h - 0.022,
            f"Faced {n_total} shots ({xg_faced:.2f} xG) · "
            f"saved {n_save} · conceded {n_goal}",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=9,
            family=FONT_MONO,
        )

    # Bottom strip
    h_faced = events[(events["team_id"] == aid) & (events["is_shot"] == True)]
    a_faced = events[(events["team_id"] == hid) & (events["is_shot"] == True)]
    h_xg_faced = float(h_faced["xG"].fillna(0).sum() if not h_faced.empty else 0)
    a_xg_faced = float(a_faced["xG"].fillna(0).sum() if not a_faced.empty else 0)
    h_saves = int(
        (
            (h_faced.get("shot_whoscored_type") == "SavedShot")
            if "shot_whoscored_type" in h_faced.columns
            else (h_faced["type"] == "SavedShot")
        ).sum()
    )
    a_saves = int(
        (
            (a_faced.get("shot_whoscored_type") == "SavedShot")
            if "shot_whoscored_type" in a_faced.columns
            else (a_faced["type"] == "SavedShot")
        ).sum()
    )
    cards = [
        (f"{hn[:14]} xG Faced", f"{h_xg_faced:.2f}", hc),
        (f"{hn[:14]} Saves", str(h_saves), hc),
        ("Total Shots", str(len(h_faced) + len(a_faced)), TEXT_BR),
        (f"{an[:14]} Saves", str(a_saves), ac),
        (f"{an[:14]} xG Faced", f"{a_xg_faced:.2f}", ac),
    ]
    metric_strip(fig, cards=cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  MATCH STATISTICS v2  (fig 16) — wraps the existing extension page
# ═════════════════════════════════════════════════════════════════════════
def render_legacy_chart_v2(
    *,
    section,
    title,
    subtitle,
    hn,
    an,
    score,
    footer_note,
    team_color,
    draw_legacy,
    sidebar_title,
    sidebar_headers,
    sidebar_rows,
    insight_text,
    metric_cards,
    legacy_box=None,
):
    """
    Hybrid layout: v2 chrome + sidebar + bottom strip, but the central chart
    is drawn by a LEGACY panel function (the user-preferred legacy look for
    Pass Target Zones, GK Saves, Ball Touches, High Turnovers).

    `draw_legacy(fig, ax)` is a callback that paints the chart on `ax`.
    """
    team_color = _clean_dark_navy(team_color)
    fig = plt.figure(figsize=(16, 10), facecolor=BG_DARK)
    chrome(
        fig,
        section=section,
        title=title,
        subtitle=subtitle,
        hn=hn,
        an=an,
        score=score,
        footer_note=footer_note,
    )

    # Reserve the same left area as render_pitch_overlay_v2, so the sidebar
    # + insight + metric strip fall into the familiar v2 grid.
    PX, PY, PW, PH = 0.05, 0.16, 0.46, 0.72
    header_h, body_h = panel_header_geom(PH)
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (PX, PY + PH - header_h),
            PW,
            header_h,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor=BG_HEADER,
            edgecolor=GRID_COL,
            linewidth=1.0,
            zorder=1,
        )
    )
    dot_y = PY + PH - header_h / 2
    fig.add_artist(
        mpatches.Circle(
            (PX + 0.018, dot_y),
            0.0035,
            transform=fig.transFigure,
            facecolor=team_color,
            edgecolor="none",
            zorder=2,
        )
    )
    fig.text(
        PX + 0.030,
        dot_y,
        section,
        ha="left",
        va="center",
        color=TEXT_BR,
        fontsize=10.5,
        fontweight="bold",
        family=FONT_MONO,
        zorder=2,
    )
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (PX, PY),
            PW,
            body_h,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor=BG_MID,
            edgecolor=GRID_COL,
            linewidth=1.0,
            zorder=-2,
        )
    )

    default_box = [PX + 0.035, PY + 0.020, PW - 0.07, body_h - 0.045]
    ax = fig.add_axes(legacy_box or default_box)
    ax.set_facecolor(BG_MID)
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL)
        s.set_linewidth(0.5)
    try:
        draw_legacy(fig, ax)
    except Exception:
        ax.text(
            0.5,
            0.5,
            "Chart unavailable",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=14,
            family=FONT_SANS,
            transform=ax.transAxes,
        )

    # Sidebar
    ax2 = panel_card(
        fig, 0.55, 0.50, 0.41, 0.38, title=sidebar_title, accent=team_color
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    n_cols = len(sidebar_headers)
    if n_cols == 1:
        xs = [0.04]
    elif n_cols == 2:
        xs = [0.04, 0.95]
    else:
        xs = [0.04] + [0.04 + (i * 0.91 / (n_cols - 1)) for i in range(1, n_cols)]
    for i, (lbl, x) in enumerate(zip(sidebar_headers, xs)):
        ha = "left" if i == 0 else ("right" if i == n_cols - 1 else "center")
        ax2.text(
            x,
            0.90,
            lbl,
            ha=ha,
            va="center",
            color=TEXT_DIM,
            fontsize=8.7,
            fontweight="bold",
            family=FONT_MONO,
            transform=ax2.transAxes,
        )
    ax2.plot(
        [0.03, 0.97], [0.84, 0.84], color=GRID_COL, lw=1.0, transform=ax2.transAxes
    )
    if sidebar_rows:
        n = max(len(sidebar_rows), 1)
        rh = 0.74 / n
        for i, row in enumerate(sidebar_rows):
            cy = 0.78 - (i + 0.5) * rh
            if i > 0:
                ax2.plot(
                    [0.03, 0.97],
                    [cy + rh / 2, cy + rh / 2],
                    color=GRID_SOFT,
                    lw=0.8,
                    transform=ax2.transAxes,
                    zorder=1,
                )
            for j, (val, x) in enumerate(zip(row, xs)):
                ha = "left" if j == 0 else ("right" if j == n_cols - 1 else "center")
                is_last = j == n_cols - 1
                col = TEXT_BR if j == 0 else (team_color if is_last else TEXT_DIM)
                fam = FONT_SANS if j == 0 else FONT_MONO
                ax2.text(
                    x,
                    cy,
                    str(val),
                    ha=ha,
                    va="center",
                    color=col,
                    fontsize=10.5 if j == 0 else (11 if is_last else 10),
                    fontweight="bold" if (j == 0 or is_last) else "normal",
                    family=fam,
                    transform=ax2.transAxes,
                    zorder=2,
                )
    else:
        ax2.text(
            0.5,
            0.4,
            "No data recorded",
            ha="center",
            va="center",
            color=TEXT_FAD,
            fontsize=10,
            style="italic",
            family=FONT_SANS,
            transform=ax2.transAxes,
        )

    key_insight(fig, 0.55, 0.16, 0.41, 0.30, text=insight_text, wrap=52)
    metric_strip(fig, cards=metric_cards)
    return fig


def make_match_stats_v2(events, info, ppda):
    """
    Reuses the polished _draw_team_stats_compare_page from match_report
    (which already uses the unified identity + 'Reading this page' panel).
    Captures the figure it builds via a tiny Pdf-like shim and prevents the
    inner plt.close from disposing it before we can save it later.
    """
    from match_report import _draw_team_stats_compare_page

    captured = []

    class _Capture:
        def savefig(self, fig, **kw):
            captured.append(fig)

    # Temporarily neutralise plt.close so the captured fig stays alive.
    _orig_close = plt.close
    plt.close = lambda *a, **k: None
    try:
        _draw_team_stats_compare_page(_Capture(), info, events, ppda)
    finally:
        plt.close = _orig_close
    return captured[-1] if captured else plt.figure(facecolor=BG_DARK)
