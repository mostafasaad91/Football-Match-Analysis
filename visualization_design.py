"""
visualization_design.py
═════════════════════════════════════════════════════════════════════════════
Unified visual identity for the WhoScored Match Analyzer.

Same visual language as the PPDA dial — but with reusable components
across every visual type (pitch maps, networks, timelines, tables, charts).

Every new visual uses:
    1. apply_unified_frame()  — header with AMOLED pure-black frame + section number + footer caption
    2. themed_pitch()         — pitch with the same unified colours and grid
    3. metric_strip()         — metric strip (like the one under the PPDA dial)
    4. tagline_card()         — coloured verdict card
    5. legend_chip()          — unified legend chip
"""

from __future__ import annotations
from typing import Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# ═════════════════════════════════════════════════════════════════════════════
# PALETTE — one source, shared with visualization_components
# ═════════════════════════════════════════════════════════════════════════════
# This module used to carry its own copy of the palette and of the readability
# helpers. The copy was written before the light theme existed and never learnt
# about it: BG_DARK was the literal "#000000" and TEXT_BRIGHT the literal
# "#FFFFFF", so apply_unified_frame painted a black page in the middle of a
# light package and readable_team_text answered "#FFFFFF" for every kit --
# white type on the white panel, a contrast ratio of 1.0.
#
# The values are imported now rather than restated. visualization_components
# reads MATCH_ANALYSIS_THEME once at import and is the only place that decides
# what a colour is, so the frame helpers below follow the theme without knowing
# it exists. TEXT_BRIGHT and TEXT_FADED are kept as the names this module's own
# code and its callers use, aliased to the canonical ones.
from visualization_components import (  # noqa: E402
    BG_DARK,
    BG_MID,
    BG_PANEL,
    BG_HEADER,
    BG_PITCH,
    GRID_COL,
    GRID_SOFT,
    PITCH_LINE,
    PITCH_LINE_ALPHA,
    PITCH_LINE_WIDTH,
    TEXT_MAIN,
    TEXT_DIM,
    TEXT_BR as TEXT_BRIGHT,
    TEXT_FAD as TEXT_FADED,
    C_HOME,
    C_AWAY,
    C_GOLD,
    C_MAGENTA,
    C_ACCENT,
    C_LIME,
    FONT_SANS,
    FONT_MONO,
    ACCENT_TEXT,
    IS_LIGHT_THEME,
    contrast_ratio,
    readable_on,
    readable_team_text,
    apply_amoled_defaults,
    _relative_luminance,
)

# Accent colours this module uses that the shared palette does not name.
C_GREEN = "#15803D" if IS_LIGHT_THEME else "#3DDC84"
C_PURPLE = "#86198F" if IS_LIGHT_THEME else "#a855f7"
C_TEAL = C_GREEN
C_ORANGE = "#B45309" if IS_LIGHT_THEME else "#f97316"
OG_COLOR = "#B3129B" if IS_LIGHT_THEME else "#ff00ff"

# The stroke behind text drawn over busy imagery is the page it sits on, not a
# fixed black: on the light page a black halo is the thing the reader sees.
TEXT_SHADOW = [pe.withStroke(linewidth=2.6, foreground=BG_DARK)]
TEXT_SHADOW_STRONG = [pe.withStroke(linewidth=3.4, foreground=BG_DARK)]


# ═════════════════════════════════════════════════════════════════════════════
# Typography
# ═════════════════════════════════════════════════════════════════════════════
def _shadow(width: float = 0, fg: str = BG_DARK) -> list:
    """Soft text shadow. The Pure-Black identity uses flat colour on flat
    black (no drop-shadows on text), so this returns an empty list by
    default; pass width>0 for a thin stroke in rare cases that still need
    one (e.g. text drawn directly over busy imagery)."""
    if width <= 0:
        return []
    return [pe.withStroke(linewidth=width, foreground=fg)]


def _neon_backdrop(fig):
    """No-op: the reference identity is a flat true-black background with
    no gradient/glow texture. Kept as a function so existing call sites
    do not need to change."""
    fig._neon_backdrop_applied = True
    return


def _pitch_glow(ax):
    """No-op: the reference pitch panels are flat black with hairline
    markings only, no radial glow. Kept for API compatibility."""
    return


# ═════════════════════════════════════════════════════════════════════════════
# 1) Unified frame — wraps any figure in the same visual frame
# ═════════════════════════════════════════════════════════════════════════════
def apply_unified_frame(
    fig,
    section: str = "",
    title: str = "",
    subtitle: str = "",
    accent: str = C_GOLD,
    home_name: str | None = None,
    away_name: str | None = None,
    score: str | None = None,
    footer_note: str | None = None,
):
    """
    Adds to a figure: eyebrow + title + subtitle header, and a footer rule
    with score / report tag / methodology note — matching chrome() in
    visualization_components.py (flat true-black background, no glow frame around the page).

    Args:
        section: small mono eyebrow label (e.g. "SHOT MAP").
        title / subtitle: header text.
        accent: colour used for the small eyebrow dot.
        home_name/away_name/score: if provided, rendered in the footer.
        footer_note: methodology note at the page bottom.
    """
    fig.patch.set_facecolor(BG_DARK)

    if section:
        fig.text(0.0335, 0.964, "●", color=accent, fontsize=7, family=FONT_SANS)
        fig.text(
            0.044,
            0.958,
            section,
            color=TEXT_DIM,
            fontsize=9.5,
            fontweight="bold",
            family=FONT_MONO,
        )
    if title:
        fig.text(
            0.030,
            0.925,
            title,
            color=TEXT_BRIGHT,
            fontsize=20,
            fontweight="bold",
            family=FONT_SANS,
        )
    if subtitle:
        fig.text(
            0.030, 0.895, subtitle, color=TEXT_DIM, fontsize=10.5, family=FONT_SANS
        )

    # ── Footer: hairline rule + score / report tag / note ──
    fig.add_artist(
        mpatches.Rectangle(
            (0.030, 0.045),
            0.940,
            0.0012,
            transform=fig.transFigure,
            facecolor=GRID_COL,
            edgecolor="none",
            zorder=5,
        )
    )
    foot_y = 0.018
    if home_name and away_name:
        match_str = f"{home_name}  {score or '–'}  {away_name}"
        fig.text(
            0.030,
            foot_y,
            match_str,
            color=TEXT_BRIGHT,
            fontsize=9.5,
            fontweight="bold",
            family=FONT_MONO,
        )
    fig.text(
        0.5,
        foot_y,
        "MATCH ANALYSIS REPORT",
        ha="center",
        color=TEXT_DIM,
        fontsize=8.5,
        fontweight="bold",
        family=FONT_MONO,
    )
    if footer_note:
        fig.text(
            0.970,
            foot_y,
            footer_note,
            ha="right",
            color=TEXT_FADED,
            fontsize=8.5,
            family=FONT_MONO,
        )

    return fig


def make_themed_figure(w: float = 14, h: float = 9):
    """figure   dark theme —  ."""
    fig = plt.figure(figsize=(w, h), facecolor=BG_DARK)
    fig.patch.set_facecolor(BG_DARK)
    _neon_backdrop(fig)
    return fig


def rebrand_figure(
    fig,
    home_name: str | None = None,
    away_name: str | None = None,
    score: str | None = None,
    accent: str = C_GOLD,
    footer_note: str | None = None,
):
    """
    Re-brand any pre-built figure (eg. one of the 47+ legacy visuals in
    football_match_analysis.py) so it carries the unified visual identity
    WITHOUT touching the original drawing code.

    Adds:
        • AMOLED pure-black border
        • subtle charcoal shadow
        • Footer strip with score + 'MATCH ANALYSIS REPORT' + optional note

    Does NOT add a title (the original figure already has its own title
    rendered inside its drawing code) — only the chrome that defines the
    visual identity.
    """
    fig.patch.set_facecolor(BG_DARK)

    foot_y = 0.012
    if home_name and away_name:
        match_str = f"{home_name}  {score or '–'}  {away_name}"
        fig.text(
            0.030,
            foot_y,
            match_str,
            color=TEXT_BRIGHT,
            fontsize=8.5,
            fontweight="bold",
            family=FONT_MONO,
        )
    fig.text(
        0.5,
        foot_y,
        "MATCH ANALYSIS REPORT",
        ha="center",
        color=TEXT_DIM,
        fontsize=7.5,
        fontweight="bold",
        family=FONT_MONO,
    )
    if footer_note:
        fig.text(
            0.975,
            foot_y,
            footer_note,
            ha="right",
            color=TEXT_FADED,
            fontsize=8,
            family=FONT_MONO,
        )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 2) Themed pitch — pitch with the unified visual identity
# ═════════════════════════════════════════════════════════════════════════════
def themed_pitch(
    ax,
    attacking_only: bool = False,
    line_color: str | None = None,
    line_alpha: float | None = None,
):
    """
    Draws a pitch in the report style: pure-black background with white
    markings held at a low alpha, so the content (dots/arrows) still reads
    clearly on top.
    """
    line_color = line_color or PITCH_LINE
    line_alpha = PITCH_LINE_ALPHA if line_alpha is None else line_alpha
    ax.set_facecolor(BG_PITCH)
    ax.set_aspect("equal")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    # Legacy call sites pass near-black greys that vanish on pure black.
    if line_color.upper() in ("#2A2A2A", "#3A3A3A", "#A8B8CA", "#C4CEDD", GRID_COL.upper()):
        line_color = PITCH_LINE
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL)
        s.set_linewidth(1.0)
        s.set_alpha(1.0)

    lc = dict(color=line_color, lw=PITCH_LINE_WIDTH, alpha=line_alpha, zorder=2)

    # boundary + halfway lines
    ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], **lc)
    ax.plot([50, 50], [0, 100], **lc)

    circ = plt.Circle((50, 50), 9.15, fill=False, **lc)
    ax.add_patch(circ)
    ax.scatter([50], [50], color=line_color, s=8, alpha=line_alpha, zorder=2)

    for x0 in (0, 100):
        sign = 1 if x0 == 0 else -1
        ax.plot(
            [x0, x0 + sign * 16.5, x0 + sign * 16.5, x0], [21.1, 21.1, 78.9, 78.9], **lc
        )
        ax.plot(
            [x0, x0 + sign * 5.5, x0 + sign * 5.5, x0], [36.8, 36.8, 63.2, 63.2], **lc
        )
        ax.scatter(
            [x0 + sign * 11], [50], color=line_color, s=6, alpha=line_alpha, zorder=2
        )

    for x0 in (0, 100):
        sign = 1 if x0 == 0 else -1
        ax.plot(
            [x0, x0 + sign * 1.5, x0 + sign * 1.5, x0],
            [44, 44, 56, 56],
            color=line_color,
            lw=1.2,
            alpha=line_alpha,
            zorder=2,
        )

    if attacking_only:
        ax.set_xlim(48, 102)


# ═════════════════════════════════════════════════════════════════════════════
# 3) Metric strip — metric strip (like the one under the PPDA dial)
# ═════════════════════════════════════════════════════════════════════════════
def metric_strip(
    fig,
    x: float,
    y: float,
    w: float,
    h: float,
    metrics: list[tuple[str, Any, str]],
    bg: str = BG_MID,
):
    """
    Draws a strip split into N flat metric cards, each:
        (label, value, color)
    at (x, y, w, h) as fractions of the figure. Single hairline border,
    no glow/glass effects — matching `.stat-card` in the reference CSS.
    """
    n = len(metrics)
    if n == 0:
        return
    gap = 0.012
    cell_w = (w - gap * (n - 1)) / n
    for i, (label, value, color) in enumerate(metrics):
        cx = x + i * (cell_w + gap)
        card = mpatches.FancyBboxPatch(
            (cx, y),
            cell_w,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor=bg if bg != BG_MID else BG_MID,
            edgecolor=GRID_COL,
            linewidth=1.0,
            zorder=-1,
        )
        fig.add_artist(card)
        ax = fig.add_axes([cx, y, cell_w, h])
        ax.set_facecolor((0, 0, 0, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(
            0.5,
            0.58,
            str(value),
            ha="center",
            va="center",
            color=readable_on(color, BG_PANEL),
            fontsize=22,
            fontweight="bold",
            family=FONT_MONO,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.20,
            label.upper(),
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=8.5,
            fontweight="bold",
            family=FONT_MONO,
            transform=ax.transAxes,
        )


def tagline_card(
    fig,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    color: str = C_GOLD,
    bg: str = BG_MID,
):
    """Raised coloured verdict card."""
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(BG_MID)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(color)
        s.set_linewidth(1.2)
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        color=readable_on(color, BG_PANEL),
        fontsize=12,
        fontweight="bold",
        family=FONT_SANS,
        transform=ax.transAxes,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4) Legend chips — unified
# ═════════════════════════════════════════════════════════════════════════════
def legend_chips(
    ax, items: list[tuple[str, str, str]], y: float = -0.06, fontsize: float = 9
):
    """
    items: list of (label, color, marker) — marker ∈ {'o','s','*','D','X','—'}
    """
    handles = []
    for label, color, marker in items:
        if marker == "—":
            handles.append(Line2D([0], [0], color=color, lw=2.2, label=label))
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color=color,
                    markeredgecolor="white",
                    markersize=10,
                    lw=0,
                    label=label,
                )
            )
    leg = ax.legend(
        handles=handles,
        ncol=min(len(items), 6),
        fontsize=fontsize,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        facecolor=BG_MID,
        edgecolor=GRID_COL,
        labelcolor=TEXT_MAIN,
        framealpha=0.95,
    )
    leg.get_frame().set_linewidth(0.6)
    return leg


# ═════════════════════════════════════════════════════════════════════════════
# 5) Intensity bar — strength bar (colour-coded like a gauge)
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# 6) Side panel — info/value list, vertical block beside the visual
# ═════════════════════════════════════════════════════════════════════════════
def side_panel(
    fig,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    rows: list[tuple[str, Any, str]],
    accent: str = C_GOLD,
):
    """
    Panel beside the visual with a title + rows (label, value, colour).
    """
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(BG_MID)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(
        mpatches.Rectangle(
            (0, 0.92),
            1,
            0.08,
            facecolor=accent,
            alpha=0.15,
            lw=0,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.05,
        0.96,
        title.upper(),
        ha="left",
        va="center",
        color=readable_on(accent, BG_PANEL),
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.plot(
        [0.05, 0.95],
        [0.92, 0.92],
        color=accent,
        lw=0.8,
        alpha=0.5,
        transform=ax.transAxes,
    )

    n = len(rows)
    if n == 0:
        return
    spacing = 0.85 / n
    for i, (label, value, val_color) in enumerate(rows):
        cy = 0.88 - (i + 0.5) * spacing
        ax.text(
            0.05,
            cy,
            label,
            ha="left",
            va="center",
            color=TEXT_DIM,
            fontsize=9,
            transform=ax.transAxes,
        )
        ax.text(
            0.95,
            cy,
            str(value),
            ha="right",
            va="center",
            color=val_color,
            fontsize=11,
            fontweight="bold",
            transform=ax.transAxes,
            path_effects=_shadow(2),
        )
