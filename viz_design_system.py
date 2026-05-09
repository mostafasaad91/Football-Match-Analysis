"""
viz_design_system.py
═════════════════════════════════════════════════════════════════════════════
Unified visual identity for the WhoScored Match Analyzer.

Same visual language as the PPDA dial — but with reusable components
across every visual type (pitch maps, networks, timelines, tables, charts).

Every new visual uses:
    1. apply_unified_frame()  — header with colour bar + section number + footer caption
    2. themed_pitch()         — pitch with the same unified colours and grid
    3. metric_strip()         — metric strip (like the one under the PPDA dial)
    4. tagline_card()         — coloured verdict card
    5. legend_chip()          — unified legend chip
"""

from __future__ import annotations
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D


# ═════════════════════════════════════════════════════════════════════════════
# PALETTE — unified across every visual
# ═════════════════════════════════════════════════════════════════════════════
BG_DARK     = "#050508"
BG_MID      = "#0d1117"
BG_PANEL    = "#0a0e16"
BG_PITCH    = "#040c04"
GRID_COL    = "#1e2836"
GRID_SOFT   = "#141b27"

TEXT_MAIN   = "#f0f4ff"
TEXT_BRIGHT = "#ffffff"
TEXT_DIM    = "#94a3b8"
TEXT_FADED  = "#64748b"

C_HOME      = "#e63946"
C_AWAY      = "#1e90ff"
C_GOLD      = "#facc15"
C_GREEN     = "#22c55e"
C_PURPLE    = "#a855f7"
C_TEAL      = "#14b8a6"
C_ORANGE    = "#f97316"
OG_COLOR    = "#ff00ff"


# ═════════════════════════════════════════════════════════════════════════════
# Typography
# ═════════════════════════════════════════════════════════════════════════════
def _shadow(width: float = 3, fg: str = BG_DARK) -> list:
    """Soft text shadow — improves legibility on any background."""
    return [pe.withStroke(linewidth=width, foreground=fg)]


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
    Adds to a figure: header + side colour bar + section number + footer caption.
    Sized for standard figures (e.g. 8.27in landscape).

    Args:
        section: section number/code (e.g. "01", "SHOT MAP").
        title:  .
        subtitle:  .
        accent:  .
        home_name/away_name/score: if provided, rendered in the footer.
        footer_note: methodology note at the page bottom.
    """
    fig.patch.set_facecolor(BG_DARK)

    # ── very thin top colour bar ──
    top_bar = fig.add_axes([0.0, 0.985, 1.0, 0.015], zorder=10)
    top_bar.set_facecolor(accent)
    top_bar.set_xticks([]); top_bar.set_yticks([])
    for s in top_bar.spines.values():
        s.set_visible(False)

    # ── left side bar ──
    side_bar = fig.add_axes([0.0, 0.05, 0.006, 0.93], zorder=10)
    side_bar.set_facecolor(accent)
    side_bar.set_xticks([]); side_bar.set_yticks([])
    for s in side_bar.spines.values():
        s.set_visible(False)

    # ── text header ──
    if section:
        fig.text(0.025, 0.965, section,
                 color=accent, fontsize=10, fontweight="bold")
    if title:
        fig.text(0.025, 0.940, title,
                 color=TEXT_BRIGHT, fontsize=18, fontweight="bold",
                 path_effects=_shadow(2))
    if subtitle:
        fig.text(0.025, 0.913, subtitle,
                 color=TEXT_DIM, fontsize=10, style="italic")

    # ── Footer ──
    foot_y = 0.018
    if home_name and away_name:
        match_str = f"{home_name}  {score or '–'}  {away_name}"
        fig.text(0.025, foot_y, match_str,
                 color=TEXT_DIM, fontsize=8.5, fontweight="bold")
    fig.text(0.5, foot_y, "M A T C H   A N A L Y S I S   R E P O R T",
             ha="center", color=TEXT_FADED, fontsize=7.5, fontweight="bold")
    if footer_note:
        fig.text(0.975, foot_y, footer_note,
                 ha="right", color=TEXT_FADED, fontsize=8, style="italic")

    return fig


def make_themed_figure(w: float = 14, h: float = 9):
    """ figure   dark theme —  ."""
    fig = plt.figure(figsize=(w, h), facecolor=BG_DARK)
    fig.patch.set_facecolor(BG_DARK)
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
    Match_Analysis_Dark.py) so it carries the unified visual identity
    WITHOUT touching the original drawing code.

    Adds:
        • Yellow top bar (slim)
        • Yellow side bar on the left
        • Footer strip with score + 'MATCH ANALYSIS REPORT' + optional note

    Does NOT add a title (the original figure already has its own title
    rendered inside its drawing code) — only the chrome that defines the
    visual identity.
    """
    fig.patch.set_facecolor(BG_DARK)

    # Top bar
    top_bar = fig.add_axes([0.0, 0.985, 1.0, 0.012], zorder=10)
    top_bar.set_facecolor(accent)
    top_bar.set_xticks([]); top_bar.set_yticks([])
    for s in top_bar.spines.values():
        s.set_visible(False)

    # Side bar
    side_bar = fig.add_axes([0.0, 0.04, 0.005, 0.945], zorder=10)
    side_bar.set_facecolor(accent)
    side_bar.set_xticks([]); side_bar.set_yticks([])
    for s in side_bar.spines.values():
        s.set_visible(False)

    foot_y = 0.012
    if home_name and away_name:
        match_str = f"{home_name}  {score or '–'}  {away_name}"
        fig.text(0.025, foot_y, match_str,
                 color=TEXT_DIM, fontsize=8.5, fontweight="bold")
    fig.text(0.5, foot_y, "M A T C H   A N A L Y S I S   R E P O R T",
             ha="center", color=TEXT_FADED, fontsize=7.5, fontweight="bold")
    if footer_note:
        fig.text(0.975, foot_y, footer_note,
                 ha="right", color=TEXT_FADED, fontsize=8, style="italic")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 2) Themed pitch — pitch with the unified visual identity
# ═════════════════════════════════════════════════════════════════════════════
def themed_pitch(ax, attacking_only: bool = False, line_color: str = "#2a3a4a",
                 line_alpha: float = 0.85):
    """
    Draws a pitch in the report style: BG_PITCH background, GRID-like lines,
    low intensity so the content (dots/arrows) reads clearly on top.
    """
    ax.set_facecolor(BG_PITCH)
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL); s.set_linewidth(0.8)

    lc = dict(color=line_color, lw=1.0, alpha=line_alpha, zorder=2)

    # boundary + halfway lines
    ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], **lc)
    ax.plot([50, 50], [0, 100], **lc)
    
    circ = plt.Circle((50, 50), 9.15, fill=False, **lc)
    ax.add_patch(circ)
    ax.scatter([50], [50], color=line_color, s=8, alpha=line_alpha, zorder=2)

    
    for x0 in (0, 100):
        sign = 1 if x0 == 0 else -1
        ax.plot([x0, x0 + sign * 16.5, x0 + sign * 16.5, x0],
                [21.1, 21.1, 78.9, 78.9], **lc)
        ax.plot([x0, x0 + sign * 5.5, x0 + sign * 5.5, x0],
                [36.8, 36.8, 63.2, 63.2], **lc)
        ax.scatter([x0 + sign * 11], [50], color=line_color, s=6,
                   alpha=line_alpha, zorder=2)

    
    for x0 in (0, 100):
        sign = 1 if x0 == 0 else -1
        ax.plot([x0, x0 + sign * 1.5, x0 + sign * 1.5, x0],
                [44, 44, 56, 56], color=line_color, lw=1.2,
                alpha=line_alpha, zorder=2)

    if attacking_only:
        ax.set_xlim(48, 102)


def themed_panel(ax, fill: str = BG_MID, lw: float = 0.8):
    """    subplot."""
    ax.set_facecolor(fill)
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL); s.set_linewidth(lw)


# ═════════════════════════════════════════════════════════════════════════════
# 3) Metric strip — metric strip (like the one under the PPDA dial)
# ═════════════════════════════════════════════════════════════════════════════
def metric_strip(fig, x: float, y: float, w: float, h: float,
                 metrics: list[tuple[str, Any, str]],
                 bg: str = BG_MID):
    """
    Draws a strip split into N metric cards, each:
        (label, value, color)
    at (x, y, w, h) as fractions of the figure.
    """
    n = len(metrics)
    if n == 0:
        return
    cell_w = w / n
    for i, (label, value, color) in enumerate(metrics):
        cx = x + i * cell_w
        ax = fig.add_axes([cx, y, cell_w, h])
        ax.set_facecolor(bg)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(color); s.set_linewidth(1.0)
        ax.text(0.5, 0.62, str(value), ha="center", va="center",
                color=color, fontsize=20, fontweight="bold",
                transform=ax.transAxes,
                path_effects=_shadow(2))
        ax.text(0.5, 0.20, label.upper(), ha="center", va="center",
                color=TEXT_DIM, fontsize=8.5, fontweight="bold",
                transform=ax.transAxes)


def tagline_card(fig, x: float, y: float, w: float, h: float,
                 text: str, color: str = C_GOLD, bg: str = BG_MID):
    """coloured verdict card — /  visual."""
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(bg)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(1.4)
    ax.text(0.5, 0.5, text, ha="center", va="center",
            color=color, fontsize=12, fontweight="bold",
            transform=ax.transAxes, path_effects=_shadow(2))


# ═════════════════════════════════════════════════════════════════════════════
# 4) Legend chips — unified
# ═════════════════════════════════════════════════════════════════════════════
def legend_chips(ax, items: list[tuple[str, str, str]], y: float = -0.06,
                 fontsize: float = 9):
    """
    items: list of (label, color, marker) — marker ∈ {'o','s','*','D','X','—'}
    """
    handles = []
    for label, color, marker in items:
        if marker == "—":
            handles.append(Line2D([0], [0], color=color, lw=2.2, label=label))
        else:
            handles.append(Line2D([0], [0], marker=marker, color=color,
                                  markeredgecolor="white",
                                  markersize=10, lw=0, label=label))
    leg = ax.legend(handles=handles, ncol=min(len(items), 6),
                    fontsize=fontsize, loc="lower center",
                    bbox_to_anchor=(0.5, y),
                    facecolor=BG_MID, edgecolor=GRID_COL,
                    labelcolor=TEXT_MAIN, framealpha=0.95)
    leg.get_frame().set_linewidth(0.6)
    return leg


# ═════════════════════════════════════════════════════════════════════════════
# 5) Intensity bar — strength bar (colour-coded like a gauge)
# ═════════════════════════════════════════════════════════════════════════════
def intensity_bar(fig, x: float, y: float, w: float, h: float,
                  value: float, vmin: float, vmax: float,
                  label: str, color: str,
                  reverse: bool = False):
    """
    Horizontal bar with value + label + tick marks. reverse=True = smaller is better.
    """
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(BG_MID)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL)

    # shadow
    ax.barh(0, 1, color="#1a2030", height=0.6, zorder=1)
    
    ratio = max(0, min(1, (value - vmin) / (vmax - vmin)))
    if reverse:
        ratio = 1 - ratio
    ax.barh(0, ratio, color=color, height=0.6, zorder=2,
            edgecolor=TEXT_BRIGHT, lw=0.6)

    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.6)
    ax.text(0.01, 0, label, ha="left", va="center",
            color=TEXT_BRIGHT, fontsize=10, fontweight="bold",
            transform=ax.transAxes, path_effects=_shadow(2))
    ax.text(0.99, 0, f"{value:.2f}" if isinstance(value, float) else str(value),
            ha="right", va="center",
            color=color, fontsize=11, fontweight="bold",
            transform=ax.transAxes, path_effects=_shadow(2))


# ═════════════════════════════════════════════════════════════════════════════
# 6) Side panel — info/value list, vertical block beside the visual
# ═════════════════════════════════════════════════════════════════════════════
def side_panel(fig, x: float, y: float, w: float, h: float,
               title: str, rows: list[tuple[str, Any, str]],
               accent: str = C_GOLD):
    """
    Panel beside the visual with a title + rows (label, value, colour).
    """
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(BG_MID)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    
    ax.add_patch(mpatches.Rectangle((0, 0.92), 1, 0.08,
                                     facecolor=accent, alpha=0.15, lw=0,
                                     transform=ax.transAxes))
    ax.text(0.05, 0.96, title.upper(), ha="left", va="center",
            color=accent, fontsize=10, fontweight="bold",
            transform=ax.transAxes)
    ax.plot([0.05, 0.95], [0.92, 0.92], color=accent, lw=0.8,
            alpha=0.5, transform=ax.transAxes)

    
    n = len(rows)
    if n == 0:
        return
    spacing = 0.85 / n
    for i, (label, value, val_color) in enumerate(rows):
        cy = 0.88 - (i + 0.5) * spacing
        ax.text(0.05, cy, label, ha="left", va="center",
                color=TEXT_DIM, fontsize=9, transform=ax.transAxes)
        ax.text(0.95, cy, str(value), ha="right", va="center",
                color=val_color, fontsize=11, fontweight="bold",
                transform=ax.transAxes,
                path_effects=_shadow(2))
