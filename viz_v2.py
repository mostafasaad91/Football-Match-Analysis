"""
viz_v2.py — Reusable design-system components for the redesigned (v2)
match-analysis visuals.

Every v2 visual composes the same primitives:
    • chrome()         — yellow top + side bar + 3-line header + footer
    • panel_card()     — panel with top accent stripe (no full border)
    • metric_card()    — single big-number card for the bottom strip
    • metric_strip()   — convenience wrapper for the 5-card bottom row
    • key_insight()    — gold callout box with one tactical sentence
    • themed_pitch()   — pitch with subdued lines so data overlay pops

Strict colour discipline:
    red/blue → teams only · gold → accent/leader/chrome · grey → labels
"""
from __future__ import annotations
import os
import textwrap as _tw
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ── Palette ─────────────────────────────────────────────────────────────
_THEME = os.environ.get("MATCH_ANALYSIS_THEME", "dark").strip().lower()
if _THEME == "light":
    BG_DARK   = "#FFFFFF"
    BG_MID    = "#F3F4F6"
    BG_PANEL  = "#FFFFFF"
    BG_PITCH  = "#EEF7EE"
    GRID_COL  = "#D1D5DB"
    GRID_SOFT = "#E5E7EB"

    TEXT_BR   = "#111827"
    TEXT_MAIN = "#1F2937"
    TEXT_DIM  = "#4B5563"
    TEXT_FAD  = "#6B7280"
else:
    BG_DARK   = "#050508"
    BG_MID    = "#0d1117"
    BG_PANEL  = "#0a0e16"
    BG_PITCH  = "#070d14"
    GRID_COL  = "#1e2836"
    GRID_SOFT = "#141b27"

    TEXT_BR   = "#ffffff"
    TEXT_MAIN = "#f0f4ff"
    TEXT_DIM  = "#94a3b8"
    TEXT_FAD  = "#64748b"

C_HOME = "#e63946"
C_AWAY = "#1e90ff"
C_GOLD = "#D97706" if _THEME == "light" else "#facc15"


def shadow(w: float = 2.0, fg: str = BG_DARK):
    return [pe.withStroke(linewidth=w, foreground=fg)]


# ── 1. CHROME (top + side bars + header + footer) ───────────────────────
def chrome(fig, *, section: str, title: str, subtitle: str,
           hn: str, an: str, score: str, footer_note: str):
    fig.patch.set_facecolor(BG_DARK)
    # top + side accent bars
    top = fig.add_axes([0, 0.985, 1, 0.012], zorder=10)
    top.set_facecolor(C_GOLD); top.set_xticks([]); top.set_yticks([])
    for s in top.spines.values():
        s.set_visible(False)
    side = fig.add_axes([0, 0.05, 0.005, 0.935], zorder=10)
    side.set_facecolor(C_GOLD); side.set_xticks([]); side.set_yticks([])
    for s in side.spines.values():
        s.set_visible(False)
    # header
    fig.text(0.025, 0.965, section, color=C_GOLD, fontsize=10,
             fontweight="bold")
    fig.text(0.025, 0.940, title, color=TEXT_BR, fontsize=20,
             fontweight="bold", path_effects=shadow(2))
    fig.text(0.025, 0.913, subtitle, color=TEXT_DIM, fontsize=10,
             style="italic")
    # footer
    fig.text(0.025, 0.018, f"{hn}  {score}  {an}",
             color=TEXT_DIM, fontsize=9, fontweight="bold")
    fig.text(0.5, 0.018, "M A T C H   A N A L Y S I S   R E P O R T",
             ha="center", color=TEXT_FAD, fontsize=8, fontweight="bold")
    fig.text(0.975, 0.018, footer_note,
             ha="right", color=TEXT_FAD, fontsize=8, style="italic")


# ── 2. PANEL CARD (top stripe, no full border) ──────────────────────────
def panel_card(fig, x, y, w, h, *, title: str, accent: str = C_GOLD,
               body: bool = True):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(BG_MID)
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL); s.set_linewidth(0.5)
    ax.set_xticks([]); ax.set_yticks([])
    # 7%-tall accent strip at the top
    ax.add_patch(mpatches.Rectangle((0, 0.93), 1, 0.07,
                  facecolor=accent, alpha=0.25, lw=0,
                  transform=ax.transAxes))
    ax.text(0.025, 0.965, title.upper(), ha="left", va="center",
            color=accent, fontsize=11, fontweight="bold",
            transform=ax.transAxes, path_effects=shadow(2))
    if body:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return ax


# ── 3. METRIC CARD ──────────────────────────────────────────────────────
def metric_card(fig, x, y, w, h, *, label: str, value, accent: str,
                big: bool = True):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(BG_MID)
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL); s.set_linewidth(0.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # 3px accent stripe
    ax.add_patch(mpatches.Rectangle((0, 0.965), 1, 0.035,
                  facecolor=accent, lw=0, transform=ax.transAxes))
    ax.text(0.5, 0.55, str(value), ha="center", va="center",
            color=accent, fontsize=28 if big else 22, fontweight="bold",
            transform=ax.transAxes, path_effects=shadow(2))
    ax.text(0.5, 0.18, label.upper(), ha="center", va="center",
            color=TEXT_DIM, fontsize=9, fontweight="bold",
            transform=ax.transAxes)


def metric_strip(fig, *, cards, x: float = 0.04, w: float = 0.92,
                 y: float = 0.045, h: float = 0.085):
    """cards: list[(label, value, accent_color)]"""
    n = len(cards)
    if n == 0:
        return
    cw = w / n
    for i, (lbl, val, col) in enumerate(cards):
        metric_card(fig, x + i * cw + 0.005, y,
                    cw - 0.010, h, label=lbl, value=val, accent=col,
                    big=True)


# ── 4. KEY INSIGHT (gold callout) ───────────────────────────────────────
def key_insight(fig, x, y, w, h, *, text: str, wrap: int = 42):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(BG_MID)
    for s in ax.spines.values():
        s.set_edgecolor(C_GOLD); s.set_linewidth(1.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    header_text = "#78350F" if _THEME == "light" else C_GOLD
    header_alpha = 0.14 if _THEME == "light" else 0.18
    ax.add_patch(mpatches.Rectangle((0, 0.78), 1, 0.22,
                  facecolor=C_GOLD, alpha=header_alpha, lw=0,
                  transform=ax.transAxes))
    ax.text(0.04, 0.89, "KEY INSIGHT", ha="left", va="center",
            color=header_text, fontsize=10, fontweight="bold",
            transform=ax.transAxes,
            path_effects=([] if _THEME == "light" else shadow(2)))
    body = "\n".join(_tw.wrap(text, width=wrap))
    ax.text(0.04, 0.62, body, ha="left", va="top",
            color=TEXT_MAIN, fontsize=11, transform=ax.transAxes,
            linespacing=1.45)


# ── 5. THEMED PITCH (subdued lines) ─────────────────────────────────────
def themed_pitch(ax, *, attacking_only: bool = False,
                 line_color: str = "#2a3a4a", line_alpha: float = 0.55):
    import matplotlib.pyplot as plt
    ax.set_facecolor(BG_PITCH)
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL); s.set_linewidth(0.5)
    lc = dict(color=line_color, lw=0.9, alpha=line_alpha, zorder=2)
    # Boundaries + halfway
    ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], **lc)
    ax.plot([50, 50], [0, 100], **lc)
    # Centre circle + mark
    circ = plt.Circle((50, 50), 9.15, fill=False, **lc)
    ax.add_patch(circ)
    ax.scatter([50], [50], color=line_color, s=6, alpha=line_alpha, zorder=2)
    # Penalty + 6-yard boxes + spots
    for x0 in (0, 100):
        sign = 1 if x0 == 0 else -1
        ax.plot([x0, x0 + sign * 16.5, x0 + sign * 16.5, x0],
                [21.1, 21.1, 78.9, 78.9], **lc)
        ax.plot([x0, x0 + sign * 5.5, x0 + sign * 5.5, x0],
                [36.8, 36.8, 63.2, 63.2], **lc)
        ax.scatter([x0 + sign * 11], [50], color=line_color, s=5,
                   alpha=line_alpha, zorder=2)
    # Goals
    for x0 in (0, 100):
        sign = 1 if x0 == 0 else -1
        ax.plot([x0, x0 + sign * 1.5, x0 + sign * 1.5, x0],
                [44, 44, 56, 56], color=line_color, lw=1.0,
                alpha=line_alpha, zorder=2)
    if attacking_only:
        ax.set_xlim(48, 102)


# ── 6. LEGEND CHIPS ─────────────────────────────────────────────────────
def legend_chips(ax, items, y: float = -0.10, fontsize: float = 10):
    """items: list[(label, color)] — small dot + label inline."""
    x = 0.02
    for lbl, col in items:
        ax.text(x, y, "●", ha="left", va="center", color=col,
                fontsize=fontsize + 2, transform=ax.transAxes)
        ax.text(x + 0.022, y, lbl, ha="left", va="center", color=TEXT_MAIN,
                fontsize=fontsize, fontweight="bold",
                transform=ax.transAxes)
        # Approximate width per char to advance x
        x += 0.022 + 0.0085 * len(lbl) + 0.025
