"""
visualization_components.py — Reusable design-system components for the redesigned (v2)
match-analysis visuals.

This module mirrors, as closely as matplotlib allows, the "Pure Black"
HTML/CSS visual identity used for the reference dashboards (xG Flow,
Shot Breakdown, Pass Network):

    bg            #000000   true black, no gradients/glow
    panel         #0a0a0a
    panel header  #101010
    border        #1c1c1c   hairline, 1px-equivalent
    border soft   #141414
    text hi       #ffffff
    text mid      #9a9a9a
    text low      #5a5a5a
    text dim      #3a3a3a
    scotland blue #4d8dff
    morocco red   #ff4d4d
    gold accent   #ffc23c

Typography: Inter (headings/body) + JetBrains Mono (numbers/labels/eyebrows),
matching the reference HTML's font stack. Falls back gracefully to the
default sans/monospace fonts if those families are not installed.

Every v2 visual composes the same primitives:
    • chrome()         — pure-black frame + 3-line header + footer
    • panel_card()      — panel with thin border + dark header strip
    • metric_card()     — single big-number card for the bottom strip
    • metric_strip()    — convenience wrapper for the 5-card bottom row
    • key_insight()     — gold-accented callout box with one tactical sentence
    • themed_pitch()    — pitch with subdued lines so data overlay pops

Strict colour discipline:
    team colours → data only · UI uses black/grey/gold only
"""

from __future__ import annotations
import os
import textwrap as _tw
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib as _mpl
import matplotlib.font_manager as _fm

# ── Font registration ───────────────────────────────────────────────────
# Mirrors the HTML identity's font stack: Inter for UI text, JetBrains Mono
# for numbers / eyebrows / mono-style labels. Registers any local copies
# found under ~/.fonts or /usr/share/fonts so matplotlib can pick them up
# by family name; silently falls back to generic sans/monospace otherwise.
FONT_SANS = "Inter Variable"
FONT_MONO = "JetBrains Mono"


def _register_fonts() -> None:
    candidate_dirs = [
        os.path.expanduser("~/.fonts"),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ]
    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for fn in files:
                if fn.lower().endswith((".ttf", ".otf")):
                    try:
                        _fm.fontManager.addfont(os.path.join(root, fn))
                    except Exception:
                        pass

    available = {f.name for f in _fm.fontManager.ttflist}
    global FONT_SANS, FONT_MONO
    if FONT_SANS not in available:
        FONT_SANS = "DejaVu Sans"
    if FONT_MONO not in available:
        FONT_MONO = "DejaVu Sans Mono"


_register_fonts()


# ── Palette ─────────────────────────────────────────────────────────────
_THEME = os.environ.get("MATCH_ANALYSIS_THEME", "dark").strip().lower()
if _THEME == "light":
    BG_DARK = "#FFFFFF"
    BG_MID = "#F3F4F6"
    BG_PANEL = "#FFFFFF"
    BG_HEADER = "#E5E7EB"
    BG_PITCH = "#EEF7EE"
    GRID_COL = "#D1D5DB"
    GRID_SOFT = "#F0F4FF"

    TEXT_BR = "#111827"
    TEXT_MAIN = "#1F2937"
    TEXT_DIM = "#4B5563"
    TEXT_FAD = "#6B7280"
    TEXT_FAINT = "#9CA3AF"
else:
    # "Pure Black" identity — matches the reference HTML dashboards exactly.
    BG_DARK = "#000000"  # true black page background
    BG_MID = "#0a0a0a"  # panel background
    BG_PANEL = "#0a0a0a"  # panel background (alias kept for compatibility)
    BG_HEADER = "#101010"  # panel header-strip background
    BG_PITCH = "#0a0a0a"  # pitch surface sits on the same panel black
    GRID_COL = "#1c1c1c"  # hairline border
    GRID_SOFT = "#141414"  # softer divider/grid line

    TEXT_BR = "#FFFFFF"
    TEXT_MAIN = "#FFFFFF"
    TEXT_DIM = "#9A9A9A"
    TEXT_FAD = "#5A5A5A"
    TEXT_FAINT = "#3A3A3A"

# Canonical production matchup palette. Every fixture uses the same two roles
# so a reader learns one stable visual language instead of relearning kit
# colours from match to match.
C_HOME = "#9A99B4"
C_AWAY = "#F37680"
C_GOLD = "#FFC23C"
C_MAGENTA = "#E879F9"
C_ACCENT = "#38BDF8"
C_LIME = "#22C55E"

# Shared semantic styles for path-heavy pitch visuals. Colour identifies the
# event outcome; line style keeps the distinction available in grayscale.
EVENT_SUCCESS = C_HOME
EVENT_FAILURE = C_AWAY
EVENT_NEUTRAL = "#B8BDC8"
EVENT_HIGHLIGHT = C_GOLD
QUIET_DASH = (0, (2.2, 3.2))
FAILURE_DASH = (0, (4.0, 3.0))


# ── Readability helpers ─────────────────────────────────────────────
def _relative_luminance(color: str) -> float:
    """WCAG relative luminance for automatic text contrast decisions."""
    try:
        r, g, b = mcolors.to_rgb(color)
    except Exception:
        r, g, b = mcolors.to_rgb(TEXT_BR)
    vals = []
    for c in (r, g, b):
        vals.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]


def contrast_ratio(fg: str, bg: str = BG_PANEL) -> float:
    l1, l2 = _relative_luminance(fg), _relative_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def readable_on(
    color: str,
    bg: str = BG_PANEL,
    *,
    min_ratio: float = 5.0,
    fallback: str | None = None,
) -> str:
    """Return the colour if readable on bg, otherwise return white/dark fallback."""
    fallback = fallback or (TEXT_BR if _THEME != "light" else "#111827")
    try:
        return color if contrast_ratio(color, bg) >= min_ratio else fallback
    except Exception:
        return fallback


def readable_team_text(team_color: str, bg: str = BG_PANEL) -> str:
    """Use the kit colour for labels only when it is legible; otherwise use white."""
    return readable_on(team_color, bg, min_ratio=5.0, fallback=TEXT_BR)


def _rgb_distance(first: str, second: str) -> float:
    """Return a compact RGB distance used for visual collision checks."""
    try:
        a = mcolors.to_rgb(first)
        b = mcolors.to_rgb(second)
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    except Exception:
        return 1.0


def network_link_palette(team_color: str) -> tuple[str, str, str]:
    """Return low/medium/strong link colours distinct from player nodes.

    Passing links encode relationship strength, while node fill encodes team
    identity. The neutral ramp is preferred because it works with almost every
    kit. If a team itself is grey/white, an indigo ramp is selected so links
    can never disappear into the nodes.
    """
    neutral = ("#3F4652", "#98A2B3", "#F1F5F9")
    indigo = ("#2E3A67", "#647DFF", "#B8C2FF")
    return indigo if min(_rgb_distance(team_color, c) for c in neutral) < 0.22 else neutral


ACCENT_TEXT = readable_on(C_GOLD, BG_PANEL, min_ratio=5.0, fallback=TEXT_BR)


# Global defaults for all v2 visuals and any Matplotlib objects drawn inside them.
def apply_amoled_defaults() -> None:
    try:
        _mpl.rcParams.update(
            {
                "figure.facecolor": BG_DARK,
                "axes.facecolor": BG_PANEL,
                "savefig.facecolor": BG_DARK,
                "savefig.edgecolor": BG_DARK,
                "text.color": TEXT_MAIN,
                "axes.labelcolor": TEXT_DIM,
                "xtick.color": TEXT_DIM,
                "ytick.color": TEXT_DIM,
                "axes.edgecolor": GRID_COL,
                "grid.color": GRID_COL,
                "legend.facecolor": BG_PANEL,
                "legend.edgecolor": GRID_COL,
                "font.family": FONT_SANS,
            }
        )
    except Exception:
        pass


apply_amoled_defaults()


def _neon_backdrop(fig):
    """No-op: the reference identity is a flat true-black background with
    no gradient/glow texture. Kept as a function (rather than removed) so
    existing call sites in tactical_visualizations.py do not need to change."""
    fig._neon_backdrop_applied = True
    return


def _pitch_glow(ax):
    """No-op for the same reason as _neon_backdrop — the reference pitch
    panels are flat black with hairline markings, no radial glow."""
    return


def shadow(w: float = 0.0, fg: str = BG_DARK):
    """Text shadow helper. The reference identity does not use drop-shadows
    on text (flat colour on flat black gives enough contrast), so this
    returns an empty path-effects list by default. A small stroke is still
    available on request (w > 0) for text drawn directly over imagery,
    e.g. player labels on top of pitch lines."""
    if w <= 0:
        return []
    return [pe.withStroke(linewidth=w, foreground=fg)]


def raised_panel_backdrop(
    fig,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    edge: str | None = None,
    glow: str | None = None,
    depth: float = 0.006,
    radius: float = 0.010,
    zorder: float = -4,
):
    """No-op in the Pure-Black identity: panels are flat with a 1px hairline
    border only (no drop shadow, no glow ring). Kept for API compatibility
    with existing call sites."""
    return


def _panel_rect(fig, x, y, w, h, *, zorder=-2):
    """Flat panel background + thin hairline border, matching `.panel` in
    the reference CSS (background: panel; border: 1px solid line)."""
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.006",
        transform=fig.transFigure,
        facecolor=BG_MID,
        edgecolor=GRID_COL,
        linewidth=1.0,
        zorder=zorder,
    )
    fig.add_artist(rect)
    return rect


# ── 1. CHROME (frame + header + footer) ─────────────────────────────────
def chrome(
    fig,
    *,
    section: str,
    title: str,
    subtitle: str,
    hn: str,
    an: str,
    score: str,
    footer_note: str,
):
    fig.patch.set_facecolor(BG_DARK)

    # eyebrow (small gold dot + mono label), matching `.eyebrow` in the CSS
    fig.text(0.0335, 0.964, "●", color=C_GOLD, fontsize=7, family=FONT_SANS)
    fig.text(
        0.044,
        0.958,
        section,
        color=TEXT_DIM,
        fontsize=9.5,
        fontweight="bold",
        family=FONT_MONO,
    )
    fig.text(
        0.030,
        0.925,
        title,
        color=TEXT_BR,
        fontsize=20,
        fontweight="bold",
        family=FONT_SANS,
    )
    fig.text(0.030, 0.895, subtitle, color=TEXT_DIM, fontsize=10.5, family=FONT_SANS)

    # footer — thin hairline rule + score / report tag / note
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
    fig.text(
        0.030,
        0.018,
        f"{hn}  {score}  {an}",
        color=TEXT_BR,
        fontsize=9.5,
        fontweight="bold",
        family=FONT_MONO,
    )
    fig.text(
        0.5,
        0.018,
        "MATCH ANALYSIS REPORT",
        ha="center",
        color=TEXT_DIM,
        fontsize=8.5,
        fontweight="bold",
        family=FONT_MONO,
    )
    fig.text(
        0.970,
        0.018,
        footer_note,
        ha="right",
        color=TEXT_FAD,
        fontsize=8.5,
        family=FONT_MONO,
    )


def make_themed_figure(w: float = 16, h: float = 10):
    """Create a blank pure-black figure (no header/footer chrome)."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(w, h), facecolor=BG_DARK)
    fig.patch.set_facecolor(BG_DARK)
    return fig


def panel_header_geom(h: float) -> tuple[float, float]:
    """Returns (header_h, body_h) for a panel of total height h, matching
    the same header sizing rule used inside panel_card()/key_insight()."""
    header_h = min(0.052, h * 0.16)
    return header_h, h - header_h


# ── 2. PANEL CARD (hairline border + dark header strip) ────────────────
def panel_card(fig, x, y, w, h, *, title: str, accent: str = C_GOLD, body: bool = True):
    """Panel matching `.panel` + `.panel-head` in the reference CSS:
    flat panel background, 1px hairline border, and a slightly darker
    header strip (with a small accent dot) carrying the title."""
    _panel_rect(fig, x, y, w, h, zorder=-2)

    header_h = min(0.052, h * 0.16)
    header = mpatches.FancyBboxPatch(
        (x, y + h - header_h),
        w,
        header_h,
        boxstyle="round,pad=0.0,rounding_size=0.006",
        transform=fig.transFigure,
        facecolor=BG_HEADER,
        edgecolor=GRID_COL,
        linewidth=1.0,
        zorder=-1,
    )
    fig.add_artist(header)
    # small accent dot in the header, like `.panel-head .ico`
    dot_y = y + h - header_h / 2
    fig.add_artist(
        mpatches.Circle(
            (x + 0.018, dot_y),
            0.0035,
            transform=fig.transFigure,
            facecolor=accent,
            edgecolor="none",
            zorder=0,
        )
    )
    fig.text(
        x + 0.030,
        dot_y,
        title.upper(),
        ha="left",
        va="center",
        color=TEXT_BR,
        fontsize=10.5,
        fontweight="bold",
        family=FONT_MONO,
        zorder=1,
    )

    body_h = h - header_h
    ax = fig.add_axes([x, y, w, body_h])
    ax.set_facecolor(BG_MID)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if body:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    return ax


# ── 3. METRIC CARD ──────────────────────────────────────────────────────
def metric_card(fig, x, y, w, h, *, label: str, value, accent: str, big: bool = True):
    """Flat stat card matching `.stat-card`: hairline border, big mono
    value, small grey mono label underneath."""
    _panel_rect(fig, x, y, w, h, zorder=-1)
    ax = fig.add_axes([x, y, w, h])
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
        color=readable_on(accent, BG_PANEL),
        fontsize=27 if big else 21,
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


def metric_strip(
    fig, *, cards, x: float = 0.04, w: float = 0.92, y: float = 0.075, h: float = 0.095
):
    """cards: list[(label, value, accent_color)]"""
    n = len(cards)
    if n == 0:
        return
    gap = 0.012
    cw = (w - gap * (n - 1)) / n
    for i, (lbl, val, col) in enumerate(cards):
        metric_card(
            fig,
            x + i * (cw + gap),
            y,
            cw,
            h,
            label=lbl,
            value=val,
            accent=col,
            big=True,
        )


# ── 4. KEY INSIGHT (gold-accented callout) ──────────────────────────────
def key_insight(fig, x, y, w, h, *, text: str, wrap: int = 42):
    _panel_rect(fig, x, y, w, h, zorder=-2)
    header_h = min(0.052, h * 0.16)
    header = mpatches.FancyBboxPatch(
        (x, y + h - header_h),
        w,
        header_h,
        boxstyle="round,pad=0.0,rounding_size=0.006",
        transform=fig.transFigure,
        facecolor=BG_HEADER,
        edgecolor=GRID_COL,
        linewidth=1.0,
        zorder=-1,
    )
    fig.add_artist(header)
    dot_y = y + h - header_h / 2
    fig.add_artist(
        mpatches.Circle(
            (x + 0.018, dot_y),
            0.0035,
            transform=fig.transFigure,
            facecolor=C_GOLD,
            edgecolor="none",
            zorder=0,
        )
    )
    fig.text(
        x + 0.030,
        dot_y,
        "KEY INSIGHT",
        ha="left",
        va="center",
        color=TEXT_BR,
        fontsize=10.5,
        fontweight="bold",
        family=FONT_MONO,
        zorder=1,
    )

    body_h = h - header_h
    ax = fig.add_axes([x, y, w, body_h])
    ax.set_facecolor((0, 0, 0, 0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for s in ax.spines.values():
        s.set_visible(False)
    body = "\n".join(_tw.wrap(text, width=wrap))
    ax.text(
        0.06,
        0.86,
        body,
        ha="left",
        va="top",
        color=TEXT_DIM,
        fontsize=11,
        family=FONT_SANS,
        linespacing=1.55,
        transform=ax.transAxes,
    )


# ── 5. THEMED PITCH (subdued lines, flat black) ─────────────────────────
def themed_pitch(
    ax,
    *,
    attacking_only: bool = False,
    line_color: str = "#A8B8CA",
    line_alpha: float = 0.56,
):
    import matplotlib.pyplot as plt

    if line_color in ("#2A2A2A", GRID_COL):
        line_color = "#3A3A3A"
    ax.set_facecolor(BG_PITCH)
    ax.set_aspect("equal")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    lc = dict(color=line_color, lw=1.05, alpha=line_alpha * 0.88, zorder=2)
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
        ax.plot(
            [x0, x0 + sign * 16.5, x0 + sign * 16.5, x0], [21.1, 21.1, 78.9, 78.9], **lc
        )
        ax.plot(
            [x0, x0 + sign * 5.5, x0 + sign * 5.5, x0], [36.8, 36.8, 63.2, 63.2], **lc
        )
        ax.scatter(
            [x0 + sign * 11], [50], color=line_color, s=5, alpha=line_alpha, zorder=2
        )
    # Goals
    for x0 in (0, 100):
        sign = 1 if x0 == 0 else -1
        ax.plot(
            [x0, x0 + sign * 1.5, x0 + sign * 1.5, x0],
            [44, 44, 56, 56],
            color=line_color,
            lw=1.0,
            alpha=line_alpha,
            zorder=2,
        )
    if attacking_only:
        ax.set_xlim(48, 102)


# ── 6. LEGEND CHIPS ─────────────────────────────────────────────────────
def legend_chips(ax, items, y: float = -0.10, fontsize: float = 10):
    """items: list[(label, color)] — small dot + label inline, mono caps,
    matching `.legend-item` in the reference CSS."""
    x = 0.02
    for lbl, col in items:
        ax.text(
            x,
            y,
            "●",
            ha="left",
            va="center",
            color=col,
            fontsize=fontsize + 1,
            transform=ax.transAxes,
        )
        ax.text(
            x + 0.022,
            y,
            lbl.upper(),
            ha="left",
            va="center",
            color=col,
            fontsize=fontsize,
            fontweight="bold",
            family=FONT_MONO,
            transform=ax.transAxes,
        )
        # Approximate width per char to advance x
        x += 0.022 + 0.0085 * len(lbl) + 0.030
