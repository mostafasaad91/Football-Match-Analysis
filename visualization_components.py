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
IS_LIGHT_THEME = _THEME == "light"

# Team colour mode.
#   "kit"   (default) — each side is drawn in its real home-kit colour, taken
#                       from team_palettes.py. Two teams whose kits are too
#                       close, or a kit that is unreadable on the page, fall
#                       back to the fixed roles below.
#   "roles"           — every fixture uses the same two production colours.
# Real kits are the production default; the fixture resolver selects an
# alternate kit whenever the two primaries are too similar.
TEAM_COLOR_MODE = os.environ.get("MATCH_ANALYSIS_TEAM_COLORS", "kit").strip().lower()
USE_REAL_TEAM_KIT_COLORS = TEAM_COLOR_MODE in {"kit", "kits", "real", "real_kits"}
if IS_LIGHT_THEME:
    # LIGHT_PALETTE — "Ink & Burnt Orange" print identity. This block is the
    # single source of truth for the light theme; every light value used by the
    # renderers derives from here.
    #   background   #F5F5F5 · text #333333 (charcoal, never pure black so type
    #   separates from the black team) · secondary text #666666 · pitch lines
    #   #B0B0B0 · dashed reference lines #888888 (grey, never black).
    BG_DARK = "#F5F5F5"
    BG_MID = "#F5F5F5"
    BG_PANEL = "#FFFFFF"
    BG_HEADER = "#EDEDED"
    BG_PITCH = "#F5F5F5"
    GRID_COL = "#CCCCCC"
    GRID_SOFT = "#E0E0E0"

    TEXT_BR = "#333333"
    TEXT_MAIN = "#333333"
    TEXT_DIM = "#666666"
    TEXT_FAD = "#999999"
    TEXT_FAINT = "#BBBBBB"
else:
    # "Pure Black" identity — matches the reference HTML dashboards exactly.
    BG_DARK = "#000000"  # true black page background
    BG_MID = "#0a0a0a"  # panel background
    BG_PANEL = "#0a0a0a"  # panel background (alias kept for compatibility)
    BG_HEADER = "#101010"  # panel header-strip background
    BG_PITCH = "#000000"  # pitch surface is the page itself: pure black
    GRID_COL = "#1c1c1c"  # hairline border
    GRID_SOFT = "#141414"  # softer divider/grid line

    TEXT_BR = "#FFFFFF"
    TEXT_MAIN = "#FFFFFF"
    TEXT_DIM = "#9A9A9A"
    TEXT_FAD = "#5A5A5A"
    TEXT_FAINT = "#3A3A3A"

# Canonical production matchup palette. Every fixture uses the same two roles
# so a reader learns one stable visual language instead of relearning kit
# colours from match to match. Each theme carries its own pair because the
# AMOLED roles (electric blue / true yellow) are unreadable on white paper.
if IS_LIGHT_THEME:
    C_HOME = "#0A0A0A"  # ink black — first-listed team role
    C_AWAY = "#E76F51"  # burnt orange — second-listed team role (no reds)
    C_GOLD = "#14505C"  # petrol accent — must stay far from the away orange
    C_MAGENTA = "#86198F"
    C_ACCENT = "#1D4ED8"
    C_LIME = "#15803D"
    PITCH_LINE = "#B0B0B0"
    DASHED_LINE = "#888888"  # dashed refs are grey, never black (team clash)
else:
    C_HOME = "#2F5BFF"
    C_AWAY = "#FFD400"
    C_GOLD = "#FFC23C"
    C_MAGENTA = "#E879F9"
    C_ACCENT = "#38BDF8"
    C_LIME = "#22C55E"
    # Pitch markings are drawn in white on the pure-black page, the way a real
    # broadcast pitch reads. Grey markings disappeared into the background and
    # made every map look washed out; white lines at a controlled alpha give
    # the outline without competing with the data layer on top of it.
    PITCH_LINE = "#FFFFFF"
    DASHED_LINE = "#3A3A3A"

# Alpha applied to pitch markings so white lines frame the pitch without
# out-shouting the marks plotted on them.
PITCH_LINE_ALPHA = 0.42 if IS_LIGHT_THEME else 0.68
PITCH_LINE_WIDTH = 1.25

# ── Shot-outcome palette ────────────────────────────────────────────────
# Outcome, not team, is what a shot map encodes: one fixed colour per result
# so the reader learns the key once and it holds for every match and player.
SHOT_GOAL = "#D62728"  # red
SHOT_SAVED = "#2077B4"  # steel blue
SHOT_MISS = "#F0A05A"  # sand orange
SHOT_BLOCKED = "#B4B4B4"  # neutral grey
SHOT_POST = "#8B5FBF"  # violet
SHOT_OWN_GOAL = "#FF00FF"  # magenta — deliberately jarring, never ambiguous

SHOT_OUTCOME_COLORS: dict[str, str] = {
    "goal": SHOT_GOAL,
    "saved": SHOT_SAVED,
    "savedshot": SHOT_SAVED,
    "attemptsaved": SHOT_SAVED,
    "miss": SHOT_MISS,
    "missed": SHOT_MISS,
    "missedshots": SHOT_MISS,
    "blocked": SHOT_BLOCKED,
    "shotblocked": SHOT_BLOCKED,
    "post": SHOT_POST,
    "woodwork": SHOT_POST,
    "shotonpost": SHOT_POST,
    "owngoal": SHOT_OWN_GOAL,
}


def shot_outcome_color(outcome: str, fallback: str = SHOT_BLOCKED) -> str:
    """Map any provider spelling of a shot outcome to its fixed colour."""
    key = "".join(ch for ch in str(outcome or "").lower() if ch.isalnum())
    return SHOT_OUTCOME_COLORS.get(key, fallback)


# Shared semantic styles for path-heavy pitch visuals. Colour identifies the
# event outcome; line style keeps the distinction available in grayscale.
EVENT_SUCCESS = C_HOME
EVENT_FAILURE = C_AWAY
EVENT_NEUTRAL = "#B8BDC8" if not IS_LIGHT_THEME else "#7A828C"
# The highlight layer must contrast against the page AND against both team
# colours. On paper that rules out white, charcoal (black team) and any warm
# orange (away team) — deep petrol is the only remaining strong hue.
EVENT_HIGHLIGHT = "#14505C" if IS_LIGHT_THEME else TEXT_BR
QUIET_DASH = (0, (2.2, 3.2))
FAILURE_DASH = (0, (4.0, 3.0))

# Wording for the highlight layer, so legends stay truthful per theme.
HIGHLIGHT_LABEL = "Petrol solid" if IS_LIGHT_THEME else "White solid"


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


# ── Text drawn on top of a coloured fill ────────────────────────────────
# Three tiers rather than a fixed white, because the fill under a label is a
# team colour and team colours run from near-black to near-white. A white
# number on Juventus silver or on the top of a light heat ramp is invisible.
TEXT_ON_LIGHT = "#0A0A0A"  # near-black, for light fills
TEXT_ON_MID = "#111827"  # ink, for mid-value fills that still read dark
TEXT_ON_DARK = "#FFFFFF"  # white, for dark fills


def text_on_fill(fill: str) -> str:
    """Return the most readable label colour for text drawn ON ``fill``.

    Picks whichever tier gives the higher contrast ratio against the fill
    itself — not against the page — so a label never inherits the page's white
    and vanish into a light cell.
    """
    try:
        candidates = (TEXT_ON_DARK, TEXT_ON_LIGHT, TEXT_ON_MID)
        return max(candidates, key=lambda candidate: contrast_ratio(candidate, fill))
    except Exception:
        return TEXT_ON_DARK


def label_outline(fill: str, linewidth: float = 1.6) -> list:
    """Return a path-effect stroke for labels over mid-luminance fills.

    Where no text colour clears a comfortable ratio on its own — mid greys and
    mid-saturation kits — a thin stroke in the opposite tier keeps the digits
    legible without changing their colour.
    """
    chosen = text_on_fill(fill)
    if contrast_ratio(chosen, fill) >= 4.5:
        return []
    opposite = TEXT_ON_DARK if chosen != TEXT_ON_DARK else TEXT_ON_LIGHT
    return [pe.withStroke(linewidth=linewidth, foreground=opposite)]


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
        # Team identity belongs to the card accent/mark. Exact values remain
        # white everywhere so neither team receives accidental emphasis.
        color=TEXT_BR,
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
    line_color: str | None = None,
    line_alpha: float | None = None,
):
    import matplotlib.pyplot as plt

    line_color = line_color or PITCH_LINE
    line_alpha = PITCH_LINE_ALPHA if line_alpha is None else line_alpha
    # Legacy call sites pass their own near-black greys, which vanish on the
    # pure-black page. Any such value is promoted to the shared pitch line.
    if line_color.upper() in ("#2A2A2A", "#3A3A3A", "#5C6470", "#A8B8CA", GRID_COL.upper()):
        line_color = PITCH_LINE
    ax.set_facecolor(BG_PITCH)
    ax.set_aspect("equal")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    lc = dict(color=line_color, lw=PITCH_LINE_WIDTH, alpha=line_alpha, zorder=2)
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


# ── Saving a figure when the machine is out of commit ────────────────────────
#
# A four-hour run died on its first PNG with "MemoryError: bad allocation" from
# matplotlib's Agg renderer. The data was ordinary — 1450 events, minutes 0 to
# 95 — and the machine was the problem: 22.5 GB of a 23.7 GB commit limit was
# already spoken for, so a fourteen-by-eight figure at 155 dpi could not get
# the ten megabytes its buffer needs.
#
# That is a fair reason for one image to fail and a bad reason to lose the
# whole package. bbox_inches="tight" renders twice, and every step down in dpi
# quarters the buffer, so a save that cannot fit is retried smaller before it
# is allowed to raise.
SAVE_DPI_STEPS = (1.0, 0.72, 0.5)


def save_figure(fig, path, *, dpi=155, **kwargs):
    """Write a figure, stepping the resolution down rather than dying.

    Returns the dpi it managed. A MemoryError at the smallest step is re-raised
    — at that point the machine genuinely cannot draw and a silent half-sized
    image would be worse than the failure.
    """
    last = None
    for step in SAVE_DPI_STEPS:
        try:
            fig.savefig(path, dpi=max(int(dpi * step), 40), **kwargs)
            return max(int(dpi * step), 40)
        except MemoryError as error:            # pragma: no cover - machine state
            last = error
            # The half-built buffer is still held until the next allocation
            # fails too, and the retry needs the room.
            import gc

            gc.collect()
    raise last
