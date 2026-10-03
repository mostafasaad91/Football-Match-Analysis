from __future__ import annotations

from football_analysis.paths import PROJECT_ROOT

import colorsys
import gc
import hashlib
import json
import math
import os

from football_analysis.metrics.frame_values import surname as _surname
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as path_effects
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Patch, Polygon, Rectangle, Wedge
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from football_analysis.visuals.visualization_components import (
    slope_label_offsets,
    C_AWAY,
    C_HOME,
    EVENT_FAILURE,
    EVENT_HIGHLIGHT,
    EVENT_NEUTRAL,
    EVENT_SUCCESS,
    FAILURE_DASH,
    HIGHLIGHT_LABEL,
    IS_LIGHT_THEME,
    PITCH_LINE,
    QUIET_DASH,
    SHOT_BLOCKED,
    SHOT_GOAL,
    SHOT_MISS,
    SHOT_POST,
    SHOT_SAVED,
    USE_REAL_TEAM_KIT_COLORS,
    label_outline,
    network_link_palette,
    text_on_fill,
)

from football_analysis.visuals import visual_redesign_preview as base
from football_analysis.visuals.typography import display
from football_analysis.metrics.match_metrics import (
    advanced_metrics_frames,
    defensive_line_height,
    duel_map,
    goalkeeper_distribution,
    goalkeeper_shot_stopping,
    line_breaking_passes,
    network_centrality,
    pass_length_profile,
    average_positions as team_average_positions,
    field_tilt_timeline,
    goal_origin_chains,
    pitch_control,
    pressing_triggers,
    receptions_between_lines,
    rest_defence_structure,
    second_ball_recovery,
    sequence_typology,
    substitution_impact,
    switches_of_play,
    time_to_progress,
    player_action_value,
    post_shot_xg,
    press_resistance,
    set_piece_breakdown,
    shot_placement_zones,
    team_compactness,
    turnover_events,
    win_probability,
    xg_momentum,
    build_possessions,
    box_entry_mask,
    cross_mask,
    deep_completion_mask,
    defensive_block_events,
    defensive_blocks_count,
    final_third_entry_mask,
    fouls_committed_count,
    fouls_committed_mask,
    high_regain_events,
    progressive_pass_mask,
    touch_mask,
)
from football_analysis.reports.match_report import compute_ppda_both


ROOT = PROJECT_ROOT
MATCH_KEY = "France_vs_England_4-6"
OUT = ROOT / "output" / MATCH_KEY
DATA = ROOT / "sample_data" / MATCH_KEY

BG = base.BG
PANEL = base.PANEL
PANEL_2 = base.PANEL_2
TEXT = base.TEXT
MUTED = base.MUTED
GRID = base.GRID
HOME = base.HOME
AWAY = base.AWAY
VALUE = base.VALUE
FOCUS = base.FOCUS
NEUTRAL = base.NEUTRAL
# The went-off ring is drawn in FOCUS, which is white on AMOLED and petrol on
# paper. Keep the legend wording honest per theme.
_FOCUS_WORD = "petrol" if IS_LIGHT_THEME else "white"
# Translucent shaded regions lose presence on a light page; lift their alpha.
_SHADE_ALPHA = 0.24 if IS_LIGHT_THEME else 0.16
# Five lanes, five hues, left to right across the page. Away from red and sky
# blue so no lane reads as a team, and darker on the light page.
ZONE_LANE_COLOURS = (
    ("#7A4FD8", "#1F7BD8", "#0E9486", "#C98A00", "#D2452F")
    if IS_LIGHT_THEME
    else ("#A77BFF", "#4EA8FF", "#2FD3BE", "#FFC247", "#FF7A5C")
)
_HATCH_ALPHA = 0.20 if IS_LIGHT_THEME else 0.13

HOME_ID = base.HOME_ID
AWAY_ID = base.AWAY_ID
HOME_NAME = base.HOME_NAME
AWAY_NAME = base.AWAY_NAME
TEAM_COLOR = base.TEAM_COLOR
TEAM_NAME = base.TEAM_NAME
MATCH_SCORE = "4-6"

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Overlay accents for marks drawn on top of a team-colour heatmap.
_HEATMAP_ACCENT_WARM = "#FFC23C"
_HEATMAP_ACCENT_COOL = "#38BDF8"


def _safe_slug(value: str) -> str:
    """Return a filesystem-safe, stable slug for team-labelled exports."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "team").strip().lower())
    return slug.strip("_") or "team"


def _team_slug(team_id: int) -> str:
    return _safe_slug(TEAM_NAME.get(team_id, str(team_id)))


def _display_score(value: object) -> str:
    """Normalize provider score strings such as ``*1 : 0`` for headers."""
    numbers = re.findall(r"\d+", str(value or ""))
    if len(numbers) >= 2:
        return f"{numbers[0]} — {numbers[1]}"
    return str(value or "-").lstrip("*").strip()


def _team_series_palette(team_color: str) -> tuple[str, str]:
    """Return primary and secondary shades from one team's identity colour."""
    primary = team_color
    try:
        primary_rgb = np.asarray(mcolors.to_rgb(primary), dtype=float)
    except ValueError:
        primary = "#94A3B8"
        primary_rgb = np.asarray(mcolors.to_rgb(primary), dtype=float)
    # A light tint stays recognisably within the same team identity while
    # separating origins from destinations on the pure-black background.
    secondary_rgb = primary_rgb * 0.58 + np.ones(3) * 0.42
    return mcolors.to_hex(primary_rgb), mcolors.to_hex(secondary_rgb)


# Minimum contrast a drawn mark must reach against the page. WCAG puts the
# floor for non-text graphics at 3:1; this sits above it so a thin arrow or a
# 1px network link still reads, not only a filled bar.
MARK_CONTRAST_FLOOR = 3.6


def _relative_luminance(rgb) -> float:
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_on_bg(rgb) -> float:
    bright, dark = sorted((_relative_luminance(rgb), _relative_luminance(mcolors.to_rgb(BG))))
    return (dark + 0.05) / (bright + 0.05)


def lift_to_floor(color: str, floor: float = MARK_CONTRAST_FLOOR) -> str:
    """Move a colour's lightness until it clears the floor, keeping its hue.

    Real kit colours are frequently dark — navy, claret, maroon, near-black.
    Drawn unmodified on a pure black page they measure under 2:1: PSG's
    #004170 rendered at 1.99 and Aston Villa's #7A003C at 1.89, roughly half
    the readable minimum, which is why arrows and network links looked muted.
    Hue and saturation are preserved, so the side is still recognisably itself.

    Which way lightness moves is decided by the page, not assumed. Written for
    the black page it only ever searched upward, which is correct there and
    exactly wrong on the light one: Manchester City's #6CABDD and Juventus'
    #DCE3EC were both driven to pure white against #F5F5F5, a contrast of
    1.09, and disappeared. A light page needs the same colours darkened.
    """
    try:
        rgb = mcolors.to_rgb(color)
    except ValueError:
        return color
    if _contrast_on_bg(rgb) >= floor:
        # Returned verbatim, not round-tripped through to_hex, so a colour that
        # already reads keeps the exact string the palette defined — including
        # its case, which callers compare against.
        return color

    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    # Move away from the page: brighten on a dark ground, darken on a light one.
    target = 0.0 if _relative_luminance(mcolors.to_rgb(BG)) > 0.5 else 1.0
    # ``far`` is the end known to satisfy the floor, ``near`` the end known not
    # to; which of the two is numerically larger depends on the page.
    near, far = lightness, target
    for _ in range(24):
        mid = (near + far) / 2
        if _contrast_on_bg(colorsys.hls_to_rgb(hue, mid, saturation)) >= floor:
            far = mid
        else:
            near = mid

    # The search runs on floats, but the returned colour is an 8-bit hex. That
    # rounding can drop the result a hundredth under the floor, which makes a
    # second call move it again — so step until the *rounded* value clears.
    step = 0.004 if target > lightness else -0.004
    for _ in range(12):
        hex_value = mcolors.to_hex(colorsys.hls_to_rgb(hue, far, saturation))
        if _contrast_on_bg(mcolors.to_rgb(hex_value)) >= floor or far == target:
            return hex_value
        far = float(np.clip(far + step, 0.0, 1.0))
    return hex_value


def _team_mark_color(team_id: int) -> str:
    """The team's colour as drawn on the pitch, lifted to stay legible.

    The chrome — headers, rules, panel text — keeps the exact kit value,
    because it sits on a panel rather than on the black ground.
    """
    return lift_to_floor(TEAM_COLOR.get(team_id, "#94A3B8"))


def _on_team_heatmap_accent(team_color: str) -> str:
    """Return an accent that stays visible ON a team-colour heatmap.

    Heatmap ramps run black → team colour, so any overlay drawn in the team's
    own colour disappears into the hot cells. Pick a hue far from the team's
    instead: amber by default, cyan when the team itself is warm/amber.
    """
    try:
        team_rgb = np.asarray(mcolors.to_rgb(team_color), dtype=float)
    except ValueError:
        return _HEATMAP_ACCENT_WARM
    warm = np.asarray(mcolors.to_rgb(_HEATMAP_ACCENT_WARM), dtype=float)
    if float(np.linalg.norm(team_rgb - warm)) < 0.45:
        return _HEATMAP_ACCENT_COOL
    return _HEATMAP_ACCENT_WARM


def _team_density_palette(team_id: int) -> tuple[str, str, str]:
    """Sequential heatmap palette anchored to the visual's own team role."""
    return BG, PANEL_2, _team_mark_color(team_id)


def _resolve_fixture_colors(match_info: dict) -> tuple[str, str]:
    """Pick the two display colours for a fixture.

    Kit mode uses the colours the caller resolved (``home_color`` /
    ``away_color``), but only when both are present, parseable and visibly
    different from each other — a renderer that draws two sides in the same
    colour is worse than one that ignores the kits. Anything short of that
    falls back to the fixed role pair.
    """
    if not USE_REAL_TEAM_KIT_COLORS:
        return C_HOME, C_AWAY

    home = str(match_info.get("home_color") or "").strip()
    away = str(match_info.get("away_color") or "").strip()
    if not home or not away:
        return C_HOME, C_AWAY
    try:
        home_rgb = np.asarray(mcolors.to_rgb(home), dtype=float)
        away_rgb = np.asarray(mcolors.to_rgb(away), dtype=float)
    except ValueError:
        return C_HOME, C_AWAY
    if float(np.linalg.norm(home_rgb - away_rgb)) < 0.22:
        return C_HOME, C_AWAY
    # Return the caller's own strings, not a normalised form: downstream code
    # and tests compare these against the palette constants by value.
    return home, away


def configure_match(match_info: dict, output_dir: Path | str) -> None:
    """Inject one fixture's identity into the reusable AMOLED renderer.

    The sample renderer originally carried France/England module constants.
    Production calls now configure the same renderer from parsed match data,
    so every fixture receives the new identity rather than the legacy path.
    """
    global OUT, MATCH_KEY, MATCH_SCORE
    global HOME_ID, AWAY_ID, HOME_NAME, AWAY_NAME, HOME, AWAY
    global TEAM_COLOR, TEAM_NAME

    HOME_ID = int(match_info["home_id"])
    AWAY_ID = int(match_info["away_id"])
    HOME_NAME = str(match_info.get("home_name") or "Home")
    AWAY_NAME = str(match_info.get("away_name") or "Away")
    # In kit mode each side keeps the real colours the caller resolved through
    # choose_matchup_colors (already clash- and contrast-checked). In roles mode
    # the visual roles are fixed instead: first-listed team is electric blue,
    # second-listed team is true yellow, for every fixture.
    # Lift once, here, rather than at each draw call. Most visuals reach for the
    # HOME/AWAY globals directly instead of going through _team_mark_color, so
    # lifting only there left arrows, bars and heatmap ramps on the raw kit
    # value — PSG's navy measured 1.99:1 against the black page and Aston
    # Villa's claret 1.89:1, against a readable minimum of 3.
    raw_home, raw_away = _resolve_fixture_colors(match_info)
    HOME, AWAY = (lift_to_floor(colour) for colour in (raw_home, raw_away))
    # The report cover draws kit colour inside its bars, where the colour the
    # club wears is the point, rather than the one lifted for text on the page.
    # It also names the venue, the managers and the shape, which only the parse
    # knows; build_pdf passes both on.
    global _MATCH_INFO, _KIT_COLORS
    _MATCH_INFO = dict(match_info or {})
    _KIT_COLORS = (raw_home, raw_away)
    MATCH_SCORE = _display_score(match_info.get("score"))
    OUT = Path(output_dir).resolve()
    MATCH_KEY = OUT.name
    TEAM_COLOR = {HOME_ID: HOME, AWAY_ID: AWAY}
    TEAM_NAME = {HOME_ID: HOME_NAME, AWAY_ID: AWAY_NAME}

    # Shared sample helpers draw comparison pages and headers from their own
    # module globals, so update them at the same configuration boundary.
    base.HOME_ID = HOME_ID
    base.AWAY_ID = AWAY_ID
    base.HOME_NAME = HOME_NAME
    base.AWAY_NAME = AWAY_NAME
    base.HOME = HOME
    base.AWAY = AWAY
    base.TEAM_COLOR = {HOME_ID: base.HOME, AWAY_ID: base.AWAY}
    base.TEAM_NAME = dict(TEAM_NAME)
    base.MATCH_SCORE = MATCH_SCORE.replace("-", "—")
    base.MATCH_KEY = MATCH_KEY
    base.OUT_DIR = OUT
    base.COMPARE_DIR = OUT / "comparisons"


def as_bool(series: pd.Series) -> pd.Series:
    return base._bool(series)


def load_all():
    return base.load_data()


def attack_xy(x, y):
    """Pitch coordinates for a board that attacks up the page.

    The feed numbers the width from the attacking side's RIGHT touchline: a right
    winger's touches average y near 18 and a left back's near 81. Drawn
    straight, a side's right flank landed on the left of every board that used
    this, the mirror image of the pass network beside it and of the lane names
    the rest of the package uses. The lateral axis is corrected here, once, so
    the right flank is on the right of the page everywhere.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    display_x = (50.0 - y_arr) * (PITCH_WIDTH / 100.0)
    display_y = x_arr * (PITCH_LENGTH / 100.0)
    return display_x, display_y


def player_position_xy(x, y):
    """Display positional maps with the provider's lateral axis corrected."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    display_x = (50.0 - y_arr) * (PITCH_WIDTH / 100.0)
    display_y = x_arr * (PITCH_LENGTH / 100.0)
    return display_x, display_y


def pitch_axes(title: str, subtitle: str):
    fig = plt.figure(figsize=(12, 9), facecolor=BG)
    active_team = (
        HOME_NAME
        if HOME_NAME.lower() in title.lower()
        else (AWAY_NAME if AWAY_NAME.lower() in title.lower() else None)
    )
    base.amoled_header(fig, title, subtitle, active_team=active_team)
    pitch = fig.add_axes([0.075, 0.105, 0.48, 0.72])
    side = fig.add_axes([0.615, 0.145, 0.325, 0.62])
    side.set_facecolor(PANEL)
    for spine in side.spines.values():
        spine.set_color(GRID)
    side.set_xticks([])
    side.set_yticks([])
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    fig.text(
        0.945,
        0.035,
        "FULL VISUAL REDESIGN · REAL MATCH DATA",
        ha="right",
        fontsize=8,
        color=NEUTRAL,
    )
    return fig, pitch, side


def draw_long_pitch(ax, line_color=PITCH_LINE, lw=1.15):
    half_w = PITCH_WIDTH / 2
    ax.set_xlim(-half_w - 5, half_w + 5)
    ax.set_ylim(-3, PITCH_LENGTH + 3)
    ax.set_aspect("equal")
    ax.add_patch(
        Rectangle((-half_w, 0), PITCH_WIDTH, PITCH_LENGTH, fill=False, ec=line_color, lw=lw)
    )
    ax.plot([-half_w, half_w], [PITCH_LENGTH / 2, PITCH_LENGTH / 2], color=line_color, lw=lw)
    ax.add_patch(
        Circle((0, PITCH_LENGTH / 2), PITCH_LENGTH * 0.0915, fill=False, ec=line_color, lw=lw)
    )
    penalty_w = PITCH_WIDTH * 0.595
    six_w = PITCH_WIDTH * 0.265
    box_l = PITCH_LENGTH * 0.157
    six_l = PITCH_LENGTH * 0.052
    for y0, direction in [(0, 1), (PITCH_LENGTH, -1)]:
        ax.add_patch(
            Rectangle(
                (-penalty_w / 2, y0 if direction > 0 else y0 - box_l),
                penalty_w,
                box_l,
                fill=False,
                ec=line_color,
                lw=lw,
            )
        )
        ax.add_patch(
            Rectangle(
                (-six_w / 2, y0 if direction > 0 else y0 - six_l),
                six_w,
                six_l,
                fill=False,
                ec=line_color,
                lw=lw,
            )
        )
        spot_y = y0 + direction * PITCH_LENGTH * 0.105
        ax.scatter([0], [spot_y], s=7, color=line_color)
        arc_center = y0 + direction * box_l
        if direction > 0:
            ax.add_patch(Arc((0, spot_y), 18.3, 18.3, theta1=37, theta2=143, ec=line_color, lw=lw))
        else:
            ax.add_patch(Arc((0, spot_y), 18.3, 18.3, theta1=217, theta2=323, ec=line_color, lw=lw))
    ax.annotate(
        "ATTACK",
        xy=(0, PITCH_LENGTH + 1.5),
        ha="center",
        va="bottom",
        color=FOCUS,
        fontsize=8,
        fontweight="bold",
    )
    ax.axis("off")


def crop_to_attack(pitch, side, low, bottom_in=1.1):
    """Show only the pitch from ``low`` (a distance up the pitch) to the goal line.

    Boards about the attacking end drew the whole pitch and used its last third,
    so most of the picture was empty grass and every mark was small. The pitch is
    cropped, widened, and the figure is cut down to fit it, so the part that
    holds data is larger and nothing below it is blank.

    ``bottom_in`` is the room, in inches, kept under the pitch for a key or a
    strip. Returns the figure fraction where the pitch now ends.
    """
    top = PITCH_LENGTH + 3.0
    low = max(-3.0, low)
    pitch.set_ylim(low, top)
    figure = pitch.figure
    fig_w = figure.get_size_inches()[0]
    span_x = (PITCH_WIDTH / 2 + 5) * 2
    width = 0.57
    pitch_in = min(6.4, width * fig_w * (top - low) / span_x)
    header_in = 1.45
    height_in = header_in + pitch_in + bottom_in
    figure.set_size_inches(fig_w, height_in)
    pitch_top = 1 - header_in / height_in
    pitch_bottom = pitch_top - pitch_in / height_in
    pitch.set_position([0.04, pitch_bottom, width, pitch_in / height_in])
    side.set_position([0.65, pitch_bottom, 0.295, pitch_in / height_in])
    return pitch_bottom


def side_title(ax, text: str):
    ax.text(0.07, 0.94, text.upper(), color=MUTED, fontsize=9, fontweight="bold", va="top")
    ax.plot([0.07, 0.93], [0.89, 0.89], color=GRID, lw=1)


def side_kpis(ax, items: list[tuple[str, str]], start=0.82, gap=0.14) -> float:
    """Draw the stacked KPI block and return the y its last value reaches.

    Callers used to place the next section at a hand-picked y, which held only
    for the number of KPIs they happened to have when it was written: a fourth
    KPI pushed its 16pt value straight through the heading below it. Returning
    the bottom lets the next block start from where this one actually ended.
    """
    bottom = start
    for idx, (label, value) in enumerate(items):
        y = start - idx * gap
        if y < 0.06:
            break
        ax.text(0.08, y, label.upper(), color=MUTED, fontsize=7.5, fontweight="bold", va="top")
        ax.text(0.08, y - 0.052, str(value), color=TEXT, va="top", **display(23))
        bottom = y - 0.055 - 0.045  # value baseline plus its own height
    return bottom


def pitch_legend(ax, items, ncol: int | None = None, y: float = -0.075):
    """Draw a key beneath a pitch for anything the marks encode.

    ``items`` are (kind, colour, label) where kind is "patch" for a filled
    swatch, or any matplotlib marker string for a point. Several visuals used
    colour or shape to carry meaning and then never said what the meaning was —
    a reader looking at Pitch Control had no way to learn that blue is one side,
    silver the other, and dark the space neither held.
    """
    handles = []
    for kind, colour, label in items:
        if kind == "patch":
            # A near-black swatch on a black page is invisible without an
            # outline — the "contested" key read as a gap in the legend.
            handles.append(
                Patch(facecolor=colour, edgecolor=EVENT_NEUTRAL, linewidth=0.7, label=label)
            )
        else:
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=kind,
                    markerfacecolor=colour,
                    markeredgecolor=BG,
                    markeredgewidth=0.9,
                    markersize=7,
                    label=label,
                )
            )
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol or min(len(handles), 4),
        frameon=False,
        labelcolor=TEXT,
        fontsize=7.5,
        handletextpad=0.6,
        columnspacing=1.8,
    )


def side_rows(
    ax,
    rows: list[tuple[str, str]],
    start=0.82,
    gap=0.085,
    value_color=TEXT,
    label_color=TEXT,
    label_weight="normal",
):
    for idx, (label, value) in enumerate(rows):
        y = start - idx * gap
        if y < 0.04:
            break
        ax.text(
            0.08,
            y,
            str(label),
            color=label_color,
            fontsize=8.5,
            fontweight=label_weight,
            va="center",
        )
        ax.text(
            0.92,
            y,
            str(value),
            color=value_color,
            fontsize=8.5,
            fontweight="bold",
            ha="right",
            va="center",
        )
        ax.plot([0.08, 0.92], [y - gap * 0.45, y - gap * 0.45], color=GRID, lw=0.55, alpha=0.7)


def save(fig, filename: str) -> Path:
    if not getattr(fig, "_amoled_header_applied", False):
        candidates = []
        for item in fig.texts:
            try:
                if item.get_visible() and item.get_position()[1] >= 0.84:
                    candidates.append(item)
            except Exception:
                continue
        title_item = max(candidates, key=lambda item: float(item.get_fontsize()), default=None)
        title = (
            title_item.get_text()
            if title_item is not None
            else filename.rsplit(".", 1)[0].replace("_", " ").title()
        )
        subtitle_items = [
            item for item in candidates if item is not title_item and item.get_text().strip()
        ]
        subtitle = (
            subtitle_items[0].get_text()
            if subtitle_items
            else f"{HOME_NAME} vs {AWAY_NAME} · real match data"
        )
        for item in candidates:
            item.set_visible(False)
        active_team = (
            HOME_NAME
            if HOME_NAME.lower() in title.lower()
            else (AWAY_NAME if AWAY_NAME.lower() in title.lower() else None)
        )
        base.amoled_header(fig, title, subtitle, active_team=active_team)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    from football_analysis.visuals.visual_guide import guide
    import textwrap

    fig.text(0.055, -0.015, textwrap.fill(guide(filename), 140), color=MUTED, fontsize=9, va="top")
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.16, facecolor=BG)
    finally:
        # Long match packages can contain 70+ figures.  Release every canvas
        # immediately so Windows does not retain Agg buffers until process exit.
        fig.clear()
        plt.close(fig)
        gc.collect()
    return path


def compact_player_label(name: str, limit: int = 7) -> str:
    """Return a readable surname, shortened only as far as the space demands.

    The default suits a name written next to a pitch marker, where anything
    longer starts colliding with its neighbours. Side-panel rows have most of
    a column to themselves and pass a larger limit — truncating "Locatelli" to
    "Locate…" there threw away a legible name to save space nothing needed.
    """
    surname = _surname(name, "?")
    return surname if len(surname) <= limit else f"{surname[: limit - 1]}…"


def shirt_number_map(players) -> dict[str, str]:
    """Return {player name: shirt number} for labelling nodes."""
    numbers: dict[str, str] = {}
    if players is None or getattr(players, "empty", True):
        return numbers
    if "name" not in players.columns or "shirt_no" not in players.columns:
        return numbers
    for row in players.itertuples():
        shirt = pd.to_numeric(pd.Series([row.shirt_no]), errors="coerce").iloc[0]
        if pd.notna(shirt):
            numbers[str(row.name)] = str(int(shirt))
    return numbers


# A name drawn at fontsize 5.9 on the long pitch measures about 0.48 pitch
# units per character either side of centre, and roughly 1.4 units tall. Good
# enough to keep a label off a neighbouring marker, which is all the placement
# search below needs to decide.
_LABEL_HALF_WIDTH_PER_CHAR = 0.48
_LABEL_HALF_HEIGHT = 1.4


def draw_node_label(
    ax,
    x: float,
    y: float,
    name: str,
    touches: float,
    max_touch: float,
    node_color: str | None = None,
    shirt: str | None = None,
    node_radius: float = 2.6,
    neighbours: "tuple[tuple[float, float, float], ...]" = (),
):
    """Put the shirt number inside the node and the player's name beside it.

    A surname squeezed inside the marker has to shrink to fit and gets clipped
    on longer names. A number always fits at a readable size, and the name then
    has the space outside the node to be written in full.

    ``neighbours`` carries every *other* node as (x, y, radius). Writing the
    name above its own marker clears that marker but says nothing about the one
    sitting just above it, which is how a midfield pair ends up with one man's
    name printed across the other's circle. With the neighbours known, the four
    sides are scored by how far the drawn label lands from any other marker and
    the roomiest one wins; above still wins ties, so an uncrowded network looks
    exactly as it did.
    """
    ratio = float(touches) / max(float(max_touch), 1.0)
    fill = node_color or BG
    inside = text_on_fill(fill)

    if shirt:
        number = ax.text(
            x,
            y,
            str(shirt),
            color=inside,
            fontsize=6.6 if ratio >= 0.25 else 5.8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=7,
            clip_on=True,
        )
        number.set_path_effects(
            [path_effects.withStroke(linewidth=1.6, foreground=fill, alpha=0.6)]
        )

    label = compact_player_label(name) if not shirt else _surname(name)[:12]

    # Offset, alignment, and where the label's centre ends up relative to the
    # anchor — the last part is what makes the clearance test meaningful, since
    # a centred label extends half its width to each side.
    half_width = _LABEL_HALF_WIDTH_PER_CHAR * len(label)
    placements = (
        (0.0, node_radius, "center", "bottom", 0.0, _LABEL_HALF_HEIGHT),
        (0.0, -node_radius, "center", "top", 0.0, -_LABEL_HALF_HEIGHT),
        (node_radius, 0.0, "left", "center", half_width, 0.0),
        (-node_radius, 0.0, "right", "center", -half_width, 0.0),
    )

    dx, dy, ha, va = placements[0][:4]
    if neighbours:
        best_score = None
        for cand_dx, cand_dy, cand_ha, cand_va, box_dx, box_dy in placements:
            centre_x = x + cand_dx + box_dx
            centre_y = y + cand_dy + box_dy
            score = min(
                math.hypot(centre_x - other_x, centre_y - other_y) - other_r
                for other_x, other_y, other_r in neighbours
            )
            # Clearing the neighbours is worthless if the label then runs off
            # the pitch: the axis clips it and the name loses its first letter,
            # which is how a wide player ended up labelled "ostic". Overflow is
            # penalised rather than forbidden so a node with no clean side
            # still gets the least bad one.
            overflow = max(0.0, abs(centre_x) + half_width - PITCH_WIDTH / 2)
            overflow += max(0.0, _LABEL_HALF_HEIGHT - centre_y)
            overflow += max(0.0, centre_y + _LABEL_HALF_HEIGHT - PITCH_LENGTH)
            score -= 10.0 * overflow
            if best_score is None or score > best_score + 1e-9:
                best_score = score
                dx, dy, ha, va = cand_dx, cand_dy, cand_ha, cand_va

    name_text = ax.text(
        x + dx,
        y + dy,
        label,
        color=TEXT,
        fontsize=5.9,
        fontweight="bold",
        ha=ha,
        va=va,
        zorder=7,
        clip_on=True,
    )
    name_text.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground=BG, alpha=0.95)])


def _role_fallback_position(position: str) -> tuple[float, float]:
    role = str(position or "").upper()
    x = 50.0
    if "GK" in role:
        x = 8.0
    elif role in {"DC", "DL", "DR", "DLC", "DRC"} or role.startswith("D"):
        x = 30.0
    elif "DMC" in role:
        x = 43.0
    elif role in {"MC", "ML", "MR"} or role.startswith("M"):
        x = 55.0
    elif "AM" in role:
        x = 68.0
    elif role in {"FW", "ST", "CF"} or "FW" in role:
        x = 80.0
    y = 50.0
    if "L" in role and "LC" not in role:
        y = 22.0
    elif "R" in role and "RC" not in role:
        y = 78.0
    return x, y


def _network_node_radius(touches: float, max_touch: float) -> float:
    """Marker radius in pitch units for a node sized by touches.

    Marker area is set in points squared; the pitch axis runs at about 4.2
    points per unit, so convert before using it as a pitch-space offset.
    """
    area = 260 + 640 * float(touches) / max(float(max_touch), 1.0)
    return math.sqrt(area / math.pi) / 4.2 + 0.7


def _node_neighbours(
    display: dict[str, tuple[float, float, float]], radii: dict[str, float], exclude: str
):
    """Every node except ``exclude``, as the (x, y, radius) triples the label
    placement search needs."""
    return tuple(
        (float(x), float(y), radii[name])
        for name, (x, y, _touches) in display.items()
        if name != exclude
    )


def _separate_network_positions(display: dict[str, tuple[float, float, float]], min_gap=5.3):
    """Apply small collision-only nudges while preserving each player's anchor."""
    names = list(display)
    if len(names) < 2:
        return display
    anchors = np.array([[display[name][0], display[name][1]] for name in names], dtype=float)
    coords = anchors.copy()
    for _ in range(90):
        shift = np.zeros_like(coords)
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                vector = coords[i] - coords[j]
                distance = float(np.hypot(vector[0], vector[1]))
                if distance >= min_gap:
                    continue
                if distance < 1e-6:
                    angle = (i * 37 + j * 19) % 360
                    vector = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                    distance = 1.0
                push = (min_gap - distance) * 0.22 * vector / distance
                shift[i] += push
                shift[j] -= push
        coords += shift
        coords += (anchors - coords) * 0.025
        coords[:, 0] = np.clip(coords[:, 0], -PITCH_WIDTH / 2 + 2.4, PITCH_WIDTH / 2 - 2.4)
        coords[:, 1] = np.clip(coords[:, 1], 2.5, PITCH_LENGTH - 2.5)
    return {
        name: (float(coords[idx, 0]), float(coords[idx, 1]), display[name][2])
        for idx, name in enumerate(names)
    }


def team_event_counts(events, team_id):
    team = events[events["team_id"].eq(team_id)]
    types = team["type"].astype(str)
    opponent_id = AWAY_ID if team_id == HOME_ID else HOME_ID
    return {
        "Tackles": int(types.eq("Tackle").sum()),
        "Interceptions": int(types.eq("Interception").sum()),
        "Recoveries": int(types.eq("BallRecovery").sum()),
        "Clearances": int(types.eq("Clearance").sum()),
        "Blocks": defensive_blocks_count(events, team_id, opponent_id),
        "Fouls": fouls_committed_count(events, team_id),
    }


def xg_row(xg, team_name):
    row = xg[xg["team"].astype(str).str.lower().eq(team_name.lower())]
    return row.iloc[0] if not row.empty else pd.Series(dtype=float)


def shot_timeline(fig, shots, outcome, top, team_id):
    """Every shot at its minute, standing as high as its xG.

    The map says where a chance came from and the cumulative curve says how the
    total built; neither says when each one came or how big it was. Stems share
    the shot map's colours and glyphs, so a goal is the same star here.
    """
    if shots.empty or top < 0.13:
        return None
    ax = fig.add_axes([0.075, 0.075, 0.54, max(top - 0.075, 0.05)])
    ax.set_facecolor(BG)
    minutes = (
        pd.to_numeric(shots["minute"], errors="coerce").fillna(0)
        + pd.to_numeric(shots.get("second", 0), errors="coerce").fillna(0) / 60
    )
    end = max(95.0, float(minutes.max()) + 2.0)
    peak = max(float(shots["xG"].max()), 0.2)
    ax.set_xlim(0, end)
    ax.set_ylim(0, peak * 1.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
    ax.set_yticks([])
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xticklabels([f"{m}′" for m in (0, 15, 30, 45, 60, 75, 90)])
    ax.axvline(45, color=GRID, lw=1.0, ls=(0, (3, 3)), zorder=0)
    kinds = {
        "MissedShots": ("X", SHOT_MISS),
        "BlockedShot": ("s", SHOT_BLOCKED),
        "ShotOnPost": ("D", SHOT_POST),
        "SavedShot": ("o", SHOT_SAVED),
        "Goal": ("*", SHOT_GOAL),
    }
    for index in shots.index:
        marker, colour = kinds.get(str(outcome.at[index]), ("o", MUTED))
        x, height = float(minutes.at[index]), float(shots.at[index, "xG"])
        ax.vlines(x, 0, height, color=colour, lw=1.6, alpha=0.9, zorder=2)
        ax.scatter(
            [x],
            [height],
            s=150 if marker == "*" else 42,
            marker=marker,
            facecolors=colour,
            edgecolors=BG,
            linewidths=0.8,
            zorder=3,
        )
    ax.text(
        0,
        1.12,
        "SHOT TIMELINE · HEIGHT = xG",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )
    return ax


def shot_map(events, xg, team_id, number):
    pso = as_bool(events.get("is_penalty_shootout", pd.Series(False, index=events.index)))
    # An own goal is logged as a shot by the side that put it in. Leaving it
    # here printed "SHOTS 18" beside "xG / SHOT 0.183" in the same panel, and
    # 0.183 is 3.11/17 — the xg frame had already excluded it, so the card
    # disagreed with itself and with the cover.
    own_goal = as_bool(events.get("is_own_goal", pd.Series(False, index=events.index)))
    shots = (
        events[events["team_id"].eq(team_id) & as_bool(events["is_shot"]) & ~pso & ~own_goal]
        .copy()
        .dropna(subset=["x", "y"])
    )
    shots["xG"] = pd.to_numeric(shots["xG"], errors="coerce").fillna(0).clip(lower=0)
    fig, pitch, side = pitch_axes(
        f"Shot Map · {TEAM_NAME[team_id]}",
        "Shot location, outcome and chance quality · marker size = xG",
    )
    draw_long_pitch(pitch)
    # Colour carries the outcome (shared shot palette, one key across the whole
    # Zoom without dropping any recorded shot, including unusual long shots:
    # the crop runs back to the furthest one, in pitch metres.
    reach = float(shots["x"].min()) * PITCH_LENGTH / 100 if not shots.empty else 60.0
    pitch_bottom = crop_to_attack(pitch, side, min(60.0, reach - 6.0), bottom_in=2.5)
    # report); marker shape repeats it so the map still reads in grayscale.
    # Goals are listed last so they draw on top of the other outcomes, and the
    # star glyph is scaled up because it reads much smaller than a disc of the
    # same nominal point area.
    markers = [
        ("MissedShots", "X", SHOT_MISS, "Off target", 1.0),
        ("BlockedShot", "s", SHOT_BLOCKED, "Blocked", 1.0),
        ("ShotOnPost", "D", SHOT_POST, "Woodwork", 1.0),
        ("SavedShot", "o", SHOT_SAVED, "Saved", 1.0),
        ("Goal", "*", SHOT_GOAL, "Goal", 2.4),
    ]
    # The outcome is the classified shot type, not the raw event type: the feed
    # logs a blocked shot as a SavedShot, so reading ``type`` filed nine blocks
    # under "Saved (11)" for a side whose keeper-board count was two, and the
    # Blocked series above never drew a mark.
    outcome = (
        shots["shot_whoscored_type"].astype(str)
        if "shot_whoscored_type" in shots
        else pd.Series("", index=shots.index)
    )
    outcome = outcome.where(~outcome.isin(["", "nan", "None"]), shots["type"].astype(str))
    for event_type, marker, color, label, scale in markers:
        subset = shots[outcome.eq(event_type)]
        if subset.empty:
            continue
        px, py = attack_xy(subset["x"], subset["y"])
        sizes = (90 + subset["xG"].to_numpy() * 900) * scale
        pitch.scatter(
            px,
            py,
            s=sizes,
            marker=marker,
            facecolors=color,
            edgecolors=BG,
            linewidths=1.0,
            alpha=0.95,
            label=f"{label} ({len(subset)})",
            zorder=5 if event_type == "Goal" else 4,
        )
    # The chances worth naming: a figure beside the mark, so the three biggest
    # do not have to be read off a marker's area.
    # Only chances big enough to matter, and never two figures on top of each
    # other: a cluster of 0.13 and 0.14 in the six-yard box is one smudge.
    named = []
    for index, row in shots.nlargest(3, "xG").iterrows():
        if row["xG"] < 0.2 and named:
            continue
        lx_, ly_ = attack_xy([row["x"]], [row["y"]])
        if any(np.hypot(lx_[0] - ox, ly_[0] - oy) < 7 for ox, oy in named):
            continue
        named.append((lx_[0], ly_[0]))
        lx, ly = attack_xy([row["x"]], [row["y"]])
        pitch.annotate(
            f"{row['xG']:.2f}",
            (lx[0], ly[0]),
            xytext=(11, -4),
            textcoords="offset points",
            color=TEXT,
            fontsize=8.5,
            fontweight="bold",
            zorder=8,
            path_effects=[path_effects.withStroke(linewidth=3, foreground=BG)],
        )
    pitch.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=5,
        frameon=False,
        labelcolor=TEXT,
        fontsize=8.5,
        columnspacing=1.4,
        handletextpad=0.4,
    )
    shot_timeline(fig, shots, outcome, pitch_bottom - 0.115, team_id)
    xr = xg_row(xg, TEAM_NAME[team_id])
    side_title(side, "SHOT OUTPUT")
    kpi_bottom = side_kpis(
        side,
        [
            ("Shots", f"{len(shots)}"),
            ("xG", f"{float(xr.get('xG', 0)):.2f}"),
            ("xG / shot", f"{float(xr.get('xG_per_shot', 0)):.3f}"),
            # Counted from the shots drawn on this board, so the card agrees with
            # the legend and with the goal-frame split under it. The provider's
            # own tally, used elsewhere, can file a block as a shot on target.
            ("On target", f"{int(outcome.isin(['Goal', 'SavedShot']).sum())}"),
        ],
    )

    # Shot location says where the chance came from; this says which part of
    # the goal the keeper actually had to cover. Anchored under the KPI block
    # rather than at a fixed 0.30, which the fourth KPI's value ran into.
    zones = shot_placement_zones(events, team_id)
    if sum(zones.values()):
        heading_y = kpi_bottom - 0.03
        side.text(
            0.08, heading_y, "GOAL FRAME TARGETED", color=MUTED, fontsize=7.5, fontweight="bold"
        )
        ranked = [item for item in sorted(zones.items(), key=lambda pair: -pair[1]) if item[1]][:3]
        for idx, (zone, count) in enumerate(ranked):
            y = heading_y - 0.05 - idx * 0.055
            side.text(0.08, y, zone.replace("_", " ").title(), color=TEXT, fontsize=8, va="center")
            side.text(
                0.92,
                y,
                str(count),
                color=TEXT,
                fontsize=8.5,
                fontweight="bold",
                ha="right",
                va="center",
            )
    return save(fig, f"{number:02d}_shot_map_{_team_slug(team_id)}.png")


def goals_breakdown(events):
    pso = as_bool(events.get("is_penalty_shootout", pd.Series(False, index=events.index)))
    goals = events[as_bool(events["is_goal"]) & ~pso].copy()
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.text(0.055, 0.94, "Goal Breakdown", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(
        0.055,
        0.898,
        "Scoring timeline and goal profile · penalty shootout excluded",
        fontsize=11,
        color=MUTED,
    )
    fig.add_artist(
        Line2D([0.055, 0.945], [0.86, 0.86], transform=fig.transFigure, color=GRID, lw=1)
    )
    ax = fig.add_axes([0.08, 0.18, 0.84, 0.56])
    base.clean_ax(ax)
    max_min = max(90, int(pd.to_numeric(events["minute"], errors="coerce").max()) + 2)
    ax.set_xlim(0, max_min)
    ax.set_ylim(-1.4, 1.4)
    ax.axhline(0, color=GRID, lw=1.5)
    ax.set_yticks([0.75, -0.75])
    ax.set_yticklabels([HOME_NAME, AWAY_NAME], fontsize=11)
    ax.set_xlabel("Match minute")
    ordered = events.sort_values(["minute", "second", "event_id"], kind="stable").copy()
    ordered["_clock"] = pd.to_numeric(ordered["minute"], errors="coerce").fillna(
        0
    ) * 60 + pd.to_numeric(ordered["second"], errors="coerce").fillna(0)

    def assist_for(goal):
        explicit = goal.get("assist_player", np.nan)
        if pd.notna(explicit) and str(explicit).strip():
            return str(explicit)
        goal_clock = float(goal["_clock"])
        key_pass = as_bool(ordered.get("is_key_pass", pd.Series(False, index=ordered.index)))
        candidates = ordered[
            ordered["team_id"].eq(goal["team_id"])
            & ordered["type"].astype(str).eq("Pass")
            & ordered["outcome"].astype(str).str.lower().eq("successful")
            & key_pass
            & ordered["_clock"].between(goal_clock - 15, goal_clock, inclusive="left")
            & ordered["player"].notna()
        ]
        return str(candidates.iloc[-1]["player"]) if not candidates.empty else "UNASSISTED"

    goals = ordered.loc[goals.index].sort_values(["minute", "second"], kind="stable").copy()
    # An own goal is logged on the scorer's own team_id; it belongs on the
    # opponent's side of this timeline.
    goals["_credited_team"] = base.credited_team(goals)
    team_goal_count = {HOME_ID: 0, AWAY_ID: 0}
    for _, goal in goals.iterrows():
        tid = int(goal["_credited_team"])
        y = 0.75 if tid == HOME_ID else -0.75
        minute = float(goal["minute"])
        team_goal_count[tid] += 1
        ax.vlines(minute, 0, y, color=_team_mark_color(tid), lw=2)
        ax.scatter(minute, y, s=150, marker="*", color=FOCUS, edgecolor=BG, linewidth=1.2, zorder=4)
        player = _surname(goal.get("player"), "Goal")
        is_own = bool(as_bool(pd.Series([goal.get("is_own_goal", False)])).iloc[0])
        assist = assist_for(goal)
        assist_label = (
            "OWN GOAL"
            if is_own
            else ("UNASSISTED" if assist == "UNASSISTED" else f"ASSIST · {assist.split()[-1]}")
        )
        label = f"{int(minute)}′  {player}\n{assist_label}"
        horizontal_nudge = -10 if team_goal_count[tid] % 2 else 10
        ax.annotate(
            label,
            (minute, y),
            xytext=(horizontal_nudge, 18 if y > 0 else -18),
            textcoords="offset points",
            ha="center",
            va="bottom" if y > 0 else "top",
            color=TEXT,
            fontsize=7.2,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=PANEL, edgecolor=GRID, linewidth=0.65),
        )
    ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.65)
    return save(fig, "04_goals_breakdown.png")


def _half_network_data(events, players, team_id, half):
    period_code = "1h" if half == 1 else "2h"
    frame = events[events["period_code"].astype(str).str.lower().eq(period_code)].copy()
    frame = frame.sort_values(["minute", "second", "event_id"], kind="stable")
    team_frame = frame[frame["team_id"].eq(team_id)].copy()

    work = frame.copy()
    work["clock"] = pd.to_numeric(work["minute"], errors="coerce").fillna(0) * 60 + pd.to_numeric(
        work["second"], errors="coerce"
    ).fillna(0)
    work["next_team"] = work["team_id"].shift(-1)
    work["next_player"] = work["player"].shift(-1)
    work["next_clock"] = work["clock"].shift(-1)
    passes = work[
        work["team_id"].eq(team_id)
        & work["type"].astype(str).eq("Pass")
        & work["outcome"].astype(str).str.lower().eq("successful")
        & work["next_team"].eq(team_id)
        & work["player"].notna()
        & work["next_player"].notna()
        & work["next_clock"].sub(work["clock"]).between(0, 20)
    ].copy()
    passes = passes[passes["player"].astype(str).ne(passes["next_player"].astype(str))]

    touches = team_frame[touch_mask(team_frame)].dropna(subset=["player", "x", "y"]).copy()
    sub_events = team_frame[
        team_frame["type"].astype(str).isin(["SubstitutionOn", "SubstitutionOff"])
    ].copy()
    log_sub_events = sub_events.copy()
    sub_on = set(
        sub_events[sub_events["type"].astype(str).eq("SubstitutionOn")]["player"]
        .dropna()
        .astype(str)
    )
    sub_off = set(
        sub_events[sub_events["type"].astype(str).eq("SubstitutionOff")]["player"]
        .dropna()
        .astype(str)
    )
    if half == 1:
        interval_events = events[
            events["team_id"].eq(team_id)
            & events["period_code"].astype(str).str.lower().eq("2h")
            & events["type"].astype(str).isin(["SubstitutionOn", "SubstitutionOff"])
            & (pd.to_numeric(events["minute"], errors="coerce").fillna(999) <= 45)
        ].copy()
        interval_off = interval_events[interval_events["type"].astype(str).eq("SubstitutionOff")][
            "player"
        ]
        sub_off |= set(interval_off.dropna().astype(str))
        # The player introduced at the interval belongs to the second-half
        # visual only. Keep the outgoing starter marked, but do not name the
        # incoming player anywhere on the first-half card.
        log_sub_events = pd.concat(
            [
                sub_events,
                interval_events[interval_events["type"].astype(str).eq("SubstitutionOff")],
            ],
            ignore_index=False,
        )
    participants = set(touches["player"].dropna().astype(str)) | sub_on
    if half == 1:
        starters = players[
            players["team_id"].eq(team_id)
            & players["is_first_xi"].astype(str).str.lower().isin(["true", "1", "yes"])
        ]["name"]
        participants |= set(starters.dropna().astype(str))

    player_info = players[players["team_id"].eq(team_id)].set_index("name")
    all_touches = events[touch_mask(events) & events["team_id"].eq(team_id)].dropna(
        subset=["player", "x", "y"]
    )
    position_rows = []
    for name in sorted(participants):
        player_touches = touches[touches["player"].astype(str).eq(name)]
        coords = player_touches[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna()
        if coords.empty:
            event_coords = (
                team_frame[team_frame["player"].astype(str).eq(name)][["x", "y"]]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
            )
            coords = event_coords
        if coords.empty:
            whole_coords = (
                all_touches[all_touches["player"].astype(str).eq(name)][["x", "y"]]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
            )
            coords = whole_coords
        if coords.empty:
            role = player_info.at[name, "position"] if name in player_info.index else ""
            x, y = _role_fallback_position(role)
        else:
            x, y = float(coords["x"].mean()), float(coords["y"].mean())
        position_rows.append({"player": name, "x": x, "y": y, "touches": int(len(player_touches))})
    positions = (
        pd.DataFrame(position_rows).set_index("player")
        if position_rows
        else pd.DataFrame(columns=["x", "y", "touches"])
    )
    names = set(positions.index.astype(str))
    passes = passes[
        passes["player"].astype(str).isin(names) & passes["next_player"].astype(str).isin(names)
    ]
    edges = (
        passes.groupby(["player", "next_player"])
        .size()
        .reset_index(name="passes")
        .sort_values("passes", ascending=False)
        .head(22)
    )

    substitutions = []
    events_sorted = log_sub_events.sort_values(["minute", "second", "event_id"], kind="stable")
    pending_off = []
    for _, row in events_sorted.iterrows():
        minute = int(float(row.get("minute", 0) or 0))
        # These names are read in a side-panel row that spans most of the
        # column, not next to a pitch marker, so the pitch default clipped
        # "Marmoush" and "Aït-Nouri" for space that was never contested.
        name = compact_player_label(str(row.get("player", "")), 12)
        if str(row.get("type")) == "SubstitutionOff":
            pending_off.append((minute, name))
        else:
            match_idx = next(
                (idx for idx, item in enumerate(pending_off) if item[0] == minute), None
            )
            off_name = pending_off.pop(match_idx)[1] if match_idx is not None else "—"
            substitutions.append((minute, name, off_name))
    for minute, off_name in pending_off:
        substitutions.append((minute, "—", off_name))
    return positions, edges, sub_on, sub_off, substitutions, int(len(passes))


def _network_resolve(label, names):
    """Map a compact substitution label back to a full player name."""
    stem = str(label).rstrip("…").strip()
    if not stem or stem == "—":
        return None
    parts = {name: (name.split() or [name])[-1] for name in names}
    for test in (
        lambda n, last: last == stem,
        lambda n, last: last.startswith(stem),
        lambda n, last: stem in n,
    ):
        for name, last in parts.items():
            if test(name, last):
                return name
    return None


def _network_repel(anchor, radii, half_w, length, gap=1.6, steps=600):
    """Push overlapping nodes apart while keeping each near its true position."""
    names = list(anchor)
    pos = {n: np.array(anchor[n], dtype=float) for n in names}
    for _ in range(steps):
        moved = False
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                d = pos[a] - pos[b]
                dist = float(np.hypot(*d))
                if dist < 1e-6:
                    d, dist = np.array([1.0, 0.0]), 1.0
                need = radii[a] + radii[b] + gap
                if dist < need:
                    push = (need - dist) / 2 * (d / dist)
                    pos[a] += push
                    pos[b] -= push
                    moved = True
        for n in names:
            pos[n] += (np.array(anchor[n]) - pos[n]) * 0.006
            pos[n][0] = float(np.clip(pos[n][0], -half_w + radii[n], half_w - radii[n]))
            pos[n][1] = float(np.clip(pos[n][1], radii[n], length - radii[n]))
        if not moved:
            break
    return pos


def _network_place_labels(pos, radii, names, label_of, fontpt, unit_pt, half_w, length):
    """Name a node only where the name sits clearly beside it.

    Tried below, above, beside and diagonally, close in. A spot is refused when
    it covers another node, another name, or sits nearer somebody else's circle
    than its own -- a name that reads as his neighbour's is worse than none, and
    the line-up card names every number anyway.
    """
    placed, boxes = {}, []
    for n in sorted(names, key=lambda m: -radii[m]):
        text_ = label_of(n)
        w = (len(text_) * fontpt * 0.62 + 4) / unit_pt
        h = fontpt * 1.3 / unit_pt
        x, y = pos[n]
        r = radii[n]
        cands = []
        for extra in (0.0, 1.6, 3.2):
            cands += [
                (x, y - r - 0.9 - extra - h / 2),
                (x, y + r + 0.9 + extra + h / 2),
                (x + r + 0.8 + extra + w / 2, y),
                (x - r - 0.8 - extra - w / 2, y),
                (x + (r + 0.6 + extra) * 0.72 + w / 2, y - (r + 0.6 + extra) * 0.72 - h / 2),
                (x - (r + 0.6 + extra) * 0.72 - w / 2, y - (r + 0.6 + extra) * 0.72 - h / 2),
            ]
        best, best_pen = None, 1e9
        for cx, cy in cands:
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            pen = 0.0
            for m in names:
                mx, my = pos[m]
                nx, ny = min(max(mx, box[0]), box[2]), min(max(my, box[1]), box[3])
                if np.hypot(nx - mx, ny - my) < radii[m] + 0.9:
                    pen += 90
            for b in boxes:
                if box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1]:
                    pen += 60
            if box[0] < -half_w - 3 or box[2] > half_w + 3 or box[1] < -2 or box[3] > length + 2:
                pen += 60
            own = float(np.hypot(cx - x, cy - y))
            if any(np.hypot(cx - pos[m][0], cy - pos[m][1]) < own for m in names if m != n):
                pen += 70
            pen += 0.35 * own
            if pen < best_pen:
                best, best_pen = (cx, cy, box), pen
        if best is not None and best_pen < 60:
            placed[n] = best[:2]
            boxes.append(best[2])
    return placed


def pass_network(events, players, team_id, number, half):
    """Who linked with whom in one half, with the changes marked where they happened.

    Numbers sit in the circles and the names live in the line-up card beside the
    pitch: sixteen names, sixteen badges and twenty arrows on one midfield was
    unreadable. A circle is a player at his average touch position; a ring is a
    man who went off, a square a man who came on, and a short dotted line joins
    each to the player he replaced. Links carry no arrowheads -- the picture is
    the shape of the team's connections, and each pair is the sum of both ways.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    positions, edges, sub_on, sub_off, substitutions, completed_links = _half_network_data(
        events, players, team_id, half
    )
    half_label = "First Half" if half == 1 else "Second Half"
    team_name = TEAM_NAME[team_id]
    team_color = _team_mark_color(team_id)
    names = [str(n) for n in positions.index]
    fig_w, fig_h = 12.0, 10.0

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
    base.amoled_header(
        fig,
        f"Pass Network · {team_name} · {half_label}",
        f"{len(names)} participants · each circle sits at the player's average touch position",
        active_team=team_name,
    )
    pitch = fig.add_axes([0.04, 0.085, 0.46, 0.745])
    # Softer lines than the other boards: here the data is the lines and circles.
    soft_line = mcolors.to_hex(
        np.asarray(mcolors.to_rgb(PITCH_LINE)) * 0.45 + np.asarray(mcolors.to_rgb(BG)) * 0.55
    )
    draw_long_pitch(pitch, line_color=soft_line)
    x0, x1 = pitch.get_xlim()
    y0, y1 = pitch.get_ylim()
    box = pitch.get_position()
    unit_pt = min(box.width * fig_w * 72 / (x1 - x0), box.height * fig_h * 72 / (y1 - y0))

    # ── who came on and off, and for whom ───────────────────────────────────
    off_min, on_min, replaced = {}, {}, {}
    for minute, on_label, off_label in substitutions:
        off_name = _network_resolve(off_label, names)
        on_name = _network_resolve(on_label, names)
        if off_name:
            off_min[off_name] = minute
        if on_name:
            on_min[on_name] = minute
        if off_name and on_name:
            replaced[on_name] = off_name
    went_off = {n for n in names if n in sub_off}
    came_on = {n for n in names if n in sub_on}

    # ── geometry ────────────────────────────────────────────────────────────
    anchor, touches = {}, {}
    for name in names:
        px, py = player_position_xy([positions.at[name, "x"]], [positions.at[name, "y"]])
        anchor[name] = (float(px[0]), float(py[0]))
        touches[name] = float(positions.at[name, "touches"])
    max_touch = max(touches.values() or [1.0]) or 1.0
    radii = {n: 1.55 + 1.45 * np.sqrt(touches[n] / max_touch) for n in names}
    pos = _network_repel(anchor, radii, PITCH_WIDTH / 2, PITCH_LENGTH)

    # ── links: one line per pair, both ways summed, the strongest nine ──────
    _link_low, link_color, _link_strong = network_link_palette(TEAM_COLOR[team_id])
    pair_total = {}
    for _, edge in edges.iterrows():
        key = tuple(sorted((str(edge["player"]), str(edge["next_player"]))))
        pair_total[key] = pair_total.get(key, 0) + int(edge["passes"])
    ranked = sorted(pair_total.items(), key=lambda item: -item[1])
    top_count = float(ranked[0][1]) if ranked else 1.0
    for (a, b), count in ranked[:9]:
        if a in pos and b in pos:
            share = count / top_count
            pitch.plot(
                [pos[a][0], pos[b][0]],
                [pos[a][1], pos[b][1]],
                color=link_color,
                lw=1.0 + 4.2 * share**0.9,
                alpha=0.20 + 0.55 * share**0.8,
                solid_capstyle="round",
                zorder=2,
            )

    # ── replacement: a short dotted line, the shapes already give the way ───
    for on_name, off_name in replaced.items():
        gap = np.hypot(*(pos[off_name] - pos[on_name])) - radii[off_name] - radii[on_name]
        if gap < 0.6:
            continue
        pitch.add_patch(
            FancyArrowPatch(
                tuple(pos[off_name]),
                tuple(pos[on_name]),
                connectionstyle="arc3,rad=-0.25",
                arrowstyle="-",
                lw=2.0,
                linestyle=(0, (1.0, 2.0)),
                color=FOCUS,
                shrinkA=radii[off_name] * unit_pt + 2,
                shrinkB=radii[on_name] * unit_pt + 2,
                zorder=3,
            )
        )

    # ── circles ─────────────────────────────────────────────────────────────
    shirts = shirt_number_map(players)
    bg_rgb, team_rgb = np.asarray(mcolors.to_rgb(BG)), np.asarray(mcolors.to_rgb(team_color))
    faded = mcolors.to_hex(team_rgb * 0.42 + bg_rgb * 0.58)
    for name in names:
        x, y = pos[name]
        r = radii[name]
        if name in came_on:
            fill = team_color
            pitch.add_patch(
                FancyBboxPatch(
                    (x - r * 0.93, y - r * 0.93),
                    2 * r * 0.93,
                    2 * r * 0.93,
                    boxstyle=f"round,pad=0,rounding_size={r * 0.38}",
                    facecolor=fill,
                    edgecolor=FOCUS,
                    lw=2.0,
                    zorder=5,
                )
            )
        elif name in went_off:
            fill = faded
            pitch.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=FOCUS, lw=2.4, zorder=5))
        else:
            fill = team_color
            pitch.add_patch(
                Circle((x, y), r, facecolor=fill, edgecolor=link_color, lw=1.4, zorder=5)
            )
        shirt = shirts.get(name)
        if shirt:
            pitch.text(
                x,
                y,
                str(shirt),
                color=text_on_fill(fill),
                fontsize=6.4 + 2.6 * (r - 1.55) / 1.45,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=6,
            )

    def label_of(name):
        surname = (name.split() or [name])[-1]
        return f"{surname} {on_min[name]}′" if name in on_min else surname

    label_size = 8.0
    for name, (cx, cy) in _network_place_labels(
        pos, radii, names, label_of, label_size, unit_pt, PITCH_WIDTH / 2, PITCH_LENGTH
    ).items():
        pitch.text(
            cx,
            cy,
            label_of(name),
            color=TEXT,
            fontsize=label_size,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10,
            path_effects=[path_effects.withStroke(linewidth=3.0, foreground=BG)],
        )

    # ── panel: top pairs, line-up, three numbers ────────────────────────────
    panel_w, panel_h = 0.43, 0.745
    panel = fig.add_axes([0.535, 0.085, panel_w, panel_h])
    panel.axis("off")
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    size = (fig_w * panel_w, fig_h * panel_h)

    def card(x, y, w, h):
        panel.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={0.10 / size[0]}",
                mutation_aspect=size[0] / size[1],
                facecolor=PANEL,
                edgecolor=GRID,
                linewidth=1.0,
                zorder=1,
            )
        )

    def head(y, left, right=""):
        panel.text(
            0.045, y, left.upper(), color=MUTED, fontsize=8.6, fontweight="bold", va="center"
        )
        if right:
            panel.text(
                0.955,
                y,
                right.upper(),
                color=NEUTRAL,
                fontsize=7.4,
                fontweight="bold",
                ha="right",
                va="center",
            )

    card(0, 0.635, 1, 0.365)
    head(0.955, "Top pairs", "passes both ways")
    for i, ((a, b), count) in enumerate(ranked[:4]):
        y = 0.885 - i * 0.064
        pair = f"{(a.split() or [a])[-1]}  ↔  {(b.split() or [b])[-1]}"
        panel.text(0.045, y, pair, color=TEXT, fontsize=10.5, fontweight="bold", va="center")
        panel.text(
            0.955,
            y - 0.003,
            str(count),
            color=TEXT,
            fontsize=17,
            fontweight="bold",
            ha="right",
            va="center",
        )
        panel.add_patch(
            Rectangle((0.045, y - 0.036), 0.91, 0.006, facecolor=GRID, edgecolor="none", zorder=2)
        )
        panel.add_patch(
            Rectangle(
                (0.045, y - 0.036),
                0.91 * count / top_count,
                0.006,
                facecolor=team_color,
                edgecolor="none",
                zorder=3,
            )
        )

    card(0, 0.125, 1, 0.48)
    changes = len(substitutions)
    head(0.572, "Line-up", f"{changes} changes" if changes else "no changes")
    panel.text(
        0.045,
        0.527,
        "STARTED THE HALF",
        color=NEUTRAL,
        fontsize=6.8,
        fontweight="bold",
        va="center",
    )
    panel.text(0.545, 0.527, "CAME ON", color=NEUTRAL, fontsize=6.8, fontweight="bold", va="center")

    def shirt_of(name):
        value = str(shirts.get(name) or "")
        return (0, int(value)) if value.isdigit() else (1, 0)

    def triangle(cx, cy, up, color, size_=0.009):
        pts = (
            [(cx - size_, cy - size_ * 0.8), (cx + size_, cy - size_ * 0.8), (cx, cy + size_)]
            if up
            else [(cx - size_, cy + size_ * 0.8), (cx + size_, cy + size_ * 0.8), (cx, cy - size_)]
        )
        panel.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none", zorder=3))

    started = sorted((n for n in names if n not in came_on), key=shirt_of)
    for i, name in enumerate(started):
        y = 0.485 - i * 0.0335
        gone = name in went_off
        color = MUTED if gone else TEXT
        panel.text(
            0.045,
            y,
            str(shirts.get(name) or ""),
            color=color,
            fontsize=10.5,
            fontweight="bold",
            va="center",
        )
        panel.text(
            0.115,
            y,
            (name.split() or [name])[-1],
            color=color,
            fontsize=9.6,
            fontweight="bold",
            va="center",
        )
        if gone:
            triangle(0.395, y, False, MUTED)
            if name in off_min:
                panel.text(
                    0.42,
                    y,
                    f"{off_min[name]}′",
                    color=MUTED,
                    fontsize=9.6,
                    fontweight="bold",
                    va="center",
                )
    entered = sorted(
        (n for n in names if n in came_on), key=lambda n: (on_min.get(n, 999), shirt_of(n))
    )
    for i, name in enumerate(entered):
        y = 0.485 - i * 0.075
        triangle(0.555, y, True, TEXT, 0.010)
        panel.text(
            0.60,
            y,
            str(shirts.get(name) or ""),
            color=TEXT,
            fontsize=10.5,
            fontweight="bold",
            va="center",
        )
        panel.text(
            0.665,
            y,
            (name.split() or [name])[-1],
            color=TEXT,
            fontsize=9.6,
            fontweight="bold",
            va="center",
        )
        if name in on_min:
            panel.text(
                0.955,
                y,
                f"{on_min[name]}′",
                color=TEXT,
                fontsize=10.5,
                fontweight="bold",
                ha="right",
                va="center",
            )
        if name in replaced:
            panel.text(
                0.665,
                y - 0.027,
                f"for {(replaced[name].split() or [replaced[name]])[-1]}",
                color=MUTED,
                fontsize=8,
                va="center",
            )
    if not entered:
        panel.text(0.545, 0.485, "No changes in this half", color=NEUTRAL, fontsize=9, va="center")

    # Link volume names the busiest pair. Betweenness names the player the
    # network routes through -- take them out and it splits in two. Scoped to
    # this half: run over the whole match it lists players who were not on the
    # pitch for the half being drawn.
    half_events = events[
        events["period_code"].astype(str).str.lower().eq("1h" if half == 1 else "2h")
    ]
    centrality = network_centrality(half_events, team_id)
    card(0, 0, 1, 0.095)
    tiles = [(str(completed_links), "completed links"), (str(len(names)), "players used")]
    if not centrality.empty:
        top_row = centrality.iloc[0]
        tiles.append(
            (
                (str(top_row["player"]).split() or [""])[-1].upper(),
                f"top connector · {float(top_row['betweenness']):.3f}",
            )
        )
    for i, (value, label) in enumerate(tiles):
        tx = [0.045, 0.265, 0.445][i]
        panel.text(
            tx,
            0.056,
            value,
            color=TEXT,
            fontsize=14 if i < 2 else 12.5,
            fontweight="bold",
            va="center",
        )
        panel.text(
            tx, 0.020, label.upper(), color=MUTED, fontsize=6.2, fontweight="bold", va="center"
        )
        if i:
            panel.plot([tx - 0.02, tx - 0.02], [0.012, 0.083], color=GRID, lw=1.0)

    # ── key: only the shapes this half actually uses ────────────────────────
    key = fig.add_axes([0.04, 0.035, 0.46, 0.03])
    key.axis("off")
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    items = [(0.015, "o", team_color, link_color, "Started, stayed")]
    if went_off:
        items.append((0.30, "o", faded, FOCUS, "Went off"))
    if came_on:
        items.append((0.50, "s", team_color, FOCUS, "Came on"))
    for kx, marker, fill, edge, label in items:
        key.scatter([kx], [0.5], s=70, marker=marker, facecolor=fill, edgecolor=edge, linewidth=1.5)
        key.text(
            kx + 0.03, 0.5, label.upper(), color=MUTED, fontsize=7.2, fontweight="bold", va="center"
        )
    if replaced:
        key.plot([0.70, 0.78], [0.5, 0.5], color=FOCUS, lw=2.0, linestyle=(0, (1.0, 2.0)))
        key.text(
            0.80, 0.5, "REPLACEMENT", color=MUTED, fontsize=7.2, fontweight="bold", va="center"
        )
    fig.text(
        0.945,
        0.035,
        "FULL VISUAL REDESIGN · REAL MATCH DATA",
        ha="right",
        fontsize=8,
        color=NEUTRAL,
    )

    suffix = "1h" if half == 1 else "2h"
    return save(
        fig,
        f"{number:02d}{'a' if half == 1 else 'b'}_pass_network_{_team_slug(team_id)}_{suffix}.png",
    )


def xt_map(events, team_id, number):
    eligible = events[
        events["type"].isin(["Pass", "Carry"])
        & events["outcome"].astype(str).str.lower().eq("successful")
    ].dropna(subset=["x", "y", "end_x", "end_y"])
    team = eligible[eligible["team_id"].eq(team_id)].copy()
    team["xT"] = pd.to_numeric(team["xT"], errors="coerce").fillna(0).clip(lower=0)
    heat, _, _ = np.histogram2d(
        team["y"], team["x"], bins=[7, 12], range=[[0, 100], [0, 100]], weights=team["xT"]
    )
    fig, pitch, side = pitch_axes(
        f"xT Heatmap · {TEAM_NAME[team_id]}",
        "7 × 12 grid · positive successful-movement xT at origins · shared scale for both teams",
    )
    team_mark = _team_mark_color(team_id)
    team_rgb = np.asarray(mcolors.to_rgb(team_mark), dtype=float)
    team_dark = mcolors.to_hex(team_rgb * 0.42)
    cmap = LinearSegmentedColormap.from_list(
        f"xt_full_grid_{team_id}", [BG, PANEL_2, team_dark, team_mark]
    )
    grids = [
        np.histogram2d(
            g["y"],
            g["x"],
            bins=[7, 12],
            range=[[0, 100], [0, 100]],
            weights=pd.to_numeric(g["xT"], errors="coerce").fillna(0).clip(lower=0),
        )[0]
        for _, g in eligible.groupby("team_id")
    ]
    # The scale tops out at the 95th percentile of the lit cells, not at the
    # hottest one: a single cell worth 3.79 beside a median of 0.14 took the
    # whole ramp and left every other cell the colour of the page.
    lit = np.concatenate([grid[grid > 0] for grid in grids] + [np.array([0.001])])
    vmax = max(float(np.percentile(lit, 95)), 0.001)
    x_grid = np.linspace(-PITCH_WIDTH / 2, PITCH_WIDTH / 2, 8)
    y_grid = np.linspace(0, PITCH_LENGTH, 13)
    image = pitch.pcolormesh(
        x_grid,
        y_grid,
        heat.T,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        shading="flat",
        edgecolors=GRID,
        linewidth=0.58,
        alpha=0.98,
        zorder=1,
    )
    draw_long_pitch(pitch)
    # Figures only on the eight hottest cells: forty-odd numbers at five points
    # were unreadable and hid the cells they described.
    # Exactly eight, ties broken by position, so the count never creeps past it.
    labelled = set(np.argsort(heat.ravel(), kind="stable")[::-1][:8].tolist())
    for ix in range(7):
        for iy in range(12):
            value = float(heat[ix, iy])
            if value <= 0 or (ix * 12 + iy) not in labelled:
                continue
            intensity = min(value / vmax, 1.0)
            cell_fill = mcolors.to_hex(cmap(intensity))
            number_color = text_on_fill(cell_fill)
            pitch.text(
                (x_grid[ix] + x_grid[ix + 1]) / 2,
                (y_grid[iy] + y_grid[iy + 1]) / 2,
                f"{value:.2f}",
                color=number_color,
                fontsize=8.0,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=3,
            )
    top = team.nlargest(10, "xT")
    # Only the top three are drawn. Ranks 4-10 added seven dashed arrows over
    # a heatmap that already carries the where; they stay in the list beside.
    dash_color = _on_team_heatmap_accent(team_mark)
    for rank, (_, row) in enumerate(top.head(3).iterrows(), start=1):
        sx, sy = attack_xy([row["x"]], [row["y"]])
        ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        top_three = rank <= 3
        arrow_color = EVENT_HIGHLIGHT if top_three else dash_color
        arrow = pitch.annotate(
            "",
            xy=(ex[0], ey[0]),
            xytext=(sx[0], sy[0]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=arrow_color,
                lw=1.75 if top_three else 1.25,
                alpha=0.94 if top_three else 0.92,
                linestyle="-" if top_three else QUIET_DASH,
                mutation_scale=11 if top_three else 9,
            ),
        )
        if arrow.arrow_patch is not None:
            arrow.arrow_patch.set_path_effects(
                [
                    path_effects.Stroke(linewidth=3.0 if top_three else 2.6, foreground=BG),
                    path_effects.Normal(),
                ]
            )
    cbar = fig.colorbar(image, ax=pitch, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.outline.set_edgecolor(GRID)
    cbar.set_label(
        "xT added per grid square · scale tops out at the 95th percentile", color=MUTED, fontsize=8
    )
    side_title(side, "TOP 10 xT PASSES")
    side_rows(
        side,
        [
            (
                f"{rank}. {_surname(row['player'])} {int(row['minute']):02d}:{int(row.get('second', 0)):02d}",
                f"{float(row['xT']):.3f}",
            )
            for rank, (_, row) in enumerate(top.iterrows(), start=1)
        ],
        start=0.835,
        gap=0.063,
        value_color=TEXT,
        label_color=TEXT,
        label_weight="bold",
    )
    side.plot([0.08, 0.16], [0.088, 0.088], color=EVENT_HIGHLIGHT, lw=1.75)
    side.text(0.19, 0.088, "Top 3 xT passes drawn", color=MUTED, fontsize=7.2, va="center")
    return save(fig, f"{number:02d}_xt_map_{_team_slug(team_id)}.png")


def draw_zone_flow(pitch, frame, done, colour, columns=5, rows_n=7, min_count=4, labels=True):
    """One arrow per zone: the way the passes from it went, on average.

    Hundreds of origin-destination lines are a texture, not a map. A zone of a
    ``columns`` x ``rows_n`` grid is drawn once instead: an arrow in the
    direction its passes went on average, as thick as it is busy and as solid as
    its completion rate, so a side that circulates and a side that launches
    read differently at a glance. ``done`` flags each row of ``frame`` as
    completed. Arrows are kept on the pitch; zones with fewer than ``min_count``
    passes are left out.
    """
    if frame is None or len(frame) == 0:
        return 0
    start_x, start_y = attack_xy(frame["x"].to_numpy(), frame["y"].to_numpy())
    end_x, end_y = attack_xy(frame["end_x"].to_numpy(), frame["end_y"].to_numpy())
    work = pd.DataFrame(
        {
            "sx": np.asarray(start_x, dtype=float),
            "sy": np.asarray(start_y, dtype=float),
            "ex": np.asarray(end_x, dtype=float),
            "ey": np.asarray(end_y, dtype=float),
            "done": np.asarray(done, dtype=bool),
        }
    )
    work["col"] = np.clip(
        ((work["sx"] + PITCH_WIDTH / 2) / PITCH_WIDTH * columns).astype(int), 0, columns - 1
    )
    work["row"] = np.clip((work["sy"] / PITCH_LENGTH * rows_n).astype(int), 0, rows_n - 1)
    zones = work.groupby(["col", "row"])
    busiest = max(int(zones.size().max()), 1)
    cell_w, cell_h = PITCH_WIDTH / columns, PITCH_LENGTH / rows_n
    drawn = 0
    for (col, row_), group in zones:
        if len(group) < min_count:
            continue
        cx = -PITCH_WIDTH / 2 + (col + 0.5) * cell_w
        cy = (row_ + 0.5) * cell_h
        dx = float(group["ex"].mean() - group["sx"].mean())
        dy = float(group["ey"].mean() - group["sy"].mean())
        length = float(np.hypot(dx, dy))
        if length < 1.0:
            continue
        reach = min(length, cell_h * 1.55)
        dx, dy = dx / length * reach, dy / length * reach
        # Keep the whole arrow on the pitch: a goalkeeper's zone pointing
        # upfield would otherwise start behind his own goal line.
        cx = float(
            np.clip(cx, -PITCH_WIDTH / 2 + abs(dx) / 2 + 1, PITCH_WIDTH / 2 - abs(dx) / 2 - 1)
        )
        cy = float(np.clip(cy, abs(dy) / 2 + 1.5, PITCH_LENGTH - abs(dy) / 2 - 1.5))
        rate = float(group["done"].mean())
        share = len(group) / busiest
        arrow = pitch.annotate(
            "",
            xy=(cx + dx / 2, cy + dy / 2),
            xytext=(cx - dx / 2, cy - dy / 2),
            arrowprops=dict(
                arrowstyle="-|>",
                color=colour,
                alpha=0.30 + 0.62 * rate,
                lw=1.6 + 5.0 * share,
                mutation_scale=10 + 14 * share,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=3,
        )
        if arrow.arrow_patch is not None:
            arrow.arrow_patch.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2 + 5.0 * share + 1.6, foreground=BG),
                    path_effects.Normal(),
                ]
            )
        if labels:
            pitch.text(
                cx - dx / 2,
                cy - dy / 2 - 2.2,
                str(len(group)),
                color=TEXT,
                fontsize=7.5,
                ha="center",
                va="top",
                zorder=6,
                path_effects=[path_effects.withStroke(linewidth=2.4, foreground=BG)],
            )
        drawn += 1
    return drawn


def pass_map(events, team_id, number):
    frame = events[events["team_id"].eq(team_id) & events["type"].astype(str).eq("Pass")].copy()
    frame = frame.dropna(subset=["x", "y", "end_x", "end_y"])
    completed = frame["outcome"].astype(str).str.lower().eq("successful")
    key_pass = as_bool(frame.get("is_key_pass", pd.Series(False, index=frame.index)))
    fig, pitch, side = pitch_axes(
        f"Pass Map · {TEAM_NAME[team_id]}",
        "Average pass from each zone · width = passes · opacity = completion · stars = key passes",
    )
    draw_long_pitch(pitch)
    team_mark = _team_mark_color(team_id)
    # Six hundred passes drawn as hairlines are a texture, not a map: one arrow
    # per zone instead, see draw_zone_flow.
    draw_zone_flow(pitch, frame, completed, team_mark)
    for idx in frame.index[key_pass.reindex(frame.index, fill_value=False)]:
        row = frame.loc[idx]
        sx, sy = attack_xy([row["x"]], [row["y"]])
        ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        pitch.plot(
            [sx[0], ex[0]], [sy[0], ey[0]], color=EVENT_HIGHLIGHT, alpha=0.9, lw=1.6, zorder=5
        )
        pitch.scatter(
            ex[0],
            ey[0],
            s=70,
            marker="*",
            color=EVENT_HIGHLIGHT,
            edgecolor=BG,
            linewidth=0.7,
            zorder=7,
        )
    attempts = len(frame)
    complete_count = int(completed.sum())
    forward = int(
        (
            pd.to_numeric(frame["end_x"], errors="coerce")
            > pd.to_numeric(frame["x"], errors="coerce")
        ).sum()
    )
    profile = pass_length_profile(events, team_id)
    side_title(side, "PASSING OUTPUT")
    side_kpis(
        side,
        [
            ("Attempts", attempts),
            ("Completed", complete_count),
            ("Completion", f"{100 * complete_count / max(attempts, 1):.1f}%"),
            ("Forward passes", forward),
        ],
        start=0.845,
        gap=0.105,
    )
    # The same completion rate means different things at 13 m and at 25 m, so
    # length and long-ball survival sit next to the raw totals.
    side.text(0.08, 0.385, "LENGTH & DIRECTION", color=MUTED, fontsize=7.5, fontweight="bold")
    stat_rows = [
        ("Average length", f"{profile['avg_length_m']:.1f} m"),
        ("Long balls", f"{profile['long_ball_share']:.0f}%"),
        ("Long-ball completion", f"{profile['long_ball_completion']:.0f}%"),
    ]
    for idx, (label, value) in enumerate(stat_rows):
        y = 0.34 - idx * 0.05
        side.text(0.08, y, label, color=TEXT, fontsize=8, va="center")
        side.text(
            0.92, y, value, color=TEXT, fontsize=8.5, fontweight="bold", ha="right", va="center"
        )
    # The key sat at a fixed 0.235/0.165/0.095 while the rows above it ran to
    # 0.19, so the "Completed pass" swatch was drawn through the "Long balls"
    # figure. Anchored under the last row instead.
    legend_top = 0.34 - (len(stat_rows) - 1) * 0.05 - 0.06
    legend_y = [legend_top - index * 0.058 for index in range(3)]
    legend_items = [
        ("Average pass from a zone", team_mark, "-", ">"),
        ("Opacity = completion rate", team_mark, "-", "o"),
        (f"Key pass ({int(key_pass.sum())})", EVENT_HIGHLIGHT, "-", "*"),
    ]
    for y, (label, color, style, marker) in zip(legend_y, legend_items):
        side.plot([0.09, 0.25], [y, y], color=color, lw=2.0, ls=style)
        side.scatter(
            [0.25],
            [y],
            s=38 if marker == "*" else 20,
            marker=marker,
            color=color,
            edgecolor=TEXT,
            linewidth=0.45,
            zorder=4,
        )
        side.text(0.31, y, label, color=TEXT, fontsize=8, va="center")
    return save(fig, f"{number:02d}_pass_map_{_team_slug(team_id)}.png")


# ── Goalkeeper goal-frame plot ──────────────────────────────────────────
# Opta reports the crossing point in the same 0-100 scale as pitch width:
# the posts sit at 45.2 and 54.8, and the crossbar at a height of 38.
_OPTA_POST_LEFT = 45.2
_OPTA_POST_RIGHT = 54.8
_OPTA_CROSSBAR = 38.0

# Placement qualifiers, used when the provider gives the zone but not the exact
# crossing point. Values are fractions of the goal: x in -1..1 across the width,
# y in 0..1 up the height. Off-target zones deliberately sit outside that range.
_PLACEMENT_ZONES = {
    "lowleft": (-0.62, 0.18),
    "lowcentre": (0.00, 0.16),
    "lowright": (0.62, 0.18),
    "highleft": (-0.62, 0.76),
    "highcentre": (0.00, 0.80),
    "highright": (0.62, 0.76),
    "missleft": (-1.45, 0.42),
    "missright": (1.45, 0.42),
    "misshigh": (0.00, 1.20),
    "missleftandhigh": (-1.30, 1.14),
    "missrightandhigh": (1.30, 1.14),
    "missleftandlow": (-1.45, 0.14),
    "missrightandlow": (1.45, 0.14),
}
_BODY_PART_MARKERS = {
    "rightfoot": ("o", "Right foot"),
    "leftfoot": ("s", "Left foot"),
    "head": ("^", "Header"),
}


def _placement_xy(row) -> tuple[float, float] | None:
    """Return a shot's crossing point as (x in -1..1, y in 0..1) of the goal.

    Prefers the provider's exact GoalMouthY/GoalMouthZ. Falls back to the
    placement qualifier when only the zone was recorded, spreading shots inside
    the zone with a deterministic offset so repeat placements stay separable
    without moving between runs.
    """
    gy = pd.to_numeric(pd.Series([row.get("goal_mouth_y")]), errors="coerce").iloc[0]
    gz = pd.to_numeric(pd.Series([row.get("goal_mouth_z")]), errors="coerce").iloc[0]
    if pd.notna(gy) and pd.notna(gz):
        span = (_OPTA_POST_RIGHT - _OPTA_POST_LEFT) / 2.0
        x = (float(gy) - (_OPTA_POST_LEFT + span)) / span
        y = float(gz) / _OPTA_CROSSBAR
        return x, y

    tokens = {
        token.strip().strip("'\"").lower()
        for token in re.split(r"[,\[\]]", str(row.get("qualifier_names") or ""))
    }
    for token in tokens:
        if token in _PLACEMENT_ZONES:
            base_x, base_y = _PLACEMENT_ZONES[token]
            # Deterministic jitter keyed on the event so the same shot always
            # lands in the same spot.
            seed = int(hashlib.md5(str(row.get("event_id")).encode()).hexdigest(), 16)
            return base_x + ((seed % 100) / 100 - 0.5) * 0.30, base_y + (
                ((seed // 100) % 100) / 100 - 0.5
            ) * 0.22
    return None


def _draw_goal_frame(ax, accent: str) -> None:
    """Goal posts, crossbar, net grid and the surrounding off-target margin.

    Limits are kept just wide enough for the off-target zones; any more
    headroom renders as dead space above the crossbar.
    """
    ax.set_xlim(-1.78, 1.78)
    ax.set_ylim(-0.20, 1.40)
    ax.set_aspect("equal")
    ax.axis("off")

    # Net: a light grid inside the frame only.
    for gx in np.linspace(-1, 1, 13):
        ax.plot([gx, gx], [0, 1], color=PITCH_LINE, lw=0.35, alpha=0.13, zorder=1)
    for gy in np.linspace(0, 1, 7):
        ax.plot([-1, 1], [gy, gy], color=PITCH_LINE, lw=0.35, alpha=0.13, zorder=1)

    # Posts + crossbar, drawn thick the way a real goal frame reads.
    ax.plot([-1, -1], [0, 1], color=PITCH_LINE, lw=3.4, solid_capstyle="round", zorder=3)
    ax.plot([1, 1], [0, 1], color=PITCH_LINE, lw=3.4, solid_capstyle="round", zorder=3)
    ax.plot([-1, 1], [1, 1], color=PITCH_LINE, lw=3.4, solid_capstyle="round", zorder=3)
    # Ground line runs past the posts so off-target shots have context.
    ax.plot([-1.74, 1.74], [0, 0], color=PITCH_LINE, lw=1.0, alpha=0.45, zorder=2)
    ax.add_patch(Rectangle((-1, 0), 2, 1, facecolor=accent, alpha=0.045, lw=0, zorder=0))


def _keeper_name(players: pd.DataFrame, team_id: int) -> str:
    """Return the team's goalkeeper, preferring the starter."""
    if players is None or players.empty or "position" not in players.columns:
        return "Goalkeeper"
    keepers = players[
        players["team_id"].eq(team_id) & players["position"].astype(str).str.upper().eq("GK")
    ]
    if keepers.empty:
        return "Goalkeeper"
    if "is_first_xi" in keepers.columns:
        starters = keepers[as_bool(keepers["is_first_xi"])]
        if not starters.empty:
            keepers = starters
    return str(keepers.iloc[0]["name"])


def gk_saves(events, xg, players):
    """One goal frame per keeper: where every shot they faced crossed the line,
    which outcome it produced, how it was struck and what it was worth."""
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    base.amoled_header(
        fig,
        "Goalkeeper Goal Frames",
        "Every shot faced, plotted where it crossed the goal line · colour = outcome · shape = body part · marker size = xG",
    )

    shots = events[as_bool(events["is_shot"])].copy()
    if "is_penalty_shootout" in shots.columns:
        shots = shots[~as_bool(shots["is_penalty_shootout"])]
    shots["xG"] = pd.to_numeric(shots["xG"], errors="coerce").fillna(0).clip(lower=0)
    # Which goal a shot was heading for is the credited team's opponent's goal,
    # not the striker's opponent's. An own goal is logged on the scorer's own
    # team_id but entered their OWN net, so it belongs on their keeper's frame,
    # never on the opposition keeper's.
    shots["_credited_team"] = base.credited_team(shots)

    # Blocked shots never reach the keeper, so they are counted but not drawn:
    # the provider still records an intended crossing point for them, and
    # plotting it on a goalkeeper's frame implies a save situation that never
    # existed. Woodwork and off-target shots stay — the keeper had to read them.
    outcomes = [
        ("MissedShots", SHOT_MISS, "Off target"),
        ("ShotOnPost", SHOT_POST, "Woodwork"),
        ("SavedShot", SHOT_SAVED, "Saved"),
        ("Goal", SHOT_GOAL, "Goal"),
    ]

    panels = [
        (HOME_ID, AWAY_ID, HOME_NAME, AWAY_NAME, 0.055),
        (AWAY_ID, HOME_ID, AWAY_NAME, HOME_NAME, 0.545),
    ]
    for keeper_team, shooting_team, keeper_team_name, shooting_team_name, x0 in panels:
        accent = _team_mark_color(keeper_team)
        faced = shots[shots["_credited_team"].eq(shooting_team)]
        shot_type = faced["shot_whoscored_type"].astype(str)
        # The axes are aspect-locked, so height is derived from the width and
        # the frame's own limits — otherwise the drawing floats inside a box
        # that is taller than it needs to be.
        frame_h = 0.40 * (14 / 9) * (1.60 / 3.56)
        ax = fig.add_axes([x0, 0.470, 0.40, frame_h])
        _draw_goal_frame(ax, accent)

        plotted = 0
        for event_type, color, _label in outcomes:
            for _, row in faced[shot_type.eq(event_type)].iterrows():
                point = _placement_xy(row)
                if point is None:
                    continue
                # A wide or high miss is recorded well outside the frame; it is
                # kept at the edge of the drawing, since leaving the axes made
                # it vanish and the board then counted shots it did not show.
                px = float(np.clip(point[0], -1.64, 1.64))
                py = float(np.clip(point[1], -0.12, 1.30))
                body = str(row.get("body_part") or "").lower()
                marker = _BODY_PART_MARKERS.get(body, ("o", "Right foot"))[0]
                ax.scatter(
                    [px],
                    [py],
                    s=min(40 + float(row["xG"]) * 620, 380),
                    marker=marker,
                    facecolors=color,
                    edgecolors=BG,
                    linewidths=0.9,
                    alpha=0.95,
                    zorder=6 if event_type == "Goal" else 5,
                )
                plotted += 1

        if plotted == 0:
            ax.text(
                0,
                0.5,
                "No placement data recorded",
                color=MUTED,
                fontsize=9,
                ha="center",
                va="center",
                zorder=7,
            )

        keeper = _keeper_name(players, keeper_team)
        fig.text(x0, 0.815, keeper.upper(), color=accent, fontsize=14, fontweight="bold")
        fig.text(
            x0,
            0.788,
            f"{keeper_team_name} · {int(shot_type.isin(['Goal', 'SavedShot']).sum())} of "
            f"{len(faced)} shots faced on target · {int(shot_type.eq('MissedShots').sum())} off target"
            f" · {int(shot_type.eq('BlockedShot').sum())} blocked",
            color=MUTED,
            fontsize=8,
        )

        on_target = int(shot_type.isin(["Goal", "SavedShot"]).sum())
        conceded = int(shot_type.eq("Goal").sum())
        saves = int(shot_type.eq("SavedShot").sum())
        # Goals prevented: post-shot xG of the shots this keeper faced, from
        # the fitted model, minus what went in. Penalties and own goals are
        # left out, so it is not always the CONCEDED card subtracted from
        # something -- a penalty conceded does not count against him here.
        stopping = goalkeeper_shot_stopping(events, keeper_team, _keeper_name(players, keeper_team))
        prevented = f"{stopping['goals_prevented']:+.2f}".replace("-", "\N{MINUS SIGN}")
        cards = [
            ("On target", f"{on_target}"),
            ("Saves", f"{saves}"),
            ("Conceded", f"{conceded}"),
            ("Prevented", prevented),
            ("Save rate", f"{100 * saves / max(on_target, 1):.0f}%"),
            (
                "Claims",
                f"{int(events[events['team_id'].eq(keeper_team) & events['type'].eq('Claim')].shape[0])}",
            ),
            (
                "Sweeps",
                f"{int(events[events['team_id'].eq(keeper_team) & events['type'].eq('KeeperSweeper')].shape[0])}",
            ),
        ]
        card_w = 0.40 / len(cards)
        for idx, (label, value) in enumerate(cards):
            cx = x0 + idx * card_w
            fig.add_artist(
                Rectangle(
                    (cx, 0.290),
                    card_w * 0.93,
                    0.105,
                    transform=fig.transFigure,
                    facecolor=PANEL,
                    edgecolor=GRID,
                    lw=0.9,
                    zorder=1,
                )
            )
            fig.text(
                cx + card_w * 0.465,
                0.357,
                value,
                color=TEXT,
                fontsize=13,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=2,
            )
            fig.text(
                cx + card_w * 0.465,
                0.312,
                label.upper(),
                color=MUTED,
                fontsize=6.2,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=2,
            )

    # Shot-stopping is only half a keeper's match. Distribution says whether
    # they played out or launched it, and how much of that survived.
    for keeper_team, _shooting_team, _keeper_team_name, _shooter_name, x0 in panels:
        distribution = goalkeeper_distribution(
            events, keeper_team, _keeper_name(players, keeper_team)
        )
        if not distribution["distributions"]:
            continue
        fig.text(
            x0,
            0.258,
            f"DISTRIBUTION   {distribution['distributions']} passes  ·  "
            f"avg {distribution['avg_length_m']:.0f} m  ·  "
            f"{distribution['launch_share']:.0f}% launched  ·  "
            f"{distribution['completion']:.0f}% completed",
            color=MUTED,
            fontsize=7.4,
            fontweight="bold",
        )

    # What happened to the ball after each save: held, pushed somewhere safe,
    # or pushed back to an attacker. A save count treats all three alike.
    for keeper_team, _shooting_team, _keeper_team_name, _shooter_name, x0 in panels:
        stopping = goalkeeper_shot_stopping(events, keeper_team, _keeper_name(players, keeper_team))
        handled = stopping["caught"] + stopping["parried_safe"] + stopping["parried_danger"]
        parts = []
        if handled:
            parts.append(
                f"{stopping['caught']} held  ·  {stopping['parried_safe']} parried safe  ·  "
                f"{stopping['parried_danger']} into danger"
            )
        errors = stopping["errors_to_shot"]
        if errors:
            parts.append(f"{errors} error{'s' if errors != 1 else ''} led to a shot")
        if not parts:
            continue
        fig.text(
            x0,
            0.236,
            "HANDLING   " + "  ·  ".join(parts),
            color=MUTED,
            fontsize=7.4,
            fontweight="bold",
        )

    # Shared legend: outcome colour on one row, body-part shape on the next.
    for row_y, entries in (
        (0.205, [(("o"), color, label) for _t, color, label in outcomes]),
        (0.155, [(marker, TEXT, label) for marker, label in _BODY_PART_MARKERS.values()]),
    ):
        legend_x = 0.055
        for marker, color, label in entries:
            fig.add_artist(
                Line2D(
                    [legend_x],
                    [row_y],
                    marker=marker,
                    color=color,
                    lw=0,
                    markersize=8,
                    transform=fig.transFigure,
                )
            )
            fig.text(
                legend_x + 0.013,
                row_y,
                label.upper(),
                color=MUTED,
                fontsize=7.2,
                fontweight="bold",
                va="center",
            )
            legend_x += 0.013 + 0.0062 * len(label) + 0.028

    fig.text(
        0.055,
        0.098,
        "Goal frame seen from behind the shooter · off-target shots sit outside the posts · "
        "blocked shots are excluded because they never reached the keeper",
        fontsize=7,
        color=NEUTRAL,
    )
    fig.text(
        0.945,
        0.035,
        "FULL VISUAL REDESIGN · REAL MATCH DATA",
        ha="right",
        fontsize=8,
        color=NEUTRAL,
    )
    return save(fig, "11_goalkeeper_saves.png")


def zone14(events, team_id, number):
    team = events[events["team_id"].eq(team_id)].copy()
    successful = team["outcome"].astype(str).str.lower().eq("successful")
    zone = (
        successful
        & pd.to_numeric(team["end_x"], errors="coerce").between(70, 83)
        & pd.to_numeric(team["end_y"], errors="coerce").between(35, 65)
    )
    actions = team[zone].dropna(subset=["x", "y", "end_x", "end_y"])
    final_third = team[successful & (pd.to_numeric(team["end_x"], errors="coerce") >= 66.7)].copy()
    final_third["end_y_num"] = pd.to_numeric(final_third["end_y"], errors="coerce")
    final_third = final_third.dropna(subset=["end_y_num"])
    # Left to right across the page, which is high to low in the feed's width.
    lane_defs = [
        ("Left wing", 80, 100.0001),
        ("Left half-space", 60, 80),
        ("Central lane", 40, 60),
        ("Right half-space", 20, 40),
        ("Right wing", 0, 20),
    ]
    team_mark = _team_mark_color(team_id)
    # One colour a lane, none of them the team's own: the board used to paint
    # all five in the kit colour, so nothing told a wing from the centre.
    lane_colors = list(ZONE_LANE_COLOURS)
    lane_counts = [
        int(final_third["end_y_num"].between(lo, hi, inclusive="left").sum())
        for _, lo, hi in lane_defs
    ]
    fig, pitch, side = pitch_axes(
        f"Zone 14 & Five Lanes · {TEAM_NAME[team_id]}",
        "Final-third actions per lane · arrows = Zone 14 access, coloured by the lane they started in",
    )
    draw_long_pitch(pitch)
    crop_to_attack(pitch, side, 62.0)
    third_y = 66.7 * PITCH_LENGTH / 100
    for (label, lo, hi), color in zip(lane_defs, lane_colors):
        x1, _ = attack_xy([66.7], [lo])
        x2, _ = attack_xy([66.7], [min(hi, 100)])
        pitch.add_patch(
            Rectangle(
                (min(x1[0], x2[0]), third_y),
                abs(x2[0] - x1[0]),
                PITCH_LENGTH - third_y,
                facecolor=color,
                edgecolor=color,
                lw=0.9,
                alpha=0.30 if IS_LIGHT_THEME else 0.24,
                zorder=0,
            )
        )
    for idx, ((_, lo, hi), count, color) in enumerate(zip(lane_defs, lane_counts, lane_colors)):
        center_y = (lo + min(hi, 100)) / 2
        cx, cy = attack_xy([68.8], [center_y])
        pitch.text(
            cx[0],
            cy[0],
            str(count),
            ha="center",
            va="center",
            color=TEXT,
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", fc=PANEL, ec=color, lw=1.15, alpha=0.97),
            zorder=6,
        )
    zx1, zy1 = attack_xy([70], [35])
    zx2, zy2 = attack_xy([83], [65])
    pitch.add_patch(
        Rectangle(
            (min(zx1[0], zx2[0]), min(zy1[0], zy2[0])),
            abs(zx2[0] - zx1[0]),
            abs(zy2[0] - zy1[0]),
            facecolor=FOCUS,
            alpha=_HATCH_ALPHA,
            edgecolor=FOCUS,
            lw=1.8,
            hatch="//",
            zorder=1,
        )
    )
    pitch.text(
        0,
        min(zy1[0], zy2[0]) + 0.9,
        "ZONE 14",
        color=FOCUS,
        fontsize=6.5,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=5,
    )

    def lane_index(width_value):
        for index, (_, low, high) in enumerate(lane_defs):
            if low <= width_value < high:
                return index
        return len(lane_defs) // 2

    # Each arrow takes the colour of the lane it started in, so the board says
    # where the Zone 14 access came from as well as that it happened.
    for _, row in actions.iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]])
        ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        arrow = pitch.annotate(
            "",
            xy=(ex[0], ey[0]),
            xytext=(sx[0], sy[0]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=lane_colors[lane_index(float(row["y"]))],
                alpha=0.95,
                lw=1.9,
                mutation_scale=12,
            ),
            zorder=5,
        )
        if arrow.arrow_patch is not None:
            arrow.arrow_patch.set_path_effects(
                [path_effects.Stroke(linewidth=3.6, foreground=BG), path_effects.Normal()]
            )
    top = actions.groupby("player").size().sort_values(ascending=False).head(3)
    side_title(side, "FIVE ATTACKING LANES")
    for idx, ((label, _, _), value, color) in enumerate(zip(lane_defs, lane_counts, lane_colors)):
        y = 0.81 - idx * 0.083
        side.add_patch(
            Rectangle(
                (0.08, y - 0.016), 0.035, 0.032, facecolor=color, edgecolor=TEXT, lw=0.4, alpha=0.9
            )
        )
        side.text(0.15, y, label, color=TEXT, fontsize=8.5, va="center")
        side.text(
            0.92,
            y,
            str(value),
            color=TEXT,
            fontsize=8.8,
            fontweight="bold",
            ha="right",
            va="center",
        )
        side.plot([0.08, 0.92], [y - 0.037, y - 0.037], color=GRID, lw=0.55, alpha=0.7)
    side.text(0.08, 0.34, "ZONE 14 CONTRIBUTORS", color=MUTED, fontsize=7.5, fontweight="bold")
    for idx, (name, value) in enumerate(top.items()):
        y = 0.285 - idx * 0.06
        side.text(0.08, y, _surname(name), color=TEXT, fontsize=8, va="center")
        side.text(
            0.92,
            y,
            str(int(value)),
            color=TEXT,
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="center",
        )
    side.text(
        0.08, 0.075, f"Zone 14 entries: {len(actions)}", color=TEXT, fontsize=9.5, fontweight="bold"
    )
    return save(fig, f"{number:02d}_zone14_{_team_slug(team_id)}.png")


def _post_match_metric_panel(ax, title, rows):
    """Use the shared Opta-style comparison contract inside dashboard panels."""
    display_rows = [
        (f"{label} ↓" if lower_better else label, home_value, away_value, fmt)
        for label, home_value, away_value, fmt, lower_better in rows
    ]
    base.row_dot_plot(ax, display_rows, title.upper())


def _derived_expected_assists(events: pd.DataFrame, team_id: int) -> float:
    """Team xA proxy: xG of the shot following each provider-tagged key pass."""
    work = events.copy()
    work["minute_num"] = pd.to_numeric(work.get("minute"), errors="coerce").fillna(0)
    work["second_num"] = pd.to_numeric(work.get("second"), errors="coerce").fillna(0)
    work["period_num"] = pd.to_numeric(work.get("period"), errors="coerce").fillna(0)
    work = work.sort_values(["period_num", "minute_num", "second_num", "event_id"]).reset_index(
        drop=True
    )
    live = ~as_bool(work.get("is_penalty_shootout", pd.Series(False, index=work.index)))
    key_pass = as_bool(work.get("is_key_pass", pd.Series(False, index=work.index))) & live
    shot = as_bool(work.get("is_shot", pd.Series(False, index=work.index))) & live
    total = 0.0
    for idx in work.index[key_pass & work["team_id"].eq(team_id)]:
        source = work.loc[idx]
        source_time = float(source["minute_num"]) * 60 + float(source["second_num"])
        for next_idx in range(idx + 1, len(work)):
            candidate = work.loc[next_idx]
            if candidate["period_num"] != source["period_num"]:
                break
            elapsed = (
                float(candidate["minute_num"]) * 60 + float(candidate["second_num"]) - source_time
            )
            if elapsed > 20:
                break
            if candidate["team_id"] == team_id and bool(shot.loc[next_idx]):
                total += float(pd.to_numeric(candidate.get("xG"), errors="coerce") or 0.0)
                break
    return total


def post_match_advanced_dashboard(events, xg, team_metrics):
    """The report's single numeric reference: attack, creation, defence and process.

    This absorbed four pages that repeated its rows in a different arrangement —
    the shot profile, the xG summary, the match-statistics page and the separate
    advanced-metrics page — plus the unique rows of the defensive summary. Every
    exact match value now lives here once.
    """
    home_xg, away_xg = xg_row(xg, HOME_NAME), xg_row(xg, AWAY_NAME)
    info = {"home_id": HOME_ID, "away_id": AWAY_ID, "home_name": HOME_NAME, "away_name": AWAY_NAME}
    try:
        ppda = compute_ppda_both(info, events)
        home_ppda = float(ppda["home"]["ppda"] or 0)
        away_ppda = float(ppda["away"]["ppda"] or 0)
    except Exception:
        home_ppda = away_ppda = 0.0

    metric = lambda side, key: float(base.metric_lookup(team_metrics, side, key))
    home_counts, away_counts = (
        team_event_counts(events, HOME_ID),
        team_event_counts(events, AWAY_ID),
    )
    home_xa = _derived_expected_assists(events, HOME_ID)
    away_xa = _derived_expected_assists(events, AWAY_ID)
    attack_rows = [
        ("Shots", float(home_xg.get("shots", 0)), float(away_xg.get("shots", 0)), "{:.0f}", False),
        (
            "Shots on target",
            float(home_xg.get("on_target", 0)),
            float(away_xg.get("on_target", 0)),
            "{:.0f}",
            False,
        ),
        (
            "Big chances",
            float(home_xg.get("big_chances", 0)),
            float(away_xg.get("big_chances", 0)),
            "{:.0f}",
            False,
        ),
        (
            "Expected goals (xG)",
            float(home_xg.get("xG", 0)),
            float(away_xg.get("xG", 0)),
            "{:.2f}",
            False,
        ),
        (
            "Post-shot xG",
            float(home_xg.get("xGoT", 0)),
            float(away_xg.get("xGoT", 0)),
            "{:.2f}",
            False,
        ),
        (
            "xG per shot",
            float(home_xg.get("xG_per_shot", 0)),
            float(away_xg.get("xG_per_shot", 0)),
            "{:.3f}",
            False,
        ),
        (
            "Transition xG",
            metric("home", "transition_xG"),
            metric("away", "transition_xG"),
            "{:.2f}",
            False,
        ),
        (
            "Transition shot rate",
            metric("home", "transition_shot_rate"),
            metric("away", "transition_shot_rate"),
            "{:.1f}%",
            False,
        ),
    ]
    creation_rows = [
        (
            "Possession share",
            metric("home", "possession_share"),
            metric("away", "possession_share"),
            "{:.1f}%",
            False,
        ),
        ("Expected assists (xA)", home_xa, away_xa, "{:.2f}", False),
        ("Open-play xT", float(home_xg.get("xT", 0)), float(away_xg.get("xT", 0)), "{:.2f}", False),
        (
            "Field tilt",
            metric("home", "field_tilt"),
            metric("away", "field_tilt"),
            "{:.1f}%",
            False,
        ),
        (
            "Progressive passes",
            metric("home", "progressive_passes"),
            metric("away", "progressive_passes"),
            "{:.0f}",
            False,
        ),
        (
            "Final-third entries",
            metric("home", "final_third_entries"),
            metric("away", "final_third_entries"),
            "{:.0f}",
            False,
        ),
        (
            "Deep completions",
            metric("home", "deep_completions"),
            metric("away", "deep_completions"),
            "{:.0f}",
            False,
        ),
        (
            "Box entries",
            metric("home", "box_entries"),
            metric("away", "box_entries"),
            "{:.0f}",
            False,
        ),
    ]
    defence_rows = [
        ("PPDA", home_ppda, away_ppda, "{:.2f}", True),
        ("Tackles", float(home_counts["Tackles"]), float(away_counts["Tackles"]), "{:.0f}", False),
        (
            "Interceptions",
            float(home_counts["Interceptions"]),
            float(away_counts["Interceptions"]),
            "{:.0f}",
            False,
        ),
        (
            "Possession regains",
            metric("home", "possession_regains"),
            metric("away", "possession_regains"),
            "{:.0f}",
            False,
        ),
        (
            "High regains",
            metric("home", "high_regains"),
            metric("away", "high_regains"),
            "{:.0f}",
            False,
        ),
        (
            "Counterpress success",
            metric("home", "counterpress_success_rate"),
            metric("away", "counterpress_success_rate"),
            "{:.1f}%",
            False,
        ),
        # Absorbed from the former standalone defensive-summary page: these are
        # the rows it carried that were not already here.
        (
            "Recoveries",
            float(home_counts["Recoveries"]),
            float(away_counts["Recoveries"]),
            "{:.0f}",
            False,
        ),
        (
            "Clearances",
            float(home_counts["Clearances"]),
            float(away_counts["Clearances"]),
            "{:.0f}",
            False,
        ),
        ("Blocks", float(home_counts["Blocks"]), float(away_counts["Blocks"]), "{:.0f}", False),
        ("Fouls", float(home_counts["Fouls"]), float(away_counts["Fouls"]), "{:.0f}", True),
    ]
    # Absorbed from the former standalone advanced-metrics page: process,
    # sequence value and the risk carried behind the ball.
    process_rows = [
        (
            "Transitions",
            metric("home", "transitions"),
            metric("away", "transitions"),
            "{:.0f}",
            False,
        ),
        (
            "Build-up success",
            metric("home", "build_up_success_rate"),
            metric("away", "build_up_success_rate"),
            "{:.1f}%",
            False,
        ),
        (
            "Final-third entry efficiency",
            metric("home", "final_third_entry_efficiency"),
            metric("away", "final_third_entry_efficiency"),
            "{:.1f}%",
            False,
        ),
        (
            "Box entry → shot",
            metric("home", "box_entry_to_shot_rate"),
            metric("away", "box_entry_to_shot_rate"),
            "{:.1f}%",
            False,
        ),
        (
            "Sequence xT",
            metric("home", "sequence_xT"),
            metric("away", "sequence_xT"),
            "{:.2f}",
            False,
        ),
        (
            "Transition xT",
            metric("home", "transition_xT"),
            metric("away", "transition_xT"),
            "{:.2f}",
            False,
        ),
        (
            "Directness",
            metric("home", "directness"),
            metric("away", "directness"),
            "{:.1f}%",
            False,
        ),
        (
            "Transition exposure",
            metric("home", "rest_defence_exposures"),
            metric("away", "rest_defence_exposures"),
            "{:.0f}",
            True,
        ),
    ]

    fig = plt.figure(figsize=(19, 9), facecolor=BG)
    base.amoled_header(
        fig,
        "Post-Match Advanced Dashboard",
        "32 indicators · attack, creation, defence and process · shootout excluded",
        active_team=None,
    )

    panels = [
        ("Attacking output", attack_rows),
        ("Creation & control", creation_rows),
        ("Defensive control", defence_rows),
        ("Process & risk", process_rows),
    ]
    for left, (title, rows) in zip([0.035, 0.275, 0.515, 0.755], panels):
        ax = fig.add_axes([left, 0.18, 0.22, 0.61])
        _post_match_metric_panel(ax, title, rows)

    territory_team = (
        HOME_NAME if metric("home", "field_tilt") > metric("away", "field_tilt") else AWAY_NAME
    )
    quality_team = (
        HOME_NAME
        if float(home_xg.get("xG_per_shot", 0)) > float(away_xg.get("xG_per_shot", 0))
        else AWAY_NAME
    )
    secure_team = (
        HOME_NAME
        if metric("home", "rest_defence_vulnerability")
        < metric("away", "rest_defence_vulnerability")
        else AWAY_NAME
    )
    fig.text(0.035, 0.112, "MATCH READ", color=FOCUS, fontsize=7.5, fontweight="bold")
    fig.text(
        0.093,
        0.112,
        f"{territory_team} controlled more territory; {quality_team} created the cleaner average shot and {secure_team} protected attacking possessions more securely.",
        color=TEXT,
        fontsize=8.5,
    )
    fig.text(
        0.035,
        0.066,
        "xA = xG of the shot following each provider-tagged key pass (within 20 seconds).",
        color=MUTED,
        fontsize=7,
    )
    fig.text(
        0.965,
        0.066,
        "↓ LOWER IS BETTER   ·   REAL MATCH EVENTS",
        color=NEUTRAL,
        fontsize=7,
        ha="right",
    )
    return save(fig, "14_post_match_advanced_dashboard.png")


def progressive(events, team_id, number):
    prog = (
        events[progressive_pass_mask(events) & events["team_id"].eq(team_id)]
        .copy()
        .dropna(subset=["x", "y", "end_x", "end_y"])
    )
    prog["xT"] = pd.to_numeric(prog["xT"], errors="coerce").fillna(0)
    fig, pitch, side = pitch_axes(
        f"Progressive Passes · {TEAM_NAME[team_id]}",
        "Average progressive pass from each zone · the five strongest by xT added are drawn on top",
    )
    draw_long_pitch(pitch)
    # The volume is a flow map, one arrow per zone, so sixty-odd progressive
    # passes read as the routes the side used rather than as a tangle. The five
    # strongest by xT stand out on top, each with its xT figure.
    draw_zone_flow(
        pitch,
        prog,
        np.ones(len(prog), dtype=bool),
        _team_mark_color(team_id),
        min_count=3,
        labels=False,
    )
    for _, row in prog.nlargest(5, "xT").iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]])
        ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        arrow = pitch.annotate(
            "",
            xy=(ex[0], ey[0]),
            xytext=(sx[0], sy[0]),
            arrowprops=dict(
                arrowstyle="-|>", color=EVENT_HIGHLIGHT, alpha=0.97, lw=2.2, mutation_scale=13
            ),
            zorder=6,
        )
        if arrow.arrow_patch is not None:
            arrow.arrow_patch.set_path_effects(
                [path_effects.Stroke(linewidth=4.2, foreground=BG), path_effects.Normal()]
            )
        pitch.text(
            ex[0],
            ey[0] + 2.0,
            f"{row['xT']:.2f}",
            color=TEXT,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=8,
            path_effects=[path_effects.withStroke(linewidth=2.6, foreground=BG)],
        )
    top = prog.groupby("player").size().sort_values(ascending=False).head(7)
    side_title(side, "TOP PROGRESSORS")
    side_rows(side, [(_surname(name), str(int(value))) for name, value in top.items()])
    side.text(
        0.08,
        0.14,
        f"Team-colour arrows = average of all {len(prog)} progressive passes",
        color=_team_mark_color(team_id),
        fontsize=8.2,
    )
    side.text(
        0.08, 0.09, f"{HIGHLIGHT_LABEL} = top 5 by xT added", color=EVENT_HIGHLIGHT, fontsize=8.4
    )
    return save(fig, f"{number:02d}_progressive_{_team_slug(team_id)}.png")


def crosses(events, team_id, number):
    mask = cross_mask(events)
    frame = (
        events[mask & events["team_id"].eq(team_id)]
        .copy()
        .dropna(subset=["x", "y", "end_x", "end_y"])
    )
    success = frame["outcome"].astype(str).str.lower().eq("successful")
    fig, pitch, side = pitch_axes(
        f"Crosses · {TEAM_NAME[team_id]}",
        "Cross origins and targets · completed deliveries use filled arrowheads",
    )
    draw_long_pitch(pitch)
    if not frame.empty:
        pitch.set_ylim(
            max(-3, min(48, float(frame[["x", "end_x"]].min().min()) * 1.05 - 5)), PITCH_LENGTH + 3
        )
    team_mark = _team_mark_color(team_id)
    for idx, row in frame.iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]])
        ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        good = bool(success.loc[idx])
        pitch.annotate(
            "",
            xy=(ex[0], ey[0]),
            xytext=(sx[0], sy[0]),
            arrowprops=dict(
                arrowstyle="-|>" if good else "->",
                color=team_mark,
                alpha=0.82 if good else 0.38,
                lw=1.15 if good else 0.68,
                linestyle="-" if good else FAILURE_DASH,
                mutation_scale=9,
            ),
        )
    completed = int(success.sum())
    rate = 100 * completed / max(len(frame), 1)
    side_title(side, "CROSSING OUTPUT")
    side_kpis(
        side,
        [
            ("Crosses", len(frame)),
            ("Completed", completed),
            ("Completion", f"{rate:.1f}%"),
            (
                "Open-play",
                int(
                    (~frame["qualifier_names"].astype(str).str.lower().str.contains("corner")).sum()
                ),
            ),
        ],
    )
    side.plot([0.09, 0.24], [0.12, 0.12], color=team_mark, lw=1.35)
    side.text(0.28, 0.12, "Completed", color=TEXT, fontsize=7.5, va="center")
    side.plot([0.55, 0.70], [0.12, 0.12], color=team_mark, lw=1.0, linestyle=FAILURE_DASH)
    side.text(0.74, 0.12, "Incomplete", color=TEXT, fontsize=7.5, va="center")
    return save(fig, f"{number:02d}_crosses_{_team_slug(team_id)}.png")


def defensive_activity(events, team_id, number):
    event_types = events["type"].astype(str)
    non_foul_actions = event_types.isin(["Tackle", "Interception", "BallRecovery", "Clearance"])
    committed_fouls = fouls_committed_mask(events)
    own_actions = (
        events[events["team_id"].eq(team_id) & (non_foul_actions | committed_fouls)]
        .dropna(subset=["x", "y"])
        .copy()
    )
    opponent_id = AWAY_ID if team_id == HOME_ID else HOME_ID
    block_actions = defensive_block_events(events, team_id, opponent_id).dropna(subset=["x", "y"])
    actions = pd.concat([own_actions, block_actions], ignore_index=True, sort=False)
    fig, pitch, side = pitch_axes(
        f"Defensive Activity · {TEAM_NAME[team_id]}",
        "Smoothed team-colour heatmap; bright colour and shape identify each action type",
    )
    hx, hy = attack_xy(actions["x"], actions["y"])
    heat, _, _ = np.histogram2d(
        hx,
        hy,
        bins=[21, 36],
        range=[[-PITCH_WIDTH / 2, PITCH_WIDTH / 2], [0, PITCH_LENGTH]],
    )
    heat = gaussian_filter(heat.astype(float), sigma=1.45)
    heat = heat / heat.max() if heat.max() > 0 else heat
    team_mark = _team_mark_color(team_id)
    cmap = LinearSegmentedColormap.from_list(
        f"def_{team_id}", [BG, PANEL_2, TEAM_COLOR[team_id], team_mark]
    )
    pitch.imshow(
        heat.T,
        extent=[-PITCH_WIDTH / 2, PITCH_WIDTH / 2, 0, PITCH_LENGTH],
        origin="lower",
        cmap=cmap,
        aspect="equal",
        vmin=0,
        vmax=1,
        alpha=0.62,
        interpolation="bicubic",
    )
    draw_long_pitch(pitch)
    # How high up the pitch the work was done: one line at the average height of
    # the actions, so a high press and a low block are told apart by where it sits.
    average_height = float(
        np.mean(
            attack_xy(actions["x"].to_numpy(), actions["y"].to_numpy())[1]
            if len(actions)
            else [0.0]
        )
    )
    pitch.axhline(average_height, color=FOCUS, lw=1.2, ls=(0, (5, 4)), alpha=0.9, zorder=4)
    marker_map = {
        "Tackle": "o",
        "Interception": "D",
        "BallRecovery": "s",
        "Clearance": "^",
        "BlockedShot": "P",
        "Foul": "X",
    }
    action_colors = {
        "Tackle": "#67E8F9",
        "Interception": "#C4B5FD",
        "BallRecovery": "#86EFAC",
        "Clearance": "#FDE68A",
        "BlockedShot": "#F9A8D4",
        "Foul": "#F9A8D4",
    }
    for event_type, marker in marker_map.items():
        subset = actions[actions["type"].astype(str).eq(event_type)]
        if subset.empty:
            continue
        px, py = attack_xy(subset["x"], subset["y"])
        pitch.scatter(
            px,
            py,
            marker=marker,
            s=84,
            facecolors=action_colors[event_type],
            edgecolors=BG,
            linewidth=0.9,
            alpha=0.98,
            zorder=5,
        )
    counts = team_event_counts(events, team_id)
    side_title(side, "ACTION TYPE LEGEND")
    label_to_event = {
        "Tackles": "Tackle",
        "Interceptions": "Interception",
        "Recoveries": "BallRecovery",
        "Clearances": "Clearance",
        "Blocks": "BlockedShot",
        "Fouls": "Foul",
    }
    for idx, (label, value) in enumerate(counts.items()):
        y = 0.81 - idx * 0.095
        marker = marker_map[label_to_event[label]]
        event_type = label_to_event[label]
        side.scatter(
            [0.13],
            [y],
            s=52,
            marker=marker,
            facecolors=action_colors[event_type],
            edgecolors=BG,
            linewidth=0.9,
        )
        side.text(0.21, y, label, color=TEXT, fontsize=8.5, va="center")
        side.text(
            0.90,
            y,
            str(value),
            color=TEXT,
            fontsize=8.5,
            fontweight="bold",
            ha="right",
            va="center",
        )
        side.plot([0.08, 0.92], [y - 0.043, y - 0.043], color=GRID, lw=0.55, alpha=0.7)
    side.plot([0.09, 0.17], [0.235, 0.235], color=FOCUS, lw=1.4, ls=(0, (5, 4)))
    side.text(
        0.21,
        0.235,
        f"Average action height · {average_height:.0f} m",
        color=TEXT,
        fontsize=8,
        va="center",
    )
    side.add_patch(
        Rectangle((0.09, 0.105), 0.08, 0.055, facecolor=team_mark, edgecolor=GRID, alpha=0.82)
    )
    side.text(
        0.21,
        0.132,
        f"{TEAM_NAME[team_id]} colour = action density",
        color=TEXT,
        fontsize=8,
        va="center",
    )
    return save(fig, f"{number:02d}_defensive_activity_{_team_slug(team_id)}.png")


def average_positions(events, players, team_id, number, half):
    positions, _edges, sub_on, sub_off, substitutions, _completed_links = _half_network_data(
        events, players, team_id, half
    )
    half_label = "First Half" if half == 1 else "Second Half"
    fig, pitch, side = pitch_axes(
        f"Average Positions · {TEAM_NAME[team_id]} · {half_label}",
        f"All {len(positions)} participants shown · corrected left/right orientation · square = came on · {_FOCUS_WORD} outline = went off",
    )
    draw_long_pitch(pitch)
    display = {}
    for name, row in positions.iterrows():
        px, py = player_position_xy([row["x"]], [row["y"]])
        display[str(name)] = (float(px[0]), float(py[0]), float(row["touches"]))
    display = _separate_network_positions(display, min_gap=6.3)
    _link_low, outline_color, _link_strong = network_link_palette(TEAM_COLOR[team_id])
    max_touch = max([value[2] for value in display.values()] or [1])
    shirts = shirt_number_map(players)
    radii = {
        name: _network_node_radius(touches, max_touch)
        for name, (_x, _y, touches) in display.items()
    }
    for name, (px, py, touches) in display.items():
        entered = name in sub_on
        left = name in sub_off
        pitch.scatter(
            px,
            py,
            s=260 + 640 * touches / max_touch,
            marker="s" if entered else "o",
            color=_team_mark_color(team_id),
            edgecolor=FOCUS if left else outline_color,
            linewidth=2.3 if left else 1.15,
            zorder=4,
        )
        draw_node_label(
            pitch,
            px,
            py,
            name,
            touches,
            max_touch,
            node_color=_team_mark_color(team_id),
            shirt=shirts.get(str(name)),
            node_radius=radii[name],
            neighbours=_node_neighbours(display, radii, name),
        )

    side_title(side, "HALF PARTICIPATION")
    side.text(
        0.92,
        0.94,
        f"{len(positions)} players",
        color=TEXT,
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="top",
    )
    active = positions.sort_values("touches", ascending=False).head(5)
    side_rows(
        side,
        [
            (compact_player_label(name, 16), str(int(row["touches"])))
            for name, row in active.iterrows()
        ],
        start=0.81,
        gap=0.075,
    )
    side.text(0.08, 0.40, "SUBSTITUTIONS", color=MUTED, fontsize=7.5, fontweight="bold")
    if substitutions:
        for idx, (minute, on_name, off_name) in enumerate(substitutions[:5]):
            y = 0.35 - idx * 0.052
            side.text(
                0.08, y, f"{minute}′", color=TEXT, fontsize=7.5, fontweight="bold", va="center"
            )
            change = (
                f"{off_name} OFF AT INTERVAL"
                if on_name == "—"
                else f"{on_name} IN  ·  {off_name} OFF"
            )
            side.text(0.19, y, change, color=TEXT, fontsize=7.2, va="center")
    else:
        side.text(0.08, 0.34, "No in-half changes", color=MUTED, fontsize=8)
    side.scatter(
        [0.06, 0.34],
        [0.075, 0.075],
        s=[65, 65],
        marker="o",
        color=_team_mark_color(team_id),
        edgecolor=[TEXT, FOCUS],
        linewidth=[1.0, 2.1],
    )
    side.scatter(
        [0.60],
        [0.075],
        s=65,
        marker="s",
        color=_team_mark_color(team_id),
        edgecolor=TEXT,
        linewidth=1.0,
    )
    side.text(0.10, 0.075, "Began half", color=TEXT, fontsize=6.8, va="center")
    side.text(0.38, 0.075, "Went off", color=TEXT, fontsize=6.8, va="center")
    side.text(0.64, 0.075, "Came on", color=TEXT, fontsize=6.8, va="center")
    suffix = "1h" if half == 1 else "2h"
    return save(
        fig,
        f"{number:02d}{'a' if half == 1 else 'b'}_average_positions_{_team_slug(team_id)}_{suffix}.png",
    )


def dominating_zones(events):
    tmask = touch_mask(events)
    home = events[tmask & events["team_id"].eq(HOME_ID)].dropna(subset=["x", "y"])
    away = events[tmask & events["team_id"].eq(AWAY_ID)].dropna(subset=["x", "y"])
    hh, _, _ = np.histogram2d(home["y"], home["x"], bins=[5, 7], range=[[0, 100], [0, 100]])
    ah, _, _ = np.histogram2d(away["y"], away["x"], bins=[5, 7], range=[[0, 100], [0, 100]])
    total = hh + ah
    diff = np.divide(hh - ah, total, out=np.zeros_like(total), where=total > 0)
    counts = hh - ah
    fig, pitch, side = pitch_axes(
        "Dominating Zones",
        "Touch-share difference · diverging scale centred on an even 50/50 split",
    )
    cmap = LinearSegmentedColormap.from_list("dom_full", [AWAY, PANEL_2, HOME])
    image = pitch.imshow(
        diff.T,
        extent=[-PITCH_WIDTH / 2, PITCH_WIDTH / 2, 0, PITCH_LENGTH],
        origin="lower",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        aspect="equal",
    )
    draw_long_pitch(pitch)
    for ix in range(5):
        for iy in range(7):
            px = -PITCH_WIDTH / 2 + (ix + 0.5) * PITCH_WIDTH / 5
            py = (iy + 0.5) * PITCH_LENGTH / 7
            # Diverging ramp: both ends carry a team colour, so either end can
            # be light. Read the label colour off the cell it sits on.
            cell_fill = mcolors.to_hex(cmap((float(diff[ix, iy]) + 1.0) / 2.0))
            pitch.text(
                px,
                py,
                f"{int(counts[ix, iy]):+d}",
                ha="center",
                va="center",
                color=text_on_fill(cell_fill),
                fontsize=9.5,
                fontweight="bold",
                path_effects=label_outline(cell_fill),
            )
    cbar = fig.colorbar(image, ax=pitch, fraction=0.035, pad=0.02, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels([AWAY_NAME, "Balanced", HOME_NAME])
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.outline.set_edgecolor(GRID)
    side_title(side, "TERRITORY TOTALS")
    side_kpis(
        side,
        [
            (
                f"{HOME_NAME} touches",
                f"{len(home)}  ({100 * len(home) / max(len(home) + len(away), 1):.0f}%)",
            ),
            (
                f"{AWAY_NAME} touches",
                f"{len(away)}  ({100 * len(away) / max(len(home) + len(away), 1):.0f}%)",
            ),
            ("Difference", f"{len(home) - len(away):+d}"),
        ],
    )
    return save(fig, "24_dominating_zones.png")


# The lateral thirds, in the provider's width units. This feed numbers the
# width from the RIGHT touchline: a right back's touches average y≈19 and a
# left back's y≈81, which is the opposite of the reading the axis name
# suggests. Every corridor decision below goes through these two bounds rather
# than restating the convention, so there is one place to be wrong.
_RIGHT_CORRIDOR_MAX = 100.0 / 3.0
_LEFT_CORRIDOR_MIN = 200.0 / 3.0


def _attacking_corridors(events, team_id):
    """How one side's final-third entries split across the three corridors.

    An entry is the unit rather than a touch. Touches count the same posession
    many times over and reward a side that circulates in front of the block;
    an entry is one arrival in the final third, which is the thing a corridor
    share is meant to describe.

    The corridor is read off where the entry LANDED (``end_y``), not where it
    started. A switch that begins on the right and arrives on the left is an
    attack down the left, and the receiving end is the side the opponent had
    to defend.
    """
    frame = events[final_third_entry_mask(events) & events["team_id"].eq(team_id)]
    frame = frame.dropna(subset=["end_y"])
    landing = pd.to_numeric(frame["end_y"], errors="coerce").dropna()
    counts = (
        int((landing >= _LEFT_CORRIDOR_MIN).sum()),
        int(((landing > _RIGHT_CORRIDOR_MAX) & (landing < _LEFT_CORRIDOR_MIN)).sum()),
        int((landing <= _RIGHT_CORRIDOR_MAX).sum()),
    )
    total = sum(counts)
    shares = tuple(100.0 * c / total for c in counts) if total else (0.0, 0.0, 0.0)
    return counts, shares, total


def _corridor_arrow(ax, x_tail, x_head, y, half_height, colour):
    """One chevron, filled with a gradient that brightens towards the goal.

    A flat fill would say the same thing at both ends. The ramp runs dim at the
    halfway line to full colour at the head, so the direction of play is in the
    shape itself and the arrow does not need a second mark to carry it.

    Matplotlib fills a polygon with one colour, so the gradient is an image
    clipped to the chevron rather than a property of the patch.
    """
    pointing_left = x_head < x_tail
    depth = abs(x_tail - x_head) * 0.16
    direction = -1.0 if pointing_left else 1.0
    # Head tip, its two shoulders, the two tail corners, and the notch pulled
    # in from the tail edge towards the head -- the same chevron either way up.
    outline = [
        (x_head, y),
        (x_head - direction * depth, y + half_height),
        (x_tail, y + half_height),
        (x_tail + direction * depth, y),
        (x_tail, y - half_height),
        (x_head - direction * depth, y - half_height),
    ]
    # A hairline in the team colour around the whole shape. The fill fades
    # towards the page at the tail, and on the light page that end reached the
    # background before the chevron ended: the arrow lost its own outline.
    chevron = Polygon(
        outline, closed=True, facecolor="none", edgecolor=colour, linewidth=0.9, zorder=4
    )
    ax.add_patch(chevron)
    dim = mcolors.to_hex(_mix(colour, BG, 0.55 if IS_LIGHT_THEME else 0.72))
    ramp = LinearSegmentedColormap.from_list("corridor", [dim, colour])
    gradient = np.linspace(0.0, 1.0, 256).reshape(1, -1)
    if pointing_left:
        gradient = gradient[:, ::-1]
    left, right = min(x_head, x_tail), max(x_head, x_tail)
    image = ax.imshow(
        gradient,
        extent=[left, right, y - half_height, y + half_height],
        cmap=ramp,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        zorder=3,
        interpolation="bilinear",
    )
    image.set_clip_path(chevron)
    return chevron


def _mix(colour, other, weight):
    """``weight`` of ``other`` mixed into ``colour``."""
    a = np.array(mcolors.to_rgb(colour))
    b = np.array(mcolors.to_rgb(other))
    return tuple(a * (1.0 - weight) + b * weight)


def match_statistics(events, match_info):
    """The whole match as two columns of counted actions, plus the press.

    The package opens on the xG flow, which is the story of the chances and
    nothing else. A reader who wants the ordinary totals -- shots, passes,
    duels, clearances -- had to assemble them from a dozen later pages or from
    the PDF's tables.

    The page already existed: match_report draws it, and the full pipeline
    saved it, but generate_match_package never did, so the redesign package
    shipped without it. This is the same figure, captured and saved with the
    package's own chrome rather than redrawn.
    """
    from football_analysis.visuals.tactical_visualizations import make_match_stats_v2

    fig = make_match_stats_v2(events, match_info, compute_ppda_both(match_info, events))

    # The captured figure arrives wearing match_report's own chrome: an eyebrow
    # and title at the top, a score and report tag at the foot. Both are
    # replaced here by the package's, so the page carries the same identity as
    # the fifty either side of it and states the score once.
    #
    # save() can strip a header it did not draw, but it does so by reading the
    # figure's own text: it takes the largest type above 0.84 as the title and
    # the next text it finds as the subtitle. Here that next text is the
    # eyebrow's bullet, and the fixture name in the title made it pick a team
    # to highlight on a page that belongs to both. The header is set here
    # instead, and the flag tells save() the work is done.
    def _chrome_y(item):
        """Where a figure-level artist sits, in figure fractions, or None.

        The chrome is not all text: the eyebrow's bullet is a Circle and the
        footer rule a Rectangle, both added to fig.artists in figure
        coordinates. Hiding only the text left the bullet floating over the
        page with nothing beside it.
        """
        for read in (
            lambda: item.get_position()[1],
            lambda: item.center[1],
            lambda: item.get_xy()[1],
        ):
            try:
                return float(read())
            except Exception:
                continue
        return None

    for item in list(fig.texts) + list(fig.artists):
        y = _chrome_y(item)
        if y is not None and (y >= 0.84 or y < 0.06):
            item.set_visible(False)
    base.amoled_header(
        fig,
        "Match Statistics",
        "Attack, passing, pressing and defensive actions · every figure counted from the event stream",
    )
    return save(fig, "01b_match_statistics.png")


def attacking_zones(events):
    """Which corridor each side attacked through, both sides on one pitch.

    The package already located attacks -- box entries, final-third entries,
    the xT map -- but each of those answers where the ball went with a cloud of
    marks the reader has to average by eye. The one question a coach asks first
    of an opponent is which side they come down, and no page answered it as a
    number.

    Each side is drawn attacking its own way, so the corridors sit where the
    players stood: a team attacking towards the left of the page has its own
    left along the bottom touchline. Arrow length carries the share, and every
    arrow is named as well, because a reader should not have to derive which
    touchline was whose left from the direction of the chevrons.
    """
    fig = plt.figure(figsize=(13.0, 8.6), facecolor=BG)
    base.amoled_header(
        fig,
        "Attacking Zones",
        "Share of each side's final-third entries by corridor · corridor read where the entry landed",
    )
    ax = fig.add_axes([0.055, 0.105, 0.89, 0.68])
    base.draw_pitch(ax)
    # Room under the touchline for the two team names.
    ax.set_ylim(-11, 100)

    rows = (78.0, 50.0, 22.0)
    # Attacking towards the left of the page, a side's own left hand is the
    # bottom touchline; attacking towards the right it is the top one. The
    # corridor names are drawn either way, so the geometry only has to be right
    # rather than also obvious.
    layouts = (
        (HOME_ID, HOME_NAME, -1.0, 48.0, 4.0, ("Right", "Centre", "Left")),
        (AWAY_ID, AWAY_NAME, 1.0, 52.0, 96.0, ("Left", "Centre", "Right")),
    )
    summary = {}
    for team_id, team_name, direction, x_tail, x_limit, row_names in layouts:
        counts, shares, total = _attacking_corridors(events, team_id)
        by_name = dict(zip(("Left", "Centre", "Right"), zip(shares, counts)))
        # What each corridor produced, so the share of entries can be set beside
        # the share of chances: shots are filed by the lane they were taken from,
        # the feed's width running from the attacking side's right touchline.
        team_shots = events[
            events["team_id"].eq(team_id)
            & as_bool(events["is_shot"])
            & ~as_bool(events.get("is_penalty_shootout", pd.Series(False, index=events.index)))
            & ~as_bool(events.get("is_own_goal", pd.Series(False, index=events.index)))
        ].dropna(subset=["y"])
        shot_lane = pd.cut(
            pd.to_numeric(team_shots["y"], errors="coerce"),
            [-1, 100 / 3, 200 / 3, 101],
            labels=["Right", "Centre", "Left"],
        )
        shot_xg = pd.to_numeric(team_shots["xG"], errors="coerce").fillna(0)
        lane_output = {
            lane: (int((shot_lane == lane).sum()), float(shot_xg[shot_lane == lane].sum()))
            for lane in ("Left", "Centre", "Right")
        }
        summary[team_id] = (counts, shares, total)
        colour = _team_mark_color(team_id)
        span = abs(x_limit - x_tail)
        for row_y, name in zip(rows, row_names):
            share, count = by_name[name]
            # A corridor that carried nothing still gets its row, so the three
            # always read as one split. It gets a rule rather than an arrow,
            # because a zero-length chevron is a smudge.
            reach = span * (0.30 + 0.70 * (share / 100.0)) if total else 0.0
            if total and reach > 1.0:
                x_head = x_tail + direction * reach
                _corridor_arrow(ax, x_tail, x_head, row_y, 7.4, colour)
            else:
                ax.plot(
                    [x_tail, x_tail + direction * span * 0.22],
                    [row_y, row_y],
                    color=GRID,
                    lw=1.2,
                    zorder=3,
                )
            label = f"{share:.0f}%" if total else "—"
            ax.text(
                x_tail + direction * 3.0,
                row_y,
                label,
                ha="left" if direction > 0 else "right",
                va="center",
                color=text_on_fill(colour),
                fontsize=13,
                fontweight="bold",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.42", facecolor=colour, edgecolor="none"),
            )
            shots_n, lane_xg = lane_output[name]
            ax.text(
                x_tail + direction * (span * 0.30),
                row_y - 10.6,
                f"{shots_n} {'shot' if shots_n == 1 else 'shots'}  ·  {lane_xg:.2f} xG",
                ha="center",
                va="center",
                color=TEXT,
                fontsize=8.5,
                zorder=6,
            )
            ax.text(
                x_tail + direction * (span * 0.30),
                row_y + 10.4,
                f"{name.upper()}  ·  {count} {'entry' if count == 1 else 'entries'}",
                ha="center",
                va="center",
                color=MUTED,
                fontsize=8.5,
                fontweight="bold",
                zorder=6,
            )
        # Under the touchline rather than inside it: at the foot of its own
        # half the name landed on the penalty-area lines.
        ax.text(
            (x_tail + x_limit) / 2.0,
            -5.5,
            team_name.upper(),
            ha="center",
            va="center",
            color=colour,
            fontsize=11.5,
            fontweight="bold",
            zorder=6,
        )
        ax.annotate(
            "",
            xy=(x_limit - direction * 2.0, 95.0),
            xytext=(x_limit - direction * 14.0, 95.0),
            arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.4),
            zorder=6,
        )
        ax.text(
            x_limit - direction * 15.5,
            95.0,
            "ATTACKS",
            ha="right" if direction > 0 else "left",
            va="center",
            color=MUTED,
            fontsize=8,
            fontweight="bold",
            zorder=6,
        )

    home_total = summary[HOME_ID][2]
    away_total = summary[AWAY_ID][2]
    fig.text(
        0.055,
        0.045,
        f"Denominator · {HOME_NAME} {home_total} final-third "
        f"{'entry' if home_total == 1 else 'entries'}, {AWAY_NAME} {away_total}. "
        "A share of a small count moves a long way on one entry. "
        "Shots and xG under each arrow are the shots taken from that lane.",
        color=MUTED,
        fontsize=9,
    )
    return save(fig, "24b_attacking_zones.png")


def box_entries(events, team_id, number):
    frame = (
        events[box_entry_mask(events) & events["team_id"].eq(team_id)]
        .copy()
        .dropna(subset=["x", "y", "end_x", "end_y"])
    )
    fig, pitch, side = pitch_axes(
        f"Box Entries · {TEAM_NAME[team_id]}",
        "Completed actions entering the penalty area · entry method encoded by shape",
    )
    draw_long_pitch(pitch)
    reach = (
        float(frame[["x", "end_x"]].min().min()) * PITCH_LENGTH / 100 if not frame.empty else 60.0
    )
    crop_to_attack(pitch, side, min(60.0, reach - 6.0))
    team_mark = _team_mark_color(team_id)
    for _, row in frame.iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]])
        ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        event_type = str(row["type"])
        marker = "o" if event_type == "Pass" else "D"
        is_pass = marker == "o"
        entry_color = team_mark
        pitch.annotate(
            "",
            xy=(ex[0], ey[0]),
            xytext=(sx[0], sy[0]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=entry_color,
                alpha=0.72 if is_pass else 0.6,
                lw=1.5 if is_pass else 1.2,
                linestyle="-" if is_pass else FAILURE_DASH,
                mutation_scale=10,
            ),
        )
        pitch.scatter(
            ex[0], ey[0], s=58, marker=marker, color=entry_color, edgecolor=TEXT, linewidth=0.75
        )
    top = frame.groupby("player").size().sort_values(ascending=False).head(5)
    side_title(side, "ENTRY CONTRIBUTORS")
    side_rows(
        side,
        [(_surname(name), str(int(value))) for name, value in top.items()],
        start=0.81,
        gap=0.083,
    )
    pass_count = int(frame["type"].astype(str).eq("Pass").sum())
    carry_count = len(frame) - pass_count
    side.text(0.08, 0.35, "ENTRY METHOD LEGEND", color=MUTED, fontsize=7.5, fontweight="bold")
    side.scatter([0.13], [0.285], s=45, marker="o", color=team_mark, edgecolor=TEXT, linewidth=0.6)
    side.text(0.21, 0.285, f"Pass entry ({pass_count})", color=TEXT, fontsize=8, va="center")
    side.scatter([0.13], [0.22], s=45, marker="D", color=team_mark, edgecolor=TEXT, linewidth=0.6)
    side.text(0.21, 0.22, f"Carry / take-on ({carry_count})", color=TEXT, fontsize=8, va="center")
    side.annotate(
        "",
        xy=(0.18, 0.15),
        xytext=(0.08, 0.15),
        arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.1),
    )
    side.text(0.21, 0.15, "Arrow = entry path", color=TEXT, fontsize=8, va="center")
    side.text(
        0.08, 0.075, f"Total entries: {len(frame)}", color=TEXT, fontsize=9.5, fontweight="bold"
    )
    return save(fig, f"{number:02d}_box_entries_{_team_slug(team_id)}.png")


def high_regains(events, team_id, number):
    frame = high_regain_events(events, team_id).dropna(subset=["x", "y"]).copy()
    fig, pitch, side = pitch_axes(
        f"High Regains · {TEAM_NAME[team_id]}",
        "Open-play possession regains at x ≥ 60 · type encoded by shape",
    )
    draw_long_pitch(pitch)
    threshold_y = 60 * PITCH_LENGTH / 100
    crop_to_attack(pitch, side, threshold_y - 12.0)
    pitch.axhspan(threshold_y, PITCH_LENGTH, color=FOCUS, alpha=0.055)
    pitch.axhline(threshold_y, color=FOCUS, lw=1.0, ls=(0, (5, 4)))
    marker_map = {"Tackle": "o", "Interception": "D", "BallRecovery": "s"}
    team_mark = _team_mark_color(team_id)
    for event_type, marker in marker_map.items():
        subset = frame[frame["type"].astype(str).eq(event_type)]
        if subset.empty:
            continue
        px, py = attack_xy(subset["x"], subset["y"])
        pitch.scatter(
            px,
            py,
            s=130,
            marker=marker,
            color=team_mark,
            edgecolor=TEXT,
            linewidth=0.9,
            label=f"{event_type} ({len(subset)})",
        )
    other = frame[~frame["type"].astype(str).isin(marker_map)]
    if not other.empty:
        px, py = attack_xy(other["x"], other["y"])
        pitch.scatter(
            px,
            py,
            s=130,
            marker="^",
            color=team_mark,
            edgecolor=TEXT,
            linewidth=0.9,
            label=f"Other ({len(other)})",
        )
    # The minute beside each regain, so the dot can be found in the match.
    for _, row in frame.iterrows():
        rx, ry = attack_xy([row["x"]], [row["y"]])
        pitch.annotate(
            f"{int(float(row['minute']))}′",
            (rx[0], ry[0]),
            xytext=(8, 5),
            textcoords="offset points",
            color=TEXT,
            fontsize=7,
            zorder=8,
            path_effects=[path_effects.withStroke(linewidth=2.6, foreground=BG)],
        )
    pitch.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=4,
        frameon=False,
        labelcolor=TEXT,
        fontsize=8.5,
    )
    top = (
        frame.groupby("player").size().sort_values(ascending=False).head(7)
        if "player" in frame
        else pd.Series(dtype=int)
    )
    side_title(side, "HIGH-REGAIN LEADERS")
    side_rows(side, [(_surname(name), str(int(value))) for name, value in top.items()])
    side.text(
        0.08, 0.14, f"Total high regains: {len(frame)}", color=TEXT, fontsize=9, fontweight="bold"
    )
    return save(fig, f"{number:02d}_high_regains_{_team_slug(team_id)}.png")


def pass_targets(events, team_id, number):
    frame = (
        events[
            events["team_id"].eq(team_id)
            & events["type"].astype(str).eq("Pass")
            & events["outcome"].astype(str).str.lower().eq("successful")
        ]
        .dropna(subset=["end_x", "end_y"])
        .copy()
    )
    heat, _, _ = np.histogram2d(
        frame["end_y"], frame["end_x"], bins=[7, 12], range=[[0, 100], [0, 100]]
    )
    fig, pitch, side = pitch_axes(
        f"Pass Target Zones · {TEAM_NAME[team_id]}",
        "Completed-pass destinations · one sequential density scale",
    )
    team_mark = _team_mark_color(team_id)
    cmap = LinearSegmentedColormap.from_list(f"targets_{team_id}", _team_density_palette(team_id))
    # The ramp tops out at the 95th percentile of the cells that were hit, so
    # one busy corner does not turn every other cell the colour of the page.
    hit = heat[heat > 0]
    max_cell = max(float(np.percentile(hit, 95)) if hit.size else 1.0, 1.0)
    image = pitch.imshow(
        heat.T,
        extent=[-PITCH_WIDTH / 2, PITCH_WIDTH / 2, 0, PITCH_LENGTH],
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=max_cell,
        aspect="equal",
        alpha=0.95,
    )
    draw_long_pitch(pitch)
    labelled = set(np.argsort(heat.ravel(), kind="stable")[::-1][:8].tolist())
    for ix in range(7):
        for iy in range(12):
            x0 = -PITCH_WIDTH / 2 + ix * PITCH_WIDTH / 7
            y0 = iy * PITCH_LENGTH / 12
            pitch.add_patch(
                Rectangle(
                    (x0, y0),
                    PITCH_WIDTH / 7,
                    PITCH_LENGTH / 12,
                    fill=False,
                    edgecolor=GRID,
                    lw=0.42,
                    alpha=0.55,
                    zorder=3,
                )
            )
            value = int(heat[ix, iy])
            if value <= 0 or (ix * 12 + iy) not in labelled:
                continue
            # The cell fill is a step on the team-colour ramp, so the label
            # colour has to be read off that fill — a fixed white disappears
            # at the top of a light ramp (Juventus silver, Real Madrid white).
            cell_fill = mcolors.to_hex(cmap(min(value / max_cell, 1.0)))
            pitch.text(
                x0 + PITCH_WIDTH / 14,
                y0 + PITCH_LENGTH / 24,
                str(value),
                color=text_on_fill(cell_fill),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=5,
                path_effects=label_outline(cell_fill),
            )
    cbar = fig.colorbar(image, ax=pitch, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.outline.set_edgecolor(GRID)
    cbar.set_label(
        "Completed-pass targets · scale tops out at the 95th percentile", color=MUTED, fontsize=8
    )
    top = frame.groupby("player").size().sort_values(ascending=False).head(7)
    side_title(side, "TOP PASSERS")
    side_rows(side, [(_surname(name), str(int(value))) for name, value in top.items()])
    side.text(
        0.08, 0.14, f"Completed passes: {len(frame)}", color=TEXT, fontsize=9, fontweight="bold"
    )
    return save(fig, f"{number:02d}_pass_targets_{_team_slug(team_id)}.png")


def _ppda_by_state(events, info):
    """PPDA for each side inside each score state.

    One PPDA for a whole match hides the thing a coach wants. Chelsea led for
    ninety-four of the ninety-eight minutes, so their 16.41 describes a side
    protecting a lead rather than a pressing plan. Splitting it says whether a
    press was abandoned once the game turned or was never there at all.
    """
    from football_analysis.reports.match_report import calculate_ppda
    from football_analysis.metrics.match_insights import score_spells

    spells = score_spells(events, info)
    if spells is None or spells.empty:
        return {}

    minute = pd.to_numeric(events.get("minute"), errors="coerce").fillna(0)
    second = pd.to_numeric(events.get("second"), errors="coerce").fillna(0).clip(0, 59)
    clock = minute * 60 + second
    # The spells number their periods 1, 2, 3, 4; the events column spells them
    # "FirstHalf". Reading it as a number made every window empty and the panel
    # printed "not enough exposure" six times over a match with 645 passes in
    # it. PERIOD_ORDER is the same map score_spells sorted by.
    from football_analysis.metrics.match_metrics import PERIOD_ORDER, _normalise_period

    period = (
        events.get("period_code", events.get("period")).map(_normalise_period).map(PERIOD_ORDER)
    )

    out = {}
    for side, team_id, opponent_id in [
        ("home", info["home_id"], info["away_id"]),
        ("away", info["away_id"], info["home_id"]),
    ]:
        for state in ("level", "leading", "trailing"):
            windows = spells[spells.team_id.eq(team_id) & spells.state.eq(state)]
            if windows.empty:
                continue
            mask = pd.Series(False, index=events.index)
            for row in windows.itertuples():
                mask |= period.eq(row.period) & clock.between(
                    row.start * 60, row.end * 60, inclusive="left"
                )
            if not mask.any():
                continue
            result = calculate_ppda(events[mask], team_id, opponent_id)
            value = result.get("ppda")
            # A handful of opponent passes gives a ratio that swings on one
            # tackle. Thirty is the floor at which it starts describing a
            # press rather than a sample.
            if value and result.get("passes_allowed", 0) >= 30:
                out[(side, state)] = (float(value), float(windows.minutes.sum()))
    return out


def ppda(events):
    """How hard each side pressed, on one scale, and how the score changed it.

    This was two semicircular gauges. A dial is the wrong mark for a
    comparison: the reader has to turn two needle angles into two numbers and
    then difference them, and the ink spent on bands, ticks and hubs was larger
    than the two figures it carried.

    What replaced the dials had its own faults, and this is the second pass.
    The board carried two charts on two axes even though both plotted PPDA, so
    a length in one could not be carried to the other. The axis ran to 21 while
    the data stopped at 12, spending two fifths of the plot on nothing. The
    value sat in a bubble at the end of its own bar and the bubble overlapped
    the bar, so the figure was struck through by the line that produced it. The
    zone names were column headers over empty plot rather than ground behind
    the data, and a score state with too little exposure to rate still spent a
    labelled row saying so.

    One axis for everything now, ending just past the largest value; the zones
    shaded behind the bars; the figure outside the end of its own bar; and a
    state with nothing to report says so in one grey line instead of a row.
    """
    from football_analysis.reports.match_report import compute_ppda_both

    info = {"home_id": HOME_ID, "away_id": AWAY_ID}
    data = compute_ppda_both(info, events)
    hp = float(data["home"]["ppda"] or 0)
    ap = float(data["away"]["ppda"] or 0)

    by_state = _ppda_by_state(events, info)
    rows = [("Full match", hp or None, 0.0, ap or None, 0.0)]
    for state in ("level", "leading", "trailing"):
        home = by_state.get(("home", state))
        away = by_state.get(("away", state))
        if not home and not away:
            continue
        rows.append(
            (
                state.capitalize(),
                home[0] if home else None,
                home[1] if home else 0.0,
                away[0] if away else None,
                away[1] if away else 0.0,
            )
        )

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.text(0.055, 0.95, "Pressing Intensity", fontsize=23, fontweight="bold", color=TEXT)
    gap = abs(hp - ap)
    lead = HOME_NAME if hp < ap else AWAY_NAME
    fig.text(
        0.055,
        0.905,
        f"{lead} pressed harder over the match, by {gap:.1f} PPDA"
        if gap >= 0.2 and hp and ap
        else "Neither side pressed measurably harder over the match",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )

    every = [v for row in rows for v in (row[1], row[3]) if v]
    ceiling = (max(every) if every else 20.0) * 1.26

    ax = fig.add_axes([0.135, 0.225, 0.60, 0.585])
    base.clean_ax(ax)
    ax.set_xlim(0, ceiling)
    ax.set_ylim(-0.6, len(rows) - 0.22)
    ax.set_yticks([])

    # The zones as ground: they interpret the number, so they sit behind it. As
    # headers they were labelling stretches of empty plot.
    for index, (low, high, label) in enumerate(
        [
            (0, 8, "Elite press"),
            (8, 11, "High press"),
            (11, 14, "Mid block"),
            (14, ceiling, "Low block"),
        ]
    ):
        if low >= ceiling:
            continue
        high = min(high, ceiling)
        ax.axvspan(low, high, color=PANEL_2 if index % 2 else PANEL, alpha=0.9, lw=0, zorder=0)
        ax.text(
            (low + high) / 2,
            len(rows) - 0.33,
            label.upper(),
            color=MUTED,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=2,
        )

    height = 0.30
    for row, (label, home, home_min, away, away_min) in enumerate(rows):
        y = len(rows) - 1 - row
        for value, minutes, colour, name, offset in (
            (home, home_min, HOME, HOME_NAME, +height / 1.85),
            (away, away_min, AWAY, AWAY_NAME, -height / 1.85),
        ):
            if not value:
                ax.text(
                    ceiling * 0.012,
                    y + offset,
                    f"{name} — under 30 opponent passes in this state",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    color=NEUTRAL,
                    style="italic",
                    zorder=3,
                )
                continue
            ax.barh(y + offset, value, height=height, color=colour, zorder=3)
            # Outside the end of its own bar. In a bubble at the end it was
            # overlapped by the bar that produced it.
            ax.text(
                value + ceiling * 0.014,
                y + offset,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=11.5,
                fontweight="bold",
                color=TEXT,
                zorder=4,
            )
            ax.text(
                value + ceiling * 0.075,
                y + offset,
                f"{name} · {minutes:.0f} min" if minutes else name,
                va="center",
                ha="left",
                fontsize=8.5,
                color=MUTED,
                zorder=4,
            )
        ax.text(
            -ceiling * 0.014,
            y,
            label,
            va="center",
            ha="right",
            fontsize=10.5,
            fontweight="bold" if row == 0 else "normal",
            color=TEXT,
        )
        if row == 0 and len(rows) > 1:
            ax.axhline(y - 0.5, color=GRID, lw=0.8, zorder=2)

    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
    ax.set_xlabel(
        "PPDA  —  opponent passes allowed per defensive action, lower is a harder press",
        color=MUTED,
        fontsize=9,
        labelpad=10,
    )

    # The denominators, beside the plot. Floating inside it they read as data.
    fig.text(0.775, 0.755, "WHAT EACH RATIO DIVIDES", color=MUTED, fontsize=7.5, fontweight="bold")
    top = 0.715
    for name, colour, side in ((HOME_NAME, HOME, "home"), (AWAY_NAME, AWAY, "away")):
        entry = data[side]
        fig.text(0.775, top, name.upper(), color=colour, fontsize=10, fontweight="bold")
        for step, (figure, caption) in enumerate(
            (
                (entry.get("passes_allowed", 0), "opponent\npasses"),
                (entry.get("defensive_actions", 0), "defensive\nactions"),
            )
        ):
            row_y = top - 0.042 - step * 0.054
            fig.text(
                0.775,
                row_y,
                f"{float(figure or 0):.0f}",
                color=TEXT,
                fontsize=15,
                fontweight="bold",
            )
            fig.text(0.832, row_y + 0.006, caption, color=MUTED, fontsize=7.5, linespacing=1.3)
        top -= 0.170

    fig.text(
        0.135,
        0.105,
        "Opponent passes ÷ tackles + interceptions + fouls + challenges + recoveries, "
        "in the opponent 60% of the pitch. A state under 30 opponent passes is left "
        "unrated: one tackle would move it.",
        color=NEUTRAL,
        fontsize=8.5,
    )
    return save(fig, "31_ppda_pressing.png")


def press_profile(events, xg=None, team_metrics=None):
    """The whole press as one shape, beside the board that measures its rate.

    PPDA is one number, and the board before this one plots it four times --
    the match and three score states. That is one measure under four
    conditions, so a radar of it encloses an area meaning nothing, and because
    a lower PPDA is a harder press the smaller shape would be the better side.

    Six different questions do make a shape worth reading: how early the ball
    is contested, how often it is won in the last third, how often that becomes
    a shot, how often it is won straight back, how far up the side plays at
    all, and how much of the risk it takes is punished. The two that are better
    when low are inverted, so on every axis further out is better -- the one
    rule a radar has to obey to be read at a glance.

    Every axis is scaled between a floor and a ceiling fixed here rather than
    against the two sides on the day: a radar normalised to its own match makes
    the better side touch the rim whatever it did, and two matches cannot then
    be compared.
    """
    from football_analysis.reports.match_report import compute_ppda_both

    info = {"home_id": HOME_ID, "away_id": AWAY_ID}
    ppda_data = compute_ppda_both(info, events)

    def metric(side, key, default=0.0):
        frame = team_metrics
        if frame is None or key not in getattr(frame, "columns", []):
            return default
        team_id = HOME_ID if side == "home" else AWAY_ID
        rows = frame[frame.get("team_id").eq(team_id)] if "team_id" in frame else frame
        if rows.empty:
            return default
        try:
            return float(pd.to_numeric(rows[key], errors="coerce").iloc[0])
        except Exception:
            return default

    def counted(team_id, kinds, high_only=False):
        own = events[events["team_id"].eq(team_id)]
        if "type" in own:
            own = own[own["type"].isin(kinds)]
        if high_only:
            x = pd.to_numeric(own.get("x"), errors="coerce")
            own = own[x >= 66.7]
        return int(len(own))

    axes = []
    for label, floor, ceiling, lower_better, digits, home, away in (
        (
            "Press\nintensity",
            6.0,
            20.0,
            True,
            1,
            ppda_data["home"]["ppda"],
            ppda_data["away"]["ppda"],
        ),
        # high_regains is the pipeline own count, used by every other
        # board. Counting final-third recoveries again here produced a
        # second, quieter definition of the same thing.
        (
            "High\nrecoveries",
            0.0,
            22.0,
            False,
            0,
            metric("home", "high_regains"),
            metric("away", "high_regains"),
        ),
        (
            "Regain\nconversion",
            0.0,
            20.0,
            False,
            1,
            metric("home", "regain_to_shot_rate"),
            metric("away", "regain_to_shot_rate"),
        ),
        (
            "Counter\npress",
            0.0,
            25.0,
            False,
            1,
            metric("home", "counterpress_success_rate"),
            metric("away", "counterpress_success_rate"),
        ),
        (
            "Territory",
            0.0,
            100.0,
            False,
            1,
            metric("home", "field_tilt"),
            metric("away", "field_tilt"),
        ),
        (
            "Rest\ndefence",
            0.0,
            30.0,
            True,
            1,
            _punished_share(
                metric("home", "rest_defence_dangerous_counters"),
                metric("home", "rest_defence_exposures"),
            ),
            _punished_share(
                metric("away", "rest_defence_dangerous_counters"),
                metric("away", "rest_defence_exposures"),
            ),
        ),
    ):
        axes.append(
            (label, float(home or 0), float(away or 0), floor, ceiling, lower_better, digits)
        )

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.text(0.055, 0.95, "Pressing Profile", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(
        0.055,
        0.905,
        "Six questions about the same press · further from the centre is better on every axis",
        fontsize=11,
        color=MUTED,
    )

    ax = fig.add_axes([0.345, 0.245, 0.295, 0.455], projection="polar")
    ax.set_facecolor(BG)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(axes), endpoint=False)

    for ring in (0.25, 0.5, 0.75, 1.0):
        ax.plot(np.linspace(0, 2 * np.pi, 200), [ring] * 200, color=GRID, lw=0.7, zorder=1)
    for angle in angles:
        ax.plot([angle, angle], [0, 1], color=GRID, lw=0.7, zorder=1)

    for colour, index in ((HOME, 1), (AWAY, 2)):
        values = []
        for row in axes:
            share = (row[index] - row[3]) / (row[4] - row[3]) if row[4] != row[3] else 0.0
            share = min(max(share, 0.0), 1.0)
            values.append(1.0 - share if row[5] else share)
        loop = list(angles) + [angles[0]]
        ax.plot(loop, values + values[:1], color=colour, lw=2.4, zorder=4)
        ax.fill(loop, values + values[:1], color=colour, alpha=0.16, zorder=3)
        ax.scatter(angles, values, color=colour, s=34, zorder=5, edgecolors=BG, linewidths=1.2)

    for angle, row in zip(angles, axes):
        ax.text(
            angle,
            1.19,
            row[0].upper(),
            ha="center",
            va="center",
            color=TEXT,
            fontsize=8,
            fontweight="bold",
            linespacing=1.25,
        )
        # One figure per line, each in its own side's colour. Side by side at an
        # angular offset they overlapped at the top and bottom of the ring,
        # where a small rotation is almost no horizontal distance.
        for radius, value, colour in ((1.36, row[1], HOME), (1.50, row[2], AWAY)):
            ax.text(
                angle,
                radius,
                f"{value:.{row[6]}f}",
                ha="center",
                va="center",
                color=colour,
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    top = 0.72
    for name, colour, side in ((HOME_NAME, HOME, "home"), (AWAY_NAME, AWAY, "away")):
        fig.text(0.055, top, name.upper(), color=colour, fontsize=11, fontweight="bold")
        # A club name on its own is a key, not a reading. Two figures under it
        # say what its shape is made of.
        fig.text(
            0.055,
            top - 0.052,
            f"{metric(side, 'field_tilt'):.0f}% of the territory"
            + chr(10)
            + f"{metric(side, 'high_regains'):.0f} high recoveries",
            color=MUTED,
            fontsize=8.5,
            linespacing=1.5,
        )
        top -= 0.14

    fig.text(0.055, 0.46, "HOW TO READ IT", color=MUTED, fontsize=7.5, fontweight="bold")
    fig.text(
        0.055,
        0.30,
        "Press intensity is PPDA inverted and rest\n"
        "defence is the share of advanced losses\n"
        "punished, also inverted, so that further\n"
        "out is better on all six.\n\n"
        "The two figures on each axis are the raw\n"
        "values, each in its own side's colour.",
        color=MUTED,
        fontsize=8.5,
        linespacing=1.6,
    )

    fig.text(
        0.135,
        0.062,
        "A shape that reaches on intensity and recoveries but not on conversion is a side "
        "winning the ball back without threatening from it. Axis floors and ceilings are "
        "fixed, not scaled to this fixture, so two matches can be compared.",
        color=NEUTRAL,
        fontsize=8.5,
    )
    return save(fig, "31b_press_profile.png")


def _punished_share(dangerous, exposures):
    """The share of a side's advanced losses that became a dangerous counter."""
    try:
        exposures = float(exposures or 0)
        return 100.0 * float(dangerous or 0) / exposures if exposures else 0.0
    except (TypeError, ValueError):
        return 0.0


def transition_outcomes(events):
    annotated, possessions = build_possessions(events)
    team_data = {}
    for team_id in [HOME_ID, AWAY_ID]:
        transitions = possessions[
            possessions["team_id"].eq(team_id) & possessions["is_transition"].astype(bool)
        ].copy()
        total = len(transitions)
        chances = (
            int(
                (
                    pd.to_numeric(transitions["transition_shots"], errors="coerce").fillna(0) > 0
                ).sum()
            )
            if total
            else 0
        )
        shots = (
            int(pd.to_numeric(transitions["transition_shots"], errors="coerce").fillna(0).sum())
            if total
            else 0
        )
        goals = (
            int(pd.to_numeric(transitions["transition_goals"], errors="coerce").fillna(0).sum())
            if total
            else 0
        )
        paths = []
        for transition in transitions.itertuples():
            window = annotated[
                annotated["possession_id"].eq(int(transition.possession_id))
                & annotated["team_id"].eq(team_id)
                & (
                    pd.to_numeric(annotated["_clock_seconds"], errors="coerce")
                    <= float(transition.start_time) + 12.0
                )
            ].copy()
            points = [(float(transition.start_x), float(transition.start_y))]
            for _, row in window.iterrows():
                for x_value, y_value in [
                    (row.get("_x", np.nan), row.get("_y", np.nan)),
                    (row.get("_end_x", np.nan), row.get("_end_y", np.nan)),
                ]:
                    if pd.notna(x_value) and pd.notna(y_value):
                        points.append((float(x_value), float(y_value)))
            end_x, end_y = max(points, key=lambda point: point[0])
            paths.append(
                {
                    "start_x": float(transition.start_x),
                    "start_y": float(transition.start_y),
                    "end_x": end_x,
                    "end_y": end_y,
                    "shot": int(transition.transition_shots) > 0,
                    "goal": int(transition.transition_goals) > 0,
                    "minute": int(float(transition.start_time) // 60),
                }
            )
        team_data[team_id] = {
            "total": total,
            "chances": chances,
            "shots": shots,
            "goals": goals,
            "xg": float(
                pd.to_numeric(transitions["transition_xG"], errors="coerce").fillna(0).sum()
            )
            if total
            else 0.0,
            "box_entries": int(
                pd.to_numeric(transitions["transition_box_entries"], errors="coerce")
                .fillna(0)
                .sum()
            )
            if total
            else 0,
            "paths": paths,
        }

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.text(0.055, 0.95, "Transition Map & Outcomes", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(
        0.055,
        0.908,
        "Each line runs from the regain/turnover location to the most advanced point reached inside 12 seconds",
        fontsize=10.5,
        color=MUTED,
    )
    fig.add_artist(
        Line2D([0.055, 0.945], [0.872, 0.872], transform=fig.transFigure, color=GRID, lw=1)
    )

    layouts = [
        (HOME_ID, [0.055, 0.145, 0.265, 0.675], [0.335, 0.215, 0.135, 0.515], 0.19),
        (AWAY_ID, [0.525, 0.145, 0.265, 0.675], [0.805, 0.215, 0.135, 0.515], 0.66),
    ]
    for team_id, pitch_rect, card_rect, title_x in layouts:
        data = team_data[team_id]
        fig.text(
            title_x,
            0.835,
            TEAM_NAME[team_id],
            color=TEXT,
            fontsize=14,
            fontweight="bold",
            ha="center",
        )
        pitch = fig.add_axes(pitch_rect)
        pitch.axhspan(66.7 * PITCH_LENGTH / 100, PITCH_LENGTH, color=FOCUS, alpha=0.035, zorder=0)
        draw_long_pitch(pitch)
        ordered_paths = sorted(data["paths"], key=lambda item: (item["goal"], item["shot"]))
        for path in ordered_paths:
            sx, sy = player_position_xy([path["start_x"]], [path["start_y"]])
            ex, ey = player_position_xy([path["end_x"]], [path["end_y"]])
            if path["goal"]:
                color, alpha, width, marker, size, line_style = (
                    EVENT_HIGHLIGHT,
                    0.95,
                    2.25,
                    "*",
                    75,
                    "-",
                )
            elif path["shot"]:
                color, alpha, width, marker, size, line_style = (
                    EVENT_SUCCESS,
                    0.82,
                    1.35,
                    "D",
                    25,
                    "-",
                )
            else:
                color, alpha, width, marker, size, line_style = (
                    EVENT_NEUTRAL,
                    0.24,
                    0.62,
                    None,
                    0,
                    QUIET_DASH,
                )
            pitch.annotate(
                "",
                xy=(ex[0], ey[0]),
                xytext=(sx[0], sy[0]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    alpha=alpha,
                    lw=width,
                    linestyle=line_style,
                    mutation_scale=7 if not path["goal"] else 10,
                ),
                zorder=3,
            )
            pitch.scatter(
                sx[0],
                sy[0],
                s=8,
                facecolors=BG,
                edgecolors=color,
                linewidth=0.55,
                alpha=max(alpha, 0.35),
                zorder=4,
            )
            if marker:
                pitch.scatter(
                    ex[0],
                    ey[0],
                    s=size,
                    marker=marker,
                    color=color,
                    edgecolor=TEXT,
                    linewidth=0.55,
                    zorder=5,
                )
            if path["goal"]:
                pitch.text(
                    ex[0],
                    ey[0] + 2.0,
                    f"{path['minute']}′",
                    color=EVENT_HIGHLIGHT,
                    fontsize=6.5,
                    fontweight="bold",
                    ha="center",
                    zorder=6,
                )

        card = fig.add_axes(card_rect)
        card.set_facecolor(PANEL)
        card.set_xlim(0, 1)
        card.set_ylim(0, 1)
        card.set_xticks([])
        card.set_yticks([])
        for spine in card.spines.values():
            spine.set_color(GRID)
        side_title(card, "OUTCOMES")
        side_kpis(
            card,
            [
                ("Transitions", data["total"]),
                ("Created chance", data["chances"]),
                ("Goals", data["goals"]),
                ("Transition xG", f"{data['xg']:.2f}"),
                ("Box entries", data["box_entries"]),
            ],
            start=0.82,
            gap=0.135,
        )
        chance_rate = 100 * data["chances"] / max(data["total"], 1)
        card.text(
            0.08,
            0.065,
            f"Chance rate: {chance_rate:.1f}%",
            color=TEXT,
            fontsize=7.5,
            fontweight="bold",
        )

    legend = [
        Line2D(
            [0],
            [0],
            color=EVENT_NEUTRAL,
            lw=1.35,
            alpha=0.55,
            linestyle=QUIET_DASH,
            label="Transition without shot",
        ),
        Line2D(
            [0], [0], color=EVENT_SUCCESS, lw=2, marker="D", markersize=5, label="Created a chance"
        ),
        Line2D(
            [0],
            [0],
            color=EVENT_HIGHLIGHT,
            lw=2.5,
            marker="*",
            markersize=8,
            label="Ended in a goal",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.067),
        ncol=3,
        frameon=False,
        labelcolor=TEXT,
        fontsize=8,
    )
    fig.text(
        0.055,
        0.032,
        "DEFINITION  Open-play regain/turnover that within 12s progresses ≥20m, reaches the final third/box, or produces a shot · restarts excluded.",
        color=NEUTRAL,
        fontsize=7.5,
    )
    fig.text(
        0.945, 0.032, "ARROW = START → MOST ADVANCED POINT", ha="right", fontsize=7.5, color=NEUTRAL
    )
    return save(fig, "32_transition_outcomes.png")


def _game_state_durations(events):
    period = events.get("period_code", pd.Series("", index=events.index)).astype(str).str.lower()
    live = events[
        ~period.isin(["pre", "prematch", "post", "postgame", "pso", "penaltyshootout"])
    ].copy()
    live["_clock"] = pd.to_numeric(live["minute"], errors="coerce").fillna(0) * 60 + pd.to_numeric(
        live["second"], errors="coerce"
    ).fillna(0)
    shootout = as_bool(live.get("is_penalty_shootout", pd.Series(False, index=live.index)))
    goals = (
        live[as_bool(live.get("is_goal", pd.Series(False, index=live.index))) & ~shootout]
        .sort_values(["_clock", "event_id"], kind="stable")
        .copy()
    )
    # Game state is driven by the scoreline, so own goals must be credited to
    # the side that benefits from them.
    goals["_credited_team"] = (
        base.credited_team(goals) if not goals.empty else pd.Series(dtype=float)
    )
    end_time = float(live["_clock"].max()) if not live.empty else 0.0
    durations = {"drawing": 0.0, "home_ahead": 0.0, "away_ahead": 0.0}
    home_score = away_score = 0
    previous = 0.0

    def current_state():
        if home_score == away_score:
            return "drawing"
        return "home_ahead" if home_score > away_score else "away_ahead"

    for _, goal in goals.iterrows():
        clock = float(goal["_clock"])
        durations[current_state()] += max(clock - previous, 0.0)
        credited = int(float(goal.get("_credited_team", 0) or 0))
        if credited == HOME_ID:
            home_score += 1
        elif credited == AWAY_ID:
            away_score += 1
        previous = clock
    durations[current_state()] += max(end_time - previous, 0.0)
    return durations, end_time


def game_state(events, team_metrics):
    durations, match_seconds = _game_state_durations(events)
    scenarios = [
        {
            "title": "SCORE LEVEL",
            "subtitle": f"{HOME_NAME} level · {AWAY_NAME} level",
            "duration_key": "drawing",
            "home_state": "drawing",
            "away_state": "drawing",
            "color": NEUTRAL,
        },
        {
            "title": f"{HOME_NAME.upper()} AHEAD",
            "subtitle": f"{HOME_NAME} leading · {AWAY_NAME} trailing",
            "duration_key": "home_ahead",
            "home_state": "leading",
            "away_state": "trailing",
            "color": HOME,
        },
        {
            "title": f"{AWAY_NAME.upper()} AHEAD",
            "subtitle": f"{HOME_NAME} trailing · {AWAY_NAME} leading",
            "duration_key": "away_ahead",
            "home_state": "trailing",
            "away_state": "leading",
            "color": AWAY,
        },
    ]
    specs = [
        ("Shots", "shots", "{:.0f}"),
        ("xG", "xG", "{:.2f}"),
        ("Transitions", "transitions", "{:.0f}"),
        ("Box entries", "box_entries", "{:.0f}"),
    ]
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.text(
        0.055,
        0.95,
        "Game-State Output by Match Situation",
        fontsize=23,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.055,
        0.908,
        "Each card is one shared scoreboard situation · values are totals from possessions starting in that situation",
        fontsize=10.5,
        color=MUTED,
    )
    fig.add_artist(
        Line2D([0.055, 0.945], [0.872, 0.872], transform=fig.transFigure, color=GRID, lw=1)
    )

    timeline = fig.add_axes([0.075, 0.775, 0.85, 0.065])
    timeline.set_xlim(0, max(match_seconds, 1))
    timeline.set_ylim(0, 1)
    timeline.axis("off")
    left = 0.0
    for scenario in scenarios:
        duration = durations[scenario["duration_key"]]
        if duration > 0:
            timeline.barh(
                0.56,
                duration,
                left=left,
                height=0.36,
                color=scenario["color"],
                alpha=0.82,
                edgecolor=BG,
                linewidth=1.0,
            )
            if duration / max(match_seconds, 1) > 0.10:
                timeline.text(
                    left + duration / 2,
                    0.56,
                    scenario["title"],
                    color=TEXT,
                    fontsize=7.5,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )
        left += duration
    timeline.text(0, 0.03, "MATCH TIME SHARE", color=MUTED, fontsize=6.8, ha="left")
    timeline.text(
        max(match_seconds, 1),
        0.03,
        f"TOTAL · {match_seconds / 60:.1f} min",
        color=MUTED,
        fontsize=6.8,
        ha="right",
    )
    legend_x = [0.15, 0.42, 0.70]
    for x, scenario in zip(legend_x, scenarios):
        minutes = durations[scenario["duration_key"]] / 60
        fig.add_artist(
            Rectangle(
                (x, 0.735),
                0.014,
                0.014,
                transform=fig.transFigure,
                facecolor=scenario["color"],
                edgecolor=TEXT,
                lw=0.45,
                alpha=0.9,
            )
        )
        fig.text(
            x + 0.020,
            0.742,
            f"{scenario['title'].title()} · {minutes:.1f} min",
            color=TEXT,
            fontsize=8,
            va="center",
        )

    active_scenarios = [
        scenario for scenario in scenarios if durations[scenario["duration_key"]] >= 3.0
    ]
    inactive_scenarios = [
        scenario for scenario in scenarios if durations[scenario["duration_key"]] < 3.0
    ]
    if len(active_scenarios) == 3:
        card_positions = [
            [0.055, 0.16, 0.285, 0.535],
            [0.3575, 0.16, 0.285, 0.535],
            [0.66, 0.16, 0.285, 0.535],
        ]
    elif len(active_scenarios) == 2:
        card_positions = [[0.06, 0.16, 0.415, 0.535], [0.525, 0.16, 0.415, 0.535]]
    else:
        card_positions = [[0.20, 0.16, 0.60, 0.535]]
    if inactive_scenarios:
        inactive_labels = {
            f"{HOME_NAME.upper()} AHEAD": f"{HOME_NAME} never led",
            f"{AWAY_NAME.upper()} AHEAD": f"{AWAY_NAME} never led",
            "SCORE LEVEL": "The score was never level",
        }
        note = " · ".join(inactive_labels[scenario["title"]] for scenario in inactive_scenarios)
        fig.text(
            0.50,
            0.705,
            f"NOT PLAYED · {note}",
            color=FOCUS,
            fontsize=8.2,
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.38", fc=PANEL, ec=GRID, lw=0.7),
        )
    for rect, scenario in zip(card_positions, active_scenarios):
        ax = fig.add_axes(rect)
        ax.set_facecolor(PANEL)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(scenario["color"])
            spine.set_linewidth(1.2)
        duration_minutes = durations[scenario["duration_key"]] / 60
        ax.text(0.06, 0.93, scenario["title"], color=TEXT, fontsize=12, fontweight="bold", va="top")
        ax.text(0.06, 0.865, scenario["subtitle"], color=MUTED, fontsize=7.4, va="top")
        ax.text(
            0.94,
            0.93,
            f"{duration_minutes:.1f} min",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="top",
        )
        ax.plot([0.06, 0.94], [0.81, 0.81], color=GRID, lw=0.8)
        ax.text(
            0.57, 0.755, HOME_NAME.upper(), color=HOME, fontsize=7.5, fontweight="bold", ha="center"
        )
        ax.text(
            0.84, 0.755, AWAY_NAME.upper(), color=AWAY, fontsize=7.5, fontweight="bold", ha="center"
        )
        # The rate belongs on this card, not on a page of its own. It used to
        # live in a second chart that repeated these same three states with a
        # different denominator, so a reader comparing "Chelsea ahead" totals
        # with "Chelsea ahead" rates had to hold two pages open. Twelve minutes
        # level and ninety-four ahead cannot be read from totals alone.
        for idx, (label, key, fmt) in enumerate(specs):
            y = 0.66 - idx * 0.115
            home_value = base.metric_lookup(
                team_metrics, "home", f"game_state_{scenario['home_state']}_{key}"
            )
            away_value = base.metric_lookup(
                team_metrics, "away", f"game_state_{scenario['away_state']}_{key}"
            )
            ax.text(0.07, y, label, color=TEXT, fontsize=8.5, va="center")
            ax.text(
                0.57,
                y,
                fmt.format(home_value),
                color=TEXT,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
            )
            ax.text(
                0.84,
                y,
                fmt.format(away_value),
                color=TEXT,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
            )
            ax.plot([0.07, 0.93], [y - 0.055, y - 0.055], color=GRID, lw=0.55, alpha=0.75)
            if key == "xG":
                home_xg, away_xg = home_value, away_value
        rate_y = 0.66 - len(specs) * 0.115
        ax.text(0.07, rate_y, "xG / 30 min", color=TEXT, fontsize=8.5, va="center", style="italic")
        for x, value in [(0.57, home_xg), (0.84, away_xg)]:
            # Below five minutes the ratio says more about the clock than the
            # football, which is the rule the separate chart already used.
            text = f"{value * 30 / duration_minutes:.2f}" if duration_minutes >= 5 else "—"
            ax.text(
                x,
                rate_y,
                text,
                color=MUTED,
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
            )
        ax.text(
            0.07,
            0.075,
            "Rows above are totals; the italic row is the rate over this state's minutes.",
            color=NEUTRAL,
            fontsize=6.7,
        )

    fig.text(
        0.055,
        0.095,
        f"HOW TO READ  {HOME_NAME} ahead pairs {HOME_NAME}'s leading output with {AWAY_NAME}'s trailing output; {AWAY_NAME} ahead does the reverse.",
        color=MUTED,
        fontsize=8.2,
    )
    fig.text(
        0.055,
        0.066,
        "Game state is assigned at possession start · Timeline duration is reconstructed from goal times.",
        color=NEUTRAL,
        fontsize=7.5,
    )
    fig.text(
        0.945,
        0.035,
        f"{HOME_NAME.upper()} · {AWAY_NAME.upper()} · REAL MATCH DATA",
        ha="right",
        fontsize=7.5,
        color=NEUTRAL,
    )
    return save(fig, "33_game_state_splits.png")


def player_sequence(player_metrics):
    metrics = [("xGChain", "xGChain"), ("xGBuildup", "xGBuildup"), ("Sequence xT", "sequence_xT")]
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.text(
        0.055, 0.95, "Player Sequence Contribution", fontsize=23, fontweight="bold", color=TEXT
    )
    fig.text(
        0.055,
        0.91,
        "Top players by xGChain, xGBuildup and sequence xT involvement",
        fontsize=11,
        color=MUTED,
    )
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=0.075, right=0.96, top=0.82, bottom=0.10, wspace=0.42)
    for ax, (title, column) in zip(axes, metrics):
        top = player_metrics.sort_values(column, ascending=False).head(5).sort_values(column)
        colors = [HOME if str(team).lower() == HOME_NAME.lower() else AWAY for team in top["team"]]
        ax.barh(top["player"].astype(str).str.split().str[-1], top[column], color=colors, alpha=0.9)
        base.clean_ax(ax)
        ax.grid(axis="x", color=GRID, lw=0.65)
        ax.set_title(title.upper(), loc="left", color=MUTED, fontsize=10, fontweight="bold")
        # Room for the value that sits past the end of the bar, or the
        # longest one prints outside the axes and over its neighbour.
        widest = max(float(top[column].max()), 0.01)
        ax.set_xlim(0, widest * 1.22)
        for idx, value in enumerate(top[column]):
            ax.text(
                value + widest * 0.025, idx, f"{value:.2f}", color=TEXT, va="center", fontsize=8
            )
    fig.text(
        0.945,
        0.035,
        f"{HOME_NAME.upper()} · {AWAY_NAME.upper()}",
        ha="right",
        fontsize=8,
        color=NEUTRAL,
    )
    return save(fig, "34_player_sequence_leaders.png")


def _momentum_rolling(events, column, mask, window=8.0, step=0.25):
    """A triangular-weighted rolling total per side of whatever ``column`` measures.

    The board this replaced binned into fixed five-minute blocks. Two shots
    forty seconds apart landed in different bars whenever a block boundary fell
    between them, and every event inside a block counted the same wherever it
    sat, which is what made the page a row of plateaus rather than a shape.

    The window slides instead, and the weight falls off towards its edges, so
    every point is the same length of football measured from where it actually
    happened.
    """
    frame = events[mask].copy()
    frame["minute"] = pd.to_numeric(frame.get("minute"), errors="coerce")
    frame[column] = pd.to_numeric(frame.get(column), errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["minute"])
    last = float(pd.to_numeric(events.get("minute"), errors="coerce").max() or 90.0)
    grid = np.arange(0.0, last + step, step)
    series = {}
    for team_id in (HOME_ID, AWAY_ID):
        side = frame[frame["team_id"].eq(team_id)]
        totals = np.zeros_like(grid)
        for minute, value in zip(side["minute"].to_numpy(), side[column].to_numpy()):
            totals += np.clip(1.0 - np.abs(grid - minute) / (window / 2.0), 0, None) * float(value)
        series[team_id] = totals
    return grid, series[HOME_ID], series[AWAY_ID], last


def _momentum_spells(grid, signal, last, floor=8.0):
    """The passages the match divided into, as ``[start, end, sign]``.

    Two mistakes are worth recording, because both produced something that
    looked plausible.

    Folding every run shorter than ``floor`` into the one before it, in a
    single left-to-right pass, collapses the whole match: the run after the one
    just absorbed now matches its neighbour's sign, so it merges too, and the
    cascade reaches the end. The output was one block covering the fixture. The
    shortest run is absorbed first here, into whichever neighbour is longer,
    and same-sign neighbours are coalesced after each step.

    Segmenting on the sign of the lead is the other one. One side's threat sat
    above the other's for ninety-eight per cent of the fixture this was built
    on, which is true and tells a reader nothing. The caller passes the share
    measured against its own match average instead, so the boundaries fall
    where the balance turned rather than where the lead did.
    """
    sign = np.sign(np.where(np.abs(signal) < 1e-4, 0, signal))
    runs, start, current = [], 0, sign[0]
    for i in range(1, len(sign)):
        if sign[i] != current:
            runs.append([grid[start], grid[i], current])
            start, current = i, sign[i]
    runs.append([grid[start], last, current])

    def coalesce(items):
        out = []
        for run in items:
            if out and out[-1][2] == run[2]:
                out[-1][1] = run[1]
            else:
                out.append(list(run))
        return out

    runs = coalesce(runs)
    while len(runs) > 1:
        lengths = [hi - lo for lo, hi, _ in runs]
        shortest = int(np.argmin(lengths))
        if lengths[shortest] >= floor:
            break
        before = lengths[shortest - 1] if shortest > 0 else -1.0
        after = lengths[shortest + 1] if shortest + 1 < len(runs) else -1.0
        into = shortest - 1 if before >= after else shortest + 1
        runs[into][0] = min(runs[into][0], runs[shortest][0])
        runs[into][1] = max(runs[into][1], runs[shortest][1])
        runs.pop(shortest)
        runs = coalesce(runs)
    return runs


def momentum(events):
    """Who was on top, minute to minute, and the spells the match divided into.

    Three changes from the board this replaces, and the last two matter more
    than the first.

    The window slides rather than binning, so the page is a shape rather than a
    row of blocks whose edges were decided by the clock.

    It reads threat rather than shots. A side can pin the other in its own box
    for ten minutes and not get a shot away; the old page drew that spell as an
    empty bar, because a shot was the only thing it counted.

    Each side keeps its own area rather than being differenced away. Ten
    minutes in which nothing happened and ten in which both sides traded
    chances both come out near zero once you subtract one from the other, and
    the old page drew them identically.

    The band underneath names the passages and prices them in chances. Each
    block is split by the share of threat inside it, so a fixture one side
    controlled throughout does not print the same sentence four times, and a
    close match -- which a solid fill could not tell from a rout -- reads as
    slivers either side of the middle.
    """
    threat = pd.to_numeric(events.get("xT"), errors="coerce").fillna(0.0)
    grid, home, away, last = _momentum_rolling(events, "xT", threat > 0)
    if not len(grid) or float(np.max(home) + np.max(away)) <= 0:
        fig, ax = base.page("Match Momentum", "No recorded movement to measure")
        base.clean_ax(ax)
        ax.text(0.5, 0.5, "No recorded movement", color=MUTED, ha="center", va="center")
        return save(fig, "35_match_momentum.png")

    # The spells come off a wider window than the area above them: the area is
    # meant to move with play, and a lead that flips every ninety seconds is
    # not a passage of play.
    _, slow_home, slow_away, _ = _momentum_rolling(events, "xT", threat > 0, window=16.0)
    pair = slow_home + slow_away
    share = np.divide(slow_home, pair, out=np.full_like(slow_home, 0.5), where=pair > 0)
    runs = _momentum_spells(grid, share - float(share.mean()), last)

    is_shot = events.get("is_shot", pd.Series(False, index=events.index))
    shots = events[is_shot.astype(str).str.lower().isin(("true", "1"))].copy()
    shots["minute"] = pd.to_numeric(shots.get("minute"), errors="coerce")
    shots["xG"] = pd.to_numeric(shots.get("xG"), errors="coerce").fillna(0.0)

    is_goal = events.get("is_goal", pd.Series(False, index=events.index))
    goals = events[is_goal.astype(str).str.lower().isin(("true", "1"))].copy()
    goals["minute"] = pd.to_numeric(goals.get("minute"), errors="coerce")
    goals = goals.dropna(subset=["minute"])

    home_colour = _team_mark_color(HOME_ID)
    away_colour = _team_mark_color(AWAY_ID)

    fig = plt.figure(figsize=(14, 8.6), facecolor=BG)
    base.amoled_header(
        fig,
        "Match Momentum",
        "Threat built in a rolling eight minutes, and the spells the match divided into",
    )
    fig.text(
        0.945,
        0.035,
        "FULL VISUAL REDESIGN · REAL MATCH DATA",
        ha="right",
        fontsize=8,
        color=NEUTRAL,
    )

    ax = fig.add_axes([0.075, 0.335, 0.875, 0.435])
    band = fig.add_axes([0.075, 0.150, 0.875, 0.075])
    for panel in (ax, band):
        base.clean_ax(panel)
        panel.set_xlim(0, last)
        panel.patch.set_alpha(0)

    top = max(float(home.max()), float(away.max()), 0.05) * 1.18
    ax.set_ylim(-top, top)
    ax.fill_between(grid, 0, home, color=home_colour, alpha=0.85, lw=0, zorder=3)
    ax.fill_between(grid, 0, -away, color=away_colour, alpha=0.85, lw=0, zorder=3)
    ax.plot(grid, home, color=home_colour, lw=1.2, zorder=4)
    ax.plot(grid, -away, color=away_colour, lw=1.2, zorder=4)
    ax.axhline(0, color=TEXT, lw=1.2, alpha=0.75, zorder=5)
    ax.axvline(45, color=MUTED, lw=0.9, ls=(0, (3, 4)), zorder=2)
    ax.text(45.5, top * 0.95, "HT", color=MUTED, fontsize=7.5, va="top")
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.55, zorder=0)
    marks = np.linspace(-top, top, 5)
    ax.set_yticks(marks)
    ax.set_yticklabels([f"{abs(v):.1f}" for v in marks])
    ax.set_xticks([])
    ax.tick_params(labelsize=8)
    ax.set_ylabel("threat built in the window", fontsize=8.5, color=MUTED)
    ax.text(
        0.004,
        0.97,
        HOME_NAME.upper(),
        transform=ax.transAxes,
        color=home_colour,
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.004,
        0.03,
        AWAY_NAME.upper(),
        transform=ax.transAxes,
        color=away_colour,
        fontsize=9.5,
        fontweight="bold",
        va="bottom",
    )

    for row in goals.itertuples():
        colour = home_colour if row.team_id == HOME_ID else away_colour
        y = float(np.interp(row.minute, grid, home if row.team_id == HOME_ID else -away))
        ax.scatter(
            [row.minute], [y], s=95, marker="o", color=colour, edgecolor=BG, lw=1.8, zorder=7
        )
        ax.annotate(
            f"{_surname(str(row.player))} {int(row.minute)}'",
            (row.minute, y),
            textcoords="offset points",
            xytext=(0, 15 if y >= 0 else -21),
            ha="center",
            color=TEXT,
            fontsize=8.5,
            fontweight="bold",
            zorder=9,
        )

    band.set_ylim(0, 1)
    band.set_yticks([])
    narrow_side = 1
    for lo, hi, _sign in runs:
        window = (grid >= lo) & (grid < hi)
        home_threat = float(slow_home[window].sum()) if window.any() else 0.0
        away_threat = float(slow_away[window].sum()) if window.any() else 0.0
        both = home_threat + away_threat
        held = home_threat / both if both else 0.5
        segment = shots[(shots["minute"] >= lo) & (shots["minute"] < hi)]
        home_xg = float(segment.loc[segment["team_id"].eq(HOME_ID), "xG"].sum())
        away_xg = float(segment.loc[segment["team_id"].eq(AWAY_ID), "xG"].sum())

        band.add_patch(
            Rectangle((lo, 0.0), hi - lo, 1.0, facecolor=away_colour, alpha=0.88, lw=0, zorder=3)
        )
        band.add_patch(
            Rectangle(
                (lo, 0.0), (hi - lo) * held, 1.0, facecolor=home_colour, alpha=0.88, lw=0, zorder=4
            )
        )
        band.plot([hi, hi], [0, 1], color=BG, lw=1.4, zorder=6)

        span = f"{int(round(lo))}–{int(round(hi))}'"
        split = f"{100 * held:.0f} / {100 * (1 - held):.0f}"
        figures = f"{home_xg:.2f} – {away_xg:.2f} xG"
        if (hi - lo) >= 11.0:
            # On the wider of the two slices, so the boundary between them
            # never runs through the middle of the label.
            anchor = lo + (hi - lo) * (held / 2 if held >= 0.5 else (1 + held) / 2)
            fill = home_colour if held >= 0.5 else away_colour
            band.text(
                anchor,
                0.66,
                f"{span}   {split}",
                ha="center",
                va="center",
                color=text_on_fill(fill),
                fontsize=9,
                fontweight="bold",
                zorder=7,
            )
            band.text(
                anchor,
                0.30,
                figures,
                ha="center",
                va="center",
                color=text_on_fill(fill),
                fontsize=8,
                zorder=7,
            )
        else:
            # Too narrow for two lines. It goes under the band on alternating
            # rows, so two short spells side by side cannot collide.
            narrow_side *= -1
            y = -0.55 if narrow_side < 0 else -1.35
            mid = (lo + hi) / 2.0
            band.plot(
                [mid, mid],
                [-0.08, y + 0.30],
                color=MUTED,
                lw=0.9,
                alpha=0.8,
                clip_on=False,
                zorder=4,
            )
            band.text(
                mid,
                y,
                f"{span}  {split}\n{figures}",
                ha="center",
                va="top",
                color=MUTED,
                fontsize=7.6,
                linespacing=1.5,
                clip_on=False,
                zorder=6,
            )

    band.text(
        0.0,
        1.42,
        "SPELLS",
        transform=band.transAxes,
        color=MUTED,
        fontsize=7.5,
        fontweight="bold",
        va="center",
    )
    band.text(
        1.0,
        1.42,
        "each block is split by the share of threat inside it · the pair beneath is the xG it produced",
        transform=band.transAxes,
        color=MUTED,
        fontsize=7.5,
        va="center",
        ha="right",
    )
    ticks = [t for t in (0, 15, 30, 45, 60, 75, 90) if t <= last]
    band.set_xticks(ticks)
    band.set_xticklabels([str(t) for t in ticks], fontsize=8)
    band.tick_params(colors=MUTED, labelsize=8, pad=44)
    band.set_xlabel("Match minute", fontsize=9, color=MUTED, labelpad=16)
    return save(fig, "35_match_momentum.png")


NOTE_STRUCK_AS_DESERVED = "struck as well as" + chr(10) + "the chance was worth"


def finishing_quality(events):
    """Chance quality, striking quality and what actually went in.

    The package priced every chance before the shot and counted what went in,
    and had nothing in between. Those two numbers cannot separate a side that
    finished well from one that was handed better chances: 3.11 xG and four
    goals is the same line whether the shots were placed in the corners or
    scuffed past a keeper who should have saved them.

    Post-shot xG is the missing middle. It re-prices an attempt once the
    placement is known, so the step from xG to post-shot xG is how well the
    ball was struck and the step from post-shot xG to goals is what the
    goalkeepers and the woodwork did with it.

    Off-target attempts have no placement to price, so they carry a post-shot
    value of zero and the first stage counts every attempt while the second
    counts only the twelve that reached the frame. That is the point of the
    panel, not a gap in it: a side that misses the target has already spent
    the chance.
    """
    from football_analysis.metrics.match_metrics import post_shot_xg

    shots = events[as_bool(events.get("is_shot"))].copy()
    shots = shots[~as_bool(shots.get("is_penalty_shootout", pd.Series(False, index=shots.index)))]
    own_goal = as_bool(shots.get("is_own_goal", pd.Series(False, index=shots.index)))
    shots = shots[~own_goal]
    if shots.empty:
        shots["_psxg"] = []
    else:
        shots["_psxg"] = (
            pd.to_numeric(post_shot_xg(events), errors="coerce").reindex(shots.index).fillna(0.0)
        )
    shots["xG"] = pd.to_numeric(shots["xG"], errors="coerce").fillna(0).clip(lower=0)
    on_target = shots["shot_whoscored_type"].astype(str).isin(["Goal", "SavedShot"])

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    base.amoled_header(
        fig,
        "Chance Quality against Finishing",
        "Pre-shot xG, post-shot xG once the placement is known, and the goals that followed",
    )

    # -- the three stages, as a slope -----------------------------------------
    ax = fig.add_axes([0.075, 0.435, 0.40, 0.33])
    base.clean_ax(ax)
    # Four stages, not three. A three-stage version set the xG of every attempt
    # beside the post-shot xG of only those that reached the frame and then
    # compared them, which is two different denominators: the second figure is
    # smaller because eleven attempts are missing from it, not because the
    # striking was poor. Splitting out the on-target subset makes every step a
    # like-for-like comparison -- what reached the frame, how it was struck,
    # and what the goalkeeper did with it.
    stages = [
        "Created\n(all attempts)",
        "Reached the\nframe (xG)",
        "Struck\n(post-shot xG)",
        "Scored",
    ]
    sides = [(HOME_ID, HOME), (AWAY_ID, AWAY)]
    series = []
    for team_id, _colour in sides:
        own = shots[shots["team_id"].eq(team_id)]
        framed_rows = own.loc[on_target.reindex(own.index, fill_value=False)]
        series.append(
            [
                float(own["xG"].sum()),
                float(framed_rows["xG"].sum()),
                float(framed_rows["_psxg"].sum()),
                float(as_bool(own.get("is_goal", pd.Series(False, index=own.index))).sum()),
            ]
        )
    ceiling = max(max(values) for values in series)
    # The label goes on the outside of the pair rather than by team, so the
    # upper line's figure is never pushed down onto the lower line's.
    offsets = slope_label_offsets(series, ceiling * 1.28)
    for (team_id, colour), values, label_dy in zip(sides, series, offsets):
        ax.plot(
            range(4),
            values,
            color=colour,
            lw=2.8,
            marker="o",
            markersize=11,
            markeredgecolor=BG,
            markeredgewidth=1.4,
            zorder=4,
            label=TEAM_NAME[team_id],
        )
        for index, value in enumerate(values):
            ax.annotate(
                f"{value:.2f}" if index < 3 else f"{value:.0f}",
                (index, value),
                xytext=(0, label_dy[index]),
                textcoords="offset points",
                color=colour,
                fontsize=10,
                fontweight="bold",
                ha="center",
            )
    ax.set_xticks(range(4), stages, color=TEXT, fontsize=8.5)
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(0, ceiling * 1.28)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
    ax.tick_params(labelsize=8.5)
    # Above the plot, not in it. Inside, the top-right corner is exactly where
    # the higher side's goal count is written: Brighton 3-0 Arsenal printed the
    # "3" through the Arsenal key.
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        frameon=False,
        labelcolor=TEXT,
        fontsize=8.5,
        ncol=2,
        borderaxespad=0.0,
    )
    fig.text(0.075, 0.795, "From chance to goal", color=TEXT, fontsize=13, fontweight="bold")

    # -- every shot that reached the frame ------------------------------------
    ax2 = fig.add_axes([0.565, 0.435, 0.37, 0.33])
    base.clean_ax(ax2)
    framed = shots[on_target.reindex(shots.index, fill_value=False)]
    limit = (
        max(
            float(framed["xG"].max() if not framed.empty else 0),
            float(framed["_psxg"].max() if not framed.empty else 0),
            0.4,
        )
        * 1.15
    )
    ax2.plot([0, limit], [0, limit], color=NEUTRAL, lw=0.9, ls=(0, (3, 3)), zorder=2)
    # The note labels the dashed line, and it sat on the line's top end --
    # which is exactly where a big chance struck well lands. Crystal Palace
    # 2-3 Ipswich printed it through the goal star at 0.42/0.47. Put it in
    # whichever corner the shots left empty: the top-left needs a small chance
    # placed perfectly and the bottom-right a big one placed badly, and one of
    # the two is free in every match seen so far. The boxes are generous, so a
    # near miss moves the note rather than shaving past a marker.
    placements = (
        (0.02, 0.98, "left", "top", (0.00, 0.34, 0.82, 1.00)),
        (0.98, 0.02, "right", "bottom", (0.64, 1.00, 0.00, 0.18)),
    )
    marks = list(
        zip(
            pd.to_numeric(framed["xG"], errors="coerce").fillna(0.0),
            pd.to_numeric(framed["_psxg"], errors="coerce").fillna(0.0),
        )
    )
    for note_x, note_y, note_ha, note_va, (x0, x1, y0, y1) in placements:
        if not any(
            x0 * limit <= mx <= x1 * limit and y0 * limit <= my <= y1 * limit for mx, my in marks
        ):
            break
    ax2.text(
        limit * note_x,
        limit * note_y,
        NOTE_STRUCK_AS_DESERVED,
        color=NEUTRAL,
        fontsize=7.5,
        ha=note_ha,
        va=note_va,
    )
    for team_id, colour in [(HOME_ID, HOME), (AWAY_ID, AWAY)]:
        own = framed[framed["team_id"].eq(team_id)]
        if own.empty:
            continue
        scored = as_bool(own.get("is_goal", pd.Series(False, index=own.index)))
        ax2.scatter(
            own.loc[~scored, "xG"],
            own.loc[~scored, "_psxg"],
            s=70,
            color=colour,
            alpha=0.55,
            edgecolors=TEXT,
            linewidths=0.5,
            zorder=4,
        )
        ax2.scatter(
            own.loc[scored, "xG"],
            own.loc[scored, "_psxg"],
            s=190,
            color=colour,
            marker="*",
            edgecolors=TEXT,
            linewidths=0.6,
            zorder=5,
        )
    # A little room below zero: a goal Opta prices near nothing after the strike
    # (Groß at Brighton, 0.01) otherwise sits on the axis with half its star cut.
    pad = limit * 0.03
    ax2.set_xlim(-pad, limit)
    ax2.set_ylim(-pad, limit)
    ax2.set_xlabel("Pre-shot xG", fontsize=9, color=MUTED)
    ax2.set_ylabel("Post-shot xG", fontsize=9, color=MUTED)
    ax2.grid(color=GRID, lw=0.7, alpha=0.6)
    ax2.tick_params(labelsize=8.5)
    fig.text(
        0.565,
        0.795,
        "Each attempt that reached the frame",
        color=TEXT,
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.565,
        0.775,
        "Above the line the ball was struck better than the chance deserved · star = goal",
        color=MUTED,
        fontsize=8.5,
    )

    # -- the reading ----------------------------------------------------------
    lines = []
    for team_id in (HOME_ID, AWAY_ID):
        own = shots[shots["team_id"].eq(team_id)]
        framed_rows = own.loc[on_target.reindex(own.index, fill_value=False)]
        pre = float(own["xG"].sum())
        framed_xg = float(framed_rows["xG"].sum())
        post = float(framed_rows["_psxg"].sum())
        goals = int(as_bool(own.get("is_goal", pd.Series(False, index=own.index))).sum())
        # Both figures in this comparison now cover the same attempts. The old
        # one read post-shot xG against the xG of every shot including the
        # eleven that never reached the frame, and so called every side in
        # every match a poor finishing team.
        verdict = (
            "struck them better than they arrived"
            if post > framed_xg
            else "struck them worse than they arrived"
        )
        lines.append(
            f"{TEAM_NAME[team_id]}: {pre:.2f} xG created, {framed_xg:.2f} of it reached the "
            f"frame, worth {post:.2f} once struck, {goals} scored - {verdict}."
        )
    fig.text(0.075, 0.345, lines[0], color=HOME, fontsize=10)
    fig.text(0.075, 0.315, lines[1], color=AWAY, fontsize=10)
    # Say where the middle figure came from. Opta's post-shot value knows the
    # pace of the shot and where the keeper stood; the local estimate knows the
    # placement only, and reads well below it on the chances that matter.
    framed_all = shots[on_target.reindex(shots.index, fill_value=False)]
    from_opta = (
        pd.to_numeric(framed_all.get("xgot_reference"), errors="coerce").notna().sum()
        if "xgot_reference" in framed_all
        else 0
    )
    if len(framed_all) and from_opta == len(framed_all):
        source_note = (
            "Post-shot xG here is a published reference value: where the ball crossed the "
            "line, how hard it was struck and where the goalkeeper stood."
        )
    elif from_opta:
        source_note = (
            f"Post-shot xG is a published reference value for {from_opta} of {len(framed_all)} "
            "attempts on target; the rest are a local placement estimate without pace or "
            "keeper position."
        )
    else:
        source_note = (
            "Post-shot xG is a local placement estimate. It has no shot velocity and no goalkeeper "
            "position, so it prices where the ball went, not how hard it was to stop."
        )
    fig.text(0.075, 0.265, source_note, color=NEUTRAL, fontsize=8.5)

    fig.text(
        0.945,
        0.035,
        "FULL VISUAL REDESIGN · REAL MATCH DATA",
        ha="right",
        fontsize=8,
        color=NEUTRAL,
    )
    return save(fig, "44_finishing_quality.png")


def pass_sonar(events):
    """Which way each side passed, how far, and how often it came off.

    ``pass_angle`` and ``pass_length`` arrive on every pass in the feed and
    nothing in the package read them. A pass network says who connected to
    whom and a progression map says which passes went forward, but neither
    answers the first question asked of a build-up: what shape was it. A side
    that plays 40-metre diagonals and one that plays 8-metre triangles can
    produce the same network.

    One wedge per direction. Its length is the median pass in that direction,
    so a long wedge is a long ball rather than a frequent one; its opacity is
    the completion rate, so a pale wedge is a direction that did not come off;
    and the count sits outside it, because volume is the one thing the wedge
    geometry cannot also carry without becoming unreadable.

    The median rather than the mean: one goal-kick hoofed clear drags a mean
    into claiming the whole direction was long.
    """
    BINS = 16
    edges = np.linspace(0, 2 * np.pi, BINS + 1)
    width = edges[1] - edges[0]
    centres = edges[:-1] + width / 2

    passes = events[as_bool(events.get("is_pass"))].copy()
    passes["pass_angle"] = pd.to_numeric(passes.get("pass_angle"), errors="coerce")
    passes["pass_length"] = pd.to_numeric(passes.get("pass_length"), errors="coerce")
    passes = passes.dropna(subset=["pass_angle", "pass_length"])
    completed = (
        passes.get("outcome", pd.Series("", index=passes.index))
        .astype(str)
        .str.lower()
        .eq("successful")
    )
    passes["_ok"] = completed

    reach = float(passes["pass_length"].quantile(0.92)) if not passes.empty else 30.0
    reach = max(reach, 12.0)

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    base.amoled_header(
        fig,
        "Passing Sonar",
        "Direction, distance and completion of every recorded pass · attacking direction is to the right",
    )

    for column, (team_id, team_name, colour) in enumerate(
        [(HOME_ID, HOME_NAME, HOME), (AWAY_ID, AWAY_NAME, AWAY)]
    ):
        own = passes[passes["team_id"].eq(team_id)]
        ax = fig.add_axes([0.08 + column * 0.47, 0.19, 0.34, 0.49], projection="polar")
        ax.set_facecolor(BG)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_ylim(0, reach)
        ax.set_yticks([reach / 3, 2 * reach / 3, reach])
        ax.set_yticklabels(
            [f"{reach / 3:.0f}m", f"{2 * reach / 3:.0f}m", f"{reach:.0f}m"],
            color=MUTED,
            fontsize=7.5,
        )
        # The distance ticks default to the forward axis, which is where the
        # longest wedges are, so the metre labels sat on top of them.
        ax.set_rlabel_position(112)
        ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax.set_xticklabels(
            ["FORWARD", "", "LEFT", "", "BACK", "", "RIGHT", ""],
            color=MUTED,
            fontsize=8,
            fontweight="bold",
        )
        ax.grid(color=GRID, lw=0.6, alpha=0.8)
        ax.spines["polar"].set_color(GRID)

        binned = pd.cut(
            own["pass_angle"] % (2 * np.pi), bins=edges, labels=False, include_lowest=True
        )
        for index in range(BINS):
            slice_rows = own[binned.eq(index)]
            if slice_rows.empty:
                continue
            median_length = float(slice_rows["pass_length"].median())
            rate = float(slice_rows["_ok"].mean())
            # Opacity carries completion. The floor keeps a direction that was
            # tried and always lost from disappearing, which would read as a
            # direction nobody tried.
            ax.bar(
                centres[index],
                min(median_length, reach),
                width=width * 0.88,
                bottom=0,
                color=colour,
                alpha=0.25 + 0.65 * rate,
                edgecolor=BG,
                linewidth=0.8,
                zorder=3,
            )
            if len(slice_rows) >= max(3, len(own) * 0.03):
                ax.text(
                    centres[index],
                    min(median_length, reach) + reach * 0.07,
                    str(len(slice_rows)),
                    color=TEXT,
                    fontsize=7.5,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

        forward = own[(own["pass_angle"] < np.pi / 2) | (own["pass_angle"] > 3 * np.pi / 2)]
        share = 100 * len(forward) / max(len(own), 1)
        # Clear of the polar frame's own direction labels, which are drawn
        # outside the axes box and were landing on this subtitle.
        fig.text(
            0.08 + column * 0.47 + 0.17,
            0.795,
            team_name.upper(),
            color=colour,
            fontsize=15,
            fontweight="bold",
            ha="center",
        )
        fig.text(
            0.08 + column * 0.47 + 0.17,
            0.772,
            f"{len(own)} passes · {share:.0f}% forward · "
            f"median {own['pass_length'].median():.0f} m · "
            f"{100 * own['_ok'].mean():.0f}% completed",
            color=MUTED,
            fontsize=9,
            ha="center",
        )

    fig.text(
        0.5,
        0.135,
        "Wedge length = median pass distance in that direction · "
        "opacity = completion rate · figure outside the wedge = passes attempted",
        color=MUTED,
        fontsize=9,
        ha="center",
    )
    fig.text(
        0.5,
        0.108,
        "Direction is the recorded pass angle, not the player's body orientation. "
        "A short wedge is a short pass, not a rare one.",
        color=NEUTRAL,
        fontsize=8.5,
        ha="center",
    )
    fig.text(
        0.945,
        0.035,
        "FULL VISUAL REDESIGN · REAL MATCH DATA",
        ha="right",
        fontsize=8,
        color=NEUTRAL,
    )
    return save(fig, "43_pass_sonar.png")


def set_pieces(events):
    """Where each side's shots came from, and what the dead ball was worth."""
    sources = [
        ("open_play", "Open play"),
        ("corner", "Corners"),
        ("free_kick", "Free kicks"),
        ("throw_in", "Throw-ins"),
        ("penalty", "Penalties"),
    ]
    # A throw is a restart, not a dead ball: the defence is not set, nothing is
    # rehearsed, and no analyst counts a goal three passes after a throw as a
    # set-piece goal. Keeping it in this share had Chelsea at 39% "from a dead
    # ball" in the same package whose article credited them with one set-piece
    # goal, the corner. The row stays — where a possession started is worth
    # showing — but it is not part of the claim in the line above the chart.
    DEAD_BALL = ("corner", "free_kick", "penalty")
    home = set_piece_breakdown(events, HOME_ID)
    away = set_piece_breakdown(events, AWAY_ID)
    # A source neither side shot from is an empty row twice over; both panels
    # drop it together so the rows still line up side by side.
    sources = [
        (key, label) for key, label in sources if home[key]["shots"] + away[key]["shots"] > 0
    ] or sources[:1]
    # One shot scale for both panels: auto-scaled, a four-shot bar and a
    # fourteen-shot bar came out the same length.
    shot_max = max([data[key]["shots"] for data in (home, away) for key, _ in sources] + [1])

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    base.amoled_header(
        fig,
        "Set-Piece Contribution",
        "Every shot traced back to how the possession started \u00b7 the source is read off the delivery, not the shot",
    )
    for column, (team_name, data, color) in enumerate(
        [(HOME_NAME, home, HOME), (AWAY_NAME, away, AWAY)]
    ):
        left = 0.075 + column * 0.475
        fig.text(left, 0.795, team_name.upper(), color=color, fontsize=13, fontweight="bold")
        dead_ball_xg = sum(data[key]["xG"] for key in DEAD_BALL)
        total_xg = sum(data[key]["xG"] for key, _ in sources)
        share = 100 * dead_ball_xg / max(total_xg, 0.01)
        fig.text(
            left,
            0.768,
            f"{share:.0f}% of xG came from a dead ball (corners, free kicks, penalties)",
            color=MUTED,
            fontsize=8.5,
        )

        ax = fig.add_axes([left, 0.20, 0.39, 0.52])
        base.clean_ax(ax)
        labels = [label for _, label in sources]
        shots = [data[key]["shots"] for key, _ in sources]
        xgs = [data[key]["xG"] for key, _ in sources]
        positions = np.arange(len(sources))
        ax.barh(positions, shots, height=0.62, color=color, alpha=0.9)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=9, color=TEXT)
        ax.invert_yaxis()
        ax.set_xlim(0, shot_max * 1.45)
        ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
        ax.set_xlabel("Shots", fontsize=8.5, color=MUTED)
        ax.tick_params(labelsize=8)
        for position, (count, value, (key, _label)) in enumerate(zip(shots, xgs, sources)):
            if count == 0:
                continue
            ax.text(
                count + shot_max * 0.03,
                position,
                f"{count}  \u00b7  {value:.2f} xG  \u00b7  {data[key]['goals']}G",
                color=TEXT,
                fontsize=8,
                va="center",
                fontweight="bold",
            )

    fig.text(
        0.945,
        0.035,
        "FULL VISUAL REDESIGN \u00b7 REAL MATCH DATA",
        ha="right",
        fontsize=8,
        color=NEUTRAL,
    )
    return save(fig, "36_set_pieces.png")


def turnovers(events, team_id, number):
    """Where a team lost the ball, and which losses the opponent punished."""
    frame = turnover_events(events, team_id)
    fig, pitch, side = pitch_axes(
        f"Ball Losses \u00b7 {TEAM_NAME[team_id]}",
        "Every possession lost \u00b7 filled marks were punished with a shot inside 15 seconds",
    )
    draw_long_pitch(pitch)
    team_mark = _team_mark_color(team_id)

    if not frame.empty:
        safe = frame[~frame["punished"]]
        punished = frame[frame["punished"]]
        if not safe.empty:
            px, py = attack_xy(safe["x"], safe["y"])
            pitch.scatter(
                px,
                py,
                s=34,
                marker="o",
                facecolors="none",
                edgecolors=team_mark,
                linewidths=0.9,
                alpha=0.55,
                zorder=4,
            )
        if not punished.empty:
            px, py = attack_xy(punished["x"], punished["y"])
            pitch.scatter(
                px,
                py,
                s=70 + punished["conceded_xG"].to_numpy() * 900,
                marker="o",
                facecolors=SHOT_GOAL,
                edgecolors=BG,
                linewidths=0.9,
                alpha=0.95,
                zorder=6,
            )

    total = len(frame)
    punished_count = int(frame["punished"].sum()) if not frame.empty else 0
    conceded = float(frame["conceded_xG"].sum()) if not frame.empty else 0.0
    own_half = int((frame["x"] < 50).sum()) if not frame.empty else 0
    side_title(side, "LOSS PROFILE")
    side_kpis(
        side,
        [
            ("Possessions lost", f"{total}"),
            ("Punished", f"{punished_count}"),
            ("Punish rate", f"{100 * punished_count / max(total, 1):.0f}%"),
            ("xG conceded from losses", f"{conceded:.2f}"),
            ("Lost in own half", f"{own_half}"),
        ],
    )
    return save(fig, f"{number:02d}_ball_losses_{_team_slug(team_id)}.png")


def _smoothed(frame, column, window=3):
    """Rolling median of a five-minute series, centred, with short ends kept.

    A five-minute window holds a handful of defensive actions, so the mean of
    their x swings ten metres on one clearance. Plotted raw, both teams became
    zigzags of equal amplitude and the eye could not separate the trend from
    the sampling. The median rather than the mean because the thing being
    smoothed away is exactly the single long clearance.
    """
    if frame.empty:
        return frame
    out = frame.copy()
    out["_smooth"] = out[column].rolling(window, center=True, min_periods=1).median()
    return out


def shape_over_time(events):
    """Where each side engaged, and when that changed.

    Two raw five-minute series drawn as polylines produced two overlapping
    zigzags: the chart had a shape but no reading. What a coach asks of it is
    which side was higher and at what point that reversed, so the smoothed
    lines are the subject, the gap between them is filled in the colour of
    whoever is higher, and the raw samples stay on the page as faint dots so
    nothing is hidden by the smoothing.
    """
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    base.amoled_header(
        fig,
        "Defensive Action Height and Touch Spread",
        "Engagement height and compactness per five-minute window · the band shows which side was higher",
    )
    height_ax = fig.add_axes([0.075, 0.475, 0.86, 0.29])
    spread_ax = fig.add_axes([0.075, 0.135, 0.86, 0.27])
    for ax in (height_ax, spread_ax):
        base.clean_ax(ax)
        ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
        ax.tick_params(labelsize=8)
        ax.axvline(45, color=GRID, lw=0.9, ls=(0, (3, 4)))
    height_ax.text(45.6, 2, "half time", color=NEUTRAL, fontsize=6.5, va="bottom")

    panels = [
        (height_ax, defensive_line_height, "height", "Mean x of defensive actions"),
        (spread_ax, team_compactness, "vertical_spread", "Vertical spread of touches (IQR)"),
    ]
    for ax, source, column, label in panels:
        series = {}
        for team_id, colour in [(HOME_ID, HOME), (AWAY_ID, AWAY)]:
            frame = _smoothed(source(events, team_id), column)
            if frame.empty:
                continue
            series[team_id] = frame
            minutes = frame["window_start"] + 2.5
            ax.scatter(minutes, frame[column], s=11, color=colour, alpha=0.30, zorder=2)
            ax.plot(
                minutes,
                frame["_smooth"],
                color=colour,
                lw=2.6,
                zorder=4,
                solid_capstyle="round",
                label=TEAM_NAME[team_id],
            )

        # The band is the finding: its colour names the side engaging higher
        # and its thickness is by how much, so a crossover is a place where the
        # fill changes hands rather than a point the reader has to hunt for.
        if len(series) == 2:
            home_frame, away_frame = series[HOME_ID], series[AWAY_ID]
            shared = sorted(set(home_frame["window_start"]) & set(away_frame["window_start"]))
            if shared:
                home_line = home_frame.set_index("window_start").loc[shared, "_smooth"]
                away_line = away_frame.set_index("window_start").loc[shared, "_smooth"]
                minutes = np.array(shared, dtype=float) + 2.5
                ax.fill_between(
                    minutes,
                    home_line,
                    away_line,
                    where=home_line >= away_line,
                    interpolate=True,
                    color=HOME,
                    alpha=0.16,
                    zorder=1,
                    linewidth=0,
                )
                ax.fill_between(
                    minutes,
                    home_line,
                    away_line,
                    where=home_line < away_line,
                    interpolate=True,
                    color=AWAY,
                    alpha=0.16,
                    zorder=1,
                    linewidth=0,
                )
        ax.set_ylabel(label, fontsize=8.5, color=MUTED)

    height_ax.set_ylim(0, 100)
    height_ax.axhline(50, color=PITCH_LINE, lw=0.8, alpha=0.35)
    height_ax.text(1, 51, "halfway", color=MUTED, fontsize=6.5, va="bottom")
    height_ax.legend(loc="upper right", frameon=False, labelcolor=TEXT, fontsize=8, ncol=2)
    spread_ax.set_xlabel("Match minute", fontsize=9, color=MUTED)

    for ax, source, column in [
        (height_ax, defensive_line_height, "height"),
        (spread_ax, team_compactness, "vertical_spread"),
    ]:
        combined = pd.concat([source(events, HOME_ID), source(events, AWAY_ID)])
        if combined.empty:
            continue
        median_value = float(combined[column].median())
        ax.axhline(median_value, color=TEXT, lw=0.8, ls=(0, (2, 3)), alpha=0.45)
        ax.text(
            99,
            median_value + 1.5,
            f"match median {median_value:.0f}",
            color=MUTED,
            fontsize=6.5,
            ha="right",
        )

    fig.text(
        0.075,
        0.075,
        "Line = rolling median of three windows · dots = the five-minute samples behind it",
        color=NEUTRAL,
        fontsize=8,
    )
    fig.text(
        0.945,
        0.035,
        "FULL VISUAL REDESIGN · REAL MATCH DATA",
        ha="right",
        fontsize=8,
        color=NEUTRAL,
    )
    return save(fig, "39_defensive_shape.png")


def win_probability_curve(events):
    """How the result hardened: the same goal is worth more the later it lands."""
    frame = win_probability(events, HOME_ID, AWAY_ID, window=5)
    fig, ax = base.page(
        "Win Probability",
        "Modelled from the scoreline, the time still to play and each side's xG rate \u00b7 a heuristic, not a market price",
    )
    base.clean_ax(ax)
    if frame.empty:
        ax.text(0.5, 0.5, "No live events", color=MUTED, ha="center", va="center")
        return save(fig, "40_win_probability.png")

    minutes = frame["minute"].to_numpy()
    home = frame["home_win"].to_numpy() * 100
    draw = frame["draw"].to_numpy() * 100
    away = frame["away_win"].to_numpy() * 100

    ax.stackplot(
        minutes,
        home,
        draw,
        away,
        colors=[HOME, NEUTRAL, AWAY],
        alpha=0.88,
        labels=[HOME_NAME, "Draw", AWAY_NAME],
    )
    ax.set_xlim(minutes.min(), minutes.max())
    ax.set_ylim(0, 100)
    ax.set_xlabel("Match minute", fontsize=9, color=MUTED)
    ax.set_ylabel("Probability (%)", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8)
    ax.axvline(45, color=BG, lw=1.1, ls=(0, (3, 4)))
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        labelcolor=TEXT,
        fontsize=8.5,
    )

    # Mark every goal: the step in the curve is the point of the chart.
    goals = events[as_bool(events["is_goal"])].copy()
    if not goals.empty:
        goals["_credited"] = base.credited_team(goals)
        for _, goal in goals.sort_values("minute").iterrows():
            minute = float(goal["minute"])
            color = HOME if int(goal["_credited"]) == HOME_ID else AWAY
            ax.axvline(minute, color=BG, lw=2.4, alpha=0.9)
            ax.axvline(minute, color=color, lw=1.1)
            ax.text(
                minute,
                102,
                f"{int(minute)}\u2032",
                color=color,
                fontsize=6.4,
                fontweight="bold",
                ha="center",
                va="bottom",
            )
    return save(fig, "40_win_probability.png")


def playing_through(events, team_id, opponent_id, number):
    """Passes that beat the opponent's line, and how the side held up when pressed."""
    breaks = line_breaking_passes(events, team_id, opponent_id)
    fig, pitch, side = pitch_axes(
        f"Reception and Progression Routes \u00b7 {TEAM_NAME[team_id]}",
        "Passes starting behind the opponent's defensive line and finishing beyond it \u00b7 line height per five-minute window",
    )
    draw_long_pitch(pitch)
    team_mark = _team_mark_color(team_id)

    if not breaks.empty:
        # The line being broken, at its average height over the match. Each
        # arrow starts behind it and ends beyond it, so drawing it shows what
        # "line-breaking" meant instead of leaving the reader to take it on trust.
        line_y = float(breaks["line_height"].mean()) * PITCH_LENGTH / 100
        pitch.axhline(line_y, color=FOCUS, lw=1.2, ls=(0, (5, 4)), zorder=1, alpha=0.9)
        for _, row in breaks.iterrows():
            sx, sy = attack_xy([row["x"]], [row["y"]])
            ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
            if bool(row["successful"]):
                arrow = pitch.annotate(
                    "",
                    xy=(ex[0], ey[0]),
                    xytext=(sx[0], sy[0]),
                    arrowprops=dict(
                        arrowstyle="-|>", color=team_mark, alpha=0.95, lw=2.0, mutation_scale=12
                    ),
                    zorder=5,
                )
                if arrow.arrow_patch is not None:
                    arrow.arrow_patch.set_path_effects(
                        [path_effects.Stroke(linewidth=3.8, foreground=BG), path_effects.Normal()]
                    )
            else:
                # A pass that did not arrive is a mark where it started, not a
                # second tangle of lines under the ones that worked.
                pitch.scatter(
                    sx,
                    sy,
                    s=46,
                    marker="o",
                    facecolors="none",
                    edgecolors=EVENT_NEUTRAL,
                    linewidths=1.3,
                    zorder=4,
                )

    resistance = press_resistance(events, team_id)
    completed = int(breaks["successful"].sum()) if not breaks.empty else 0
    side_title(side, "PLAYING THROUGH")
    side_kpis(
        side,
        [
            ("Line-breaking passes", f"{len(breaks)}"),
            ("Completed", f"{completed}"),
            ("Completion", f"{100 * completed / max(len(breaks), 1):.0f}%"),
        ],
        start=0.82,
        gap=0.13,
    )

    side.text(0.08, 0.40, "UNDER PRESSURE", color=MUTED, fontsize=7.5, fontweight="bold")
    for idx, (label, value) in enumerate(
        [
            ("Passes pressed", f"{resistance['passes_under_pressure']}"),
            ("Share of passes", f"{resistance['pressed_share']:.0f}%"),
            ("Completion pressed", f"{resistance['pressed_completion']:.0f}%"),
            ("Completion free", f"{resistance['free_completion']:.0f}%"),
            ("Resistance gap", f"{resistance['resistance_gap']:+.0f} pts"),
        ]
    ):
        y = 0.35 - idx * 0.048
        side.text(0.08, y, label, color=TEXT, fontsize=8, va="center")
        side.text(
            0.92, y, value, color=TEXT, fontsize=8.5, fontweight="bold", ha="right", va="center"
        )

    duels = duel_map(events, team_id)
    if not duels.empty:
        won = int(duels["won"].sum())
        aerial = duels[duels["kind"] == "aerial"]
        side.text(0.08, 0.085, "DUELS", color=MUTED, fontsize=7.5, fontweight="bold")
        side.text(
            0.08,
            0.04,
            f"{won}/{len(duels)} won  \u00b7  aerial {int(aerial['won'].sum())}/{len(aerial)}",
            color=TEXT,
            fontsize=8,
            va="center",
        )

    pitch.plot([], [], color=FOCUS, lw=1.2, ls=(0, (5, 4)), label="Opponent line (average)")
    pitch.plot([], [], color=team_mark, lw=2.0, label="Completed line-breaking pass")
    pitch.scatter(
        [],
        [],
        s=46,
        facecolors="none",
        edgecolors=EVENT_NEUTRAL,
        linewidths=1.3,
        label="Incomplete (origin)",
    )
    pitch.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=3,
        frameon=False,
        labelcolor=TEXT,
        fontsize=7.5,
    )
    return save(fig, f"{number:02d}_playing_through_{_team_slug(team_id)}.png")


def action_value_leaders(events):
    """One value ranking every position can appear in."""
    ranked = player_action_value(events)
    # An event with no player attached is not a player: it printed as a nameless
    # row with a value of +0.000 at the foot of the ranking.
    if not ranked.empty:
        named = ranked["player"].notna() & ~ranked["player"].astype(str).str.strip().isin(
            ["", "nan", "None"]
        )
        ranked = ranked[named]
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    base.amoled_header(
        fig,
        "Action Value",
        "Every action priced on one scale in goals · on-ball gains and losses plus the threat denied by defensive work",
    )
    if ranked.empty:
        fig.text(0.5, 0.5, "No valued actions", color=MUTED, ha="center")
        return save(fig, "43_action_value.png")

    top = ranked.head(14).iloc[::-1]
    ax = fig.add_axes([0.20, 0.13, 0.62, 0.63])
    base.clean_ax(ax)
    positions = np.arange(len(top))
    offensive = top["offensive_value"].to_numpy()
    defensive = top["defensive_value"].to_numpy()

    # Stacked from zero in both directions so a player who lost value on the
    # ball but won it back still reads honestly.
    for index, (off, dfn) in enumerate(zip(offensive, defensive)):
        color = HOME if int(top.iloc[index]["team_id"]) == HOME_ID else AWAY
        ax.barh(index, off, height=0.62, color=color, alpha=0.95)
        ax.barh(index, dfn, left=max(off, 0.0), height=0.62, color=color, alpha=0.42)

    ax.set_yticks(positions)
    ax.set_yticklabels([_surname(name) for name in top["player"]], fontsize=8.5, color=TEXT)
    ax.axvline(0, color=PITCH_LINE, lw=1.0, alpha=0.6)
    ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
    ax.set_xlabel("Action value (goals)", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8)
    # Room on the right for those labels. Without it the widest bar reaches
    # the axis edge and its number is drawn past it, over whatever sits beside
    # the panel.
    _reach = max([max(o, 0.0) + max(d, 0.0) for o, d in zip(offensive, defensive)] + [0.01])
    _left = min(list(offensive) + [0.0])
    _value_pad = _reach * 0.03
    ax.set_xlim(_left * 1.15 if _left < 0 else 0.0, _reach * 1.30)

    # Label at the bar's right edge, not at the total. When a player lost value
    # on the ball the bar starts left of zero, so the two do not coincide and
    # the number ended up printed across the bar.
    for index, (off, dfn, total) in enumerate(
        zip(offensive, defensive, top["total_value"].to_numpy())
    ):
        ax.text(
            max(off, 0.0) + max(dfn, 0.0) + _value_pad,
            index,
            f"{total:+.3f}",
            color=TEXT,
            fontsize=8,
            fontweight="bold",
            va="center",
        )

    fig.text(
        0.20,
        0.80,
        "SOLID = ON-BALL   ·   FADED = DEFENSIVE",
        color=MUTED,
        fontsize=7.5,
        fontweight="bold",
    )
    # Which bar is whose: the bars are coloured by team and nothing said so.
    for slot, (name, colour) in enumerate([(HOME_NAME, HOME), (AWAY_NAME, AWAY)]):
        key_x = 0.60 + slot * 0.11
        fig.add_artist(
            Rectangle((key_x, 0.797), 0.011, 0.016, transform=fig.transFigure, color=colour)
        )
        fig.text(
            key_x + 0.016,
            0.805,
            name.upper(),
            color=TEXT,
            fontsize=7.5,
            fontweight="bold",
            va="center",
        )
    fig.text(
        0.20,
        0.075,
        "Zone-value model, not a fitted VAEP: no labelled training data, so the value surface is explicit rather than learned.",
        color=NEUTRAL,
        fontsize=7,
    )
    fig.text(
        0.945,
        0.035,
        "FULL VISUAL REDESIGN · REAL MATCH DATA",
        ha="right",
        fontsize=8,
        color=NEUTRAL,
    )
    return save(fig, "43_action_value.png")


def control_surface(events, players=None):
    """Which side held which parts of the pitch, and where it was contested."""
    grid, shares = pitch_control(events, HOME_ID, AWAY_ID)
    fig, pitch, side = pitch_axes(
        "Average-position Influence",
        "Time-weighted touch windows split at substitutions · illustrative event model, not tracking",
    )
    cmap = LinearSegmentedColormap.from_list("control", [AWAY, PANEL_2, HOME])
    # grid is indexed [pitch_y, pitch_x]; this display puts pitch x up the page,
    # so it has to be transposed or the control reads across the pitch instead
    # of up it.
    pitch.imshow(
        grid.T,
        extent=[-PITCH_WIDTH / 2, PITCH_WIDTH / 2, 0, PITCH_LENGTH],
        origin="lower",
        cmap=cmap,
        vmin=0.25,
        vmax=0.75,
        aspect="equal",
        alpha=0.92,
    )
    draw_long_pitch(pitch)

    for team_id, color in [(HOME_ID, HOME), (AWAY_ID, AWAY)]:
        positions = team_average_positions(events, team_id)
        if positions.empty:
            continue
        # Both sides are plotted in the home team's attacking frame, which is
        # the frame the control grid was built in.
        xs = positions["x"] if team_id == HOME_ID else 100 - positions["x"]
        ys = positions["y"] if team_id == HOME_ID else 100 - positions["y"]
        px, py = attack_xy(xs, ys)
        # Shirt numbers in the marks, so the white dots are people and not a cloud.
        numbers = {}
        if players is not None and "shirt_no" in players.columns:
            numbers = dict(zip(players["name"].astype(str), players["shirt_no"]))
        pitch.scatter(
            px, py, s=230, marker="o", facecolors=color, edgecolors=BG, linewidths=1.2, zorder=6
        )
        for name, x_, y_ in zip(positions["player"], px, py):
            shirt = numbers.get(str(name))
            if shirt is None or pd.isna(shirt):
                continue
            pitch.text(
                x_,
                y_,
                str(int(float(shirt))),
                color=text_on_fill(color),
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=7,
            )

    # Everything on this map is encoded: the field's colour is which side held
    # the space, the dots are average positions. Neither was stated anywhere.
    pitch_legend(
        pitch,
        [
            ("patch", HOME, f"{HOME_NAME} holds"),
            ("patch", PANEL_2, "Contested"),
            ("patch", AWAY, f"{AWAY_NAME} holds"),
            ("o", TEXT, "Average position"),
        ],
        ncol=4,
    )

    side_title(side, "TERRITORY")
    side_kpis(
        side,
        [
            (f"{HOME_NAME} control", f"{shares['home']:.0f}%"),
            (f"{AWAY_NAME} control", f"{shares['away']:.0f}%"),
            ("Contested", f"{shares['contested']:.0f}%"),
        ],
        start=0.82,
        gap=0.14,
    )
    side.text(
        0.08,
        0.36,
        "Contested = similar modeled influence.\nAverages include different participants\nat different times; this is not tracking.",
        color=MUTED,
        fontsize=7.6,
        va="top",
        linespacing=1.6,
    )
    return save(fig, "44_pitch_control.png")


def sequence_types(events):
    """How each side actually built its danger, not just how much."""
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    base.amoled_header(
        fig,
        "How the Danger Was Built",
        "Every possession classified by how it was constructed \u00b7 two sides can reach the same xG by completely different routes",
    )
    labels = {
        "build_up": "Build-up",
        "sustained": "Sustained",
        "direct": "Direct",
        "counter": "Counter",
        "set_piece": "Set piece",
        "other": "Other",
    }
    order = ["sustained", "build_up", "direct", "counter", "set_piece", "other"]

    # One shared x-scale. Two panels auto-scaled to their own maximum make a
    # 0.29 bar look identical to a 1.05 bar, which inverts the comparison the
    # page exists to make.
    both = {
        team_id: sequence_typology(events, team_id).set_index("type")
        for team_id in (HOME_ID, AWAY_ID)
    }
    scale_max = (
        max([float(frame["xG"].max()) for frame in both.values() if not frame.empty] or [0.1])
        * 1.45
    )
    # A route neither side used is an empty row on both panels; drop it from both.
    order = [
        key
        for key in order
        if any(
            key in frame.index and float(frame.loc[key, "sequences"]) > 0 for frame in both.values()
        )
    ] or order[:1]

    for column, (team_id, team_name, color) in enumerate(
        [(HOME_ID, HOME_NAME, HOME), (AWAY_ID, AWAY_NAME, AWAY)]
    ):
        typology = both[team_id]
        left = 0.075 + column * 0.475
        fig.text(left, 0.795, team_name.upper(), color=color, fontsize=13, fontweight="bold")

        ax = fig.add_axes([left, 0.20, 0.39, 0.52])
        base.clean_ax(ax)
        values = [float(typology["xG"].get(key, 0.0)) for key in order]
        positions = np.arange(len(order))
        ax.barh(positions, values, height=0.62, color=color, alpha=0.9)
        ax.set_yticks(positions)
        ax.set_yticklabels([labels[key] for key in order], fontsize=9, color=TEXT)
        ax.invert_yaxis()
        ax.set_xlim(0, scale_max)
        ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
        ax.set_xlabel("xG", fontsize=8.5, color=MUTED)
        ax.tick_params(labelsize=8)

        for index, key in enumerate(order):
            if key not in typology.index:
                continue
            row = typology.loc[key]
            ax.text(
                float(row["xG"]) + scale_max * 0.02,
                index,
                f"{int(row['sequences'])} seq  \u00b7  {int(row['goals'])}G  \u00b7  {float(row['share_of_xG']):.0f}% of xG",
                color=TEXT,
                fontsize=7.6,
                va="center",
                fontweight="bold",
            )

        if not typology.empty:
            best = typology["xG"].idxmax()
            fig.text(
                left,
                0.768,
                f"Most dangerous route: {labels[best].lower()} ({typology.loc[best, 'share_of_xG']:.0f}% of xG)",
                color=MUTED,
                fontsize=8.5,
            )

    fig.text(
        0.945,
        0.035,
        "FULL VISUAL REDESIGN \u00b7 REAL MATCH DATA",
        ha="right",
        fontsize=8,
        color=NEUTRAL,
    )
    return save(fig, "45_sequence_types.png")


def _goal_sequence(annotated, goal_row):
    """The scoring side's own touches in the possession a goal ended, in order.

    Passes and carries that reached a team-mate, plus the shot itself. Events by
    the opponent inside the same possession window (a clearance that dropped to
    the scorer, say) are not part of the build-up and are left out.
    """
    window = annotated[annotated["possession_id"].eq(goal_row["possession_id"])]
    window = window[window["team_id"].eq(goal_row["team_id"])]
    shot = window[as_bool(window["is_goal"])].tail(1)
    moves = window[window["type"].astype(str).isin(["Pass", "Carry"])].dropna(
        subset=["x", "y", "end_x", "end_y"]
    )
    moves = moves[moves["outcome"].astype(str).str.lower().eq("successful")]
    return moves, shot


def goal_origins(events):
    """The sequence behind every goal, drawn on the pitch it travelled over."""
    chains = goal_origin_chains(events, HOME_ID, AWAY_ID)
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    base.amoled_header(
        fig,
        "Goal Origins",
        "Every goal drawn from where its possession began · numbered touches, arrows for the passes that built it",
    )
    if chains.empty:
        fig.text(0.5, 0.5, "No goals from open sequences", color=MUTED, ha="center")
        return save(fig, "46_goal_origins.png")

    annotated, _possessions = build_possessions(events)
    goals = annotated[as_bool(annotated["is_goal"]) & ~as_bool(annotated["is_own_goal"])]
    goals = goals.dropna(subset=["possession_id"]).sort_values(["minute", "second"], kind="stable")

    count = len(chains)
    columns = min(count, 4)
    rows_n = int(np.ceil(count / columns))
    area_top, area_bottom = 0.84, 0.07
    cell_h = (area_top - area_bottom) / rows_n
    caption_h = 0.17
    pitch_h = cell_h - caption_h - 0.03
    fig_w, fig_h = fig.get_size_inches()
    # Pitch width follows its height so the grass keeps its proportions.
    pitch_w = pitch_h * fig_h / fig_w * (PITCH_WIDTH + 10) / (PITCH_LENGTH + 6)
    pitch_w = min(pitch_w, 0.9 / columns - 0.02)
    cell_w = 0.93 / columns

    for index, (row, goal) in enumerate(zip(chains.itertuples(), goals.itertuples())):
        r, c = divmod(index, columns)
        centre = 0.035 + cell_w * (c + 0.5)
        top = area_top - r * cell_h
        color = HOME if int(row.team_id) == HOME_ID else AWAY
        mark = _team_mark_color(int(row.team_id))
        ax = fig.add_axes([centre - pitch_w / 2, top - pitch_h, pitch_w, pitch_h])
        ax.set_facecolor(BG)
        draw_long_pitch(ax)
        ax.axis("off")

        moves, shot = _goal_sequence(annotated, goal._asdict())
        for step, (_, move) in enumerate(moves.iterrows(), start=1):
            sx, sy = attack_xy([move["x"]], [move["y"]])
            ex, ey = attack_xy([move["end_x"]], [move["end_y"]])
            arrow = ax.annotate(
                "",
                xy=(ex[0], ey[0]),
                xytext=(sx[0], sy[0]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=mark,
                    alpha=0.9,
                    lw=1.6,
                    mutation_scale=9,
                    shrinkA=3,
                    shrinkB=3,
                ),
                zorder=4,
            )
            if arrow.arrow_patch is not None:
                arrow.arrow_patch.set_path_effects(
                    [path_effects.Stroke(linewidth=3.2, foreground=BG), path_effects.Normal()]
                )
        # Numbered touches: one per move, at the place it was played from.
        for step, (_, move) in enumerate(moves.iterrows(), start=1):
            sx, sy = attack_xy([move["x"]], [move["y"]])
            ax.scatter(
                sx,
                sy,
                s=190,
                facecolor=BG,
                edgecolor=mark,
                linewidth=1.4,
                zorder=6,
            )
            ax.text(
                sx[0],
                sy[0],
                str(step),
                color=TEXT,
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=7,
            )
        if not shot.empty:
            gx, gy = attack_xy(shot["x"].to_numpy(), shot["y"].to_numpy())
            if len(moves):
                lx, ly = attack_xy([moves.iloc[-1]["end_x"]], [moves.iloc[-1]["end_y"]])
                ax.plot(
                    [lx[0], gx[0]],
                    [ly[0], gy[0]],
                    color=EVENT_HIGHLIGHT,
                    lw=1.6,
                    ls=(0, (3, 2)),
                    zorder=5,
                )
            ax.scatter(
                gx,
                gy,
                s=340,
                marker="*",
                facecolor=EVENT_HIGHLIGHT,
                edgecolor=BG,
                linewidth=1.0,
                zorder=8,
            )

        text_x = centre - pitch_w / 2
        caption_top = top - pitch_h - 0.012
        fig.text(
            text_x,
            caption_top,
            f"{int(row.minute)}′",
            color=color,
            va="top",
            **display(26),
        )
        fig.text(
            text_x + 0.062,
            caption_top - 0.004,
            _surname(row.scorer)[:18],
            color=TEXT,
            va="top",
            **display(18),
        )
        fig.text(
            text_x + 0.062,
            caption_top - 0.034,
            str(row.sequence_type).replace("_", " ").upper(),
            color=MUTED,
            fontsize=7.5,
            fontweight="bold",
            va="top",
        )
        facts = (
            f"{int(row.passes)} passes  ·  {float(row.duration):.0f} s  ·  "
            f"{int(row.players)} players"
        )
        fig.text(text_x, caption_top - 0.066, facts, color=TEXT, fontsize=8.5, va="top")
        fig.text(
            text_x,
            caption_top - 0.088,
            "Started: " + str(row.started_from).replace("_", " "),
            color=MUTED,
            fontsize=8,
            va="top",
        )

    fig.text(
        0.06,
        0.04,
        "Route is read from the possession the goal ended, so a first-time finish from a regain shows as a counter with few passes. Star = the goal.",
        color=NEUTRAL,
        fontsize=7,
    )
    fig.text(
        0.945,
        0.035,
        "FULL VISUAL REDESIGN · REAL MATCH DATA",
        ha="right",
        fontsize=8,
        color=NEUTRAL,
    )
    return save(fig, "46_goal_origins.png")


def unlocking_the_block(events, team_id, opponent_id, number):
    """Receptions in the pocket in front of the opponent's defensive line."""
    pockets = receptions_between_lines(events, team_id, opponent_id)
    switches = switches_of_play(events, team_id)
    tempo = time_to_progress(events, team_id)

    fig, pitch, side = pitch_axes(
        f"Unlocking the Block \u00b7 {TEAM_NAME[team_id]}",
        "Passes received just in front of the opponent's defensive line \u00b7 through a block, not around it",
    )
    draw_long_pitch(pitch)
    team_mark = _team_mark_color(team_id)

    if not pockets.empty:
        px, py = attack_xy(pockets["x"], pockets["y"])
        pitch.scatter(
            px,
            py,
            s=52,
            marker="o",
            facecolors=team_mark,
            edgecolors=BG,
            linewidths=0.9,
            alpha=0.9,
            zorder=6,
        )
        # The estimated line the receptions were taken behind.
        mean_line = float(pockets["line_height"].mean())
        line_y = mean_line * PITCH_LENGTH / 100.0
        pitch.plot(
            [-PITCH_WIDTH / 2, PITCH_WIDTH / 2],
            [line_y, line_y],
            color=EVENT_NEUTRAL,
            lw=1.1,
            ls=(0, (5, 4)),
            alpha=0.7,
            zorder=3,
        )
        pitch.text(
            PITCH_WIDTH / 2,
            line_y + 0.8,
            "avg line",
            color=MUTED,
            fontsize=6.4,
            ha="right",
            va="bottom",
        )
        # Neither mark said what it was: one dot is a single reception, and
        # "avg line" alone does not tell the reader whose line it is.
        pitch_legend(
            pitch,
            [
                ("o", team_mark, "Reception in the pocket"),
                ("_", EVENT_NEUTRAL, f"{TEAM_NAME[opponent_id]} average defensive line"),
            ],
            ncol=2,
        )

    side_title(side, "PLAYING THROUGH")
    side_kpis(
        side,
        [
            ("Receptions in pocket", f"{len(pockets)}"),
            ("Switches of play", f"{len(switches)}"),
            ("Reached final third", f"{tempo['reach_rate']:.0f}%"),
        ],
        start=0.82,
        gap=0.13,
    )

    side.text(0.08, 0.40, "TEMPO", color=MUTED, fontsize=7.5, fontweight="bold")
    side.text(0.08, 0.355, "Median regain to final third", color=TEXT, fontsize=8, va="center")
    side.text(
        0.92,
        0.355,
        f"{tempo['median_seconds']:.1f}s",
        color=TEXT,
        fontsize=8.5,
        fontweight="bold",
        ha="right",
        va="center",
    )

    if not pockets.empty:
        receivers = pockets[pockets["player"] != ""]["player"].value_counts().head(4)
        side.text(
            0.08, 0.28, "MOST OFTEN IN THE POCKET", color=MUTED, fontsize=7.5, fontweight="bold"
        )
        for index, (name, count) in enumerate(receivers.items()):
            y = 0.235 - index * 0.048
            side.text(0.08, y, _surname(name)[:14], color=TEXT, fontsize=8, va="center")
            side.text(
                0.92,
                y,
                str(int(count)),
                color=TEXT,
                fontsize=8.5,
                fontweight="bold",
                ha="right",
                va="center",
            )
    return save(fig, f"{number:02d}_unlocking_{_team_slug(team_id)}.png")


def press_and_rest(events):
    """What the press fed on, and what each side left behind the ball."""
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    base.amoled_header(
        fig,
        "Press Triggers & Rest Defence",
        "Recorded actions preceding high regains and opponent shots within 12 seconds after losses",
    )
    for column, (team_id, team_name, color) in enumerate(
        [(HOME_ID, HOME_NAME, HOME), (AWAY_ID, AWAY_NAME, AWAY)]
    ):
        left = 0.075 + column * 0.475
        fig.text(left, 0.80, team_name.upper(), color=color, fontsize=13, fontweight="bold")

        triggers = pressing_triggers(events, team_id).head(6)
        ax = fig.add_axes([left, 0.40, 0.39, 0.33])
        base.clean_ax(ax)
        if triggers.empty:
            ax.text(0.5, 0.5, "No high regains", color=MUTED, ha="center", va="center")
        else:
            positions = np.arange(len(triggers))
            counts = [int(v) for v in triggers["regains"]]
            ax.barh(positions, counts, height=0.6, color=color, alpha=0.9)
            ax.set_yticks(positions)
            ax.set_yticklabels(triggers["trigger"], fontsize=8.5, color=TEXT)
            ax.invert_yaxis()
            ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
            ax.set_xlabel("High regains", fontsize=8.5, color=MUTED)
            ax.tick_params(labelsize=8)

            # Room for the value that sits at the end of the bar. Without it a
            # longest bar reaches the axis edge and its label is drawn past it,
            # over the other side's category names — which is exactly what
            # happened whenever a press produced only two or three regains.
            top = max(counts + [1])
            ax.set_xlim(0, top * 1.42)
            # These are counts of regains: a tick at 0.25 of one does not exist.
            ax.set_xticks(range(0, top + 1, max(1, top // 5)))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))

            for index, (count, share) in enumerate(zip(counts, triggers["share"])):
                ax.text(
                    count + top * 0.045,
                    index,
                    f"{count}  ({share:.0f}%)",
                    color=TEXT,
                    fontsize=7.8,
                    va="center",
                    fontweight="bold",
                )

        losses = turnover_events(events, team_id)
        second = second_ball_recovery(events, team_id)
        rows = [
            ("Open-play losses", f"{len(losses)}"),
            ("Opponent shot within 12 seconds", f"{int(losses['punished'].sum())}/{len(losses)}"),
            (
                "Second balls won",
                f"{second['won']}/{second['contests']} ({second['win_rate']:.0f}%)",
            ),
        ]
        fig.text(
            left, 0.30, "REST DEFENCE & SECOND BALLS", color=MUTED, fontsize=7.8, fontweight="bold"
        )
        for index, (label, value) in enumerate(rows):
            y = 0.255 - index * 0.045
            fig.text(left, y, label, color=TEXT, fontsize=8.4)
            fig.text(left + 0.39, y, value, color=TEXT, fontsize=8.6, fontweight="bold", ha="right")

    fig.text(
        0.945,
        0.035,
        "FULL VISUAL REDESIGN \u00b7 REAL MATCH DATA",
        ha="right",
        fontsize=8,
        color=NEUTRAL,
    )
    return save(fig, "49_press_triggers.png")


def non_pitch_pages(events, xg, team_metrics):
    """Render the chart-only pages that live in visual_redesign_preview.

    The shot-profile, ball-touches and advanced-metrics pages used to be built
    here too. All three repeated rows of the post-match dashboard in a
    different arrangement, so they were dropped and their unique rows folded
    into that one page.
    """
    base.OUT_DIR = OUT
    sources = {
        "01_xg_flow.png": base.xg_flow(events),
        "15_xt_per_minute.png": base.xt_per_minute(events),
    }
    result = {}
    for filename, src in sources.items():
        dst = OUT / filename
        if src != dst:
            shutil.copy2(src, dst)
            src.unlink(missing_ok=True)
        result[filename] = dst
    return result


def build_pdf(
    paths: list[Path],
    events: pd.DataFrame | None = None,
    xg: pd.DataFrame | None = None,
    team_metrics: pd.DataFrame | None = None,
    player_metrics: pd.DataFrame | None = None,
):
    """Build one connected tactical report followed by the full player appendix."""
    if any(value is None for value in [events, xg, team_metrics, player_metrics]):
        loaded_events, _players, loaded_xg, loaded_team_metrics, loaded_player_metrics = load_all()
        events = loaded_events if events is None else events
        xg = loaded_xg if xg is None else xg
        team_metrics = loaded_team_metrics if team_metrics is None else team_metrics
        player_metrics = loaded_player_metrics if player_metrics is None else player_metrics

    from football_analysis.reports.tactical_pdf_report import build_tactical_pdf

    return build_tactical_pdf(
        paths,
        OUT / "full_visual_redesign_real_data.pdf",
        events,
        xg,
        team_metrics,
        player_metrics,
        {
            "home_id": HOME_ID,
            "away_id": AWAY_ID,
            "home_name": HOME_NAME,
            "away_name": AWAY_NAME,
            "home_color": HOME,
            "away_color": AWAY,
            "score": MATCH_SCORE,
            "date": events.attrs.get("match_date", ""),
            # What the cover names beyond the scoreline. Read with defaults, so
            # a renderer configured before these existed still builds a report.
            **{
                key: globals().get("_MATCH_INFO", {}).get(key)
                for key in (
                    "venue",
                    "managers",
                    "formations",
                    "round_name",
                    "home_form",
                    "away_form",
                )
            },
            "home_kit": (globals().get("_KIT_COLORS") or (None, None))[0],
            "away_kit": (globals().get("_KIT_COLORS") or (None, None))[1],
        },
    )


def build_catalog(paths: list[Path]):
    rows = []
    for order, path in enumerate(paths, start=1):
        stem = path.stem
        prefix, separator, remainder = stem.partition("_")
        if separator and re.fullmatch(r"\d+[a-z]?", prefix, flags=re.IGNORECASE):
            number = prefix
            title_source = remainder
        else:
            # Player radar exports use the player's display name directly
            # (for example ``Pedri.png``), so they have no numbered prefix.
            # Give them a stable catalogue reference without assuming that
            # every filename contains a separator.
            number = f"P{order:02d}"
            title_source = stem
        try:
            file_name = path.relative_to(OUT.resolve()).as_posix()
        except ValueError:
            file_name = path.name
        rows.append(
            {
                "_order": order,
                "number": number,
                "title": title_source.replace("_", " ").replace("-", " ").title(),
                "file": file_name,
                "has_pitch": any(
                    token in stem
                    for token in [
                        "shot_map",
                        "pass_network",
                        "pass_map",
                        "xt_map",
                        "danger",
                        "zone14",
                        "progressive",
                        "crosses",
                        "defensive_activity",
                        "average_positions",
                        "dominating",
                        "box_entries",
                        "high_regains",
                        "pass_targets",
                        "transition_outcomes",
                    ]
                ),
            }
        )
    catalog = pd.DataFrame(rows).sort_values("_order").drop(columns="_order")
    catalog.to_csv(OUT / "visual_catalog.csv", index=False, encoding="utf-8-sig")
    return catalog


# How many player radars the report carries per side. Every participant is
# still exported to the team folders; this is only what reaches the PDF.
PDF_RADARS_PER_TEAM = 5


def player_pizzas(events: pd.DataFrame, players: pd.DataFrame | None = None) -> list[Path]:
    """Export every player's radar, and return the five per side for the report."""
    from football_analysis.visuals.player_radar import export_player_radars

    info = {
        "home_id": HOME_ID,
        "away_id": AWAY_ID,
        "home_name": HOME_NAME,
        "away_name": AWAY_NAME,
        "home_color": HOME,
        "away_color": AWAY,
        "score": MATCH_SCORE,
    }
    # The squad is handed down rather than left to be found on disk. The radar
    # falls back to reading players.csv out of the folder it writes into, which
    # the dark package has and the light one did not, so every light radar lost
    # its position and printed the player's substitution role instead — "sub_out"
    # where the black copy of the same player said "Defensive midfielder".
    ranking = export_player_radars(events, info, str(OUT), dpi=135, squad=players)

    # Every participant's radar is written to disk — the folders are the
    # reference and the article picks its own three a side from them. The
    # report carries the five that mattered most per team instead of all
    # thirty-odd, which was half the document.
    source_root = OUT / "player_radars"
    exported = []
    for side, team in (("home", HOME_NAME), ("away", AWAY_NAME)):
        team_dir = source_root / team.replace(" ", "_")
        if not team_dir.exists():
            team_dir = source_root / team
        if not team_dir.exists():
            continue
        best = [name for name, _rating in (ranking or {}).get(side, [])][:PDF_RADARS_PER_TEAM]
        for name in best:
            candidate = team_dir / (str(name).replace(" ", "_") + ".png")
            if candidate.exists():
                exported.append(candidate)
                continue
            wanted = str(name).replace(" ", "_").lower()
            match = next((f for f in team_dir.glob("*.png") if f.stem.lower() == wanted), None)
            if match is not None:
                exported.append(match)
    return exported


def _corrected_xgot(events: pd.DataFrame, xg: pd.DataFrame, match_info: dict) -> pd.DataFrame:
    """Recompute xGoT from the events, so the package agrees with itself.

    xGoT used to be the sum of *pre-shot* xG over on-target attempts, which
    knows nothing about where the ball went. The collector now prices it from
    the placement, but a fixture parsed before that still carries the old
    number on disk — and the player radars, which read the events directly,
    would then disagree with the team card in the same document.

    Recomputing here means an old export and a new one produce the same
    package. Only touched when the events carry the placement the model needs.
    """
    if events is None or events.empty or xg is None or xg.empty:
        return xg
    if not {"goal_mouth_y", "goal_mouth_z"}.issubset(events.columns):
        return xg
    try:
        from football_analysis.metrics.match_metrics import team_post_shot_xg
    except Exception:
        return xg

    corrected = xg.copy()
    for side in ("home", "away"):
        try:
            team_id = int(match_info[f"{side}_id"])
        except (KeyError, TypeError, ValueError):
            continue
        name = str(match_info.get(f"{side}_name") or "")
        rows = corrected.index[corrected["team"].astype(str).str.lower().eq(name.lower())]
        if not len(rows):
            continue
        corrected.loc[rows, "xGoT"] = team_post_shot_xg(events, team_id)
    return corrected


from football_analysis.pipeline.package_io import transactional_package


@transactional_package
def generate_match_package(
    events: pd.DataFrame,
    players: pd.DataFrame,
    xg: pd.DataFrame,
    team_metrics: pd.DataFrame,
    player_metrics: pd.DataFrame,
    match_info: dict,
    output_dir: Path | str,
    *,
    clean: bool = True,
) -> dict:
    """Generate the production AMOLED package for any parsed fixture."""
    configure_match(match_info, output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    if clean:
        for pattern in ("*.png", "*.pdf"):
            for old in OUT.glob(pattern):
                old.unlink(missing_ok=True)
        for generated_dir in (OUT / "player_radars", OUT / "comparisons"):
            if generated_dir.exists():
                shutil.rmtree(generated_dir)
    base.theme()
    xg = _corrected_xgot(events, xg, match_info)
    # Persist it beside the other frames. The documents were built from the
    # corrected value while xg.csv kept the old one, so anything re-reading the
    # folder — the tests, a second render, the article — disagreed with the PDF
    # it sat next to.
    xg.to_csv(OUT / "xg.csv", index=False, encoding="utf-8-sig")
    team_metrics.to_csv(OUT / "team_advanced_metrics.csv", index=False, encoding="utf-8-sig")
    player_metrics.to_csv(OUT / "player_sequence_metrics.csv", index=False, encoding="utf-8-sig")
    # The squad too. It is what carries each player's position, and anything
    # reading the folder back — the radars first among them — has no other
    # source for it. The light package went without one and its radars named
    # every substitute "sub_out".
    players.to_csv(OUT / "players.csv", index=False, encoding="utf-8-sig")
    # The fixture's identity, next to the frames it describes. Without it the
    # output folder held every number about the match and no record of which
    # match it was, so nothing downstream could re-render from it.
    (OUT / "match_info.json").write_text(
        json.dumps(match_info, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    generated = non_pitch_pages(events, xg, team_metrics)
    plt.close("all")
    gc.collect()
    paths = [
        generated["01_xg_flow.png"],
        match_statistics(events, match_info),
        shot_map(events, xg, HOME_ID, 2),
        shot_map(events, xg, AWAY_ID, 3),
        pass_network(events, players, HOME_ID, 5, 1),
        pass_network(events, players, HOME_ID, 5, 2),
        pass_network(events, players, AWAY_ID, 6, 1),
        pass_network(events, players, AWAY_ID, 6, 2),
        xt_map(events, HOME_ID, 7),
        xt_map(events, AWAY_ID, 8),
        pass_map(events, HOME_ID, 9),
        pass_map(events, AWAY_ID, 10),
        gk_saves(events, xg, players),
        zone14(events, HOME_ID, 12),
        zone14(events, AWAY_ID, 13),
        generated["15_xt_per_minute.png"],
        progressive(events, HOME_ID, 16),
        progressive(events, AWAY_ID, 17),
        # Crosses were dropped from the package. Nine deliveries drew two
        # arrows on an otherwise empty pitch, and the crossing totals that
        # carried the finding already sit on the poster's delivery panel.
        defensive_activity(events, HOME_ID, 20),
        defensive_activity(events, AWAY_ID, 21),
        dominating_zones(events),
        attacking_zones(events),
        box_entries(events, HOME_ID, 25),
        box_entries(events, AWAY_ID, 26),
        high_regains(events, HOME_ID, 27),
        high_regains(events, AWAY_ID, 28),
        pass_targets(events, HOME_ID, 29),
        pass_targets(events, AWAY_ID, 30),
        ppda(events),
        press_profile(events, xg, team_metrics),
        transition_outcomes(events),
        game_state(events, team_metrics),
        player_sequence(player_metrics),
        momentum(events),
        set_pieces(events),
        pass_sonar(events),
        finishing_quality(events),
        shape_over_time(events),
        playing_through(events, HOME_ID, AWAY_ID, 41),
        playing_through(events, AWAY_ID, HOME_ID, 42),
        action_value_leaders(events),
        control_surface(events, players),
        sequence_types(events),
        goal_origins(events),
        press_and_rest(events),
    ]
    # Role profiles supersede the old pizza/radar export in new packages.
    from football_analysis.visuals.insight_visuals import build_insight_visuals

    new_paths, _insights = build_insight_visuals(events, players, match_info, OUT)
    paths.extend(new_paths)
    events.attrs["match_date"] = match_info.get("date", "")
    paths = sorted({path.resolve() for path in paths}, key=lambda path: path.name)
    from football_analysis.visuals.visual_numbering import number_visuals

    paths = number_visuals(paths, OUT, events.attrs.get("chart_contracts", {}))
    catalog = build_catalog(paths)
    pdf = build_pdf(paths, events, xg, team_metrics, player_metrics)
    from football_analysis.visuals.poster_dashboard import build_match_posters

    posters = build_match_posters(
        events,
        xg,
        team_metrics,
        player_metrics,
        players,
        out_dir=OUT,
        home_id=HOME_ID,
        away_id=AWAY_ID,
        home_name=HOME_NAME,
        away_name=AWAY_NAME,
        home_color=HOME,
        away_color=AWAY,
        score=MATCH_SCORE.replace("-", "—"),
        competition=str(match_info.get("competition") or "MATCH ANALYSIS"),
        match_date=str(match_info.get("date") or ""),
    )

    # The publishable read, beside the reference report. Never fatal: a package
    # that cannot write its article still has every visual and the PDF.
    from football_analysis.prose.match_article import build_match_article

    article = build_match_article(
        events, xg, team_metrics, player_metrics, match_info, OUT, players
    )

    print(f"Generated {len(paths)} full redesigned visuals")
    print(f"Pitch visuals: {int(catalog['has_pitch'].sum())}")
    print(f"Posters: {len(posters)}")
    print(f"Article: {article}" if article else "Article: not written")
    print(f"PDF: {pdf}")
    result = {
        "visuals": paths,
        "catalog": OUT / "visual_catalog.csv",
        "pdf": pdf,
        "posters": posters,
        "article": article,
        "output_dir": OUT,
    }

    # The light copy belongs here, not in the collector that used to call it.
    # Re-rendering a fixture from its saved frames only rebuilt the black
    # package, so a corrected report could sit next to a light one carrying
    # numbers fixed weeks earlier, and nothing said so.
    #
    # The theme is fixed when visualization_components is first imported, so
    # this runs in a child process. That child is given LIGHT_COPY=0, which is
    # what stops this from recursing. A failure here never costs the package
    # that is already written.
    # Normal renders always publish both themes.  The light child sets this to
    # 0 so it does not recursively launch another child process.
    if os.environ.get("MATCH_ANALYSIS_LIGHT_COPY", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        # Release the dark render before the light one starts.
        #
        # The light copy is a second full render in a child process, and it was
        # being launched while this process still held every figure it had just
        # drawn -- fifty-one boards, four posters and a profile card per player.
        # On a machine without several spare gigabytes the child could not
        # allocate its first canvas and died inside matplotlib's Agg backend:
        #
        #     MemoryError: bad allocation
        #       backend_agg.py, in __init__
        #       self._renderer = _RendererAgg(int(width), int(height), dpi)
        #
        # The package was already written by then, so the failure looked like
        # "the light copy does not generate" rather than like running out of
        # memory. Nothing below needs the figures, so they are dropped first.
        # gc and plt are module-level imports. Importing them again here made
        # both names local to the whole function, and the plt.close/gc.collect
        # pair a hundred lines above -- which had been running on the module
        # ones -- died on UnboundLocalError before the boards were drawn.
        plt.close("all")
        gc.collect()
        try:
            from football_analysis.render.render_light import render_light_package

            light_out = render_light_package(OUT)
            if light_out is not None:
                result["light_output_dir"] = light_out
                print(f"Light package: {light_out}")
        except Exception as error:
            print(f"  ! light package not written — {type(error).__name__}: {error}")

    return result


def _sample_kit_colors() -> tuple[str, str]:
    """Resolve the sample fixture's kit colours the same way production does.

    Imported lazily: football_match_analysis imports this module, so a
    top-level import here would be circular.
    """
    if not USE_REAL_TEAM_KIT_COLORS:
        return HOME, AWAY
    try:
        from football_analysis.pipeline.football_match_analysis import choose_matchup_colors

        return choose_matchup_colors(HOME_NAME, AWAY_NAME)
    except Exception:
        return HOME, AWAY


def main():
    events, players, xg, team_metrics, player_metrics = load_all()
    home_color, away_color = _sample_kit_colors()
    return generate_match_package(
        events,
        players,
        xg,
        team_metrics,
        player_metrics,
        {
            "home_id": HOME_ID,
            "away_id": AWAY_ID,
            "home_name": HOME_NAME,
            "away_name": AWAY_NAME,
            "home_color": home_color,
            "away_color": away_color,
            "score": MATCH_SCORE,
        },
        OUT,
    )


if __name__ == "__main__":
    main()
