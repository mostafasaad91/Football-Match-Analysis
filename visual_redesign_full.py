from __future__ import annotations

import colorsys
import gc
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Patch, Rectangle, Wedge
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from visualization_components import (
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

import visual_redesign_preview as base
from match_metrics import (
    advanced_metrics_frames,
    defensive_line_height,
    duel_map,
    goalkeeper_distribution,
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
from match_report import compute_ppda_both


ROOT = Path(__file__).resolve().parent
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
_HATCH_ALPHA = 0.20 if IS_LIGHT_THEME else 0.13

HOME_ID = base.HOME_ID
AWAY_ID = base.AWAY_ID
HOME_NAME = base.HOME_NAME
AWAY_NAME = base.AWAY_NAME
TEAM_COLOR = base.TEAM_COLOR
TEAM_NAME = base.TEAM_NAME
MATCH_SCORE = "4-6"

PITCH_LENGTH = 105.0
PITCH_WIDTH = 58.0

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
    HOME, AWAY = (lift_to_floor(colour) for colour in _resolve_fixture_colors(match_info))
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
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    display_x = (y_arr - 50.0) * (PITCH_WIDTH / 100.0)
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
    active_team = HOME_NAME if HOME_NAME.lower() in title.lower() else (AWAY_NAME if AWAY_NAME.lower() in title.lower() else None)
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
    fig.text(0.945, 0.035, "FULL VISUAL REDESIGN · REAL MATCH DATA", ha="right", fontsize=8, color=NEUTRAL)
    return fig, pitch, side


def draw_long_pitch(ax, line_color=PITCH_LINE, lw=1.15):
    half_w = PITCH_WIDTH / 2
    ax.set_xlim(-half_w - 5, half_w + 5)
    ax.set_ylim(-3, PITCH_LENGTH + 3)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((-half_w, 0), PITCH_WIDTH, PITCH_LENGTH, fill=False, ec=line_color, lw=lw))
    ax.plot([-half_w, half_w], [PITCH_LENGTH / 2, PITCH_LENGTH / 2], color=line_color, lw=lw)
    ax.add_patch(Circle((0, PITCH_LENGTH / 2), PITCH_LENGTH * 0.0915, fill=False, ec=line_color, lw=lw))
    penalty_w = PITCH_WIDTH * 0.595
    six_w = PITCH_WIDTH * 0.265
    box_l = PITCH_LENGTH * 0.157
    six_l = PITCH_LENGTH * 0.052
    for y0, direction in [(0, 1), (PITCH_LENGTH, -1)]:
        ax.add_patch(Rectangle((-penalty_w / 2, y0 if direction > 0 else y0 - box_l), penalty_w, box_l, fill=False, ec=line_color, lw=lw))
        ax.add_patch(Rectangle((-six_w / 2, y0 if direction > 0 else y0 - six_l), six_w, six_l, fill=False, ec=line_color, lw=lw))
        spot_y = y0 + direction * PITCH_LENGTH * 0.105
        ax.scatter([0], [spot_y], s=7, color=line_color)
        arc_center = y0 + direction * box_l
        if direction > 0:
            ax.add_patch(Arc((0, spot_y), 18.3, 18.3, theta1=37, theta2=143, ec=line_color, lw=lw))
        else:
            ax.add_patch(Arc((0, spot_y), 18.3, 18.3, theta1=217, theta2=323, ec=line_color, lw=lw))
    ax.annotate("ATTACK", xy=(0, PITCH_LENGTH + 1.5), ha="center", va="bottom", color=FOCUS, fontsize=8, fontweight="bold")
    ax.axis("off")


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
        ax.text(0.08, y - 0.055, str(value), color=TEXT, fontsize=16, fontweight="bold", va="top")
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
            handles.append(Patch(facecolor=colour, edgecolor=EVENT_NEUTRAL,
                                 linewidth=0.7, label=label))
        else:
            handles.append(Line2D([], [], linestyle="none", marker=kind,
                                  markerfacecolor=colour, markeredgecolor=BG,
                                  markeredgewidth=0.9, markersize=7, label=label))
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
        ax.text(0.92, y, str(value), color=value_color, fontsize=8.5, fontweight="bold", ha="right", va="center")
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
        title = title_item.get_text() if title_item is not None else filename.rsplit(".", 1)[0].replace("_", " ").title()
        subtitle_items = [item for item in candidates if item is not title_item and item.get_text().strip()]
        subtitle = subtitle_items[0].get_text() if subtitle_items else f"{HOME_NAME} vs {AWAY_NAME} · real match data"
        for item in candidates:
            item.set_visible(False)
        active_team = HOME_NAME if HOME_NAME.lower() in title.lower() else (AWAY_NAME if AWAY_NAME.lower() in title.lower() else None)
        base.amoled_header(fig, title, subtitle, active_team=active_team)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
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
    surname = str(name).strip().split()[-1] if str(name).strip() else "?"
    return surname if len(surname) <= limit else f"{surname[:limit - 1]}…"


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


def draw_node_label(ax, x: float, y: float, name: str, touches: float, max_touch: float,
                    node_color: str | None = None, shirt: str | None = None,
                    node_radius: float = 2.6,
                    neighbours: "tuple[tuple[float, float, float], ...]" = ()):
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
        number = ax.text(x, y, str(shirt), color=inside, fontsize=6.6 if ratio >= 0.25 else 5.8,
                         fontweight="bold", ha="center", va="center", zorder=7, clip_on=True)
        number.set_path_effects([path_effects.withStroke(linewidth=1.6, foreground=fill, alpha=0.6)])

    label = compact_player_label(name) if not shirt else str(name).strip().split()[-1][:12]

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
        x + dx, y + dy, label, color=TEXT, fontsize=5.9, fontweight="bold",
        ha=ha, va=va, zorder=7, clip_on=True,
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


def _node_neighbours(display: dict[str, tuple[float, float, float]],
                     radii: dict[str, float], exclude: str):
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


def shot_map(events, xg, team_id, number):
    pso = as_bool(events.get("is_penalty_shootout", pd.Series(False, index=events.index)))
    shots = events[events["team_id"].eq(team_id) & as_bool(events["is_shot"]) & ~pso].copy().dropna(subset=["x", "y"])
    shots["xG"] = pd.to_numeric(shots["xG"], errors="coerce").fillna(0).clip(lower=0)
    fig, pitch, side = pitch_axes(f"Shot Map · {TEAM_NAME[team_id]}", "Shot location, outcome and chance quality · marker size = xG")
    draw_long_pitch(pitch)
    # Colour carries the outcome (shared shot palette, one key across the whole
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
    for event_type, marker, color, label, scale in markers:
        subset = shots[shots["type"].astype(str).eq(event_type)]
        if subset.empty:
            continue
        px, py = attack_xy(subset["x"], subset["y"])
        sizes = (45 + subset["xG"].to_numpy() * 520) * scale
        pitch.scatter(px, py, s=sizes, marker=marker, facecolors=color, edgecolors=BG, linewidths=1.0, alpha=0.95, label=f"{label} ({len(subset)})", zorder=5 if event_type == "Goal" else 4)
    pitch.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, labelcolor=TEXT, fontsize=7.5)
    xr = xg_row(xg, TEAM_NAME[team_id])
    side_title(side, "SHOT OUTPUT")
    kpi_bottom = side_kpis(side, [("Shots", f"{len(shots)}"), ("xG", f"{float(xr.get('xG', 0)):.2f}"), ("xG / shot", f"{float(xr.get('xG_per_shot', 0)):.3f}"), ("On target", f"{int(float(xr.get('on_target', 0)))}")])

    # Shot location says where the chance came from; this says which part of
    # the goal the keeper actually had to cover. Anchored under the KPI block
    # rather than at a fixed 0.30, which the fourth KPI's value ran into.
    zones = shot_placement_zones(events, team_id)
    if sum(zones.values()):
        heading_y = kpi_bottom - 0.03
        side.text(0.08, heading_y, "GOAL FRAME TARGETED", color=MUTED, fontsize=7.5, fontweight="bold")
        ranked = [item for item in sorted(zones.items(), key=lambda pair: -pair[1]) if item[1]][:3]
        for idx, (zone, count) in enumerate(ranked):
            y = heading_y - 0.05 - idx * 0.055
            side.text(0.08, y, zone.replace("_", " ").title(), color=TEXT, fontsize=8, va="center")
            side.text(0.92, y, str(count), color=TEXT, fontsize=8.5, fontweight="bold",
                      ha="right", va="center")
    return save(fig, f"{number:02d}_shot_map_{_team_slug(team_id)}.png")


def goals_breakdown(events):
    pso = as_bool(events.get("is_penalty_shootout", pd.Series(False, index=events.index)))
    goals = events[as_bool(events["is_goal"]) & ~pso].copy()
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.text(0.055, 0.94, "Goal Breakdown", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.898, "Scoring timeline and goal profile · penalty shootout excluded", fontsize=11, color=MUTED)
    fig.add_artist(Line2D([0.055, 0.945], [0.86, 0.86], transform=fig.transFigure, color=GRID, lw=1))
    ax = fig.add_axes([0.08, 0.18, 0.84, 0.56])
    base.clean_ax(ax)
    max_min = max(90, int(pd.to_numeric(events["minute"], errors="coerce").max()) + 2)
    ax.set_xlim(0, max_min)
    ax.set_ylim(-1.4, 1.4)
    ax.axhline(0, color=GRID, lw=1.5)
    ax.set_yticks([0.75, -0.75]); ax.set_yticklabels([HOME_NAME, AWAY_NAME], fontsize=11)
    ax.set_xlabel("Match minute")
    ordered = events.sort_values(["minute", "second", "event_id"], kind="stable").copy()
    ordered["_clock"] = (
        pd.to_numeric(ordered["minute"], errors="coerce").fillna(0) * 60
        + pd.to_numeric(ordered["second"], errors="coerce").fillna(0)
    )

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
        player = str(goal.get("player", "Goal")).split()[-1]
        is_own = bool(as_bool(pd.Series([goal.get("is_own_goal", False)])).iloc[0])
        assist = assist_for(goal)
        assist_label = (
            "OWN GOAL" if is_own
            else ("UNASSISTED" if assist == "UNASSISTED" else f"ASSIST · {assist.split()[-1]}")
        )
        label = f"{int(minute)}′  {player}\n{assist_label}"
        horizontal_nudge = -10 if team_goal_count[tid] % 2 else 10
        ax.annotate(
            label, (minute, y),
            xytext=(horizontal_nudge, 18 if y > 0 else -18), textcoords="offset points",
            ha="center", va="bottom" if y > 0 else "top", color=TEXT, fontsize=7.2,
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
    work["clock"] = pd.to_numeric(work["minute"], errors="coerce").fillna(0) * 60 + pd.to_numeric(work["second"], errors="coerce").fillna(0)
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
    sub_events = team_frame[team_frame["type"].astype(str).isin(["SubstitutionOn", "SubstitutionOff"])].copy()
    log_sub_events = sub_events.copy()
    sub_on = set(sub_events[sub_events["type"].astype(str).eq("SubstitutionOn")]["player"].dropna().astype(str))
    sub_off = set(sub_events[sub_events["type"].astype(str).eq("SubstitutionOff")]["player"].dropna().astype(str))
    if half == 1:
        interval_events = events[
            events["team_id"].eq(team_id)
            & events["period_code"].astype(str).str.lower().eq("2h")
            & events["type"].astype(str).isin(["SubstitutionOn", "SubstitutionOff"])
            & (pd.to_numeric(events["minute"], errors="coerce").fillna(999) <= 45)
        ].copy()
        interval_off = interval_events[interval_events["type"].astype(str).eq("SubstitutionOff")]["player"]
        sub_off |= set(interval_off.dropna().astype(str))
        # The player introduced at the interval belongs to the second-half
        # visual only. Keep the outgoing starter marked, but do not name the
        # incoming player anywhere on the first-half card.
        log_sub_events = pd.concat(
            [sub_events, interval_events[interval_events["type"].astype(str).eq("SubstitutionOff")]],
            ignore_index=False,
        )
    participants = set(touches["player"].dropna().astype(str)) | sub_on
    if half == 1:
        starters = players[players["team_id"].eq(team_id) & players["is_first_xi"].astype(str).str.lower().isin(["true", "1", "yes"])]["name"]
        participants |= set(starters.dropna().astype(str))

    player_info = players[players["team_id"].eq(team_id)].set_index("name")
    all_touches = events[touch_mask(events) & events["team_id"].eq(team_id)].dropna(subset=["player", "x", "y"])
    position_rows = []
    for name in sorted(participants):
        player_touches = touches[touches["player"].astype(str).eq(name)]
        coords = player_touches[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna()
        if coords.empty:
            event_coords = team_frame[team_frame["player"].astype(str).eq(name)][["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna()
            coords = event_coords
        if coords.empty:
            whole_coords = all_touches[all_touches["player"].astype(str).eq(name)][["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna()
            coords = whole_coords
        if coords.empty:
            role = player_info.at[name, "position"] if name in player_info.index else ""
            x, y = _role_fallback_position(role)
        else:
            x, y = float(coords["x"].mean()), float(coords["y"].mean())
        position_rows.append({"player": name, "x": x, "y": y, "touches": int(len(player_touches))})
    positions = pd.DataFrame(position_rows).set_index("player") if position_rows else pd.DataFrame(columns=["x", "y", "touches"])
    names = set(positions.index.astype(str))
    passes = passes[passes["player"].astype(str).isin(names) & passes["next_player"].astype(str).isin(names)]
    edges = passes.groupby(["player", "next_player"]).size().reset_index(name="passes").sort_values("passes", ascending=False).head(22)

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
            match_idx = next((idx for idx, item in enumerate(pending_off) if item[0] == minute), None)
            off_name = pending_off.pop(match_idx)[1] if match_idx is not None else "—"
            substitutions.append((minute, name, off_name))
    for minute, off_name in pending_off:
        substitutions.append((minute, "—", off_name))
    return positions, edges, sub_on, sub_off, substitutions, int(len(passes))


def pass_network(events, players, team_id, number, half):
    positions, edges, sub_on, sub_off, substitutions, completed_links = _half_network_data(events, players, team_id, half)
    half_label = "First Half" if half == 1 else "Second Half"
    fig, pitch, side = pitch_axes(
        f"Pass Network · {TEAM_NAME[team_id]} · {half_label}",
        f"All {len(positions)} participants shown · node size = touches · square = came on · {_FOCUS_WORD} outline = went off",
    )
    draw_long_pitch(pitch)
    display = {}
    for name, row in positions.iterrows():
        px, py = player_position_xy([row["x"]], [row["y"]])
        display[str(name)] = (float(px[0]), float(py[0]), float(row["touches"]))
    display = _separate_network_positions(display, min_gap=6.3)
    _link_low, link_color, _link_strong = network_link_palette(TEAM_COLOR[team_id])
    max_edge = max(float(edges["passes"].max()) if not edges.empty else 1, 1)
    for _, edge in edges.iterrows():
        a, b = str(edge["player"]), str(edge["next_player"])
        if a not in display or b not in display:
            continue
        ax, ay, _ = display[a]; bx, by, _ = display[b]
        pitch.plot([ax, bx], [ay, by], color=link_color, alpha=0.52,
                   lw=0.75 + 4.4 * float(edge["passes"]) / max_edge, zorder=2)
    max_touch = max([value[2] for value in display.values()] or [1])
    shirts = shirt_number_map(players)
    radii = {name: _network_node_radius(touches, max_touch)
             for name, (_x, _y, touches) in display.items()}
    for name, (px, py, touches) in display.items():
        entered = name in sub_on
        left = name in sub_off
        pitch.scatter(px, py, s=260 + 640 * touches / max_touch, marker="s" if entered else "o",
                      color=_team_mark_color(team_id), edgecolor=FOCUS if left else link_color,
                      linewidth=2.3 if left else 1.15, zorder=4)
        draw_node_label(pitch, px, py, name, touches, max_touch,
                        node_color=_team_mark_color(team_id),
                        shirt=shirts.get(str(name)), node_radius=radii[name],
                        neighbours=_node_neighbours(display, radii, name))

    side_title(side, "TOP HALF CONNECTIONS")
    side.text(0.92, 0.94, f"{len(positions)} players", color=TEXT, fontsize=8,
              fontweight="bold", ha="right", va="top")
    # Four link rows rather than five: the fifth was the weakest pair anyway,
    # and the space now carries the centrality read instead.
    # Two names share this row, so each gets less room than a single-name row
    # further down — but still far more than the pitch-side default.
    side_rows(side, [(f"{compact_player_label(r.player, 11)} → {compact_player_label(r.next_player, 11)}", str(int(r.passes))) for r in edges.head(4).itertuples()], start=0.81, gap=0.075)

    # Link volume names the busiest pair. Betweenness names the player the
    # network routes through — take them out and it splits in two.
    #
    # Scoped to this half, like everything else on the page. Run over the whole
    # match it lists players who were not on the pitch for the half being drawn.
    half_events = events[
        events["period_code"].astype(str).str.lower().eq("1h" if half == 1 else "2h")
    ]
    centrality = network_centrality(half_events, team_id)
    if not centrality.empty:
        side.text(0.08, 0.520, "CONNECTORS", color=MUTED, fontsize=7.5, fontweight="bold")
        for idx, row in enumerate(centrality.head(3).itertuples()):
            y = 0.472 - idx * 0.043
            side.text(0.08, y, compact_player_label(row.player, 16), color=TEXT, fontsize=8, va="center")
            side.text(0.92, y, f"{row.betweenness:.3f}", color=TEXT, fontsize=8.5,
                      fontweight="bold", ha="right", va="center")

    side.text(0.08, 0.325, "SUBSTITUTIONS", color=MUTED, fontsize=7.5, fontweight="bold")
    if substitutions:
        # A fixed 0.042 step fits four rows above the footer and puts a fifth
        # exactly on top of it. Tighten the step only when the extra row needs
        # it, so the common case keeps its existing spacing.
        shown = substitutions[:5]
        top, floor = 0.278, 0.150
        gap = 0.042 if len(shown) < 2 else min(0.042, (top - floor) / (len(shown) - 1))
        for idx, (minute, on_name, off_name) in enumerate(shown):
            y = top - idx * gap
            side.text(0.08, y, f"{minute}′", color=TEXT, fontsize=7.5, fontweight="bold", va="center")
            change = f"{off_name} OFF AT INTERVAL" if on_name == "—" else f"{on_name} IN  ·  {off_name} OFF"
            side.text(0.19, y, change, color=TEXT, fontsize=7.2, va="center")
    else:
        side.text(0.08, 0.278, "No in-half changes", color=MUTED, fontsize=8)
    side.text(0.08, 0.105, f"Completed pass links: {completed_links}", color=TEXT, fontsize=8, fontweight="bold")
    # "Began half" is wider than the old 0.19 gap between markers, so it ran
    # under the next swatch. Spaced to the widest label rather than to an
    # eyeballed step.
    side.scatter([0.06, 0.34], [0.055, 0.055], s=[65, 65], marker="o", color=_team_mark_color(team_id), edgecolor=[TEXT, FOCUS], linewidth=[1.0, 2.1])
    side.scatter([0.60], [0.055], s=65, marker="s", color=_team_mark_color(team_id), edgecolor=TEXT, linewidth=1.0)
    side.text(0.10, 0.055, "Began half", color=TEXT, fontsize=6.8, va="center")
    side.text(0.38, 0.055, "Went off", color=TEXT, fontsize=6.8, va="center")
    side.text(0.64, 0.055, "Came on", color=TEXT, fontsize=6.8, va="center")
    suffix = "1h" if half == 1 else "2h"
    return save(fig, f"{number:02d}{'a' if half == 1 else 'b'}_pass_network_{_team_slug(team_id)}_{suffix}.png")


def xt_map(events, team_id, number):
    team = events[events["team_id"].eq(team_id) & events["type"].astype(str).eq("Pass")].copy().dropna(subset=["x", "y", "end_x", "end_y"])
    team["xT"] = pd.to_numeric(team["xT"], errors="coerce").fillna(0).clip(lower=0)
    heat, _, _ = np.histogram2d(
        team["y"], team["x"], bins=[7, 12], range=[[0, 100], [0, 100]], weights=team["xT"]
    )
    fig, pitch, side = pitch_axes(
        f"xT Heatmap · {TEAM_NAME[team_id]}",
        "Full-pitch 7 × 12 square grid · every cell sums threat added from pass origins",
    )
    team_mark = _team_mark_color(team_id)
    team_rgb = np.asarray(mcolors.to_rgb(team_mark), dtype=float)
    team_dark = mcolors.to_hex(team_rgb * 0.42)
    cmap = LinearSegmentedColormap.from_list(
        f"xt_full_grid_{team_id}", [BG, PANEL_2, team_dark, team_mark]
    )
    nonzero = heat[heat > 0]
    vmax = max(float(np.percentile(nonzero, 92)) if nonzero.size else 0.0, 0.001)
    x_grid = np.linspace(-PITCH_WIDTH / 2, PITCH_WIDTH / 2, 8)
    y_grid = np.linspace(0, PITCH_LENGTH, 13)
    image = pitch.pcolormesh(
        x_grid, y_grid, heat.T, cmap=cmap, vmin=0, vmax=vmax, shading="flat",
        edgecolors=GRID, linewidth=0.58, alpha=0.98, zorder=1,
    )
    draw_long_pitch(pitch)
    for ix in range(7):
        for iy in range(12):
            value = float(heat[ix, iy])
            if value <= 0:
                continue
            intensity = min(value / vmax, 1.0)
            cell_fill = mcolors.to_hex(cmap(intensity))
            number_color = text_on_fill(cell_fill)
            pitch.text(
                (x_grid[ix] + x_grid[ix + 1]) / 2,
                (y_grid[iy] + y_grid[iy + 1]) / 2,
                f"{value:.2f}", color=number_color,
                fontsize=5.0, fontweight="bold", ha="center", va="center", zorder=3,
            )
    top = team.nlargest(10, "xT")
    # Ranks 4–10 used to be drawn in the team colour, which is exactly the
    # colour of the hot cells underneath them — the dashed arrows vanished
    # into the heatmap. Use an accent hue away from the team's own ramp, at
    # full opacity with a black halo, so they stay legible over every cell.
    dash_color = _on_team_heatmap_accent(team_mark)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        sx, sy = attack_xy([row["x"]], [row["y"]]); ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        top_three = rank <= 3
        arrow_color = EVENT_HIGHLIGHT if top_three else dash_color
        arrow = pitch.annotate(
            "", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]),
            arrowprops=dict(
                arrowstyle="-|>", color=arrow_color,
                lw=1.75 if top_three else 1.25,
                alpha=0.94 if top_three else 0.92,
                linestyle="-" if top_three else QUIET_DASH,
                mutation_scale=11 if top_three else 9,
            ),
        )
        if arrow.arrow_patch is not None:
            arrow.arrow_patch.set_path_effects([
                path_effects.Stroke(linewidth=3.0 if top_three else 2.6, foreground=BG),
                path_effects.Normal(),
            ])
    cbar = fig.colorbar(image, ax=pitch, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.outline.set_edgecolor(GRID)
    cbar.set_label("xT added per grid square", color=MUTED, fontsize=8)
    side_title(side, "TOP 10 xT PASSES")
    side_rows(
        side,
        [(f"{rank}. {str(row['player']).split()[-1]}", f"{float(row['xT']):.3f}") for rank, (_, row) in enumerate(top.iterrows(), start=1)],
        start=0.835,
        gap=0.063,
        value_color=TEXT,
        label_color=TEXT,
        label_weight="bold",
    )
    side.plot([0.08, 0.16], [0.088, 0.088], color=EVENT_HIGHLIGHT, lw=1.75)
    side.text(0.19, 0.088, "Top 3 xT passes", color=MUTED, fontsize=7.2, va="center")
    side.plot([0.55, 0.63], [0.088, 0.088], color=dash_color, lw=1.25, linestyle=QUIET_DASH)
    side.text(0.66, 0.088, "Ranks 4–10", color=MUTED, fontsize=7.2, va="center")
    return save(fig, f"{number:02d}_xt_map_{_team_slug(team_id)}.png")


def pass_map(events, team_id, number):
    frame = events[events["team_id"].eq(team_id) & events["type"].astype(str).eq("Pass")].copy()
    frame = frame.dropna(subset=["x", "y", "end_x", "end_y"])
    completed = frame["outcome"].astype(str).str.lower().eq("successful")
    key_pass = as_bool(frame.get("is_key_pass", pd.Series(False, index=frame.index)))
    fig, pitch, side = pitch_axes(
        f"Pass Map · {TEAM_NAME[team_id]}",
        "Every pass in the match · completed, incomplete and key passes are explicitly distinguished",
    )
    draw_long_pitch(pitch)
    team_mark = _team_mark_color(team_id)
    for idx, row in frame.iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]])
        ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        if bool(key_pass.loc[idx]):
            color, alpha, width, style = EVENT_HIGHLIGHT, 0.95, 1.8, "-"
        elif bool(completed.loc[idx]):
            color, alpha, width, style = team_mark, 0.24, 0.62, "-"
        else:
            color, alpha, width, style = team_mark, 0.34, 0.62, FAILURE_DASH
        pitch.plot([sx[0], ex[0]], [sy[0], ey[0]], color=color, alpha=alpha,
                   lw=width, ls=style, zorder=2)
        if bool(key_pass.loc[idx]):
            pitch.scatter(ex[0], ey[0], s=22, marker="*", color=EVENT_HIGHLIGHT,
                          edgecolor=TEXT, linewidth=0.45, zorder=4)
    attempts = len(frame)
    complete_count = int(completed.sum())
    forward = int((pd.to_numeric(frame["end_x"], errors="coerce") > pd.to_numeric(frame["x"], errors="coerce")).sum())
    profile = pass_length_profile(events, team_id)
    side_title(side, "PASSING OUTPUT")
    side_kpis(side, [
        ("Attempts", attempts),
        ("Completed", complete_count),
        ("Completion", f"{100 * complete_count / max(attempts, 1):.1f}%"),
        ("Forward passes", forward),
    ], start=0.82, gap=0.115)
    # The same completion rate means different things at 13 m and at 25 m, so
    # length and long-ball survival sit next to the raw totals.
    side.text(0.08, 0.335, "LENGTH & DIRECTION", color=MUTED, fontsize=7.5, fontweight="bold")
    stat_rows = [
        ("Average length", f"{profile['avg_length_m']:.1f} m"),
        ("Long balls", f"{profile['long_ball_share']:.0f}%"),
        ("Long-ball completion", f"{profile['long_ball_completion']:.0f}%"),
    ]
    for idx, (label, value) in enumerate(stat_rows):
        y = 0.29 - idx * 0.05
        side.text(0.08, y, label, color=TEXT, fontsize=8, va="center")
        side.text(0.92, y, value, color=TEXT, fontsize=8.5, fontweight="bold",
                  ha="right", va="center")
    # The key sat at a fixed 0.235/0.165/0.095 while the rows above it ran to
    # 0.19, so the "Completed pass" swatch was drawn through the "Long balls"
    # figure. Anchored under the last row instead.
    legend_top = 0.29 - (len(stat_rows) - 1) * 0.05 - 0.06
    legend_y = [legend_top - index * 0.058 for index in range(3)]
    legend_items = [
        ("Completed pass", team_mark, "-", "o"),
        ("Incomplete pass", team_mark, FAILURE_DASH, "o"),
        (f"Key pass ({int(key_pass.sum())})", EVENT_HIGHLIGHT, "-", "*"),
    ]
    for y, (label, color, style, marker) in zip(legend_y, legend_items):
        side.plot([0.09, 0.25], [y, y], color=color, lw=2.0, ls=style)
        side.scatter([0.25], [y], s=38 if marker == "*" else 20, marker=marker,
                     color=color, edgecolor=TEXT, linewidth=0.45, zorder=4)
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
        players["team_id"].eq(team_id)
        & players["position"].astype(str).str.upper().eq("GK")
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
                px, py = point
                body = str(row.get("body_part") or "").lower()
                marker = _BODY_PART_MARKERS.get(body, ("o", "Right foot"))[0]
                ax.scatter(
                    [px], [py],
                    s=min(40 + float(row["xG"]) * 620, 380),
                    marker=marker, facecolors=color, edgecolors=BG, linewidths=0.9,
                    alpha=0.95, zorder=6 if event_type == "Goal" else 5,
                )
                plotted += 1

        if plotted == 0:
            ax.text(0, 0.5, "No placement data recorded", color=MUTED, fontsize=9,
                    ha="center", va="center", zorder=7)

        keeper = _keeper_name(players, keeper_team)
        fig.text(x0, 0.815, keeper.upper(), color=accent, fontsize=14, fontweight="bold")
        fig.text(x0, 0.788, f"{keeper_team_name} · {plotted} of {len(faced)} shots faced reached the frame",
                 color=MUTED, fontsize=8)

        on_target = int(shot_type.isin(["Goal", "SavedShot"]).sum())
        conceded = int(shot_type.eq("Goal").sum())
        saves = int(shot_type.eq("SavedShot").sum())
        # Post-shot xG of the shots this keeper actually faced. The report's
        # old "xGoT" was the sum of xG over on-target shots, which ignores
        # placement — a shot rolled at the keeper scored the same as one in the
        # top corner, so "goals prevented" measured nothing about the keeper.
        psxg = round(float(post_shot_xg(faced).sum()), 2)
        cards = [
            ("On target", f"{on_target}"),
            ("Saves", f"{saves}"),
            ("Conceded", f"{conceded}"),
            ("Save rate", f"{100 * saves / max(on_target, 1):.0f}%"),
            ("PSxG faced", f"{psxg:.2f}"),
            ("Prevented", f"{psxg - conceded:+.2f}"),
        ]
        card_w = 0.40 / len(cards)
        for idx, (label, value) in enumerate(cards):
            cx = x0 + idx * card_w
            fig.add_artist(Rectangle(
                (cx, 0.290), card_w * 0.93, 0.105, transform=fig.transFigure,
                facecolor=PANEL, edgecolor=GRID, lw=0.9, zorder=1,
            ))
            fig.text(cx + card_w * 0.465, 0.357, value, color=TEXT, fontsize=13,
                     fontweight="bold", ha="center", va="center", zorder=2)
            fig.text(cx + card_w * 0.465, 0.312, label.upper(), color=MUTED, fontsize=6.2,
                     fontweight="bold", ha="center", va="center", zorder=2)

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

    # Shared legend: outcome colour on one row, body-part shape on the next.
    for row_y, entries in (
        (0.205, [(("o"), color, label) for _t, color, label in outcomes]),
        (0.155, [(marker, TEXT, label) for marker, label in _BODY_PART_MARKERS.values()]),
    ):
        legend_x = 0.055
        for marker, color, label in entries:
            fig.add_artist(Line2D([legend_x], [row_y], marker=marker, color=color, lw=0,
                                  markersize=8, transform=fig.transFigure))
            fig.text(legend_x + 0.013, row_y, label.upper(), color=MUTED, fontsize=7.2,
                     fontweight="bold", va="center")
            legend_x += 0.013 + 0.0062 * len(label) + 0.028

    fig.text(0.055, 0.098,
             "Goal frame seen from behind the shooter · off-target shots sit outside the posts · "
             "blocked shots are excluded because they never reached the keeper",
             fontsize=7, color=NEUTRAL)
    fig.text(0.945, 0.035, "FULL VISUAL REDESIGN · REAL MATCH DATA",
             ha="right", fontsize=8, color=NEUTRAL)
    return save(fig, "11_goalkeeper_saves.png")


def zone14(events, team_id, number):
    team = events[events["team_id"].eq(team_id)].copy()
    successful = team["outcome"].astype(str).str.lower().eq("successful")
    zone = successful & pd.to_numeric(team["end_x"], errors="coerce").between(70, 83) & pd.to_numeric(team["end_y"], errors="coerce").between(35, 65)
    actions = team[zone].dropna(subset=["x", "y", "end_x", "end_y"])
    final_third = team[successful & (pd.to_numeric(team["end_x"], errors="coerce") >= 66.7)].copy()
    final_third["end_y_num"] = pd.to_numeric(final_third["end_y"], errors="coerce")
    final_third = final_third.dropna(subset=["end_y_num"])
    lane_defs = [
        ("Left wing", 0, 20),
        ("Left half-space", 20, 40),
        ("Central lane", 40, 60),
        ("Right half-space", 60, 80),
        ("Right wing", 80, 100.0001),
    ]
    team_mark = _team_mark_color(team_id)
    lane_colors = [team_mark] * len(lane_defs)
    lane_counts = [int(final_third["end_y_num"].between(lo, hi, inclusive="left").sum()) for _, lo, hi in lane_defs]
    fig, pitch, side = pitch_axes(
        f"Zone 14 & Five Lanes · {TEAM_NAME[team_id]}",
        "Lane numbers = completed actions ending in the final third · arrows = completed Zone 14 access",
    )
    draw_long_pitch(pitch)
    third_y = 66.7 * PITCH_LENGTH / 100
    for (label, lo, hi), color in zip(lane_defs, lane_colors):
        x1, _ = attack_xy([66.7], [lo])
        x2, _ = attack_xy([66.7], [min(hi, 100)])
        pitch.add_patch(Rectangle(
            (min(x1[0], x2[0]), third_y), abs(x2[0] - x1[0]), PITCH_LENGTH - third_y,
            facecolor=color, edgecolor=color, lw=0.9, alpha=_SHADE_ALPHA, zorder=0,
        ))
    for idx, ((_, lo, hi), count, color) in enumerate(zip(lane_defs, lane_counts, lane_colors)):
        center_y = (lo + min(hi, 100)) / 2
        cx, cy = attack_xy([68.8], [center_y])
        pitch.text(cx[0], cy[0], str(count), ha="center", va="center", color=TEXT,
                   fontsize=8, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.28", fc=PANEL, ec=color, lw=1.15, alpha=0.97), zorder=6)
    zx1, zy1 = attack_xy([70], [35]); zx2, zy2 = attack_xy([83], [65])
    pitch.add_patch(Rectangle(
        (min(zx1[0], zx2[0]), min(zy1[0], zy2[0])),
        abs(zx2[0] - zx1[0]), abs(zy2[0] - zy1[0]),
        facecolor=FOCUS, alpha=_HATCH_ALPHA, edgecolor=FOCUS, lw=1.8, hatch="//", zorder=1,
    ))
    pitch.text(0, min(zy1[0], zy2[0]) + 0.9, "ZONE 14", color=FOCUS, fontsize=6.5,
               fontweight="bold", ha="center", va="bottom", zorder=5)
    for _, row in actions.iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]]); ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        pitch.annotate(
            "", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]),
            arrowprops=dict(
                arrowstyle="-|>", color=team_mark, alpha=0.42, lw=0.78,
                linestyle=QUIET_DASH, mutation_scale=8,
            ),
        )
    top = actions.groupby("player").size().sort_values(ascending=False).head(3)
    side_title(side, "FIVE ATTACKING LANES")
    for idx, ((label, _, _), value, color) in enumerate(zip(lane_defs, lane_counts, lane_colors)):
        y = 0.81 - idx * 0.083
        side.add_patch(Rectangle((0.08, y - 0.016), 0.035, 0.032, facecolor=color,
                                 edgecolor=TEXT, lw=0.4, alpha=0.9))
        side.text(0.15, y, label, color=TEXT, fontsize=8.5, va="center")
        side.text(0.92, y, str(value), color=TEXT, fontsize=8.8, fontweight="bold",
                  ha="right", va="center")
        side.plot([0.08, 0.92], [y - 0.037, y - 0.037], color=GRID, lw=0.55, alpha=0.7)
    side.text(0.08, 0.34, "ZONE 14 CONTRIBUTORS", color=MUTED, fontsize=7.5, fontweight="bold")
    for idx, (name, value) in enumerate(top.items()):
        y = 0.285 - idx * 0.06
        side.text(0.08, y, str(name).split()[-1], color=TEXT, fontsize=8, va="center")
        side.text(0.92, y, str(int(value)), color=TEXT, fontsize=8, fontweight="bold", ha="right", va="center")
    side.text(0.08, 0.075, f"Zone 14 entries: {len(actions)}", color=TEXT, fontsize=9.5, fontweight="bold")
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
    work = work.sort_values(["period_num", "minute_num", "second_num", "event_id"]).reset_index(drop=True)
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
            elapsed = float(candidate["minute_num"]) * 60 + float(candidate["second_num"]) - source_time
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
    home_counts, away_counts = team_event_counts(events, HOME_ID), team_event_counts(events, AWAY_ID)
    home_xa = _derived_expected_assists(events, HOME_ID)
    away_xa = _derived_expected_assists(events, AWAY_ID)
    attack_rows = [
        ("Shots", float(home_xg.get("shots", 0)), float(away_xg.get("shots", 0)), "{:.0f}", False),
        ("Shots on target", float(home_xg.get("on_target", 0)), float(away_xg.get("on_target", 0)), "{:.0f}", False),
        ("Big chances", float(home_xg.get("big_chances", 0)), float(away_xg.get("big_chances", 0)), "{:.0f}", False),
        ("Expected goals (xG)", float(home_xg.get("xG", 0)), float(away_xg.get("xG", 0)), "{:.2f}", False),
        ("xG on target (xGoT)", float(home_xg.get("xGoT", 0)), float(away_xg.get("xGoT", 0)), "{:.2f}", False),
        ("xG per shot", float(home_xg.get("xG_per_shot", 0)), float(away_xg.get("xG_per_shot", 0)), "{:.3f}", False),
        ("Transition xG", metric("home", "transition_xG"), metric("away", "transition_xG"), "{:.2f}", False),
        ("Transition shot rate", metric("home", "transition_shot_rate"), metric("away", "transition_shot_rate"), "{:.1f}%", False),
    ]
    creation_rows = [
        ("Possession share", metric("home", "possession_share"), metric("away", "possession_share"), "{:.1f}%", False),
        ("Expected assists (xA)", home_xa, away_xa, "{:.2f}", False),
        ("Open-play xT", float(home_xg.get("xT", 0)), float(away_xg.get("xT", 0)), "{:.2f}", False),
        ("Field tilt", metric("home", "field_tilt"), metric("away", "field_tilt"), "{:.1f}%", False),
        ("Progressive passes", metric("home", "progressive_passes"), metric("away", "progressive_passes"), "{:.0f}", False),
        ("Final-third entries", metric("home", "final_third_entries"), metric("away", "final_third_entries"), "{:.0f}", False),
        ("Deep completions", metric("home", "deep_completions"), metric("away", "deep_completions"), "{:.0f}", False),
        ("Box entries", metric("home", "box_entries"), metric("away", "box_entries"), "{:.0f}", False),
    ]
    defence_rows = [
        ("PPDA", home_ppda, away_ppda, "{:.2f}", True),
        ("Tackles", float(home_counts["Tackles"]), float(away_counts["Tackles"]), "{:.0f}", False),
        ("Interceptions", float(home_counts["Interceptions"]), float(away_counts["Interceptions"]), "{:.0f}", False),
        ("Possession regains", metric("home", "possession_regains"), metric("away", "possession_regains"), "{:.0f}", False),
        ("High regains", metric("home", "high_regains"), metric("away", "high_regains"), "{:.0f}", False),
        ("Counterpress success", metric("home", "counterpress_success_rate"), metric("away", "counterpress_success_rate"), "{:.1f}%", False),
        # Absorbed from the former standalone defensive-summary page: these are
        # the rows it carried that were not already here.
        ("Recoveries", float(home_counts["Recoveries"]), float(away_counts["Recoveries"]), "{:.0f}", False),
        ("Clearances", float(home_counts["Clearances"]), float(away_counts["Clearances"]), "{:.0f}", False),
        ("Blocks", float(home_counts["Blocks"]), float(away_counts["Blocks"]), "{:.0f}", False),
        ("Fouls", float(home_counts["Fouls"]), float(away_counts["Fouls"]), "{:.0f}", True),
    ]
    # Absorbed from the former standalone advanced-metrics page: process,
    # sequence value and the risk carried behind the ball.
    process_rows = [
        ("Transitions", metric("home", "transitions"), metric("away", "transitions"), "{:.0f}", False),
        ("Build-up success", metric("home", "build_up_success_rate"), metric("away", "build_up_success_rate"), "{:.1f}%", False),
        ("Final-third entry efficiency", metric("home", "final_third_entry_efficiency"), metric("away", "final_third_entry_efficiency"), "{:.1f}%", False),
        ("Box entry → shot", metric("home", "box_entry_to_shot_rate"), metric("away", "box_entry_to_shot_rate"), "{:.1f}%", False),
        ("Sequence xT", metric("home", "sequence_xT"), metric("away", "sequence_xT"), "{:.2f}", False),
        ("Transition xT", metric("home", "transition_xT"), metric("away", "transition_xT"), "{:.2f}", False),
        ("Directness", metric("home", "directness"), metric("away", "directness"), "{:.1f}%", False),
        ("Transition exposure", metric("home", "rest_defence_exposures"), metric("away", "rest_defence_exposures"), "{:.0f}", True),
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

    territory_team = HOME_NAME if metric("home", "field_tilt") > metric("away", "field_tilt") else AWAY_NAME
    quality_team = HOME_NAME if float(home_xg.get("xG_per_shot", 0)) > float(away_xg.get("xG_per_shot", 0)) else AWAY_NAME
    secure_team = HOME_NAME if metric("home", "rest_defence_vulnerability") < metric("away", "rest_defence_vulnerability") else AWAY_NAME
    fig.text(0.035, 0.112, "MATCH READ", color=FOCUS, fontsize=7.5, fontweight="bold")
    fig.text(
        0.093,
        0.112,
        f"{territory_team} controlled more territory; {quality_team} created the cleaner average shot and {secure_team} protected attacking possessions more securely.",
        color=TEXT,
        fontsize=8.5,
    )
    fig.text(0.035, 0.066, "xA = xG of the shot following each provider-tagged key pass (within 20 seconds).", color=MUTED, fontsize=7)
    fig.text(0.965, 0.066, "↓ LOWER IS BETTER   ·   REAL MATCH EVENTS", color=NEUTRAL, fontsize=7, ha="right")
    return save(fig, "14_post_match_advanced_dashboard.png")


def progressive(events, team_id, number):
    prog = events[progressive_pass_mask(events) & events["team_id"].eq(team_id)].copy().dropna(subset=["x", "y", "end_x", "end_y"])
    prog["xT"] = pd.to_numeric(prog["xT"], errors="coerce").fillna(0)
    fig, pitch, side = pitch_axes(f"Progressive Passes · {TEAM_NAME[team_id]}", "Canonical zone-aware thresholds · strongest ten actions highlighted by xT added")
    draw_long_pitch(pitch)
    # The dashed layer carries the full progressive volume, so it has to stay
    # legible under the ten white highlight arrows. Denser dash, heavier stroke
    # and a much higher alpha than the generic QUIET_DASH treatment.
    volume_dash = (0, (3.4, 2.2))
    for _, row in prog.iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]]); ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        pitch.annotate(
            "", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]),
            arrowprops=dict(
                arrowstyle="-|>", color=_team_mark_color(team_id), alpha=0.55,
                lw=1.15, linestyle=volume_dash, mutation_scale=10,
            ),
        )
    for _, row in prog.nlargest(10, "xT").iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]]); ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        pitch.annotate(
            "", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]),
            arrowprops=dict(
                arrowstyle="-|>", color=EVENT_HIGHLIGHT, alpha=0.95,
                lw=1.9, mutation_scale=11,
            ),
        )
    top = prog.groupby("player").size().sort_values(ascending=False).head(7)
    side_title(side, "TOP PROGRESSORS")
    side_rows(side, [(str(name).split()[-1], str(int(value))) for name, value in top.items()])
    side.text(0.08, 0.14, f"Team-colour dashed = all progressive passes ({len(prog)})", color=_team_mark_color(team_id), fontsize=8.2)
    side.text(0.08, 0.09, f"{HIGHLIGHT_LABEL} = top 10 by xT added", color=EVENT_HIGHLIGHT, fontsize=8.4)
    return save(fig, f"{number:02d}_progressive_{_team_slug(team_id)}.png")


def crosses(events, team_id, number):
    mask = cross_mask(events)
    frame = events[mask & events["team_id"].eq(team_id)].copy().dropna(subset=["x", "y", "end_x", "end_y"])
    success = frame["outcome"].astype(str).str.lower().eq("successful")
    fig, pitch, side = pitch_axes(f"Crosses · {TEAM_NAME[team_id]}", "Cross origins and targets · completed deliveries use filled arrowheads")
    draw_long_pitch(pitch)
    team_mark = _team_mark_color(team_id)
    for idx, row in frame.iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]]); ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        good = bool(success.loc[idx])
        pitch.annotate(
            "", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]),
            arrowprops=dict(
                arrowstyle="-|>" if good else "->",
                color=team_mark,
                alpha=0.82 if good else 0.38,
                lw=1.15 if good else 0.68,
                linestyle="-" if good else FAILURE_DASH,
                mutation_scale=9,
            ),
        )
    completed = int(success.sum()); rate = 100 * completed / max(len(frame), 1)
    side_title(side, "CROSSING OUTPUT")
    side_kpis(side, [("Crosses", len(frame)), ("Completed", completed), ("Completion", f"{rate:.1f}%"), ("Open-play", int((~frame["qualifier_names"].astype(str).str.lower().str.contains("corner")).sum()))])
    side.plot([0.09, 0.24], [0.12, 0.12], color=team_mark, lw=1.35)
    side.text(0.28, 0.12, "Completed", color=TEXT, fontsize=7.5, va="center")
    side.plot([0.55, 0.70], [0.12, 0.12], color=team_mark, lw=1.0, linestyle=FAILURE_DASH)
    side.text(0.74, 0.12, "Incomplete", color=TEXT, fontsize=7.5, va="center")
    return save(fig, f"{number:02d}_crosses_{_team_slug(team_id)}.png")


def defensive_activity(events, team_id, number):
    event_types = events["type"].astype(str)
    non_foul_actions = event_types.isin(["Tackle", "Interception", "BallRecovery", "Clearance"])
    committed_fouls = fouls_committed_mask(events)
    own_actions = events[
        events["team_id"].eq(team_id) & (non_foul_actions | committed_fouls)
    ].dropna(subset=["x", "y"]).copy()
    opponent_id = AWAY_ID if team_id == HOME_ID else HOME_ID
    block_actions = defensive_block_events(events, team_id, opponent_id).dropna(
        subset=["x", "y"]
    )
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
        alpha=0.92,
        interpolation="bicubic",
    )
    draw_long_pitch(pitch)
    marker_map = {"Tackle": "o", "Interception": "D", "BallRecovery": "s", "Clearance": "^", "BlockedShot": "P", "Foul": "X"}
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
        if subset.empty: continue
        px, py = attack_xy(subset["x"], subset["y"])
        pitch.scatter(
            px,
            py,
            marker=marker,
            s=34,
            facecolors=action_colors[event_type],
            edgecolors=BG,
            linewidth=0.75,
            alpha=0.96,
            zorder=5,
        )
    counts = team_event_counts(events, team_id)
    side_title(side, "ACTION TYPE LEGEND")
    label_to_event = {
        "Tackles": "Tackle", "Interceptions": "Interception", "Recoveries": "BallRecovery",
        "Clearances": "Clearance", "Blocks": "BlockedShot", "Fouls": "Foul",
    }
    for idx, (label, value) in enumerate(counts.items()):
        y = 0.81 - idx * 0.095
        marker = marker_map[label_to_event[label]]
        event_type = label_to_event[label]
        side.scatter(
            [0.13], [y], s=52, marker=marker,
            facecolors=action_colors[event_type], edgecolors=BG, linewidth=0.9,
        )
        side.text(0.21, y, label, color=TEXT, fontsize=8.5, va="center")
        side.text(0.90, y, str(value), color=TEXT, fontsize=8.5, fontweight="bold", ha="right", va="center")
        side.plot([0.08, 0.92], [y - 0.043, y - 0.043], color=GRID, lw=0.55, alpha=0.7)
    side.add_patch(Rectangle((0.09, 0.105), 0.08, 0.055, facecolor=team_mark, edgecolor=GRID, alpha=0.82))
    side.text(0.21, 0.132, f"{TEAM_NAME[team_id]} colour = action density", color=TEXT, fontsize=8, va="center")
    return save(fig, f"{number:02d}_defensive_activity_{_team_slug(team_id)}.png")


def average_positions(events, players, team_id, number, half):
    positions, _edges, sub_on, sub_off, substitutions, _completed_links = _half_network_data(events, players, team_id, half)
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
    radii = {name: _network_node_radius(touches, max_touch)
             for name, (_x, _y, touches) in display.items()}
    for name, (px, py, touches) in display.items():
        entered = name in sub_on
        left = name in sub_off
        pitch.scatter(px, py, s=260 + 640 * touches / max_touch, marker="s" if entered else "o",
                      color=_team_mark_color(team_id), edgecolor=FOCUS if left else outline_color,
                      linewidth=2.3 if left else 1.15, zorder=4)
        draw_node_label(pitch, px, py, name, touches, max_touch,
                        node_color=_team_mark_color(team_id),
                        shirt=shirts.get(str(name)), node_radius=radii[name],
                        neighbours=_node_neighbours(display, radii, name))

    side_title(side, "HALF PARTICIPATION")
    side.text(0.92, 0.94, f"{len(positions)} players", color=TEXT, fontsize=8,
              fontweight="bold", ha="right", va="top")
    active = positions.sort_values("touches", ascending=False).head(5)
    side_rows(side, [(compact_player_label(name, 16), str(int(row["touches"]))) for name, row in active.iterrows()], start=0.81, gap=0.075)
    side.text(0.08, 0.40, "SUBSTITUTIONS", color=MUTED, fontsize=7.5, fontweight="bold")
    if substitutions:
        for idx, (minute, on_name, off_name) in enumerate(substitutions[:5]):
            y = 0.35 - idx * 0.052
            side.text(0.08, y, f"{minute}′", color=TEXT, fontsize=7.5, fontweight="bold", va="center")
            change = f"{off_name} OFF AT INTERVAL" if on_name == "—" else f"{on_name} IN  ·  {off_name} OFF"
            side.text(0.19, y, change, color=TEXT, fontsize=7.2, va="center")
    else:
        side.text(0.08, 0.34, "No in-half changes", color=MUTED, fontsize=8)
    side.scatter([0.06, 0.34], [0.075, 0.075], s=[65, 65], marker="o", color=_team_mark_color(team_id), edgecolor=[TEXT, FOCUS], linewidth=[1.0, 2.1])
    side.scatter([0.60], [0.075], s=65, marker="s", color=_team_mark_color(team_id), edgecolor=TEXT, linewidth=1.0)
    side.text(0.10, 0.075, "Began half", color=TEXT, fontsize=6.8, va="center")
    side.text(0.38, 0.075, "Went off", color=TEXT, fontsize=6.8, va="center")
    side.text(0.64, 0.075, "Came on", color=TEXT, fontsize=6.8, va="center")
    suffix = "1h" if half == 1 else "2h"
    return save(fig, f"{number:02d}{'a' if half == 1 else 'b'}_average_positions_{_team_slug(team_id)}_{suffix}.png")


def dominating_zones(events):
    tmask = touch_mask(events)
    home = events[tmask & events["team_id"].eq(HOME_ID)].dropna(subset=["x", "y"])
    away = events[tmask & events["team_id"].eq(AWAY_ID)].dropna(subset=["x", "y"])
    hh, _, _ = np.histogram2d(home["y"], home["x"], bins=[5, 7], range=[[0, 100], [0, 100]])
    ah, _, _ = np.histogram2d(away["y"], away["x"], bins=[5, 7], range=[[0, 100], [0, 100]])
    total = hh + ah
    diff = np.divide(hh - ah, total, out=np.zeros_like(total), where=total > 0)
    counts = hh - ah
    fig, pitch, side = pitch_axes("Dominating Zones", "Touch-share difference · diverging scale centred on an even 50/50 split")
    cmap = LinearSegmentedColormap.from_list("dom_full", [AWAY, PANEL_2, HOME])
    image = pitch.imshow(diff.T, extent=[-PITCH_WIDTH / 2, PITCH_WIDTH / 2, 0, PITCH_LENGTH], origin="lower", cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    draw_long_pitch(pitch)
    for ix in range(5):
        for iy in range(7):
            px = -PITCH_WIDTH / 2 + (ix + 0.5) * PITCH_WIDTH / 5
            py = (iy + 0.5) * PITCH_LENGTH / 7
            # Diverging ramp: both ends carry a team colour, so either end can
            # be light. Read the label colour off the cell it sits on.
            cell_fill = mcolors.to_hex(cmap((float(diff[ix, iy]) + 1.0) / 2.0))
            pitch.text(px, py, f"{int(counts[ix, iy]):+d}", ha="center", va="center",
                       color=text_on_fill(cell_fill), fontsize=6.5, fontweight="bold",
                       path_effects=label_outline(cell_fill))
    cbar = fig.colorbar(image, ax=pitch, fraction=0.035, pad=0.02, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels([AWAY_NAME, "Balanced", HOME_NAME]); cbar.ax.tick_params(colors=MUTED, labelsize=7); cbar.outline.set_edgecolor(GRID)
    side_title(side, "TERRITORY TOTALS")
    side_kpis(
        side,
        [
            (f"{HOME_NAME} touches", len(home)),
            (f"{AWAY_NAME} touches", len(away)),
            ("Difference", f"{len(home)-len(away):+d}"),
            ("Cell label", f"{HOME_NAME} − {AWAY_NAME}"),
        ],
    )
    return save(fig, "24_dominating_zones.png")


def box_entries(events, team_id, number):
    frame = events[box_entry_mask(events) & events["team_id"].eq(team_id)].copy().dropna(subset=["x", "y", "end_x", "end_y"])
    fig, pitch, side = pitch_axes(f"Box Entries · {TEAM_NAME[team_id]}", "Completed actions entering the penalty area · entry method encoded by shape")
    draw_long_pitch(pitch)
    team_mark = _team_mark_color(team_id)
    for _, row in frame.iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]]); ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        event_type = str(row["type"])
        marker = "o" if event_type == "Pass" else "D"
        is_pass = marker == "o"
        entry_color = team_mark
        pitch.annotate(
            "", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]),
            arrowprops=dict(
                arrowstyle="-|>", color=entry_color, alpha=0.62 if is_pass else 0.52,
                lw=1.0 if is_pass else 0.82,
                linestyle="-" if is_pass else FAILURE_DASH, mutation_scale=8,
            ),
        )
        pitch.scatter(ex[0], ey[0], s=32, marker=marker, color=entry_color, edgecolor=TEXT, linewidth=0.65)
    top = frame.groupby("player").size().sort_values(ascending=False).head(5)
    side_title(side, "ENTRY CONTRIBUTORS")
    side_rows(side, [(str(name).split()[-1], str(int(value))) for name, value in top.items()], start=0.81, gap=0.083)
    pass_count = int(frame["type"].astype(str).eq("Pass").sum())
    carry_count = len(frame) - pass_count
    side.text(0.08, 0.35, "ENTRY METHOD LEGEND", color=MUTED, fontsize=7.5, fontweight="bold")
    side.scatter([0.13], [0.285], s=45, marker="o", color=team_mark, edgecolor=TEXT, linewidth=0.6)
    side.text(0.21, 0.285, f"Pass entry ({pass_count})", color=TEXT, fontsize=8, va="center")
    side.scatter([0.13], [0.22], s=45, marker="D", color=team_mark, edgecolor=TEXT, linewidth=0.6)
    side.text(0.21, 0.22, f"Carry / take-on ({carry_count})", color=TEXT, fontsize=8, va="center")
    side.annotate("", xy=(0.18, 0.15), xytext=(0.08, 0.15), arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.1))
    side.text(0.21, 0.15, "Arrow = entry path", color=TEXT, fontsize=8, va="center")
    side.text(0.08, 0.075, f"Total entries: {len(frame)}", color=TEXT, fontsize=9.5, fontweight="bold")
    return save(fig, f"{number:02d}_box_entries_{_team_slug(team_id)}.png")


def high_regains(events, team_id, number):
    frame = high_regain_events(events, team_id).dropna(subset=["x", "y"]).copy()
    fig, pitch, side = pitch_axes(f"High Regains · {TEAM_NAME[team_id]}", "Open-play possession regains at x ≥ 60 · type encoded by shape")
    draw_long_pitch(pitch)
    threshold_y = 60 * PITCH_LENGTH / 100
    pitch.axhspan(threshold_y, PITCH_LENGTH, color=FOCUS, alpha=0.055)
    pitch.axhline(threshold_y, color=FOCUS, lw=1.0, ls=(0, (5, 4)))
    marker_map = {"Tackle": "o", "Interception": "D", "BallRecovery": "s"}
    team_mark = _team_mark_color(team_id)
    for event_type, marker in marker_map.items():
        subset = frame[frame["type"].astype(str).eq(event_type)]
        if subset.empty: continue
        px, py = attack_xy(subset["x"], subset["y"])
        pitch.scatter(px, py, s=65, marker=marker, color=team_mark, edgecolor=TEXT, linewidth=0.8, label=f"{event_type} ({len(subset)})")
    other = frame[~frame["type"].astype(str).isin(marker_map)]
    if not other.empty:
        px, py = attack_xy(other["x"], other["y"]); pitch.scatter(px, py, s=65, marker="^", color=team_mark, edgecolor=TEXT, linewidth=0.8, label=f"Other ({len(other)})")
    pitch.legend(loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, labelcolor=TEXT, fontsize=7)
    top = frame.groupby("player").size().sort_values(ascending=False).head(7) if "player" in frame else pd.Series(dtype=int)
    side_title(side, "HIGH-REGAIN LEADERS")
    side_rows(side, [(str(name).split()[-1], str(int(value))) for name, value in top.items()])
    side.text(0.08, 0.14, f"Total high regains: {len(frame)}", color=TEXT, fontsize=9, fontweight="bold")
    return save(fig, f"{number:02d}_high_regains_{_team_slug(team_id)}.png")


def pass_targets(events, team_id, number):
    frame = events[events["team_id"].eq(team_id) & events["type"].astype(str).eq("Pass") & events["outcome"].astype(str).str.lower().eq("successful")].dropna(subset=["end_x", "end_y"]).copy()
    heat, _, _ = np.histogram2d(frame["end_y"], frame["end_x"], bins=[7, 12], range=[[0, 100], [0, 100]])
    fig, pitch, side = pitch_axes(f"Pass Target Zones · {TEAM_NAME[team_id]}", "Completed-pass destinations · one sequential density scale")
    team_mark = _team_mark_color(team_id)
    cmap = LinearSegmentedColormap.from_list(
        f"targets_{team_id}", _team_density_palette(team_id)
    )
    image = pitch.imshow(heat.T, extent=[-PITCH_WIDTH / 2, PITCH_WIDTH / 2, 0, PITCH_LENGTH], origin="lower", cmap=cmap, aspect="equal", alpha=0.95)
    draw_long_pitch(pitch)
    max_cell = max(float(heat.max()), 1.0)
    for ix in range(7):
        for iy in range(12):
            x0 = -PITCH_WIDTH / 2 + ix * PITCH_WIDTH / 7
            y0 = iy * PITCH_LENGTH / 12
            pitch.add_patch(Rectangle((x0, y0), PITCH_WIDTH / 7, PITCH_LENGTH / 12,
                                      fill=False, edgecolor=GRID, lw=0.42, alpha=0.55, zorder=3))
            value = int(heat[ix, iy])
            # The cell fill is a step on the team-colour ramp, so the label
            # colour has to be read off that fill — a fixed white disappears
            # at the top of a light ramp (Juventus silver, Real Madrid white).
            cell_fill = mcolors.to_hex(cmap(value / max_cell))
            pitch.text(x0 + PITCH_WIDTH / 14, y0 + PITCH_LENGTH / 24, str(value),
                       color=text_on_fill(cell_fill), fontsize=5.6, fontweight="bold",
                       ha="center", va="center", zorder=5,
                       path_effects=label_outline(cell_fill))
    cbar = fig.colorbar(image, ax=pitch, fraction=0.035, pad=0.02); cbar.ax.tick_params(colors=MUTED, labelsize=7); cbar.outline.set_edgecolor(GRID); cbar.set_label("Completed-pass targets", color=MUTED, fontsize=8)
    top = frame.groupby("player").size().sort_values(ascending=False).head(7)
    side_title(side, "TOP PASSERS")
    side_rows(side, [(str(name).split()[-1], str(int(value))) for name, value in top.items()])
    side.text(0.08, 0.14, f"Completed passes: {len(frame)}", color=TEXT, fontsize=9, fontweight="bold")
    return save(fig, f"{number:02d}_pass_targets_{_team_slug(team_id)}.png")


def ppda(events):
    info = {"home_id": HOME_ID, "away_id": AWAY_ID, "home_name": HOME_NAME, "away_name": AWAY_NAME}
    data = compute_ppda_both(info, events)
    hp = float(data["home"]["ppda"] or 0); ap = float(data["away"]["ppda"] or 0)
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.text(0.055, 0.95, "PPDA & Pressing Intensity", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.91, "Opponent passes allowed per defensive action · lower PPDA means more aggressive pressure", fontsize=10.5, color=MUTED)
    fig.add_artist(Line2D([0.055, 0.945], [0.875, 0.875], transform=fig.transFigure, color=GRID, lw=1))

    def gauge(rect, team_name, value, color, passes, actions):
        ax = fig.add_axes(rect)
        ax.set_facecolor(BG); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.72, 1.18); ax.set_aspect("equal"); ax.axis("off")
        bounds = [(5, 10, VALUE), (10, 15, "#8E7CE8"), (15, 20, FOCUS), (20, 25, "#B86A4B")]
        for low, high, band_color in bounds:
            theta_high = 180 - (low - 5) / 20 * 180
            theta_low = 180 - (high - 5) / 20 * 180
            ax.add_patch(Wedge((0, 0), 1.0, theta_low, theta_high, width=0.18,
                               facecolor=band_color, edgecolor=BG, lw=1.0, alpha=0.86))
        for tick in [5, 10, 15, 20, 25]:
            angle = np.deg2rad(180 - (tick - 5) / 20 * 180)
            ax.text(1.09 * np.cos(angle), 1.09 * np.sin(angle), str(tick), color=MUTED,
                    fontsize=7.5, ha="center", va="center")
        clipped = float(np.clip(value, 5, 25))
        angle = np.deg2rad(180 - (clipped - 5) / 20 * 180)
        ax.plot([0, 0.82 * np.cos(angle)], [0, 0.82 * np.sin(angle)], color=color, lw=4.2, solid_capstyle="round", zorder=5)
        ax.scatter([0.82 * np.cos(angle)], [0.82 * np.sin(angle)], s=65, color=TEXT, edgecolor=color, linewidth=2.0, zorder=6)
        ax.scatter([0], [0], s=62, color=color, edgecolor=TEXT, linewidth=0.8, zorder=6)
        ax.text(0, 1.12, team_name, color=TEXT, fontsize=14, fontweight="bold", ha="center")
        ax.text(0, -0.18, f"{value:.2f}", color=TEXT, fontsize=27, fontweight="bold", ha="center", va="center")
        level = "ELITE PRESS" if value < 7.5 else "HIGH PRESS" if value < 10 else "MID PRESS" if value < 14 else "LOW PRESS"
        ax.text(0, -0.38, level, color=FOCUS, fontsize=9, fontweight="bold", ha="center")
        ax.text(-0.48, -0.56, f"{passes}", color=TEXT, fontsize=13, fontweight="bold", ha="center")
        ax.text(-0.48, -0.67, "OPP PASSES", color=MUTED, fontsize=6.5, fontweight="bold", ha="center")
        ax.text(0.48, -0.56, f"{actions}", color=TEXT, fontsize=13, fontweight="bold", ha="center")
        ax.text(0.48, -0.67, "DEF ACTIONS", color=MUTED, fontsize=6.5, fontweight="bold", ha="center")

    gauge([0.08, 0.42, 0.38, 0.41], HOME_NAME, hp, HOME, data["home"]["passes_allowed"], data["home"]["defensive_actions"])
    gauge([0.54, 0.42, 0.38, 0.41], AWAY_NAME, ap, AWAY, data["away"]["passes_allowed"], data["away"]["defensive_actions"])
    leader = HOME_NAME if hp < ap else AWAY_NAME
    leader_color = HOME if hp < ap else AWAY
    fig.add_artist(Rectangle((0.285, 0.302), 0.43, 0.076, transform=fig.transFigure,
                             facecolor=PANEL, edgecolor=GRID, linewidth=0.8))
    fig.text(0.5, 0.349, f"{leader} pressed more aggressively", color=leader_color,
             fontsize=13, fontweight="bold", ha="center")
    fig.text(0.5, 0.318, f"PPDA differential: {abs(hp - ap):.2f}", color=MUTED, fontsize=9, ha="center")
    zone = fig.add_axes([0.10, 0.12, 0.80, 0.14]); zone.set_xlim(0, 100); zone.set_ylim(0, 1); zone.axis("off")
    zone.add_patch(Rectangle((0, 0.12), 40, 0.66, facecolor=PANEL, edgecolor=GRID, lw=1.0))
    zone.add_patch(Rectangle((40, 0.12), 60, 0.66, facecolor=FOCUS, edgecolor=GRID, lw=1.0, alpha=0.16))
    zone.axvline(40, ymin=0.12, ymax=0.78, color=FOCUS, lw=1.2, ls=(0, (4, 3)))
    zone.text(20, 0.45, "OWN 40%", color=MUTED, fontsize=10, fontweight="bold", ha="center")
    zone.text(70, 0.45, "PPDA PRESSING ZONE · OPPONENT 60%", color=FOCUS, fontsize=10, fontweight="bold", ha="center")
    zone.text(0, 0.89, "← own goal", color=MUTED, fontsize=7, ha="left")
    zone.text(100, 0.89, "opponent goal →", color=MUTED, fontsize=7, ha="right")
    fig.text(0.945, 0.035, "METHOD: OPPONENT PASSES ÷ TACKLES + INTERCEPTIONS + FOULS + CHALLENGES + RECOVERIES", ha="right", fontsize=7.5, color=NEUTRAL)
    return save(fig, "31_ppda_pressing.png")


def transition_outcomes(events):
    annotated, possessions = build_possessions(events)
    team_data = {}
    for team_id in [HOME_ID, AWAY_ID]:
        transitions = possessions[
            possessions["team_id"].eq(team_id)
            & possessions["is_transition"].astype(bool)
        ].copy()
        total = len(transitions)
        chances = int((pd.to_numeric(transitions["transition_shots"], errors="coerce").fillna(0) > 0).sum()) if total else 0
        shots = int(pd.to_numeric(transitions["transition_shots"], errors="coerce").fillna(0).sum()) if total else 0
        goals = int(pd.to_numeric(transitions["transition_goals"], errors="coerce").fillna(0).sum()) if total else 0
        paths = []
        for transition in transitions.itertuples():
            window = annotated[
                annotated["possession_id"].eq(int(transition.possession_id))
                & annotated["team_id"].eq(team_id)
                & (pd.to_numeric(annotated["_clock_seconds"], errors="coerce") <= float(transition.start_time) + 12.0)
            ].copy()
            points = [(float(transition.start_x), float(transition.start_y))]
            for _, row in window.iterrows():
                for x_value, y_value in [(row.get("_x", np.nan), row.get("_y", np.nan)),
                                         (row.get("_end_x", np.nan), row.get("_end_y", np.nan))]:
                    if pd.notna(x_value) and pd.notna(y_value):
                        points.append((float(x_value), float(y_value)))
            end_x, end_y = max(points, key=lambda point: point[0])
            paths.append({
                "start_x": float(transition.start_x),
                "start_y": float(transition.start_y),
                "end_x": end_x,
                "end_y": end_y,
                "shot": int(transition.transition_shots) > 0,
                "goal": int(transition.transition_goals) > 0,
                "minute": int(float(transition.start_time) // 60),
            })
        team_data[team_id] = {
            "total": total,
            "chances": chances,
            "shots": shots,
            "goals": goals,
            "xg": float(pd.to_numeric(transitions["transition_xG"], errors="coerce").fillna(0).sum()) if total else 0.0,
            "box_entries": int(pd.to_numeric(transitions["transition_box_entries"], errors="coerce").fillna(0).sum()) if total else 0,
            "paths": paths,
        }

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.text(0.055, 0.95, "Transition Map & Outcomes", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.908, "Each line runs from the regain/turnover location to the most advanced point reached inside 12 seconds", fontsize=10.5, color=MUTED)
    fig.add_artist(Line2D([0.055, 0.945], [0.872, 0.872], transform=fig.transFigure, color=GRID, lw=1))

    layouts = [
        (HOME_ID, [0.055, 0.145, 0.265, 0.675], [0.335, 0.215, 0.135, 0.515], 0.19),
        (AWAY_ID, [0.525, 0.145, 0.265, 0.675], [0.805, 0.215, 0.135, 0.515], 0.66),
    ]
    for team_id, pitch_rect, card_rect, title_x in layouts:
        data = team_data[team_id]
        fig.text(title_x, 0.835, TEAM_NAME[team_id], color=TEXT, fontsize=14,
                 fontweight="bold", ha="center")
        pitch = fig.add_axes(pitch_rect)
        pitch.axhspan(66.7 * PITCH_LENGTH / 100, PITCH_LENGTH, color=FOCUS, alpha=0.035, zorder=0)
        draw_long_pitch(pitch)
        ordered_paths = sorted(data["paths"], key=lambda item: (item["goal"], item["shot"]))
        for path in ordered_paths:
            sx, sy = player_position_xy([path["start_x"]], [path["start_y"]])
            ex, ey = player_position_xy([path["end_x"]], [path["end_y"]])
            if path["goal"]:
                color, alpha, width, marker, size, line_style = EVENT_HIGHLIGHT, 0.95, 2.25, "*", 75, "-"
            elif path["shot"]:
                color, alpha, width, marker, size, line_style = EVENT_SUCCESS, 0.82, 1.35, "D", 25, "-"
            else:
                color, alpha, width, marker, size, line_style = EVENT_NEUTRAL, 0.24, 0.62, None, 0, QUIET_DASH
            pitch.annotate("", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]),
                           arrowprops=dict(arrowstyle="-|>", color=color, alpha=alpha,
                                           lw=width, linestyle=line_style,
                                           mutation_scale=7 if not path["goal"] else 10), zorder=3)
            pitch.scatter(sx[0], sy[0], s=8, facecolors=BG, edgecolors=color,
                          linewidth=0.55, alpha=max(alpha, 0.35), zorder=4)
            if marker:
                pitch.scatter(ex[0], ey[0], s=size, marker=marker, color=color,
                              edgecolor=TEXT, linewidth=0.55, zorder=5)
            if path["goal"]:
                pitch.text(ex[0], ey[0] + 2.0, f"{path['minute']}′", color=EVENT_HIGHLIGHT,
                           fontsize=6.5, fontweight="bold", ha="center", zorder=6)

        card = fig.add_axes(card_rect)
        card.set_facecolor(PANEL); card.set_xlim(0, 1); card.set_ylim(0, 1); card.set_xticks([]); card.set_yticks([])
        for spine in card.spines.values():
            spine.set_color(GRID)
        side_title(card, "OUTCOMES")
        side_kpis(card, [
            ("Transitions", data["total"]),
            ("Created chance", data["chances"]),
            ("Goals", data["goals"]),
            ("Transition xG", f"{data['xg']:.2f}"),
            ("Box entries", data["box_entries"]),
        ], start=0.82, gap=0.135)
        chance_rate = 100 * data["chances"] / max(data["total"], 1)
        card.text(0.08, 0.065, f"Chance rate: {chance_rate:.1f}%", color=TEXT,
                  fontsize=7.5, fontweight="bold")

    legend = [
        Line2D([0], [0], color=EVENT_NEUTRAL, lw=1.35, alpha=0.55,
               linestyle=QUIET_DASH, label="Transition without shot"),
        Line2D([0], [0], color=EVENT_SUCCESS, lw=2, marker="D", markersize=5,
               label="Created a chance"),
        Line2D([0], [0], color=EVENT_HIGHLIGHT, lw=2.5, marker="*", markersize=8,
               label="Ended in a goal"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.067), ncol=3,
               frameon=False, labelcolor=TEXT, fontsize=8)
    fig.text(0.055, 0.032,
             "DEFINITION  Open-play regain/turnover that within 12s progresses ≥20m, reaches the final third/box, or produces a shot · restarts excluded.",
             color=NEUTRAL, fontsize=7.5)
    fig.text(0.945, 0.032, "ARROW = START → MOST ADVANCED POINT", ha="right", fontsize=7.5, color=NEUTRAL)
    return save(fig, "32_transition_outcomes.png")


def _game_state_durations(events):
    period = events.get("period_code", pd.Series("", index=events.index)).astype(str).str.lower()
    live = events[~period.isin(["pre", "prematch", "post", "postgame", "pso", "penaltyshootout"])].copy()
    live["_clock"] = pd.to_numeric(live["minute"], errors="coerce").fillna(0) * 60 + pd.to_numeric(live["second"], errors="coerce").fillna(0)
    shootout = as_bool(live.get("is_penalty_shootout", pd.Series(False, index=live.index)))
    goals = live[as_bool(live.get("is_goal", pd.Series(False, index=live.index))) & ~shootout].sort_values(["_clock", "event_id"], kind="stable").copy()
    # Game state is driven by the scoreline, so own goals must be credited to
    # the side that benefits from them.
    goals["_credited_team"] = base.credited_team(goals) if not goals.empty else pd.Series(dtype=float)
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
    specs = [("Shots", "shots", "{:.0f}"), ("xG", "xG", "{:.2f}"), ("Transitions", "transitions", "{:.0f}"), ("Box entries", "box_entries", "{:.0f}")]
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.text(0.055, 0.95, "Game-State Output by Match Situation", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.908, "Each card is one shared scoreboard situation · values are totals from possessions starting in that situation", fontsize=10.5, color=MUTED)
    fig.add_artist(Line2D([0.055, 0.945], [0.872, 0.872], transform=fig.transFigure, color=GRID, lw=1))

    timeline = fig.add_axes([0.075, 0.775, 0.85, 0.065])
    timeline.set_xlim(0, max(match_seconds, 1)); timeline.set_ylim(0, 1); timeline.axis("off")
    left = 0.0
    for scenario in scenarios:
        duration = durations[scenario["duration_key"]]
        if duration > 0:
            timeline.barh(0.56, duration, left=left, height=0.36, color=scenario["color"], alpha=0.82, edgecolor=BG, linewidth=1.0)
            if duration / max(match_seconds, 1) > 0.10:
                timeline.text(left + duration / 2, 0.56, scenario["title"], color=TEXT, fontsize=7.5, fontweight="bold", ha="center", va="center")
        left += duration
    timeline.text(0, 0.03, "MATCH TIME SHARE", color=MUTED, fontsize=6.8, ha="left")
    timeline.text(max(match_seconds, 1), 0.03, f"TOTAL · {match_seconds / 60:.1f} min", color=MUTED, fontsize=6.8, ha="right")
    legend_x = [0.15, 0.42, 0.70]
    for x, scenario in zip(legend_x, scenarios):
        minutes = durations[scenario["duration_key"]] / 60
        fig.add_artist(Rectangle((x, 0.735), 0.014, 0.014, transform=fig.transFigure,
                                 facecolor=scenario["color"], edgecolor=TEXT, lw=0.45, alpha=0.9))
        fig.text(x + 0.020, 0.742, f"{scenario['title'].title()} · {minutes:.1f} min", color=TEXT, fontsize=8, va="center")

    active_scenarios = [scenario for scenario in scenarios if durations[scenario["duration_key"]] >= 3.0]
    inactive_scenarios = [scenario for scenario in scenarios if durations[scenario["duration_key"]] < 3.0]
    if len(active_scenarios) == 3:
        card_positions = [[0.055, 0.16, 0.285, 0.535], [0.3575, 0.16, 0.285, 0.535], [0.66, 0.16, 0.285, 0.535]]
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
        fig.text(0.50, 0.705, f"NOT PLAYED · {note}", color=FOCUS, fontsize=8.2,
                 fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.38", fc=PANEL, ec=GRID, lw=0.7))
    for rect, scenario in zip(card_positions, active_scenarios):
        ax = fig.add_axes(rect)
        ax.set_facecolor(PANEL); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(scenario["color"]); spine.set_linewidth(1.2)
        duration_minutes = durations[scenario["duration_key"]] / 60
        ax.text(0.06, 0.93, scenario["title"], color=TEXT, fontsize=12, fontweight="bold", va="top")
        ax.text(0.06, 0.865, scenario["subtitle"], color=MUTED, fontsize=7.4, va="top")
        ax.text(0.94, 0.93, f"{duration_minutes:.1f} min", color=TEXT, fontsize=9,
                fontweight="bold", ha="right", va="top")
        ax.plot([0.06, 0.94], [0.81, 0.81], color=GRID, lw=0.8)
        ax.text(0.57, 0.755, HOME_NAME.upper(), color=HOME, fontsize=7.5, fontweight="bold", ha="center")
        ax.text(0.84, 0.755, AWAY_NAME.upper(), color=AWAY, fontsize=7.5, fontweight="bold", ha="center")
        for idx, (label, key, fmt) in enumerate(specs):
            y = 0.66 - idx * 0.135
            home_value = base.metric_lookup(team_metrics, "home", f"game_state_{scenario['home_state']}_{key}")
            away_value = base.metric_lookup(team_metrics, "away", f"game_state_{scenario['away_state']}_{key}")
            ax.text(0.07, y, label, color=TEXT, fontsize=8.5, va="center")
            ax.text(0.57, y, fmt.format(home_value), color=TEXT, fontsize=12, fontweight="bold", ha="center", va="center")
            ax.text(0.84, y, fmt.format(away_value), color=TEXT, fontsize=12, fontweight="bold", ha="center", va="center")
            ax.plot([0.07, 0.93], [y - 0.064, y - 0.064], color=GRID, lw=0.55, alpha=0.75)
        ax.text(0.07, 0.075, "Values are totals, not per-90 rates.", color=NEUTRAL, fontsize=6.7)

    fig.text(0.055, 0.095, f"HOW TO READ  {HOME_NAME} ahead pairs {HOME_NAME}'s leading output with {AWAY_NAME}'s trailing output; {AWAY_NAME} ahead does the reverse.", color=MUTED, fontsize=8.2)
    fig.text(0.055, 0.066, "Game state is assigned at possession start · Timeline duration is reconstructed from goal times.", color=NEUTRAL, fontsize=7.5)
    fig.text(0.945, 0.035, f"{HOME_NAME.upper()} · {AWAY_NAME.upper()} · REAL MATCH DATA", ha="right", fontsize=7.5, color=NEUTRAL)
    return save(fig, "33_game_state_splits.png")


def player_sequence(player_metrics):
    metrics = [("xGChain", "xGChain"), ("xGBuildup", "xGBuildup"), ("Sequence xT", "sequence_xT")]
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.text(0.055, 0.95, "Player Sequence Contribution", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.91, "Top players by xGChain, xGBuildup and sequence xT involvement", fontsize=11, color=MUTED)
    axes = fig.subplots(1, 3); fig.subplots_adjust(left=0.075, right=0.96, top=0.82, bottom=0.10, wspace=0.42)
    for ax, (title, column) in zip(axes, metrics):
        top = player_metrics.sort_values(column, ascending=False).head(8).sort_values(column)
        colors = [HOME if str(team).lower() == HOME_NAME.lower() else AWAY for team in top["team"]]
        ax.barh(top["player"].astype(str).str.split().str[-1], top[column], color=colors, alpha=0.9)
        base.clean_ax(ax); ax.grid(axis="x", color=GRID, lw=0.65); ax.set_title(title.upper(), loc="left", color=MUTED, fontsize=10, fontweight="bold")
        for idx, value in enumerate(top[column]): ax.text(value + max(top[column].max(), 0.01) * 0.025, idx, f"{value:.2f}", color=TEXT, va="center", fontsize=8)
    fig.text(0.945, 0.035, f"{HOME_NAME.upper()} · {AWAY_NAME.upper()}", ha="right", fontsize=8, color=NEUTRAL)
    return save(fig, "34_player_sequence_leaders.png")


def momentum(events):
    """Windowed xG differential — who was actually on top, and when."""
    frame = xg_momentum(events, HOME_ID, AWAY_ID, window=5)
    fig, ax = base.page(
        "Match Momentum",
        "Expected-goal difference per five-minute window \u00b7 the xG flow shows who finished ahead, this shows when",
    )
    base.clean_ax(ax)
    if frame.empty:
        ax.text(0.5, 0.5, "No shots recorded", color=MUTED, ha="center", va="center")
        return save(fig, "35_match_momentum.png")

    starts = frame["window_start"].to_numpy()
    diff = frame["differential"].to_numpy()
    colors = [HOME if value >= 0 else AWAY for value in diff]
    ax.bar(starts + 2.5, diff, width=4.4, color=colors, alpha=0.9, edgecolor=BG, linewidth=0.8)
    ax.axhline(0, color=PITCH_LINE, lw=1.0, alpha=0.55)
    ax.axvline(45, color=GRID, lw=0.9, ls=(0, (3, 4)))
    ax.text(45, ax.get_ylim()[1], " HT", color=MUTED, fontsize=7, va="top")

    ax.set_xlabel("Match minute", fontsize=9, color=MUTED)
    ax.set_ylabel("xG difference in window", fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
    ax.tick_params(labelsize=8)

    limit = max(float(np.abs(diff).max()), 0.05) * 1.35
    ax.set_ylim(-limit, limit)
    ax.text(0.005, 0.97, HOME_NAME.upper(), transform=ax.transAxes, color=HOME,
            fontsize=9, fontweight="bold", va="top")
    ax.text(0.005, 0.03, AWAY_NAME.upper(), transform=ax.transAxes, color=AWAY,
            fontsize=9, fontweight="bold", va="bottom")

    best_home = frame.loc[frame["differential"].idxmax()]
    best_away = frame.loc[frame["differential"].idxmin()]
    fig.text(0.08, 0.815,
             f"Strongest spell \u00b7 {HOME_NAME}: {int(best_home['window_start'])}\u2013{int(best_home['window_start']) + 5}'  "
             f"({best_home['differential']:+.2f})     "
             f"{AWAY_NAME}: {int(best_away['window_start'])}\u2013{int(best_away['window_start']) + 5}'  "
             f"({best_away['differential']:+.2f})",
             color=MUTED, fontsize=8.5)
    return save(fig, "35_match_momentum.png")


def set_pieces(events):
    """Where each side's shots came from, and what the dead ball was worth."""
    sources = [
        ("open_play", "Open play"),
        ("corner", "Corners"),
        ("free_kick", "Free kicks"),
        ("throw_in", "Throw-ins"),
        ("penalty", "Penalties"),
    ]
    home = set_piece_breakdown(events, HOME_ID)
    away = set_piece_breakdown(events, AWAY_ID)

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
        dead_ball_xg = sum(data[key]["xG"] for key, _ in sources if key != "open_play")
        total_xg = sum(data[key]["xG"] for key, _ in sources)
        share = 100 * dead_ball_xg / max(total_xg, 0.01)
        fig.text(left, 0.768, f"{share:.0f}% of xG came from a dead ball", color=MUTED, fontsize=8.5)

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
        ax.set_xlim(0, max(max(shots), 1) * 1.35)
        ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
        ax.set_xlabel("Shots", fontsize=8.5, color=MUTED)
        ax.tick_params(labelsize=8)
        for position, (count, value, (key, _label)) in enumerate(zip(shots, xgs, sources)):
            if count == 0:
                continue
            ax.text(count + max(max(shots), 1) * 0.03, position,
                    f"{count}  \u00b7  {value:.2f} xG  \u00b7  {data[key]['goals']}G",
                    color=TEXT, fontsize=8, va="center", fontweight="bold")

    fig.text(0.945, 0.035, "FULL VISUAL REDESIGN \u00b7 REAL MATCH DATA",
             ha="right", fontsize=8, color=NEUTRAL)
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
            pitch.scatter(px, py, s=34, marker="o", facecolors="none",
                          edgecolors=team_mark, linewidths=0.9, alpha=0.55, zorder=4)
        if not punished.empty:
            px, py = attack_xy(punished["x"], punished["y"])
            pitch.scatter(px, py, s=70 + punished["conceded_xG"].to_numpy() * 900,
                          marker="o", facecolors=SHOT_GOAL, edgecolors=BG,
                          linewidths=0.9, alpha=0.95, zorder=6)

    total = len(frame)
    punished_count = int(frame["punished"].sum()) if not frame.empty else 0
    conceded = float(frame["conceded_xG"].sum()) if not frame.empty else 0.0
    own_half = int((frame["x"] < 50).sum()) if not frame.empty else 0
    side_title(side, "LOSS PROFILE")
    side_kpis(side, [
        ("Possessions lost", f"{total}"),
        ("Punished", f"{punished_count}"),
        ("Punish rate", f"{100 * punished_count / max(total, 1):.0f}%"),
        ("xG conceded from losses", f"{conceded:.2f}"),
        ("Lost in own half", f"{own_half}"),
    ])
    return save(fig, f"{number:02d}_ball_losses_{_team_slug(team_id)}.png")


def shape_over_time(events):
    """Defensive line height and compactness, sampled across the match."""
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    base.amoled_header(
        fig,
        "Defensive Shape Over Time",
        "Engagement height and compactness per five-minute window \u00b7 a single average hides when a side dropped off",
    )
    height_ax = fig.add_axes([0.075, 0.475, 0.86, 0.29])
    spread_ax = fig.add_axes([0.075, 0.135, 0.86, 0.27])
    for ax in (height_ax, spread_ax):
        base.clean_ax(ax)
        ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
        ax.tick_params(labelsize=8)
        ax.axvline(45, color=GRID, lw=0.9, ls=(0, (3, 4)))

    for team_id, color in [(HOME_ID, HOME), (AWAY_ID, AWAY)]:
        height = defensive_line_height(events, team_id)
        if not height.empty:
            height_ax.plot(height["window_start"] + 2.5, height["height"],
                           color=color, lw=2.4, marker="o", markersize=3.4,
                           markeredgecolor=BG, label=TEAM_NAME[team_id])
        compact = team_compactness(events, team_id)
        if not compact.empty:
            spread_ax.plot(compact["window_start"] + 2.5, compact["vertical_spread"],
                           color=color, lw=2.4, marker="o", markersize=3.4,
                           markeredgecolor=BG, label=TEAM_NAME[team_id])

    height_ax.set_ylabel("Mean x of defensive actions", fontsize=8.5, color=MUTED)
    height_ax.set_ylim(0, 100)
    height_ax.axhline(50, color=PITCH_LINE, lw=0.8, alpha=0.35)
    height_ax.text(1, 51, "halfway", color=MUTED, fontsize=6.5, va="bottom")
    height_ax.legend(loc="upper right", frameon=False, labelcolor=TEXT, fontsize=8, ncol=2)

    spread_ax.set_ylabel("Vertical spread of touches (IQR)", fontsize=8.5, color=MUTED)
    spread_ax.set_xlabel("Match minute", fontsize=9, color=MUTED)

    fig.text(0.945, 0.035, "FULL VISUAL REDESIGN \u00b7 REAL MATCH DATA",
             ha="right", fontsize=8, color=NEUTRAL)
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
        minutes, home, draw, away,
        colors=[HOME, NEUTRAL, AWAY], alpha=0.88,
        labels=[HOME_NAME, "Draw", AWAY_NAME],
    )
    ax.set_xlim(minutes.min(), minutes.max())
    ax.set_ylim(0, 100)
    ax.set_xlabel("Match minute", fontsize=9, color=MUTED)
    ax.set_ylabel("Probability (%)", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8)
    ax.axvline(45, color=BG, lw=1.1, ls=(0, (3, 4)))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              frameon=False, labelcolor=TEXT, fontsize=8.5)

    # Mark every goal: the step in the curve is the point of the chart.
    goals = events[as_bool(events["is_goal"])].copy()
    if not goals.empty:
        goals["_credited"] = base.credited_team(goals)
        for _, goal in goals.sort_values("minute").iterrows():
            minute = float(goal["minute"])
            color = HOME if int(goal["_credited"]) == HOME_ID else AWAY
            ax.axvline(minute, color=BG, lw=2.4, alpha=0.9)
            ax.axvline(minute, color=color, lw=1.1)
            ax.text(minute, 102, f"{int(minute)}\u2032", color=color, fontsize=6.4,
                    fontweight="bold", ha="center", va="bottom")
    return save(fig, "40_win_probability.png")


def playing_through(events, team_id, opponent_id, number):
    """Passes that beat the opponent's line, and how the side held up when pressed."""
    breaks = line_breaking_passes(events, team_id, opponent_id)
    fig, pitch, side = pitch_axes(
        f"Playing Through \u00b7 {TEAM_NAME[team_id]}",
        "Passes starting behind the opponent's defensive line and finishing beyond it \u00b7 line height per five-minute window",
    )
    draw_long_pitch(pitch)
    team_mark = _team_mark_color(team_id)

    if not breaks.empty:
        for _, row in breaks.iterrows():
            sx, sy = attack_xy([row["x"]], [row["y"]])
            ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
            completed = bool(row["successful"])
            pitch.annotate(
                "", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]),
                arrowprops=dict(
                    arrowstyle="-|>", color=team_mark if completed else EVENT_NEUTRAL,
                    alpha=0.85 if completed else 0.4,
                    lw=1.35 if completed else 0.75,
                    linestyle="-" if completed else FAILURE_DASH,
                    mutation_scale=9,
                ),
            )

    resistance = press_resistance(events, team_id)
    completed = int(breaks["successful"].sum()) if not breaks.empty else 0
    side_title(side, "PLAYING THROUGH")
    side_kpis(side, [
        ("Line-breaking passes", f"{len(breaks)}"),
        ("Completed", f"{completed}"),
        ("Completion", f"{100 * completed / max(len(breaks), 1):.0f}%"),
    ], start=0.82, gap=0.13)

    side.text(0.08, 0.40, "UNDER PRESSURE", color=MUTED, fontsize=7.5, fontweight="bold")
    for idx, (label, value) in enumerate([
        ("Passes pressed", f"{resistance['passes_under_pressure']}"),
        ("Share of passes", f"{resistance['pressed_share']:.0f}%"),
        ("Completion pressed", f"{resistance['pressed_completion']:.0f}%"),
        ("Completion free", f"{resistance['free_completion']:.0f}%"),
        ("Resistance gap", f"{resistance['resistance_gap']:+.0f} pts"),
    ]):
        y = 0.35 - idx * 0.048
        side.text(0.08, y, label, color=TEXT, fontsize=8, va="center")
        side.text(0.92, y, value, color=TEXT, fontsize=8.5, fontweight="bold",
                  ha="right", va="center")

    duels = duel_map(events, team_id)
    if not duels.empty:
        won = int(duels["won"].sum())
        aerial = duels[duels["kind"] == "aerial"]
        side.text(0.08, 0.085, "DUELS", color=MUTED, fontsize=7.5, fontweight="bold")
        side.text(
            0.08, 0.04,
            f"{won}/{len(duels)} won  \u00b7  aerial {int(aerial['won'].sum())}/{len(aerial)}",
            color=TEXT, fontsize=8, va="center",
        )

    pitch.plot([], [], color=team_mark, lw=1.35, label="Completed")
    pitch.plot([], [], color=EVENT_NEUTRAL, lw=0.9, ls=FAILURE_DASH, label="Incomplete")
    pitch.legend(loc="lower center", bbox_to_anchor=(0.5, -0.075), ncol=2,
                 frameon=False, labelcolor=TEXT, fontsize=7.5)
    return save(fig, f"{number:02d}_playing_through_{_team_slug(team_id)}.png")


def action_value_leaders(events):
    """One value ranking every position can appear in."""
    ranked = player_action_value(events)
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
    ax.set_yticklabels([str(name).split()[-1] for name in top["player"]], fontsize=8.5, color=TEXT)
    ax.axvline(0, color=PITCH_LINE, lw=1.0, alpha=0.6)
    ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
    ax.set_xlabel("Action value (goals)", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8)
    # Label at the bar's right edge, not at the total. When a player lost value
    # on the ball the bar starts left of zero, so the two do not coincide and
    # the number ended up printed across the bar.
    for index, (off, dfn, total) in enumerate(
        zip(offensive, defensive, top["total_value"].to_numpy())
    ):
        ax.text(max(off, 0.0) + max(dfn, 0.0) + 0.012, index, f"{total:+.3f}",
                color=TEXT, fontsize=8, fontweight="bold", va="center")

    fig.text(0.20, 0.80, "SOLID = ON-BALL   ·   FADED = DEFENSIVE", color=MUTED,
             fontsize=7.5, fontweight="bold")
    fig.text(0.20, 0.075,
             "Zone-value model, not a fitted VAEP: no labelled training data, so the value surface is explicit rather than learned.",
             color=NEUTRAL, fontsize=7)
    fig.text(0.945, 0.035, "FULL VISUAL REDESIGN · REAL MATCH DATA",
             ha="right", fontsize=8, color=NEUTRAL)
    return save(fig, "43_action_value.png")


def control_surface(events):
    """Which side held which parts of the pitch, and where it was contested."""
    grid, shares = pitch_control(events, HOME_ID, AWAY_ID)
    fig, pitch, side = pitch_axes(
        "Pitch Control",
        "Influence decays with distance · space is held strongly, weakly, or genuinely contested",
    )
    cmap = LinearSegmentedColormap.from_list("control", [AWAY, PANEL_2, HOME])
    # grid is indexed [pitch_y, pitch_x]; this display puts pitch x up the page,
    # so it has to be transposed or the control reads across the pitch instead
    # of up it.
    pitch.imshow(
        grid.T,
        extent=[-PITCH_WIDTH / 2, PITCH_WIDTH / 2, 0, PITCH_LENGTH],
        origin="lower", cmap=cmap, vmin=0.25, vmax=0.75, aspect="equal", alpha=0.92,
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
        pitch.scatter(px, py, s=44, marker="o", facecolors=color,
                      edgecolors=BG, linewidths=1.0, zorder=6)

    # Everything on this map is encoded: the field's colour is which side held
    # the space, the dots are average positions. Neither was stated anywhere.
    pitch_legend(pitch, [
        ("patch", HOME, f"{HOME_NAME} holds"),
        ("patch", PANEL_2, "Contested"),
        ("patch", AWAY, f"{AWAY_NAME} holds"),
        ("o", TEXT, "Average position"),
    ], ncol=4)

    side_title(side, "TERRITORY")
    side_kpis(side, [
        (f"{HOME_NAME} control", f"{shares['home']:.0f}%"),
        (f"{AWAY_NAME} control", f"{shares['away']:.0f}%"),
        ("Contested", f"{shares['contested']:.0f}%"),
    ], start=0.82, gap=0.14)
    side.text(0.08, 0.36,
              "Contested space is not neutral: it is\nwhere both sides committed bodies and\nneither held a clear advantage.",
              color=MUTED, fontsize=7.6, va="top", linespacing=1.6)
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
    scale_max = max(
        [float(frame["xG"].max()) for frame in both.values() if not frame.empty] or [0.1]
    ) * 1.45

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
            ax.text(float(row["xG"]) + scale_max * 0.02, index,
                    f"{int(row['sequences'])} seq  \u00b7  {int(row['goals'])}G  \u00b7  {float(row['share_of_xG']):.0f}% of xG",
                    color=TEXT, fontsize=7.6, va="center", fontweight="bold")

        if not typology.empty:
            best = typology["xG"].idxmax()
            fig.text(left, 0.768,
                     f"Most dangerous route: {labels[best].lower()} ({typology.loc[best, 'share_of_xG']:.0f}% of xG)",
                     color=MUTED, fontsize=8.5)

    fig.text(0.945, 0.035, "FULL VISUAL REDESIGN \u00b7 REAL MATCH DATA",
             ha="right", fontsize=8, color=NEUTRAL)
    return save(fig, "45_sequence_types.png")


def goal_origins(events):
    """The sequence behind every goal, not just the scorer and the minute."""
    chains = goal_origin_chains(events, HOME_ID, AWAY_ID)
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    base.amoled_header(
        fig,
        "Goal Origins",
        "Where each goal began, how long it took and how many players touched it",
    )
    if chains.empty:
        fig.text(0.5, 0.5, "No goals from open sequences", color=MUTED, ha="center")
        return save(fig, "46_goal_origins.png")

    headers = ["MIN", "SCORER", "ROUTE", "PASSES", "SECONDS", "PLAYERS", "STARTED"]
    xs = [0.065, 0.115, 0.30, 0.44, 0.53, 0.63, 0.73]
    top = 0.755
    for x, header in zip(xs, headers):
        fig.text(x, top, header, color=MUTED, fontsize=7.6, fontweight="bold")
    fig.add_artist(Line2D([0.06, 0.94], [top - 0.018, top - 0.018],
                          transform=fig.transFigure, color=GRID, lw=1))

    row_height = min(0.055, 0.60 / max(len(chains), 1))
    for index, row in enumerate(chains.itertuples()):
        y = top - 0.045 - index * row_height
        color = HOME if int(row.team_id) == HOME_ID else AWAY
        values = [
            f"{int(row.minute)}'",
            str(row.scorer).split()[-1][:16],
            str(row.sequence_type).replace("_", " ").title(),
            str(int(row.passes)),
            f"{float(row.duration):.0f}",
            str(int(row.players)),
            str(row.started_from).replace("_", " "),
        ]
        for x, value, is_first in zip(xs, values, [True] + [False] * 6):
            fig.text(x, y, value, color=color if is_first else TEXT,
                     fontsize=8.4, fontweight="bold" if is_first else "normal", va="center")
        fig.add_artist(Line2D([0.06, 0.94], [y - row_height * 0.42, y - row_height * 0.42],
                              transform=fig.transFigure, color=GRID, lw=0.5, alpha=0.6))

    fig.text(0.06, 0.075,
             "Route is read from the possession the goal ended, so a first-time finish from a regain shows as a counter with few passes.",
             color=NEUTRAL, fontsize=7)
    fig.text(0.945, 0.035, "FULL VISUAL REDESIGN \u00b7 REAL MATCH DATA",
             ha="right", fontsize=8, color=NEUTRAL)
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
        pitch.scatter(px, py, s=52, marker="o", facecolors=team_mark,
                      edgecolors=BG, linewidths=0.9, alpha=0.9, zorder=6)
        # The estimated line the receptions were taken behind.
        mean_line = float(pockets["line_height"].mean())
        line_y = mean_line * PITCH_LENGTH / 100.0
        pitch.plot([-PITCH_WIDTH / 2, PITCH_WIDTH / 2], [line_y, line_y],
                   color=EVENT_NEUTRAL, lw=1.1, ls=(0, (5, 4)), alpha=0.7, zorder=3)
        pitch.text(PITCH_WIDTH / 2, line_y + 0.8, "avg line", color=MUTED,
                   fontsize=6.4, ha="right", va="bottom")
        # Neither mark said what it was: one dot is a single reception, and
        # "avg line" alone does not tell the reader whose line it is.
        pitch_legend(pitch, [
            ("o", team_mark, "Reception in the pocket"),
            ("_", EVENT_NEUTRAL, f"{TEAM_NAME[opponent_id]} average defensive line"),
        ], ncol=2)

    side_title(side, "PLAYING THROUGH")
    side_kpis(side, [
        ("Receptions in pocket", f"{len(pockets)}"),
        ("Switches of play", f"{len(switches)}"),
        ("Reached final third", f"{tempo['reach_rate']:.0f}%"),
    ], start=0.82, gap=0.13)

    side.text(0.08, 0.40, "TEMPO", color=MUTED, fontsize=7.5, fontweight="bold")
    side.text(0.08, 0.355, "Median regain to final third", color=TEXT, fontsize=8, va="center")
    side.text(0.92, 0.355, f"{tempo['median_seconds']:.1f}s", color=TEXT, fontsize=8.5,
              fontweight="bold", ha="right", va="center")

    if not pockets.empty:
        receivers = pockets[pockets["player"] != ""]["player"].value_counts().head(4)
        side.text(0.08, 0.28, "MOST OFTEN IN THE POCKET", color=MUTED, fontsize=7.5, fontweight="bold")
        for index, (name, count) in enumerate(receivers.items()):
            y = 0.235 - index * 0.048
            side.text(0.08, y, str(name).split()[-1][:14], color=TEXT, fontsize=8, va="center")
            side.text(0.92, y, str(int(count)), color=TEXT, fontsize=8.5,
                      fontweight="bold", ha="right", va="center")
    return save(fig, f"{number:02d}_unlocking_{_team_slug(team_id)}.png")


def press_and_rest(events):
    """What the press fed on, and what each side left behind the ball."""
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    base.amoled_header(
        fig,
        "Press Triggers & Rest Defence",
        "What the opponent was doing when the ball was won high, and how many bodies were behind the ball when it was lost",
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
            ax.barh(positions, triggers["regains"], height=0.6, color=color, alpha=0.9)
            ax.set_yticks(positions)
            ax.set_yticklabels(triggers["trigger"], fontsize=8.5, color=TEXT)
            ax.invert_yaxis()
            ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
            ax.set_xlabel("High regains", fontsize=8.5, color=MUTED)
            ax.tick_params(labelsize=8)
            for index, (count, share) in enumerate(zip(triggers["regains"], triggers["share"])):
                ax.text(count + 0.15, index, f"{int(count)}  ({share:.0f}%)", color=TEXT,
                        fontsize=7.8, va="center", fontweight="bold")

        structure = rest_defence_structure(events, team_id)
        second = second_ball_recovery(events, team_id)
        rows = [
            ("Players behind the ball at loss", f"{structure['avg_players_behind']:.1f}"),
            ("Losses with three or fewer behind", f"{structure['exposed_losses']} ({structure['exposed_share']:.0f}%)"),
            ("Second balls won", f"{second['won']}/{second['contests']} ({second['win_rate']:.0f}%)"),
        ]
        fig.text(left, 0.30, "REST DEFENCE & SECOND BALLS", color=MUTED, fontsize=7.8, fontweight="bold")
        for index, (label, value) in enumerate(rows):
            y = 0.255 - index * 0.045
            fig.text(left, y, label, color=TEXT, fontsize=8.4)
            fig.text(left + 0.39, y, value, color=TEXT, fontsize=8.6,
                     fontweight="bold", ha="right")

    fig.text(0.945, 0.035, "FULL VISUAL REDESIGN \u00b7 REAL MATCH DATA",
             ha="right", fontsize=8, color=NEUTRAL)
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

    from tactical_pdf_report import build_tactical_pdf

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


def player_pizzas(events: pd.DataFrame) -> list[Path]:
    """Export every player's radar, and return the five per side for the report."""
    from player_radar import export_player_radars

    info = {
        "home_id": HOME_ID,
        "away_id": AWAY_ID,
        "home_name": HOME_NAME,
        "away_name": AWAY_NAME,
        "home_color": HOME,
        "away_color": AWAY,
        "score": MATCH_SCORE,
    }
    ranking = export_player_radars(events, info, str(OUT), dpi=135)

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
            match = next((f for f in team_dir.glob("*.png")
                          if f.stem.lower() == wanted), None)
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
        from match_metrics import post_shot_xg
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
        shots = events[events["team_id"].eq(team_id)]
        corrected.loc[rows, "xGoT"] = round(float(post_shot_xg(shots).sum()), 2)
    return corrected


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
    team_metrics.to_csv(OUT / "team_advanced_metrics.csv", index=False, encoding="utf-8-sig")
    player_metrics.to_csv(OUT / "player_sequence_metrics.csv", index=False, encoding="utf-8-sig")
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
        shot_map(events, xg, HOME_ID, 2),
        shot_map(events, xg, AWAY_ID, 3),
        goals_breakdown(events),
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
        post_match_advanced_dashboard(events, xg, team_metrics),
        generated["15_xt_per_minute.png"],
        progressive(events, HOME_ID, 16),
        progressive(events, AWAY_ID, 17),
        crosses(events, HOME_ID, 18),
        crosses(events, AWAY_ID, 19),
        defensive_activity(events, HOME_ID, 20),
        defensive_activity(events, AWAY_ID, 21),
        average_positions(events, players, HOME_ID, 22, 1),
        average_positions(events, players, HOME_ID, 22, 2),
        average_positions(events, players, AWAY_ID, 23, 1),
        average_positions(events, players, AWAY_ID, 23, 2),
        dominating_zones(events),
        box_entries(events, HOME_ID, 25),
        box_entries(events, AWAY_ID, 26),
        high_regains(events, HOME_ID, 27),
        high_regains(events, AWAY_ID, 28),
        pass_targets(events, HOME_ID, 29),
        pass_targets(events, AWAY_ID, 30),
        ppda(events),
        transition_outcomes(events),
        game_state(events, team_metrics),
        player_sequence(player_metrics),
        momentum(events),
        set_pieces(events),
        turnovers(events, HOME_ID, 37),
        turnovers(events, AWAY_ID, 38),
        shape_over_time(events),
        win_probability_curve(events),
        playing_through(events, HOME_ID, AWAY_ID, 41),
        playing_through(events, AWAY_ID, HOME_ID, 42),
        action_value_leaders(events),
        control_surface(events),
        sequence_types(events),
        goal_origins(events),
        unlocking_the_block(events, HOME_ID, AWAY_ID, 47),
        unlocking_the_block(events, AWAY_ID, HOME_ID, 48),
        press_and_rest(events),
    ]
    paths.extend(player_pizzas(events))
    paths = sorted({path.resolve() for path in paths}, key=lambda path: path.name)
    catalog = build_catalog(paths)
    pdf = build_pdf(paths, events, xg, team_metrics, player_metrics)
    from match_posters import build_match_posters
    posters = build_match_posters(
        events, xg, team_metrics, player_metrics, players,
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
    from match_article import build_match_article
    article = build_match_article(events, xg, team_metrics, player_metrics,
                                  match_info, OUT, players)

    print(f"Generated {len(paths)} full redesigned visuals")
    print(f"Pitch visuals: {int(catalog['has_pitch'].sum())}")
    print(f"Posters: {len(posters)}")
    print(f"Article: {article}" if article else "Article: not written")
    print(f"PDF: {pdf}")
    return {
        "visuals": paths,
        "catalog": OUT / "visual_catalog.csv",
        "pdf": pdf,
        "posters": posters,
        "article": article,
        "output_dir": OUT,
    }


def _sample_kit_colors() -> tuple[str, str]:
    """Resolve the sample fixture's kit colours the same way production does.

    Imported lazily: football_match_analysis imports this module, so a
    top-level import here would be circular.
    """
    if not USE_REAL_TEAM_KIT_COLORS:
        return HOME, AWAY
    try:
        from football_match_analysis import choose_matchup_colors

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
