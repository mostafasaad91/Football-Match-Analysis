"""Two post-match posters, each complete on its own.

These replace the twelve contact sheets. The sheets composited already-rendered
PNGs -- every visual was reopened, scaled to a 1180x720 thumbnail and pasted,
which shrank the type inside it past reading. Here every panel is drawn onto
the poster's own canvas at the poster's own scale, so a number on a poster is
as sharp as a number anywhere else in the project.

Four boards, so a fixture is covered without the report around it:

1. POST-MATCH REPORT -- what happened. Passing shape, sixteen indicators, the
   shot map, and who was on top minute by minute.
2. HOW IT WAS PLAYED -- why. Territory, progression, the penalty area,
   pressing, and who connected the danger.
3. THE TRANSITION GAME -- the match between the phases. Where possession was
   surrendered, what the scoreline did to each side, delivery from wide, and
   the middle nobody wants to give up.
4. THE FINAL BALL -- the last action. Every shot, where it crossed the line,
   what the restarts produced, and how much of the pitch each side had to
   cross to get there.

Both are 1640x2048 -- 4:5, the tallest frame a timeline shows without cropping.
"""
from __future__ import annotations

import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Rectangle
from scipy.ndimage import gaussian_filter

import crests
from match_metrics import (
    box_entry_mask,
    cross_mask,
    deep_completion_mask,
    final_third_entry_mask,
    high_regain_events,
    pass_length_profile,
    pitch_control,
    progressive_pass_mask,
    set_piece_breakdown,
    shot_placement_zones,
    touch_mask,
    turnover_events,
    xg_momentum,
)
from visualization_components import IS_LIGHT_THEME, text_on_fill

if IS_LIGHT_THEME:
    BG, PANEL, GRID = "#F5F5F5", "#FFFFFF", "#CCCCCC"
    TEXT, MUTED, NEUTRAL, LINE = "#1A1A1A", "#555B63", "#7C838C", "#B9BEC5"
else:
    BG, PANEL, GRID = "#000000", "#08090B", "#22262D"
    TEXT, MUTED, NEUTRAL, LINE = "#F5F7FA", "#9BA3AE", "#626A75", "#31363E"

# Marks that have to separate from the page rather than carry a team: the
# goal star's rim, the average-height rule, a scatter edge. Written as #FFFFFF
# throughout the first pass, which is right on black and invisible on paper.
INK = "#1A1A1A" if IS_LIGHT_THEME else "#FFFFFF"
# Network links are drawn in the team colour at partial alpha. The same
# 0.34 that reads as a bright line on black washes a dark line out on paper.
LINK_ALPHA = 0.55 if IS_LIGHT_THEME else 0.34

PITCH_LENGTH, PITCH_WIDTH = 105.0, 58.0

W_PX, H_PX = 1640, 2048
DPI = 200
# Figure size is in inches and the saved pixel count is truncated, not rounded,
# so a width that is not exactly representable in binary loses its last pixel:
# 1640/200 stores as 8.199999999999999 and saved a 1639-wide poster. Half a
# pixel of slack lands on the intended size without moving anything.
FIGSIZE = ((W_PX + 0.5) / DPI, (H_PX + 0.5) / DPI)

# Three columns and three rows of panels, in figure fractions. Held as
# constants because every panel title, rule and label is positioned against
# them -- the contact sheets recomputed spacing per sheet and drifted.
COL = {"left": (0.030, 0.316), "mid": (0.352, 0.648), "right": (0.684, 0.970)}

# Each row is a block of title, axes and footnote. Sized as one unit because
# the first pass tied only the axes to the grid: every pitch then printed its
# footnote into the title of the row below it.
ROW_TOP = 0.9045          # title baseline of the first row
ROW_PITCH = 0.288         # distance between one row's title and the next
AXES_HEIGHT = 0.222
TITLE_GAP = 0.020         # title baseline down to the top of its axes


def _row_bounds(row: int) -> tuple[float, float]:
    top = ROW_TOP - row * ROW_PITCH
    y1 = top - TITLE_GAP
    return y1 - AXES_HEIGHT, y1


def _axes(fig, column: str, row: int, *, span=None):
    x0, x1 = COL[column]
    if span:
        x0 = COL[span[0]][0]
        x1 = COL[span[1]][1]
    y0, y1 = _row_bounds(row)
    ax = fig.add_axes([x0, y0, x1 - x0, y1 - y0])
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def _panel_title(fig, column: str, row: int, title: str, *, span=None):
    """Panel heading.

    A right-aligned note used to share this line. A third of the poster does
    not hold a title and a note together once a club is called "Aston Villa",
    and the club name -- not the layout -- decides how wide the title is. What
    each panel encodes is stated in the footnote beneath it instead.
    """
    x0 = COL[span[0]][0] if span else COL[column][0]
    y = ROW_TOP - row * ROW_PITCH
    fig.text(x0, y, title.upper(), color=TEXT, fontsize=8.6, fontweight="bold", va="bottom")


# --------------------------------------------------------------------------
# pitch
# --------------------------------------------------------------------------

def _pitch(ax, lw=0.75, margin=4.0, *, attack: bool | None = None):
    """Draw the pitch. ``attack`` marks which way the side plays, when given."""
    half = PITCH_WIDTH / 2
    ax.set_xlim(-half - margin, half + margin)
    ax.set_ylim(-margin, PITCH_LENGTH + margin)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-half, 0), PITCH_WIDTH, PITCH_LENGTH, fill=False, ec=LINE, lw=lw))
    ax.plot([-half, half], [PITCH_LENGTH / 2] * 2, color=LINE, lw=lw)
    ax.add_patch(Circle((0, PITCH_LENGTH / 2), PITCH_LENGTH * 0.0915, fill=False, ec=LINE, lw=lw))
    box_w, six_w = PITCH_WIDTH * 0.595, PITCH_WIDTH * 0.265
    box_l, six_l = PITCH_LENGTH * 0.157, PITCH_LENGTH * 0.052
    for base_y, direction in ((0.0, 1), (PITCH_LENGTH, -1)):
        top = base_y if direction > 0 else base_y - box_l
        ax.add_patch(Rectangle((-box_w / 2, top), box_w, box_l, fill=False, ec=LINE, lw=lw))
        top6 = base_y if direction > 0 else base_y - six_l
        ax.add_patch(Rectangle((-six_w / 2, top6), six_w, six_l, fill=False, ec=LINE, lw=lw))
        spot = base_y + direction * PITCH_LENGTH * 0.105
        ax.scatter([0], [spot], s=3, color=LINE)
        if direction > 0:
            ax.add_patch(Arc((0, spot), 18.3, 18.3, theta1=37, theta2=143, ec=LINE, lw=lw))
        else:
            ax.add_patch(Arc((0, spot), 18.3, 18.3, theta1=217, theta2=323, ec=LINE, lw=lw))
    if attack is not None:
        # Each side is drawn attacking its own end, so the direction has to be
        # stated or the two columns look like the same map twice.
        ax.text(-half - 1.4, PITCH_LENGTH / 2, "ATTACK ▲" if attack else "ATTACK ▼",
                color=NEUTRAL, fontsize=5.2, fontweight="bold", rotation=90,
                ha="center", va="center")


def _xy(x, y, *, flip=False):
    """Provider coordinates to pitch display coordinates, attacking upward."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    x = np.clip(x, 0.0, 100.0)
    y = np.clip(y, 0.0, 100.0)
    if flip:
        return (y - 50.0) * (PITCH_WIDTH / 100.0), (100.0 - x) * (PITCH_LENGTH / 100.0)
    return (50.0 - y) * (PITCH_WIDTH / 100.0), x * (PITCH_LENGTH / 100.0)


def _arrow_xy(frame, *, flip=False):
    sx, sy = _xy(frame["x"], frame["y"], flip=flip)
    ex, ey = _xy(frame["end_x"], frame["end_y"], flip=flip)
    return sx, sy, ex, ey


def _bool(series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _footnote(ax, text: str, dy: float = 0.0):
    """Caption under a panel. ``dy`` drops it clear of a panel that has ticks."""
    ax.text(0.5, -0.018 - dy, text.upper(), transform=ax.transAxes, color=NEUTRAL,
            fontsize=5.6, fontweight="bold", ha="center", va="top")


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------

def panel_pass_network(ax, events, players, team_id, colour, *, flip=False):
    """Average position and the heaviest links, over the whole match."""
    _pitch(ax, attack=not flip)
    team = events[events["team_id"].eq(team_id)]
    touches = team[touch_mask(team)].dropna(subset=["player", "x", "y"]).copy()
    if touches.empty:
        return

    position = touches.groupby("player").agg(
        x=("x", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        y=("y", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        touches=("player", "size"),
    )
    # Only the eleven busiest, or a poster cell fills with substitutes who
    # played ten minutes and sat on top of the player they replaced.
    position = position.sort_values("touches", ascending=False).head(11)

    work = events.sort_values(["minute", "second", "event_id"], kind="stable").copy()
    work["next_team"] = work["team_id"].shift(-1)
    work["next_player"] = work["player"].shift(-1)
    links = work[
        work["team_id"].eq(team_id)
        & work["type"].astype(str).eq("Pass")
        & work["outcome"].astype(str).str.lower().eq("successful")
        & work["next_team"].eq(team_id)
        & work["player"].notna()
        & work["next_player"].notna()
    ]
    links = links[links["player"].astype(str).ne(links["next_player"].astype(str))]
    names = set(position.index.astype(str))
    links = links[links["player"].astype(str).isin(names)
                  & links["next_player"].astype(str).isin(names)]
    edges = (links.groupby(["player", "next_player"]).size()
             .reset_index(name="n").sort_values("n", ascending=False).head(14))

    px, py = _xy(position["x"], position["y"], flip=flip)
    coords = {str(name): (px[i], py[i]) for i, name in enumerate(position.index)}

    top = float(edges["n"].max()) if not edges.empty else 1.0
    for row in edges.itertuples():
        a, b = coords.get(str(row.player)), coords.get(str(row.next_player))
        if not a or not b:
            continue
        ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, alpha=LINK_ALPHA,
                lw=0.5 + 3.4 * row.n / top, zorder=2, solid_capstyle="round")

    shirts = {}
    if players is not None and "shirt_no" in players.columns:
        squad = players[players["team_id"].eq(team_id)]
        shirts = {str(r.name_): str(int(r.shirt)) for r in squad.assign(
            name_=squad["name"].astype(str),
            shirt=pd.to_numeric(squad["shirt_no"], errors="coerce"),
        ).dropna(subset=["shirt"]).itertuples()}

    peak = float(position["touches"].max())
    for name, row in position.iterrows():
        x, y = coords[str(name)]
        ax.scatter(x, y, s=90 + 260 * row["touches"] / peak, color=colour,
                   edgecolor=BG, linewidth=0.9, zorder=4)
        ax.text(x, y, shirts.get(str(name), str(name)[:2].upper()),
                color=text_on_fill(colour), fontsize=5.4, fontweight="bold",
                ha="center", va="center", zorder=5)
    _footnote(ax, f"node = touches · {len(links)} completed links")


def panel_stat_table(ax, rows, home_colour, away_colour):
    """Sixteen indicators, each with the split that produced it."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(rows)
    step = 1.0 / n
    bar_l, bar_r = 0.225, 0.775
    mid = (bar_l + bar_r) / 2
    half = (bar_r - bar_l) / 2

    for i, (label, home_text, away_text, home_w, away_w) in enumerate(rows):
        y = 1.0 - (i + 0.5) * step
        total = float(home_w) + float(away_w)
        hs = float(home_w) / total if total else 0.5
        aws = float(away_w) / total if total else 0.5

        ax.text(0.5, y + step * 0.21, label.upper(), color=MUTED, fontsize=5.5,
                fontweight="bold", ha="center", va="center")
        bar_y = y - step * 0.16
        bar_h = step * 0.15
        ax.add_patch(Rectangle((bar_l, bar_y), bar_r - bar_l, bar_h,
                               facecolor=PANEL, edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((mid - hs * half, bar_y), hs * half, bar_h,
                               facecolor=home_colour, edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((mid, bar_y), aws * half, bar_h,
                               facecolor=away_colour, edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((mid - 0.0016, bar_y - step * 0.05), 0.0032,
                               bar_h + step * 0.10, facecolor=GRID, edgecolor="none",
                               zorder=4))

        ax.text(0.195, y, home_text, color=home_colour if home_w >= away_w else NEUTRAL,
                fontsize=7.6, fontweight="bold", ha="right", va="center")
        ax.text(0.805, y, away_text, color=away_colour if away_w >= home_w else NEUTRAL,
                fontsize=7.6, fontweight="bold", ha="left", va="center")
        if i:
            ax.plot([0.03, 0.97], [1.0 - i * step] * 2, color=GRID, lw=0.4, alpha=0.8)


def panel_shot_map(ax, events, home_id, away_id, home_colour, away_colour):
    """Both sides on one pitch, each attacking its own end."""
    _pitch(ax)
    pso = _bool(events.get("is_penalty_shootout"))
    if pso.empty:
        pso = pd.Series(False, index=events.index)
    shots = events[_bool(events["is_shot"]) & ~pso].dropna(subset=["x", "y"]).copy()
    shots["xG"] = pd.to_numeric(shots["xG"], errors="coerce").fillna(0).clip(lower=0)

    for team_id, colour, flip in ((home_id, home_colour, False), (away_id, away_colour, True)):
        side = shots[shots["team_id"].eq(team_id)]
        if side.empty:
            continue
        goals = _bool(side["is_goal"])
        for subset, marker, scale, edge in (
            (side[~goals], "o", 1.0, BG),
            (side[goals], "*", 2.6, INK),
        ):
            if subset.empty:
                continue
            sx, sy = _xy(subset["x"], subset["y"], flip=flip)
            ax.scatter(sx, sy, s=(22 + subset["xG"].to_numpy() * 420) * scale,
                       marker=marker, facecolor=colour, edgecolor=edge,
                       linewidth=0.7, alpha=0.92,
                       zorder=6 if marker == "*" else 4)
    _footnote(ax, "size = xg · star = goal")


def panel_momentum(ax, events, home_id, away_id, home_colour, away_colour):
    """Rolling expected-goal difference, five minutes at a time."""
    frame = xg_momentum(events, home_id, away_id, window=5)
    ax.set_facecolor(BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(GRID)
        ax.spines[spine].set_linewidth(0.6)
    if frame.empty:
        return
    minute = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    diff = pd.to_numeric(frame["differential"], errors="coerce").fillna(0).to_numpy(dtype=float)
    ax.fill_between(minute, 0, diff, where=diff >= 0, color=home_colour, alpha=0.85,
                    interpolate=True, linewidth=0)
    ax.fill_between(minute, 0, diff, where=diff < 0, color=away_colour, alpha=0.85,
                    interpolate=True, linewidth=0)
    ax.axhline(0, color=GRID, lw=0.7)

    goals = events[_bool(events["is_goal"])]
    span = float(np.abs(diff).max() or 0.2)
    for row in goals.itertuples():
        m = float(getattr(row, "minute", 0) or 0)
        ax.plot([m, m], [-span * 1.16, span * 1.16], color=NEUTRAL, lw=0.6,
                linestyle=(0, (2, 2)), zorder=1)
        ax.scatter([m], [span * 1.16], marker="*", s=42, color=INK,
                   edgecolor=BG, linewidth=0.5, zorder=5)
    ax.set_xlim(0, max(93.0, float(minute.max()) if len(minute) else 93.0))
    ax.set_ylim(-span * 1.32, span * 1.32)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.tick_params(axis="x", colors=MUTED, labelsize=5.8, length=2, width=0.5, pad=1)
    ax.tick_params(axis="y", length=0, labelleft=False)
    _footnote(ax, "above the line = home on top", dy=0.055)


def panel_xt_zones(ax, events, team_id, colour, *, flip=False):
    """Where possession added threat, as a smoothed surface."""
    _pitch(ax)
    team = events[events["team_id"].eq(team_id) & events["type"].astype(str).eq("Pass")]
    team = team.dropna(subset=["x", "y"]).copy()
    team["xT"] = pd.to_numeric(team["xT"], errors="coerce").fillna(0).clip(lower=0)
    team = team[team["xT"] > 0]
    if team.empty:
        return
    sx, sy = _xy(team["x"], team["y"], flip=flip)
    half = PITCH_WIDTH / 2
    grid, _, _ = np.histogram2d(
        sx, sy, bins=(26, 40), range=[[-half, half], [0, PITCH_LENGTH]],
        weights=team["xT"].to_numpy(dtype=float),
    )
    grid = gaussian_filter(grid, sigma=1.25)
    if grid.max() <= 0:
        return
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "xt", [(0, 0, 0, 0), matplotlib.colors.to_rgba(colour, 0.30),
               matplotlib.colors.to_rgba(colour, 1.0)]
    )
    ax.imshow(grid.T, origin="lower", extent=[-half, half, 0, PITCH_LENGTH],
              cmap=cmap, vmin=0.0, vmax=float(np.percentile(grid[grid > 0], 96)),
              aspect="equal", zorder=1, interpolation="bilinear")
    _pitch(ax, attack=not flip)
    _footnote(ax, "pass origin × threat added")


def panel_defensive_actions(ax, events, team_id, colour, *, flip=False):
    """Tackles, interceptions, recoveries and clearances, plus the line held."""
    _pitch(ax, attack=not flip)
    team = events[events["team_id"].eq(team_id)]
    kinds = team["type"].astype(str)
    actions = team[kinds.isin(["Tackle", "Interception", "BallRecovery", "Clearance"])]
    actions = actions.dropna(subset=["x", "y"])
    if actions.empty:
        return
    won = actions[actions["type"].astype(str).isin(["Tackle", "Interception", "BallRecovery"])]
    cleared = actions[actions["type"].astype(str).eq("Clearance")]
    if not won.empty:
        sx, sy = _xy(won["x"], won["y"], flip=flip)
        ax.scatter(sx, sy, s=22, marker="o", facecolor=colour, edgecolor=BG,
                   linewidth=0.5, alpha=0.85, zorder=4)
    if not cleared.empty:
        sx, sy = _xy(cleared["x"], cleared["y"], flip=flip)
        # "x" is an unfilled marker; giving it an edgecolor only warns.
        ax.scatter(sx, sy, s=16, marker="x", color=NEUTRAL, linewidth=0.7,
                   alpha=0.85, zorder=4)
    height = pd.to_numeric(actions["x"], errors="coerce").mean()
    hx, hy = _xy([50.0, 50.0], [0.0, 100.0], flip=flip)
    _, line_y = _xy([height], [50.0], flip=flip)
    ax.plot([hx[0], hx[1]], [line_y[0], line_y[0]], color=INK, lw=0.9,
            linestyle=(0, (4, 3)), alpha=0.75, zorder=5)
    ax.text(0, line_y[0] + 1.6, f"AVG {height:.0f}", color=INK, fontsize=5.6,
            fontweight="bold", ha="center", va="bottom", zorder=5)
    _footnote(ax, f"{len(won)} won · {len(cleared)} cleared · avg height {height:.0f}")


def panel_pitch_control(ax, events, home_id, away_id, home_colour, away_colour, shares):
    """Distance-decayed influence: who held which space."""
    surface, _ = pitch_control(events, home_id, away_id)
    _pitch(ax)
    half = PITCH_WIDTH / 2
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "control", [away_colour, PANEL, home_colour]
    )
    # The metric grid is [provider y, provider x]; the display wants
    # [length, width] with the lateral axis mirrored, the same turn _xy makes.
    grid = np.asarray(surface).T[:, ::-1]
    ax.imshow(grid, origin="lower", extent=[-half, half, 0, PITCH_LENGTH],
              cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal", zorder=1,
              interpolation="bilinear", alpha=0.92)
    _pitch(ax, attack=True)
    home_share, away_share, contested = shares
    _footnote(ax, f"{home_share:.0f}% · {contested:.0f}% contested · {away_share:.0f}%")


def panel_zone_dominance(ax, events, home_id, away_id, home_colour, away_colour):
    """Touch difference by zone -- who owned which third of which channel."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    touches = events[touch_mask(events)].dropna(subset=["x", "y"]).copy()
    touches["_x"] = pd.to_numeric(touches["x"], errors="coerce")
    touches["_y"] = pd.to_numeric(touches["y"], errors="coerce")
    rows, cols = 6, 5
    grid = np.zeros((rows, cols))
    for team_id, sign in ((home_id, 1), (away_id, -1)):
        side = touches[touches["team_id"].eq(team_id)]
        if side.empty:
            continue
        # The away side attacks the other way, so its coordinates are mirrored
        # into the home frame before the two are differenced.
        gx = side["_x"] if sign > 0 else 100.0 - side["_x"]
        gy = side["_y"] if sign > 0 else 100.0 - side["_y"]
        counts, _, _ = np.histogram2d(gx, gy, bins=(rows, cols),
                                      range=[[0, 100], [0, 100]])
        grid += sign * counts
    # Columns run in provider y; mirror them so the grid faces the same way as
    # every pitch on the poster.
    grid = grid[:, ::-1]
    peak = float(np.abs(grid).max() or 1.0)

    cell_w, cell_h = 1.0 / cols, 1.0 / rows
    for r in range(rows):
        for c in range(cols):
            value = grid[r, c]
            colour = home_colour if value >= 0 else away_colour
            alpha = 0.12 + 0.78 * abs(value) / peak
            ax.add_patch(Rectangle((c * cell_w, r * cell_h), cell_w, cell_h,
                                   facecolor=colour, alpha=alpha,
                                   edgecolor=BG, linewidth=1.1, zorder=2))
            # A faint cell is mostly page, a saturated one mostly team colour,
            # so the label is measured against the blend it actually sits on.
            blend = mcolors.to_hex(
                (1 - alpha) * np.asarray(mcolors.to_rgb(BG))
                + alpha * np.asarray(mcolors.to_rgb(colour))
            )
            ax.text((c + 0.5) * cell_w, (r + 0.5) * cell_h, f"{value:+.0f}",
                    color=text_on_fill(blend), fontsize=6.4, fontweight="bold",
                    ha="center", va="center", zorder=3)
    ax.text(0.5, -0.022, "OWN THIRD  →  ATTACKING THIRD, HOME DIRECTION",
            transform=ax.transAxes, color=NEUTRAL, fontsize=6.0,
            fontweight="bold", ha="center", va="top")


def panel_arrows(ax, frame, colour, *, flip=False, lw=0.7, alpha=0.7, head=1.7):
    if frame.empty:
        return
    sx, sy, ex, ey = _arrow_xy(frame, flip=flip)
    for i in range(len(sx)):
        if not np.isfinite([sx[i], sy[i], ex[i], ey[i]]).all():
            continue
        ax.annotate("", xy=(ex[i], ey[i]), xytext=(sx[i], sy[i]),
                    arrowprops=dict(arrowstyle=f"-|>,head_width={head * 0.11},"
                                              f"head_length={head * 0.18}",
                                    color=colour, lw=lw, alpha=alpha,
                                    shrinkA=0, shrinkB=0), zorder=4)


def panel_progressive(ax, events, team_id, colour, *, flip=False):
    _pitch(ax, attack=not flip)
    team = events[events["team_id"].eq(team_id)]
    frame = team[progressive_pass_mask(team)].dropna(subset=["x", "y", "end_x", "end_y"])
    panel_arrows(ax, frame, colour, flip=flip)
    _footnote(ax, f"{len(frame)} progressive passes")


def panel_box_entries(ax, events, team_id, colour, *, flip=False):
    _pitch(ax, attack=not flip)
    team = events[events["team_id"].eq(team_id)]
    frame = team[box_entry_mask(team)].dropna(subset=["x", "y", "end_x", "end_y"])
    panel_arrows(ax, frame, colour, flip=flip, lw=0.9, alpha=0.85, head=2.1)
    crossed = int(cross_mask(frame).sum()) if len(frame) else 0
    _footnote(ax, f"{len(frame)} entries · {crossed} crossed")


def panel_high_regains(ax, events, team_id, colour, *, flip=False):
    _pitch(ax, attack=not flip)
    frame = high_regain_events(events, team_id).dropna(subset=["x", "y"])
    if not frame.empty:
        sx, sy = _xy(frame["x"], frame["y"], flip=flip)
        ax.scatter(sx, sy, s=34, color=colour, edgecolor=INK, linewidth=0.6,
                   alpha=0.9, zorder=4)
    # The zone the regains are counted in, so the reader sees the rule not just
    # the dots that passed it.
    band_y = _xy([60.0], [50.0], flip=flip)[1][0]
    top_y = _xy([100.0], [50.0], flip=flip)[1][0]
    lo, hi = sorted((band_y, top_y))
    ax.add_patch(Rectangle((-PITCH_WIDTH / 2, lo), PITCH_WIDTH, hi - lo,
                           facecolor=colour, alpha=0.06, edgecolor="none", zorder=1))
    _footnote(ax, f"{len(frame)} regains in opposition territory")


def panel_player_leaders(ax, player_metrics, home_id, away_id, home_colour, away_colour,
                         home_name, away_name):
    """Who connected the valuable attacks, six a side."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if player_metrics is None or player_metrics.empty:
        return
    _footnote(ax, "bar = sequence xt · xgc = xg chain")
    frame = player_metrics.copy()
    frame["sequence_xT"] = pd.to_numeric(frame["sequence_xT"], errors="coerce").fillna(0)
    frame["xGChain"] = pd.to_numeric(frame["xGChain"], errors="coerce").fillna(0)
    peak = float(frame["sequence_xT"].max() or 1.0)

    for column, (team_id, colour, name) in enumerate(
        ((home_id, home_colour, home_name), (away_id, away_colour, away_name))
    ):
        side = (frame[frame["team_id"].eq(team_id)]
                .sort_values("sequence_xT", ascending=False).head(6))
        x0 = 0.02 + column * 0.51
        width = 0.45
        ax.text(x0, 0.965, name.upper()[:14], color=colour, fontsize=6.2,
                fontweight="bold", va="top")
        ax.text(x0 + width, 0.965, "xT     xGC", color=NEUTRAL, fontsize=5.2,
                fontweight="bold", va="top", ha="right")
        for i, row in enumerate(side.itertuples()):
            y = 0.855 - i * 0.142
            share = float(row.sequence_xT) / peak
            ax.add_patch(Rectangle((x0, y - 0.052), width * share, 0.030,
                                   facecolor=colour, alpha=0.55, edgecolor="none",
                                   zorder=2))
            label = str(row.player)
            if len(label) > 12:
                label = label.split()[-1]
            ax.text(x0, y, label[:12], color=TEXT, fontsize=6.4, va="center", zorder=3)
            ax.text(x0 + width - 0.085, y, f"{row.sequence_xT:.2f}", color=TEXT,
                    fontsize=6.4, fontweight="bold", ha="right", va="center", zorder=3)
            ax.text(x0 + width, y, f"{row.xGChain:.2f}", color=MUTED,
                    fontsize=6.4, ha="right", va="center", zorder=3)


# --------------------------------------------------------------------------
# panels — the transition game
# --------------------------------------------------------------------------

def panel_ball_losses(ax, events, team_id, colour, *, flip=False):
    """Where possession was surrendered, and which losses were punished."""
    _pitch(ax, attack=not flip)
    frame = turnover_events(events, team_id).dropna(subset=["x", "y"])
    if frame.empty:
        _footnote(ax, "no losses recorded")
        return
    punished = _bool(frame["punished"]) if "punished" in frame.columns else pd.Series(
        False, index=frame.index)
    for subset, size, alpha, edge in ((frame[~punished], 16, 0.55, "none"),
                                      (frame[punished], 46, 0.95, INK)):
        if subset.empty:
            continue
        sx, sy = _xy(subset["x"], subset["y"], flip=flip)
        ax.scatter(sx, sy, s=size, color=colour, alpha=alpha, edgecolor=edge,
                   linewidth=0.7, zorder=4)
    conceded = float(pd.to_numeric(frame.get("conceded_xG"), errors="coerce").fillna(0).sum())
    _footnote(ax, f"{len(frame)} losses · {int(punished.sum())} punished · {conceded:.2f} xg conceded")


def panel_game_state(ax, team_metrics, home_colour, away_colour):
    """What the scoreline did to each side.

    The most under-read board in a match report: a team that stops attacking
    once ahead and a team that only attacks once behind produce the same
    ninety-minute totals as two sides who played the game evenly.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    home = _side_metrics(team_metrics, "home")
    away = _side_metrics(team_metrics, "away")
    states = ("leading", "drawing", "trailing")
    peak = max(
        (_num(side, f"game_state_{state}_xG") for side in (home, away) for state in states),
        default=0.0,
    ) or 1.0

    band = 1.0 / len(states)
    for i, state in enumerate(states):
        top = 1.0 - i * band
        ax.text(0.5, top - band * 0.10, state.upper(), color=MUTED, fontsize=6.4,
                fontweight="bold", ha="center", va="center")
        for j, (side, colour, align) in enumerate(
            ((home, home_colour, -1), (away, away_colour, 1))
        ):
            xg = _num(side, f"game_state_{state}_xG")
            share = xg / peak
            centre = 0.5
            # The bar has to stop short of the value printed beside it: at
            # 0.40 the longest one ran straight under its own number.
            width = 0.30 * share
            x0 = centre - width if align < 0 else centre
            bar_y = top - band * 0.46
            ax.add_patch(Rectangle((x0, bar_y), width, band * 0.16,
                                   facecolor=colour, edgecolor="none", zorder=3))
            label_x = centre - 0.44 if align < 0 else centre + 0.44
            ax.text(label_x, bar_y + band * 0.08, f"{xg:.2f}", color=colour,
                    fontsize=7.6, fontweight="bold", va="center",
                    ha="left" if align < 0 else "right", zorder=4)
            detail = (f"{int(_num(side, f'game_state_{state}_shots'))} shots · "
                      f"{int(_num(side, f'game_state_{state}_box_entries'))} box")
            ax.text(label_x, bar_y - band * 0.14, detail, color=NEUTRAL, fontsize=5.4,
                    fontweight="bold", va="center",
                    ha="left" if align < 0 else "right", zorder=4)
        if i:
            ax.plot([0.03, 0.97], [top, top], color=GRID, lw=0.5)
    _footnote(ax, "expected goals produced in each scoreline state")


def panel_crosses(ax, events, team_id, colour, *, flip=False):
    """Delivery from wide, completed and not."""
    _pitch(ax, attack=not flip)
    team = events[events["team_id"].eq(team_id)]
    frame = team[cross_mask(team)].dropna(subset=["x", "y", "end_x", "end_y"])
    if frame.empty:
        _footnote(ax, "no crosses")
        return
    complete = frame["outcome"].astype(str).str.lower().eq("successful")
    panel_arrows(ax, frame[~complete], NEUTRAL, flip=flip, lw=0.55, alpha=0.5)
    panel_arrows(ax, frame[complete], colour, flip=flip, lw=1.0, alpha=0.9, head=2.1)
    _footnote(ax, f"{len(frame)} crosses · {int(complete.sum())} completed")


def panel_thirds(ax, team_metrics, home_colour, away_colour, home_name, away_name):
    """Touch distribution by third: where each side actually spent the match."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bands = (("touch_def_pct", "DEFENSIVE THIRD"), ("touch_mid_pct", "MIDDLE THIRD"),
             ("touch_att_pct", "ATTACKING THIRD"))
    for column, (side, colour, name) in enumerate((
        (_side_metrics(team_metrics, "home"), home_colour, home_name),
        (_side_metrics(team_metrics, "away"), away_colour, away_name),
    )):
        x0 = 0.06 + column * 0.50
        width = 0.38
        ax.text(x0 + width / 2, 0.965, name.upper()[:16], color=colour, fontsize=6.6,
                fontweight="bold", ha="center", va="top")
        bottom = 0.06
        # Stacked from the defensive third up, so the column reads like the
        # pitch it describes.
        for depth, (key, label) in enumerate(bands):
            share = _num(side, key) / 100.0
            height = share * 0.80
            alpha = 0.30 + 0.55 * (depth / (len(bands) - 1))
            ax.add_patch(Rectangle((x0, bottom), width, height, facecolor=colour,
                                   alpha=alpha, edgecolor=BG, linewidth=1.0, zorder=2))
            blend = mcolors.to_hex(
                (1 - alpha) * np.asarray(mcolors.to_rgb(BG))
                + alpha * np.asarray(mcolors.to_rgb(colour))
            )
            ink = text_on_fill(blend)
            ax.text(x0 + width / 2, bottom + height / 2 + 0.026,
                    f"{_num(side, key):.0f}%", color=ink, fontsize=7.4,
                    fontweight="bold", ha="center", va="center", zorder=3)
            ax.text(x0 + width / 2, bottom + height / 2 - 0.026, label,
                    color=ink, fontsize=5.0, fontweight="bold", alpha=0.75,
                    ha="center", va="center", zorder=3)
            bottom += height
    _footnote(ax, "share of each side's own touches")


def panel_zone14(ax, events, team_id, colour, *, flip=False):
    """Passes into zone 14 -- the pocket a settled block exists to protect."""
    _pitch(ax, attack=not flip)
    team = events[events["team_id"].eq(team_id)
                  & events["type"].astype(str).eq("Pass")
                  & events["outcome"].astype(str).str.lower().eq("successful")]
    team = team.dropna(subset=["x", "y", "end_x", "end_y"]).copy()
    end_x = pd.to_numeric(team["end_x"], errors="coerce")
    end_y = pd.to_numeric(team["end_y"], errors="coerce")
    # Zone 14: the central channel immediately outside the penalty area.
    into = team[end_x.between(70, 83) & end_y.between(33, 67)]
    half = PITCH_WIDTH / 2
    lo = _xy([70.0], [50.0], flip=flip)[1][0]
    hi = _xy([83.0], [50.0], flip=flip)[1][0]
    left = _xy([75.0], [33.0], flip=flip)[0][0]
    right = _xy([75.0], [67.0], flip=flip)[0][0]
    ax.add_patch(Rectangle((min(left, right), min(lo, hi)), abs(right - left),
                           abs(hi - lo), facecolor=colour, alpha=0.10,
                           edgecolor=colour, linewidth=0.6, zorder=1))
    panel_arrows(ax, into, colour, flip=flip, lw=0.8, alpha=0.8, head=1.9)
    del half
    _footnote(ax, f"{len(into)} completed passes into zone 14")


# --------------------------------------------------------------------------
# panels — the final ball
# --------------------------------------------------------------------------

def panel_shots(ax, events, team_id, colour, *, flip=False):
    """One side's shots, by outcome."""
    _pitch(ax, attack=not flip)
    pso = _bool(events.get("is_penalty_shootout"))
    if pso.empty:
        pso = pd.Series(False, index=events.index)
    shots = events[events["team_id"].eq(team_id) & _bool(events["is_shot"]) & ~pso]
    shots = shots.dropna(subset=["x", "y"]).copy()
    if shots.empty:
        _footnote(ax, "no shots")
        return
    shots["xG"] = pd.to_numeric(shots["xG"], errors="coerce").fillna(0).clip(lower=0)
    kinds = shots["type"].astype(str)
    layers = (
        (shots[kinds.eq("MissedShots")], "x", NEUTRAL, 1.0),
        (shots[kinds.eq("BlockedShot")], "s", MUTED, 1.0),
        (shots[kinds.eq("ShotOnPost")], "D", INK, 1.0),
        (shots[kinds.eq("SavedShot")], "o", colour, 1.0),
        (shots[_bool(shots["is_goal"])], "*", colour, 2.6),
    )
    for subset, marker, face, scale in layers:
        if subset.empty:
            continue
        sx, sy = _xy(subset["x"], subset["y"], flip=flip)
        sizes = (18 + subset["xG"].to_numpy() * 380) * scale
        if marker == "x":
            ax.scatter(sx, sy, s=sizes * 0.6, marker=marker, color=face,
                       linewidth=0.8, alpha=0.85, zorder=4)
        else:
            ax.scatter(sx, sy, s=sizes, marker=marker, facecolor=face,
                       edgecolor=INK if marker == "*" else BG, linewidth=0.7,
                       alpha=0.92, zorder=6 if marker == "*" else 4)
    total = float(shots["xG"].sum())
    _footnote(ax, f"{len(shots)} shots · {total:.2f} xg · x off · ◇ post · ★ goal")


def panel_goal_frame(ax, events, xg, home_id, away_id, home_colour, away_colour,
                     home_name, away_name):
    """Where the shots on target crossed the line, on one goal.

    A goal is three times wider than it is tall, and this panel is nearly
    square, so an equal-aspect frame can only ever fill one dimension. The band
    beneath the goal-line carries each side's on-target record rather than
    being left as the empty third of the cell it was.
    """
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.50, 1.30)
    ax.set_aspect("equal")
    ax.axis("off")
    for gx in np.linspace(-1, 1, 13):
        ax.plot([gx, gx], [0, 1], color=LINE, lw=0.35, alpha=0.5, zorder=1)
    for gy in np.linspace(0, 1, 7):
        ax.plot([-1, 1], [gy, gy], color=LINE, lw=0.35, alpha=0.5, zorder=1)
    for post in (-1, 1):
        ax.plot([post, post], [0, 1], color=LINE, lw=2.6, solid_capstyle="round", zorder=3)
    ax.plot([-1, 1], [1, 1], color=LINE, lw=2.6, solid_capstyle="round", zorder=3)
    ax.plot([-1.38, 1.38], [0, 0], color=LINE, lw=0.9, alpha=0.6, zorder=2)

    # The nine placement zones the provider records, counted per side and
    # printed in the cell they belong to. The two sides are offset vertically
    # by more than a marker radius so a shared zone reads as two counts.
    grid = {"high": 0.80, "mid": 0.50, "low": 0.20}
    lanes = {"left": -0.60, "centre": 0.0, "right": 0.60}
    for team_id, colour, dx in ((home_id, home_colour, -0.145),
                                (away_id, away_colour, 0.145)):
        zones = shot_placement_zones(events, team_id)
        for zone, count in zones.items():
            if not count:
                continue
            level, lane = zone.split("_", 1)
            x, y = lanes[lane] + dx, grid[level]
            ax.scatter([x], [y], s=96 + 28 * count, color=colour, alpha=0.9,
                       edgecolor=BG, linewidth=0.8, zorder=5)
            ax.text(x, y, str(count), color=text_on_fill(colour), fontsize=6.2,
                    fontweight="bold", ha="center", va="center", zorder=6)

    for i, (name, colour, row) in enumerate((
        (home_name, home_colour, _xg_row(xg, home_name)),
        (away_name, away_colour, _xg_row(xg, away_name)),
    )):
        y = -0.52 - i * 0.40
        ax.add_patch(Rectangle((-1.34, y - 0.062), 0.15, 0.124, facecolor=colour,
                               edgecolor="none", zorder=4))
        ax.text(-1.13, y, name.upper()[:18], color=TEXT, fontsize=6.4,
                fontweight="bold", va="center", zorder=4)
        ax.text(1.36, y, f"{int(_num(row, 'on_target'))} ON TARGET  ·  "
                         f"{_num(row, 'xGoT'):.2f} xGOT",
                color=MUTED, fontsize=6.0, fontweight="bold", ha="right",
                va="center", zorder=4)
    _footnote(ax, "shots on target by placement · size = count")


def panel_set_pieces(ax, events, team_id, colour, name, *, peak=None):
    """What the restarts produced.

    ``peak`` is shared across both sides so the two columns of this panel can
    be read against each other; scaled to its own maximum, the weaker side
    drew the same bar as the stronger one.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    breakdown = set_piece_breakdown(events, team_id)
    order = [("open_play", "Open play"), ("corner", "Corners"),
             ("free_kick", "Free kicks"), ("throw_in", "Throw-ins"),
             ("penalty", "Penalties")]
    peak = peak or max(
        (float(breakdown.get(key, {}).get("xG", 0.0)) for key, _ in order), default=0.0
    ) or 1.0
    ax.text(0.04, 0.965, name.upper()[:18], color=colour, fontsize=6.6,
            fontweight="bold", va="top")
    ax.text(0.96, 0.965, "SHOTS   xG", color=NEUTRAL, fontsize=5.2,
            fontweight="bold", va="top", ha="right")
    for i, (key, label) in enumerate(order):
        row = breakdown.get(key, {})
        shots = int(float(row.get("shots", 0) or 0))
        xg = float(row.get("xG", 0.0) or 0.0)
        goals = int(float(row.get("goals", 0) or 0))
        y = 0.83 - i * 0.175
        ax.add_patch(Rectangle((0.04, y - 0.062), 0.62 * (xg / peak), 0.036,
                               facecolor=colour, alpha=0.5, edgecolor="none", zorder=2))
        text = f"{label}  ·  {goals} goal{'s' if goals != 1 else ''}" if goals else label
        ax.text(0.04, y, text, color=TEXT if goals else MUTED, fontsize=6.6,
                fontweight="bold" if goals else "normal", va="center", zorder=3)
        ax.text(0.855, y, str(shots), color=TEXT, fontsize=6.6, fontweight="bold",
                ha="right", va="center", zorder=3)
        ax.text(0.96, y, f"{xg:.2f}", color=MUTED, fontsize=6.6, ha="right",
                va="center", zorder=3)
    _footnote(ax, "bar = expected goals from that restart")


def set_piece_peak(events, *team_ids) -> float:
    """Largest expected-goal total any restart type produced, either side."""
    values = [
        float(row.get("xG", 0.0) or 0.0)
        for team_id in team_ids
        for row in set_piece_breakdown(events, team_id).values()
    ]
    return max(values, default=0.0) or 1.0


def panel_funnel(ax, team_metrics, xg, home_id, away_id, home_name, away_name,
                 home_colour, away_colour):
    """Possessions narrowed down to goals, one stage at a time.

    Totals say what each side produced; the funnel says where each of them
    stopped producing it, which is the part a match report usually leaves out.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    home, away = _side_metrics(team_metrics, "home"), _side_metrics(team_metrics, "away")
    hx, ax_ = _xg_row(xg, home_name), _xg_row(xg, away_name)
    stages = [
        ("Possessions", _num(home, "possession_count"), _num(away, "possession_count")),
        ("Final third", _num(home, "final_third_entries"), _num(away, "final_third_entries")),
        ("Box entries", _num(home, "box_entries"), _num(away, "box_entries")),
        ("Shots", _num(hx, "shots"), _num(ax_, "shots")),
        ("On target", _num(hx, "on_target"), _num(ax_, "on_target")),
        ("Goals", _num(hx, "goals"), _num(ax_, "goals")),
    ]
    peak = max(max(h, a) for _label, h, a in stages) or 1.0
    band = 1.0 / len(stages)
    for i, (label, home_value, away_value) in enumerate(stages):
        top = 1.0 - i * band
        y = top - band * 0.52
        ax.text(0.5, top - band * 0.20, label.upper(), color=MUTED, fontsize=5.8,
                fontweight="bold", ha="center", va="center")
        for value, colour, align in ((home_value, home_colour, -1),
                                     (away_value, away_colour, 1)):
            width = 0.30 * (value / peak)
            x0 = 0.5 - width if align < 0 else 0.5
            ax.add_patch(Rectangle((x0, y - band * 0.10), width, band * 0.20,
                                   facecolor=colour, edgecolor="none", zorder=3))
            ax.text(0.5 - 0.44 if align < 0 else 0.5 + 0.44, y,
                    f"{value:.0f}", color=colour, fontsize=7.4, fontweight="bold",
                    va="center", ha="left" if align < 0 else "right", zorder=4)
        if i:
            ax.plot([0.05, 0.95], [top, top], color=GRID, lw=0.4)
    _footnote(ax, "every stage a possession has to survive to become a goal")


def panel_pass_length(ax, events, team_id, colour, name):
    """The passing profile: how far, how forward, how often it arrived."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    profile = pass_length_profile(events, team_id)
    ax.text(0.04, 0.965, name.upper()[:18], color=colour, fontsize=6.6,
            fontweight="bold", va="top")
    # Each row is a share of 100 except the average, which is metres; it is
    # drawn against a 40m ceiling so the bar still means something.
    rows = [
        ("Completion", profile.get("completion", 0.0), 100.0, "{:.1f}%"),
        ("Forward share", profile.get("forward_share", 0.0), 100.0, "{:.1f}%"),
        ("Long-ball share", profile.get("long_ball_share", 0.0), 100.0, "{:.1f}%"),
        ("Long-ball completion", profile.get("long_ball_completion", 0.0), 100.0, "{:.1f}%"),
        ("Average length", profile.get("avg_length_m", 0.0), 40.0, "{:.1f} m"),
    ]
    for i, (label, value, ceiling, fmt) in enumerate(rows):
        y = 0.82 - i * 0.175
        ax.add_patch(Rectangle((0.04, y - 0.062), 0.62 * min(value / ceiling, 1.0), 0.036,
                               facecolor=colour, alpha=0.5, edgecolor="none", zorder=2))
        ax.text(0.04, y, label, color=TEXT, fontsize=6.6, va="center", zorder=3)
        ax.text(0.96, y, fmt.format(value), color=TEXT, fontsize=6.6,
                fontweight="bold", ha="right", va="center", zorder=3)
    _footnote(ax, f"{int(profile.get('passes', 0))} passes attempted")


# --------------------------------------------------------------------------
# indicators
# --------------------------------------------------------------------------

def _side_metrics(team_metrics, side: str) -> pd.Series:
    row = team_metrics[team_metrics["side"].astype(str).eq(side)]
    return row.iloc[0] if not row.empty else pd.Series(dtype=float)


def _xg_row(xg, team_name: str) -> pd.Series:
    row = xg[xg["team"].astype(str).str.lower().eq(str(team_name).lower())]
    return row.iloc[0] if not row.empty else pd.Series(dtype=float)


def _num(series: pd.Series, key: str, default=0.0) -> float:
    try:
        value = float(series.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return default if pd.isna(value) else value


def build_indicator_rows(events, xg, team_metrics, home_id, away_id,
                         home_name, away_name, ppda, control_shares):
    """The sixteen rows of the poster's stat table.

    Every value is read from the pipeline's own frames rather than recomputed,
    so the poster and the report can never disagree about the same match.
    """
    home, away = _side_metrics(team_metrics, "home"), _side_metrics(team_metrics, "away")
    hx, ax_ = _xg_row(xg, home_name), _xg_row(xg, away_name)
    home_ppda, away_ppda = ppda
    home_control, away_control, _contested = control_shares

    def touches_share(row):
        return _num(row, "touches", 1.0)

    home_touch, away_touch = touches_share(home), touches_share(away)
    home_poss = 100.0 * home_touch / max(home_touch + away_touch, 1.0)

    rows = [
        ("Expected goals", f"{_num(hx, 'xG'):.2f}", f"{_num(ax_, 'xG'):.2f}",
         _num(hx, "xG"), _num(ax_, "xG")),
        ("xG on target", f"{_num(hx, 'xGoT'):.2f}", f"{_num(ax_, 'xGoT'):.2f}",
         _num(hx, "xGoT"), _num(ax_, "xGoT")),
        ("Shots (on target)",
         f"{int(_num(hx, 'shots'))} ({int(_num(hx, 'on_target'))})",
         f"{int(_num(ax_, 'shots'))} ({int(_num(ax_, 'on_target'))})",
         _num(hx, "shots"), _num(ax_, "shots")),
        ("Big chances", f"{int(_num(hx, 'big_chances'))}", f"{int(_num(ax_, 'big_chances'))}",
         _num(hx, "big_chances"), _num(ax_, "big_chances")),
        ("Possession", f"{home_poss:.1f}%", f"{100 - home_poss:.1f}%",
         home_poss, 100 - home_poss),
        ("Field tilt", f"{_num(home, 'field_tilt'):.1f}%", f"{_num(away, 'field_tilt'):.1f}%",
         _num(home, "field_tilt"), _num(away, "field_tilt")),
        ("Pitch control", f"{home_control:.0f}%", f"{away_control:.0f}%",
         home_control, away_control),
        ("Final third entries", f"{int(_num(home, 'final_third_entries'))}",
         f"{int(_num(away, 'final_third_entries'))}",
         _num(home, "final_third_entries"), _num(away, "final_third_entries")),
        ("Box entries", f"{int(_num(home, 'box_entries'))}", f"{int(_num(away, 'box_entries'))}",
         _num(home, "box_entries"), _num(away, "box_entries")),
        ("Box entry → shot", f"{_num(home, 'box_entry_to_shot_rate'):.0f}%",
         f"{_num(away, 'box_entry_to_shot_rate'):.0f}%",
         _num(home, "box_entry_to_shot_rate"), _num(away, "box_entry_to_shot_rate")),
        ("Progressive passes", f"{int(_num(home, 'progressive_passes'))}",
         f"{int(_num(away, 'progressive_passes'))}",
         _num(home, "progressive_passes"), _num(away, "progressive_passes")),
        ("Deep completions", f"{int(_num(home, 'deep_completions'))}",
         f"{int(_num(away, 'deep_completions'))}",
         _num(home, "deep_completions"), _num(away, "deep_completions")),
        ("Crosses (completed)",
         f"{int(_num(home, 'crosses'))} ({int(_num(home, 'completed_crosses'))})",
         f"{int(_num(away, 'crosses'))} ({int(_num(away, 'completed_crosses'))})",
         _num(home, "crosses"), _num(away, "crosses")),
        # A press is better when the number is lower, so the bar is inverted --
        # the wider half is still the side pressing harder.
        ("PPDA · lower is harder", f"{home_ppda:.2f}", f"{away_ppda:.2f}",
         away_ppda, home_ppda),
        ("High regains", f"{int(_num(home, 'high_regains'))}",
         f"{int(_num(away, 'high_regains'))}",
         _num(home, "high_regains"), _num(away, "high_regains")),
        ("Sequence threat (xT)", f"{_num(home, 'sequence_xT'):.2f}",
         f"{_num(away, 'sequence_xT'):.2f}",
         _num(home, "sequence_xT"), _num(away, "sequence_xT")),
    ]
    return rows


def build_transition_rows(team_metrics):
    """Twelve indicators about the game between the phases.

    None of these appear on posters 1 and 2. A side can lose every board there
    and still be the one who actually punished the other in broken play.
    """
    home, away = _side_metrics(team_metrics, "home"), _side_metrics(team_metrics, "away")
    return [
        ("Transitions", f"{int(_num(home, 'transitions'))}", f"{int(_num(away, 'transitions'))}",
         _num(home, "transitions"), _num(away, "transitions")),
        ("Transition shots", f"{int(_num(home, 'transition_shots'))}",
         f"{int(_num(away, 'transition_shots'))}",
         _num(home, "transition_shots"), _num(away, "transition_shots")),
        ("Transition xG", f"{_num(home, 'transition_xG'):.2f}", f"{_num(away, 'transition_xG'):.2f}",
         _num(home, "transition_xG"), _num(away, "transition_xG")),
        ("Transition → shot", f"{_num(home, 'transition_shot_rate'):.1f}%",
         f"{_num(away, 'transition_shot_rate'):.1f}%",
         _num(home, "transition_shot_rate"), _num(away, "transition_shot_rate")),
        ("Metres per transition", f"{_num(home, 'avg_transition_progress'):.1f}",
         f"{_num(away, 'avg_transition_progress'):.1f}",
         _num(home, "avg_transition_progress"), _num(away, "avg_transition_progress")),
        ("Possession regains", f"{int(_num(home, 'possession_regains'))}",
         f"{int(_num(away, 'possession_regains'))}",
         _num(home, "possession_regains"), _num(away, "possession_regains")),
        ("Regain → shot", f"{_num(home, 'regain_to_shot_rate'):.1f}%",
         f"{_num(away, 'regain_to_shot_rate'):.1f}%",
         _num(home, "regain_to_shot_rate"), _num(away, "regain_to_shot_rate")),
        ("Counterpress success", f"{_num(home, 'counterpress_success_rate'):.1f}%",
         f"{_num(away, 'counterpress_success_rate'):.1f}%",
         _num(home, "counterpress_success_rate"), _num(away, "counterpress_success_rate")),
        # Being exposed more often is worse, so the bar is inverted: the wider
        # half is the side that kept its shape.
        ("Rest-defence vulnerability", f"{_num(home, 'rest_defence_vulnerability'):.1f}%",
         f"{_num(away, 'rest_defence_vulnerability'):.1f}%",
         _num(away, "rest_defence_vulnerability"), _num(home, "rest_defence_vulnerability")),
        ("Dangerous counters", f"{int(_num(home, 'rest_defence_dangerous_counters'))}",
         f"{int(_num(away, 'rest_defence_dangerous_counters'))}",
         _num(away, "rest_defence_dangerous_counters"),
         _num(home, "rest_defence_dangerous_counters")),
        ("Build-up success", f"{_num(home, 'build_up_success_rate'):.1f}%",
         f"{_num(away, 'build_up_success_rate'):.1f}%",
         _num(home, "build_up_success_rate"), _num(away, "build_up_success_rate")),
        ("Directness", f"{_num(home, 'directness'):.1f}", f"{_num(away, 'directness'):.1f}",
         _num(home, "directness"), _num(away, "directness")),
    ]


def build_shooting_rows(xg, team_metrics, home_name, away_name):
    """Twelve indicators about the last action, and what it was worth."""
    hx, ax_ = _xg_row(xg, home_name), _xg_row(xg, away_name)
    home, away = _side_metrics(team_metrics, "home"), _side_metrics(team_metrics, "away")

    def per_shot(row, key):
        shots = _num(row, "shots")
        return _num(row, key) / shots if shots else 0.0

    def rate(row, key, base="shots"):
        total = _num(row, base)
        return 100.0 * _num(row, key) / total if total else 0.0

    home_over = _num(hx, "goals") - _num(hx, "xG")
    away_over = _num(ax_, "goals") - _num(ax_, "xG")
    return [
        ("Goals", f"{int(_num(hx, 'goals'))}", f"{int(_num(ax_, 'goals'))}",
         _num(hx, "goals"), _num(ax_, "goals")),
        ("Expected goals", f"{_num(hx, 'xG'):.2f}", f"{_num(ax_, 'xG'):.2f}",
         _num(hx, "xG"), _num(ax_, "xG")),
        ("Finishing vs expected", f"{home_over:+.2f}", f"{away_over:+.2f}",
         home_over - min(home_over, away_over, 0.0) + 0.01,
         away_over - min(home_over, away_over, 0.0) + 0.01),
        ("xG per shot", f"{per_shot(hx, 'xG'):.3f}", f"{per_shot(ax_, 'xG'):.3f}",
         per_shot(hx, "xG"), per_shot(ax_, "xG")),
        ("Shots on target", f"{int(_num(hx, 'on_target'))}", f"{int(_num(ax_, 'on_target'))}",
         _num(hx, "on_target"), _num(ax_, "on_target")),
        ("Shot accuracy", f"{rate(hx, 'on_target'):.1f}%", f"{rate(ax_, 'on_target'):.1f}%",
         rate(hx, "on_target"), rate(ax_, "on_target")),
        # Threat created and final-third efficiency stand where xG on target
        # and big chances were: both already have a row on poster 1, and a cell
        # that restates one is a cell the match did not get.
        ("Threat created (xT)", f"{_num(hx, 'xT'):.2f}", f"{_num(ax_, 'xT'):.2f}",
         _num(hx, "xT"), _num(ax_, "xT")),
        # How much of what was created ever reached the goalkeeper.
        ("xG reaching the frame",
         f"{100 * _num(hx, 'xGoT') / _num(hx, 'xG'):.0f}%" if _num(hx, "xG") else "—",
         f"{100 * _num(ax_, 'xGoT') / _num(ax_, 'xG'):.0f}%" if _num(ax_, "xG") else "—",
         _num(hx, "xGoT") / max(_num(hx, "xG"), 1e-9),
         _num(ax_, "xGoT") / max(_num(ax_, "xG"), 1e-9)),
        ("Final-third efficiency", f"{_num(home, 'final_third_entry_efficiency'):.1f}%",
         f"{_num(away, 'final_third_entry_efficiency'):.1f}%",
         _num(home, "final_third_entry_efficiency"),
         _num(away, "final_third_entry_efficiency")),
        ("Blocked", f"{int(_num(hx, 'blocked'))}", f"{int(_num(ax_, 'blocked'))}",
         _num(hx, "blocked"), _num(ax_, "blocked")),
        ("Off target", f"{int(_num(hx, 'off_target'))}", f"{int(_num(ax_, 'off_target'))}",
         # Missing more is worse, so the bar is inverted.
         _num(ax_, "off_target"), _num(hx, "off_target")),
        ("Sequence xT per possession",
         f"{_num(home, 'sequence_xT_per_possession'):.3f}",
         f"{_num(away, 'sequence_xT_per_possession'):.3f}",
         _num(home, "sequence_xT_per_possession"), _num(away, "sequence_xT_per_possession")),
    ]


# --------------------------------------------------------------------------
# chrome
# --------------------------------------------------------------------------

def _header(fig, *, home_id, away_id, home_name, away_name, home_colour, away_colour,
            score, competition, match_date, poster_label, subhead, allow_download):
    fig.text(0.030, 0.9885, poster_label.upper(), color=NEUTRAL, fontsize=7.0,
             fontweight="bold", va="top")
    # One line, above the crest band. On its own row the date landed at 0.9705,
    # inside the crest drawn at 0.9525, and was printed behind it.
    strap = competition.upper()
    if match_date:
        strap = f"{strap}  ·  {match_date}"
    fig.text(0.970, 0.9885, strap, color=NEUTRAL, fontsize=7.0,
             fontweight="bold", va="top", ha="right")

    crests.place_crest(fig, 0.068, 0.9525, home_id, monogram=home_name[:3].upper(),
                       colour=home_colour, width=86 / W_PX, background=BG,
                       allow_download=allow_download)
    crests.place_crest(fig, 0.932, 0.9525, away_id, monogram=away_name[:3].upper(),
                       colour=away_colour, width=86 / W_PX, background=BG,
                       allow_download=allow_download)

    fig.text(0.5, 0.9605, f"{home_name}  {score}  {away_name}", color=TEXT,
             fontsize=17.5, fontweight="bold", ha="center", va="center")
    # The line under the score carried the two club names again, which the
    # score already gives. It carries the result the match deserved instead.
    fig.text(0.5, 0.9345, subhead.upper(), color=NEUTRAL, fontsize=6.8,
             fontweight="bold", ha="center", va="center")

    rule = 0.9215
    fig.add_artist(Line2D([0.030, 0.5], [rule, rule], transform=fig.transFigure,
                          color=home_colour, lw=1.8))
    fig.add_artist(Line2D([0.5, 0.970], [rule, rule], transform=fig.transFigure,
                          color=away_colour, lw=1.8))


def _footer(fig, home_colour, away_colour, byline: str):
    fig.add_artist(Line2D([0.030, 0.970], [0.0495, 0.0495], transform=fig.transFigure,
                          color=GRID, lw=0.8))
    # The badge goes between the two footer lines rather than beside either:
    # the header is already carrying two crests and a score.
    crests.place_logo(fig, 0.5, 0.029, width=46 / W_PX, background=BG)
    fig.text(0.030, 0.030, byline.upper(), color=MUTED, fontsize=7.0, fontweight="bold",
             va="center")
    fig.text(0.970, 0.030, "DATA VIA OPTA / WHOSCORED", color=NEUTRAL, fontsize=7.0,
             fontweight="bold", va="center", ha="right")
    fig.add_artist(Rectangle((0.0, 0.0), 0.5, 0.0055, transform=fig.transFigure,
                             facecolor=home_colour, edgecolor="none"))
    fig.add_artist(Rectangle((0.5, 0.0), 0.5, 0.0055, transform=fig.transFigure,
                             facecolor=away_colour, edgecolor="none"))


def _new_figure():
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=BG)
    return fig


# --------------------------------------------------------------------------
# posters
# --------------------------------------------------------------------------

def build_match_posters(
    events,
    xg,
    team_metrics,
    player_metrics,
    players,
    *,
    out_dir: Path | str,
    home_id: int,
    away_id: int,
    home_name: str,
    away_name: str,
    home_color: str,
    away_color: str,
    score: str,
    competition: str = "MATCH ANALYSIS",
    match_date: str = "",
    byline: str = "MOSTAFA SAAD",
    allow_download: bool = True,
) -> list[Path]:
    """Render all four posters for one fixture and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    home_id, away_id = int(home_id), int(away_id)

    from match_report import compute_ppda_both

    ppda = compute_ppda_both(
        {"home_id": home_id, "away_id": away_id,
         "home_name": home_name, "away_name": away_name},
        events,
    )
    ppda_pair = (float(ppda["home"]["ppda"] or 0.0), float(ppda["away"]["ppda"] or 0.0))

    # pitch_control already returns percentages.
    _surface, shares = pitch_control(events, home_id, away_id)
    control = (float(shares.get("home", 0.0)), float(shares.get("away", 0.0)),
               float(shares.get("contested", 0.0)))

    home_xg = _num(_xg_row(xg, home_name), "xG")
    away_xg = _num(_xg_row(xg, away_name), "xG")

    header = dict(
        home_id=home_id, away_id=away_id, home_name=home_name, away_name=away_name,
        home_colour=home_color, away_colour=away_color, score=score,
        competition=competition, match_date=match_date,
        allow_download=allow_download,
        subhead=f"EXPECTED GOALS  {home_xg:.2f}  —  {away_xg:.2f}",
    )
    generated: list[Path] = []

    # ---- poster 1: the match -------------------------------------------
    fig = _new_figure()
    _header(fig, poster_label="POST-MATCH REPORT · 1 OF 4", **header)

    _panel_title(fig, "left", 0, f"{home_name} shape")
    panel_pass_network(_axes(fig, "left", 0), events, players, home_id, home_color)
    _panel_title(fig, "mid", 0, "The match in sixteen indicators")
    panel_stat_table(
        _axes(fig, "mid", 0),
        build_indicator_rows(events, xg, team_metrics, home_id, away_id,
                             home_name, away_name, ppda_pair, control),
        home_color, away_color,
    )
    _panel_title(fig, "right", 0, f"{away_name} shape")
    panel_pass_network(_axes(fig, "right", 0), events, players, away_id, away_color, flip=True)

    _panel_title(fig, "left", 1, f"{home_name} threat zones")
    panel_xt_zones(_axes(fig, "left", 1), events, home_id, home_color)
    _panel_title(fig, "mid", 1, "Shot map")
    panel_shot_map(_axes(fig, "mid", 1), events, home_id, away_id, home_color, away_color)
    _panel_title(fig, "right", 1, f"{away_name} threat zones")
    panel_xt_zones(_axes(fig, "right", 1), events, away_id, away_color, flip=True)

    _panel_title(fig, "left", 2, f"{home_name} defending")
    panel_defensive_actions(_axes(fig, "left", 2), events, home_id, home_color)
    _panel_title(fig, "mid", 2, "Game control")
    panel_momentum(_axes(fig, "mid", 2), events, home_id, away_id, home_color, away_color)
    _panel_title(fig, "right", 2, f"{away_name} defending")
    panel_defensive_actions(_axes(fig, "right", 2), events, away_id, away_color, flip=True)

    _footer(fig, home_color, away_color, byline)
    path = out / "match_poster_1_report.png"
    fig.savefig(path, facecolor=BG, dpi=DPI)
    plt.close(fig)
    generated.append(path)
    del fig
    gc.collect()

    # ---- poster 2: how it was played ------------------------------------
    fig = _new_figure()
    _header(fig, poster_label="HOW IT WAS PLAYED · 2 OF 4", **header)

    _panel_title(fig, "left", 0, f"{home_name} into the box")
    panel_box_entries(_axes(fig, "left", 0), events, home_id, home_color)
    _panel_title(fig, "mid", 0, "Pitch control")
    panel_pitch_control(_axes(fig, "mid", 0), events, home_id, away_id,
                        home_color, away_color, control)
    _panel_title(fig, "right", 0, f"{away_name} into the box")
    panel_box_entries(_axes(fig, "right", 0), events, away_id, away_color, flip=True)

    _panel_title(fig, "left", 1, f"{home_name} progression")
    panel_progressive(_axes(fig, "left", 1), events, home_id, home_color)
    _panel_title(fig, "mid", 1, "Zone dominance")
    panel_zone_dominance(_axes(fig, "mid", 1), events, home_id, away_id,
                         home_color, away_color)
    _panel_title(fig, "right", 1, f"{away_name} progression")
    panel_progressive(_axes(fig, "right", 1), events, away_id, away_color, flip=True)

    _panel_title(fig, "left", 2, f"{home_name} pressing")
    panel_high_regains(_axes(fig, "left", 2), events, home_id, home_color)
    _panel_title(fig, "mid", 2, "Who connected the danger")
    panel_player_leaders(_axes(fig, "mid", 2), player_metrics, home_id, away_id,
                         home_color, away_color, home_name, away_name)
    _panel_title(fig, "right", 2, f"{away_name} pressing")
    panel_high_regains(_axes(fig, "right", 2), events, away_id, away_color, flip=True)

    _footer(fig, home_color, away_color, byline)
    path = out / "match_poster_2_tactics.png"
    fig.savefig(path, facecolor=BG, dpi=DPI)
    plt.close(fig)
    generated.append(path)
    del fig
    gc.collect()

    # ---- poster 3: the transition game ----------------------------------
    fig = _new_figure()
    _header(fig, poster_label="THE TRANSITION GAME · 3 OF 4", **header)

    _panel_title(fig, "left", 0, f"{home_name} ball losses")
    panel_ball_losses(_axes(fig, "left", 0), events, home_id, home_color)
    _panel_title(fig, "mid", 0, "What the scoreline did")
    panel_game_state(_axes(fig, "mid", 0), team_metrics, home_color, away_color)
    _panel_title(fig, "right", 0, f"{away_name} ball losses")
    panel_ball_losses(_axes(fig, "right", 0), events, away_id, away_color, flip=True)

    _panel_title(fig, "left", 1, f"{home_name} from wide")
    panel_crosses(_axes(fig, "left", 1), events, home_id, home_color)
    _panel_title(fig, "mid", 1, "Transition and press")
    panel_stat_table(_axes(fig, "mid", 1), build_transition_rows(team_metrics),
                     home_color, away_color)
    _panel_title(fig, "right", 1, f"{away_name} from wide")
    panel_crosses(_axes(fig, "right", 1), events, away_id, away_color, flip=True)

    _panel_title(fig, "left", 2, f"{home_name} into zone 14")
    panel_zone14(_axes(fig, "left", 2), events, home_id, home_color)
    _panel_title(fig, "mid", 2, "Where the match was spent")
    panel_thirds(_axes(fig, "mid", 2), team_metrics, home_color, away_color,
                 home_name, away_name)
    _panel_title(fig, "right", 2, f"{away_name} into zone 14")
    panel_zone14(_axes(fig, "right", 2), events, away_id, away_color, flip=True)

    _footer(fig, home_color, away_color, byline)
    path = out / "match_poster_3_transitions.png"
    fig.savefig(path, facecolor=BG, dpi=DPI)
    plt.close(fig)
    generated.append(path)
    del fig
    gc.collect()

    # ---- poster 4: the final ball ---------------------------------------
    fig = _new_figure()
    _header(fig, poster_label="THE FINAL BALL · 4 OF 4", **header)

    _panel_title(fig, "left", 0, f"{home_name} shots")
    panel_shots(_axes(fig, "left", 0), events, home_id, home_color)
    _panel_title(fig, "mid", 0, "On the goal frame")
    panel_goal_frame(_axes(fig, "mid", 0), events, xg, home_id, away_id,
                     home_color, away_color, home_name, away_name)
    _panel_title(fig, "right", 0, f"{away_name} shots")
    panel_shots(_axes(fig, "right", 0), events, away_id, away_color, flip=True)

    restart_peak = set_piece_peak(events, home_id, away_id)
    _panel_title(fig, "left", 1, f"{home_name} restarts")
    panel_set_pieces(_axes(fig, "left", 1), events, home_id, home_color, home_name,
                     peak=restart_peak)
    _panel_title(fig, "mid", 1, "Possession to goal")
    panel_funnel(_axes(fig, "mid", 1), team_metrics, xg, home_id, away_id,
                 home_name, away_name, home_color, away_color)
    _panel_title(fig, "right", 1, f"{away_name} restarts")
    panel_set_pieces(_axes(fig, "right", 1), events, away_id, away_color, away_name,
                     peak=restart_peak)

    _panel_title(fig, "left", 2, f"{home_name} passing profile")
    panel_pass_length(_axes(fig, "left", 2), events, home_id, home_color, home_name)
    _panel_title(fig, "mid", 2, "The last action")
    panel_stat_table(_axes(fig, "mid", 2),
                     build_shooting_rows(xg, team_metrics, home_name, away_name),
                     home_color, away_color)
    _panel_title(fig, "right", 2, f"{away_name} passing profile")
    panel_pass_length(_axes(fig, "right", 2), events, away_id, away_color, away_name)

    _footer(fig, home_color, away_color, byline)
    path = out / "match_poster_4_final_ball.png"
    fig.savefig(path, facecolor=BG, dpi=DPI)
    plt.close(fig)
    generated.append(path)
    del fig
    gc.collect()

    return generated
