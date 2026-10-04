"""Every statistic and comparison for a block of rounds, as 16:9 timeline graphics.

Clubs and players each get their own folder under ``visuals``. A ranking for every
measure the tables hold (all twenty clubs; the top players), then the comparisons that
need two measures at once -- scatters, dumbbells, stacked bars, back-to-back bars, a heat
table of the league -- and a profile card for each club and player. All are 1600 x 900.
The title sits in a left-hand panel and the chart takes the rest, so the chart keeps the
full height. Text in the panel is stacked from its measured size, a scatter's marks are
spread apart where they overlap, and labels are placed where nothing else is.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from football_analysis.visuals import league_boards as lb
from football_analysis.visuals import visual_redesign_preview as base
from football_analysis.visuals.typography import display

SIZE, DPI = lb.SIZE, lb.DPI
GOOD, BAD, GOLD, GREY = "#3DDC84", "#FF6B5B", "#F2B134", "#8E99A4"
PARTS = ["#4EA8FF", "#F2B134", "#E8452C", "#A77BFF"]
MIN_MINUTES = 270.0
# Substitutes carry no position in the feed, so a player who only ever came on has role "Unknown".
# He belongs in a ranking that does not depend on position, and in none that does.
ALL_ROLES = ["Defender", "Midfielder", "Forward", "Unknown"]
LIST_RECT = (0.485, 0.07, 0.475, 0.88)  # a chart with a label for every row
PLOT_RECT = (0.40, 0.10, 0.575, 0.85)  # a scatter
TABLE_LEFT = 0.315  # where a full-width table begins
PANEL_RIGHT = 0.30  # the title panel may not run past this


@dataclass
class Card:
    scope: str  # "teams" or "players"
    group: str
    name: str  # file stem
    draw: Callable[[lb.League], Path]


# ── furniture ────────────────────────────────────────────────────────────────────────
def frame(league, group, title, subtitle, note="", key=None, reading=None, right=PANEL_RIGHT):
    """The page: a left panel (kicker, title, one line of explanation, key, how to read it).

    Each block is placed from the measured bottom of the one above, and the title is made
    smaller until it fits in the panel, so nothing in the panel can run into another block
    or into the chart.
    """
    fig = plt.figure(figsize=SIZE, facecolor=base.BG)
    renderer = fig.canvas.get_renderer()
    height, width = SIZE[1] * fig.dpi, SIZE[0] * fig.dpi  # the canvas still measures at its own dpi

    def put(text, y, **kw):
        artist = fig.text(0.035, y, text, va="top", **kw)
        return artist, artist.get_window_extent(renderer).y0 / height

    _, y = put(
        f"{league.title}  ·  {league.rounds}",
        0.95,
        color=lb.ACCENT,
        fontsize=8.5,
        fontweight="bold",
    )
    _, y = put(group.upper(), y - 0.01, color=base.MUTED, fontsize=8.5, fontweight="bold")
    y -= 0.035
    size = 46
    while True:
        lines = textwrap.wrap(title.upper(), 14 if size > 36 else 17)
        artist = fig.text(
            0.035, y, "\n".join(lines), va="top", color=base.TEXT, linespacing=0.92, **display(size)
        )
        extent = artist.get_window_extent(renderer)
        if extent.x1 <= right * width and (len(lines) <= 4 or size <= 28) or size <= 24:
            break
        artist.remove()
        size -= 3
    y = extent.y0 / height - 0.028
    fig.add_artist(
        Rectangle((0.035, y), 0.07, 0.006, transform=fig.transFigure, color=lb.ACCENT, lw=0)
    )
    y -= 0.028
    _, y = put(textwrap.fill(subtitle, 38), y, color=base.MUTED, fontsize=9.5, linespacing=1.45)
    y -= 0.03
    for symbol, label, colour in key or []:
        fig.text(0.037, y, symbol, color=colour, fontsize=10, va="top", fontweight="bold")
        fig.text(0.064, y, label, color=base.MUTED, fontsize=8.5, va="top", fontweight="bold")
        y -= 0.05
    if reading:
        y -= 0.01
        for symbol, text in reading:
            fig.text(0.037, y, symbol, color=lb.ACCENT, fontsize=9, va="top", fontweight="bold")
            _, y = put_at(
                fig,
                renderer,
                height,
                0.062,
                y,
                textwrap.fill(text, 30),
                color=base.MUTED,
                fontsize=7.8,
                linespacing=1.3,
            )
            y -= 0.018
    if note:
        fig.text(
            0.035,
            0.085,
            textwrap.fill(note, 50),
            color=base.NEUTRAL,
            fontsize=7,
            va="bottom",
            linespacing=1.4,
        )
    fig.text(0.035, 0.03, "MOSTAFA SAAD", color=base.MUTED, va="center", **display(14))
    return fig


def put_at(fig, renderer, height, x, y, text, **kw):
    artist = fig.text(x, y, text, va="top", **kw)
    return artist, artist.get_window_extent(renderer).y0 / height


def save(fig, league, scope, filename) -> Path:
    path = league.folder / "visuals" / scope / f"{filename}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=base.BG)
    plt.close(fig)
    return path


def clean(ax, grid="x"):
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=base.MUTED, labelsize=9, length=0)
    if grid:
        ax.grid(axis=grid, color=base.GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def list_axes(fig, shrink=1.0):
    x, y, w, h = LIST_RECT
    return fig.add_axes([x, y, w * shrink, h])


def row_labels(ax, league, labels, teams, size=9.5, zoom=0.085):
    """A name and a badge for each row; the badge sits just left of the axes."""
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=size, color=base.TEXT)
    ax.tick_params(axis="y", pad=24)
    for i, team in enumerate(teams):
        lb.put_crest(ax, league, team, (-0.045, i), zoom=zoom, transform=ax.get_yaxis_transform())


def value_labels(ax, ys, values, fmt, pad, size=9.5):
    """Figures beside the bars, on a patch of page colour so a gridline never runs through one."""
    for y, v in zip(ys, values):
        ax.text(
            v + pad,
            y,
            fmt.format(v),
            va="center",
            color=base.TEXT,
            fontsize=size,
            fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.12", facecolor=base.BG, edgecolor="none", alpha=0.9),
            zorder=6,
        )


# ── chart kinds ─────────────────────────────────────────────────────────────────────────
def ranked_teams(
    league, group, name, title, subtitle, series, fmt, note, lower_better=False, highlight=3
):
    """All twenty clubs as bars, best first, the top few in their own colour; a line marks the average."""
    order = series.sort_values(ascending=lower_better)
    teams, values = list(order.index)[::-1], order.to_numpy()[::-1]
    fig = frame(league, group, title, subtitle, note)
    ax = list_axes(fig)
    clean(ax)
    n = len(teams)
    colours = [league.colour(t) if i >= n - highlight else "#3A434C" for i, t in enumerate(teams)]
    ax.barh(range(n), values, color=colours, height=0.68)
    row_labels(ax, league, teams, teams)
    top = float(np.nanmax(values))
    ax.set_xlim(min(0, float(np.nanmin(values)) * 1.2), top * 1.16 if top > 0 else 1)
    value_labels(ax, range(n), values, fmt, top * 0.012 if top > 0 else 0.01)
    mean = float(series.mean())
    ax.axvline(mean, color=GREY, lw=1.1, ls=(0, (4, 4)), zorder=1)
    ax.text(
        mean,
        n - 0.15,
        "LEAGUE AVERAGE",
        color=GREY,
        fontsize=7.2,
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    ax.set_ylim(-0.7, n + 0.35)
    ax.tick_params(axis="x", labelbottom=False)
    return save(fig, league, "teams", name)


def ranked_players(league, group, name, title, subtitle, rows, key, fmt, note, n=15):
    """The leaders as a stem and a dot in their club's colour, with the club under the name."""
    rows = rows.sort_values(key, ascending=False).head(n).iloc[::-1]
    fig = frame(league, group, title, subtitle, note)
    ax = list_axes(fig, shrink=0.94)
    clean(ax)
    ys = np.arange(len(rows))
    top = float(rows[key].max())
    for y, r in zip(ys, rows.itertuples()):
        value = getattr(r, key)
        ax.plot([0, value], [y, y], color=base.GRID, lw=3.0, zorder=1)
        ax.scatter(
            [value],
            [y],
            s=210,
            color=league.colour(r.team),
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            value + top * 0.04, y, fmt.format(value), va="center", color=base.TEXT, **display(16)
        )
    labels = [f"{lb.surname(r.player)}" for r in rows.itertuples()]
    row_labels(ax, league, labels, [r.team for r in rows.itertuples()], size=9.5, zoom=0.075)
    for y, r in zip(ys, rows.itertuples()):
        ax.text(
            -0.066,
            y - 0.34,
            r.team,
            color=base.NEUTRAL,
            fontsize=6.2,
            va="center",
            ha="right",
            transform=ax.get_yaxis_transform(),
        )
    ax.set_xlim(0, top * 1.2 if top > 0 else 1)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.tick_params(axis="x", labelbottom=False)
    return save(fig, league, "players", name)


def dumbbell(
    league,
    scope,
    group,
    name,
    title,
    subtitle,
    frame_,
    a_key,
    b_key,
    a_label,
    b_label,
    note,
    row_label,
    row_team,
    b_higher_is_good=True,
):
    """Two values per row joined by a line: a dot for what was expected, a ring for what happened."""
    rows = frame_.copy()
    rows["_gap"] = rows[b_key] - rows[a_key]
    rows = rows.sort_values("_gap", ascending=b_higher_is_good)
    fig = frame(
        league, group, title, subtitle, note, key=[("●", a_label, GREY), ("○", b_label, base.TEXT)]
    )
    ax = list_axes(fig)
    clean(ax)
    ys = np.arange(len(rows))
    top = float(rows[[a_key, b_key]].to_numpy().max())
    for y, r in zip(ys, rows.itertuples()):
        a, b = getattr(r, a_key), getattr(r, b_key)
        colour = GOOD if ((b >= a) if b_higher_is_good else (b <= a)) else BAD
        ax.plot([a, b], [y, y], color=colour, lw=3.2, solid_capstyle="round", zorder=2)
        ax.scatter([a], [y], s=60, color=GREY, zorder=3)
        ax.scatter([b], [y], s=130, facecolor=colour, edgecolor="white", linewidth=1.3, zorder=4)
        ax.text(
            max(a, b) + top * 0.04,
            y,
            f"{b - a:+.1f}",
            va="center",
            color=colour,
            fontsize=9.5,
            fontweight="bold",
        )
    row_labels(
        ax,
        league,
        [row_label(r) for r in rows.itertuples()],
        [row_team(r) for r in rows.itertuples()],
    )
    ax.set_ylim(-0.7, len(rows) - 0.3)
    low = float(rows[[a_key, b_key]].to_numpy().min())
    ax.set_xlim(min(0, low) - 0.05 * top, top * 1.2)
    return save(fig, league, scope, name)


def stacked(
    league,
    scope,
    group,
    name,
    title,
    subtitle,
    rows,
    keys,
    labels,
    note,
    row_name,
    row_team,
    fmt="{:.1f}",
):
    """What each row is made of: one bar split into its parts, longest first."""
    order = rows.assign(_t=rows[keys].sum(axis=1)).sort_values("_t", ascending=True)
    fig = frame(
        league,
        group,
        title,
        subtitle,
        note,
        key=[("■", label, c) for label, c in zip(labels, PARTS)],
    )
    ax = list_axes(fig, shrink=0.97)
    clean(ax)
    ys = np.arange(len(order))
    left = np.zeros(len(order))
    for key, colour in zip(keys, PARTS):
        values = order[key].to_numpy()
        ax.barh(ys, values, left=left, color=colour, height=0.68, edgecolor=base.BG, linewidth=0.8)
        left += values
    row_labels(
        ax,
        league,
        [row_name(r) for r in order.itertuples()],
        [row_team(r) for r in order.itertuples()],
    )
    value_labels(ax, ys, order["_t"].to_numpy(), fmt, float(order["_t"].max()) * 0.012)
    ax.set_xlim(0, float(order["_t"].max()) * 1.14)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.tick_params(axis="x", labelbottom=False)
    return save(fig, league, scope, name)


def quadrant(
    league,
    scope,
    group,
    name,
    title,
    subtitle,
    frame_,
    x_key,
    y_key,
    x_label,
    y_label,
    note,
    corners,
    as_teams=True,
    label_n=14,
):
    """Two measures at once. Clubs carry a badge (spread apart where they overlap); players a dot and a surname.

    What each corner means is written in the title panel, not on the chart, so no label can
    sit on a mark.
    """
    arrows = ["↗", "↖", "↘", "↙"]
    reading = [(a, f"{c.capitalize()}") for a, c in zip(arrows, corners) if c]
    fig = frame(league, group, title, subtitle, note, reading=reading)
    ax = fig.add_axes(list(PLOT_RECT))
    ax.set_facecolor(base.PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=base.MUTED, labelsize=8.5, length=0)
    ax.grid(color=base.GRID, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    x, y = frame_[x_key].to_numpy(float), frame_[y_key].to_numpy(float)
    pad_x, pad_y = (x.max() - x.min()) * 0.09, (y.max() - y.min()) * 0.09
    ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax.set_ylim(y.min() - pad_y, y.max() + pad_y)
    ax.axvline(np.median(x), color=base.GRID, lw=1.1, ls=(0, (4, 4)))
    ax.axhline(np.median(y), color=base.GRID, lw=1.1, ls=(0, (4, 4)))
    ax.set_xlabel(x_label, color=base.MUTED, fontsize=9, fontweight="bold")
    ax.set_ylabel(y_label, color=base.MUTED, fontsize=9, fontweight="bold")
    if as_teams:
        radius = 21.0
        moved = lb.spread_apart(ax, x, y, radius)
        for team, (xv, yv), (mx, my) in zip(frame_.index, zip(x, y), moved):
            colour = league.colour(team)
            if abs(mx - xv) + abs(my - yv) > 0:
                span_x, span_y = np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim())
                if abs(mx - xv) / span_x + abs(my - yv) / span_y > 0.004:
                    ax.plot([xv, mx], [yv, my], color="#6B7580", lw=0.7, zorder=2)
                    ax.scatter([xv], [yv], s=12, color="#9AA4AB", zorder=3)
            ax.scatter(mx, my, s=430, facecolor=base.BG, edgecolor=colour, linewidth=1.8, zorder=4)
            if not lb.put_crest(ax, league, team, (mx, my), zoom=0.115):
                ax.text(
                    mx,
                    my,
                    team[:3].upper(),
                    color=colour,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    zorder=6,
                )
        lb.place_labels(
            ax, list(moved), list(frame_.index), radius_px=radius, fontsize=7.5, badges=True
        )
    else:
        spread = max(frame_["minutes"].max() - frame_["minutes"].min(), 1)
        sizes = 36 + 120 * (frame_["minutes"] - frame_["minutes"].min()) / spread
        ax.scatter(
            x,
            y,
            s=sizes,
            c=[league.colour(t) for t in frame_["team"]],
            alpha=0.9,
            edgecolor=base.BG,
            linewidth=0.8,
            zorder=3,
        )
        zx, zy = (x - x.mean()) / (x.std() or 1), (y - y.mean()) / (y.std() or 1)
        named = np.argsort(-np.hypot(zx, zy))[:label_n]
        lb.place_labels(
            ax,
            [(x[i], y[i]) for i in named],
            [lb.surname(frame_["player"].iloc[i]) for i in named],
            radius_px=6.0,
            fontsize=8,
            avoid=list(zip(x, y)),
        )
    return save(fig, league, scope, name)


def butterfly(
    league,
    group,
    name,
    title,
    subtitle,
    left,
    right,
    left_label,
    right_label,
    note,
    fmt="{:.0f}",
    right_lower_better=False,
):
    """Two measures back to back, clubs ordered by the two ranks together."""
    both = pd.DataFrame({"l": left, "r": right}).dropna()
    both["_rank"] = both["l"].rank(ascending=False) + both["r"].rank(ascending=right_lower_better)
    order = both.sort_values("_rank", ascending=False)
    fig = frame(
        league,
        group,
        title,
        subtitle,
        note,
        key=[("◀", left_label, "#4EA8FF"), ("▶", right_label, GOLD)],
    )
    ax = list_axes(fig)
    clean(ax, grid=None)
    ys = np.arange(len(order))
    lmax, rmax = float(order["l"].max()), float(order["r"].max())
    ax.barh(ys, -order["l"] / lmax, color="#4EA8FF", height=0.68)
    ax.barh(ys, order["r"] / rmax, color=GOLD, height=0.68)
    for y, (_, row) in zip(ys, order.iterrows()):
        ax.text(
            -row["l"] / lmax - 0.03,
            y,
            fmt.format(row["l"]),
            ha="right",
            va="center",
            color=base.TEXT,
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            row["r"] / rmax + 0.03,
            y,
            fmt.format(row["r"]),
            ha="left",
            va="center",
            color=base.TEXT,
            fontsize=9,
            fontweight="bold",
        )
    ax.axvline(0, color=base.TEXT, lw=1.0, alpha=0.6)
    row_labels(ax, league, list(order.index), list(order.index))
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xticks([])
    return save(fig, league, "teams", name)


def table_axes(fig):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_zorder(-1)
    return ax


def percentile_ramp():
    return mcolors.LinearSegmentedColormap.from_list("heat", [base.PANEL, "#0F4A44", lb.ACCENT])


# ── data ───────────────────────────────────────────────────────────────────────────────────
def per_match(league):
    return league.per_match.set_index("team")


def totals(league):
    return league.teams.set_index("team")


def player_frame(league):
    """Per-90 rates with the matching totals alongside (``*_tot``), outfield players over the minutes floor."""
    keep = [
        "goals",
        "assists",
        "shots",
        "shots_on_target",
        "takeons",
        "tackles",
        "aerials",
        "aerials_won",
        "passes",
        "crosses",
        "long_balls",
        "yellow_cards",
        "fouls_committed",
        "goal_contributions",
        "goals_minus_xG",
        "xG",
    ]
    t = league.players[["player", "team"] + [c for c in keep if c in league.players]]
    t = t.rename(columns={c: f"{c}_tot" for c in t.columns if c not in ("player", "team")})
    merged = league.per90.merge(t, on=["player", "team"], how="left")
    return merged[(merged["minutes"] >= MIN_MINUTES) & ~lb.is_goalkeeper(merged)].copy()


def squad_totals(league):
    p = league.players
    return p[(p["minutes"] >= MIN_MINUTES) & ~lb.is_goalkeeper(p)].copy()


# ── club rankings: one for every measure the table holds ──────────────────────────────────────
# (key, group, title, subtitle, format, lower_is_better)
TEAM_RANKINGS = [
    ("goals_for", "Attack", "Goals scored", "Goals per match", "{:.2f}", False),
    ("xG", "Attack", "Expected goals", "xG created per match", "{:.2f}", False),
    (
        "xGoT",
        "Attack",
        "Expected goals on target",
        "Post-shot xG per match: chances weighted by where the shot went",
        "{:.2f}",
        False,
    ),
    ("shots", "Attack", "Shots", "Shots per match", "{:.1f}", False),
    ("on_target", "Attack", "Shots on target", "Shots on target per match", "{:.1f}", False),
    (
        "shot_accuracy",
        "Attack",
        "Shot accuracy",
        "Share of shots that hit the target",
        "{:.0f}%",
        False,
    ),
    (
        "goal_conversion",
        "Attack",
        "Shot conversion",
        "Share of shots that scored",
        "{:.0f}%",
        False,
    ),
    ("xG_per_shot", "Attack", "Shot quality", "xG per shot", "{:.3f}", False),
    (
        "big_chances",
        "Attack",
        "Big chances created",
        "Clear-cut chances per match",
        "{:.1f}",
        False,
    ),
    (
        "box_entries",
        "Attack",
        "Entries into the box",
        "Passes and carries into the penalty area per match",
        "{:.1f}",
        False,
    ),
    ("final_third_entries", "Attack", "Entries into the final third", "Per match", "{:.0f}", False),
    (
        "deep_completions",
        "Attack",
        "Deep completions",
        "Completed passes into the zone near goal, per match",
        "{:.1f}",
        False,
    ),
    ("xT", "Attack", "Threat created", "xT added per match", "{:.2f}", False),
    ("crosses", "Attack", "Crosses", "Crosses per match", "{:.1f}", False),
    ("corners", "Attack", "Corners won", "Corners per match", "{:.1f}", False),
    (
        "transition_xG",
        "Attack",
        "Counter-attack chances",
        "xG from fast breaks per match",
        "{:.2f}",
        False,
    ),
    (
        "goals_against",
        "Defence",
        "Fewest goals conceded",
        "Goals conceded per match",
        "{:.2f}",
        True,
    ),
    ("shots_against", "Defence", "Fewest shots faced", "Shots faced per match", "{:.1f}", True),
    ("on_target_against", "Defence", "Fewest shots on target faced", "Per match", "{:.1f}", True),
    (
        "xGoT_against",
        "Defence",
        "Post-shot xG faced",
        "Per match · the quality of the shots on target a goalkeeper had to save",
        "{:.2f}",
        True,
    ),
    (
        "big_chances_against",
        "Defence",
        "Fewest big chances conceded",
        "Clear-cut chances against per match",
        "{:.1f}",
        True,
    ),
    ("box_entries_against", "Defence", "Fewest box entries conceded", "Per match", "{:.1f}", True),
    (
        "goals_prevented",
        "Defence",
        "Goalkeeping above expectation",
        "xGOT faced minus goals conceded, per match",
        "{:+.2f}",
        False,
    ),
    ("tackles", "Defence", "Tackles", "Tackles per match", "{:.1f}", False),
    ("interceptions", "Defence", "Interceptions", "Interceptions per match", "{:.1f}", False),
    ("blocks", "Defence", "Blocks", "Shots blocked per match", "{:.1f}", False),
    ("clearances", "Defence", "Clearances", "Clearances per match", "{:.1f}", False),
    ("fouls_committed", "Defence", "Fouls committed", "Per match", "{:.1f}", False),
    ("yellow_cards", "Defence", "Yellow cards", "Per match", "{:.2f}", False),
    ("aerial_pct", "Defence", "Aerial duels won", "Share of aerial duels won", "{:.0f}%", False),
    (
        "ground_duel_pct",
        "Defence",
        "Ground duels won",
        "Share of ground duels won",
        "{:.0f}%",
        False,
    ),
    (
        "high_regains",
        "Pressing",
        "High regains",
        "Possessions won in the final third, per match",
        "{:.1f}",
        False,
    ),
    (
        "possession_regains",
        "Pressing",
        "Possessions won back",
        "Regains per match anywhere on the pitch",
        "{:.1f}",
        False,
    ),
    (
        "regain_to_shot_rate",
        "Pressing",
        "Regains that become shots",
        "Share of won possessions ending in a shot",
        "{:.1f}%",
        False,
    ),
    ("regain_xG", "Pressing", "xG from regains", "Per match", "{:.2f}", False),
    (
        "counterpress_regains",
        "Pressing",
        "Counter-press regains",
        "Possessions won back within seconds, per match",
        "{:.1f}",
        False,
    ),
    ("passes", "Passing", "Most passes", "Passes attempted per match", "{:.0f}", False),
    ("pass_pct", "Passing", "Pass completion", "Share of passes completed", "{:.1f}%", False),
    ("progressive_passes", "Passing", "Progressive passes", "Per match", "{:.0f}", False),
    (
        "line_breaking_completed",
        "Passing",
        "Line-breaking passes",
        "Completed per match",
        "{:.0f}",
        False,
    ),
    (
        "key_passes",
        "Passing",
        "Key passes",
        "Passes that led to a shot, per match",
        "{:.1f}",
        False,
    ),
    (
        "passes_into_final_third",
        "Passing",
        "Passes into the final third",
        "Completed, per match",
        "{:.0f}",
        False,
    ),
    (
        "long_balls",
        "Passing",
        "Long balls",
        "Passes of 32 metres or more, per match",
        "{:.0f}",
        False,
    ),
    (
        "long_ball_pct",
        "Passing",
        "Long-ball accuracy",
        "Share of long balls that arrived",
        "{:.0f}%",
        False,
    ),
    ("cross_pct", "Passing", "Crossing accuracy", "Share of crosses completed", "{:.0f}%", False),
    (
        "directness",
        "Passing",
        "Directness",
        "Share of progress made towards goal",
        "{:.0f}%",
        False,
    ),
    ("field_tilt", "Passing", "Field tilt", "Share of final-third touches", "{:.0f}%", False),
    ("possession_share", "Passing", "Possession", "Share of possessions", "{:.0f}%", False),
    ("touches", "Passing", "Touches", "Per match", "{:.0f}", False),
    (
        "build_up_success_rate",
        "Passing",
        "Build-up success",
        "Share of build-up attempts that worked",
        "{:.0f}%",
        False,
    ),
    (
        "final_third_entry_efficiency",
        "Passing",
        "Final-third efficiency",
        "How often a final-third entry ends in a shot",
        "{:.1f}%",
        False,
    ),
]
# rankings already told by a curated card below, so they are not repeated
TEAM_SKIP = {"xG_against", "ppda", "counterpress_success_rate", "passes_into_box"}


def team_ranking_card(spec) -> Card | None:
    key, group, title, subtitle, fmt, lower = spec
    slug = key.lower()
    index = TEAM_RANKINGS.index(spec) + 1

    def draw(league):
        series = per_match(league)[key].dropna()
        if len(series) < 10:
            raise ValueError(f"{key} is missing for too many clubs")
        return ranked_teams(
            league,
            group,
            f"{group.lower()}_{index:02d}_{slug}",
            title,
            subtitle,
            series,
            fmt,
            "Top three in club colour. Dashed line is the league average."
            + (" Lower is better." if lower else ""),
            lower_better=lower,
        )

    return Card("teams", group, f"{group.lower()}_{index:02d}_{slug}", draw)


# ── player rankings ──────────────────────────────────────────────────────────────────────────
# (key, group, title, subtitle, format, roles, where)
PLAYER_RANKINGS = [
    ("xG_p90", "Attack", "Expected goals", "xG per 90", "{:.2f}", ALL_ROLES, None),
    ("xA_p90", "Attack", "Expected assists", "xA per 90", "{:.2f}", ALL_ROLES, None),
    ("xG_xA_p90", "Attack", "Goal threat", "xG plus xA per 90", "{:.2f}", ALL_ROLES, None),
    (
        "goal_contributions_p90",
        "Attack",
        "Goal contributions",
        "Goals plus assists per 90",
        "{:.2f}",
        ALL_ROLES,
        None,
    ),
    ("shots_p90", "Attack", "Shot volume", "Shots per 90", "{:.1f}", ALL_ROLES, None),
    (
        "shots_on_target_p90",
        "Attack",
        "Shots on target",
        "Shots on target per 90",
        "{:.1f}",
        ALL_ROLES,
        None,
    ),
    (
        "shot_accuracy",
        "Attack",
        "Shot accuracy",
        "Share of shots on target · 8+ shots",
        "{:.0f}%",
        ALL_ROLES,
        lambda d: d["shots_tot"] >= 8,
    ),
    (
        "goal_conversion",
        "Attack",
        "Shot conversion",
        "Goals per 100 shots · 8+ shots",
        "{:.0f}%",
        ALL_ROLES,
        lambda d: d["shots_tot"] >= 8,
    ),
    (
        "takeons_won_p90",
        "Attack",
        "The dribblers",
        "Take-ons won per 90",
        "{:.1f}",
        ALL_ROLES,
        None,
    ),
    (
        "takeon_success_pct",
        "Attack",
        "Dribble success",
        "Share of take-ons won · 10+ attempts",
        "{:.0f}%",
        ALL_ROLES,
        lambda d: d["takeons_tot"] >= 10,
    ),
    ("box_entries_p90", "Attack", "Entries into the box", "Per 90", "{:.1f}", ALL_ROLES, None),
    (
        "final_third_receptions_p90",
        "Attack",
        "Receiving in the final third",
        "Passes received in the attacking third, per 90",
        "{:.1f}",
        ALL_ROLES,
        None,
    ),
    ("positive_xT_p90", "Attack", "Threat added", "xT added per 90", "{:.2f}", ALL_ROLES, None),
    (
        "xGChain_p90",
        "Attack",
        "Involved in chances",
        "xG chain per 90: every move that led to a shot",
        "{:.2f}",
        ALL_ROLES,
        None,
    ),
    (
        "xGBuildup_p90",
        "Attack",
        "Build-up involvement",
        "xG build-up per 90, shooter and key passer excluded",
        "{:.2f}",
        ALL_ROLES,
        None,
    ),
    (
        "key_passes_p90",
        "Passing",
        "Key passes",
        "Passes that led to a shot, per 90",
        "{:.2f}",
        ALL_ROLES,
        None,
    ),
    (
        "assists_p90",
        "Passing",
        "Who sets up the goals",
        "Assists per 90, as the feed credits them",
        "{:.2f}",
        ALL_ROLES,
        None,
    ),
    (
        "passes_into_box_p90",
        "Passing",
        "Passes into the box",
        "Completed, per 90",
        "{:.1f}",
        ALL_ROLES,
        None,
    ),
    ("crosses_p90", "Passing", "Crossing", "Crosses per 90", "{:.1f}", ALL_ROLES, None),
    (
        "cross_pct",
        "Passing",
        "Crossing accuracy",
        "Share of crosses completed · 10+ crosses",
        "{:.0f}%",
        ALL_ROLES,
        lambda d: d["crosses_tot"] >= 10,
    ),
    ("passes_p90", "Passing", "Most passes", "Passes per 90", "{:.0f}", ALL_ROLES, None),
    (
        "pass_pct",
        "Passing",
        "Pass accuracy",
        "Share of passes completed · 40+ passes per 90",
        "{:.0f}%",
        ALL_ROLES,
        lambda d: d["passes_p90"] >= 40,
    ),
    (
        "progressive_passes_p90",
        "Passing",
        "Progressive passes",
        "Per 90",
        "{:.1f}",
        ALL_ROLES,
        None,
    ),
    (
        "progressive_pass_pct",
        "Passing",
        "Forward-looking passers",
        "Progressive share of completed passes · 40+ passes per 90",
        "{:.0f}%",
        ALL_ROLES,
        lambda d: d["passes_p90"] >= 40,
    ),
    (
        "progression_metres_p90",
        "Passing",
        "Ground gained",
        "Metres moved towards goal per 90 by completed passes and carries",
        "{:.0f}",
        ALL_ROLES,
        None,
    ),
    (
        "progressions_p90",
        "Passing",
        "Progressions",
        "Progressive passes and carries per 90",
        "{:.1f}",
        ALL_ROLES,
        None,
    ),
    (
        "line_breaking_passes_p90",
        "Passing",
        "Breaking the lines",
        "Line-breaking passes per 90",
        "{:.1f}",
        ALL_ROLES,
        None,
    ),
    (
        "passes_into_final_third_p90",
        "Passing",
        "Into the final third",
        "Completed passes into the attacking third, per 90",
        "{:.1f}",
        ALL_ROLES,
        None,
    ),
    (
        "long_balls_p90",
        "Passing",
        "Long balls",
        "Passes of 32 metres or more, per 90",
        "{:.1f}",
        ALL_ROLES,
        None,
    ),
    (
        "long_ball_pct",
        "Passing",
        "Long-ball accuracy",
        "Share of long balls completed · 10+ long balls",
        "{:.0f}%",
        ALL_ROLES,
        lambda d: d["long_balls_tot"] >= 10,
    ),
    ("touches_p90", "Passing", "Most on the ball", "Touches per 90", "{:.0f}", ALL_ROLES, None),
    (
        "xT_per_100_touches",
        "Passing",
        "Threat per touch",
        "xT added per 100 touches",
        "{:.2f}",
        ALL_ROLES,
        lambda d: d["touches_p90"] >= 30,
    ),
    (
        "tackles_p90",
        "Defence",
        "Tackling",
        "Tackles per 90",
        "{:.1f}",
        ["Defender", "Midfielder"],
        None,
    ),
    (
        "tackles_won_p90",
        "Defence",
        "Tackles won",
        "Per 90",
        "{:.1f}",
        ["Defender", "Midfielder"],
        None,
    ),
    (
        "tackle_pct",
        "Defence",
        "Tackle success",
        "Share of tackles won · 8+ tackles",
        "{:.0f}%",
        ["Defender", "Midfielder"],
        lambda d: d["tackles_tot"] >= 8,
    ),
    ("interceptions_p90", "Defence", "Interceptions", "Per 90", "{:.1f}", ALL_ROLES, None),
    ("clearances_p90", "Defence", "Clearances", "Per 90", "{:.1f}", ["Defender"], None),
    (
        "aerial_pct",
        "Defence",
        "Aerial duels won",
        "Share of aerial duels won · 15+ duels",
        "{:.0f}%",
        ALL_ROLES,
        lambda d: d["aerials_tot"] >= 15,
    ),
    (
        "defensive_actions_p90",
        "Defence",
        "Defensive actions",
        "Tackles, interceptions, recoveries, clearances and blocks per 90",
        "{:.1f}",
        ["Defender", "Midfielder"],
        None,
    ),
    (
        "fouls_committed_p90",
        "Defence",
        "Most fouls",
        "Fouls committed per 90",
        "{:.1f}",
        ALL_ROLES,
        None,
    ),
    (
        "recoveries_p90",
        "Pressing",
        "The ball winners",
        "Recoveries per 90",
        "{:.1f}",
        ALL_ROLES,
        None,
    ),
    ("fouls_won_p90", "Pressing", "Fouled most", "Fouls won per 90", "{:.1f}", ALL_ROLES, None),
    (
        "dispossessed_p90",
        "Pressing",
        "Most dispossessed",
        "Times the ball was lost under challenge, per 90",
        "{:.1f}",
        ALL_ROLES,
        None,
    ),
    ("goals_tot", "Overview", "Top scorers", "Goals over the block", "{:.0f}", ALL_ROLES, None),
    (
        "assists_tot",
        "Overview",
        "Most assists",
        "Assists over the block",
        "{:.0f}",
        ALL_ROLES,
        None,
    ),
    (
        "goal_contributions_tot",
        "Overview",
        "Goals and assists",
        "Total contributions over the block",
        "{:.0f}",
        ALL_ROLES,
        None,
    ),
    (
        "goals_minus_xG_tot",
        "Overview",
        "Beating the model",
        "Goals scored minus xG",
        "{:+.1f}",
        ALL_ROLES,
        None,
    ),
    (
        "yellow_cards_tot",
        "Overview",
        "Most booked",
        "Yellow cards over the block",
        "{:.0f}",
        ALL_ROLES,
        None,
    ),
]


def who_note(roles) -> str:
    """The footnote naming who is ranked: every outfield player, or just the positions listed."""
    if set(roles) >= set(ALL_ROLES):
        return f"Outfield players with {MIN_MINUTES:.0f}+ minutes."
    return f"{', '.join(r.lower() + 's' for r in roles)} with {MIN_MINUTES:.0f}+ minutes."


def player_ranking_card(spec) -> Card:
    key, group, title, subtitle, fmt, roles, where = spec
    index = PLAYER_RANKINGS.index(spec) + 1
    stem = f"{group.lower()}_{index:02d}_{key.lower().replace('_p90', '').replace('_tot', '')}"

    def draw(league):
        rows = player_frame(league)
        rows = rows[rows["role"].isin(roles)]
        if where is not None:
            rows = rows[where(rows)]
        rows = rows.dropna(subset=[key])
        return ranked_players(
            league,
            group,
            stem,
            title,
            subtitle,
            rows,
            key,
            fmt,
            who_note(roles),
        )

    return Card("players", group, stem, draw)


# ── club comparisons ─────────────────────────────────────────────────────────────────────────
def team_scatter(group, name, title, subtitle, x_key, y_key, x_label, y_label, note, corners):
    def draw(league):
        return quadrant(
            league,
            "teams",
            group,
            name,
            title,
            subtitle,
            per_match(league),
            x_key,
            y_key,
            x_label,
            y_label,
            note,
            corners,
        )

    return Card("teams", group, name, draw)


def t_finishing(league):
    t = totals(league).reset_index()
    return dumbbell(
        league,
        "teams",
        "Attack",
        "attack_goals_against_xg",
        "Goals against xG",
        "Who scored more than their chances said, and who fewer · five matches is a lean, not a verdict",
        t,
        "xG",
        "goals_for",
        "xG created",
        "Goals scored",
        "Green: scored more than expected. Red: fewer.",
        row_label=lambda r: r.team,
        row_team=lambda r: r.team,
    )


def t_goalkeeping(league):
    t = totals(league).reset_index()
    return dumbbell(
        league,
        "teams",
        "Defence",
        "defence_shots_faced_goals_let_in",
        "Shots faced and goals let in",
        "Post-shot xG of the shots on target faced, against goals conceded · green: conceded fewer than deserved",
        t,
        "xGoT_against",
        "goals_against",
        "xGOT faced",
        "Goals conceded",
        "Penalties and own goals included, so a figure here can differ from a goalkeeper's own.",
        row_label=lambda r: r.team,
        row_team=lambda r: r.team,
        b_higher_is_good=False,
    )


def t_conceding(league):
    t = totals(league).reset_index()
    return dumbbell(
        league,
        "teams",
        "Defence",
        "defence_goals_against_xg_against",
        "Goals conceded against xG",
        "What the opposition's chances were worth, against what they scored · green: conceded fewer than the chances said",
        t,
        "xG_against",
        "goals_against",
        "xG conceded",
        "Goals conceded",
        "Green: conceded fewer than expected. Red: more.",
        row_label=lambda r: r.team,
        row_team=lambda r: r.team,
        b_higher_is_good=False,
    )


def t_big_chances(league):
    t = per_match(league)
    return butterfly(
        league,
        "Attack",
        "attack_big_chances_for_against",
        "Big chances for and against",
        "Clear-cut chances created and conceded per match · ordered by both together",
        t["big_chances"],
        t["big_chances_against"],
        "Created",
        "Conceded (fewer is better)",
        "A big chance is one the provider marks as a clear opportunity to score.",
        fmt="{:.1f}",
        right_lower_better=True,
    )


def t_progress(league):
    t = per_match(league)
    return butterfly(
        league,
        "Passing",
        "passing_progressive_vs_line_breaking",
        "Getting the ball forward",
        "Progressive passes against line-breaking passes per match · ordered by both together",
        t["progressive_passes"],
        t["line_breaking_completed"],
        "Progressive passes",
        "Line-breaking (completed)",
        "A line-breaking pass starts behind the opponent's line and ends beyond it.",
    )


def t_set_pieces(league):
    t = per_match(league).assign(fk_pen=lambda x: x["free_kick_xG"] + x["penalty_xG"]).reset_index()
    return stacked(
        league,
        "teams",
        "Attack",
        "attack_chance_sources",
        "Where the chances come from",
        "Expected goals per match by how the move began",
        t,
        ["open_play_xG", "corner_xG", "fk_pen", "throw_in_xG"],
        ["Open play", "Corners", "Free kicks and pens", "Throw-ins"],
        "A throw-in is a restart, not a rehearsed set piece, so it sits apart from the dead balls.",
        row_name=lambda r: r.team,
        row_team=lambda r: r.team,
        fmt="{:.2f}",
    )


def t_defence_makeup(league):
    t = per_match(league).reset_index()
    return stacked(
        league,
        "teams",
        "Defence",
        "defence_what_a_defence_is_made_of",
        "What a defence is made of",
        "Tackles, interceptions, blocks and clearances per match",
        t,
        ["tackles", "interceptions", "blocks", "clearances"],
        ["Tackles", "Interceptions", "Blocks", "Clearances"],
        "Clearances count a club defending its own box, so a long bar is not always a good one.",
        row_name=lambda r: r.team,
        row_team=lambda r: r.team,
    )


def t_ppda(league):
    return ranked_teams(
        league,
        "Pressing",
        "pressing_00_ppda",
        "Who presses hardest",
        "PPDA: opponent passes allowed per defensive action in the front 60% · lower is harder",
        per_match(league)["ppda"],
        "{:.1f}",
        "Bars run from hardest to softest. Dashed line is the league average.",
        lower_better=True,
    )


def t_xg_against(league):
    return ranked_teams(
        league,
        "Defence",
        "defence_00_xg_against",
        "Hardest to break down",
        "Expected goals conceded per match · the shorter the bar, the better the defence",
        per_match(league)["xG_against"],
        "{:.2f}",
        "Top three in club colour. Dashed line is the league average.",
        lower_better=True,
    )


def t_counterpress(league):
    return ranked_teams(
        league,
        "Pressing",
        "pressing_00_counterpress",
        "Winning it back at once",
        "Share of lost possessions won back within seconds · the counter-press",
        per_match(league)["counterpress_success_rate"],
        "{:.1f}%",
        "Top three in club colour. Dashed line is the league average.",
    )


def t_into_box(league):
    return ranked_teams(
        league,
        "Passing",
        "passing_00_passes_into_box",
        "Into the box",
        "Completed passes into the penalty area per match",
        per_match(league)["passes_into_box"],
        "{:.1f}",
        "Top three in club colour. Dashed line is the league average.",
    )


def heat_table(league):
    """The league on one page: points, goal difference and twelve per-match figures, shaded by rank."""
    columns = [
        ("xG", "xG", False, "{:.2f}"),
        ("xG_against", "xGA", True, "{:.2f}"),
        ("shots", "SH", False, "{:.1f}"),
        ("shots_against", "SH-A", True, "{:.1f}"),
        ("ppda", "PPDA", True, "{:.1f}"),
        ("field_tilt", "TILT", False, "{:.0f}"),
        ("pass_pct", "PASS%", False, "{:.0f}"),
        ("progressive_passes", "PROG", False, "{:.0f}"),
        ("line_breaking_completed", "LBP", False, "{:.0f}"),
        ("high_regains", "HIGH", False, "{:.1f}"),
        ("box_entries", "BOX", False, "{:.1f}"),
        ("key_passes", "KEY", False, "{:.1f}"),
    ]
    tot, per = totals(league), per_match(league)
    per = per.loc[tot.index]
    fig = frame(
        league,
        "Overview",
        "The league at a glance",
        "Per-match figures shaded by rank in the league · brighter is better, whichever way the number runs",
        "Ordered by points. PPDA, xGA and shots against are shaded the other way round: lower is better.",
        reading=[
            ("SH", "shots per match"),
            ("LBP", "line-breaking passes"),
            ("HIGH", "regains in the final third"),
            ("TILT", "field tilt, %"),
            ("PROG", "progressive passes"),
        ],
    )
    ax = table_axes(fig)
    ramp = percentile_ramp()
    n, top, bottom = len(per), 0.915, 0.075
    row_h = (top - bottom - 0.04) / n
    left, right = TABLE_LEFT, 0.975
    name_w, lead_w = 0.135, 0.08
    grid_w = (right - left - name_w - lead_w) / len(columns)
    heads = ["PTS", "GD"] + [c[1] for c in columns]
    xs = [left + name_w + lead_w * (0.25 + 0.5 * i) for i in range(2)] + [
        left + name_w + lead_w + grid_w * (i + 0.5) for i in range(len(columns))
    ]
    for x, head in zip(xs, heads):
        fig.text(
            x,
            top - 0.005,
            head,
            color=base.MUTED,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
    ranks = {c[0]: lb.percentile_rank(per[c[0]], c[2]) for c in columns}
    for i, team in enumerate(per.index):
        y = top - 0.03 - (i + 1) * row_h
        ax.add_patch(
            Rectangle(
                (left - 0.005, y),
                right - left + 0.005,
                row_h - 0.002,
                facecolor=base.PANEL if i % 2 == 0 else base.BG,
                edgecolor="none",
            )
        )
        fig.text(left + 0.003, y + row_h / 2, f"{i + 1}", color=base.MUTED, fontsize=8, va="center")
        lb.put_crest(ax, league, team, (left + 0.03, y + row_h / 2), zoom=0.075)
        fig.text(
            left + 0.045,
            y + row_h / 2,
            team,
            color=base.TEXT,
            fontsize=8.5,
            fontweight="bold",
            va="center",
        )
        for x, text in zip(
            xs[:2], (f"{tot.loc[team, 'points']:.0f}", f"{tot.loc[team, 'goal_difference']:+.0f}")
        ):
            fig.text(
                x, y + row_h / 2, text, color=base.TEXT, va="center", ha="center", **display(14)
            )
        for j, (key, _h, _low, fmt) in enumerate(columns):
            cx = left + name_w + lead_w + grid_w * j
            rank = float(ranks[key].loc[team])
            ax.add_patch(
                Rectangle(
                    (cx + 0.002, y + 0.003),
                    grid_w - 0.004,
                    row_h - 0.008,
                    facecolor=ramp(rank),
                    edgecolor="none",
                )
            )
            fig.text(
                cx + grid_w / 2,
                y + row_h / 2,
                fmt.format(per.loc[team, key]),
                color=base.BG if rank > 0.62 else base.TEXT,
                fontsize=7.8,
                fontweight="bold",
                va="center",
                ha="center",
            )
    return save(fig, league, "teams", "overview_01_league_heat_table")


def _xg_quadrant(league):
    t = per_match(league).assign(fewer_conceded=lambda x: -x["xG_against"])
    return quadrant(
        league,
        "teams",
        "Overview",
        "overview_02_chances_for_against",
        "Chances for and against",
        "xG created against xG conceded per match · the best sides sit top right",
        t,
        "xG",
        "fewer_conceded",
        "xG created per match  →",
        "Fewer xG conceded  → (axis flipped)",
        "Up is better on both axes.",
        ("DOMINANT", "SOLID BUT BLUNT", "OPEN GAMES", "STRUGGLING"),
    )


def goalkeepers(league, minimum_minutes=180.0):
    keepers = league.players[
        lb.is_goalkeeper(league.players) & (league.players["minutes"] >= minimum_minutes)
    ]
    keepers = keepers.sort_values("goals_prevented", ascending=False)
    fig = frame(
        league,
        "Defence",
        "Goalkeepers",
        f"Goals prevented is the post-shot xG of the shots faced less the goals conceded · keepers with {minimum_minutes:.0f}+ minutes",
        "Penalties and own goals are left out of goals prevented.",
    )
    ax = table_axes(fig)
    heads = [
        ("MIN", "minutes", "{:.0f}", 0.58),
        ("SAVES", "saves", "{:.0f}", 0.64),
        ("CLAIMS", "claims", "{:.0f}", 0.70),
        ("SWEEPS", "sweeps", "{:.0f}", 0.76),
        ("PASS %", "pass_pct", "{:.0f}", 0.82),
        ("PREVENTED", "goals_prevented", "{:+.2f}", 0.89),
    ]
    top = 0.915
    row_h = min(0.04, (top - 0.09) / max(len(keepers), 1))
    for head, _, _, x in heads:
        fig.text(
            x, top, head, color=base.MUTED, fontsize=8, fontweight="bold", ha="center", va="bottom"
        )
    fig.text(
        TABLE_LEFT + 0.04,
        top,
        "GOALKEEPER",
        color=base.MUTED,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )
    span = float(np.abs(keepers["goals_prevented"]).max()) or 1.0
    for i, (_, row) in enumerate(keepers.iterrows()):
        y = top - 0.012 - (i + 1) * row_h
        ax.add_patch(
            Rectangle(
                (TABLE_LEFT, y),
                0.975 - TABLE_LEFT,
                row_h - 0.003,
                facecolor=base.PANEL if i % 2 == 0 else base.BG,
                edgecolor="none",
            )
        )
        lb.put_crest(ax, league, row["team"], (TABLE_LEFT + 0.022, y + row_h / 2), zoom=0.07)
        fig.text(
            TABLE_LEFT + 0.04,
            y + row_h / 2,
            lb.surname(row["player"]),
            color=base.TEXT,
            fontsize=8.8,
            fontweight="bold",
            va="center",
        )
        fig.text(
            TABLE_LEFT + 0.125,
            y + row_h / 2,
            row["team"],
            color=base.NEUTRAL,
            fontsize=7.2,
            va="center",
        )
        for _, key, fmt, x in heads:
            value = row[key]
            if pd.isna(value):
                continue
            if key == "goals_prevented":
                width = 0.04 * abs(value) / span
                ax.add_patch(
                    Rectangle(
                        (x if value >= 0 else x - width, y + row_h * 0.3),
                        width,
                        row_h * 0.35,
                        facecolor=league.colour(row["team"]),
                        edgecolor="none",
                    )
                )
                fig.text(
                    0.94,
                    y + row_h / 2,
                    fmt.format(value),
                    color=base.TEXT,
                    va="center",
                    ha="left",
                    **display(13),
                )
            else:
                fig.text(
                    x,
                    y + row_h / 2,
                    fmt.format(value),
                    color=base.TEXT,
                    fontsize=8.8,
                    va="center",
                    ha="center",
                )
    return save(fig, league, "players", "defence_00_goalkeepers")


# ── player comparisons ───────────────────────────────────────────────────────────────────────
def player_scatter(
    group,
    name,
    title,
    subtitle,
    x_key,
    y_key,
    x_label,
    y_label,
    note,
    corners,
    roles=ALL_ROLES,
    where=None,
    label_n=14,
):
    def draw(league):
        rows = player_frame(league)
        rows = rows[rows["role"].isin(roles)]
        if where is not None:
            rows = rows[where(rows)]
        rows = rows.dropna(subset=[x_key, y_key])
        return quadrant(
            league,
            "players",
            group,
            name,
            title,
            subtitle,
            rows,
            x_key,
            y_key,
            x_label,
            y_label,
            f"{note} Players with {MIN_MINUTES:.0f}+ minutes; named: the furthest from the pack.",
            corners,
            as_teams=False,
            label_n=label_n,
        )

    return Card("players", group, name, draw)


def p_finishers(league):
    p = squad_totals(league).sort_values("xG", ascending=False).head(14)
    return dumbbell(
        league,
        "players",
        "Attack",
        "attack_00_goals_against_xg",
        "Scoring more than expected",
        f"Goals against xG for the 14 players with the most xG · {MIN_MINUTES:.0f}+ minutes",
        p,
        "xG",
        "goals",
        "xG",
        "Goals",
        "Green: scored more than expected. Red: fewer.",
        row_label=lambda r: lb.surname(r.player),
        row_team=lambda r: r.team,
    )


def p_contributions(league):
    p = squad_totals(league)
    p = (
        p.assign(_t=p["goals"] + p["assists"])
        .sort_values(["_t", "goals"], ascending=False)
        .head(14)
    )
    return stacked(
        league,
        "players",
        "Attack",
        "attack_00_goals_and_assists",
        "Goals and assists",
        "Direct goal contributions · the players with the most",
        p,
        ["goals", "assists"],
        ["Goals", "Assists"],
        f"Players with {MIN_MINUTES:.0f}+ minutes. Assists are those the feed credits on the goal.",
        row_name=lambda r: lb.surname(r.player),
        row_team=lambda r: r.team,
        fmt="{:.0f}",
    )


def p_ballwinners(league):
    r = player_frame(league)
    r = r[r["role"].isin(["Defender", "Midfielder"])]
    r = r.assign(_t=r[["tackles_won_p90", "interceptions_p90"]].sum(axis=1)).nlargest(14, "_t")
    return stacked(
        league,
        "players",
        "Defence",
        "defence_00_who_stops_the_attack",
        "Who stops the attack",
        "Tackles won plus interceptions per 90 · the ball won from the opponent, not cleared",
        r,
        ["tackles_won_p90", "interceptions_p90"],
        ["Tackles won", "Interceptions"],
        f"Defenders and midfielders with {MIN_MINUTES:.0f}+ minutes.",
        row_name=lambda r: lb.surname(r.player),
        row_team=lambda r: r.team,
    )


def p_air(league):
    p = squad_totals(league)
    p = (
        p.assign(aerials_lost=p["aerials"] - p["aerials_won"])
        .sort_values("aerials_won", ascending=False)
        .head(14)
    )
    return stacked(
        league,
        "players",
        "Defence",
        "defence_00_who_wins_in_the_air",
        "Who wins in the air",
        "Aerial duels won and lost over the block · the players who contest the most",
        p,
        ["aerials_won", "aerials_lost"],
        ["Won", "Lost"],
        f"Players with {MIN_MINUTES:.0f}+ minutes. A duel is logged for both players.",
        row_name=lambda r: lb.surname(r.player),
        row_team=lambda r: r.team,
        fmt="{:.0f}",
    )


def p_forwards_press(league):
    r = player_frame(league)
    r = r[r["role"].eq("Forward")]
    return ranked_players(
        league,
        "Pressing",
        "pressing_00_forwards_who_work",
        "Forwards who work",
        "Defensive actions per 90 by forwards · who starts the press",
        r,
        "defensive_actions_p90",
        "{:.1f}",
        f"Forwards with {MIN_MINUTES:.0f}+ minutes.",
        n=12,
    )


def _team_scatter_cards():
    return [
        team_scatter(
            "Attack",
            "attack_00_volume_against_quality",
            "Volume against quality",
            "Shots per match against the xG of each shot · top right is plenty of good chances",
            "shots",
            "xG_per_shot",
            "Shots per match  →",
            "xG per shot  →",
            "Shot quality is the model's xG divided by shots.",
            ("HIGH VOLUME, HIGH QUALITY", "PATIENT AND CLINICAL", "SHOOT ON SIGHT", "FEW AND POOR"),
        ),
        team_scatter(
            "Attack",
            "attack_00_entries_against_xg",
            "Getting into the box",
            "Entries into the penalty area against xG per match · who turns territory into chances",
            "box_entries",
            "xG",
            "Box entries per match  →",
            "xG per match  →",
            "A box entry is a pass or carry that ends in the penalty area.",
            ("MANY ENTRIES, MANY CHANCES", "EFFICIENT", "ENTRIES THAT GO NOWHERE", "STUCK OUTSIDE"),
        ),
        team_scatter(
            "Attack",
            "attack_00_creation",
            "Creating the shot",
            "Key passes against passes into the box per match",
            "passes_into_box",
            "key_passes",
            "Passes into the box per match  →",
            "Key passes per match  →",
            "A key pass is the pass before a shot.",
            ("BOTH", "SHOT SUPPLIERS", "BOX PASSERS", "NEITHER"),
        ),
        team_scatter(
            "Defence",
            "defence_00_fouls_and_bookings",
            "Fouls and bookings",
            "Fouls committed against yellow cards per match · who plays on the edge",
            "fouls_committed",
            "yellow_cards",
            "Fouls per match  →",
            "Yellow cards per match  →",
            "Red cards are rare enough that they are left out.",
            ("FOUL OFTEN, BOOKED OFTEN", "CAUTIOUS, BOOKED", "FOUL OFTEN, NOT BOOKED", "CLEAN"),
        ),
        team_scatter(
            "Defence",
            "defence_00_tackles_and_interceptions",
            "Two ways to win it",
            "Tackles against interceptions per match · meeting the attacker or reading the pass",
            "tackles",
            "interceptions",
            "Tackles per match  →",
            "Interceptions per match  →",
            "Both are counted from the event feed.",
            ("BOTH", "READERS", "TACKLERS", "LITTLE OF EITHER"),
        ),
        team_scatter(
            "Defence",
            "defence_00_pressing_and_conceding",
            "Does pressing protect?",
            "PPDA against xG conceded · harder pressing to the left",
            "ppda",
            "xG_against",
            "PPDA → softer press",
            "xG conceded per match  →",
            "Lower PPDA is a harder press.",
            ("SOFT PRESS, OPEN", "HARD PRESS, OPEN", "SOFT PRESS, SOLID", "HARD PRESS, SOLID"),
        ),
        team_scatter(
            "Pressing",
            "pressing_00_press_wins",
            "Pressing that pays",
            "Possessions won in the final third per match against how many turned into a shot",
            "high_regains",
            "regain_to_shot_rate",
            "High regains per match  →",
            "Regains that became a shot (%)  →",
            "A regain is a possession won back; high means in the attacking third.",
            ("WIN IT HIGH, USE IT", "SELECTIVE", "WIN IT HIGH, WASTE IT", "NOT PRESSING"),
        ),
        team_scatter(
            "Pressing",
            "pressing_00_style",
            "Press and territory",
            "PPDA against field tilt · harder pressing to the left",
            "ppda",
            "field_tilt",
            "PPDA → softer press",
            "Field tilt (%)  →",
            "Field tilt is the share of final-third touches.",
            (
                "SOFT PRESS, HIGH TERRITORY",
                "HARD PRESS, HIGH TERRITORY",
                "SOFT PRESS, LOW BLOCK",
                "HARD PRESS, LOW BLOCK",
            ),
        ),
        team_scatter(
            "Pressing",
            "pressing_00_counterpress_vs_regains",
            "Counter-press and regains",
            "Counter-press success against possessions won overall, per match",
            "possession_regains",
            "counterpress_success_rate",
            "Possessions won per match  →",
            "Counter-press success (%)  →",
            "Counter-press: winning the ball back within seconds of losing it.",
            ("WIN IT BACK AT ONCE", "SELECTIVE", "BUSY, NOT IMMEDIATE", "PASSIVE"),
        ),
        team_scatter(
            "Passing",
            "passing_00_safe_or_direct",
            "Safe or direct",
            "How often passes arrive against how direct the play is · top right is accurate and vertical",
            "pass_pct",
            "directness",
            "Pass completion (%)  →",
            "Directness (%)  →",
            "Directness is the share of progress made towards goal.",
            ("ACCURATE AND DIRECT", "LONG AND RISKY", "SHORT AND SAFE", "NEITHER"),
        ),
        team_scatter(
            "Passing",
            "passing_00_crosses_or_long_balls",
            "Crosses or long balls",
            "Two ways round a block, per match · which one a side reaches for",
            "crosses",
            "long_balls",
            "Crosses per match  →",
            "Long balls per match  →",
            "A long ball is a pass of 32 metres or more.",
            ("BOTH", "LONG AND DIRECT", "WIDE DELIVERY", "SHORT GAME"),
        ),
        team_scatter(
            "Passing",
            "passing_00_possession_pays",
            "Does possession pay?",
            "Possession against the xG difference per match",
            "possession_share",
            "xG_difference",
            "Possession (%)  →",
            "xG created minus xG conceded, per match  →",
            "Possession is the share of possessions, not of the time.",
            (
                "DOMINANT AND DANGEROUS",
                "EFFICIENT WITHOUT THE BALL",
                "ALL THE BALL, LITTLE TO SHOW",
                "OUTPLAYED",
            ),
        ),
        team_scatter(
            "Passing",
            "passing_00_accuracy_and_territory",
            "Accuracy and territory",
            "Pass completion against field tilt per match",
            "pass_pct",
            "field_tilt",
            "Pass completion (%)  →",
            "Field tilt (%)  →",
            "Field tilt is the share of final-third touches.",
            ("CLEAN AND HIGH UP", "ROUGH BUT HIGH UP", "CLEAN BUT DEEP", "ROUGH AND DEEP"),
        ),
    ]


def _player_scatter_cards():
    return [
        player_scatter(
            "Attack",
            "attack_00_finishers_and_creators",
            "Finishers and creators",
            "xG against xA per 90 · bigger dots played more",
            "xG_p90",
            "xA_p90",
            "xG per 90  →",
            "xA per 90  →",
            "",
            ("COMPLETE FORWARDS", "CREATORS", "FINISHERS", ""),
        ),
        player_scatter(
            "Attack",
            "attack_00_who_makes_the_chances",
            "Who makes the chances",
            "Key passes against passes into the box per 90",
            "key_passes_p90",
            "passes_into_box_p90",
            "Key passes per 90  →",
            "Passes into the box per 90  →",
            "",
            ("CREATORS", "FINAL BALL", "SHOT PROVIDERS", ""),
        ),
        player_scatter(
            "Attack",
            "attack_00_shoot_often_shoot_straight",
            "Shoot often, shoot straight",
            "Shots per 90 against the share that hit the target · averaging a shot a game or more",
            "shots_p90",
            "shot_accuracy",
            "Shots per 90  →",
            "Shots on target (%)  →",
            "",
            ("VOLUME AND ACCURACY", "SELECTIVE", "SPRAYING IT", ""),
            roles=["Forward", "Midfielder"],
            where=lambda d: d["shots_p90"] >= 1.0,
        ),
        player_scatter(
            "Attack",
            "attack_00_dribbling_and_losing_it",
            "Dribble and lose it",
            "Take-ons won against times dispossessed per 90",
            "takeons_won_p90",
            "dispossessed_p90",
            "Take-ons won per 90  →",
            "Dispossessed per 90  →",
            "",
            ("RISK TAKERS", "CARELESS", "SECURE DRIBBLERS", "SAFE"),
        ),
        player_scatter(
            "Attack",
            "attack_00_threat_and_touches",
            "Threat and touches",
            "xT added per 90 against touches per 90",
            "touches_p90",
            "positive_xT_p90",
            "Touches per 90  →",
            "xT added per 90  →",
            "",
            ("BUSY AND DANGEROUS", "EFFICIENT", "BUSY, LITTLE THREAT", "QUIET"),
        ),
        player_scatter(
            "Attack",
            "attack_00_chain_and_buildup",
            "In the move, not at the end",
            "xG chain against xG build-up per 90",
            "xGBuildup_p90",
            "xGChain_p90",
            "xG build-up per 90  →",
            "xG chain per 90  →",
            "",
            ("INVOLVED THROUGHOUT", "NEAR THE SHOT", "DEEP BUILDERS", ""),
        ),
        player_scatter(
            "Defence",
            "defence_00_where_they_defend",
            "Where they defend",
            "Defensive actions per 90 against how far up the pitch they happen",
            "defensive_height",
            "defensive_actions_p90",
            "Average height of defensive actions  → further upfield",
            "Defensive actions per 90  →",
            "",
            ("HIGH AND BUSY", "DEEP AND BUSY", "HIGH AND QUIET", "DEEP AND QUIET"),
            roles=["Defender", "Midfielder"],
            where=lambda d: d["defensive_height"].notna(),
        ),
        player_scatter(
            "Defence",
            "defence_00_tackling",
            "Tackling: how many, how well",
            "Tackles per 90 against the share won · players making a tackle a game or more",
            "tackles_p90",
            "tackle_pct",
            "Tackles per 90  →",
            "Tackles won (%)  →",
            "",
            ("BUSY AND WINNING", "SELECTIVE", "BUSY AND LOSING", ""),
            roles=["Defender", "Midfielder"],
            where=lambda d: d["tackles_p90"] >= 1.0,
        ),
        player_scatter(
            "Defence",
            "defence_00_clearing_and_reading",
            "Clear it or read it",
            "Clearances against interceptions per 90 · defenders",
            "clearances_p90",
            "interceptions_p90",
            "Clearances per 90  →",
            "Interceptions per 90  →",
            "",
            ("BOTH", "READERS", "CLEARERS", ""),
            roles=["Defender"],
        ),
        player_scatter(
            "Defence",
            "defence_00_fouls_won_and_given",
            "Fouls given and won",
            "Fouls committed against fouls won per 90",
            "fouls_committed_p90",
            "fouls_won_p90",
            "Fouls committed per 90  →",
            "Fouls won per 90  →",
            "",
            ("PHYSICAL BOTH WAYS", "FOULED OFTEN", "FOUL OFTEN", ""),
        ),
        player_scatter(
            "Pressing",
            "pressing_00_recoveries_and_tackles",
            "Winning the ball",
            "Recoveries against tackles won per 90",
            "recoveries_p90",
            "tackles_won_p90",
            "Recoveries per 90  →",
            "Tackles won per 90  →",
            "",
            ("BALL WINNERS", "TACKLERS", "SWEEPERS", ""),
            roles=["Defender", "Midfielder"],
        ),
        player_scatter(
            "Passing",
            "passing_00_progress_and_line_breaking",
            "Moving the ball forward",
            "Progression metres against line-breaking passes per 90",
            "progression_metres_p90",
            "line_breaking_passes_p90",
            "Progression metres per 90  →",
            "Line-breaking passes per 90  →",
            "",
            ("CARRY AND BREAK", "BREAK LINES", "CARRY THE BALL", ""),
        ),
        player_scatter(
            "Passing",
            "passing_00_safe_or_ambitious",
            "Safe or ambitious",
            "Pass completion against the share of completed passes that were progressive · defenders and midfielders",
            "pass_pct",
            "progressive_pass_pct",
            "Pass completion (%)  →",
            "Progressive share of completed passes (%)  →",
            "",
            ("SAFE AND FORWARD", "AMBITIOUS", "SAFE AND SIDEWAYS", "RECYCLERS"),
            roles=["Defender", "Midfielder"],
            where=lambda d: d["passes_p90"] >= 30,
        ),
        player_scatter(
            "Passing",
            "passing_00_aerial_duels",
            "In the air",
            "Aerial duels contested against share won",
            "aerials_tot",
            "aerial_pct",
            "Aerial duels contested  →",
            "Share won (%)  →",
            "",
            ("WIN A LOT", "SELECTIVE", "CONTEST A LOT, LOSE", ""),
            where=lambda d: d["aerials_tot"] >= 10,
        ),
    ]


# ── the full set ───────────────────────────────────────────────────────────────────────────────
def cards(league: lb.League) -> list[Card]:
    """Every card for this block: club and player rankings, comparisons and profiles."""
    out = [team_ranking_card(spec) for spec in TEAM_RANKINGS if spec[0] not in TEAM_SKIP]
    out += [
        Card("teams", "Overview", "overview_01_league_heat_table", heat_table),
        Card("teams", "Overview", "overview_02_chances_for_against", _xg_quadrant),
        Card("teams", "Attack", "attack_goals_against_xg", t_finishing),
        Card("teams", "Attack", "attack_big_chances_for_against", t_big_chances),
        Card("teams", "Attack", "attack_chance_sources", t_set_pieces),
        Card("teams", "Defence", "defence_00_xg_against", t_xg_against),
        Card("teams", "Defence", "defence_shots_faced_goals_let_in", t_goalkeeping),
        Card("teams", "Defence", "defence_goals_against_xg_against", t_conceding),
        Card("teams", "Defence", "defence_what_a_defence_is_made_of", t_defence_makeup),
        Card("teams", "Pressing", "pressing_00_ppda", t_ppda),
        Card("teams", "Pressing", "pressing_00_counterpress", t_counterpress),
        Card("teams", "Passing", "passing_progressive_vs_line_breaking", t_progress),
        Card("teams", "Passing", "passing_00_passes_into_box", t_into_box),
    ]
    out += _team_scatter_cards()
    out += [player_ranking_card(spec) for spec in PLAYER_RANKINGS]
    out += [
        Card("players", "Defence", "defence_00_goalkeepers", goalkeepers),
        Card("players", "Attack", "attack_00_goals_against_xg", p_finishers),
        Card("players", "Attack", "attack_00_goals_and_assists", p_contributions),
        Card("players", "Defence", "defence_00_who_stops_the_attack", p_ballwinners),
        Card("players", "Defence", "defence_00_who_wins_in_the_air", p_air),
        Card("players", "Pressing", "pressing_00_forwards_who_work", p_forwards_press),
    ]
    out += _player_scatter_cards()
    return out


def profiles(league: lb.League, min_minutes: float = MIN_MINUTES):
    """A profile card for each club, then for each player over the minutes floor."""
    out = [
        Card(
            "teams",
            "Profile",
            f"profile_{t.replace(' ', '_')}",
            lambda lg, t=t: lb.team_profile(lg, t),
        )
        for t in league.teams["team"]
    ]
    pool = lb._qualified(league, min_minutes)
    out += [
        Card(
            "players",
            "Profile",
            f"profile_{r.team}_{r.player}",
            lambda lg, p=r.player, t=r.team: lb.player_profile(lg, p, t, min_minutes),
        )
        for r in pool.itertuples()
    ]
    return out


def number_in_group(path: Path, card: Card, sequence: dict) -> Path:
    """Rename a card to ``<group>_<nn>_<what>`` with ``nn`` counting up inside its group and folder."""
    group = card.group.lower()
    stem = re.sub(rf"^{group}_(\d+_)?", "", path.stem)
    slot = (card.scope, group)
    sequence[slot] = sequence.get(slot, 0) + 1
    target = path.with_name(f"{group}_{sequence[slot]:02d}_{stem}.png")
    path.replace(target)
    return target


def build_all(
    league: lb.League,
    only: list[str] | None = None,
    scope: str | None = None,
    with_profiles: bool = True,
    report=None,
) -> list[Path]:
    """Draw the cards; ``only`` filters by a substring of the file stem, ``scope`` by teams or players."""
    base.theme()
    sequence: dict[tuple[str, str], int] = {}
    chosen = cards(league) + (profiles(league) if with_profiles else [])
    paths = []
    for card in chosen:
        if scope and card.scope != scope:
            continue
        if only and not any(word in card.name for word in only):
            continue
        try:
            path = card.draw(league)
        except Exception as error:  # one broken card must not lose the other two hundred
            if report:
                report(f"skipped {card.scope}/{card.name}: {type(error).__name__}: {error}")
            continue
        if path is None:
            continue
        if card.group != "Profile":
            path = number_in_group(path, card, sequence)
        paths.append(path)
    return paths
