"""Single-message graphics for a block of rounds, sized for a timeline.

Each card makes one point about attack, defence, pressing or passing, for clubs or for
players, and each uses the chart that suits that point (a dumbbell for results against
expectation, stacked bars for what a defence is made of, a scatter for two measures at
once) rather than one template. Every card is drawn twice from the same code: 4:5
(1200 x 1500), the tallest frame a timeline shows uncropped, and 16:9 (1600 x 900), the
widest. The wide frame moves the title into a left-hand panel so the chart keeps the full
height. All read from the aggregate tables ``scripts/aggregate_rounds.py`` writes.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from football_analysis.visuals import league_boards as lb
from football_analysis.visuals import visual_redesign_preview as base
from football_analysis.visuals.typography import display

GOOD, BAD, GOLD = "#3DDC84", "#FF6B5B", "#F2B134"
GREY = "#8E99A4"
PARTS = ["#4EA8FF", "#F2B134", "#E8452C", "#A77BFF"]
MIN_MINUTES = 270.0


@dataclass(frozen=True)
class Geo:
    """Where things go in one frame."""

    name: str
    size: tuple
    dpi: float
    wide: bool
    list_rect: tuple  # a chart with a label for every row
    plot_rect: tuple  # a scatter
    crest: float  # badge zoom beside a row
    row_pad: float  # room for the badge between the row label and the bars


FOUR_BY_FIVE = Geo(
    "4x5", (8.0, 10.0), 150, False, (0.255, 0.07, 0.66, 0.735), (0.11, 0.10, 0.83, 0.70), 0.11, 30
)
SIXTEEN_BY_NINE = Geo(
    "16x9",
    (12.0, 6.75),
    1600 / 12,
    True,
    (0.50, 0.07, 0.46, 0.88),
    (0.40, 0.095, 0.57, 0.86),
    0.085,
    24,
)
GEO = FOUR_BY_FIVE  # the frame being drawn; build_all switches it


@dataclass
class Card:
    name: str
    group: str
    draw: Callable[[lb.League], Path]


# ── furniture ────────────────────────────────────────────────────────────────────────
def frame(league: lb.League, group: str, title: str, subtitle: str, note: str = ""):
    """The shared page. Tall: title across the top. Wide: title in a panel on the left."""
    g = GEO
    fig = plt.figure(figsize=g.size, facecolor=base.BG)
    if not g.wide:
        kicker = f"{league.title}  ·  {league.rounds}  ·  {group.upper()}"
        fig.text(0.06, 0.957, kicker, color=lb.ACCENT, fontsize=9.5, fontweight="bold")
        size = 54 if len(title) <= 18 else (46 if len(title) <= 24 else 38)
        fig.text(0.06, 0.905, title.upper(), color=base.TEXT, va="center", **display(size))
        fig.text(0.06, 0.857, subtitle, color=base.MUTED, fontsize=10.5, va="center")
        fig.add_artist(
            Rectangle((0.06, 0.835), 0.88, 0.003, transform=fig.transFigure, color=lb.ACCENT, lw=0)
        )
        fig.text(0.06, 0.028, note, color=base.NEUTRAL, fontsize=7.5, va="center")
        fig.text(
            0.94, 0.028, "MOSTAFA SAAD", color=base.MUTED, ha="right", va="center", **display(14)
        )
        return fig
    fig.text(
        0.035,
        0.945,
        f"{league.title}  ·  {league.rounds}",
        color=lb.ACCENT,
        fontsize=8.5,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.035, 0.915, group.upper(), color=base.MUTED, fontsize=8.5, fontweight="bold", va="top"
    )
    lines = textwrap.wrap(title.upper(), 15)
    fig.text(
        0.035,
        0.86,
        "\n".join(lines),
        color=base.TEXT,
        va="top",
        linespacing=0.92,
        **display(44 if len(lines) <= 3 else 36),
    )
    top = 0.86 - 0.092 * len(lines) - 0.02
    fig.add_artist(
        Rectangle((0.035, top), 0.07, 0.006, transform=fig.transFigure, color=lb.ACCENT, lw=0)
    )
    wrapped = textwrap.wrap(subtitle, 40)
    fig.text(
        0.035,
        top - 0.03,
        "\n".join(wrapped),
        color=base.MUTED,
        fontsize=9.5,
        va="top",
        linespacing=1.45,
    )
    fig.text(
        0.035,
        0.085,
        textwrap.fill(note, 52),
        color=base.NEUTRAL,
        fontsize=7.2,
        va="bottom",
        linespacing=1.4,
    )
    fig.text(0.035, 0.03, "MOSTAFA SAAD", color=base.MUTED, va="center", **display(14))
    fig._card_top = top - 0.03 - 0.04 * len(wrapped) - 0.04
    return fig


def legend(fig, items):
    """A key of (symbol, label, colour): a row under the rule when tall, a column in the panel when wide."""
    if GEO.wide:
        y = getattr(fig, "_card_top", 0.4)
        for symbol, label, colour in items:
            fig.text(0.035, y, symbol, color=colour, fontsize=10, va="center", fontweight="bold")
            fig.text(
                0.062, y, label, color=base.MUTED, fontsize=8.5, va="center", fontweight="bold"
            )
            y -= 0.052
        return
    x = 0.255
    for symbol, label, colour in items:
        fig.text(x, 0.817, symbol, color=colour, fontsize=10, va="center", fontweight="bold")
        fig.text(
            x + 0.022, 0.817, label, color=base.MUTED, fontsize=8.5, va="center", fontweight="bold"
        )
        x += 0.04 + 0.0115 * len(label)


def save(fig, league: lb.League, filename: str) -> Path:
    folder = league.folder / "visuals" / f"twitter_{GEO.name}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{filename}.png"
    fig.savefig(path, dpi=GEO.dpi, facecolor=base.BG)
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


def crest_at(ax, league, team, y, zoom=None):
    """A badge just left of the axes, at row ``y``."""
    lb.put_crest(
        ax, league, team, (-0.045, y), zoom=zoom or GEO.crest, transform=ax.get_yaxis_transform()
    )


def row_labels(ax, league, labels, teams, size=9.5, zoom=None):
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=size, color=base.TEXT)
    ax.tick_params(axis="y", pad=GEO.row_pad)
    for i, team in enumerate(teams):
        crest_at(ax, league, team, i, zoom)


def list_axes(fig, shrink=1.0):
    x, y, w, h = GEO.list_rect
    return fig.add_axes([x, y, w * shrink, h])


def pool(league: lb.League, roles=None) -> pd.DataFrame:
    frame_ = lb._qualified(league, MIN_MINUTES)
    return frame_ if roles is None else frame_[frame_["role"].isin(roles)]


def value_labels(ax, ys, values, fmt, pad, size=9.5):
    for y, v in zip(ys, values):
        ax.text(
            v + pad,
            y,
            fmt.format(v),
            va="center",
            color=base.TEXT,
            fontsize=size,
            fontweight="bold",
        )


# ── chart kinds ─────────────────────────────────────────────────────────────────────────
def ranked_teams(
    league, group, filename, title, subtitle, series, fmt, note, lower_better=False, highlight=3
):
    """All twenty clubs as bars, best first, the top few in their own colour; a line marks the average."""
    order = series.sort_values(ascending=lower_better)
    teams, values = list(order.index)[::-1], order.to_numpy()[::-1]
    fig = frame(league, group, title, subtitle, note)
    ax = list_axes(fig)
    clean(ax)
    n = len(teams)
    ax.barh(
        range(n),
        values,
        color=[league.colour(t) if i >= n - highlight else "#3A434C" for i, t in enumerate(teams)],
        height=0.68,
    )
    row_labels(ax, league, teams, teams)
    top = float(np.nanmax(values))
    ax.set_xlim(0, top * 1.16)
    value_labels(ax, range(n), values, fmt, top * 0.012)
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
    return save(fig, league, filename)


def dumbbell(
    league,
    group,
    filename,
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
    fig = frame(league, group, title, subtitle, note)
    ax = list_axes(fig)
    clean(ax)
    ys = np.arange(len(rows))
    top = float(rows[[a_key, b_key]].to_numpy().max())
    for y, r in zip(ys, rows.itertuples()):
        a, b = getattr(r, a_key), getattr(r, b_key)
        good = (b >= a) if b_higher_is_good else (b <= a)
        colour = GOOD if good else BAD
        ax.plot([a, b], [y, y], color=colour, lw=3.2, solid_capstyle="round", zorder=2)
        ax.scatter([a], [y], s=70, color=GREY, zorder=3)
        ax.scatter([b], [y], s=150, facecolor=colour, edgecolor="white", linewidth=1.4, zorder=4)
        ax.text(
            max(a, b) + top * 0.04,
            y,
            f"{b - a:+.1f}",
            va="center",
            ha="left",
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
    legend(fig, [("●", a_label, GREY), ("○", b_label, base.TEXT)])
    return save(fig, league, filename)


def stacked(
    league,
    group,
    filename,
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
    fig = frame(league, group, title, subtitle, note)
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
    for y, total in zip(ys, order["_t"]):
        ax.text(
            total + float(order["_t"].max()) * 0.012,
            y,
            fmt.format(total),
            va="center",
            color=base.TEXT,
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_xlim(0, float(order["_t"].max()) * 1.14)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.tick_params(axis="x", labelbottom=False)
    legend(fig, [("■", label, colour) for label, colour in zip(labels, PARTS)])
    return save(fig, league, filename)


def quadrant(
    league,
    group,
    filename,
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
    """Two measures at once. Clubs carry their badge; players carry a dot and a surname."""
    fig = frame(league, group, title, subtitle, note)
    ax = fig.add_axes(list(GEO.plot_rect))
    ax.set_facecolor(base.PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=base.MUTED, labelsize=8.5, length=0)
    ax.grid(color=base.GRID, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    x, y = frame_[x_key], frame_[y_key]
    px, py = (x.max() - x.min()) * 0.10, (y.max() - y.min()) * 0.10
    ax.set_xlim(x.min() - px, x.max() + px)
    ax.set_ylim(y.min() - py, y.max() + py)
    ax.axvline(x.median(), color=base.GRID, lw=1.1, ls=(0, (4, 4)))
    ax.axhline(y.median(), color=base.GRID, lw=1.1, ls=(0, (4, 4)))
    ax.set_xlabel(x_label, color=base.MUTED, fontsize=9, fontweight="bold")
    ax.set_ylabel(y_label, color=base.MUTED, fontsize=9, fontweight="bold")
    lb._corners(ax, *corners)
    if as_teams:
        for team, xv, yv in zip(frame_.index, x, y):
            colour = league.colour(team)
            ax.scatter(xv, yv, s=520, facecolor=base.BG, edgecolor=colour, linewidth=1.8, zorder=4)
            if not lb.put_crest(ax, league, team, (xv, yv), zoom=0.145):
                ax.text(
                    xv,
                    yv,
                    team[:3].upper(),
                    color=colour,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    zorder=6,
                )
        lb.place_labels(ax, list(zip(x, y)), list(frame_.index), marker_pt=11, fontsize=7.5)
    else:
        spread = max(frame_["minutes"].max() - frame_["minutes"].min(), 1)
        sizes = 40 + 140 * (frame_["minutes"] - frame_["minutes"].min()) / spread
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
        for _, r in frame_.assign(_s=np.hypot(zx, zy)).nlargest(label_n, "_s").iterrows():
            ax.annotate(
                lb.surname(r["player"]),
                (r[x_key], r[y_key]),
                xytext=(6, 5),
                textcoords="offset points",
                color=base.TEXT,
                fontsize=8.5,
                fontweight="bold",
                zorder=6,
            )
    return save(fig, league, filename)


def butterfly(
    league,
    group,
    filename,
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
    fig = frame(league, group, title, subtitle, note)
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
    legend(fig, [("◀", left_label, "#4EA8FF"), ("▶", right_label, GOLD)])
    return save(fig, league, filename)


def lollipop_players(league, group, filename, title, subtitle, rows, key, fmt, note, n=12):
    """The leaders as a stem and a dot in their club's colour."""
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
            s=240,
            color=league.colour(r.team),
            edgecolor="white",
            linewidth=1.3,
            zorder=3,
        )
        ax.text(
            value + top * 0.045, y, fmt.format(value), va="center", color=base.TEXT, **display(17)
        )
        ax.text(0, y - 0.38, r.team, color=base.NEUTRAL, fontsize=6.8, va="center")
    row_labels(
        ax,
        league,
        [lb.surname(r.player) for r in rows.itertuples()],
        [r.team for r in rows.itertuples()],
        size=10,
        zoom=GEO.crest * 0.9,
    )
    ax.set_xlim(0, top * 1.22)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.tick_params(axis="x", labelbottom=False)
    return save(fig, league, filename)


# ── data ───────────────────────────────────────────────────────────────────────────────────
def per_match(league):
    return league.per_match.set_index("team")


def totals(league):
    return league.teams.set_index("team")


def squad_totals(league):
    """Player totals for outfield players over the minutes floor."""
    p = league.players
    return p[(p["minutes"] >= MIN_MINUTES) & ~lb.is_goalkeeper(p)].copy()


# ── club cards ──────────────────────────────────────────────────────────────────────────────
def t_attack_finishing(league):
    t = totals(league).reset_index()
    return dumbbell(
        league,
        "Attack",
        "t01_attack_finishing",
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


def t_attack_quality(league):
    return quadrant(
        league,
        "Attack",
        "t02_attack_quality",
        "Volume against quality",
        "Shots per match against the xG of each shot · top right is plenty of good chances",
        per_match(league),
        "shots",
        "xG_per_shot",
        "Shots per match  →",
        "xG per shot  →",
        "Shot quality is the model's xG divided by shots.",
        ("HIGH VOLUME, HIGH QUALITY", "PATIENT AND CLINICAL", "SHOOT ON SIGHT", "FEW AND POOR"),
    )


def t_attack_big_chances(league):
    t = per_match(league)
    return butterfly(
        league,
        "Attack",
        "t03_attack_big_chances",
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


def t_attack_sources(league):
    t = per_match(league).assign(fk_pen=lambda x: x["free_kick_xG"] + x["penalty_xG"]).reset_index()
    return stacked(
        league,
        "Attack",
        "t04_attack_sources",
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


def t_defence_xg(league):
    return ranked_teams(
        league,
        "Defence",
        "t05_defence_xg",
        "Hardest to break down",
        "Expected goals conceded per match · the shorter the bar, the better the defence",
        per_match(league)["xG_against"],
        "{:.2f}",
        "Top three in club colour. Dashed line is the league average.",
        lower_better=True,
    )


def t_defence_actions(league):
    t = per_match(league).reset_index()
    return stacked(
        league,
        "Defence",
        "t06_defence_actions",
        "What a defence is made of",
        "Tackles, interceptions, blocks and clearances per match",
        t,
        ["tackles", "interceptions", "blocks", "clearances"],
        ["Tackles", "Interceptions", "Blocks", "Clearances"],
        "Clearances count a club defending its own box, so a long bar is not always a good one.",
        row_name=lambda r: r.team,
        row_team=lambda r: r.team,
    )


def t_defence_goalkeeping(league):
    t = totals(league).reset_index()
    return dumbbell(
        league,
        "Defence",
        "t07_defence_goalkeeping",
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


def t_defence_discipline(league):
    return quadrant(
        league,
        "Defence",
        "t08_defence_discipline",
        "Fouls and bookings",
        "Fouls committed against yellow cards per match · who plays on the edge",
        per_match(league),
        "fouls_committed",
        "yellow_cards",
        "Fouls per match  →",
        "Yellow cards per match  →",
        "Red cards are rare enough that they are left out.",
        ("FOUL OFTEN, BOOKED OFTEN", "CAUTIOUS, BOOKED", "FOUL OFTEN, NOT BOOKED", "CLEAN"),
    )


def t_press_ppda(league):
    return ranked_teams(
        league,
        "Pressing",
        "t09_press_ppda",
        "Who presses hardest",
        "PPDA: opponent passes allowed per defensive action in the front 60% · lower is harder",
        per_match(league)["ppda"],
        "{:.1f}",
        "Bars run from hardest to softest. Dashed line is the league average.",
        lower_better=True,
    )


def t_press_wins(league):
    return quadrant(
        league,
        "Pressing",
        "t10_press_wins",
        "Pressing that pays",
        "Possessions won in the final third per match against how many turned into a shot",
        per_match(league),
        "high_regains",
        "regain_to_shot_rate",
        "High regains per match  →",
        "Regains that became a shot (%)  →",
        "A regain is a possession won back; high means in the attacking third.",
        ("WIN IT HIGH, USE IT", "SELECTIVE", "WIN IT HIGH, WASTE IT", "NOT PRESSING"),
    )


def t_press_counter(league):
    return ranked_teams(
        league,
        "Pressing",
        "t11_press_counter",
        "Winning it back at once",
        "Share of lost possessions won back within seconds · the counter-press",
        per_match(league)["counterpress_success_rate"],
        "{:.1f}%",
        "Top three in club colour. Dashed line is the league average.",
    )


def t_pass_progress(league):
    t = per_match(league)
    return butterfly(
        league,
        "Passing",
        "t12_pass_progress",
        "Getting the ball forward",
        "Progressive passes against line-breaking passes per match · ordered by both together",
        t["progressive_passes"],
        t["line_breaking_completed"],
        "Progressive passes",
        "Line-breaking (completed)",
        "A line-breaking pass starts behind the opponent's line and ends beyond it.",
    )


def t_pass_delivery(league):
    return quadrant(
        league,
        "Passing",
        "t13_pass_delivery",
        "Safe or direct",
        "How often passes arrive against how direct the play is · top right is accurate and vertical",
        per_match(league),
        "pass_pct",
        "directness",
        "Pass completion (%)  →",
        "Directness (%)  →",
        "Directness is the share of progress made towards goal.",
        ("ACCURATE AND DIRECT", "LONG AND RISKY", "SHORT AND SAFE", "NEITHER"),
    )


def t_pass_routes(league):
    return quadrant(
        league,
        "Passing",
        "t14_pass_routes",
        "Crosses or long balls",
        "Two ways round a block, per match · which one a side reaches for",
        per_match(league),
        "crosses",
        "long_balls",
        "Crosses per match  →",
        "Long balls per match  →",
        "A long ball is a pass of 32 metres or more.",
        ("BOTH", "LONG AND DIRECT", "WIDE DELIVERY", "SHORT GAME"),
    )


def t_pass_box(league):
    return ranked_teams(
        league,
        "Passing",
        "t15_pass_box",
        "Into the box",
        "Completed passes into the penalty area per match",
        per_match(league)["passes_into_box"],
        "{:.1f}",
        "Top three in club colour. Dashed line is the league average.",
    )


# ── player cards ────────────────────────────────────────────────────────────────────────────
def p_attack_finishers(league):
    p = league.players
    p = p[p["minutes"] >= MIN_MINUTES].sort_values("xG", ascending=False).head(12)
    return dumbbell(
        league,
        "Attack",
        "p01_attack_finishers",
        "Scoring more than expected",
        f"Goals against xG for the 12 players with the most xG · {MIN_MINUTES:.0f}+ minutes",
        p,
        "xG",
        "goals",
        "xG",
        "Goals",
        "Green: scored more than expected. Red: fewer.",
        row_label=lambda r: lb.surname(r.player),
        row_team=lambda r: r.team,
    )


def p_attack_contributions(league):
    p = squad_totals(league)
    p = (
        p.assign(_t=p["goals"] + p["assists"])
        .sort_values(["_t", "goals"], ascending=False)
        .head(12)
    )
    return stacked(
        league,
        "Attack",
        "p02_attack_contributions",
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


def p_attack_creators(league):
    return quadrant(
        league,
        "Attack",
        "p03_attack_creators",
        "Who makes the chances",
        "Key passes against passes into the box per 90 · players with 270+ minutes",
        pool(league, ["Midfielder", "Forward", "Defender"]),
        "key_passes_p90",
        "passes_into_box_p90",
        "Key passes per 90  →",
        "Passes into the box per 90  →",
        "Named: the players furthest from the middle of the pack.",
        ("CREATORS", "FINAL BALL", "SHOT PROVIDERS", ""),
        as_teams=False,
    )


def p_attack_shooters(league):
    r = pool(league, ["Forward", "Midfielder"])
    r = r[r["shots_p90"] >= 1.0]
    return quadrant(
        league,
        "Attack",
        "p04_attack_shooters",
        "Shoot often, shoot straight",
        "Shots per 90 against the share that hit the target · players averaging a shot a game or more",
        r,
        "shots_p90",
        "shot_accuracy",
        "Shots per 90  →",
        "Shots on target (%)  →",
        "Named: the players furthest from the middle of the pack.",
        ("VOLUME AND ACCURACY", "SELECTIVE", "SPRAYING IT", ""),
        as_teams=False,
    )


def p_attack_dribblers(league):
    return lollipop_players(
        league,
        "Attack",
        "p05_attack_dribblers",
        "The dribblers",
        "Take-ons won per 90 · players with 270+ minutes",
        pool(league, ["Midfielder", "Forward", "Defender"]),
        "takeons_won_p90",
        "{:.1f}",
        "A take-on is an attempt to beat an opponent on the dribble.",
    )


def p_defence_ballwinners(league):
    r = pool(league, ["Defender", "Midfielder"])
    r = r.assign(_t=r[["tackles_won_p90", "interceptions_p90"]].sum(axis=1)).nlargest(12, "_t")
    return stacked(
        league,
        "Defence",
        "p06_defence_ballwinners",
        "Who stops the attack",
        "Tackles won plus interceptions per 90 · the ball won from the opponent, not cleared",
        r,
        ["tackles_won_p90", "interceptions_p90"],
        ["Tackles won", "Interceptions"],
        f"Defenders and midfielders with {MIN_MINUTES:.0f}+ minutes.",
        row_name=lambda r: lb.surname(r.player),
        row_team=lambda r: r.team,
    )


def p_defence_height(league):
    r = pool(league, ["Defender", "Midfielder"])
    r = r[r["defensive_height"].notna()]
    return quadrant(
        league,
        "Defence",
        "p07_defence_height",
        "Where they defend",
        "Defensive actions per 90 against how far up the pitch they happen",
        r,
        "defensive_height",
        "defensive_actions_p90",
        "Average height of defensive actions  → further upfield",
        "Defensive actions per 90  →",
        "Named: the players furthest from the middle of the pack.",
        ("HIGH AND BUSY", "DEEP AND BUSY", "HIGH AND QUIET", "DEEP AND QUIET"),
        as_teams=False,
    )


def p_defence_air(league):
    p = squad_totals(league)
    p = (
        p.assign(aerials_lost=p["aerials"] - p["aerials_won"])
        .sort_values("aerials_won", ascending=False)
        .head(12)
    )
    return stacked(
        league,
        "Defence",
        "p08_defence_air",
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


def p_defence_tackling(league):
    r = pool(league, ["Defender", "Midfielder"])
    r = r[r["tackles_p90"] >= 1.0]
    return quadrant(
        league,
        "Defence",
        "p09_defence_tackling",
        "Tackling: how many, how well",
        "Tackles per 90 against the share won · players making a tackle a game or more",
        r,
        "tackles_p90",
        "tackle_pct",
        "Tackles per 90  →",
        "Tackles won (%)  →",
        "Named: the players furthest from the middle of the pack.",
        ("BUSY AND WINNING", "SELECTIVE", "BUSY AND LOSING", ""),
        as_teams=False,
    )


def p_press_recoveries(league):
    return lollipop_players(
        league,
        "Pressing",
        "p10_press_recoveries",
        "The ball winners",
        "Recoveries per 90 · players with 270+ minutes",
        pool(league, ["Midfielder", "Forward", "Defender"]),
        "recoveries_p90",
        "{:.1f}",
        "A recovery is a loose or lost ball won back.",
    )


def p_press_forwards(league):
    return lollipop_players(
        league,
        "Pressing",
        "p11_press_forwards",
        "Forwards who work",
        "Defensive actions per 90 by forwards · who starts the press",
        pool(league, ["Forward"]),
        "defensive_actions_p90",
        "{:.1f}",
        f"Forwards with {MIN_MINUTES:.0f}+ minutes.",
        n=10,
    )


def p_pass_linebreakers(league):
    return lollipop_players(
        league,
        "Passing",
        "p12_pass_linebreakers",
        "Breaking the lines",
        "Line-breaking passes per 90 · passes that start behind the opponent's line and end beyond it",
        pool(league, ["Defender", "Midfielder", "Forward"]),
        "line_breaking_passes_p90",
        "{:.1f}",
        f"Players with {MIN_MINUTES:.0f}+ minutes.",
    )


def p_pass_distance(league):
    return lollipop_players(
        league,
        "Passing",
        "p13_pass_distance",
        "Ground gained",
        "Metres moved towards goal per 90 by completed passes and carries",
        pool(league, ["Defender", "Midfielder", "Forward"]),
        "progression_metres_p90",
        "{:.0f}",
        f"Players with {MIN_MINUTES:.0f}+ minutes.",
    )


def p_pass_final_third(league):
    return lollipop_players(
        league,
        "Passing",
        "p14_pass_final_third",
        "Into the final third",
        "Completed passes into the attacking third per 90",
        pool(league, ["Defender", "Midfielder", "Forward"]),
        "passes_into_final_third_p90",
        "{:.1f}",
        f"Players with {MIN_MINUTES:.0f}+ minutes.",
    )


def p_pass_assists(league):
    return lollipop_players(
        league,
        "Passing",
        "p15_pass_assists",
        "Who sets up the goals",
        "Assists per 90 · the final pass, as the feed credits it",
        pool(league, ["Defender", "Midfielder", "Forward"]),
        "assists_p90",
        "{:.2f}",
        f"Players with {MIN_MINUTES:.0f}+ minutes.",
    )


def p_pass_style(league):
    r = pool(league, ["Defender", "Midfielder"])
    r = r[r["passes_p90"] >= r["passes_p90"].quantile(0.25)]
    return quadrant(
        league,
        "Passing",
        "p16_pass_style",
        "Safe or ambitious",
        "Pass completion against the share of completed passes that were progressive",
        r,
        "pass_pct",
        "progressive_pass_pct",
        "Pass completion (%)  →",
        "Progressive share of completed passes (%)  →",
        "Defenders and midfielders in the top three quarters for passes per 90.",
        ("SAFE AND FORWARD", "AMBITIOUS", "SAFE AND SIDEWAYS", "RECYCLERS"),
        as_teams=False,
    )


CARDS = [
    Card("t01", "Attack", t_attack_finishing),
    Card("t02", "Attack", t_attack_quality),
    Card("t03", "Attack", t_attack_big_chances),
    Card("t04", "Attack", t_attack_sources),
    Card("t05", "Defence", t_defence_xg),
    Card("t06", "Defence", t_defence_actions),
    Card("t07", "Defence", t_defence_goalkeeping),
    Card("t08", "Defence", t_defence_discipline),
    Card("t09", "Pressing", t_press_ppda),
    Card("t10", "Pressing", t_press_wins),
    Card("t11", "Pressing", t_press_counter),
    Card("t12", "Passing", t_pass_progress),
    Card("t13", "Passing", t_pass_delivery),
    Card("t14", "Passing", t_pass_routes),
    Card("t15", "Passing", t_pass_box),
    Card("p01", "Attack", p_attack_finishers),
    Card("p02", "Attack", p_attack_contributions),
    Card("p03", "Attack", p_attack_creators),
    Card("p04", "Attack", p_attack_shooters),
    Card("p05", "Attack", p_attack_dribblers),
    Card("p06", "Defence", p_defence_ballwinners),
    Card("p07", "Defence", p_defence_height),
    Card("p08", "Defence", p_defence_air),
    Card("p09", "Defence", p_defence_tackling),
    Card("p10", "Pressing", p_press_recoveries),
    Card("p11", "Pressing", p_press_forwards),
    Card("p12", "Passing", p_pass_linebreakers),
    Card("p13", "Passing", p_pass_distance),
    Card("p14", "Passing", p_pass_final_third),
    Card("p15", "Passing", p_pass_assists),
    Card("p16", "Passing", p_pass_style),
]

FORMATS = {"4x5": FOUR_BY_FIVE, "16x9": SIXTEEN_BY_NINE}


def build_all(
    league: lb.League, only: list[str] | None = None, formats=("4x5", "16x9")
) -> list[Path]:
    """Draw the chosen cards in each frame; a card's file is its id and name, e.g. ``t05_defence_xg.png``."""
    global GEO
    base.theme()
    paths = []
    try:
        for name in formats:
            GEO = FORMATS[name]
            for card in CARDS:
                if only is None or card.name in only:
                    paths.append(card.draw(league))
    finally:
        GEO = FOUR_BY_FIVE
    return paths
