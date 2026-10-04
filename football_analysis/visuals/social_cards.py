"""Single-message graphics for a block of rounds, sized for a timeline.

Each card makes one point about attack, defence, pressing or passing, for clubs or for
players, and each uses the chart that suits that point (a dumbbell for results against
expectation, stacked bars for what a defence is made of, a scatter for two measures at
once) rather than one template. All are 4:5 (1200 x 1500), the tallest frame a timeline shows
without cropping, and read from the aggregate tables ``scripts/aggregate_rounds.py`` writes.
"""

from __future__ import annotations

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

SIZE = (8.0, 10.0)
DPI = 150  # 1200 x 1500
GOOD, BAD, GOLD = "#3DDC84", "#FF6B5B", "#F2B134"
PARTS = ["#4EA8FF", "#F2B134", "#E8452C", "#A77BFF"]
MIN_MINUTES = 270.0


@dataclass
class Card:
    """What a card needs to be drawn and filed."""

    name: str
    group: str
    draw: Callable[[lb.League], Path]


# ── furniture ────────────────────────────────────────────────────────────────────────
def frame(league: lb.League, group: str, title: str, subtitle: str, note: str = ""):
    """The shared 4:5 page: kicker, big condensed title, one line of explanation, footer."""
    fig = plt.figure(figsize=SIZE, facecolor=base.BG)
    fig.text(
        0.06,
        0.957,
        f"{league.title}  ·  {league.rounds}  ·  {group.upper()}",
        color=lb.ACCENT,
        fontsize=9.5,
        fontweight="bold",
    )
    size = 54 if len(title) <= 18 else (46 if len(title) <= 24 else 38)
    fig.text(0.06, 0.905, title.upper(), color=base.TEXT, va="center", **display(size))
    fig.text(0.06, 0.857, subtitle, color=base.MUTED, fontsize=10.5, va="center", wrap=True)
    fig.add_artist(
        Rectangle((0.06, 0.835), 0.88, 0.003, transform=fig.transFigure, color=lb.ACCENT, lw=0)
    )
    fig.text(0.06, 0.028, note, color=base.NEUTRAL, fontsize=7.5, va="center")
    fig.text(0.94, 0.028, "MOSTAFA SAAD", color=base.MUTED, ha="right", va="center", **display(14))
    return fig


def save(fig, league: lb.League, filename: str) -> Path:
    folder = league.folder / "twitter"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
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


def crest_at(ax, league, team, y, zoom=0.11):
    """A badge just left of the axes, at row ``y``."""
    lb.put_crest(ax, league, team, (-0.045, y), zoom=zoom, transform=ax.get_yaxis_transform())


def team_rows(ax, league, teams):
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(teams, fontsize=9.5, color=base.TEXT)
    ax.tick_params(axis="y", pad=30)
    for i, team in enumerate(teams):
        crest_at(ax, league, team, i)


def player_rows(ax, league, rows, label_size=10):
    """Surname with the club underneath, for a row of players."""
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [lb.surname(r.player) for r in rows.itertuples()], fontsize=label_size, color=base.TEXT
    )
    ax.tick_params(axis="y", pad=34)
    for i, r in enumerate(rows.itertuples()):
        crest_at(ax, league, r.team, i, zoom=0.10)


def pool(league: lb.League, roles=None) -> pd.DataFrame:
    frame_ = lb._qualified(league, MIN_MINUTES)
    return frame_ if roles is None else frame_[frame_["role"].isin(roles)]


def value_labels(ax, ys, values, fmt, pad, colour=None, size=9.5):
    for y, v in zip(ys, values):
        ax.text(
            v + pad,
            y,
            fmt.format(v),
            va="center",
            color=colour or base.TEXT,
            fontsize=size,
            fontweight="bold",
        )


def short(frame_, key, n, ascending=False):
    return frame_.sort_values(key, ascending=ascending).head(n)


# ── charts ─────────────────────────────────────────────────────────────────────────────
def ranked_teams(
    league,
    group,
    filename,
    title,
    subtitle,
    series: pd.Series,
    fmt,
    note,
    lower_better=False,
    highlight=3,
    ref=None,
    ref_label="LEAGUE AVERAGE",
):
    """All twenty clubs as bars, best first, the top few in their own colour.

    The rest are grey so the eye lands on the leaders; a line marks the league average.
    """
    order = series.sort_values(ascending=lower_better)
    teams = list(order.index)[::-1]
    values = order.to_numpy()[::-1]
    fig = frame(league, group, title, subtitle, note)
    ax = fig.add_axes([0.255, 0.065, 0.66, 0.74])
    clean(ax)
    n = len(teams)
    colours = [league.colour(t) if i >= n - highlight else "#3A434C" for i, t in enumerate(teams)]
    ax.barh(range(n), values, color=colours, height=0.68)
    team_rows(ax, league, teams)
    top = float(np.nanmax(values))
    ax.set_xlim(0, top * 1.16)
    value_labels(ax, range(n), values, fmt, top * 0.012)
    mean = float(series.mean()) if ref is None else ref
    ax.axvline(mean, color="#8E99A4", lw=1.1, ls=(0, (4, 4)), zorder=1)
    ax.text(
        mean,
        n - 0.15,
        ref_label,
        color="#8E99A4",
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
    row_team=None,
    n=None,
    fmt="{:.1f}",
    sort_key=None,
):
    """Two values per row joined by a line: a dot for expectation, a ring for the result."""
    rows = frame_.copy()
    rows["_gap"] = rows[b_key] - rows[a_key]
    rows = rows.sort_values(sort_key or "_gap", ascending=True)
    if n:
        rows = rows.tail(n)
    fig = frame(league, group, title, subtitle, note)
    ax = fig.add_axes([0.255, 0.075, 0.66, 0.72])
    clean(ax)
    ys = np.arange(len(rows))
    for y, r in zip(ys, rows.itertuples()):
        a, b = getattr(r, a_key), getattr(r, b_key)
        colour = GOOD if b >= a else BAD
        ax.plot([a, b], [y, y], color=colour, lw=3.2, solid_capstyle="round", zorder=2)
        ax.scatter([a], [y], s=70, color="#8E99A4", zorder=3)
        ax.scatter([b], [y], s=150, facecolor=colour, edgecolor="white", linewidth=1.4, zorder=4)
        ax.text(
            b + (0.05 * (rows[[a_key, b_key]].to_numpy().max())) * (1 if b >= a else -1),
            y,
            f"{b - a:+.1f}",
            va="center",
            ha="left" if b >= a else "right",
            color=colour,
            fontsize=9.5,
            fontweight="bold",
        )
    labels = [row_label(r) for r in rows.itertuples()]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9.5, color=base.TEXT)
    ax.tick_params(axis="y", pad=30)
    for y, r in zip(ys, rows.itertuples()):
        crest_at(ax, league, row_team(r), y)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    top = float(rows[[a_key, b_key]].to_numpy().max())
    low = float(rows[[a_key, b_key]].to_numpy().min())
    ax.set_xlim(min(0, low) - 0.12 * top, top * 1.18)
    fig.text(0.255, 0.815, f"● {a_label}", color="#8E99A4", fontsize=9, fontweight="bold")
    fig.text(0.44, 0.815, f"○ {b_label}", color=base.TEXT, fontsize=9, fontweight="bold")
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
    """What each row is made of: one bar split into its parts, longest total first."""
    totals = rows[keys].sum(axis=1)
    order = rows.assign(_t=totals).sort_values("_t", ascending=True)
    fig = frame(league, group, title, subtitle, note)
    ax = fig.add_axes([0.255, 0.075, 0.66, 0.70])
    clean(ax)
    ys = np.arange(len(order))
    left = np.zeros(len(order))
    for key, label, colour in zip(keys, labels, PARTS):
        values = order[key].to_numpy()
        ax.barh(ys, values, left=left, color=colour, height=0.68, edgecolor=base.BG, linewidth=0.8)
        left += values
    ax.set_yticks(ys)
    ax.set_yticklabels([row_name(r) for r in order.itertuples()], fontsize=9.5, color=base.TEXT)
    ax.tick_params(axis="y", pad=30)
    for y, r in zip(ys, order.itertuples()):
        crest_at(ax, league, row_team(r), y)
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
    x = 0.255
    for label, colour in zip(labels, PARTS):
        fig.add_artist(
            Rectangle((x, 0.812), 0.014, 0.012, transform=fig.transFigure, color=colour, lw=0)
        )
        fig.text(
            x + 0.02, 0.818, label, color=base.MUTED, fontsize=8.5, fontweight="bold", va="center"
        )
        x += 0.03 + 0.012 * len(label)
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
    invert_x=False,
    invert_y=False,
    as_teams=True,
    label_n=14,
    size_key=None,
):
    """Two measures at once. Clubs carry their badge; players carry a dot and a surname."""
    fig = frame(league, group, title, subtitle, note)
    ax = fig.add_axes([0.11, 0.10, 0.83, 0.70])
    ax.set_facecolor(base.PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=base.MUTED, labelsize=8.5, length=0)
    ax.grid(color=base.GRID, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    x, y = frame_[x_key], frame_[y_key]
    px, py = (x.max() - x.min()) * 0.10, (y.max() - y.min()) * 0.10
    ax.set_xlim((x.max() + px, x.min() - px) if invert_x else (x.min() - px, x.max() + px))
    ax.set_ylim((y.max() + py, y.min() - py) if invert_y else (y.min() - py, y.max() + py))
    ax.axvline(x.median(), color=base.GRID, lw=1.1, ls=(0, (4, 4)))
    ax.axhline(y.median(), color=base.GRID, lw=1.1, ls=(0, (4, 4)))
    ax.set_xlabel(x_label, color=base.MUTED, fontsize=9.5, fontweight="bold")
    ax.set_ylabel(y_label, color=base.MUTED, fontsize=9.5, fontweight="bold")
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
        sizes = 40 + 140 * (frame_["minutes"] - frame_["minutes"].min()) / max(
            frame_["minutes"].max() - frame_["minutes"].min(), 1
        )
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
    series_left,
    series_right,
    left_label,
    right_label,
    note,
    fmt="{:.0f}",
):
    """Two measures back to back, clubs ordered by the sum of their ranks."""
    both = pd.DataFrame({"l": series_left, "r": series_right}).dropna()
    both["_rank"] = both["l"].rank(ascending=False) + both["r"].rank(ascending=False)
    order = both.sort_values("_rank", ascending=False)
    fig = frame(league, group, title, subtitle, note)
    ax = fig.add_axes([0.255, 0.075, 0.66, 0.70])
    clean(ax, grid=None)
    ys = np.arange(len(order))
    lmax, rmax = float(order["l"].max()), float(order["r"].max())
    ax.barh(ys, -order["l"] / lmax, color="#4EA8FF", height=0.68)
    ax.barh(ys, order["r"] / rmax, color=GOLD, height=0.68)
    for y, (team, row) in zip(ys, order.iterrows()):
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
    team_rows(ax, league, list(order.index))
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xticks([])
    fig.text(0.255, 0.812, f"◀ {left_label}", color="#4EA8FF", fontsize=9.5, fontweight="bold")
    fig.text(
        0.915, 0.812, f"{right_label} ▶", color=GOLD, fontsize=9.5, fontweight="bold", ha="right"
    )
    return save(fig, league, filename)


def lollipop_players(
    league,
    group,
    filename,
    title,
    subtitle,
    rows,
    key,
    fmt,
    note,
    tint_key=None,
    tint_label="",
    n=12,
):
    """The leaders as a stem and a dot in their club's colour, with a second figure as the dot's shade."""
    rows = rows.sort_values(key, ascending=False).head(n).iloc[::-1]
    fig = frame(league, group, title, subtitle, note)
    ax = fig.add_axes([0.255, 0.075, 0.62, 0.72])
    clean(ax)
    ys = np.arange(len(rows))
    top = float(rows[key].max())
    for y, r in zip(ys, rows.itertuples()):
        ax.plot([0, getattr(r, key)], [y, y], color=base.GRID, lw=3.0, zorder=1)
        ax.scatter(
            [getattr(r, key)],
            [y],
            s=240,
            color=league.colour(r.team),
            edgecolor="white",
            linewidth=1.3,
            zorder=3,
        )
        ax.text(
            getattr(r, key) + top * 0.045,
            y,
            fmt.format(getattr(r, key)),
            va="center",
            color=base.TEXT,
            **display(17),
        )
        ax.text(0, y - 0.38, r.team, color=base.NEUTRAL, fontsize=6.8, va="center")
    player_rows(ax, league, rows)
    ax.set_xlim(0, top * 1.22)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.tick_params(axis="x", labelbottom=False)
    return save(fig, league, filename)


# ── the cards ────────────────────────────────────────────────────────────────────────────
def per_match(league):
    return league.per_match.set_index("team")


def totals(league):
    return league.teams.set_index("team")


def t_attack_finishing(league):
    t = totals(league).reset_index()
    return dumbbell(
        league,
        "Attack",
        "t1_attack_finishing.png",
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
    t = per_match(league)
    return quadrant(
        league,
        "Attack",
        "t2_attack_quality.png",
        "Volume against quality",
        "Shots per match against the xG of each shot · top right is plenty of good chances",
        t,
        "shots",
        "xG_per_shot",
        "Shots per match  →",
        "xG per shot  →",
        "Shot quality is the model's xG divided by shots.",
        corners=(
            "HIGH VOLUME, HIGH QUALITY",
            "PATIENT AND CLINICAL",
            "SHOOT ON SIGHT",
            "FEW AND POOR",
        ),
    )


def t_defence_xg(league):
    t = per_match(league)
    return ranked_teams(
        league,
        "Defence",
        "t3_defence_xg.png",
        "Hardest to break down",
        "Expected goals conceded per match · the shorter the bar, the better the defence",
        t["xG_against"],
        "{:.2f}",
        "Top three in club colour. Dashed line is the league average.",
        lower_better=True,
        highlight=3,
    )


def t_defence_actions(league):
    t = per_match(league).reset_index()
    return stacked(
        league,
        "Defence",
        "t4_defence_actions.png",
        "What a defence is made of",
        "Tackles, interceptions, blocks and clearances per match",
        t,
        ["tackles", "interceptions", "blocks", "clearances"],
        ["Tackles", "Interceptions", "Blocks", "Clearances"],
        "Clearances count a club defending its own box, so a long bar is not always a good one.",
        row_name=lambda r: r.team,
        row_team=lambda r: r.team,
    )


def t_press_ppda(league):
    t = per_match(league)
    return ranked_teams(
        league,
        "Pressing",
        "t5_press_ppda.png",
        "Who presses hardest",
        "PPDA: opponent passes allowed per defensive action in the front 60% · lower is harder",
        t["ppda"],
        "{:.1f}",
        "Bars run from hardest to softest. Dashed line is the league average.",
        lower_better=True,
        highlight=3,
    )


def t_press_wins(league):
    t = per_match(league)
    return quadrant(
        league,
        "Pressing",
        "t6_press_wins.png",
        "Pressing that pays",
        "Possessions won in the final third per match against how many turned into a shot",
        t,
        "high_regains",
        "regain_to_shot_rate",
        "High regains per match  →",
        "Regains that became a shot (%)  →",
        "A regain is a possession won back; high means in the attacking third.",
        corners=("WIN IT HIGH, USE IT", "SELECTIVE", "WIN IT HIGH, WASTE IT", "NOT PRESSING"),
    )


def t_pass_progress(league):
    t = per_match(league)
    return butterfly(
        league,
        "Passing",
        "t7_pass_progress.png",
        "Getting the ball forward",
        "Progressive passes against line-breaking passes per match · ordered by both together",
        t["progressive_passes"],
        t["line_breaking_completed"],
        "Progressive passes",
        "Line-breaking passes (completed)",
        "A line-breaking pass starts behind the opponent's line and ends beyond it.",
    )


def t_pass_delivery(league):
    t = per_match(league)
    return quadrant(
        league,
        "Passing",
        "t8_pass_delivery.png",
        "Safe or direct",
        "How often passes arrive against how direct the play is · top right is accurate and vertical",
        t,
        "pass_pct",
        "directness",
        "Pass completion (%)  →",
        "Directness (%)  →",
        "Directness is the share of progress made towards goal.",
        corners=("ACCURATE AND DIRECT", "LONG AND RISKY", "SHORT AND SAFE", "NEITHER"),
    )


def p_attack_finishers(league):
    p = league.players
    p = p[p["minutes"] >= MIN_MINUTES].copy()
    p = p.sort_values("xG", ascending=False).head(12)
    return dumbbell(
        league,
        "Attack",
        "p1_attack_finishers.png",
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


def p_attack_creators(league):
    r = pool(league, roles=["Midfielder", "Forward", "Defender"])
    return quadrant(
        league,
        "Attack",
        "p2_attack_creators.png",
        "Who makes the chances",
        "Key passes against passes into the box per 90 · players with 270+ minutes",
        r,
        "key_passes_p90",
        "passes_into_box_p90",
        "Key passes per 90  →",
        "Passes into the box per 90  →",
        "Named: the players furthest from the middle of the pack.",
        corners=("CREATORS", "FINAL BALL", "SHOT PROVIDERS", ""),
        as_teams=False,
    )


def p_defence_ballwinners(league):
    r = pool(league, roles=["Defender", "Midfielder"])
    r = r.assign(total=r[["tackles_won_p90", "interceptions_p90"]].sum(axis=1)).nlargest(
        12, "total"
    )
    rows = r.assign(name=r["player"].map(lb.surname))
    return stacked(
        league,
        "Defence",
        "p3_defence_ballwinners.png",
        "Who stops the attack",
        "Tackles won plus interceptions per 90 · the ball won from the opponent, not cleared",
        rows,
        ["tackles_won_p90", "interceptions_p90"],
        ["Tackles won", "Interceptions"],
        f"Defenders and midfielders with {MIN_MINUTES:.0f}+ minutes.",
        row_name=lambda x: x.name,
        row_team=lambda x: x.team,
    )


def p_defence_height(league):
    r = pool(league, roles=["Defender", "Midfielder"])
    r = r[r["defensive_height"].notna()]
    return quadrant(
        league,
        "Defence",
        "p4_defence_height.png",
        "Where they defend",
        "Defensive actions per 90 against the average height of those actions (metres up the pitch)",
        r,
        "defensive_height",
        "defensive_actions_p90",
        "Average height of defensive actions  → further upfield",
        "Defensive actions per 90  →",
        "Named: the players furthest from the middle of the pack.",
        corners=("HIGH AND BUSY", "DEEP AND BUSY", "HIGH AND QUIET", "DEEP AND QUIET"),
        as_teams=False,
    )


def p_press_recoveries(league):
    r = pool(league, roles=["Midfielder", "Forward", "Defender"])
    return lollipop_players(
        league,
        "Pressing",
        "p5_press_recoveries.png",
        "The ball winners",
        "Recoveries per 90 · players with 270+ minutes",
        r,
        "recoveries_p90",
        "{:.1f}",
        "A recovery is a loose or lost ball won back.",
    )


def p_press_forwards(league):
    r = pool(league, roles=["Forward"])
    return lollipop_players(
        league,
        "Pressing",
        "p6_press_forwards.png",
        "Forwards who work",
        "Defensive actions per 90 by forwards · who starts the press",
        r,
        "defensive_actions_p90",
        "{:.1f}",
        f"Forwards with {MIN_MINUTES:.0f}+ minutes.",
        n=10,
    )


def p_pass_progressors(league):
    r = pool(league, roles=["Defender", "Midfielder", "Forward"])
    return lollipop_players(
        league,
        "Passing",
        "p7_pass_linebreakers.png",
        "Breaking the lines",
        "Line-breaking passes per 90 · passes that start behind the opponent's line and end beyond it",
        r,
        "line_breaking_passes_p90",
        "{:.1f}",
        f"Players with {MIN_MINUTES:.0f}+ minutes.",
    )


def p_pass_style(league):
    r = pool(league, roles=["Defender", "Midfielder"])
    r = r[r["passes_p90"] >= r["passes_p90"].quantile(0.25)]
    return quadrant(
        league,
        "Passing",
        "p8_pass_style.png",
        "Safe or ambitious",
        "Pass completion against the share of completed passes that were progressive",
        r,
        "pass_pct",
        "progressive_pass_pct",
        "Pass completion (%)  →",
        "Progressive share of completed passes (%)  →",
        "Defenders and midfielders in the top three quarters for passes per 90.",
        corners=("SAFE AND FORWARD", "AMBITIOUS", "SAFE AND SIDEWAYS", "RECYCLERS"),
        as_teams=False,
    )


CARDS = [
    Card("t1", "Attack", t_attack_finishing),
    Card("t2", "Attack", t_attack_quality),
    Card("t3", "Defence", t_defence_xg),
    Card("t4", "Defence", t_defence_actions),
    Card("t5", "Pressing", t_press_ppda),
    Card("t6", "Pressing", t_press_wins),
    Card("t7", "Passing", t_pass_progress),
    Card("t8", "Passing", t_pass_delivery),
    Card("p1", "Attack", p_attack_finishers),
    Card("p2", "Attack", p_attack_creators),
    Card("p3", "Defence", p_defence_ballwinners),
    Card("p4", "Defence", p_defence_height),
    Card("p5", "Pressing", p_press_recoveries),
    Card("p6", "Pressing", p_press_forwards),
    Card("p7", "Passing", p_pass_progressors),
    Card("p8", "Passing", p_pass_style),
]


def build_all(league: lb.League, only: list[str] | None = None) -> list[Path]:
    base.theme()
    return [card.draw(league) for card in CARDS if only is None or card.name in only]
