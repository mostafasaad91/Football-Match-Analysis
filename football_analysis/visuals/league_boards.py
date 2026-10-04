"""Boards for a block of rounds: every team and the players who stood out.

The match boards answer "what happened on Saturday". These answer "what does the
season so far look like" from the tables ``scripts/aggregate_rounds.py`` writes, so
they read CSVs and know nothing about events. They share the match boards' tokens
(page, panel and grid colours, the bundled condensed face) and use each club's own
colour and crest where there is one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, Rectangle

from football_analysis.visuals import crests
from football_analysis.visuals import visual_redesign_preview as base
from football_analysis.visuals.typography import display

matplotlib.use("Agg")

DPI = 170
ACCENT = "#2FD3BE"


@dataclass
class League:
    """The aggregate tables of one league block, with each club's id and colour."""

    folder: Path
    title: str
    rounds: str
    teams: pd.DataFrame
    per_match: pd.DataFrame
    players: pd.DataFrame
    per90: pd.DataFrame
    colours: dict[str, str] = field(default_factory=dict)
    ids: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, folder: Path | str, packages: Path | str | None = None) -> "League":
        folder = Path(folder)

        def read(name):
            return pd.read_csv(folder / f"{name}.csv", encoding="utf-8-sig")

        label = re.search(r"R(\d+-\d+|\d+)$", folder.name)
        colours, ids = {}, {}
        if packages is not None:
            for info_file in Path(packages).glob("Matchweek_*/*/match_info.json"):
                info = json.loads(info_file.read_text(encoding="utf-8-sig"))
                for side in ("home", "away"):
                    name = info.get(f"{side}_name")
                    if name and f"{side}_color" in info:
                        colours.setdefault(name, info[f"{side}_color"])
                        ids.setdefault(name, info[f"{side}_id"])
        return cls(
            folder=folder,
            title="PREMIER LEAGUE"
            if "Premier" in folder.name
            else folder.name.split("_20")[0].replace("_", " ").upper(),
            rounds=f"ROUNDS {label.group(1)}" if label else "",
            teams=read("teams_totals"),
            per_match=read("teams_per_match"),
            players=read("players_totals"),
            per90=read("players_per90"),
            colours=colours,
            ids=ids,
        )

    def colour(self, team: str) -> str:
        return readable(self.colours.get(team, "#9AA4AB"))


def readable(colour: str, floor: float = 0.30) -> str:
    """Lift a club colour until it can be read on the page (navy and maroon vanish on black)."""
    rgb = np.array(mcolors.to_rgb(colour))
    for _ in range(12):
        luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        if luminance >= floor:
            break
        rgb = rgb + (1.0 - rgb) * 0.18
    return mcolors.to_hex(rgb)


def is_goalkeeper(frame: pd.DataFrame) -> pd.Series:
    """Rows whose role is goalkeeper (the aggregate writes the role as a word)."""
    return frame["role"].astype(str).str.casefold().isin({"goalkeeper", "gk"})


def surname(name: str) -> str:
    parts = str(name).split()
    return parts[-1] if parts else str(name)


# ── page furniture ──────────────────────────────────────────────────────────────
def page(league: League, title: str, subtitle: str, size=(14, 9)):
    fig = plt.figure(figsize=size, facecolor=base.BG)
    fig.text(
        0.05,
        0.955,
        f"{league.title}  ·  {league.rounds}",
        color=base.MUTED,
        fontsize=10,
        fontweight="bold",
    )
    fig.text(0.05, 0.898, title.upper(), color=base.TEXT, va="center", **display(38))
    fig.text(0.05, 0.855, subtitle, color=base.MUTED, fontsize=11)
    fig.add_artist(
        Rectangle((0.05, 0.835), 0.9, 0.003, transform=fig.transFigure, color=ACCENT, lw=0)
    )
    fig.text(0.95, 0.022, "MOSTAFA SAAD", color=base.MUTED, ha="right", **display(13))
    return fig


def finish(fig, league: League, name: str, note: str = "") -> Path:
    if note:
        fig.text(0.05, 0.022, note, color=base.NEUTRAL, fontsize=8)
    out = league.folder / "visuals"
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=DPI, facecolor=base.BG)
    plt.close(fig)
    return path


def put_crest(ax, league: League, team: str, xy, zoom=0.24, transform=None):
    """A club badge at ``xy``; returns False when there is none to draw."""
    team_id = league.ids.get(team)
    image = crests.crest_image(team_id, allow_download=False) if team_id is not None else None
    if image is None:
        return False
    box = AnnotationBbox(
        OffsetImage(image, zoom=zoom),
        xy,
        frameon=False,
        xycoords=transform or "data",
        box_alignment=(0.5, 0.5),
        zorder=6,
    )
    ax.add_artist(box)
    return True


def place_labels(ax, points, names, marker_pt=15.0, fontsize=8):
    """Name each point beside its badge, on the side where nothing else is.

    Tries below, above, right and left in turn and keeps the first position whose text
    does not touch another label or any badge. Falls back to below, which is where a
    reader looks first. Positions are measured in the figure's own pixels.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = fig.dpi / 72.0
    pixels = ax.transData.transform(np.asarray(points, dtype=float))
    radius = marker_pt * scale
    badges = [(x - radius, y - radius, x + radius, y + radius) for x, y in pixels]
    taken = []

    def overlaps(box, others):
        return any(
            box[0] < o[2] and box[2] > o[0] and box[1] < o[3] and box[3] > o[1] for o in others
        )

    for index, ((x, y), name) in enumerate(zip(pixels, names)):
        probe = ax.text(0, 0, name, fontsize=fontsize, fontweight="bold")
        width = probe.get_window_extent(renderer).width
        height = probe.get_window_extent(renderer).height
        probe.remove()
        gap = 4 * scale
        options = [
            ("center", "top", x, y - radius - gap),
            ("center", "bottom", x, y + radius + gap),
            ("left", "center", x + radius + gap, y),
            ("right", "center", x - radius - gap, y),
        ]
        chosen = options[0]
        for option in options:
            ha, va, px, py = option
            left = px - width / 2 if ha == "center" else (px if ha == "left" else px - width)
            bottom = py - height if va == "top" else (py if va == "bottom" else py - height / 2)
            box = (left, bottom, left + width, bottom + height)
            others = [b for i, b in enumerate(badges) if i != index] + taken
            if not overlaps(box, others):
                chosen = option
                taken.append(box)
                break
        else:
            ha, va, px, py = chosen
            left = px - width / 2
            taken.append((left, py - height, left + width, py))
        ha, va, px, py = chosen
        data = ax.transData.inverted().transform((px, py))
        ax.text(
            data[0],
            data[1],
            name,
            ha=ha,
            va=va,
            color=base.TEXT,
            fontsize=fontsize,
            fontweight="bold",
            zorder=7,
        )


# ── 1. what each side created against what it allowed ───────────────────────────────
def xg_quadrant(league: League) -> Path:
    t = league.per_match.set_index("team")
    fig = page(
        league,
        "Chances for and against",
        "Expected goals created and conceded per match · top right is the best of both",
    )
    ax = fig.add_axes([0.08, 0.10, 0.86, 0.69])
    ax.set_facecolor(base.PANEL)
    x, y = t["xG"], t["xG_against"]
    pad_x, pad_y = (x.max() - x.min()) * 0.12, (y.max() - y.min()) * 0.12
    ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax.set_ylim(y.max() + pad_y, y.min() - pad_y)  # up is fewer chances conceded
    ax.axvline(x.median(), color=base.GRID, lw=1.1, ls=(0, (4, 4)))
    ax.axhline(y.median(), color=base.GRID, lw=1.1, ls=(0, (4, 4)))
    ax.grid(color=base.GRID, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=base.MUTED, labelsize=9, length=0)
    ax.set_xlabel("xG created per match  →", color=base.MUTED, fontsize=10, fontweight="bold")
    ax.set_ylabel(
        "← more   xG conceded per match   fewer →", color=base.MUTED, fontsize=10, fontweight="bold"
    )
    corner = dict(color=base.NEUTRAL, fontsize=11, fontweight="bold", alpha=0.85)
    ax.text(0.985, 0.975, "DOMINANT", ha="right", va="top", transform=ax.transAxes, **corner)
    ax.text(0.015, 0.975, "SOLID BUT BLUNT", ha="left", va="top", transform=ax.transAxes, **corner)
    ax.text(0.985, 0.025, "OPEN GAMES", ha="right", va="bottom", transform=ax.transAxes, **corner)
    ax.text(0.015, 0.025, "STRUGGLING", ha="left", va="bottom", transform=ax.transAxes, **corner)
    for team, row in t.iterrows():
        colour = league.colour(team)
        ax.scatter(
            row["xG"],
            row["xG_against"],
            s=900,
            facecolor=base.BG,
            edgecolor=colour,
            linewidth=2.0,
            zorder=4,
        )
        if not put_crest(ax, league, team, (row["xG"], row["xG_against"]), zoom=0.20):
            ax.text(
                row["xG"],
                row["xG_against"],
                team[:3].upper(),
                color=colour,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                zorder=6,
            )
    place_labels(ax, list(zip(t["xG"], t["xG_against"])), list(t.index))
    return finish(
        fig,
        league,
        "01_xg_for_against.png",
        "xG is the model value of each chance; the dotted lines are the league medians.",
    )


# ── 2. the league as a heat table ──────────────────────────────────────────────────────
# (column, header, lower_is_better, format) read from the per-match table
HEAT_COLUMNS = [
    ("xG", "xG", False, "{:.2f}"),
    ("xG_against", "xGA", True, "{:.2f}"),
    ("shots", "SHOTS", False, "{:.1f}"),
    ("shots_against", "SHOTS V", True, "{:.1f}"),
    ("ppda", "PPDA", True, "{:.1f}"),
    ("field_tilt", "TILT %", False, "{:.0f}"),
    ("pass_pct", "PASS %", False, "{:.0f}"),
    ("progressive_passes", "PROG", False, "{:.0f}"),
    ("line_breaking_completed", "LINE BRK", False, "{:.0f}"),
    ("high_regains", "HIGH REG", False, "{:.1f}"),
    ("box_entries", "BOX ENT", False, "{:.1f}"),
    ("key_passes", "KEY P", False, "{:.1f}"),
]


def percentile_rank(series: pd.Series, lower_is_better: bool) -> pd.Series:
    """0 for the worst in the league to 1 for the best, ties sharing the average."""
    ranks = series.rank(pct=False, method="average")
    score = (ranks - 1) / max(len(series) - 1, 1)
    return 1 - score if lower_is_better else score


def league_heat_table(league: League) -> Path:
    totals = league.teams.set_index("team")
    per = league.per_match.set_index("team").loc[totals.index]
    columns = [c for c in HEAT_COLUMNS if c[0] in per.columns]
    rows = len(per)
    fig = page(
        league,
        "The league at a glance",
        "Per-match figures, shaded by rank in the league · brighter is better, whichever way the number runs",
    )
    top, bottom = 0.80, 0.07
    left, right = 0.05, 0.95
    name_w, lead_w = 0.17, 0.115
    grid_w = (right - left - name_w - lead_w) / len(columns)
    row_h = (top - bottom - 0.035) / rows
    y0 = top - 0.03
    heads = ["PTS", "GD"] + [h for _, h, _, _ in columns]
    xs = [left + name_w + lead_w * (0.25 + 0.5 * i) for i in range(2)] + [
        left + name_w + lead_w + grid_w * (i + 0.5) for i in range(len(columns))
    ]
    for x, head in zip(xs, heads):
        fig.text(
            x,
            top,
            head,
            color=base.MUTED,
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
    ramp = mcolors.LinearSegmentedColormap.from_list("heat", [base.PANEL, "#0F4A44", ACCENT])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_zorder(-1)
    ranks = {c[0]: percentile_rank(per[c[0]], c[2]) for c in columns}
    for i, (team, _row) in enumerate(per.iterrows()):
        y = y0 - (i + 1) * row_h
        ax.add_patch(
            Rectangle(
                (left, y + 0.002),
                right - left,
                row_h - 0.004,
                facecolor=base.PANEL if i % 2 == 0 else base.BG,
                edgecolor="none",
            )
        )
        fig.text(
            left + 0.012,
            y + row_h / 2,
            f"{i + 1}",
            color=base.MUTED,
            fontsize=9,
            va="center",
            ha="left",
        )
        put_crest(ax, league, team, (left + 0.05, y + row_h / 2), zoom=0.12)
        fig.text(
            left + 0.072,
            y + row_h / 2,
            team,
            color=base.TEXT,
            fontsize=10,
            fontweight="bold",
            va="center",
        )
        for x, value in zip(
            xs[:2], (totals.loc[team, "points"], totals.loc[team, "goal_difference"])
        ):
            text = f"{value:.0f}" if x == xs[0] else f"{value:+.0f}"
            fig.text(
                x, y + row_h / 2, text, color=base.TEXT, va="center", ha="center", **display(18)
            )
        for j, (key, _head, _low, fmt) in enumerate(columns):
            cx = left + name_w + lead_w + grid_w * j
            rank = float(ranks[key].loc[team])
            ax.add_patch(
                Rectangle(
                    (cx + 0.003, y + 0.004),
                    grid_w - 0.006,
                    row_h - 0.008,
                    facecolor=ramp(rank),
                    edgecolor="none",
                )
            )
            fill = mcolors.to_hex(ramp(rank))
            fig.text(
                cx + grid_w / 2,
                y + row_h / 2,
                fmt.format(per.loc[team, key]),
                color=base.BG if rank > 0.62 else base.TEXT,
                fontsize=9,
                fontweight="bold",
                va="center",
                ha="center",
            )
    return finish(
        fig,
        league,
        "02_league_heat_table.png",
        "Ordered by points. PPDA, xGA and shots faced are shaded the other way round: lower is better.",
    )


# ── 3. players who led each measure ───────────────────────────────────────────────────────
LEADERBOARDS = [
    ("xG_xA_p90", "xG + xA per 90", "{:.2f}"),
    ("progression_metres_p90", "Progression metres per 90", "{:.0f}"),
    ("key_passes_p90", "Key passes per 90", "{:.2f}"),
    ("line_breaking_passes_p90", "Line-breaking passes per 90", "{:.1f}"),
    ("defensive_actions_p90", "Defensive actions per 90", "{:.1f}"),
    ("positive_xT_p90", "xT added per 90", "{:.2f}"),
]


def player_leaderboards(league: League, minimum_minutes: float = 270.0) -> Path:
    pool = league.per90[league.per90["minutes"] >= minimum_minutes].copy()
    pool = pool[~is_goalkeeper(pool)]
    fig = page(
        league,
        "Who led the way",
        f"Per 90 minutes · players with at least {minimum_minutes:.0f} minutes ({len(pool)} qualify) · outfield only",
    )
    panels = [(k, t, f) for k, t, f in LEADERBOARDS if k in pool.columns]
    for index, (key, title, fmt) in enumerate(panels):
        column, row = index % 3, index // 3
        left = 0.05 + column * 0.305
        top = 0.80 - row * 0.385
        card = fig.add_axes([left, top - 0.355, 0.285, 0.355])
        card.axis("off")
        card.set_xlim(0, 1)
        card.set_ylim(0, 1)
        card.add_patch(
            FancyBboxPatch(
                (0, 0),
                1,
                1,
                boxstyle="round,pad=0,rounding_size=0.025",
                mutation_aspect=0.355 * 9 / (0.285 * 14),
                facecolor=base.PANEL,
                edgecolor=base.GRID,
                linewidth=1.0,
            )
        )
        card.text(
            0.05,
            0.93,
            title.upper(),
            color=base.MUTED,
            fontsize=8.5,
            fontweight="bold",
            va="center",
        )
        best = pool.nlargest(8, key)
        top_value = float(best[key].max()) or 1.0
        for rank, (_, player) in enumerate(best.iterrows()):
            y = 0.80 - rank * 0.098
            colour = league.colour(player["team"])
            card.text(0.05, y, f"{rank + 1}", color=base.MUTED, fontsize=8.5, va="center")
            card.text(
                0.10,
                y,
                surname(player["player"]),
                color=base.TEXT,
                fontsize=9.5,
                fontweight="bold",
                va="center",
            )
            card.add_patch(
                Rectangle(
                    (0.43, y - 0.027),
                    0.40 * float(player[key]) / top_value,
                    0.054,
                    facecolor=colour,
                    edgecolor="none",
                )
            )
            card.text(
                0.43 + 0.40 * float(player[key]) / top_value + 0.012,
                y,
                fmt.format(player[key]),
                color=base.TEXT,
                va="center",
                **display(15),
            )
            card.text(
                0.10, y - 0.036, str(player["team"]), color=base.NEUTRAL, fontsize=6.5, va="center"
            )
    return finish(
        fig,
        league,
        "03_player_leaderboards.png",
        "Bars share a scale within each panel. Minutes are the sum over the block.",
    )


# ── 4. how each side plays: pressing against territory ────────────────────────────────
def _crest_scatter(league: League, ax, xs, ys, names) -> None:
    for team, x, y in zip(names, xs, ys):
        colour = league.colour(team)
        ax.scatter(x, y, s=900, facecolor=base.BG, edgecolor=colour, linewidth=2.0, zorder=4)
        if not put_crest(ax, league, team, (x, y), zoom=0.20):
            ax.text(
                x,
                y,
                team[:3].upper(),
                color=colour,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                zorder=6,
            )
    place_labels(ax, list(zip(xs, ys)), list(names))


def _quadrant_axes(fig, x_label: str, y_label: str):
    ax = fig.add_axes([0.08, 0.10, 0.86, 0.69])
    ax.set_facecolor(base.PANEL)
    ax.grid(color=base.GRID, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=base.MUTED, labelsize=9, length=0)
    ax.set_xlabel(x_label, color=base.MUTED, fontsize=10, fontweight="bold")
    ax.set_ylabel(y_label, color=base.MUTED, fontsize=10, fontweight="bold")
    return ax


def _corners(ax, top_right, top_left, bottom_right, bottom_left) -> None:
    style = dict(
        color=base.NEUTRAL, fontsize=11, fontweight="bold", alpha=0.85, transform=ax.transAxes
    )
    ax.text(0.985, 0.975, top_right, ha="right", va="top", **style)
    ax.text(0.015, 0.975, top_left, ha="left", va="top", **style)
    ax.text(0.985, 0.025, bottom_right, ha="right", va="bottom", **style)
    ax.text(0.015, 0.025, bottom_left, ha="left", va="bottom", **style)


def style_map(league: League) -> Path:
    t = league.per_match.set_index("team")
    fig = page(
        league,
        "How they play",
        "Pressing intensity against territory · PPDA is passes the opponent makes per defensive action, so lower is harder",
    )
    ax = _quadrant_axes(
        fig,
        "← softer       PPDA       harder press →",
        "Field tilt: share of final-third touches (%)",
    )
    x, y = t["ppda"], t["field_tilt"]
    ax.set_xlim(x.max() + 0.6, x.min() - 0.6)  # harder pressing to the right
    ax.set_ylim(y.min() - 4, y.max() + 4)
    ax.axvline(x.median(), color=base.GRID, lw=1.1, ls=(0, (4, 4)))
    ax.axhline(y.median(), color=base.GRID, lw=1.1, ls=(0, (4, 4)))
    _corners(ax, "HIGH PRESS, HIGH TERRITORY", "PATIENT CONTROL", "PRESS AND COUNTER", "LOW BLOCK")
    _crest_scatter(league, ax, x, y, t.index)
    return finish(
        fig,
        league,
        "04_style_map.png",
        "PPDA counts the opponent's passes in the front 60% of the pitch over the side's tackles, interceptions, fouls and recoveries there.",
    )


# ── 5. results against what the chances said ────────────────────────────────────────────
def over_under(league: League) -> Path:
    t = league.teams.set_index("team")
    fig = page(
        league,
        "Luck, finishing and goalkeeping",
        "Where results ran ahead of the chances and where they fell behind · five matches is a short run, so read it as a lean",
    )
    panels = [
        (
            "goals_minus_xG",
            "FINISHING",
            "Goals scored minus xG created",
            "scored more than the chances said",
        ),
        (
            "goals_prevented",
            "GOALKEEPING",
            "xGOT faced minus goals conceded",
            "conceded fewer than deserved",
        ),
    ]
    for index, (key, heading, caption, good) in enumerate(panels):
        ordered = t[key].sort_values()
        left = 0.05 + index * 0.475
        ax = fig.add_axes([left + 0.075, 0.075, 0.33, 0.69])
        ax.set_facecolor(base.BG)
        positions = np.arange(len(ordered))
        ax.barh(
            positions,
            ordered.values,
            color=[league.colour(team) for team in ordered.index],
            height=0.66,
        )
        ax.axvline(0, color=base.TEXT, lw=1.0, alpha=0.7)
        ax.set_yticks(positions)
        ax.set_yticklabels(ordered.index, fontsize=9, color=base.TEXT)
        ax.tick_params(axis="x", colors=base.MUTED, labelsize=8.5, length=0)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=base.GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        span = float(np.abs(ordered.values).max()) or 1.0
        ax.set_xlim(-span * 1.28, span * 1.28)
        for position, value in zip(positions, ordered.values):
            ax.text(
                value + (0.04 * span if value >= 0 else -0.04 * span),
                position,
                f"{value:+.2f}",
                ha="left" if value >= 0 else "right",
                va="center",
                color=base.TEXT,
                fontsize=8.5,
                fontweight="bold",
            )
        fig.text(left + 0.075, 0.795, heading, color=base.TEXT, **display(22))
        fig.text(left + 0.075, 0.772, caption, color=base.MUTED, fontsize=9)
        fig.text(
            left + 0.075 + 0.33,
            0.772,
            f"right = {good}",
            color=base.NEUTRAL,
            fontsize=7.5,
            ha="right",
        )
    return finish(
        fig,
        league,
        "05_finishing_goalkeeping.png",
        "Team totals over the block. A goalkeeper's figure is the shots on target faced, priced after the strike, less the goals that went in.",
    )


# ── 6. where the chances came from ────────────────────────────────────────────────────────
SOURCES = [
    ("open_play_xG", "Open play", "#4EA8FF"),
    ("corner_xG", "Corners", "#FFC247"),
    ("free_kick_xG", "Free kicks", "#2FD3BE"),
    ("throw_in_xG", "Throw-ins", "#A77BFF"),
    ("penalty_xG", "Penalties", "#FF7A5C"),
]


def chance_sources(league: League) -> Path:
    t = league.per_match.set_index("team")
    columns = [s for s in SOURCES if s[0] in t.columns]
    order = t[[c for c, _, _ in columns]].sum(axis=1).sort_values()
    fig = page(
        league,
        "Where the chances came from",
        "Expected goals per match, split by how the move began · the dead-ball share shows who lives off set pieces",
    )
    ax = fig.add_axes([0.16, 0.075, 0.76, 0.69])
    ax.set_facecolor(base.BG)
    positions = np.arange(len(order))
    left = np.zeros(len(order))
    for column, label, colour in columns:
        values = t.loc[order.index, column].to_numpy()
        ax.barh(
            positions,
            values,
            left=left,
            color=colour,
            height=0.68,
            label=label,
            edgecolor=base.BG,
            linewidth=0.8,
        )
        left += values
    ax.set_yticks(positions)
    ax.set_yticklabels(order.index, fontsize=9.5, color=base.TEXT)
    ax.tick_params(axis="x", colors=base.MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=base.GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    dead_cols = [c for c in ("corner_xG", "free_kick_xG", "penalty_xG") if c in t.columns]
    dead = t.loc[order.index, dead_cols].sum(axis=1)
    for position, (team, total) in enumerate(order.items()):
        share = 100 * dead[team] / total if total else 0
        ax.text(
            total + 0.03,
            position,
            f"{total:.2f}   ·   {share:.0f}% dead ball",
            va="center",
            color=base.TEXT,
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xlim(0, float(order.max()) * 1.32)
    ax.legend(loc="lower right", frameon=False, labelcolor=base.TEXT, fontsize=9)
    return finish(
        fig,
        league,
        "06_chance_sources.png",
        "Dead ball = corners, free kicks and penalties. A throw-in is a restart, not a rehearsed set piece, so it is shown but not counted as one.",
    )


# ── 7. players: who creates, who progresses, who wins it back ────────────────────────────────
def _qualified(league: League, minimum_minutes: float, outfield=True) -> pd.DataFrame:
    pool = league.per90[league.per90["minutes"] >= minimum_minutes].copy()
    return pool[~is_goalkeeper(pool)] if outfield else pool


def _player_scatter(
    league: League,
    key_x: str,
    key_y: str,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    name: str,
    note: str,
    minimum_minutes: float = 270.0,
    labels: int = 14,
) -> Path:
    pool = _qualified(league, minimum_minutes)
    fig = page(league, title, subtitle.format(n=len(pool), m=int(minimum_minutes)))
    ax = _quadrant_axes(fig, x_label, y_label)
    ax.axvline(pool[key_x].median(), color=base.GRID, lw=1.1, ls=(0, (4, 4)))
    ax.axhline(pool[key_y].median(), color=base.GRID, lw=1.1, ls=(0, (4, 4)))
    spread = max(pool["minutes"].max() - pool["minutes"].min(), 1)
    sizes = 30 + 150 * (pool["minutes"] - pool["minutes"].min()) / spread
    ax.scatter(
        pool[key_x],
        pool[key_y],
        s=sizes,
        c=[league.colour(t) for t in pool["team"]],
        alpha=0.85,
        edgecolor=base.BG,
        linewidth=0.8,
        zorder=3,
    )
    # name the players furthest from the middle on either measure
    zx = (pool[key_x] - pool[key_x].mean()) / (pool[key_x].std() or 1)
    zy = (pool[key_y] - pool[key_y].mean()) / (pool[key_y].std() or 1)
    standout = pool.assign(_score=np.hypot(zx, zy)).nlargest(labels, "_score")
    for _, row in standout.iterrows():
        ax.annotate(
            surname(row["player"]),
            (row[key_x], row[key_y]),
            xytext=(6, 6),
            textcoords="offset points",
            color=base.TEXT,
            fontsize=8.5,
            fontweight="bold",
            zorder=6,
        )
    return finish(fig, league, name, note)


def scoring_and_creating(league: League) -> Path:
    return _player_scatter(
        league,
        "xG_p90",
        "xA_p90",
        "Finishers and creators",
        "xG against xA per 90 · {n} players with at least {m} minutes · bigger dots played more",
        "xG per 90  →",
        "xA per 90  →",
        "07_finishers_creators.png",
        "Named: the players furthest from the middle of the pack. Dotted lines are the medians.",
    )


def progressing_the_ball(league: League) -> Path:
    return _player_scatter(
        league,
        "progression_metres_p90",
        "line_breaking_passes_p90",
        "Moving the ball forward",
        "Progression metres against line-breaking passes per 90 · {n} players with at least {m} minutes",
        "Progression metres per 90  →",
        "Line-breaking passes per 90  →",
        "08_ball_progressors.png",
        "Progression metres are ground gained toward goal by completed passes and carries.",
    )


def winning_the_ball(league: League) -> Path:
    return _player_scatter(
        league,
        "recoveries_p90",
        "tackles_won_p90",
        "Winning the ball back",
        "Recoveries against tackles won per 90 · {n} players with at least {m} minutes",
        "Recoveries per 90  →",
        "Tackles won per 90  →",
        "09_ball_winners.png",
        "Named: the players furthest from the middle of the pack.",
    )


# ── 8. goalkeepers ────────────────────────────────────────────────────────────────────────
def goalkeepers(league: League, minimum_minutes: float = 180.0) -> Path:
    keepers = league.players[
        is_goalkeeper(league.players) & (league.players["minutes"] >= minimum_minutes)
    ]
    keepers = keepers.sort_values("goals_prevented", ascending=False)
    fig = page(
        league,
        "Goalkeepers",
        f"Goals prevented is the post-shot xG of the shots faced less the goals conceded · keepers with at least {minimum_minutes:.0f} minutes",
    )
    heads = [
        ("MIN", "minutes", "{:.0f}", 0.34),
        ("SAVES", "saves", "{:.0f}", 0.42),
        ("CLAIMS", "claims", "{:.0f}", 0.50),
        ("SWEEPS", "sweeps", "{:.0f}", 0.58),
        ("PASS %", "pass_pct", "{:.0f}", 0.66),
        ("PREVENTED", "goals_prevented", "{:+.2f}", 0.76),
    ]
    top = 0.78
    row_h = min(0.036, 0.66 / max(len(keepers), 1))
    for head, _, _, x in heads:
        fig.text(x, top, head, color=base.MUTED, fontsize=8.5, fontweight="bold", ha="center")
    fig.text(0.10, top, "GOALKEEPER", color=base.MUTED, fontsize=8.5, fontweight="bold")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_zorder(-1)
    span = float(np.abs(keepers["goals_prevented"]).max()) or 1.0
    for i, (_, row) in enumerate(keepers.iterrows()):
        y = top - 0.03 - i * row_h
        ax.add_patch(
            Rectangle(
                (0.05, y - row_h / 2 + 0.002),
                0.9,
                row_h - 0.004,
                facecolor=base.PANEL if i % 2 == 0 else base.BG,
                edgecolor="none",
            )
        )
        put_crest(ax, league, row["team"], (0.075, y), zoom=0.11)
        fig.text(
            0.10,
            y,
            surname(row["player"]),
            color=base.TEXT,
            fontsize=10,
            fontweight="bold",
            va="center",
        )
        fig.text(0.20, y, row["team"], color=base.NEUTRAL, fontsize=8, va="center")
        for _, key, fmt, x in heads:
            value = row[key]
            if pd.isna(value):
                continue
            if key == "goals_prevented":
                width = 0.07 * float(value) / span
                ax.add_patch(
                    Rectangle(
                        (x, y - 0.009),
                        width,
                        0.018,
                        facecolor=league.colour(row["team"]),
                        edgecolor="none",
                    )
                )
                fig.text(
                    x + 0.08,
                    y,
                    fmt.format(value),
                    color=base.TEXT,
                    va="center",
                    ha="left",
                    **display(15),
                )
            else:
                fig.text(
                    x, y, fmt.format(value), color=base.TEXT, fontsize=9.5, va="center", ha="center"
                )
    return finish(
        fig,
        league,
        "10_goalkeepers.png",
        "Penalties and own goals are left out of goals prevented.",
    )


# ── profile cards: one club or one player against the rest of the league ───────────────────
# Fully saturated, so a slice stays vivid against the near-black page and the three groups
# cannot be mistaken for one another at a glance.
GROUP_COLOURS = {"ATTACK": "#FF3B1D", "BUILD-UP": "#0A84FF", "DEFENCE": "#00D95F"}

# (column, label, lower_is_better, format, group)
TEAM_SLICES = [
    ("xG", "xG", False, "{:.2f}", "ATTACK"),
    ("shots", "Shots", False, "{:.1f}", "ATTACK"),
    ("box_entries", "Box entries", False, "{:.1f}", "ATTACK"),
    ("key_passes", "Key passes", False, "{:.1f}", "ATTACK"),
    ("pass_pct", "Pass %", False, "{:.0f}", "BUILD-UP"),
    ("progressive_passes", "Progressive", False, "{:.0f}", "BUILD-UP"),
    ("line_breaking_completed", "Line-breaking", False, "{:.0f}", "BUILD-UP"),
    ("field_tilt", "Field tilt", False, "{:.0f}", "BUILD-UP"),
    ("xG_against", "xG against", True, "{:.2f}", "DEFENCE"),
    ("shots_against", "Shots against", True, "{:.1f}", "DEFENCE"),
    ("ppda", "PPDA", True, "{:.1f}", "DEFENCE"),
    ("high_regains", "High regains", False, "{:.1f}", "DEFENCE"),
]

PLAYER_SLICES = [
    ("xG_p90", "xG", False, "{:.2f}", "ATTACK"),
    ("xA_p90", "xA", False, "{:.2f}", "ATTACK"),
    ("shots_p90", "Shots", False, "{:.1f}", "ATTACK"),
    ("takeons_won_p90", "Take-ons won", False, "{:.1f}", "ATTACK"),
    ("key_passes_p90", "Key passes", False, "{:.2f}", "BUILD-UP"),
    ("positive_xT_p90", "xT added", False, "{:.2f}", "BUILD-UP"),
    ("progression_metres_p90", "Progression m", False, "{:.0f}", "BUILD-UP"),
    ("line_breaking_passes_p90", "Line-breaking", False, "{:.1f}", "BUILD-UP"),
    ("tackles_won_p90", "Tackles won", False, "{:.1f}", "DEFENCE"),
    ("interceptions_p90", "Interceptions", False, "{:.1f}", "DEFENCE"),
    ("recoveries_p90", "Recoveries", False, "{:.1f}", "DEFENCE"),
    ("clearances_p90", "Clearances", False, "{:.1f}", "DEFENCE"),
]


def pizza(ax, slices, shares, raws, colour_by_group=GROUP_COLOURS) -> None:
    """Twelve slices, each as long as the share of the league it beats (full is the best).

    ``shares`` run 0 to 1; a missing value draws an empty slice and prints a dash. The
    raw figure is printed beside its label so the picture never has to be decoded.
    """
    n = len(slices)
    width = 2 * np.pi / n
    theta = np.arange(n) * width
    ax.set_theta_offset(np.pi / 2 + width / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.set_facecolor("none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.bar(
        theta, 1.0, width=width * 0.96, color=base.PANEL, edgecolor=base.BG, linewidth=1.2, zorder=1
    )
    for ring in (0.25, 0.5, 0.75):
        ax.plot(np.linspace(0, 2 * np.pi, 200), [ring] * 200, color=base.GRID, lw=0.7, zorder=2)
    for i, (key, label, _low, fmt, group) in enumerate(slices):
        share = shares[i]
        colour = colour_by_group[group]
        if share is not None and not np.isnan(share):
            ax.bar(
                theta[i],
                max(share, 0.04),
                width=width * 0.96,
                color=colour,
                edgecolor=base.BG,
                linewidth=1.2,
                alpha=1.0,
                zorder=3,
            )
        angle = theta[i]
        raw = raws[i]
        value = "—" if raw is None or pd.isna(raw) else fmt.format(raw)
        # The label and its figure share one anchor just outside the slice, aligned away
        # from the centre: left of the anchor on the left of the chart and right of it on
        # the right, so a long name never runs back under its own number.
        visual = np.pi / 2 + width / 2 - angle
        side = np.cos(visual)
        ha = "left" if side > 0.35 else ("right" if side < -0.35 else "center")
        shift = 4 if ha == "left" else (-4 if ha == "right" else 0)
        common = dict(
            xy=(angle, 1.06), textcoords="offset points", ha=ha, annotation_clip=False, zorder=5
        )
        lift = 10 if np.sin(visual) > 0.7 else (-6 if np.sin(visual) < -0.7 else 0)
        ax.annotate(
            label.upper(),
            xytext=(shift, 7 + lift),
            va="bottom",
            color=base.MUTED,
            fontsize=7.6,
            fontweight="bold",
            **common,
        )
        ax.annotate(
            value,
            xytext=(shift, 6 + lift),
            va="top",
            color=base.TEXT,
            fontsize=13,
            fontweight="bold",
            fontfamily="Barlow Condensed",
            **common,
        )


def _percentiles(frame: pd.DataFrame, row_index, slices) -> tuple[list, list]:
    shares, raws = [], []
    for key, _label, lower, _fmt, _group in slices:
        if key not in frame:
            shares.append(None)
            raws.append(None)
            continue
        ranks = percentile_rank(frame[key], lower)
        shares.append(float(ranks.loc[row_index]) if pd.notna(ranks.loc[row_index]) else None)
        raws.append(frame.loc[row_index, key])
    return shares, raws


def _legend(fig, x, y) -> None:
    for i, (group, colour) in enumerate(GROUP_COLOURS.items()):
        fig.add_artist(
            Rectangle(
                (x + i * 0.095, y), 0.011, 0.016, transform=fig.transFigure, color=colour, lw=0
            )
        )
        fig.text(
            x + i * 0.095 + 0.016,
            y + 0.008,
            group,
            color=base.MUTED,
            fontsize=8,
            fontweight="bold",
            va="center",
        )


def _card_header(fig, league: League, team: str, title: str, subtitle: str) -> None:
    ax = fig.add_axes([0.03, 0.865, 0.1, 0.11])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    put_crest(ax, league, team, (0.5, 0.5), zoom=0.40)
    fig.text(0.14, 0.935, title, color=base.TEXT, va="center", **display(36))
    fig.text(0.14, 0.893, subtitle, color=base.MUTED, fontsize=10.5, va="center")
    fig.add_artist(
        Rectangle(
            (0.03, 0.852), 0.94, 0.003, transform=fig.transFigure, color=league.colour(team), lw=0
        )
    )
    fig.text(0.97, 0.022, "MOSTAFA SAAD", color=base.MUTED, ha="right", **display(13))


def team_profile(league: League, team: str) -> Path:
    per = league.per_match.set_index("team")
    totals = league.teams.set_index("team")
    shares, raws = _percentiles(per, team, TEAM_SLICES)
    row = totals.loc[team]
    position = int(list(totals.index).index(team)) + 1
    fig = plt.figure(figsize=(12, 9), facecolor=base.BG)
    _card_header(
        fig,
        league,
        team,
        team.upper(),
        f"{league.title.title()} · {league.rounds.title()} · {int(row['matches'])} matches · {int(row['won'])}W {int(row['drawn'])}D {int(row['lost'])}L · {int(row['points'])} points · {position}{_ordinal(position)}",
    )
    ax = fig.add_axes([0.27, 0.13, 0.46, 0.60], projection="polar")
    pizza(ax, TEAM_SLICES, shares, raws)
    ax.text(0, 0, "", transform=ax.transAxes)
    _legend(fig, 0.05, 0.075)
    fig.text(
        0.97,
        0.075,
        "Slice length = share of the league this side beats · per match",
        color=base.NEUTRAL,
        fontsize=8,
        ha="right",
        va="center",
    )
    # four figures that carry the story, with the rank each one holds
    figures = [
        ("GOALS", f"{int(row['goals_for'])}–{int(row['goals_against'])}", None),
        ("xG", f"{row['xG']:.1f}–{row['xG_against']:.1f}", None),
        ("GOALS − xG", f"{row['goals_minus_xG']:+.1f}", None),
        ("PREVENTED", f"{row['goals_prevented']:+.1f}", None),
    ]
    for i, (label, value, _) in enumerate(figures):
        x = 0.03 + (i % 2) * 0.115
        y = 0.70 - (i // 2) * 0.115
        fig.text(x, y + 0.045, label, color=base.MUTED, fontsize=8, fontweight="bold")
        fig.text(x, y, value, color=base.TEXT, **display(24))
    return finish_card(fig, league, f"teams/{team.replace(' ', '_')}.png")


def _ordinal(n: int) -> str:
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def finish_card(fig, league: League, relative: str) -> Path:
    path = league.folder / "visuals" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=base.BG)
    plt.close(fig)
    return path


def player_profile(
    league: League, player: str, team: str, minimum_minutes: float = 270.0
) -> Path | None:
    pool = _qualified(league, minimum_minutes)
    role_rows = league.per90.set_index(["player", "team"]).loc[(player, team)]
    role = role_rows["role"]
    peers = (
        pool[pool["role"].eq(role)].reset_index(drop=True)
        if role in set(pool["role"])
        else pool.reset_index(drop=True)
    )
    peers = peers.set_index(["player", "team"])
    if (player, team) not in peers.index:
        return None
    shares, raws = _percentiles(peers, (player, team), PLAYER_SLICES)
    totals = league.players.set_index(["player", "team"]).loc[(player, team)]
    fig = plt.figure(figsize=(12, 9), facecolor=base.BG)
    rating = totals.get("avg_rating")
    _card_header(
        fig,
        league,
        team,
        player.upper(),
        f"{team} · {role} · {int(totals['matches'])} matches · {totals['minutes']:.0f} minutes"
        + (f" · rating {rating:.2f}" if pd.notna(rating) else ""),
    )
    ax = fig.add_axes([0.27, 0.13, 0.46, 0.60], projection="polar")
    pizza(ax, PLAYER_SLICES, shares, raws)
    _legend(fig, 0.05, 0.075)
    fig.text(
        0.97,
        0.075,
        f"Slice length = share of the {len(peers)} {role.lower()}s with {minimum_minutes:.0f}+ minutes this player beats · per 90",
        color=base.NEUTRAL,
        fontsize=8,
        ha="right",
        va="center",
    )
    figures = [
        ("GOALS", f"{totals['goals']:.0f}"),
        ("ASSISTS", f"{totals['assists']:.0f}"),
        ("KEY PASSES", f"{totals['key_passes']:.0f}"),
        ("SHOTS", f"{totals['shots']:.0f}"),
    ]
    for i, (label, value) in enumerate(figures):
        x = 0.03 + (i % 2) * 0.115
        y = 0.70 - (i // 2) * 0.115
        fig.text(x, y + 0.045, label, color=base.MUTED, fontsize=8, fontweight="bold")
        fig.text(x, y, value, color=base.TEXT, **display(24))
    safe = re.sub(r"[^\w\-]+", "_", player).strip("_")
    return finish_card(fig, league, f"players/{team.replace(' ', '_')}/{safe}.png")
