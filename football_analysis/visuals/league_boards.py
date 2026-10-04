"""Shared pieces for the season graphics, and the club and player profile cards.

Everything here reads the aggregate tables ``scripts/aggregate_rounds.py`` writes and
knows nothing about events. ``League`` holds the tables and each club's id and colour;
the helpers draw badges, spread overlapping marks apart and place labels where nothing
else is; ``team_profile`` and ``player_profile`` draw the ringed pizza cards. All are
16:9 (1600 x 900): the widest frame a timeline shows, and the one every graphic uses.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle

from football_analysis.visuals import crests
from football_analysis.visuals import visual_redesign_preview as base
from football_analysis.visuals.typography import display

matplotlib.use("Agg")

SIZE = (12.0, 6.75)
DPI = 1600 / 12  # 1600 x 900
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


def _qualified(league: League, minimum_minutes: float, outfield=True) -> pd.DataFrame:
    pool = league.per90[league.per90["minutes"] >= minimum_minutes].copy()
    return pool[~is_goalkeeper(pool)] if outfield else pool


def percentile_rank(series: pd.Series, lower_is_better: bool) -> pd.Series:
    """0 for the worst in the league to 1 for the best, ties sharing the average."""
    ranks = series.rank(pct=False, method="average")
    score = (ranks - 1) / max(len(series) - 1, 1)
    return 1 - score if lower_is_better else score


def put_crest(ax, league: League, team: str, xy, zoom=0.24, transform=None):
    """A club badge at ``xy``; returns False when there is none to draw."""
    team_id = league.ids.get(team)
    image = crests.crest_image(team_id, allow_download=False) if team_id is not None else None
    if image is None:
        return False
    ax.add_artist(
        AnnotationBbox(
            OffsetImage(image, zoom=zoom),
            xy,
            frameon=False,
            xycoords=transform or "data",
            box_alignment=(0.5, 0.5),
            zorder=6,
        )
    )
    return True


# ── keeping marks apart ───────────────────────────────────────────────────────────────
def spread_apart(ax, xs, ys, radius_px: float, reach: float = 3.0):
    """Nudge marks that overlap until they do not, and say where each one ended up.

    Positions are moved in screen pixels, no further than ``reach`` radii from where the
    data puts them, and returned in data units. Callers draw a small dot at the true
    position and a hairline to the moved mark, so the chart stays honest about where a
    value really is. The marks are moved in pairs, half each, so a tight cluster opens up
    like a flower and not into a line.
    """
    ax.figure.canvas.draw()
    home = ax.transData.transform(np.column_stack([xs, ys])).astype(float)
    moved = home.copy()
    gap = 2 * radius_px + 2.0
    for _ in range(80):
        shifted = False
        for i in range(len(moved)):
            for j in range(i + 1, len(moved)):
                delta = moved[j] - moved[i]
                distance = float(np.hypot(*delta))
                if distance >= gap:
                    continue
                direction = (
                    delta / distance if distance > 1e-6 else np.array([np.cos(i), np.sin(i)])
                )
                push = (gap - distance) / 2 + 0.1
                moved[i] -= direction * push
                moved[j] += direction * push
                shifted = True
        pull = moved - home
        length = np.hypot(pull[:, 0], pull[:, 1])
        too_far = length > reach * radius_px
        if too_far.any():
            moved[too_far] = (
                home[too_far] + pull[too_far] / length[too_far, None] * reach * radius_px
            )
        if not shifted:
            break
    return ax.transData.inverted().transform(moved)


def place_labels(
    ax, points, names, radius_px=12.0, fontsize=8, avoid=None, leaders=True, badges=False
):
    """Name each point in the nearest free spot: not on another label, badge or point.

    ``points`` are data coordinates. Each label tries sixteen directions at three distances
    and keeps the first position that overlaps nothing, so a crowded cluster spreads its
    names outward instead of printing them on top of one another. A label that had to move
    away is joined to its point by a hairline. ``avoid`` is any further points a label must
    not cover (every dot on a busy scatter, when only the outliers are named).
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pixels = ax.transData.transform(np.asarray(points, dtype=float))
    others = pixels if avoid is None else ax.transData.transform(np.asarray(avoid, dtype=float))
    axes_box = ax.get_window_extent(renderer)
    taken = []
    drawn = []
    # with ``badges`` every point is a round mark of ``radius_px``; a label may not sit on one
    marks = (
        [(px - radius_px, py - radius_px, px + radius_px, py + radius_px) for px, py in pixels]
        if badges
        else []
    )

    def clash(box, own=None):
        boxes = taken + [m for k, m in enumerate(marks) if k != own]
        return sum(
            box[0] < o[2] and box[2] > o[0] and box[1] < o[3] and box[3] > o[1] for o in boxes
        )

    def covers(box):
        return int(
            (
                (others[:, 0] > box[0])
                & (others[:, 0] < box[2])
                & (others[:, 1] > box[1])
                & (others[:, 1] < box[3])
            ).sum()
        )

    scale = fig.dpi / 72.0
    for index, ((x, y), name) in enumerate(zip(pixels, names)):
        probe = ax.text(0, 0, name, fontsize=fontsize, fontweight="bold")
        extent = probe.get_window_extent(renderer)
        width, height = extent.width, extent.height
        probe.remove()
        best = None
        for step, distance in enumerate(
            (radius_px + 3 * scale, radius_px + 14 * scale, radius_px + 30 * scale)
        ):
            for angle in np.linspace(-np.pi / 2, 3 * np.pi / 2, 16, endpoint=False):
                cx, cy = x + distance * np.cos(angle), y + distance * np.sin(angle)
                ha = (
                    "left"
                    if np.cos(angle) > 0.4
                    else ("right" if np.cos(angle) < -0.4 else "center")
                )
                va = (
                    "bottom"
                    if np.sin(angle) > 0.4
                    else ("top" if np.sin(angle) < -0.4 else "center")
                )
                left = cx if ha == "left" else (cx - width if ha == "right" else cx - width / 2)
                bottom = cy if va == "bottom" else (cy - height if va == "top" else cy - height / 2)
                box = (left, bottom, left + width, bottom + height)
                inside = (
                    box[0] >= axes_box.x0
                    and box[2] <= axes_box.x1
                    and box[1] >= axes_box.y0
                    and box[3] <= axes_box.y1
                )
                cost = (
                    clash(box, index) * 1000 + covers(box) * 60 + step * 10 + (0 if inside else 400)
                )
                if best is None or cost < best[0]:
                    best = (cost, box, ha, va, cx, cy, step)
            if best[0] < 10:  # nothing in the way at this distance
                break
        _, box, ha, va, cx, cy, step = best
        taken.append(box)
        data = ax.transData.inverted().transform((cx, cy))
        drawn.append(
            ax.text(
                data[0],
                data[1],
                name,
                ha=ha,
                va=va,
                color=base.TEXT,
                fontsize=fontsize,
                fontweight="bold",
                zorder=8,
            )
        )
        if leaders and step > 0:
            anchor = ax.transData.inverted().transform((x, y))
            ax.plot([anchor[0], data[0]], [anchor[1], data[1]], color="#6B7580", lw=0.6, zorder=2)
    return drawn


# ── profile cards: one club or one player against the rest of the league ───────────────────
# Three deep, distinct hues -- vermilion, amber, blue -- that hold up on the near-black page
# and stay apart for readers who confuse red and green.
GROUP_COLOURS = {"ATTACK": "#E8452C", "BUILD-UP": "#F2B134", "DEFENCE": "#2F7FE0"}
TRACK = "#101419"  # the empty part of each slice's lane
RING = "#2A3037"  # the 20, 40, 60 and 80 per cent circles
RING_OUTER = "#5A646E"  # the outer circle and the one round the hole
INNER = 0.17  # the hole in the middle, where the club's badge sits
MUTED_SLICE = "#2B323A"  # what a weak slice fades toward

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


def _radius(share: float) -> float:
    """Where a share of the league (0 to 1) ends on the radius, outside the central hole."""
    return INNER + share * (1.0 - INNER)


def darken(colour: str, amount: float = 0.38) -> str:
    """The colour pulled toward black, for the badge that sits on a slice of that colour."""
    return mcolors.to_hex(np.array(mcolors.to_rgb(colour)) * (1.0 - amount))


def strength_colour(colour: str, share: float) -> str:
    """The group colour, faded toward grey the further the slice falls short of the league."""
    fade = float(np.clip((0.55 - share) / 0.55, 0.0, 1.0)) * 0.72
    a, b = np.array(mcolors.to_rgb(colour)), np.array(mcolors.to_rgb(MUTED_SLICE))
    return mcolors.to_hex(a * (1.0 - fade) + b * fade)


def pizza(ax, slices, shares, raws, badges, centre=None) -> None:
    """Twelve slices round a central hole, each as long as the share of the league it beats.

    Rings at 20, 40, 60 and 80 per cent make the length readable without a scale, the dashed
    ring at 50 is the league median, and each slice ends in a circle holding ``badges[i]`` (a
    rank or a percentile). A measure where less is better is marked with a down arrow: its
    slice is long when the figure is small. ``centre`` is a ``(league, club)`` pair whose
    badge fills the hole.
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
    full = np.linspace(0, 2 * np.pi, 361)
    ax.bar(
        theta,
        1.0 - INNER,
        bottom=INNER,
        width=width * 0.94,
        color=TRACK,
        edgecolor="none",
        zorder=1,
    )
    for ring in (0.2, 0.4, 0.6, 0.8):
        ax.plot(full, [_radius(ring)] * len(full), color=RING, lw=0.8, zorder=2)
    ax.plot(full, [_radius(0.5)] * len(full), color="#8E99A4", lw=1.1, ls=(0, (4, 4)), zorder=5)
    ax.plot(full, [1.0] * len(full), color=RING_OUTER, lw=1.6, zorder=4)
    ax.plot(full, [INNER] * len(full), color=RING_OUTER, lw=1.6, zorder=4)
    for i, (key, label, lower, fmt, group) in enumerate(slices):
        share = shares[i]
        angle = theta[i]
        if share is not None and not np.isnan(share):
            colour = strength_colour(GROUP_COLOURS[group], share)
            length = max(share, 0.05)
            ax.bar(
                angle,
                length * (1.0 - INNER),
                bottom=INNER,
                width=width * 0.94,
                color=colour,
                edgecolor=base.BG,
                linewidth=1.0,
                zorder=3,
            )
            tip = max(_radius(length) - 0.075, INNER + 0.06)
            ax.text(
                angle,
                tip,
                badges[i],
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
                zorder=6,
                bbox=dict(
                    boxstyle="circle,pad=0.26",
                    facecolor=darken(GROUP_COLOURS[group], 0.55),
                    edgecolor="white",
                    linewidth=1.1,
                ),
            )
        raw = raws[i]
        value = "—" if raw is None or pd.isna(raw) else fmt.format(raw)
        visual = np.pi / 2 + width / 2 - angle
        side = np.cos(visual)
        ha = "left" if side > 0.35 else ("right" if side < -0.35 else "center")
        shift = 4 if ha == "left" else (-4 if ha == "right" else 0)
        common = dict(
            xy=(angle, 1.06), textcoords="offset points", ha=ha, annotation_clip=False, zorder=5
        )
        lift = 9 if np.sin(visual) > 0.7 else (-5 if np.sin(visual) < -0.7 else 0)
        ax.annotate(
            label.upper() + ("  ↓" if lower else ""),
            xytext=(shift, 6 + lift),
            va="bottom",
            color=base.MUTED,
            fontsize=7.4,
            fontweight="bold",
            **common,
        )
        ax.annotate(
            value,
            xytext=(shift, 5 + lift),
            va="top",
            color=base.TEXT,
            fontsize=12,
            fontweight="bold",
            fontfamily="Barlow Condensed",
            **common,
        )
    if centre is not None:
        league, club = centre
        put_crest(ax, league, club, (0.5, 0.5), zoom=0.36, transform="axes fraction")


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


def _ordinal(n: int) -> str:
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _card_header(fig, colour: str, title: str, subtitle: str) -> None:
    fig.text(0.03, 0.93, title, color=base.TEXT, va="center", **display(40))
    fig.text(0.03, 0.865, subtitle, color=base.MUTED, fontsize=10, va="center")
    fig.add_artist(
        Rectangle((0.03, 0.835), 0.94, 0.004, transform=fig.transFigure, color=colour, lw=0)
    )
    fig.text(0.97, 0.03, "MOSTAFA SAAD", color=base.MUTED, ha="right", va="center", **display(13))


def _legend_column(fig, x, y) -> None:
    for group, colour in GROUP_COLOURS.items():
        fig.add_artist(
            Rectangle((x, y - 0.008), 0.011, 0.02, transform=fig.transFigure, color=colour, lw=0)
        )
        fig.text(
            x + 0.017,
            y + 0.002,
            group,
            color=base.MUTED,
            fontsize=8,
            fontweight="bold",
            va="center",
        )
        y -= 0.05


def _figure_pairs(fig, pairs, x0=0.03, y0=0.745) -> None:
    """The headline figures, stacked in one column left of the chart."""
    for i, (label, value) in enumerate(pairs):
        y = y0 - i * 0.15
        fig.text(x0, y + 0.045, label, color=base.MUTED, fontsize=8, fontweight="bold")
        fig.text(x0, y, value, color=base.TEXT, va="center", **display(26))


def _findings(fig, slices, shares, raws, ranks, x: float) -> None:
    """The three strongest and the three weakest measures, in words, right of the chart."""
    scored = sorted(
        (
            (shares[i], i)
            for i in range(len(slices))
            if shares[i] is not None and not np.isnan(shares[i])
        ),
        reverse=True,
    )
    blocks = (("STRONGEST", scored[:3], "#3DDC84"), ("WEAKEST", scored[::-1][:3], "#FF6B5B"))
    for b, (heading, rows, colour) in enumerate(blocks):
        top = 0.775 - b * 0.34
        fig.text(x, top, heading, color=colour, fontsize=9, fontweight="bold", va="center")
        fig.add_artist(
            Rectangle((x, top - 0.02), 0.17, 0.002, transform=fig.transFigure, color=colour, lw=0)
        )
        for r, (_, i) in enumerate(rows):
            y = top - 0.085 - r * 0.072
            key, label, lower, fmt, group = slices[i]
            fig.text(
                x,
                y + 0.028,
                label.upper() + (" ↓" if lower else ""),
                color=base.MUTED,
                fontsize=7.6,
                fontweight="bold",
                va="center",
            )
            fig.text(x, y - 0.012, fmt.format(raws[i]), color=base.TEXT, va="center", **display(20))
            fig.text(
                x + 0.17,
                y - 0.012,
                ranks[i],
                color=base.TEXT,
                va="center",
                ha="right",
                **display(20),
            )


def _footnote(fig, text: str, x: float, width: int) -> None:
    fig.text(
        x,
        0.075,
        textwrap.fill(text, width),
        color=base.NEUTRAL,
        fontsize=7,
        va="bottom",
        linespacing=1.4,
    )


def finish_card(fig, league: League, scope: str, filename: str) -> Path:
    path = league.folder / "visuals" / scope / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=base.BG)
    plt.close(fig)
    return path


def team_profile(league: League, team: str) -> Path:
    per = league.per_match.set_index("team")
    totals = league.teams.set_index("team")
    shares, raws = _percentiles(per, team, TEAM_SLICES)
    size = len(per)
    positions = [None if s is None else int(round(size - s * (size - 1))) for s in shares]
    badges = [str(p) if p is not None else "" for p in positions]
    ranks = [f"#{p}" if p is not None else "—" for p in positions]
    row = totals.loc[team]
    place = int(list(totals.index).index(team)) + 1
    fig = plt.figure(figsize=SIZE, facecolor=base.BG)
    _card_header(
        fig,
        league.colour(team),
        team.upper(),
        f"{league.title.title()} · {league.rounds.title()} · {int(row['matches'])} matches · "
        f"{int(row['won'])}W {int(row['drawn'])}D {int(row['lost'])}L · {int(row['points'])} points · {place}{_ordinal(place)}",
    )
    ax = fig.add_axes([0.285, 0.075, 0.43, 0.66], projection="polar")
    pizza(ax, TEAM_SLICES, shares, raws, badges, centre=(league, team))
    _figure_pairs(
        fig,
        [
            ("GOALS", f"{int(row['goals_for'])}–{int(row['goals_against'])}"),
            ("xG", f"{row['xG']:.1f}–{row['xG_against']:.1f}"),
            ("GOALS − xG", f"{row['goals_minus_xG']:+.1f}"),
        ],
    )
    _legend_column(fig, 0.03, 0.2)
    _findings(fig, TEAM_SLICES, shares, raws, ranks, 0.79)
    _footnote(
        fig,
        f"Slice length = share of the league this side beats · circle = rank of {size}, 1 is best · dashed ring = league median · ↓ fewer is better, so a long slice is a small figure",
        0.79,
        46,
    )
    return finish_card(fig, league, "teams", f"profile_{team.replace(' ', '_')}.png")


def player_profile(
    league: League, player: str, team: str, minimum_minutes: float = 270.0
) -> Path | None:
    pool = _qualified(league, minimum_minutes)
    role = league.per90.set_index(["player", "team"]).loc[(player, team)]["role"]
    # a player who only came off the bench has no listed position: he is set against every outfield player
    peers = pool[pool["role"].eq(role)] if role != "Unknown" and role in set(pool["role"]) else pool
    role_label = "outfield player" if role == "Unknown" else role.lower()
    peers = peers.reset_index(drop=True).set_index(["player", "team"])
    if (player, team) not in peers.index:
        return None
    shares, raws = _percentiles(peers, (player, team), PLAYER_SLICES)
    totals = league.players.set_index(["player", "team"]).loc[(player, team)]
    fig = plt.figure(figsize=SIZE, facecolor=base.BG)
    _card_header(
        fig,
        league.colour(team),
        player.upper(),
        f"{team} · {'position not listed' if role == 'Unknown' else role} · {int(totals['matches'])} matches · {totals['minutes']:.0f} minutes"
        + (" · small sample" if totals["minutes"] < 450 else ""),
    )
    badges = [f"{round(100 * s):d}" if s is not None else "" for s in shares]
    ranks = [
        f"{round(100 * s):d}{_ordinal(round(100 * s))}" if s is not None else "—" for s in shares
    ]
    ax = fig.add_axes([0.285, 0.075, 0.43, 0.66], projection="polar")
    pizza(ax, PLAYER_SLICES, shares, raws, badges, centre=(league, team))
    _figure_pairs(
        fig,
        [
            ("GOALS", f"{totals['goals']:.0f}"),
            ("ASSISTS", f"{totals['assists']:.0f}"),
            ("KEY PASSES", f"{totals['key_passes']:.0f}"),
        ],
    )
    _legend_column(fig, 0.03, 0.2)
    _findings(fig, PLAYER_SLICES, shares, raws, ranks, 0.79)
    _footnote(
        fig,
        f"Slice length = share of the {len(peers)} {role_label}s with {minimum_minutes:.0f}+ minutes this player beats · per 90 · circle = percentile · dashed ring = median",
        0.79,
        46,
    )
    safe = re.sub(r"[^\w\-]+", "_", player).strip("_")
    return finish_card(fig, league, "players", f"profile_{team.replace(' ', '_')}__{safe}.png")
