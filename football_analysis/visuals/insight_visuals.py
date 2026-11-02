"""Editorial charts with explicit observations, denominators and small-sample rules."""

from pathlib import Path
import json
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from football_analysis.metrics.match_insights import build_insights, stage_counts, flags, numeric
from football_analysis.metrics.frame_values import short_name as _short_name
from football_analysis.visuals.visualization_components import (
    C_HOME,
    C_AWAY,
    IS_LIGHT_THEME,
    BG_DARK,
    TEXT_MAIN,
    TEXT_DIM,
)
from football_analysis.visuals.modern_chart_components import paired_dots, stage_flow

BRAND = "#137F82" if IS_LIGHT_THEME else "#61D7CC"
BG = BG_DARK
FG = TEXT_MAIN
MUTED = TEXT_DIM


def counted(n, singular, plural=None):
    """ "1 entry", "2 entries": the noun agrees with the number printed beside it."""
    n = int(n)
    return f"{n} {singular if n == 1 else plural or singular + 's'}"


def share_limits(axes, axis="x"):
    """Give every panel of a two-team chart the same limits on one axis.

    Each panel used to size its own axis to its own team, so a side with a
    third of the values drew bars as long as the other's: the eye compared
    lengths and read the wrong answer. Only the value axis is shared; a
    category axis keeps its own order.
    """
    axes = list(axes)
    read = (lambda a: a.get_xlim()) if axis == "x" else (lambda a: a.get_ylim())
    lows, highs = zip(*(sorted(read(a)) for a in axes))
    low, high = min(lows), max(highs)
    for a in axes:
        (a.set_xlim if axis == "x" else a.set_ylim)(low, high)


RED_CARD = "#EF4444"


def bench_contributions(a, names):
    """One row per substitute: when he came on, for whom, and what he produced.

    Juventus beat Napoli with a goal and two assists from players who started
    on the bench, and the substitution board showed one window at 82' with a
    negative xG axis -- half-time changes have no "before" inside the period,
    so every one of them was skipped.
    """
    cols = ["team_id", "player", "replaced", "minute", "minutes", "shots", "xG",
            "goals", "assists", "xA", "key_passes"]
    if a is None or a.empty or "type" not in a:
        return pd.DataFrame(columns=cols)
    on = a[a.type.eq("SubstitutionOn")].sort_values("_clock_seconds")
    off = a[a.type.eq("SubstitutionOff")].sort_values("_clock_seconds")
    end = max(90.0, float(numeric(a, "minute").max() or 90))
    shots = a[flags(a, "is_shot") & ~flags(a, "is_own_goal")]
    goals = shots[flags(shots, "is_goal")]
    assist = shots.get("assist_player", pd.Series("", index=shots.index)).astype(str)
    rows = []
    used_off = set()
    for idx, sub in on.iterrows():
        tid, name = sub.team_id, str(sub.player)
        same = off[off.team_id.eq(tid) & (off._clock_seconds - sub._clock_seconds).abs().le(5)]
        same = same[~same.index.isin(used_off)]
        replaced = ""
        if len(same):
            used_off.add(same.index[0])
            replaced = str(same.iloc[0].player)
        later_off = off[off.team_id.eq(tid) & off.player.astype(str).eq(name)
                        & off._clock_seconds.gt(sub._clock_seconds)]
        left = float(later_off.minute.iloc[0]) if len(later_off) else end
        mine = shots[shots.team_id.eq(tid) & shots.player.astype(str).eq(name)]
        fed = shots[shots.team_id.eq(tid) & assist.eq(name)]
        rows.append(dict(
            team_id=tid, player=name, replaced=replaced, minute=float(sub.minute),
            minutes=max(left - float(sub.minute), 0.0), shots=len(mine),
            xG=float(numeric(mine, "xG").sum()), goals=int(flags(mine, "is_goal").sum()),
            assists=int(goals.team_id.eq(tid).mul(assist.reindex(goals.index).eq(name)).sum()),
            xA=float(numeric(fed, "xG").sum()),
            key_passes=int(flags(a[a.team_id.eq(tid) & a.player.astype(str).eq(name)],
                                 "is_key_pass").sum()),
            surname=_short_name(name), out_surname=_short_name(replaced),
        ))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


def _bench_impact_board(charts, a, bench):
    """Substitutions on a match timeline, with what each substitute produced."""

    if bench is None or bench.empty:
        return False
    end = max(90.0, float(numeric(a, "minute").max() or 90))
    fig, axes = charts.figure(
        "Substitutions and bench impact",
        "Every change on the match clock, with goals and sendings-off · what each substitute produced",
        columns=1,
        height=8.6,
    )
    axes[0].remove()
    shots = a[flags(a, "is_shot")]
    goals = shots[flags(shots, "is_goal")]
    cards = a[a.type.eq("Card")]
    q = cards.get("qualifier_names", pd.Series("", index=cards.index)).astype(str)
    reds = cards[q.str.contains("SecondYellow") | q.str.contains(r"['\"]Red['\"]", regex=True)]
    summaries = []
    for row, (tid, name) in enumerate(charts.names.items()):
        top = 0.745 - row * 0.315
        color = charts.colors[tid]
        line = fig.add_axes([0.06, top - 0.255, 0.56, 0.255], facecolor=BG)
        line.set_xlim(-2, end + 3)
        line.set_ylim(-1.75, 1.75)
        line.axis("off")
        line.text(-2, 1.72, name.upper(), color=color, size=12, weight="bold", va="top")
        line.plot([0, end], [0, 0], color=MUTED, lw=1.4, alpha=0.6, zorder=1)
        line.axvline(45, color=MUTED, lw=0.8, ls=(0, (3, 3)), alpha=0.5, zorder=0)
        for m in (0, 15, 30, 45, 60, 75, 90):
            line.text(m, -1.72, f"{m}'", color=MUTED, size=8, ha="center", va="bottom")
        mine = bench[bench.team_id.eq(tid)].sort_values("minute")
        # Changes made together share one tick and one label block, stacked;
        # neighbours alternate above and below the line so they cannot collide.
        clusters = []
        for _, sub in mine.iterrows():
            if clusters and sub.minute - clusters[-1][0] <= 1:
                clusters[-1][1].append(sub)
            else:
                clusters.append((sub.minute, [sub]))
        # Each label claims a horizontal span (about a minute of clock per
        # character at this size); a change goes to the first slot -- above or
        # below the line, near or far -- where its span collides with nothing.
        slots = {(1, 0): [], (-1, 0): [], (1, 1): [], (-1, 1): []}
        lo, hi = -2, end + 3
        for minute, group in clusters:
            rows_ = [f"↑ {s.surname}   ↓ {s.out_surname}" for s in group]
            rows_[0] = f"{minute:.0f}'  " + rows_[0]
            half = 0.47 * max(len(r) for r in rows_)
            x = min(max(minute, lo + half), hi - half)
            span = (x - half - 1, x + half + 1)
            slot = next(
                (k for k in [(1, 0), (-1, 0), (1, 1), (-1, 1)]
                 if all(span[1] < a or span[0] > b for a, b in slots[k])),
                (1, 1),
            )
            slots[slot].append(span)
            sign, level = slot
            y = sign * (0.42 + 0.42 * level)
            line.plot([minute, minute], [0, y - sign * 0.06], color=color, lw=1.1, zorder=2)
            line.scatter([minute], [0], s=46, color=color, edgecolor=BG, zorder=4)
            line.text(x, y, "\n".join(rows_), color=FG, size=8.2, ha="center",
                      va="bottom" if sign > 0 else "top", linespacing=1.25, zorder=5)
        # Goals and sendings-off sit on the line itself, named just under it, so
        # they never contend with the substitution labels for the same space.
        for _, g in goals[goals.team_id.eq(tid)].iterrows():
            line.scatter([g.minute], [0], marker="*", s=260, color="#FACC15",
                         edgecolor=BG, linewidth=0.8, zorder=6)
            line.text(g.minute, 0.12, _short_name(g.player), color="#FACC15", size=7.5,
                      ha="center", va="bottom", weight="bold", zorder=6,
                      bbox=dict(facecolor=BG, edgecolor="none", pad=0.6))
        for _, r in reds[reds.team_id.eq(tid)].iterrows():
            line.scatter([r.minute], [0], marker="s", s=70, color=RED_CARD, zorder=6)
            line.text(r.minute, -0.13, f"{_short_name(r.player)} sent off {r.minute:.0f}'",
                      color=RED_CARD, size=7.5, ha="center", va="top", weight="bold", zorder=6,
                      bbox=dict(facecolor=BG, edgecolor="none", pad=0.6))

        table = fig.add_axes([0.655, top - 0.255, 0.30, 0.255], facecolor=BG)
        table.axis("off")
        table.text(0, 1.0, "BENCH OUTPUT", color=MUTED, size=8.5, weight="bold", va="top")
        if mine.empty:
            table.text(0, 0.78, "No substitutions", color=MUTED, size=9, va="top")
        for j, (_, s) in enumerate(mine.iterrows()):
            y = 0.80 - j * (0.78 / max(len(mine), 5))
            decisive = s.goals or s.assists
            table.text(0, y, f"{s.surname}", color=color if decisive else FG, size=9,
                       weight="bold" if decisive else "normal", va="top")
            table.text(0.36, y, f"{s.minutes:.0f} min", color=MUTED, size=8.5, va="top")
            stat = f"{s.goals}G {s.assists}A · {s.xG:.2f} xG · {s.xA:.2f} xA"
            table.text(1.0, y, stat, color=FG if decisive else MUTED, size=8.5, ha="right",
                       va="top")
        tot = mine[["goals", "assists", "xG", "xA"]].sum() if len(mine) else None
        if tot is not None:
            summaries.append(
                f"{name} made {len(mine)} changes; the substitutes produced "
                f"{int(tot.goals)} goals, {int(tot.assists)} assists, {tot.xG:.2f} xG "
                f"and {tot.xA:.2f} xA."
            )
    charts.save(
        fig,
        "61_substitution_windows.png",
        "Substitutions and bench impact",
        " ".join(summaries),
        "Changes on the match clock; stars are goals, red squares sendings-off. Substitute output "
        "counts only what the player did after coming on; xA is the xG of shots he set up.",
        len(bench),
    )
    return True


def _butterfly_frame(charts, title, subtitle, height=8.0):
    """A header page with one bare axis spanning it, for side-against-side boards."""
    fig, axes = charts.figure(title, subtitle, columns=1, height=height)
    axes[0].remove()
    ax = fig.add_axes([0.06, 0.17, 0.88, 0.60])
    ax.axis("off")
    return fig, ax


def _team_heads(ax, charts, y, xs=(-0.32, 0.32)):
    for x, (tid, name) in zip(xs, charts.names.items()):
        ax.text(x, y, name.upper(), color=charts.colors[tid], ha="center", size=13, weight="bold")


def _funnel_board(charts, p):
    """Possession to shot on target as one butterfly, with the rate at each step.

    The two line charts plotted stages as a continuous curve in two panels;
    the conversion at each step -- where attacks died -- had to be worked out.
    """
    stages = ["Possessions", "Reached final third", "Then box", "Then shot", "On target"]
    counts = {tid: stage_counts(p, tid) for tid in charts.names}
    fig, ax = _butterfly_frame(
        charts,
        "From possession to a shot on target",
        "Each bar is what survived the stage above it · the % between bars is the conversion at that step",
    )
    top = max(max(c) for c in counts.values()) or 1
    for i, label in enumerate(stages):
        y = len(stages) - i
        ax.text(0, y, label, ha="center", va="center", color=FG, size=11, weight="bold")
        for (tid, _), sign in zip(charts.names.items(), (-1, 1)):
            n = counts[tid][i]
            w = 0.40 * n / top
            ax.barh(y, sign * w, left=sign * 0.11, height=0.56, color=charts.colors[tid], alpha=0.92)
            ax.text(sign * (0.11 + w + 0.012), y, str(n), va="center",
                    ha="left" if sign > 0 else "right", color=FG, size=12, weight="bold")
            if i:
                prev = counts[tid][i - 1]
                rate = 100 * n / prev if prev else 0
                ax.text(sign * 0.31, y + 0.5, f"↓ {rate:.0f}%", ha="center", va="center",
                        color=charts.colors[tid], size=10)
    ax.set_xlim(-0.64, 0.64)
    ax.set_ylim(0.4, len(stages) + 0.9)
    _team_heads(ax, charts, len(stages) + 0.65)
    readings = [
        f"{name}: possessions at successive stages: " + ", ".join(map(str, counts[tid])) + "."
        for tid, name in charts.names.items()
    ]
    charts.save(
        fig,
        "55_possession_funnel.png",
        "Which possessions reached the box and a shot",
        "This identifies where attacks stop: the largest drop between two stages is the access "
        "problem to review, rather than possession volume alone. " + " ".join(readings),
        "Same possession at every stage; the arrows give the share that survived each step.",
        len(p),
    )


def _regain_speed_board(charts, regains):
    """Every regain split by how quickly the ball reached the final third, one bar a side."""
    bands = ["0–5 s", "6–12 s", "Over 12 s", "Never reached"]
    shades = [1.0, 0.72, 0.45, 0.18]
    fig, ax = _butterfly_frame(
        charts,
        "Regains and time to reach the final third",
        "Every regain, split by how quickly the ball then reached the final third",
        height=6.4,
    )
    ax.set_position([0.17, 0.22, 0.68, 0.48])
    for row, (tid, name) in enumerate(charts.names.items()):
        times = regains.loc[regains.team_id.eq(tid), "seconds_to_third"]
        parts = [int(times.le(5).sum()), int((times.gt(5) & times.le(12)).sum()),
                 int(times.gt(12).sum()), int(times.isna().sum())]
        total = max(sum(parts), 1)
        y, left = 1 - row, 0.0
        for n, alpha in zip(parts, shades):
            w = n / total
            ax.barh(y, w, left=left, height=0.5, color=charts.colors[tid], alpha=alpha,
                    edgecolor=BG, lw=2)
            if w > 0.06:
                ax.text(left + w / 2, y, f"{n}\n{100 * w:.0f}%", ha="center", va="center",
                        color=BG if alpha > 0.6 else FG, size=10, weight="bold")
            left += w
        ax.text(-0.012, y, f"{name}\n{counted(sum(parts), 'regain')}", ha="right", va="center",
                color=charts.colors[tid], size=11, weight="bold")
        ax.text(1.012, y, f"{100 * (parts[0] + parts[1]) / total:.0f}%\nwithin 12 s",
                ha="left", va="center", color=FG, size=10)
    for k, (label, alpha) in enumerate(zip(bands, shades)):
        ax.barh(1.48, 0.03, left=(k + 0.5) / 4 - 0.075, height=0.08, color=FG, alpha=alpha)
        ax.text((k + 0.5) / 4 - 0.035, 1.48, label, va="center", color=MUTED, size=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.7)
    charts.save(
        fig,
        "54_regain_speed.png",
        "Regain progression time",
        "Counts include regains that never reached the final third.",
        "One bar per side out of 100%; zero seconds means regained inside the final third.",
        len(regains),
    )


def _loss_cost_board(charts, loss):
    """Every loss faint; the ones the opponent punished large, with totals by third."""
    fig, axes = charts.figure(
        "Where possession was lost and what it cost",
        "Every loss is a faint dot; the ones the opponent punished within 12 seconds are drawn large",
        columns=2,
        height=7.4,
    )
    summaries = []
    for ax, (tid, name) in zip(axes, charts.names.items()):
        colour = charts.colors[tid]
        g = loss[loss.team_id.eq(tid)]
        ax.set_xlim(-2, 107)
        ax.set_ylim(-16, 72)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 105, 68, fill=False, ec=MUTED, lw=0.8))
        for x in (35, 70):
            ax.plot([x, x], [0, 68], color=MUTED, lw=0.8, ls=(0, (4, 4)), alpha=0.6)
        calm = g[~(g.third_within_12 | g.box_within_12 | g.shot_within_12)]
        ax.scatter(calm.x * 1.05, calm.y * 0.68, s=12, color=MUTED, alpha=0.35, lw=0)
        tiers = [
            (g.third_within_12 & ~g.box_within_12 & ~g.shot_within_12, 55, "o"),
            (g.box_within_12 & ~g.shot_within_12, 95, "s"),
            (g.shot_within_12, 260, "*"),
        ]
        for mask, size, marker in tiers:
            h = g[mask]
            ax.scatter(h.x * 1.05, h.y * 0.68, s=size, marker=marker, color=colour,
                       edgecolor=FG, lw=0.6, zorder=3)
        for lo, hi, zone in ((0, 33.3, "Own third"), (33.3, 66.7, "Middle"), (66.7, 101, "Final third")):
            part = g[(g.x >= lo) & (g.x < hi)]
            punished = int(part.shot_within_12.sum())
            centre = (lo + min(hi, 100)) / 2 * 1.05
            ax.text(centre, -4.5, f"{zone}: {len(part)} lost", ha="center", color=FG, size=9.5)
            ax.text(centre, -10, f"{punished} led to a shot", ha="center", size=9,
                    color=colour if punished else MUTED, weight="bold" if punished else "normal")
        ax.set_title(
            f"{name} · {len(g)} losses · {int(g.shot_within_12.sum())} punished with a shot",
            color=colour, size=11.5, weight="bold", loc="left",
        )
        summaries.append(
            f"{name}: {len(g)} open-play losses; {int(g.third_within_12.sum())} followed by final-third "
            f"presence, {int(g.box_within_12.sum())} by box presence and {int(g.shot_within_12.sum())} "
            f"by a shot within 12 seconds, totaling {g.xG_conceded.sum():.2f} xG."
        )
    fig.text(
        0.5, 0.15,
        "● opponent reached the final third     ■ the box     ★ a shot     grey: nothing followed",
        ha="center", color=MUTED, size=9.5,
    )
    charts.save(
        fig,
        "60_loss_consequences.png",
        "Loss locations and immediate opponent output",
        " ".join(summaries),
        "Attacking direction →. Each loss is plotted once at its most advanced outcome within 12 "
        "seconds; coordinates are event-inferred.",
        len(loss),
    )


_SMALL_WORDS = {"a", "an", "and", "the", "to", "of", "in", "on", "by", "per", "v", "vs", "at"}


def _headline(title):
    """Title case, as on every other page of the package.

    These titles were sentence case ("Shot volume and average quality") under
    the same strip that sets every other page in title case, which was half of
    what made the last eleven pages look like a different publication.
    """
    words = title.split(" ")
    out = []
    for i, word in enumerate(words):
        if i and word.lower() in _SMALL_WORDS:
            out.append(word.lower())
        else:
            out.append(
                "-".join(
                    part[:1].upper() + part[1:] if part[:1].islower() else part
                    for part in word.split("-")
                )
            )
    return " ".join(out)


class Charts:
    def __init__(self, out, info):
        self.out = Path(out)
        self.info = info
        self.paths = []
        self.contracts = {}
        self.skipped = []
        self.names = {info["home_id"]: info["home_name"], info["away_id"]: info["away_name"]}
        from football_analysis.visuals.visual_redesign_full import (
            _resolve_fixture_colors,
            lift_to_floor,
        )

        home, away = _resolve_fixture_colors(info)
        self.colors = {info["home_id"]: lift_to_floor(home), info["away_id"]: lift_to_floor(away)}

    def figure(self, title, subtitle, columns=1, height=7):
        """One figure, wearing the same head as every other page in the package.

        These charts carried their own header — a teal "MOSTAFA SAAD / MATCH
        STUDY" line, no crests, no scoreline, its own type scale — so a reader
        turning from page 41 to page 42 of one PDF crossed into what looked
        like a different publication. The pages either side of that boundary
        describe the same match.

        The shared strip is the one already on the other forty: badge, section
        label, title, subtitle, both crests and the score. It occupies the top
        of the figure, so the axes start lower than they used to.
        """
        fig, axes = plt.subplots(1, columns, figsize=(14, height), squeeze=False)
        fig.patch.set_facecolor(BG)
        # The shared strip ends at 0.865, so anything below 0.79 leaves a
        # visible band of empty page between the head and the first axis.
        fig.subplots_adjust(left=0.09, right=0.955, bottom=0.17, top=0.79, wspace=0.30)
        from football_analysis.visuals.visual_redesign_preview import amoled_header

        amoled_header(fig, _headline(title), subtitle)
        for ax in axes.flat:
            ax.set_facecolor(BG)
            ax.tick_params(colors=MUTED, labelsize=10)
            ax.xaxis.label.set_color(FG)
            ax.yaxis.label.set_color(FG)
            for spine in ax.spines.values():
                spine.set_color(MUTED)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.12, color=MUTED)
            ax.set_axisbelow(True)
        return fig, axes[0]

    def save(self, fig, filename, title, reading, method, n):
        # The method note alone, and the package's own signature beside it: the
        # "PSG v Barcelona |" prefix repeated the header's scoreline and these
        # were the only pages without the signature every other page carries.
        fig.text(0.055, 0.055, textwrap.fill(method, 125), color=MUTED, size=9, va="bottom")
        fig.text(
            0.945, 0.055, "FULL VISUAL REDESIGN · REAL MATCH DATA",
            color=MUTED, size=8, ha="right", va="bottom",
        )
        path = self.out / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor=BG)
        plt.close(fig)
        self.paths.append(path)
        self.contracts[path.name] = {
            "title": title,
            "reading": reading,
            "method": method,
            "n": int(n),
        }
        return path

    def scatter(self, frame, x, y, filename, title, xlabel, ylabel, *, player=False, size=None):
        required = [x, y, "team_id"]
        data = (
            frame.dropna(subset=required).copy()
            if all(k in frame for k in required)
            else pd.DataFrame()
        )
        if player and not data.empty:
            data = data[
                (data.minutes >= 30) & ~data.role.astype(str).str.upper().isin(["GK", "GOALKEEPER"])
            ]
        if len(data) < (8 if player else 5):
            if player and len(data):
                fig, axes = self.figure(
                    title,
                    f"Small sample: {len(data)} eligible players; raw observations replace scatter quadrants",
                )
                ax = axes[0]
                ax.axis("off")
                ordered = data.sort_values(y, ascending=False)
                for i, row in enumerate(ordered.itertuples()):
                    ax.text(
                        0,
                        1 - i / max(len(data), 8),
                        f"{row.player}  |  {getattr(row, x):.2f} {xlabel}  |  {getattr(row, y):.2f} {ylabel}",
                        transform=ax.transAxes,
                        color=self.colors[row.team_id],
                        size=11,
                    )
                return self.save(
                    fig,
                    filename,
                    title,
                    f"Only {len(data)} players meet the minimum minutes and metric availability. Read individual values; no quadrant ranking is drawn.",
                    "Minimum 30 minutes; undefined ratios omitted",
                    len(data),
                )
            self.skipped.append(
                {
                    "chart": filename,
                    "reason": f"Only {len(data)} valid observation{'' if len(data) == 1 else 's'}",
                }
            )
            return
        # Counts make exact ties common -- one shot at 0.05 xG, two progressions
        # and no xA -- and a tied marker sat whole under the other: Rodri under
        # Marquinhos, Bernal under Pacho, Yamal under Doué. Tied points are
        # spread sideways by a sliver of the axis so every player shows.
        if len(data) > 1:
            key = data[x].round(6).astype(str) + "|" + data[y].round(6).astype(str)
            rank = data.groupby(key).cumcount()
            size_of = data.groupby(key)[x].transform("size")
            span = float(data[x].max() - data[x].min()) or 1.0
            data[x] = data[x] + (rank - (size_of - 1) / 2) * span * 0.018
        # "1 observations" reached a shipped article; a single eligible point
        # is rare but legal, so the noun agrees with the count.
        observed = len(data)
        fig, axes = self.figure(
            title,
            f"{observed} observation{'' if observed == 1 else 's'}  |  "
            + (
                "Players with at least 30 minutes; match totals"
                if player
                else "One point per possession; both teams share the same axes"
            ),
            height=8,
        )
        fig.subplots_adjust(bottom=0.23)
        ax = axes[0]
        for marker, (tid, group) in zip(["o", "s"], data.groupby("team_id")):
            sizes = 35 + 90 * group[size] / max(data[size].max(), 1) if size else 52
            ax.scatter(
                group[x],
                group[y],
                s=sizes,
                c=self.colors[tid],
                label=self.names[tid],
                marker=marker,
                alpha=0.80,
                edgecolors=FG,
                linewidths=0.5,
            )
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.set_xlim(left=min(0, data[x].min() * 0.95))
        ax.set_ylim(bottom=min(0, data[y].min() * 0.95))
        if player:
            ax.margins(x=0.15, y=0.25)
            ax.set_xlim(left=-max(data[x].max(), 1) * 0.04)
            ax.set_ylim(bottom=-max(data[y].max(), 0.01) * 0.08)
            ax.axvline(data[x].median(), color=MUTED, ls=":", lw=1)
            ax.axhline(data[y].median(), color=MUTED, ls=":", lw=1)
            from football_analysis.visuals.scatter_labels import label_players

            label_players(ax, data, x, y, color=FG, background=BG, fontsize=9.5, short=True)
        ax.legend(
            facecolor=BG,
            labelcolor=FG,
            edgecolor="none",
            fontsize=9,
            # Just above the plot, under the header strip rather than inside it.
            loc="lower right",
            bbox_to_anchor=(1, 1.01),
            ncol=2,
        )
        method = (
            (
                "Dotted lines: eligible-player medians. Bubble area: " + size
                if size
                else "Dotted lines: eligible-player medians."
            )
            if player
            else "No fitted trend or causal interpretation."
        )
        if y == "xA":
            method += " Inferred xA = shot xG credited to a prior successful key pass in the same possession (≤15 seconds)."
        if y == "positive_xT":
            method += " Positive xT = added local zone threat from successful passes/carries; not goals or net value."
        if y == "xG_per_shot":
            method += " xG per shot = total shot xG / attempts; higher means better average pre-shot chance quality."
        if y == "seconds_to_third":
            method += " Time runs from regain to first final-third presence; regains without arrival remain in the source table."
        med = data.groupby("team_id")[[x, y]].median()
        # This paragraph is what the article and the PDF print under the figure.
        # It was written in Arabic inside an otherwise English document, and
        # what it said was two medians: "Tottenham يميل إلى progressive passes
        # 1.00 وinferred xa 0.00". A scatter's argument is which quadrant the
        # points fall in and whether the two axes agree, so that is what it
        # argues now.
        parts = []
        if len(med) >= 2:
            lead_x = self.names[med[x].idxmax()]
            lead_y = self.names[med[y].idxmax()]
            if lead_x == lead_y:
                parts.append(
                    f"{lead_x} led on both axes: its typical player did more of the "
                    f"{xlabel.lower()} and got more out of it. Volume and value agreeing "
                    f"is the simple case, and it puts the difference between the sides in "
                    f"the players rather than in the route they were given."
                )
            else:
                parts.append(
                    f"{lead_x} led {xlabel.lower()} and {lead_y} led {ylabel.lower()}, so the "
                    f"side doing more of it was not the side getting more from it. Doing a "
                    f"thing often and doing it profitably are separate claims, and only the "
                    f"second one is on the vertical axis."
                )
        if player and len(data):
            both = data[(data[x] > data[x].median()) & (data[y] > data[y].median())]
            if len(both):
                picked = both.sort_values([y, x], ascending=False).head(3)
                names = ", ".join(str(row.player) for row in picked.itertuples())
                parts.append(
                    f"Above both medians: {names}. Those are the players the two measures "
                    f"agree on. Anyone high on one axis alone was doing half of what the "
                    f"chart asks about, which is the half worth checking on video."
                )
            else:
                parts.append(
                    "Nobody cleared both medians, so no player on either side combined the "
                    "volume with the value. The chart is describing a match in which the "
                    "work and the return sat with different people."
                )
        parts.append(
            "Minutes and role move a player across this chart on their own, so read a "
            "position through both."
            if player
            else "Follow the sequence behind a point before reading a tactical cause into it."
        )
        reading = " ".join(parts)
        return self.save(fig, filename, title, reading, method, len(data))


def role_group(role):
    role = str(role).upper()
    if role in ["SUB", "SUBSTITUTE", "UNKNOWN", "NAN", "NONE"]:
        return "Unknown"
    if role in ["GK", "GOALKEEPER"]:
        return "Goalkeeper"
    if role.startswith("D") and not role.startswith("DM"):
        return "Defender"
    # AML / AMR are the wide forwards of a 4-3-3 written in a 4-2-3-1 grid:
    # Yamal and Gordon at Paris were filed as midfielders and ranked against
    # Pedri and Bernal, while Doué on the same wing as FWR was a forward.
    if role.startswith("F") or role in ["ST", "SS", "STRIKER", "AML", "AMR"]:
        return "Forward"
    if role.startswith(("M", "AM", "DM")):
        return "Midfielder"
    return "Unknown"



def _history_rows(directory, match, events=None):
    """One row per side of a stored fixture for the history boards."""
    from football_analysis.reports.match_report import compute_ppda_both

    directory = Path(directory)
    tm = pd.read_csv(directory / "team_advanced_metrics.csv")
    xg = pd.read_csv(directory / "xg.csv")
    if events is None:
        events = pd.read_csv(directory / "events.csv")
    ppda = compute_ppda_both(match, events)
    rows = []
    for side in ["home", "away"]:
        name = match[side + "_name"]
        t = tm[tm.side.eq(side)]
        shot = xg[xg.team.eq(name)]
        if t.empty or shot.empty:
            continue
        rows.append(
            dict(
                team_id=match[side + "_id"],
                team=name,
                date=match.get("date", ""),
                field_tilt=float(t.iloc[0].field_tilt),
                xG=float(shot.iloc[0].xG),
                ppda=ppda[side].get("ppda"),
                risk=float(t.iloc[0].rest_defence_vulnerability),
            )
        )
    return rows


def _mark_today(ax, charts, hist, today, x, y):
    """The two sides' other matches small, this one large and named."""
    for tid, name in charts.names.items():
        g = hist[hist.team.eq(name)]
        if len(g):
            ax.scatter(
                g[x], g[y], color=charts.colors[tid], s=40, alpha=0.75,
                label=f"{name} · other matches",
            )
        t = today[today.team.eq(name)] if len(today) else today
        if len(t):
            ax.scatter(
                t[x], t[y], color=charts.colors[tid], s=150, edgecolor=FG, linewidth=1.4,
                zorder=5, label=f"{name} · this match",
            )
            ax.annotate(
                "this match", (float(t[x].iloc[0]), float(t[y].iloc[0])),
                xytext=(8, 8), textcoords="offset points", color=FG, size=8,
            )

from football_analysis.visuals.advanced_profiles import compact_profiles


def build_insight_visuals(events, players, info, out):
    # Remove the old post-49 names from copied staging packages. Otherwise a
    # rebuild would contain both the contiguous series and stale 54–65 files.
    for old_number in range(54, 66):
        for legacy in Path(out).glob(f"{old_number}_*.png"):
            legacy.unlink(missing_ok=True)
    data = build_insights(events, players, info)
    charts = Charts(out, info)
    p = data["possessions"]
    people = data["players"]
    charts.scatter(
        people,
        "progressions",
        "xA",
        "50_player_progression_creation.png",
        "Progression and chance creation",
        "Progressive passes + explicit carries",
        "Inferred xA",
        player=True,
        size="touches",
    )
    charts.scatter(
        people,
        "touches",
        "positive_xT",
        "51_player_involvement_value.png",
        "Involvement and added threat",
        "Recorded touches",
        "Positive successful-movement xT",
        player=True,
    )
    charts.scatter(
        people[people.shots > 0],
        "shots",
        "xG_per_shot",
        "52_player_shot_quality.png",
        "Shot volume and average quality",
        "Shots",
        "xG per shot",
        player=True,
    )
    fig, axes = charts.figure(
        "Possession value by duration",
        "Each box shows the distribution of positive xT; counts retain the number of possessions in each duration band.",
        columns=2,
    )
    for ax, (tid, name) in zip(axes, charts.names.items()):
        own = p[p.team_id.eq(tid)]
        bands = [
            own.duration.le(10),
            own.duration.gt(10) & own.duration.le(20),
            own.duration.gt(20) & own.duration.le(40),
            own.duration.gt(40),
        ]
        groups = [own.loc[mask, "positive_xT"].dropna().to_numpy() for mask in bands]
        for i, values in enumerate(groups):
            if len(values) >= 4:
                ax.boxplot(
                    [values],
                    positions=[i],
                    widths=0.45,
                    patch_artist=True,
                    boxprops={"facecolor": charts.colors[tid], "alpha": 0.5},
                    medianprops={"color": FG},
                    whiskerprops={"color": MUTED},
                    capprops={"color": MUTED},
                    flierprops={"marker": ".", "markeredgecolor": charts.colors[tid]},
                )
            elif len(values):
                ax.scatter([i] * len(values), values, color=charts.colors[tid])
        ax.set_xticks(
            range(4),
            [
                f"{label}\nn={len(v)}"
                for label, v in zip(["0–10s", "10–20s", "20–40s", "40s+"], groups)
            ],
        )
        # Say which possessions the bands hold: only those with a pass or carry
        # carry an xT figure, so the counts summed to 88 beside the funnel's 123.
        valued = sum(len(v) for v in groups)
        ax.set_title(
            f"{name} · {valued} of {len(own)} possessions had a pass or carry",
            color=FG,
        )
        ax.set_ylabel("Positive xT")
        ax.set_xlim(-0.6, 3.6)
    share_limits(axes, "y")
    charts.save(
        fig,
        "53_possession_speed_value.png",
        "Possession value by duration",
        "Compare typical value and its spread within each duration group.",
        "Box = middle 50%; line = median. Groups below four observations show individual points.",
        len(p),
    )
    regains = p[p.is_regain]
    _regain_speed_board(charts, regains)
    _funnel_board(charts, p)
    routes = data["routes"]
    fig, axes = charts.figure(
        "Box-entry routes and their next outcome",
        "A shot must follow the entry in the same possession; multiple entries can precede one shot",
        columns=2,
    )
    readings = []
    for ax, (tid, name) in zip(axes, charts.names.items()):
        g = routes[routes.team_id.eq(tid)]
        counts = (
            g.groupby("lane")
            .shot_followed.agg(["size", "sum"])
            .reindex(["Left", "Centre", "Right"], fill_value=0)
        )
        positions = np.arange(len(counts))
        for i, (lane, row) in enumerate(counts.iterrows()):
            total = float(row["size"])
            shot = float(row["sum"])
            rate = 100 * shot / total if total else 0
            ax.plot([0, total], [i, i], color=MUTED, lw=3, alpha=0.45)
            ax.scatter(total, i, s=90, color=MUTED, zorder=3)
            ax.scatter(shot, i, s=90, marker="s", color=charts.colors[tid], zorder=4)
            ax.text(
                total + 0.35,
                i + 0.13,
                counted(total, "entry", "entries"),
                color=FG,
                size=9,
                va="bottom",
            )
            ax.text(
                shot + 0.35,
                i - 0.15,
                f"{counted(shot, 'shot')} · {rate:.0f}%",
                color=FG,
                size=9,
                va="top",
            )
        ax.set_yticks(positions, counts.index, color=FG)
        ax.set_ylim(2.55, -0.55)
        ax.set_xlim(0, max(float(counts["size"].max()) * 1.38, 1))
        ax.set_title(name, color=FG)
        ax.set_xlabel("Entries and entries followed by a shot")
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=MUTED,
                    markersize=8,
                    label="All entries",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="none",
                    markerfacecolor=charts.colors[tid],
                    markersize=8,
                    label="Shot followed",
                ),
            ],
            facecolor=BG,
            labelcolor=FG,
            edgecolor="none",
            fontsize=9,
            loc="upper right",
        )
        readings.append(
            f"{name}: {counted(len(g), 'entry', 'entries')}, {int(g.shot_followed.sum())} followed by a shot in that possession."
        )
    share_limits(axes, "x")
    charts.save(
        fig,
        "56_entry_routes.png",
        "Entry route and subsequent shot",
        "This compares the quality of access, not only the number of entries. More entries without more following shots means access did not become sustained threat. "
        + " ".join(readings),
        "The lane identifies entry origin; a following shot is descriptive, not proof of cause.",
        len(routes),
    )
    # 57_game_state_rates was folded into the Game-State Output card, which
    # already listed the totals for the same three states. Two pages showing
    # the same split with different denominators made a reader hold both open
    # to answer one question; the rate now sits under the totals it qualifies.
    # Two charts were cut here rather than redrawn.
    #
    # 58_reception_map gave each side a scatter of every attacking-half
    # reception over a second panel of twelve arrows. Two hundred undifferentiated
    # dots on a blank rectangle is a texture, not a finding: nothing in it
    # separated a reception that mattered from one that did not, and the arrows
    # underneath repeated what the progression pages already show.
    #
    # 59_sequence_story drew the three largest chances as three pitches stacked
    # at a fifth of the page each, with the action list set beside them at 9pt.
    # At the size the layout allowed, neither the numbered dots nor the chain
    # text could be read, and the same three possessions are described in prose
    # in the article.
    loss = data["losses"]
    _loss_cost_board(charts, loss)
    # Equal, within-period windows, ending before another substitution cluster.
    a = data["actions"]
    subs = a[a.type.isin(["SubstitutionOn", "SubstitutionOff"])]
    windows = []
    for (period, clock), cluster in subs.groupby(["_period_order", "_clock_seconds"]):
        period_actions = a[a._period_order.eq(period)]
        clocks = sorted(subs[subs._period_order.eq(period)]._clock_seconds.unique())
        prev = max([t for t in clocks if t < clock] + [float(period_actions._clock_seconds.min())])
        nxt = min([t for t in clocks if t > clock] + [float(period_actions._clock_seconds.max())])
        duration = min(600, clock - prev, nxt - clock)
        if duration < 180:
            continue
        before = period_actions[
            period_actions._clock_seconds.between(clock - duration, clock, inclusive="left")
        ]
        after = period_actions[
            period_actions._clock_seconds.between(clock, clock + duration, inclusive="left")
        ]
        confounds = period_actions[
            period_actions._clock_seconds.between(clock - duration, clock + duration)
            & (flags(period_actions, "is_goal") | period_actions.type.eq("Card"))
        ]
        for tid in charts.names:
            for label, frame in [("Before", before), ("After", after)]:
                own = frame[frame.team_id.eq(tid)]
                windows.append(
                    dict(
                        team_id=tid,
                        minute=clock / 60,
                        window_minutes=duration / 60,
                        phase=label,
                        xG=numeric(own[flags(own, "is_shot")], "xG").sum(),
                        goals_or_cards=len(confounds),
                    )
                )
    window_frame = pd.DataFrame(windows)
    data["substitution_windows"] = window_frame
    bench = bench_contributions(a, charts.names)
    data["bench_contributions"] = bench
    if _bench_impact_board(charts, a, bench):
        pass
    elif len(windows):
        fig, axes = charts.figure(
            "Output around substitution windows",
            "Equal windows before and after each change \u00b7 a description of output, not a substitution effect",
            columns=2,
            # One change needs a short board, not a tall one with a lone row in it.
            height=6.2 if window_frame.minute.nunique() <= 2 else 7,
        )
        for ax, (tid, name) in zip(axes, charts.names.items()):
            g = window_frame[window_frame.team_id.eq(tid)]
            pivot = g.pivot(index="minute", columns="phase", values="xG")
            x = np.arange(len(pivot))
            detail = g.drop_duplicates("minute").set_index("minute")
            categories = [
                f"{t:.0f}' ±{detail.loc[t, 'window_minutes']:.1f}m"
                + (" *" if detail.loc[t, "goals_or_cards"] else "")
                for t in pivot.index
            ]
            if len(pivot) <= 2:
                yy = np.arange(len(pivot))
                for j, t in enumerate(pivot.index):
                    b = float(pivot.loc[t, "Before"])
                    a = float(pivot.loc[t, "After"])
                    ax.plot([b, a], [j, j], color=MUTED, lw=2, zorder=1)
                    ax.scatter(
                        [b], [j], s=80, color=MUTED, zorder=3, label="Before" if j == 0 else None
                    )
                    ax.scatter(
                        [a],
                        [j],
                        s=80,
                        color=charts.colors[tid],
                        marker="s",
                        zorder=3,
                        label="After" if j == 0 else None,
                    )
                    ax.text(b, j + 0.14, f"{b:.2f}", ha="center", color=FG, size=9)
                    ax.text(a, j - 0.20, f"{a:.2f}", ha="center", color=FG, size=9)
                ax.set_yticks(np.arange(len(categories)), categories, color=FG)
                ax.set_ylim(-0.7, max(len(categories) - 0.3, 0.7))
                ax.legend(facecolor=BG, labelcolor=FG, edgecolor="none", loc="upper right")
            else:
                paired_dots(
                    ax,
                    categories,
                    pivot.Before,
                    pivot.After,
                    [MUTED, charts.colors[tid]],
                    FG,
                    MUTED,
                    labels=("Before", "After"),
                )
                ax.legend(facecolor=BG, labelcolor=FG, edgecolor="none")
            ax.set_title(name, color=FG)
            ax.set_xlabel("xG in equal windows")
        share_limits(axes, "x")
        charts.save(
            fig,
            "61_substitution_windows.png",
            "Equal windows around substitution clusters",
            f"{window_frame.minute.nunique()} substitution clusters have equal observable windows of at least three minutes before and after. Compare the raw outputs alongside score-state changes; a change is not a measured substitution effect.",
            "* A goal or card occurs in the combined window. Nearby substitutions shorten the available window.",
            len(windows),
        )
    else:
        charts.skipped.append(
            {
                "chart": "61_substitution_windows.png",
                "reason": "No isolated substitution cluster has at least three minutes on each side within the same period.",
            }
        )
    # Historical comparisons require comparable, versioned snapshots; old files
    # with unknown models must not silently form a benchmark.
    from football_analysis.metrics.metric_registry import METRIC_VERSION

    history = []
    history_root = Path(out).resolve()
    root = next((p for p in history_root.parents if p.name == "output"), None)
    if root and info.get("competition") and info.get("xg_model_version"):
        for manifest_path in root.rglob("package_manifest.json"):
            if any(part.startswith(".") for part in manifest_path.relative_to(root).parts):
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("metric_version") != METRIC_VERSION
                    or manifest.get("fixture", {}).get("competition") != info["competition"]
                    or manifest.get("fixture", {}).get("xg_model_version")
                    != info["xg_model_version"]
                ):
                    continue
                history.extend(_history_rows(manifest_path.parent, manifest["fixture"]))
            except (ValueError, KeyError, OSError):
                continue
    hist = pd.DataFrame(history)
    if len(hist):
        hist = hist.drop_duplicates(["team", "date"])
    # This fixture, from the frames this render has just written. The history
    # walk can only find packages already published, and this one is either
    # absent or a version behind, so the coloured dots were the two sides'
    # other matches: PSG 4-2 Barcelona sat at no point on its own board.
    today = pd.DataFrame()
    try:
        today = pd.DataFrame(_history_rows(Path(out), info, events))
    except (ValueError, KeyError, OSError):
        pass
    if len(hist) and len(today):
        hist = hist[~(hist.date.eq(info.get("date", "")) & hist.team.isin(today.team))]
    data["history_comparisons"] = hist
    # Ten matches means at least twenty team observations, not ten points.
    if len(hist) >= 20:
        fig, axes = charts.figure(
            "Territory and chances across the competition",
            "Comparable metric version; one point per team-match, not the two teams of one fixture",
        )
        ax = axes[0]
        ax.scatter(hist.field_tilt, hist.xG, color=MUTED, alpha=0.45, s=28)
        _mark_today(ax, charts, hist, today, "field_tilt", "xG")
        ax.set_xlabel("Field tilt (%)")
        ax.set_ylabel("xG")
        ax.legend(facecolor=BG, labelcolor=FG)
        charts.save(
            fig,
            "66_history_territory_chances.png",
            "Territory and chance output across saved matches",
            f"{len(hist)} unique team-date observations use the same competition and metric version. Differences in opponents, score states and xG source still require checking before causal or forecasting conclusions.",
            "History is descriptive; input-model provenance is listed in each manifest.",
            len(hist),
        )
        valid = hist.dropna(subset=["ppda", "risk"])
        if len(valid) >= 20:
            fig, axes = charts.figure(
                "Pressing frequency and danger after losses",
                "Same competition and metric version | one point per team-match",
            )
            ax = axes[0]
            ax.scatter(valid.ppda, valid.risk, color=MUTED, alpha=0.55, s=30)
            _mark_today(ax, charts, valid, today.dropna(subset=["ppda", "risk"]) if len(today) else today, "ppda", "risk")
            ax.set_xlabel("PPDA (lower = fewer opponent passes per action)")
            ax.set_ylabel("Dangerous counters / advanced losses (%)")
            ax.legend(facecolor=BG, labelcolor=FG)
            charts.save(
                fig,
                "67_history_pressing_risk.png",
                "Pressing frequency and turnover risk",
                f"{len(valid)} comparable team-match observations. A relationship here cannot isolate the effect of pressing height or off-ball cover.",
                "Each axis keeps its own denominator; not a causal model.",
                len(valid),
            )
        else:
            charts.skipped.append(
                {
                    "chart": "67_history_pressing_risk.png",
                    "reason": "Fewer than 20 valid PPDA/risk observations.",
                }
            )
    else:
        for name in ["66_history_territory_chances.png", "67_history_pressing_risk.png"]:
            charts.skipped.append(
                {
                    "chart": name,
                    "reason": f"Need 20 comparable team-match observations, named competition and matching explicit xg_model_version; found {len(hist)}.",
                }
            )
    compact_profiles(charts, people, events)
    target = Path(out) / "analysis_tables"
    target.mkdir(exist_ok=True)
    for key, frame in data.items():
        if key != "actions":
            frame.to_csv(target / (key + ".csv"), index=False, encoding="utf-8-sig")
    (target / "chart_contracts.json").write_text(
        json.dumps(charts.contracts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (target / "skipped_charts.json").write_text(
        json.dumps(charts.skipped, indent=2), encoding="utf-8"
    )
    events.attrs["chart_contracts"] = charts.contracts
    # Include every meaningful appearance in the publication. Brief cameos
    # still get a standalone file, while the main PDF/catalog keeps players
    # with at least 30 minutes so every full-match player is represented.
    eligible = set(
        re.sub(r"[^\w.-]+", "_", name) + ".png"
        for name in people.loc[people.minutes >= 30, "player"]
    )
    return [p for p in charts.paths if "player_profiles" not in p.parts or p.name in eligible], data
