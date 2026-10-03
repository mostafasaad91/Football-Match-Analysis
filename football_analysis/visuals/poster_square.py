"""A one-page summary in a 1:1 frame, for places that crop to a square.

The four 4:5 boards tell the match in sequence; this is the single image a
timeline shows when it is cropped to a square. One row of headline figures, the
shots on the pitch beside the cumulative xG curve, and the chance balance with
the new post-shot and line-breaking figures. It uses the same tokens as the
other boards (colours from the active theme, the bundled condensed face for
type that has to carry) so it sits beside them without looking like a different
product.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

from football_analysis.visuals import crests
from football_analysis.visuals.typography import display

SIZE_IN = 12.0
DPI = 200  # 2400 x 2400
FILENAME = "match_summary_square.png"


def _flag(frame, column):
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].astype(str).str.lower().isin(["true", "1", "1.0"])


def _card(fig, rect, radius=0.012):
    """A rounded panel in figure fractions; returns an axes in 0-1 coordinates."""
    from football_analysis.visuals import visual_redesign_full as v

    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            mutation_aspect=rect[2] / rect[3],
            facecolor=v.PANEL,
            edgecolor=v.GRID,
            linewidth=1.0,
        )
    )
    return ax


def _team_figures(events, xg, team_metrics, team_id, name):
    """The numbers one side contributes to the page."""
    shots = events[
        events["team_id"].eq(team_id)
        & _flag(events, "is_shot")
        & ~_flag(events, "is_penalty_shootout")
        & ~_flag(events, "is_own_goal")
    ]
    xg_row = xg[xg["team"].astype(str).eq(name)].iloc[0] if "team" in xg and len(xg) else {}
    metrics = team_metrics[team_metrics["team_id"].eq(team_id)]
    metric = metrics.iloc[0] if len(metrics) else {}

    def pick(source, key, default=0.0):
        try:
            value = float(source[key])
        except (KeyError, TypeError, ValueError):
            return default
        return default if np.isnan(value) else value

    outcome = shots.get("shot_whoscored_type", pd.Series("", index=shots.index)).astype(str)
    return {
        "xG": float(pd.to_numeric(shots["xG"], errors="coerce").fillna(0).sum()),
        "xGoT": pick(metric, "xGoT"),
        "shots": int(len(shots)),
        "on_target": int(outcome.isin(["Goal", "SavedShot"]).sum()),
        "big_chances": int(pick(xg_row, "big_chances")),
        "box_entries": int(pick(metric, "box_entries")),
        "line_breaking": int(pick(metric, "line_breaking_completed")),
        "line_breaking_tried": int(pick(metric, "line_breaking_passes")),
        "field_tilt": pick(metric, "field_tilt"),
    }


def build_square_poster(events, xg, team_metrics, info, out_dir, allow_download=False) -> Path:
    from football_analysis.visuals import visual_redesign_full as v

    home_id, away_id = info["home_id"], info["away_id"]
    home_name, away_name = info["home_name"], info["away_name"]
    colours = {home_id: v.HOME, away_id: v.AWAY}
    figures = {
        home_id: _team_figures(events, xg, team_metrics, home_id, home_name),
        away_id: _team_figures(events, xg, team_metrics, away_id, away_name),
    }

    fig = plt.figure(figsize=(SIZE_IN, SIZE_IN), facecolor=v.BG)

    # -- header: crest, name, score, name, crest -------------------------------
    fig.text(0.05, 0.962, "MATCH SUMMARY", color=v.MUTED, fontsize=10, fontweight="bold")
    fig.text(
        0.95,
        0.962,
        "  ·  ".join(filter(None, [info.get("competition"), info.get("date")])).upper(),
        color=v.MUTED,
        fontsize=9,
        ha="right",
    )
    for side, x_crest, x_name, align in (
        (home_id, 0.105, 0.17, "left"),
        (away_id, 0.895, 0.83, "right"),
    ):
        name = home_name if side == home_id else away_name
        crests.place_crest(
            fig,
            x_crest,
            0.895,
            side,
            monogram=name[:3].upper(),
            colour=colours[side],
            width=0.085,
            background=v.BG,
            allow_download=allow_download,
        )
        fig.text(
            x_name,
            0.895,
            name,
            color=v.TEXT,
            ha=align,
            va="center",
            **display(40),
        )
    fig.text(
        0.5,
        0.898,
        str(info["score"]).replace(":", "–").replace(" ", ""),
        color=v.TEXT,
        ha="center",
        va="center",
        **display(64),
    )
    fig.add_artist(
        Rectangle((0.05, 0.835), 0.45, 0.004, transform=fig.transFigure, color=v.HOME, lw=0)
    )
    fig.add_artist(
        Rectangle((0.5, 0.835), 0.45, 0.004, transform=fig.transFigure, color=v.AWAY, lw=0)
    )

    # -- headline figures -------------------------------------------------------
    headline = [
        ("EXPECTED GOALS", "xG", "{:.2f}"),
        ("EXPECTED GOALS ON TARGET", "xGoT", "{:.2f}"),
        ("SHOTS", "shots", "{:.0f}"),
        ("LINE-BREAKING PASSES", "line_breaking", "{:.0f}"),
    ]
    left, width = 0.05, 0.9
    gap = 0.015
    cell = (width - gap * (len(headline) - 1)) / len(headline)
    for index, (label, key, fmt) in enumerate(headline):
        ax = _card(fig, [left + index * (cell + gap), 0.700, cell, 0.115])
        ax.text(0.06, 0.82, label, color=v.MUTED, fontsize=7.8, fontweight="bold", va="center")
        for slot, side in enumerate((home_id, away_id)):
            x = 0.06 + slot * 0.5
            ax.text(
                x,
                0.36,
                fmt.format(figures[side][key]),
                color=colours[side],
                va="center",
                **display(34),
            )
    # -- shots on the pitch -----------------------------------------------------
    pitch_card = _card(fig, [0.05, 0.285, 0.30, 0.395])
    pitch = fig.add_axes([0.062, 0.295, 0.276, 0.375])
    pitch.set_facecolor(v.PANEL)
    v.draw_long_pitch(pitch)
    pitch.axis("off")
    for side, flip in ((home_id, False), (away_id, True)):
        shots = events[
            events["team_id"].eq(side)
            & _flag(events, "is_shot")
            & ~_flag(events, "is_penalty_shootout")
            & ~_flag(events, "is_own_goal")
        ].dropna(subset=["x", "y"])
        if shots.empty:
            continue
        x = shots["x"].astype(float).to_numpy()
        y = shots["y"].astype(float).to_numpy()
        if flip:
            x, y = 100 - x, 100 - y
        px, py = v.attack_xy(x, y)
        goal = _flag(shots, "is_goal").to_numpy()
        size = 30 + 1100 * pd.to_numeric(shots["xG"], errors="coerce").fillna(0).to_numpy()
        pitch.scatter(
            px[~goal],
            py[~goal],
            s=size[~goal],
            facecolor=colours[side],
            edgecolor=v.BG,
            linewidth=1.0,
            alpha=0.88,
            zorder=4,
        )
        pitch.scatter(
            px[goal],
            py[goal],
            s=size[goal] * 1.5 + 110,
            marker="*",
            facecolor=colours[side],
            edgecolor="white",
            linewidth=1.3,
            zorder=6,
        )
    pitch_card.text(0.06, 0.975, "SHOTS", color=v.MUTED, fontsize=8, fontweight="bold", va="center")
    pitch_card.text(
        0.5,
        0.022,
        "AREA = xG · STAR = GOAL",
        color=v.NEUTRAL,
        fontsize=6.8,
        ha="center",
        va="center",
    )

    # -- cumulative xG ---------------------------------------------------------
    flow_card = _card(fig, [0.365, 0.285, 0.585, 0.395])
    flow = fig.add_axes([0.405, 0.325, 0.46, 0.29])
    flow.set_facecolor("none")
    minute = pd.to_numeric(events["minute"], errors="coerce").fillna(0) + (
        pd.to_numeric(events.get("second", 0), errors="coerce").fillna(0) / 60
    )
    end_minute = max(90.0, float(minute.max()))
    ceiling = 0.4
    for side in (home_id, away_id):
        shots = events[
            events["team_id"].eq(side)
            & _flag(events, "is_shot")
            & ~_flag(events, "is_penalty_shootout")
            & ~_flag(events, "is_own_goal")
        ].copy()
        shots["t"] = minute.loc[shots.index]
        shots = shots.sort_values("t")
        total = pd.to_numeric(shots["xG"], errors="coerce").fillna(0).cumsum()
        xs = np.r_[0, shots["t"].to_numpy(), end_minute]
        ys = np.r_[0, total.to_numpy(), total.iloc[-1] if len(total) else 0]
        flow.step(xs, ys, where="post", color=colours[side], lw=2.8, solid_capstyle="round")
        flow.fill_between(xs, ys, step="post", color=colours[side], alpha=0.11)
        ceiling = max(ceiling, float(ys.max()))
        for _, goal in shots[_flag(shots, "is_goal")].iterrows():
            at = float(goal["t"])
            height = float(total[shots["t"] <= at].iloc[-1])
            flow.scatter(
                [at],
                [height],
                s=120,
                facecolor=colours[side],
                edgecolor="white",
                linewidth=1.6,
                zorder=5,
            )
            flow.annotate(
                f"{int(goal['minute'])}′",
                (at, height),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                color=v.TEXT,
                fontsize=9,
                fontweight="bold",
            )
        flow.text(
            end_minute + 2.2,
            float(ys[-1]),
            f"{ys[-1]:.2f}",
            color=colours[side],
            va="center",
            clip_on=False,
            **display(26),
        )
    flow.set_xlim(0, end_minute + 1)
    flow.set_ylim(0, ceiling * 1.18)
    flow.grid(axis="y", color=v.GRID, lw=0.9)
    flow.set_axisbelow(True)
    flow.axvline(45, color=v.GRID, lw=1.0, ls=(0, (3, 4)))
    flow.set_xticks([0, 15, 30, 45, 60, 75, 90])
    flow.set_xticklabels([f"{m}′" for m in (0, 15, 30, 45, 60, 75, 90)])
    flow.tick_params(colors=v.MUTED, length=0, labelsize=8)
    for spine in flow.spines.values():
        spine.set_visible(False)
    flow_card.text(
        0.04,
        0.965,
        "EXPECTED GOALS THROUGH THE MATCH",
        color=v.MUTED,
        fontsize=8,
        fontweight="bold",
        va="center",
    )

    # -- chance balance -----------------------------------------------------------
    rows = [
        ("Big chances", "big_chances", "{:.0f}"),
        ("On target", "on_target", "{:.0f}"),
        ("Box entries", "box_entries", "{:.0f}"),
        ("Field tilt", "field_tilt", "{:.0f}%"),
    ]
    balance = _card(fig, [0.05, 0.05, 0.90, 0.215])
    balance.text(
        0.025, 0.91, "CHANCE BALANCE", color=v.MUTED, fontsize=8, fontweight="bold", va="center"
    )
    balance.text(
        0.975,
        0.91,
        "WITHIN EACH ROW THE LARGER SIDE FILLS THE BAR",
        color=v.NEUTRAL,
        fontsize=6.8,
        ha="right",
        va="center",
    )
    slot = 0.78 / len(rows)
    for index, (label, key, fmt) in enumerate(rows):
        y = 0.77 - index * slot
        a, b = figures[home_id][key], figures[away_id][key]
        top = max(a, b, 1e-9)
        balance.text(
            0.5,
            y + 0.075,
            label.upper(),
            color=v.MUTED,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
        )
        balance.add_patch(
            Rectangle((0.2, y - 0.03), 0.6, 0.034, facecolor=v.GRID, edgecolor="none")
        )
        balance.add_patch(
            Rectangle(
                (0.5 - 0.3 * a / top, y - 0.026),
                0.3 * a / top - 0.004,
                0.026,
                facecolor=v.HOME,
                edgecolor="none",
            )
        )
        balance.add_patch(
            Rectangle(
                (0.504, y - 0.026), 0.3 * b / top - 0.004, 0.026, facecolor=v.AWAY, edgecolor="none"
            )
        )
        balance.text(
            0.025,
            y - 0.012,
            fmt.format(a),
            color=v.HOME if a >= b else v.TEXT,
            va="center",
            **display(24),
        )
        balance.text(
            0.975,
            y - 0.012,
            fmt.format(b),
            color=v.AWAY if b >= a else v.TEXT,
            ha="right",
            va="center",
            **display(24),
        )

    fig.text(
        0.05,
        0.022,
        "Shots drawn attacking the goal they were shooting at · xGOT = post-shot xG",
        color=v.NEUTRAL,
        fontsize=8,
    )
    fig.text(0.95, 0.022, "MOSTAFA SAAD", color=v.MUTED, ha="right", **display(13))

    out = Path(out_dir) / FILENAME
    fig.savefig(out, dpi=DPI, facecolor=v.BG)
    plt.close(fig)
    return out
