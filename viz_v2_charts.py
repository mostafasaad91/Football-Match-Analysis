"""
viz_v2_charts.py — production v2 chart renderers + DataFrame adapters.

Each visual exposes two functions:
    • render_<name>_v2(...)  — pure renderer; takes simple data structures,
                              returns a matplotlib Figure (no I/O).
    • make_<name>_v2(events, info, ...)  — DataFrame adapter; converts the
                              project's events DataFrame to the renderer's
                              expected shape and calls render_*.

The renderers compose the design-system primitives from viz_v2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

from viz_v2 import (
    chrome, panel_card, metric_strip, key_insight, themed_pitch,
    BG_DARK, BG_MID, GRID_COL, TEXT_BR, TEXT_MAIN, TEXT_DIM,
    C_HOME, C_AWAY, C_GOLD, shadow,
)

IS_LIGHT_THEME = BG_DARK.upper() in {"#FFFFFF", "WHITE"}
ROW_BG = "#F8FAFC" if IS_LIGHT_THEME else "#0f1620"
MID_BG = "#F8FAFC" if IS_LIGHT_THEME else "#0a0e16"
PASS_ARROW = "#111827" if IS_LIGHT_THEME else "#ffffff"
PASS_NEG = "#7F1D1D" if IS_LIGHT_THEME else "#e63946"
GOAL_ROW_HOME = "#F8FAFC" if IS_LIGHT_THEME else "#1a0a0a"
GOAL_ROW_AWAY = "#F1F5F9" if IS_LIGHT_THEME else "#0a1630"


# ═════════════════════════════════════════════════════════════════════════
#  1. xG FLOW v2
# ═════════════════════════════════════════════════════════════════════════
def render_xg_flow_v2(hn, an, score, hc, ac, shots_h, shots_a):
    fig = plt.figure(figsize=(15, 9.5), facecolor=BG_DARK)
    chrome(fig, section="xG FLOW",
           title=f"{hn} vs {an} — xG Flow",
           subtitle="Cumulative Expected Goals minute by minute · "
                    "stars mark goals · shaded territory = chance creation",
           hn=hn, an=an, score=score,
           footer_note="Step height = shot xG · steeper curve = better chances")

    ax = fig.add_axes([0.05, 0.18, 0.62, 0.66])
    ax.set_facecolor(BG_MID)
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL); s.set_linewidth(0.5)

    def _cum(shots):
        ms = sorted(shots, key=lambda s: s["minute"])
        xs, ys = [0], [0]
        cum = 0
        for s in ms:
            xs += [s["minute"], s["minute"]]
            ys += [cum,         cum + s["xG"]]
            cum += s["xG"]
        xs.append(95); ys.append(cum)
        return xs, ys, cum

    hx, hy, h_total = _cum(shots_h)
    ax_, ay, a_total = _cum(shots_a)
    ax.fill_between(hx, 0, hy, color=hc, alpha=0.18, zorder=2)
    ax.fill_between(ax_, 0, ay, color=ac, alpha=0.18, zorder=2)
    ax.plot(hx, hy, color=hc, lw=2.6, solid_capstyle="round", zorder=3)
    ax.plot(ax_, ay, color=ac, lw=2.6, solid_capstyle="round", zorder=3)

    # Goal stars
    for shots, col in [(shots_h, hc), (shots_a, ac)]:
        ms = sorted(shots, key=lambda s: s["minute"])
        cum = 0
        for s in ms:
            cum += s["xG"]
            if s["is_goal"]:
                ax.scatter([s["minute"]], [cum], s=180, marker="*",
                           color=C_GOLD, edgecolor=col, lw=1.8, zorder=5)
                ax.annotate(s["player"].split()[-1],
                            xy=(s["minute"], cum),
                            xytext=(0, 10), textcoords="offset points",
                            ha="center", color=C_GOLD, fontsize=8,
                            fontweight="bold", path_effects=shadow(2))

    y_max = max(h_total, a_total, 0.5) * 1.15
    for xv, lb in [(45, "HT"), (90, "FT")]:
        ax.axvline(xv, color=C_GOLD, lw=0.9, ls="--", alpha=0.45, zorder=1)
        ax.text(xv, y_max * 0.96, lb, ha="center", va="top",
                color=C_GOLD, fontsize=8, fontweight="bold", alpha=0.85)

    # Hottest 10-min window
    def _best_window(shots, w=10):
        if not shots:
            return None
        best = (0, 0, 0)
        for start in range(0, 90 - w + 1):
            x = sum(s["xG"] for s in shots
                    if start <= s["minute"] < start + w)
            if x > best[0]:
                best = (x, start, start + w)
        return best
    bw_h = _best_window(shots_h)
    bw_a = _best_window(shots_a)
    best = bw_h if (bw_h and (not bw_a or bw_h[0] > bw_a[0])) else bw_a
    momentum_team = hn if best is bw_h else an
    if best and best[0] > 0:
        ax.axvspan(best[1], best[2], color=C_GOLD, alpha=0.07, zorder=0)
        ax.text((best[1] + best[2]) / 2, y_max * 0.04,
                f"hottest 10-min · {momentum_team}",
                ha="center", color=C_GOLD, fontsize=8.5,
                fontweight="bold", alpha=0.85, style="italic")

    ax.set_xlim(0, 95); ax.set_ylim(0, y_max)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID_COL, lw=0.4, alpha=0.4)
    ax.set_xlabel("Minute", color=TEXT_DIM, fontsize=9)
    ax.set_ylabel("Cumulative xG", color=TEXT_DIM, fontsize=9)
    # Inline legend chips
    ax.text(0.015, 0.97, "●", color=hc, fontsize=18,
            transform=ax.transAxes, va="top")
    ax.text(0.040, 0.96, hn, color=TEXT_BR, fontsize=10.5, fontweight="bold",
            transform=ax.transAxes, va="top")
    ax.text(0.015, 0.91, "●", color=ac, fontsize=18,
            transform=ax.transAxes, va="top")
    ax.text(0.040, 0.90, an, color=TEXT_BR, fontsize=10.5, fontweight="bold",
            transform=ax.transAxes, va="top")

    # Goals (by minute) panel
    goals = []
    for s in shots_h:
        if s["is_goal"]:
            goals.append((s, hn, hc))
    for s in shots_a:
        if s["is_goal"]:
            goals.append((s, an, ac))
    goals.sort(key=lambda g: g[0]["minute"])

    ax2 = panel_card(fig, 0.70, 0.50, 0.27, 0.34,
                     title="Goals (by minute)", accent=C_GOLD)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.text(0.05, 0.83, "MIN", color=TEXT_DIM, fontsize=9,
             fontweight="bold", transform=ax2.transAxes, va="center")
    ax2.text(0.20, 0.83, "SCORER", color=TEXT_DIM, fontsize=9,
             fontweight="bold", transform=ax2.transAxes, va="center")
    ax2.text(0.93, 0.83, "xG", ha="right", color=TEXT_DIM, fontsize=9,
             fontweight="bold", transform=ax2.transAxes, va="center")
    ax2.plot([0.04, 0.96], [0.79, 0.79], color=GRID_COL, lw=0.6,
             transform=ax2.transAxes)
    if goals:
        n = max(len(goals), 1)
        rh = 0.70 / n
        for i, (s, _team_nm, team_col) in enumerate(goals):
            cy = 0.74 - (i + 0.5) * rh
            if i % 2 == 0:
                ax2.add_patch(mpatches.Rectangle(
                    (0.04, cy - rh*0.42), 0.92, rh*0.84,
                    facecolor=ROW_BG, lw=0,
                    transform=ax2.transAxes, zorder=1))
            ax2.text(0.05, cy, f"{s['minute']}'", ha="left", va="center",
                     color=TEXT_DIM, fontsize=10.5, fontweight="bold",
                     transform=ax2.transAxes, zorder=2)
            ax2.text(0.20, cy, s["player"].split()[-1] if s["player"] else "—",
                     ha="left", va="center", color=team_col, fontsize=10.5,
                     fontweight="bold", transform=ax2.transAxes, zorder=2)
            ax2.text(0.93, cy, f"{s['xG']:.2f}", ha="right", va="center",
                     color=C_GOLD, fontsize=11, fontweight="bold",
                     transform=ax2.transAxes, zorder=2)

    diff = h_total - a_total
    leader = hn if diff > 0 else an
    if best and best[0] > 0:
        insight = (
            f"{leader} produced {abs(diff):.2f} more xG over the 90 minutes. "
            f"The hottest 10-minute spell came from {momentum_team} "
            f"({best[1]:02d}'–{best[2]:02d}') with {best[0]:.2f} xG packed "
            f"into that window."
        )
    else:
        insight = (f"{leader} produced {abs(diff):.2f} more xG over the "
                   f"90 minutes.")
    key_insight(fig, 0.70, 0.16, 0.27, 0.28, text=insight, wrap=34)

    h_goals = sum(1 for s in shots_h if s["is_goal"])
    a_goals = sum(1 for s in shots_a if s["is_goal"])
    cards = [
        (f"{hn[:14]} xG",    f"{h_total:.2f}", hc),
        (f"{hn[:14]} Goals", str(h_goals),     C_GOLD),
        ("xG Diff",          f"{'+' if diff >= 0 else ''}{diff:.2f}", C_GOLD),
        (f"{an[:14]} Goals", str(a_goals),     C_GOLD),
        (f"{an[:14]} xG",    f"{a_total:.2f}", ac),
    ]
    metric_strip(fig, cards=cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  2. SHOT MAP v2
# ═════════════════════════════════════════════════════════════════════════
def render_shot_map_v2(team_name, opp_name, score, team_color, shots):
    fig = plt.figure(figsize=(15, 10), facecolor=BG_DARK)
    chrome(fig, section="SHOT MAP",
           title=f"{team_name} — Shot Map",
           subtitle="Each dot is a shot · size = xG · filled = goal · "
                    "ring = on-target save",
           hn=team_name, an=opp_name, score=score,
           footer_note="Direction of attack →")

    ax = fig.add_axes([0.04, 0.18, 0.62, 0.66])
    themed_pitch(ax, attacking_only=True)
    ax.set_xlim(48, 102); ax.set_ylim(-2, 102)

    goals = [s for s in shots if s["is_goal"]]
    on_t  = [s for s in shots if (not s["is_goal"]) and s["is_on_target"]]
    miss  = [s for s in shots
             if (not s["is_goal"]) and (not s["is_on_target"])]

    def _scatter(group, marker, fc, ec, lw, alpha, zorder):
        if not group:
            return
        xs = [s["x"] for s in group]
        ys = [s["y"] for s in group]
        sz = [40 + s["xG"] * 1100 for s in group]
        ax.scatter(xs, ys, s=sz, marker=marker, facecolor=fc,
                   edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder)
        for s in group:
            ax.text(s["x"], s["y"] + 3.2, f"{s['xG']:.2f}",
                    ha="center", va="bottom",
                    color=TEXT_DIM, fontsize=6.5, fontweight="bold",
                    path_effects=shadow(1.6),
                    zorder=zorder + 1)

    _scatter(miss,  "o", "none",     team_color, 1.2, 0.55, 3)
    _scatter(on_t,  "o", team_color, TEXT_BR,    1.5, 0.85, 4)
    _scatter(goals, "*", C_GOLD,     team_color, 1.8, 1.00, 5)

    if goals:
        gsorted = sorted(goals, key=lambda s: s["y"])
        n_g = len(gsorted)
        anchor_x = 100.5
        y_top, y_bot = 88, 12
        for i, s in enumerate(gsorted):
            label_y = y_top - i * (y_top - y_bot) / max(n_g - 1, 1)
            surname = s["player"].split()[-1] if s["player"] else "—"
            ax.annotate(
                f"{surname} {s['minute']}'",
                xy=(s["x"], s["y"]),
                xytext=(anchor_x, label_y),
                ha="left", va="center",
                color=C_GOLD, fontsize=8, fontweight="bold",
                path_effects=shadow(2),
                arrowprops=dict(arrowstyle="-", color=C_GOLD, lw=0.7,
                                alpha=0.55, connectionstyle="arc3,rad=0.0"),
                zorder=6,
            )

    ax.annotate("", xy=(99, 5), xytext=(70, 5),
                arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.2,
                                alpha=0.55))
    ax.text(85, 8, "ATTACK", ha="center", color=C_GOLD,
            fontsize=8.5, fontweight="bold", alpha=0.7)

    chips = [("Goal", C_GOLD, "*"),
             ("On Target", team_color, "●"),
             ("Off Target / Blocked", team_color, "○"),
             ("Big chance (xG ≥ 0.30)", TEXT_BR, "●")]
    cx = 0.06
    for lbl, col, mk in chips:
        fig.text(cx, 0.140, mk, ha="left", va="center", color=col,
                 fontsize=14)
        fig.text(cx + 0.014, 0.140, lbl, ha="left", va="center",
                 color=TEXT_MAIN, fontsize=9.5, fontweight="bold")
        cx += 0.014 + 0.008 * len(lbl) + 0.04

    # Top scorers
    by_player = {}
    for s in shots:
        by_player.setdefault(s["player"],
                             {"xG": 0, "shots": 0, "goals": 0})
        by_player[s["player"]]["xG"]    += s["xG"]
        by_player[s["player"]]["shots"] += 1
        by_player[s["player"]]["goals"] += int(s["is_goal"])
    top = sorted(by_player.items(), key=lambda kv: -kv[1]["xG"])[:5]

    ax2 = panel_card(fig, 0.69, 0.50, 0.27, 0.34,
                     title="Top Shot Sources (by xG)", accent=team_color)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.text(0.05, 0.83, "PLAYER", color=TEXT_DIM, fontsize=9,
             fontweight="bold", transform=ax2.transAxes, va="center")
    ax2.text(0.62, 0.83, "SH",  ha="center", color=TEXT_DIM, fontsize=9,
             fontweight="bold", transform=ax2.transAxes, va="center")
    ax2.text(0.78, 0.83, "G",   ha="center", color=TEXT_DIM, fontsize=9,
             fontweight="bold", transform=ax2.transAxes, va="center")
    ax2.text(0.93, 0.83, "xG",  ha="right",  color=TEXT_DIM, fontsize=9,
             fontweight="bold", transform=ax2.transAxes, va="center")
    ax2.plot([0.04, 0.96], [0.79, 0.79], color=GRID_COL, lw=0.6,
             transform=ax2.transAxes)
    if top:
        n = len(top); rh = 0.70 / n
        for i, (player, d) in enumerate(top):
            cy = 0.74 - (i + 0.5) * rh
            if i % 2 == 0:
                ax2.add_patch(mpatches.Rectangle(
                    (0.04, cy - rh*0.42), 0.92, rh*0.84,
                    facecolor=ROW_BG, lw=0,
                    transform=ax2.transAxes, zorder=1))
            label = (player or "—").split()[-1] if player else "—"
            ax2.text(0.05, cy, label, ha="left", va="center",
                     color=TEXT_BR, fontsize=10.5, fontweight="bold",
                     transform=ax2.transAxes, zorder=2)
            ax2.text(0.62, cy, str(d["shots"]), ha="center", va="center",
                     color=TEXT_MAIN, fontsize=10.5,
                     transform=ax2.transAxes, zorder=2)
            ax2.text(0.78, cy, str(d["goals"]),
                     ha="center", va="center",
                     color=C_GOLD if d["goals"] else TEXT_MAIN,
                     fontsize=10.5,
                     fontweight="bold" if d["goals"] else "normal",
                     transform=ax2.transAxes, zorder=2)
            ax2.text(0.93, cy, f"{d['xG']:.2f}", ha="right", va="center",
                     color=team_color, fontsize=11, fontweight="bold",
                     transform=ax2.transAxes, zorder=2)

    n_shots = len(shots)
    n_goals = len(goals)
    n_ot    = len(on_t) + n_goals
    total_xg = sum(s["xG"] for s in shots)
    big_chances = sum(1 for s in shots if s["xG"] >= 0.30)
    insight = (
        f"{team_name} took {n_shots} shots worth {total_xg:.2f} xG. "
        f"{n_ot}/{n_shots} forced the keeper into action. "
        f"{big_chances} big chance(s); finishing was "
        f"{'over' if n_goals > total_xg else 'under'}-performing with "
        f"{n_goals} goals."
    )
    key_insight(fig, 0.69, 0.16, 0.27, 0.28, text=insight, wrap=34)

    cards = [
        ("Total Shots", str(n_shots), C_GOLD),
        ("On Target",   str(n_ot),    team_color),
        ("Goals",       str(n_goals), C_GOLD),
        ("Big Chances", str(big_chances), team_color),
        ("Total xG",    f"{total_xg:.2f}", C_GOLD),
    ]
    metric_strip(fig, cards=cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  3. SHOT BREAKDOWN v2
# ═════════════════════════════════════════════════════════════════════════
def render_shot_breakdown_v2(hn, an, score, home, away, goals, hc=None, ac=None):
    hc = hc or C_HOME
    ac = ac or C_AWAY
    fig = plt.figure(figsize=(16, 10), facecolor=BG_DARK)
    chrome(fig, section="SHOT BREAKDOWN",
           title=f"{hn} vs {an} — Shot Breakdown & Goals",
           subtitle="Volume · placement · finishing — and how every goal "
                    "actually arrived",
           hn=hn, an=an, score=score,
           footer_note="Bars = shot volume · table = every goal scored")

    ax1 = panel_card(fig, 0.04, 0.46, 0.56, 0.35,
                     title="Shot Volume by Outcome", accent=C_GOLD,
                     body=False)
    keys   = ["shots", "on_target", "off_target", "blocked", "post"]
    labels = ["Total Shots", "On Target", "Off Target", "Blocked", "Woodwork"]
    n = len(keys)
    pos = np.arange(n); w = 0.36
    h_vals = [home.get(k, 0) for k in keys]
    a_vals = [away.get(k, 0) for k in keys]
    ax1.set_xlim(-0.6, n - 0.4)
    ax1.set_ylim(0, max(h_vals + a_vals + [1]) * 1.35)
    for y in np.arange(0, max(h_vals + a_vals + [1]) * 1.35, 5):
        ax1.axhline(y, color=GRID_COL, lw=0.4, alpha=0.5, zorder=0)
    ax1.bar(pos - w/2, h_vals, w, color=hc, alpha=0.9, lw=0, zorder=2)
    ax1.bar(pos + w/2, a_vals, w, color=ac, alpha=0.9, lw=0, zorder=2)
    for i, (hv, av) in enumerate(zip(h_vals, a_vals)):
        h_col = C_GOLD if hv > av else TEXT_BR
        a_col = C_GOLD if av > hv else TEXT_BR
        ax1.text(i - w/2, hv + 0.4, str(hv), ha="center", va="bottom",
                 color=h_col, fontsize=11, fontweight="bold",
                 path_effects=shadow(2.4))
        ax1.text(i + w/2, av + 0.4, str(av), ha="center", va="bottom",
                 color=a_col, fontsize=11, fontweight="bold",
                 path_effects=shadow(2.4))
    ax1.set_xticks(pos)
    ax1.set_xticklabels(labels, color=TEXT_MAIN, fontsize=10)
    ax1.tick_params(axis="x", length=0, pad=8); ax1.set_yticks([])
    for sp in ["top", "right", "left", "bottom"]:
        ax1.spines[sp].set_visible(False)
    ax1.text(0.02, -0.16, "● " + hn, color=hc, fontsize=10,
             fontweight="bold", transform=ax1.transAxes)
    ax1.text(0.20, -0.16, "● " + an, color=ac, fontsize=10,
             fontweight="bold", transform=ax1.transAxes)

    diff_xg = home.get("xG", 0) - away.get("xG", 0)
    leader = hn if diff_xg > 0 else an
    insight = (
        f"{leader} produced the stronger chance profile "
        f"(xG {'+' if diff_xg >= 0 else ''}{diff_xg:.2f}). "
        f"{home.get('on_target', 0)} of {hn}'s {home.get('shots', 0)} shots "
        f"forced saves vs. only {away.get('on_target', 0)} from "
        f"{an}'s {away.get('shots', 0)}."
    )
    key_insight(fig, 0.62, 0.46, 0.34, 0.35, text=insight)

    ax2 = panel_card(fig, 0.04, 0.16, 0.92, 0.27,
                     title="Goals & Assists", accent=C_GOLD)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    cols = [("Min", 0.04), ("Scorer", 0.13), ("Team", 0.30),
            ("Type", 0.46), ("Assist", 0.62), ("xG", 0.92)]
    for c, x in cols:
        ax2.text(x, 0.83, c, ha="left" if c != "xG" else "right",
                 va="center", color=TEXT_DIM, fontsize=9.5, fontweight="bold",
                 transform=ax2.transAxes)
    ax2.plot([0.03, 0.97], [0.79, 0.79], color=GRID_COL, lw=0.6,
             transform=ax2.transAxes)
    type_styles = {
        "OP": ("OPEN PLAY",  "#22c55e"),
        "SP": ("SET PIECE",  "#f59e0b"),
        "PK": ("PENALTY",    "#a855f7"),
        "OG": ("OWN GOAL",   "#ec4899"),
    }
    n_g = max(len(goals), 1)
    row_h = 0.65 / n_g
    for i, g in enumerate(goals):
        mn, scorer, gtype, assist, xg, side = g
        cy = 0.74 - (i + 0.5) * row_h
        team_col = hc if side == "home" else ac
        team_nm  = hn if side == "home" else an
        fc = GOAL_ROW_HOME if side == "home" else GOAL_ROW_AWAY
        ax2.add_patch(mpatches.Rectangle(
            (0.03, cy - row_h*0.42), 0.94, row_h*0.84,
            facecolor=fc, lw=0, transform=ax2.transAxes, zorder=1))
        ax2.text(0.04, cy, mn, ha="left", va="center",
                 color=TEXT_DIM, fontsize=10, fontweight="bold",
                 transform=ax2.transAxes, zorder=2)
        ax2.text(0.13, cy, scorer, ha="left", va="center",
                 color=TEXT_BR if not IS_LIGHT_THEME else "#111827",
                 fontsize=11, fontweight="bold",
                 transform=ax2.transAxes, zorder=2)
        ax2.text(0.30, cy, team_nm, ha="left", va="center",
                 color=team_col, fontsize=10.5, fontweight="bold",
                 transform=ax2.transAxes, zorder=2)
        badge_lbl, badge_col = type_styles.get(gtype, ("OPEN PLAY", "#22c55e"))
        bw = 0.13
        ax2.add_patch(mpatches.FancyBboxPatch(
            (0.46, cy - 0.022), bw, 0.044,
            boxstyle="round,pad=0.015,rounding_size=0.022",
            facecolor=badge_col, alpha=0.22, edgecolor=badge_col, lw=0.8,
            transform=ax2.transAxes, zorder=2))
        ax2.text(0.46 + bw/2, cy, badge_lbl, ha="center", va="center",
                 color=badge_col, fontsize=8.5, fontweight="bold",
                 transform=ax2.transAxes, zorder=3)
        ax2.text(0.62, cy, assist, ha="left", va="center",
                 color=TEXT_MAIN, fontsize=10,
                 transform=ax2.transAxes, zorder=2)
        ax2.text(0.92, cy, f"{xg:.2f}", ha="right", va="center",
                 color=C_GOLD, fontsize=11, fontweight="bold",
                 transform=ax2.transAxes, zorder=2)

    home_g = sum(1 for g in goals if g[5] == "home")
    away_g = sum(1 for g in goals if g[5] == "away")
    cards = [
        ("Final Score",     f"{score}",                         C_GOLD),
        (f"{hn[:14]} xG",   f"{home.get('xG', 0):.2f}",         hc),
        ("xG Diff",         f"{'+' if diff_xg >= 0 else ''}{diff_xg:.2f}",
         C_GOLD),
        (f"{an[:14]} xG",   f"{away.get('xG', 0):.2f}",         ac),
        ("Goals — H/A",     f"{home_g} / {away_g}",             C_GOLD),
    ]
    metric_strip(fig, cards=cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  4. PASS NETWORK v2
# ═════════════════════════════════════════════════════════════════════════
def _role_color(role, team_color):
    return {
        "sub_in": "#15803D",
        "sub_out": "#92400E",
        "both_sub": "#7C3AED",
        "red_card": "#B91C1C",
    }.get(role or "", team_color)


def _role_badge(role):
    return {
        "sub_in": "↑",
        "sub_out": "↓",
        "both_sub": "↕",
        "red_card": "RC",
    }.get(role or "", "")


def render_pass_network_v2(team_name, opp_name, score, team_color,
                           players, edges):
    fig = plt.figure(figsize=(15, 10), facecolor=BG_DARK)
    chrome(fig, section="PASS NETWORK",
           title=f"{team_name} — Pass Network",
           subtitle="Nodes at average pass position · node size = passes "
                    "made · edge width = pair volume (≥ 3 passes)",
           hn=team_name, an=opp_name, score=score,
           footer_note="Direction of attack →")

    ax = fig.add_axes([0.04, 0.18, 0.62, 0.66])
    themed_pitch(ax)
    by_name = {p["name"]: p for p in players}

    max_e = max((e["count"] for e in edges), default=1)
    drawn_edges = []
    for e in edges:
        if e["count"] < 3:
            continue
        if e["from"] not in by_name or e["to"] not in by_name:
            continue
        p1 = by_name[e["from"]]; p2 = by_name[e["to"]]
        ratio = e["count"] / max_e
        lw    = 1.8 + 10.0 * ratio
        alpha = 0.58 + 0.42 * ratio
        ax.plot([p1["x"], p2["x"]], [p1["y"], p2["y"]],
                color=TEXT_BR, lw=lw + 3.2, alpha=0.20,
                solid_capstyle="round", zorder=2)
        ax.plot([p1["x"], p2["x"]], [p1["y"], p2["y"]],
                color=team_color, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=3)
        drawn_edges.append((p1, p2, e["count"], lw))

    top_for_labels = sorted(drawn_edges, key=lambda t: -t[2])[:8]
    for p1, p2, count, _lw in top_for_labels:
        mx, my = (p1["x"] + p2["x"]) / 2, (p1["y"] + p2["y"]) / 2
        ax.text(mx, my, str(count), ha="center", va="center",
                color=TEXT_BR, fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.20",
                          facecolor=BG_DARK, edgecolor=team_color,
                          lw=0.7, alpha=0.85),
                zorder=4)

    max_p = max((p["passes"] for p in players), default=1)
    for p in players:
        size = 380 + 1700 * (p["passes"] / max_p)
        role = p.get("role", "")
        node_color = _role_color(role, team_color)
        badge = _role_badge(role)
        ax.scatter([p["x"]], [p["y"]], s=size + 200, color=TEXT_BR,
                   alpha=0.95, zorder=5)
        ax.scatter([p["x"]], [p["y"]], s=size, color=node_color,
                   edgecolor=TEXT_BR, lw=1.6, alpha=0.96, zorder=6)
        short = (p["name"] or "").split()[-1][:9]
        if badge:
            short = f"{short} {badge}"
        ax.text(p["x"], p["y"] + 0.5, short, ha="center", va="center",
                color=TEXT_BR if not badge else node_color, fontsize=8.5, fontweight="bold",
                path_effects=shadow(2.0), zorder=7)
        ax.text(p["x"], p["y"] - 3.4, str(p["passes"]),
                ha="center", va="center",
                color=C_GOLD, fontsize=7, fontweight="bold",
                path_effects=shadow(2.0), zorder=7)

    ax.annotate("", xy=(99, 5), xytext=(70, 5),
                arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.2,
                                alpha=0.55))
    ax.text(85, 8, "ATTACK", ha="center", color=C_GOLD,
            fontsize=8.5, fontweight="bold", alpha=0.7)

    ax2 = panel_card(fig, 0.69, 0.50, 0.27, 0.34,
                     title="Top Partnerships (passes)", accent=team_color)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.text(0.05, 0.83, "PAIR", color=TEXT_DIM, fontsize=9,
             fontweight="bold", transform=ax2.transAxes, va="center")
    ax2.text(0.93, 0.83, "PASSES", ha="right", color=TEXT_DIM, fontsize=9,
             fontweight="bold", transform=ax2.transAxes, va="center")
    ax2.plot([0.04, 0.96], [0.79, 0.79], color=GRID_COL, lw=0.6,
             transform=ax2.transAxes)
    top_edges = sorted(edges, key=lambda e: -e["count"])[:6]
    if top_edges:
        n = len(top_edges); rh = 0.70 / n
        for i, e in enumerate(top_edges):
            cy = 0.74 - (i + 0.5) * rh
            if i % 2 == 0:
                ax2.add_patch(mpatches.Rectangle(
                    (0.04, cy - rh*0.42), 0.92, rh*0.84,
                    facecolor=ROW_BG, lw=0,
                    transform=ax2.transAxes, zorder=1))
            from_short = (e["from"] or "").split()[-1]
            to_short   = (e["to"]   or "").split()[-1]
            pair = f"{from_short} ↔ {to_short}"
            ax2.text(0.05, cy, pair, ha="left", va="center",
                     color=TEXT_BR, fontsize=10.5, fontweight="bold",
                     transform=ax2.transAxes, zorder=2)
            ax2.text(0.93, cy, str(e["count"]), ha="right", va="center",
                     color=team_color, fontsize=11, fontweight="bold",
                     transform=ax2.transAxes, zorder=2)

    total_passes = sum(p["passes"] for p in players)
    if players:
        top_player = max(players, key=lambda p: p["passes"])
        most_active = (top_player["name"] or "").split()[-1]
        avg_x = int(np.mean([p["x"] for p in players]))
        insight = (
            f"{team_name} attempted {total_passes} passes across all used players. "
            f"{most_active} was the busiest hub with {top_player['passes']} "
            f"passes. The team's centre of gravity sat at x≈{avg_x} "
            f"on the pitch."
        )
    else:
        insight = f"{team_name} pass network — insufficient pass data."
    key_insight(fig, 0.69, 0.16, 0.27, 0.28, text=insight, wrap=34)

    if players:
        avg_x = int(np.mean([p["x"] for p in players]))
        y_spread = int(max(p["y"] for p in players) -
                       min(p["y"] for p in players))
    else:
        avg_x, y_spread = 0, 0
    cards = [
        ("Total Passes",  str(total_passes), C_GOLD),
        ("Players",       str(len(players)), team_color),
        ("Top Partner.",  str(top_edges[0]["count"]) if top_edges else "0",
         C_GOLD),
        ("Avg X (depth)", f"{avg_x}",        team_color),
        ("Y Spread",      f"{y_spread}",     C_GOLD),
    ]
    metric_strip(fig, cards=cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  5. xT MAP v2
# ═════════════════════════════════════════════════════════════════════════
def render_xt_map_v2(team_name, opp_name, score, team_color, passes):
    fig = plt.figure(figsize=(16, 10), facecolor=BG_DARK)
    chrome(fig, section="xT MAP",
           title=f"{team_name} — Expected Threat (xT)",
           subtitle="Heatmap = pitch xT value · white arrows = positive-xT "
                    "passes · red arrows = negative-xT passes",
           hn=team_name, an=opp_name, score=score,
           footer_note="Direction of attack →")

    ax = fig.add_axes([0.04, 0.18, 0.62, 0.66])
    themed_pitch(ax, line_color="#2a3a4a", line_alpha=0.40)

    rows_n, cols_n = 8, 12
    cell_w, cell_h = 100 / cols_n, 100 / rows_n
    grid = np.zeros((rows_n, cols_n))
    for r in range(rows_n):
        for c in range(cols_n):
            grid[r, c] = ((c / cols_n) ** 1.6) * 0.6 + \
                         (1 - abs(r - rows_n / 2 + 0.5) / (rows_n / 2)) * 0.18

    cmap = LinearSegmentedColormap.from_list(
        "xt", ["#0a1628", "#0d3b6e", "#1a6b3c", "#f59e0b", "#e63946"]
    )
    ax.imshow(grid, extent=[0, 100, 0, 100], origin="lower", aspect="auto",
              cmap=cmap, vmin=0, vmax=0.7, alpha=0.78, zorder=1)

    for r in range(rows_n):
        for c in range(cols_n):
            v  = grid[r, c]
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            tc = "#F8FAFC" if IS_LIGHT_THEME else (TEXT_BR if v < 0.30 else "#111111")
            stroke_col = "#0F172A" if IS_LIGHT_THEME else ("#000000" if tc == TEXT_BR else "#ffffff")
            ax.text(cx, cy, f"{v:.3f}", ha="center", va="center",
                    color=tc, fontsize=6.0, fontweight="bold",
                    path_effects=[pe.withStroke(
                        linewidth=1.1 if IS_LIGHT_THEME else 1.2,
                        foreground=stroke_col)],
                    zorder=3)

    pos = [p for p in passes if p.get("successful") and p.get("xT", 0) > 0]
    neg = [p for p in passes if p.get("xT", 0) < 0]
    if pos:
        max_pos = max(p["xT"] for p in pos)
        for p in pos:
            a = 0.18 + 0.55 * (p["xT"] / max_pos)
            ax.plot([p["x"], p["end_x"]], [p["y"], p["end_y"]],
                    color=PASS_ARROW, lw=0.85, alpha=a,
                    solid_capstyle="round", zorder=4)
    for p in neg:
        ax.plot([p["x"], p["end_x"]], [p["y"], p["end_y"]],
                color=PASS_NEG, lw=0.65, alpha=0.18,
                solid_capstyle="round", zorder=4)
    top5_passes = sorted(pos, key=lambda p: -p["xT"])[:5]
    for p in top5_passes:
        ax.annotate(
            "", xy=(p["end_x"], p["end_y"]), xytext=(p["x"], p["y"]),
            arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=2.2,
                            alpha=0.90, connectionstyle="arc3,rad=0.08"),
            zorder=6,
        )

    ax.annotate("", xy=(99, 5), xytext=(70, 5),
                arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.2,
                                alpha=0.55))
    ax.text(85, 8, "ATTACK", ha="center", color=C_GOLD,
            fontsize=8.5, fontweight="bold", alpha=0.7)

    by_player = {}
    for p in pos:
        nm = p.get("player") or "—"
        by_player.setdefault(nm, {"xT": 0, "n": 0})
        by_player[nm]["xT"] += p["xT"]
        by_player[nm]["n"]  += 1
    top_creators = sorted(by_player.items(), key=lambda kv: -kv[1]["xT"])[:6]

    ax2 = panel_card(fig, 0.69, 0.50, 0.27, 0.34,
                     title="Top xT Creators", accent=team_color)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.text(0.05, 0.83, "PLAYER", color=TEXT_DIM, fontsize=9,
             fontweight="bold", transform=ax2.transAxes, va="center")
    ax2.text(0.65, 0.83, "P", ha="center", color=TEXT_DIM, fontsize=9,
             fontweight="bold", transform=ax2.transAxes, va="center")
    ax2.text(0.93, 0.83, "xT", ha="right", color=TEXT_DIM, fontsize=9,
             fontweight="bold", transform=ax2.transAxes, va="center")
    ax2.plot([0.04, 0.96], [0.79, 0.79], color=GRID_COL, lw=0.6,
             transform=ax2.transAxes)
    if top_creators:
        n = len(top_creators); rh = 0.70 / n
        for i, (player, d) in enumerate(top_creators):
            cy = 0.74 - (i + 0.5) * rh
            if i % 2 == 0:
                ax2.add_patch(mpatches.Rectangle(
                    (0.04, cy - rh*0.42), 0.92, rh*0.84,
                    facecolor=ROW_BG, lw=0,
                    transform=ax2.transAxes, zorder=1))
            ax2.text(0.05, cy, (player or "—").split()[-1],
                     ha="left", va="center",
                     color=TEXT_BR, fontsize=10.5, fontweight="bold",
                     transform=ax2.transAxes, zorder=2)
            ax2.text(0.65, cy, str(d["n"]), ha="center", va="center",
                     color=TEXT_MAIN, fontsize=10.5,
                     transform=ax2.transAxes, zorder=2)
            ax2.text(0.93, cy, f"{d['xT']:.3f}", ha="right", va="center",
                     color=team_color, fontsize=11, fontweight="bold",
                     transform=ax2.transAxes, zorder=2)

    total_xt = sum(p["xT"] for p in pos)
    n_pass = len(pos)
    if top_creators and total_xt > 0:
        leader_name = (top_creators[0][0] or "—").split()[-1]
        leader_xt = top_creators[0][1]["xT"]
        leader_share = leader_xt / total_xt * 100
        insight = (
            f"{team_name} created {total_xt:.2f} xT across {n_pass} "
            f"positive passes. {leader_name} alone delivered "
            f"{leader_share:.0f}% of the team's threat through the lines."
        )
    else:
        insight = (f"{team_name} created {total_xt:.2f} xT across "
                   f"{n_pass} positive passes.")
    key_insight(fig, 0.69, 0.16, 0.27, 0.28, text=insight, wrap=34)

    cards = [
        ("Total xT",    f"{total_xt:.2f}", C_GOLD),
        ("Pos. Passes", str(n_pass),       team_color),
        ("Avg xT/Pass", f"{(total_xt/n_pass if n_pass else 0):.3f}",
         C_GOLD),
        ("Top Pass xT", f"{max((p['xT'] for p in pos), default=0):.3f}",
         team_color),
        ("Top Creator",
         (top_creators[0][0].split()[-1] if top_creators else "—"), C_GOLD),
    ]
    metric_strip(fig, cards=cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  DataFrame ADAPTERS — accept events DataFrame + info dict
# ═════════════════════════════════════════════════════════════════════════
ON_TARGET_TYPES = {"Goal", "SavedShot"}


def _safe(v, default=0):
    try:
        if v is None:
            return default
        import math
        if isinstance(v, float) and math.isnan(v):
            return default
        return v
    except Exception:
        return default


def _match_colors(info):
    return (
        info.get("home_color") or info.get("HOME_COLOR") or C_HOME,
        info.get("away_color") or info.get("AWAY_COLOR") or C_AWAY,
    )


def _shots_for_team(events, team_id):
    out = []
    sub = events[(events["is_shot"] == True) & (events["team_id"] == team_id)]
    for _, row in sub.iterrows():
        shot_type = (row.get("shot_whoscored_type")
                     or row.get("type") or "")
        out.append({
            "x":            float(_safe(row.get("x"), 0)),
            "y":            float(_safe(row.get("y"), 0)),
            "xG":           float(_safe(row.get("xG"), 0) or 0),
            "is_goal":      bool(row.get("is_goal", False)),
            "is_on_target": shot_type in ON_TARGET_TYPES,
            "player":       str(_safe(row.get("player"), "")),
            "minute":       int(_safe(row.get("minute"), 0) or 0),
        })
    return out


def make_xg_flow_v2(events, info, xg_data=None):
    hn  = info.get("home_name") or "Home"
    an  = info.get("away_name") or "Away"
    hid = info.get("home_id"); aid = info.get("away_id")
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)
    return render_xg_flow_v2(
        hn, an, str(score), hc, ac,
        _shots_for_team(events, hid),
        _shots_for_team(events, aid),
    )


def make_shot_map_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name  = an if is_home else hn
    score = info.get("score") or "—"
    return render_shot_map_v2(team_name, opp_name, str(score), team_color,
                               _shots_for_team(events, team_id))


def make_shot_breakdown_v2(events, info, xg_data):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)
    h = xg_data.get(hn, {}) if xg_data else {}
    a = xg_data.get(an, {}) if xg_data else {}
    home = {k: int(_safe(h.get(k), 0))
            for k in ("shots", "on_target", "off_target", "blocked", "post")}
    away = {k: int(_safe(a.get(k), 0))
            for k in ("shots", "on_target", "off_target", "blocked", "post")}
    home["xG"] = float(_safe(h.get("xG"), 0) or 0)
    away["xG"] = float(_safe(a.get("xG"), 0) or 0)

    def _assist_from_context(goal_row):
        """Fallback assist lookup from the same-team action immediately before a goal."""
        explicit = goal_row.get("assist_player")
        if explicit and str(explicit).lower() != "nan":
            return str(explicit), goal_row.get("assist_type")
        if events is None or events.empty:
            return "", ""
        scoring_team = goal_row.get("scoring_team") or goal_row.get("team_id")
        minute = _safe(goal_row.get("minute"), 0) or 0
        second = _safe(goal_row.get("second"), 0) or 0
        goal_time = float(minute) * 60 + float(second)
        cand = events[events.get("team_id") == scoring_team].copy()
        if cand.empty:
            return "", ""
        cand["__t"] = cand.get("minute", 0).fillna(0).astype(float) * 60 + cand.get("second", 0).fillna(0).astype(float)
        cand = cand[(cand["__t"] < goal_time) & (cand["__t"] >= goal_time - 25)]
        if "is_pass" in cand.columns:
            cand = cand[cand["is_pass"] == True]
        if "outcome" in cand.columns:
            successful = cand[cand["outcome"].fillna("").astype(str).str.lower().eq("successful")]
            if not successful.empty:
                cand = successful
        if cand.empty:
            return "", ""
        if "is_key_pass" in cand.columns and cand["is_key_pass"].any():
            cand = cand[cand["is_key_pass"] == True]
        last = cand.sort_values("__t").iloc[-1]
        name = str(last.get("player") or "")
        q = str(last.get("qualifier_names") or "")
        if "Cross" in q:
            at = "Cross"
        elif "ThroughBall" in q:
            at = "ThroughBall"
        elif "Chipped" in q:
            at = "Chipped"
        elif "FastBreak" in q:
            at = "FastBreak"
        else:
            at = last.get("assist_type") or "Assist"
        return name, at

    # Goals list — same shape the renderer expects
    goals_list = []
    gdf = events[events["is_goal"] == True].sort_values("minute")
    for _, row in gdf.iterrows():
        side = "home" if row.get("scoring_team") == info.get("home_id") \
               else "away"
        if row.get("is_own_goal"):
            gtype = "OG"
        elif row.get("is_penalty"):
            gtype = "PK"
        elif (row.get("is_direct_fk") or row.get("is_header") is False
              and "Set Piece" in str(row.get("qualifier_names", ""))):
            gtype = "SP"
        else:
            gtype = "OP"
        ap, at = _assist_from_context(row)
        if ap and str(ap).lower() != "nan":
            assist = f"{ap}" + (f" ({at})" if at and str(at).lower() != "nan"
                                  else "")
        else:
            assist = "—"
        goals_list.append((
            f"{int(_safe(row.get('minute'), 0))}'",
            (str(row.get("player") or "")).split()[-1] or "—",
            gtype,
            assist,
            float(_safe(row.get("xG"), 0) or 0),
            side,
        ))
    return render_shot_breakdown_v2(hn, an, str(score), home, away, goals_list, hc, ac)


def make_pass_network_v2(events, info, team_id, team_color):
    """
    Build players + edges by inferring receivers from the next same-team
    event (same approach as the legacy `build_pass_network`).
    """
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name  = an if is_home else hn
    score = info.get("score") or "—"

    team_evts = (events[events["team_id"] == team_id]
                 .sort_values(["minute", "second"])
                 .reset_index(drop=True))
    passes = team_evts[team_evts["is_pass"] == True]

    # Nodes: avg pass position + pass count per player_id
    nodes = {}
    if not passes.empty:
        for pid, grp in passes.groupby("player_id"):
            try:
                nodes[pid] = {
                    "name":   str(grp["player"].iloc[0]),
                    "avg_x":  float(grp["x"].mean()),
                    "avg_y":  float(grp["y"].mean()),
                    "passes": int(len(grp)),
                }
            except Exception:
                continue

    # Edges: each successful pass paired with the very next same-team
    # event's player as the receiver (same heuristic as the legacy code).
    edges_count = {}
    succ = team_evts[(team_evts["is_pass"] == True) &
                     (team_evts["outcome"] == "Successful")]
    for i in range(len(succ)):
        curr_idx = succ.index[i]
        passer_id = succ.iloc[i].get("player_id")
        if passer_id is None:
            continue
        later = team_evts[(team_evts.index > curr_idx) &
                          team_evts["player_id"].notna()]
        if later.empty:
            continue
        recv_id = later.iloc[0]["player_id"]
        if recv_id == passer_id:
            continue
        # Backfill receiver into nodes if they passed less but received
        if recv_id not in nodes:
            rr = team_evts[team_evts["player_id"] == recv_id]
            if not rr.empty:
                nodes[recv_id] = {
                    "name":   str(rr["player"].iloc[0]),
                    "avg_x":  float(rr["x"].mean()),
                    "avg_y":  float(rr["y"].mean()),
                    "passes": 0,
                }
        key = tuple(sorted([passer_id, recv_id]))
        edges_count[key] = edges_count.get(key, 0) + 1

    sub_in = set(info.get("sub_in") or [])
    sub_out = set(info.get("sub_out") or [])
    red_cards = set(info.get("red_cards") or [])

    def _pid_role(pid):
        if pid in red_cards:
            return "red_card"
        if pid in sub_in and pid in sub_out:
            return "both_sub"
        if pid in sub_in:
            return "sub_in"
        if pid in sub_out:
            return "sub_out"
        return ""

    # Keep the main structure plus every substituted player who touched/passed.
    activity = {pid: n["passes"] for pid, n in nodes.items()}
    for (a, b), c in edges_count.items():
        activity[a] = activity.get(a, 0) + c * 0.5
        activity[b] = activity.get(b, 0) + c * 0.5
    keep = set(sorted(activity, key=activity.get, reverse=True)[:11])
    keep.update(pid for pid in activity if pid in sub_in or pid in sub_out or pid in red_cards)

    players_list = [
        {"name": nodes[pid]["name"],
         "x":     nodes[pid]["avg_x"],
         "y":     nodes[pid]["avg_y"],
         "passes": nodes[pid]["passes"],
         "role": _pid_role(pid)}
        for pid in keep if pid in nodes
        and not (np.isnan(nodes[pid]["avg_x"])
                 or np.isnan(nodes[pid]["avg_y"]))
    ]

    edges_list = []
    for (a, b), count in edges_count.items():
        if a not in keep or b not in keep:
            continue
        if a not in nodes or b not in nodes:
            continue
        edges_list.append({
            "from":  nodes[a]["name"],
            "to":    nodes[b]["name"],
            "count": int(count),
        })
    edges_list.sort(key=lambda e: -e["count"])

    return render_pass_network_v2(team_name, opp_name, str(score), team_color,
                                   players_list, edges_list)


def make_xt_map_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name  = an if is_home else hn
    score = info.get("score") or "—"

    passes_list = []
    if "xT" in events.columns:
        sub = events[(events["is_pass"] == True) &
                     (events["team_id"] == team_id) &
                     (events["xT"].notna())]
        for _, row in sub.iterrows():
            passes_list.append({
                "x":          float(_safe(row.get("x"), 0)),
                "y":          float(_safe(row.get("y"), 0)),
                "end_x":      float(_safe(row.get("end_x"), 0)),
                "end_y":      float(_safe(row.get("end_y"), 50)),
                "xT":         float(_safe(row.get("xT"), 0) or 0),
                "player":     str(_safe(row.get("player"), "")),
                "successful": row.get("outcome") == "Successful",
            })
    return render_xt_map_v2(team_name, opp_name, str(score), team_color,
                            passes_list)


# ═════════════════════════════════════════════════════════════════════════
#  GENERIC PITCH-OVERLAY v2 — used by all simple "pitch + dots/arrows"
#  visuals (defensive heatmap, average positions, box entries, crosses,
#  high turnovers, etc.). Same chrome + sidebar + metric strip shape.
# ═════════════════════════════════════════════════════════════════════════
def render_pitch_overlay_v2(
    *,
    section, title, subtitle, hn, an, score, footer_note,
    team_color, draw_overlay,
    sidebar_title, sidebar_headers, sidebar_rows, sidebar_value_cols=None,
    insight_text, metric_cards,
):
    """
    sidebar_rows: list of tuples; each tuple aligns with sidebar_headers.
    sidebar_value_cols: optional list of column-x positions for each header.
                       Defaults to evenly spaced.
    """
    fig = plt.figure(figsize=(16, 10), facecolor=BG_DARK)
    chrome(fig, section=section, title=title, subtitle=subtitle,
           hn=hn, an=an, score=score, footer_note=footer_note)

    ax = fig.add_axes([0.04, 0.18, 0.62, 0.66])
    themed_pitch(ax)
    if callable(draw_overlay):
        draw_overlay(ax)
    # Direction arrow
    ax.annotate("", xy=(99, 5), xytext=(70, 5),
                arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.2,
                                alpha=0.55))
    ax.text(85, 8, "ATTACK", ha="center", color=C_GOLD,
            fontsize=8.5, fontweight="bold", alpha=0.7)

    # Sidebar table
    ax2 = panel_card(fig, 0.69, 0.50, 0.27, 0.34,
                     title=sidebar_title, accent=team_color)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    n_cols = len(sidebar_headers)
    if sidebar_value_cols is None:
        if n_cols == 1:
            xs = [0.05]
        elif n_cols == 2:
            xs = [0.05, 0.93]
        else:
            xs = [0.05] + [0.05 + (i * 0.88 / (n_cols - 1))
                           for i in range(1, n_cols)]
    else:
        xs = sidebar_value_cols
    for i, (lbl, x) in enumerate(zip(sidebar_headers, xs)):
        ha = "left" if i == 0 else ("right" if i == n_cols - 1 else "center")
        ax2.text(x, 0.83, lbl, ha=ha, va="center",
                 color=TEXT_DIM, fontsize=9, fontweight="bold",
                 transform=ax2.transAxes)
    ax2.plot([0.04, 0.96], [0.79, 0.79], color=GRID_COL, lw=0.6,
             transform=ax2.transAxes)
    if sidebar_rows:
        n = max(len(sidebar_rows), 1)
        rh = 0.70 / n
        for i, row in enumerate(sidebar_rows):
            cy = 0.74 - (i + 0.5) * rh
            if i % 2 == 0:
                ax2.add_patch(mpatches.Rectangle(
                    (0.04, cy - rh*0.42), 0.92, rh*0.84,
                    facecolor=ROW_BG, lw=0,
                    transform=ax2.transAxes, zorder=1))
            for j, (val, x) in enumerate(zip(row, xs)):
                ha = "left" if j == 0 else ("right" if j == n_cols - 1
                                              else "center")
                col = TEXT_BR if j == 0 else (
                    team_color if j == n_cols - 1 else TEXT_MAIN)
                fs  = 10.5 if j != n_cols - 1 else 11
                fw  = "bold" if j != n_cols - 2 else "normal"
                ax2.text(x, cy, str(val), ha=ha, va="center",
                         color=col, fontsize=fs, fontweight=fw,
                         transform=ax2.transAxes, zorder=2)

    key_insight(fig, 0.69, 0.16, 0.27, 0.28, text=insight_text, wrap=34)
    metric_strip(fig, cards=metric_cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  DEFENSIVE HEATMAP v2  (figs 26 / 27)
# ═════════════════════════════════════════════════════════════════════════
DEF_TYPE_COLORS = {
    "Tackle":       "#22c55e",
    "Interception": "#f59e0b",
    "Clearance":    "#1e90ff",
    "BlockedShot":  "#a855f7",
    "BallRecovery": "#14b8a6",
    "Foul":         "#ef4444",
    "Aerial":       "#facc15",
}


def _blocked_shots_for_team(events, info, team_id) -> int:
    """Count blocked shots as defensive blocks by the opponent of the shooter."""
    if "team_id" not in events.columns:
        return 0
    hid = info.get("home_id")
    aid = info.get("away_id")

    def _blocked_shots_by(shooter_id):
        mask = events["team_id"] == shooter_id
        hit = pd.Series(False, index=events.index)
        for col in ("type", "shot_whoscored_type", "shot_category"):
            if col in events.columns:
                vals = events[col].fillna("").astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)
                hit = hit | vals.isin({"blockedshot", "blocked"})
        if "qualifier_names" in events.columns:
            hit = hit | events["qualifier_names"].fillna("").astype(str).str.contains(r"\bBlocked\b", case=False, regex=True)
        if "is_shot" in events.columns:
            mask = mask & ((events["is_shot"] == True) | hit)
        return events[mask & hit].copy()

    direct = len(_blocked_shots_by(team_id))
    opp_id = aid if team_id == hid else (hid if team_id == aid else None)
    if opp_id is None:
        return direct
    opponent_blocked_shots = len(_blocked_shots_by(opp_id))
    if not opponent_blocked_shots:
        opp_side = "away" if team_id == hid else "home"
        mc = (info.get("matchcentre_stats", {}) or {}).get(opp_side, {}) or {}
        opponent_blocked_shots = int(_safe(mc.get("blocked"), 0) or 0)
    return opponent_blocked_shots if opponent_blocked_shots else direct


def _defensive_events_for_team(events, info, team_id):
    """Return defensive events, mapping opponent BlockedShot events to this team."""
    own_types = [t for t in DEF_TYPE_COLORS.keys() if t != "BlockedShot"]
    own = events[(events["team_id"] == team_id) & events["type"].isin(own_types)].copy()
    hid = info.get("home_id")
    aid = info.get("away_id")
    opp_id = aid if team_id == hid else (hid if team_id == aid else None)

    def _blocked_shots_by(shooter_id):
        mask = events["team_id"] == shooter_id
        hit = pd.Series(False, index=events.index)
        for col in ("type", "shot_whoscored_type", "shot_category"):
            if col in events.columns:
                vals = events[col].fillna("").astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)
                hit = hit | vals.isin({"blockedshot", "blocked"})
        if "qualifier_names" in events.columns:
            hit = hit | events["qualifier_names"].fillna("").astype(str).str.contains(r"\bBlocked\b", case=False, regex=True)
        if "is_shot" in events.columns:
            mask = mask & ((events["is_shot"] == True) | hit)
        return events[mask & hit].copy()

    if opp_id is not None:
        blocks = _blocked_shots_by(opp_id)
        if blocks.empty:
            blocks = _blocked_shots_by(team_id)
    else:
        blocks = _blocked_shots_by(team_id)

    if not blocks.empty:
        blocks["team_id"] = team_id
        blocks["type"] = "BlockedShot"
        blocks["player"] = blocks.get("blocked_by", "Team block")
        blocks["player"] = blocks["player"].fillna("Team block").replace("", "Team block")
    if own.empty:
        return blocks
    if blocks.empty:
        return own
    return pd.concat([own, blocks], ignore_index=True, sort=False)


def make_defensive_heatmap_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name  = an if is_home else hn
    score = info.get("score") or "—"

    sub = _defensive_events_for_team(events, info, team_id)
    by_type = {}
    by_player = {}
    points = []
    for _, row in sub.iterrows():
        t = row.get("type")
        x = float(_safe(row.get("x"), 50))
        y = float(_safe(row.get("y"), 50))
        p = str(_safe(row.get("player"), "—"))
        points.append((x, y, t))
        by_type[t] = by_type.get(t, 0) + 1
        by_player[p] = by_player.get(p, 0) + 1

    def draw_overlay(ax):
        for x, y, t in points:
            ax.scatter([x], [y], s=85,
                       facecolor=DEF_TYPE_COLORS.get(t, team_color),
                       edgecolor="white", lw=0.8, alpha=0.78, zorder=4)
        # Inline legend chips at the bottom of pitch
        chips = list(DEF_TYPE_COLORS.items())[:6]
        x = 1
        for lbl, col in chips:
            ax.scatter([x], [-3], s=42, color=col, edgecolor="white",
                       lw=0.7, clip_on=False, zorder=5)
            ax.text(x + 1.5, -3, lbl, ha="left", va="center",
                    color=TEXT_MAIN, fontsize=8, fontweight="bold",
                    clip_on=False)
            x += 16

    top_players = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    sidebar_rows = [(name.split()[-1] if name else "—", str(cnt))
                    for name, cnt in top_players]

    total = len(points)
    leader_name = (top_players[0][0].split()[-1]
                   if top_players else "—")
    insight = (
        f"{team_name} completed {total} defensive actions. {leader_name} "
        f"led the workload with {top_players[0][1]} actions. "
        f"Tackles: {by_type.get('Tackle', 0)} · "
        f"Interceptions: {by_type.get('Interception', 0)} · "
        f"Clearances: {by_type.get('Clearance', 0)} · "
        f"Blocks: {by_type.get('BlockedShot', 0)}."
    ) if top_players else f"{team_name} — no defensive data."

    cards = [
        ("Total Actions", str(total),                       C_GOLD),
        ("Tackles",       str(by_type.get("Tackle", 0)),    team_color),
        ("Interceptions", str(by_type.get("Interception", 0)), C_GOLD),
        ("Clearances",    str(by_type.get("Clearance", 0)), team_color),
        ("Blocks",        str(by_type.get("BlockedShot", 0)), C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="DEFENSIVE ACTIONS",
        title=f"{team_name} — Defensive Heatmap",
        subtitle="Each dot = one defensive action · colour = action type · "
                 "cluster reveals the defensive line height",
        hn=team_name, an=opp_name, score=str(score),
        footer_note="Tackle · Interception · Clearance · Block · Recovery · Foul",
        team_color=team_color, draw_overlay=draw_overlay,
        sidebar_title="Top Defenders (actions)",
        sidebar_headers=["PLAYER", "ACT"],
        sidebar_rows=sidebar_rows,
        insight_text=insight, metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  AVERAGE POSITIONS v2  (figs 29 / 30)
# ═════════════════════════════════════════════════════════════════════════
def make_avg_positions_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name  = an if is_home else hn
    score = info.get("score") or "—"

    sub_in = set(info.get("sub_in") or [])
    sub_out = set(info.get("sub_out") or [])
    red_cards = set(info.get("red_cards") or [])

    def _pid_role(pid):
        if pid in red_cards:
            return "red_card"
        if pid in sub_in and pid in sub_out:
            return "both_sub"
        if pid in sub_in:
            return "sub_in"
        if pid in sub_out:
            return "sub_out"
        return ""

    sub = events[(events["team_id"] == team_id) &
                 events["player_id"].notna() &
                 events["x"].notna() & events["y"].notna()]
    grp_all = sub.groupby(["player_id", "player"], dropna=True).agg(
        x=("x", "mean"), y=("y", "mean"), touches=("event_id", "count"),
    ).reset_index().sort_values("touches", ascending=False)
    keep_ids = set(grp_all.head(11)["player_id"].tolist())
    keep_ids.update(pid for pid in grp_all["player_id"].tolist()
                    if pid in sub_in or pid in sub_out or pid in red_cards)
    grp = grp_all[grp_all["player_id"].isin(keep_ids)]

    players = []
    for _, r in grp.iterrows():
        players.append({"name": str(r["player"]),
                        "x": float(r["x"]),
                        "y": float(r["y"]),
                        "touches": int(r["touches"]),
                        "role": _pid_role(r["player_id"])})
    max_t = max((p["touches"] for p in players), default=1)

    def draw_overlay(ax):
        # Convex-ish team shape: connect each player to the centroid
        if players:
            cx = float(np.mean([p["x"] for p in players]))
            cy = float(np.mean([p["y"] for p in players]))
            for p in players:
                ax.plot([p["x"], cx], [p["y"], cy],
                        color=team_color, lw=0.6, alpha=0.18, zorder=2)
            ax.scatter([cx], [cy], s=40, color=C_GOLD, alpha=0.5,
                       zorder=2)
        for p in players:
            sz = 380 + 1500 * (p["touches"] / max_t)
            node_color = _role_color(p.get("role"), team_color)
            badge = _role_badge(p.get("role"))
            ax.scatter([p["x"]], [p["y"]], s=sz + 220, color=TEXT_BR,
                       alpha=0.95, zorder=4)
            ax.scatter([p["x"]], [p["y"]], s=sz, color=node_color,
                       edgecolor=TEXT_BR, lw=1.4, alpha=0.96, zorder=5)
            label = p["name"].split()[-1][:9] + (f" {badge}" if badge else "")
            ax.text(p["x"], p["y"] + 0.5,
                    label,
                    ha="center", va="center",
                    color=TEXT_BR if not badge else node_color,
                    fontsize=8.5, fontweight="bold",
                    path_effects=shadow(2.0), zorder=6)
            ax.text(p["x"], p["y"] - 3.4, str(p["touches"]),
                    ha="center", va="center",
                    color=C_GOLD, fontsize=7, fontweight="bold",
                    path_effects=shadow(2.0), zorder=6)

    rows = [(p["name"].split()[-1] if p["name"] else "—",
             str(p["touches"])) for p in players[:6]]

    if players:
        avg_x = int(np.mean([p["x"] for p in players]))
        spread_y = int(max(p["y"] for p in players) -
                       min(p["y"] for p in players))
        depth_x = int(max(p["x"] for p in players) -
                      min(p["x"] for p in players))
        total_touches = sum(p["touches"] for p in players)
        insight = (
            f"{team_name}'s average shape sat at x≈{avg_x} (depth) with "
            f"a vertical spread of {depth_x} and a horizontal spread of "
            f"{spread_y}. {players[0]['name'].split()[-1]} held the most "
            f"touches ({players[0]['touches']})."
        )
    else:
        avg_x = spread_y = depth_x = total_touches = 0
        insight = f"{team_name} — no positional data."

    cards = [
        ("Avg X (depth)",  str(avg_x),       team_color),
        ("X Spread",       str(depth_x),     C_GOLD),
        ("Y Spread",       str(spread_y),    team_color),
        ("Total Touches",  str(total_touches), C_GOLD),
        ("Players",        str(len(players)), team_color),
    ]
    return render_pitch_overlay_v2(
        section="AVERAGE POSITIONS",
        title=f"{team_name} — Average Positions",
        subtitle="Each node at the player's average touch position · size "
                 "= touches · faint lines connect to team centroid",
        hn=team_name, an=opp_name, score=str(score),
        footer_note="Shape signals defensive line height + width",
        team_color=team_color, draw_overlay=draw_overlay,
        sidebar_title="Most Active (touches)",
        sidebar_headers=["PLAYER", "TOUCHES"],
        sidebar_rows=rows,
        insight_text=insight, metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  BOX ENTRIES v2  (figs 32 / 33)
# ═════════════════════════════════════════════════════════════════════════
def make_box_entries_v2(events, info, team_id, team_color):
    """
    Box entry = pass or carry that ends inside the opponent box
    (end_x ≥ 83, 21 ≤ end_y ≤ 79) but starts outside.
    """
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name  = an if is_home else hn
    score = info.get("score") or "—"

    BOX_X, BOX_Y_LO, BOX_Y_HI = 83, 21, 79
    sub = events[(events["team_id"] == team_id) &
                 events["x"].notna() & events["y"].notna() &
                 events["end_x"].notna() & events["end_y"].notna() &
                 (events["outcome"] == "Successful")]

    entries = []
    for _, r in sub.iterrows():
        sx, sy = float(r["x"]),     float(r["y"])
        ex, ey = float(r["end_x"]), float(r["end_y"])
        in_box   = (ex >= BOX_X and BOX_Y_LO <= ey <= BOX_Y_HI)
        from_out = not (sx >= BOX_X and BOX_Y_LO <= sy <= BOX_Y_HI)
        if in_box and from_out:
            kind = "pass" if bool(r.get("is_pass")) else "carry"
            channel = ("left" if sy < 38 else
                       ("centre" if sy <= 62 else "right"))
            entries.append({"sx": sx, "sy": sy, "ex": ex, "ey": ey,
                            "kind": kind, "player": str(r.get("player") or "—"),
                            "channel": channel})

    def draw_overlay(ax):
        for e in entries:
            col = C_GOLD if e["kind"] == "pass" else "#22c55e"
            ax.annotate("", xy=(e["ex"], e["ey"]),
                        xytext=(e["sx"], e["sy"]),
                        arrowprops=dict(arrowstyle="->", color=col,
                                        lw=1.2, alpha=0.78,
                                        connectionstyle="arc3,rad=0.10"),
                        zorder=4)
        # legend chips bottom
        for x0, lbl, col in [(2, "Pass entry", C_GOLD),
                             (32, "Carry entry", "#22c55e")]:
            ax.annotate("", xy=(x0 + 8, -3), xytext=(x0, -3),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.4,
                                        alpha=0.9), clip_on=False)
            ax.text(x0 + 9.5, -3, lbl, ha="left", va="center",
                    color=TEXT_MAIN, fontsize=8, fontweight="bold",
                    clip_on=False)

    by_player = {}
    for e in entries:
        by_player[e["player"]] = by_player.get(e["player"], 0) + 1
    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    rows = [(p.split()[-1] if p else "—", str(c)) for p, c in top]

    n_pass = sum(1 for e in entries if e["kind"] == "pass")
    n_carry = sum(1 for e in entries if e["kind"] == "carry")
    n_left = sum(1 for e in entries if e["channel"] == "left")
    n_centre = sum(1 for e in entries if e["channel"] == "centre")
    n_right = sum(1 for e in entries if e["channel"] == "right")
    dom = max([("left", n_left), ("centre", n_centre), ("right", n_right)],
              key=lambda kv: kv[1])
    insight = (
        f"{team_name} entered the box {len(entries)} times "
        f"({n_pass} via pass, {n_carry} via carry). The dominant channel "
        f"was the {dom[0]} ({dom[1]} entries)."
    ) if entries else f"{team_name} — no successful box entries recorded."

    cards = [
        ("Total Entries", str(len(entries)), C_GOLD),
        ("Pass",          str(n_pass),       team_color),
        ("Carry",         str(n_carry),      C_GOLD),
        ("Left / Centre", f"{n_left} / {n_centre}", team_color),
        ("Right",         str(n_right),      C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="BOX ENTRIES",
        title=f"{team_name} — Box Entries",
        subtitle="Arrows show every successful entry into the opposition box "
                 "· gold = pass entry · green = carry entry",
        hn=team_name, an=opp_name, score=str(score),
        footer_note="Box = the 18-yard area",
        team_color=team_color, draw_overlay=draw_overlay,
        sidebar_title="Top Entry-Makers",
        sidebar_headers=["PLAYER", "ENTRIES"],
        sidebar_rows=rows,
        insight_text=insight, metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  HIGH TURNOVERS v2  (figs 36 / 37)
# ═════════════════════════════════════════════════════════════════════════
def make_high_turnovers_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name  = an if is_home else hn
    score = info.get("score") or "—"

    REGAIN_TYPES = {"Tackle", "Interception", "BallRecovery"}
    sub = events[(events["team_id"] == team_id) &
                 events["type"].isin(list(REGAIN_TYPES)) &
                 (events["x"] >= 60)]

    points = []; by_player = {}
    for _, r in sub.iterrows():
        x = float(_safe(r.get("x"), 70)); y = float(_safe(r.get("y"), 50))
        p = str(_safe(r.get("player"), "—"))
        points.append((x, y, str(r.get("type"))))
        by_player[p] = by_player.get(p, 0) + 1

    def draw_overlay(ax):
        # Shade the high-zone (x >= 60)
        ax.axvspan(60, 100, color=C_GOLD, alpha=0.04, zorder=0)
        for x, y, t in points:
            col = DEF_TYPE_COLORS.get(t, team_color)
            ax.scatter([x], [y], s=150, facecolor=col, edgecolor="white",
                       lw=1.0, alpha=0.85, zorder=4)

    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    rows = [(p.split()[-1] if p else "—", str(c)) for p, c in top]

    counts = {t: 0 for t in REGAIN_TYPES}
    for _, _, t in points:
        counts[t] = counts.get(t, 0) + 1
    leader = top[0][0].split()[-1] if top else "—"
    insight = (
        f"{team_name} regained possession {len(points)} times in the "
        f"final 40 metres — the press's tangible reward. {leader} led "
        f"with {top[0][1]} high turnovers."
    ) if top else f"{team_name} — no high regains recorded."

    cards = [
        ("High Regains", str(len(points)),                C_GOLD),
        ("Tackles",      str(counts.get("Tackle", 0)),    team_color),
        ("Intercepts",   str(counts.get("Interception", 0)), C_GOLD),
        ("Recoveries",   str(counts.get("BallRecovery", 0)), team_color),
        ("Top Player",   leader,                          C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="HIGH TURNOVERS",
        title=f"{team_name} — High Turnovers",
        subtitle="Each dot = a possession regain inside the final 40m · "
                 "colour = action type · gold band = high zone",
        hn=team_name, an=opp_name, score=str(score),
        footer_note="High = inside the opposition half (x ≥ 60)",
        team_color=team_color, draw_overlay=draw_overlay,
        sidebar_title="Top High-Pressers",
        sidebar_headers=["PLAYER", "REGAINS"],
        sidebar_rows=rows,
        insight_text=insight, metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  DANGER CREATION v2  (figs 10 / 11)
# ═════════════════════════════════════════════════════════════════════════
def make_danger_creation_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name  = an if is_home else hn
    score = info.get("score") or "—"

    sub = events[events["team_id"] == team_id]
    shots, kps, entries = [], [], []
    by_player = {}
    for _, r in sub.iterrows():
        x = float(_safe(r.get("x"), 50)); y = float(_safe(r.get("y"), 50))
        p = str(_safe(r.get("player"), "—"))
        if bool(r.get("is_shot")):
            shots.append((x, y, p))
            by_player[p] = by_player.get(p, 0) + 1
        if bool(r.get("is_key_pass")):
            kps.append((x, y, p))
            by_player[p] = by_player.get(p, 0) + 1
        # Box entry approximation
        if bool(r.get("is_pass")) and r.get("outcome") == "Successful":
            ex = float(_safe(r.get("end_x"), x))
            ey = float(_safe(r.get("end_y"), y))
            if ex >= 83 and 21 <= ey <= 79 and not (
                x >= 83 and 21 <= y <= 79
            ):
                entries.append((x, y, ex, ey, p))

    def draw_overlay(ax):
        # Box-entry arrows in faint gold (so they read as background)
        for sx, sy, ex, ey, _ in entries:
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=C_GOLD,
                                        lw=0.7, alpha=0.30,
                                        connectionstyle="arc3,rad=0.10"),
                        zorder=2)
        for x, y, _ in kps:
            ax.scatter([x], [y], s=110, marker="D",
                       facecolor=C_GOLD, edgecolor="white", lw=1.0,
                       alpha=0.85, zorder=4)
        for x, y, _ in shots:
            ax.scatter([x], [y], s=130, marker="o",
                       facecolor=team_color, edgecolor="white", lw=1.2,
                       alpha=0.92, zorder=5)

    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    rows = [(p.split()[-1] if p else "—", str(c)) for p, c in top]
    leader = top[0][0].split()[-1] if top else "—"
    insight = (
        f"{team_name} produced {len(shots)} shots, {len(kps)} key passes "
        f"and {len(entries)} successful box entries. {leader} was the "
        f"leading creator in danger zones."
    ) if top else f"{team_name} — no danger actions recorded."

    cards = [
        ("Shots",      str(len(shots)),   team_color),
        ("Key Passes", str(len(kps)),     C_GOLD),
        ("Box Entries", str(len(entries)), team_color),
        ("Touches",    str(len(shots) + len(kps) + len(entries)), C_GOLD),
        ("Top Creator", leader,           C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="DANGER CREATION",
        title=f"{team_name} — Danger Creation",
        subtitle="Circles = shots · diamonds = key passes · faint arrows "
                 "= box entries — every action that birthed a chance",
        hn=team_name, an=opp_name, score=str(score),
        footer_note="Cluster density reveals the side's true danger lanes",
        team_color=team_color, draw_overlay=draw_overlay,
        sidebar_title="Top Creators",
        sidebar_headers=["PLAYER", "ACT"],
        sidebar_rows=rows,
        insight_text=insight, metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  ZONE 14 + HALF-SPACES v2  (figs 14 / 15)
# ═════════════════════════════════════════════════════════════════════════
def make_zone14_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name  = an if is_home else hn
    score = info.get("score") or "—"

    # Zone 14: x ∈ [70, 83], y ∈ [37, 63]
    # Half-spaces:  x ∈ [60, 95], y ∈ [22, 37] OR [63, 78]
    Z14   = lambda x, y: 70 <= x <= 83 and 37 <= y <= 63
    HSL   = lambda x, y: 60 <= x <= 95 and 22 <= y <  37
    HSR   = lambda x, y: 60 <= x <= 95 and 63 <  y <= 78

    sub = events[(events["team_id"] == team_id) &
                 events["x"].notna() & events["y"].notna()]
    z14_pts, hs_pts = [], []
    by_player = {}
    for _, r in sub.iterrows():
        x = float(_safe(r.get("x"), 50)); y = float(_safe(r.get("y"), 50))
        p = str(_safe(r.get("player"), "—"))
        if Z14(x, y):
            z14_pts.append((x, y, p))
            by_player[p] = by_player.get(p, 0) + 1
        elif HSL(x, y) or HSR(x, y):
            hs_pts.append((x, y, p))
            by_player[p] = by_player.get(p, 0) + 0.5

    def draw_overlay(ax):
        # Zone 14 highlight
        ax.add_patch(mpatches.Rectangle((70, 37), 13, 26,
                      facecolor=C_GOLD, alpha=0.10, lw=0, zorder=0))
        ax.text(76.5, 65.5, "Z14", ha="center", color=C_GOLD,
                fontsize=8, fontweight="bold", alpha=0.85)
        # Half-spaces
        for y0 in (22, 63):
            ax.add_patch(mpatches.Rectangle((60, y0), 35, 15,
                          facecolor="#22c55e", alpha=0.07, lw=0, zorder=0))
        ax.text(78, 18, "Half-Space (L)", ha="center", color="#22c55e",
                fontsize=7, fontweight="bold", alpha=0.85)
        ax.text(78, 84, "Half-Space (R)", ha="center", color="#22c55e",
                fontsize=7, fontweight="bold", alpha=0.85)
        for x, y, _ in hs_pts:
            ax.scatter([x], [y], s=40, color="#22c55e",
                       edgecolor="white", lw=0.5, alpha=0.55, zorder=3)
        for x, y, _ in z14_pts:
            ax.scatter([x], [y], s=70, color=C_GOLD,
                       edgecolor="white", lw=0.8, alpha=0.85, zorder=4)

    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    rows = [(p.split()[-1] if p else "—", str(int(c))) for p, c in top]
    insight = (
        f"{team_name} touched the ball {len(z14_pts)} times in Zone 14 "
        f"and {len(hs_pts)} times in the half-spaces. Central-pocket "
        f"access is a leading indicator of chance creation."
    )
    cards = [
        ("Zone 14",     str(len(z14_pts)), C_GOLD),
        ("Half-Spaces", str(len(hs_pts)),  team_color),
        ("Total",       str(len(z14_pts) + len(hs_pts)), C_GOLD),
        ("Top Z14",     (top[0][0].split()[-1] if top else "—"), team_color),
        ("Top Count",   str(int(top[0][1])) if top else "0", C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="ZONE 14 + HALF-SPACES",
        title=f"{team_name} — Zone 14 & Half-Spaces",
        subtitle="Zone 14 is the gold rectangle (central pocket outside "
                 "the box) · green = the half-spaces flanking it",
        hn=team_name, an=opp_name, score=str(score),
        footer_note="More central-pocket touches → more chance creation",
        team_color=team_color, draw_overlay=draw_overlay,
        sidebar_title="Top Operators",
        sidebar_headers=["PLAYER", "ACT"],
        sidebar_rows=rows,
        insight_text=insight, metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  CROSSES v2  (figs 24 / 25)
# ═════════════════════════════════════════════════════════════════════════
def make_crosses_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name  = an if is_home else hn
    score = info.get("score") or "—"

    # Heuristic: cross = pass from x≥60 with end inside box and y near touchline
    sub = events[(events["team_id"] == team_id) &
                 (events["is_pass"] == True) &
                 events["x"].notna() & events["y"].notna() &
                 events["end_x"].notna() & events["end_y"].notna() &
                 (events["x"] >= 60) &
                 ((events["y"] <= 22) | (events["y"] >= 78)) &
                 (events["end_x"] >= 80)]

    crosses = []; by_player = {}
    for _, r in sub.iterrows():
        ok = (r.get("outcome") == "Successful")
        from_left = float(r["y"]) >= 78
        crosses.append({"sx": float(r["x"]), "sy": float(r["y"]),
                         "ex": float(r["end_x"]), "ey": float(r["end_y"]),
                         "ok": ok, "from_left": from_left,
                         "player": str(r.get("player") or "—")})
        if ok:
            p = str(r.get("player") or "—")
            by_player[p] = by_player.get(p, 0) + 1

    def draw_overlay(ax):
        for c in crosses:
            col = team_color if c["ok"] else "#64748b"
            alpha = 0.85 if c["ok"] else 0.35
            ax.annotate("", xy=(c["ex"], c["ey"]),
                        xytext=(c["sx"], c["sy"]),
                        arrowprops=dict(arrowstyle="->", color=col,
                                        lw=1.3, alpha=alpha,
                                        connectionstyle="arc3,rad=0.10"),
                        zorder=4)
        for x0, lbl, col in [(2, "Successful", team_color),
                              (32, "Unsuccessful", "#64748b")]:
            ax.annotate("", xy=(x0 + 8, -3), xytext=(x0, -3),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.4,
                                        alpha=0.9), clip_on=False)
            ax.text(x0 + 9.5, -3, lbl, ha="left", va="center",
                    color=TEXT_MAIN, fontsize=8, fontweight="bold",
                    clip_on=False)

    n = len(crosses)
    n_ok = sum(1 for c in crosses if c["ok"])
    n_left = sum(1 for c in crosses if c["from_left"])
    n_right = n - n_left
    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    rows = [(p.split()[-1] if p else "—", str(c)) for p, c in top]
    insight = (
        f"{team_name} attempted {n} crosses and completed {n_ok} "
        f"({(n_ok/n*100 if n else 0):.0f}%). Left flank: {n_left} · "
        f"right flank: {n_right}."
    )
    cards = [
        ("Total Crosses", str(n),   C_GOLD),
        ("Successful",    str(n_ok), team_color),
        ("Accuracy",      f"{(n_ok/n*100 if n else 0):.0f}%", C_GOLD),
        ("Left",          str(n_left), team_color),
        ("Right",         str(n_right), C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="CROSSES",
        title=f"{team_name} — Crosses",
        subtitle="Solid arrows = successful crosses · faded = unsuccessful "
                 "· flank reveals the side's wide-attack channel",
        hn=team_name, an=opp_name, score=str(score),
        footer_note="Cross = wide pass into the box from x ≥ 60",
        team_color=team_color, draw_overlay=draw_overlay,
        sidebar_title="Top Crossers",
        sidebar_headers=["PLAYER", "OK"],
        sidebar_rows=rows,
        insight_text=insight, metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  PROGRESSIVE PASSES v2  (figs 22 / 23)
# ═════════════════════════════════════════════════════════════════════════
def make_progressive_passes_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name  = an if is_home else hn
    score = info.get("score") or "—"

    sub = events[(events["team_id"] == team_id) &
                 (events["is_pass"] == True) &
                 (events["outcome"] == "Successful") &
                 events["x"].notna() & events["end_x"].notna()]
    progressives = []
    by_player = {}
    for _, r in sub.iterrows():
        sx, sy = float(r["x"]), float(r.get("y") or 50)
        ex, ey = float(r["end_x"]), float(r.get("end_y") or 50)
        # Progressive: reduces distance to goal by ≥25% OR ends in final third
        dist_before = 100 - sx
        dist_after  = 100 - ex
        is_progressive = (dist_before > 0 and
                          (dist_after / dist_before) <= 0.75) or \
                          (ex >= 67 and ex - sx >= 10)
        if is_progressive:
            progressives.append({"sx": sx, "sy": sy, "ex": ex, "ey": ey,
                                  "player": str(r.get("player") or "—")})
            p = str(r.get("player") or "—")
            by_player[p] = by_player.get(p, 0) + 1

    def draw_overlay(ax):
        for p in progressives:
            ax.annotate("", xy=(p["ex"], p["ey"]),
                        xytext=(p["sx"], p["sy"]),
                        arrowprops=dict(arrowstyle="->", color=team_color,
                                        lw=1.0, alpha=0.55,
                                        connectionstyle="arc3,rad=0.06"),
                        zorder=3)
        # Highlight top 10 by gain
        top10 = sorted(progressives,
                       key=lambda p: (p["ex"] - p["sx"]),
                       reverse=True)[:10]
        for p in top10:
            ax.annotate("", xy=(p["ex"], p["ey"]),
                        xytext=(p["sx"], p["sy"]),
                        arrowprops=dict(arrowstyle="->", color=C_GOLD,
                                        lw=1.8, alpha=0.95,
                                        connectionstyle="arc3,rad=0.08"),
                        zorder=5)

    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    rows = [(p.split()[-1] if p else "—", str(c)) for p, c in top]
    avg_gain = (np.mean([p["ex"] - p["sx"] for p in progressives])
                if progressives else 0)
    insight = (
        f"{team_name} played {len(progressives)} progressive passes — the "
        f"forward-progress engine. Top 10 by raw gain are highlighted in "
        f"gold. Average gain per pass: {avg_gain:.1f} m of pitch."
    )
    cards = [
        ("Progressives", str(len(progressives)), C_GOLD),
        ("Avg Gain",     f"{avg_gain:.1f}",      team_color),
        ("Top 10 Pass",  "10" if progressives else "0", C_GOLD),
        ("Top Player",   (top[0][0].split()[-1] if top else "—"), team_color),
        ("Top Count",    str(top[0][1]) if top else "0", C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="PROGRESSIVE PASSES",
        title=f"{team_name} — Progressive Passes",
        subtitle="Every pass that closed ≥25% of the distance to goal · "
                 "gold arrows = top-10 by raw forward gain",
        hn=team_name, an=opp_name, score=str(score),
        footer_note="Progressive = forward pass past the threshold",
        team_color=team_color, draw_overlay=draw_overlay,
        sidebar_title="Top Progressors",
        sidebar_headers=["PLAYER", "PROG"],
        sidebar_rows=rows,
        insight_text=insight, metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  PASS MAP BY THIRD v2  (figs 19 / 20)
# ═════════════════════════════════════════════════════════════════════════
def make_pass_thirds_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name  = an if is_home else hn
    score = info.get("score") or "—"

    sub = events[(events["team_id"] == team_id) &
                 (events["is_pass"] == True) &
                 events["x"].notna() & events["end_x"].notna()]
    def_p, mid_p, att_p = [], [], []
    n_def_ok = n_mid_ok = n_att_ok = 0
    by_player = {}
    for _, r in sub.iterrows():
        sx = float(r["x"]); sy = float(r.get("y") or 50)
        ex = float(r["end_x"]); ey = float(r.get("end_y") or 50)
        ok = r.get("outcome") == "Successful"
        rec = (sx, sy, ex, ey, ok)
        if sx < 33:
            def_p.append(rec); n_def_ok += int(ok)
        elif sx < 67:
            mid_p.append(rec); n_mid_ok += int(ok)
        else:
            att_p.append(rec); n_att_ok += int(ok)
        if ok:
            p = str(r.get("player") or "—")
            by_player[p] = by_player.get(p, 0) + 1

    def draw_overlay(ax):
        def_col = "#475569" if IS_LIGHT_THEME else "#64748b"
        att_col = "#B45309" if IS_LIGHT_THEME else C_GOLD
        for grp, col in [(def_p, def_col),
                          (mid_p, team_color),
                          (att_p, att_col)]:
            for sx, sy, ex, ey, ok in grp:
                a = 0.68 if ok else 0.32
                lw = 0.95 if ok else 0.65
                ax.annotate(
                    "", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(
                        arrowstyle="-|>", color=col, lw=lw, alpha=a,
                        mutation_scale=6.5 if ok else 5.0,
                        shrinkA=0, shrinkB=0,
                    ),
                    zorder=3,
                )
        # Third dividers
        div_col = "#334155" if IS_LIGHT_THEME else "white"
        ax.axvline(33, color=div_col, lw=0.8, ls="--", alpha=0.35, zorder=1)
        ax.axvline(67, color=div_col, lw=0.8, ls="--", alpha=0.35, zorder=1)
        ax.text(16.5, 95, "DEF",  color=TEXT_DIM, fontsize=7,
                fontweight="bold", ha="center")
        ax.text(50,   95, "MID",  color=TEXT_DIM, fontsize=7,
                fontweight="bold", ha="center")
        ax.text(83,   95, "ATT",  color=TEXT_DIM, fontsize=7,
                fontweight="bold", ha="center")

    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6]
    rows = [(p.split()[-1] if p else "—", str(c)) for p, c in top]
    n_total = len(def_p) + len(mid_p) + len(att_p)
    insight = (
        f"{team_name} played {n_total} passes — {len(def_p)} from the "
        f"defensive third, {len(mid_p)} from the middle, "
        f"{len(att_p)} from the attacking third. Final-third volume is "
        f"the cleanest read of break-down activity."
    )
    cards = [
        ("Total Passes",  str(n_total),    C_GOLD),
        ("Defensive 3rd", str(len(def_p)), team_color),
        ("Middle 3rd",    str(len(mid_p)), C_GOLD),
        ("Attacking 3rd", str(len(att_p)), team_color),
        ("Att 3rd Acc.",
         f"{(n_att_ok/len(att_p)*100 if att_p else 0):.0f}%", C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="PASS MAP BY THIRD",
        title=f"{team_name} — Pass Map by Third",
        subtitle="Grey = defensive-third passes · team colour = middle "
                 "third · gold = attacking third",
        hn=team_name, an=opp_name, score=str(score),
        footer_note="Final-third volume signals break-down efficiency",
        team_color=team_color, draw_overlay=draw_overlay,
        sidebar_title="Top Passers",
        sidebar_headers=["PLAYER", "OK"],
        sidebar_rows=rows,
        insight_text=insight, metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  PASS TARGET ZONES v2  (figs 38 / 39)
# ═════════════════════════════════════════════════════════════════════════
def make_pass_target_zones_v2(events, info, team_id, team_color):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    is_home = team_id == info.get("home_id")
    team_name = hn if is_home else an
    opp_name  = an if is_home else hn
    score = info.get("score") or "—"

    sub = events[(events["team_id"] == team_id) &
                 (events["is_pass"] == True) &
                 (events["outcome"] == "Successful") &
                 events["end_x"].notna() & events["end_y"].notna()]

    rows_n, cols_n = 6, 8
    cell_w = 100 / cols_n; cell_h = 100 / rows_n
    grid = np.zeros((rows_n, cols_n))
    for _, r in sub.iterrows():
        ex = float(r["end_x"]); ey = float(r["end_y"])
        c = min(int(ex // cell_w), cols_n - 1)
        rr = min(int(ey // cell_h), rows_n - 1)
        grid[rr, c] += 1

    def draw_overlay(ax):
        from matplotlib.colors import LinearSegmentedColormap as _LCM
        cmap = _LCM.from_list("pt", ["#0a1628", team_color])
        ax.imshow(grid, extent=[0, 100, 0, 100], origin="lower",
                  aspect="auto", cmap=cmap, alpha=0.78, zorder=1)
        # Top-3 cells get numeric label
        flat = [(grid[r, c], r, c)
                for r in range(rows_n) for c in range(cols_n)]
        for v, r, c in sorted(flat, reverse=True)[:5]:
            cx = (c + 0.5) * cell_w; cy = (r + 0.5) * cell_h
            ax.text(cx, cy, f"{int(v)}", ha="center", va="center",
                    color=TEXT_BR, fontsize=10, fontweight="bold",
                    path_effects=shadow(2), zorder=4)

    total = int(grid.sum())
    flat = [(grid[r, c], r, c)
            for r in range(rows_n) for c in range(cols_n)]
    flat.sort(reverse=True)
    hot = flat[0]
    hot_zone = ("attacking" if hot[2] >= cols_n * 2 / 3
                else ("middle" if hot[2] >= cols_n / 3
                      else "defensive")) + " third"
    insight = (
        f"{team_name} found targets {total} times. The hottest receiving "
        f"zone was in the {hot_zone} ({int(hot[0])} passes landed there)."
    )
    cards = [
        ("Targets Found", str(total), C_GOLD),
        ("Hottest Zone",  hot_zone.title()[:10], team_color),
        ("Top Cell",      str(int(hot[0])), C_GOLD),
        ("Att 3rd %",
         f"{(int(grid[:, cols_n*2//3:].sum())/total*100 if total else 0):.0f}%",
         team_color),
        ("Cells > 0",
         str(int((grid > 0).sum())), C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section="PASS TARGET ZONES",
        title=f"{team_name} — Pass Target Zones",
        subtitle="Heatmap of where successful passes landed · top-5 cells "
                 "show the raw count of passes received",
        hn=team_name, an=opp_name, score=str(score),
        footer_note="Where the team wanted the ball to arrive",
        team_color=team_color, draw_overlay=draw_overlay,
        sidebar_title="Receiving-Zone Notes",
        sidebar_headers=["WHERE", "PASSES"],
        sidebar_rows=[
            ("Att third",   str(int(grid[:, cols_n*2//3:].sum()))),
            ("Mid third",   str(int(grid[:, cols_n//3:cols_n*2//3].sum()))),
            ("Def third",   str(int(grid[:, :cols_n//3].sum()))),
            ("Top half (Y)", str(int(grid[rows_n//2:, :].sum()))),
            ("Bottom half", str(int(grid[:rows_n//2, :].sum()))),
        ],
        insight_text=insight, metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  BALL TOUCHES v2  (fig 18 — shared, uses both teams)
# ═════════════════════════════════════════════════════════════════════════
def make_ball_touches_v2(events, info, *,
                         section="BALL TOUCHES",
                         title_label="Ball Touches",
                         insight_intro=None):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    hid = info.get("home_id"); aid = info.get("away_id")
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)

    rows_n, cols_n = 6, 10
    cell_w = 100 / cols_n; cell_h = 100 / rows_n
    grid_h = np.zeros((rows_n, cols_n))
    grid_a = np.zeros((rows_n, cols_n))
    sub = events[events["x"].notna() & events["y"].notna()]
    for _, r in sub.iterrows():
        x = float(r["x"]); y = float(r["y"])
        c = min(int(x // cell_w), cols_n - 1)
        rr = min(int(y // cell_h), rows_n - 1)
        if r.get("team_id") == hid:
            grid_h[rr, c] += 1
        elif r.get("team_id") == aid:
            grid_a[rr, c] += 1

    def draw_overlay(ax):
        # Diff grid: positive = home, negative = away
        diff = grid_h - grid_a
        from matplotlib.colors import LinearSegmentedColormap as _LCM
        cmap = _LCM.from_list("dom", [ac, MID_BG, hc])
        vmax = max(abs(diff).max(), 1)
        ax.imshow(diff, extent=[0, 100, 0, 100], origin="lower",
                  aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                  alpha=0.65, zorder=1)
        # Number on EVERY cell that has at least one touch — top 5 get a
        # bigger, brighter label, the rest stay smaller and dimmer so the
        # eye still picks the hot cells first.
        flat = [(grid_h[r, c] + grid_a[r, c], r, c)
                for r in range(rows_n) for c in range(cols_n)]
        top5 = {(r, c) for _, r, c in sorted(flat, reverse=True)[:5]}
        for v, r, c in flat:
            if v <= 0:
                continue
            cx = (c + 0.5) * cell_w; cy = (r + 0.5) * cell_h
            is_top = (r, c) in top5
            ax.text(cx, cy, f"{int(v)}", ha="center", va="center",
                    color=TEXT_BR if is_top else TEXT_MAIN,
                    fontsize=9 if is_top else 7,
                    fontweight="bold",
                    path_effects=shadow(2 if is_top else 1.4),
                    zorder=4)

    n_h = int(grid_h.sum()); n_a = int(grid_a.sum())
    diff = n_h - n_a
    leader = hn if diff > 0 else an
    insight = (
        f"{leader} touched the ball {abs(diff)} more times overall. "
        f"Heatmap shows where each side dominated possession — the "
        f"colour at each cell points to the team with more touches there."
    )
    cards = [
        (f"{hn[:14]} Touches", str(n_h),         hc),
        ("Total",              str(n_h + n_a),   C_GOLD),
        (f"{an[:14]} Touches", str(n_a),         ac),
        ("Diff",               f"{'+' if diff >= 0 else ''}{diff}", C_GOLD),
        ("Leader",             leader[:10],      C_GOLD),
    ]
    return render_pitch_overlay_v2(
        section=section,
        title=f"{hn} vs {an} — {title_label}",
        subtitle="Each cell coloured by which team had more touches there "
                 "· top-5 cells show the raw combined touch count",
        hn=hn, an=an, score=str(score),
        footer_note="Where the game actually got played",
        team_color=C_GOLD, draw_overlay=draw_overlay,
        sidebar_title="Touch Distribution",
        sidebar_headers=["WHERE", "DIFF"],
        sidebar_rows=[
            ("Att 3rd (H–A)", f"{int(grid_h[:, cols_n*2//3:].sum() - grid_a[:, cols_n*2//3:].sum()):+d}"),
            ("Mid 3rd (H–A)", f"{int(grid_h[:, cols_n//3:cols_n*2//3].sum() - grid_a[:, cols_n//3:cols_n*2//3].sum()):+d}"),
            ("Def 3rd (H–A)", f"{int(grid_h[:, :cols_n//3].sum() - grid_a[:, :cols_n//3].sum()):+d}"),
            ("Total (H)",      str(n_h)),
            ("Total (A)",      str(n_a)),
        ],
        insight_text=insight, metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  TERRITORIAL CONTROL v2 + DOMINATING ZONE v2 (figs 17 / 33)
# ═════════════════════════════════════════════════════════════════════════
def make_territorial_v2(events, info):
    """Same engine as ball touches but framed as territorial control."""
    return make_ball_touches_v2(
        events, info,
        section="TERRITORIAL CONTROL",
        title_label="Territorial Control",
    )


def make_dominating_zone_v2(events, info):
    """Same engine but framed as 'who dominated each zone'."""
    return make_ball_touches_v2(
        events, info,
        section="DOMINATING ZONE",
        title_label="Dominating Zone",
    )


# ═════════════════════════════════════════════════════════════════════════
#  GENERIC BAR-COMPARISON v2 — used by Shot Comparison / xG Summary /
#  Defensive Summary (figs 9, 13, 30).
# ═════════════════════════════════════════════════════════════════════════
def render_bar_compare_v2(*, section, title, subtitle, hn, an, score,
                          footer_note, hc, ac, rows, insight_text,
                          metric_cards):
    """
    rows: list of (metric_label, home_value, away_value)
    """
    fig = plt.figure(figsize=(15, 9.5), facecolor=BG_DARK)
    chrome(fig, section=section, title=title, subtitle=subtitle,
           hn=hn, an=an, score=score, footer_note=footer_note)

    ax = panel_card(fig, 0.04, 0.20, 0.62, 0.62,
                    title="Side-by-side", accent=C_GOLD, body=False)
    n = len(rows)
    pos = np.arange(n)
    h_vals = [float(r[1]) for r in rows]
    a_vals = [float(r[2]) for r in rows]
    labels = [r[0] for r in rows]
    w = 0.38
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, max(h_vals + a_vals + [1]) * 1.30)
    for y in np.linspace(0, max(h_vals + a_vals + [1]) * 1.30, 6):
        ax.axhline(y, color=GRID_COL, lw=0.4, alpha=0.5, zorder=0)
    ax.bar(pos - w/2, h_vals, w, color=hc, alpha=0.9, lw=0, zorder=2)
    ax.bar(pos + w/2, a_vals, w, color=ac, alpha=0.9, lw=0, zorder=2)
    for i, (hv, av) in enumerate(zip(h_vals, a_vals)):
        h_col = C_GOLD if hv > av else TEXT_BR
        a_col = C_GOLD if av > hv else TEXT_BR
        ax.text(i - w/2, hv + 0.4, _fmt_num(hv), ha="center", va="bottom",
                color=h_col, fontsize=11, fontweight="bold",
                path_effects=shadow(2.4))
        ax.text(i + w/2, av + 0.4, _fmt_num(av), ha="center", va="bottom",
                color=a_col, fontsize=11, fontweight="bold",
                path_effects=shadow(2.4))
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, color=TEXT_MAIN, fontsize=9.5)
    ax.tick_params(axis="x", length=0, pad=8); ax.set_yticks([])
    for sp in ["top", "right", "left", "bottom"]:
        ax.spines[sp].set_visible(False)
    ax.text(0.02, -0.10, "● " + hn, color=hc, fontsize=10,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.20, -0.10, "● " + an, color=ac, fontsize=10,
            fontweight="bold", transform=ax.transAxes)

    key_insight(fig, 0.69, 0.30, 0.27, 0.50, text=insight_text, wrap=34)
    metric_strip(fig, cards=metric_cards)
    return fig


def _fmt_num(v):
    try:
        f = float(v)
        if f.is_integer():
            return str(int(f))
        return f"{f:.2f}"
    except Exception:
        return str(v)


def make_shot_comparison_v2(events, info, xg_data):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)
    h = (xg_data or {}).get(hn, {})
    a = (xg_data or {}).get(an, {})
    rows = [
        ("Total Shots", h.get("shots", 0),       a.get("shots", 0)),
        ("On Target",   h.get("on_target", 0),   a.get("on_target", 0)),
        ("Big Chances", h.get("big_chances", 0), a.get("big_chances", 0)),
        ("xG",          float(h.get("xG", 0) or 0),
                        float(a.get("xG", 0) or 0)),
        ("xGoT",        float(h.get("xGoT", 0) or 0),
                        float(a.get("xGoT", 0) or 0)),
    ]
    diff = float(h.get("xG", 0) or 0) - float(a.get("xG", 0) or 0)
    leader = hn if diff > 0 else an
    insight = (
        f"{leader} produced the stronger shooting profile — leading on "
        f"{sum(1 for r in rows if r[1] > r[2])} of the {len(rows)} "
        f"metrics. Total xG difference: "
        f"{'+' if diff >= 0 else ''}{diff:.2f}."
    )
    cards = [
        (f"{hn[:14]} xG", _fmt_num(h.get("xG", 0)), hc),
        ("xG Diff",       f"{'+' if diff >= 0 else ''}{diff:.2f}", C_GOLD),
        (f"{an[:14]} xG", _fmt_num(a.get("xG", 0)), ac),
        ("Total Shots",   str(h.get("shots", 0) + a.get("shots", 0)),
         C_GOLD),
        ("Total OT",      str(h.get("on_target", 0) +
                              a.get("on_target", 0)),               C_GOLD),
    ]
    return render_bar_compare_v2(
        section="SHOT COMPARISON",
        title=f"{hn} vs {an} — Shot Comparison",
        subtitle="Five headline shooting metrics side-by-side · gold "
                 "label = the metric leader",
        hn=hn, an=an, score=str(score),
        footer_note="Read top-to-bottom: who created the better profile?",
        hc=hc, ac=ac, rows=rows,
        insight_text=insight, metric_cards=cards,
    )


def make_xg_summary_v2(events, info, xg_data):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)
    h = (xg_data or {}).get(hn, {})
    a = (xg_data or {}).get(an, {})
    rows = [
        ("xG",         float(h.get("xG", 0) or 0),
                       float(a.get("xG", 0) or 0)),
        ("xGoT",       float(h.get("xGoT", 0) or 0),
                       float(a.get("xGoT", 0) or 0)),
        ("Goals",      h.get("goals", 0), a.get("goals", 0)),
        ("Big Chances", h.get("big_chances", 0), a.get("big_chances", 0)),
    ]
    h_xg = float(h.get("xG", 0) or 0)
    a_xg = float(a.get("xG", 0) or 0)
    h_g = h.get("goals", 0); a_g = a.get("goals", 0)
    over_h = h_g - h_xg
    over_a = a_g - a_xg
    insight = (
        f"xG: {hn} {h_xg:.2f} vs {an} {a_xg:.2f}. Finishing performance: "
        f"{hn} {'+' if over_h >= 0 else ''}{over_h:.2f} vs xG, "
        f"{an} {'+' if over_a >= 0 else ''}{over_a:.2f}. "
        f"xGoT shows post-shot finishing quality."
    )
    cards = [
        (f"{hn[:14]} xG",   f"{h_xg:.2f}",  hc),
        (f"{hn[:14]} Goals", str(h_g),       C_GOLD),
        ("Goals - xG (H/A)",
         f"{over_h:+.1f}/{over_a:+.1f}",     C_GOLD),
        (f"{an[:14]} Goals", str(a_g),       C_GOLD),
        (f"{an[:14]} xG",   f"{a_xg:.2f}",  ac),
    ]
    return render_bar_compare_v2(
        section="xG / xGoT SUMMARY",
        title=f"{hn} vs {an} — xG and xGoT",
        subtitle="xG measures pre-shot quality · xGoT measures post-shot "
                 "placement & power · gap to goals = finishing variance",
        hn=hn, an=an, score=str(score),
        footer_note="Below xG = wasteful · above xG = clinical",
        hc=hc, ac=ac, rows=rows,
        insight_text=insight, metric_cards=cards,
    )


def make_defensive_summary_v2(events, info):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    hid = info.get("home_id"); aid = info.get("away_id")
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)

    def _count(team_id, type_name):
        return int(((events["team_id"] == team_id) &
                    (events["type"] == type_name)).sum())

    rows = [
        ("Tackles",       _count(hid, "Tackle"),       _count(aid, "Tackle")),
        ("Interceptions", _count(hid, "Interception"), _count(aid, "Interception")),
        ("Clearances",    _count(hid, "Clearance"),    _count(aid, "Clearance")),
        ("Blocks",        _blocked_shots_for_team(events, info, hid),
                          _blocked_shots_for_team(events, info, aid)),
        ("Recoveries",    _count(hid, "BallRecovery"), _count(aid, "BallRecovery")),
        ("Fouls",         _count(hid, "Foul"),         _count(aid, "Foul")),
    ]
    h_total = sum(r[1] for r in rows)
    a_total = sum(r[2] for r in rows)
    leader = hn if h_total > a_total else an
    insight = (
        f"{leader} did more defensive work overall ({max(h_total, a_total)} "
        f"vs {min(h_total, a_total)}). Tackles + interceptions describe "
        f"duels; recoveries + clearances mark how each side escaped pressure."
    )
    cards = [
        (f"{hn[:14]} Total", str(h_total), hc),
        ("Diff",
         f"{'+' if h_total - a_total >= 0 else ''}{h_total - a_total}",
         C_GOLD),
        (f"{an[:14]} Total", str(a_total), ac),
        ("Top H Type",
         max(rows, key=lambda r: r[1])[0][:10], C_GOLD),
        ("Top A Type",
         max(rows, key=lambda r: r[2])[0][:10], C_GOLD),
    ]
    return render_bar_compare_v2(
        section="DEFENSIVE SUMMARY",
        title=f"{hn} vs {an} — Defensive Summary",
        subtitle="Six defensive-action types — head-to-head counts · gold "
                 "label = the side with more of that action",
        hn=hn, an=an, score=str(score),
        footer_note="Tackle = duel · Interception = anticipation · "
                    "Recovery = loose-ball control",
        hc=hc, ac=ac, rows=rows,
        insight_text=insight, metric_cards=cards,
    )


# ═════════════════════════════════════════════════════════════════════════
#  xT PER MINUTE v2  (fig 21 — diverging bars)
# ═════════════════════════════════════════════════════════════════════════
def make_xt_per_minute_v2(events, info):
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    hid = info.get("home_id"); aid = info.get("away_id")
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)

    if "xT" not in events.columns:
        # Fallback empty visual
        fig = plt.figure(figsize=(15, 9.5), facecolor=BG_DARK)
        chrome(fig, section="xT PER MINUTE",
               title=f"{hn} vs {an} — xT per Minute",
               subtitle="No xT data in this dataset",
               hn=hn, an=an, score=str(score),
               footer_note="—")
        ax = fig.add_axes([0.05, 0.30, 0.92, 0.5])
        ax.set_facecolor(BG_MID)
        ax.text(0.5, 0.5, "No xT data", ha="center", va="center",
                color=TEXT_DIM, fontsize=14, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    xt = events[events["xT"].notna() & (events["xT"] > 0) &
                (events["outcome"] == "Successful")].copy()
    h_min = xt[xt["team_id"] == hid].groupby("minute")["xT"].sum()
    a_min = xt[xt["team_id"] == aid].groupby("minute")["xT"].sum()
    mins = list(range(1, 95))
    h_vals = [float(h_min.get(m, 0))   for m in mins]
    a_vals = [-float(a_min.get(m, 0))  for m in mins]

    fig = plt.figure(figsize=(15, 9.5), facecolor=BG_DARK)
    chrome(fig, section="xT PER MINUTE",
           title=f"{hn} vs {an} — xT per Minute",
           subtitle="Diverging bars: home xT rises above zero · away xT "
                    "drops below · curves are 5-min rolling averages",
           hn=hn, an=an, score=str(score),
           footer_note="Spikes = momentum windows")

    ax = fig.add_axes([0.05, 0.20, 0.62, 0.62])
    ax.set_facecolor(BG_MID)
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL); s.set_linewidth(0.5)
    ax.bar(mins, h_vals, color=hc, alpha=0.78, width=0.85, zorder=3)
    ax.bar(mins, a_vals, color=ac, alpha=0.78, width=0.85, zorder=3)
    ax.axhline(0, color="#94a3b8", lw=0.9, alpha=0.6, zorder=4)

    import pandas as _pd
    _hv = _pd.Series(h_vals).rolling(5, center=True, min_periods=1).mean()
    _av = _pd.Series(a_vals).rolling(5, center=True, min_periods=1).mean()
    ax.plot(mins, _hv, color=hc, lw=2.0, alpha=0.92, zorder=5)
    ax.plot(mins, _av, color=ac, lw=2.0, alpha=0.92, zorder=5)

    ymax = max(max(h_vals + [0.001]), abs(min(a_vals + [-0.001])))
    for xv, lb in [(45, "HT"), (90, "FT")]:
        ax.axvline(xv, color=C_GOLD, lw=0.9, ls="--", alpha=0.45, zorder=2)
        ax.text(xv, ymax * 0.95, lb, ha="center", va="top",
                color=C_GOLD, fontsize=8, fontweight="bold")
    ax.set_xlim(0, 95)
    ax.set_xlabel("Minute", color=TEXT_DIM, fontsize=9)
    ax.set_ylabel("xT  (▲ Home  |  ▼ Away)", color=TEXT_DIM, fontsize=9)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    ht = float(h_min.sum()); at = float(a_min.sum())
    leader = hn if ht > at else an
    diff = abs(ht - at)
    # Hottest 5-min for each side
    def _best(values, w=5):
        if not any(values):
            return (0.0, 0, 0)
        best = (0.0, 0, 0)
        for s in range(0, len(values) - w + 1):
            x = sum(values[s:s+w])
            if x > best[0]:
                best = (x, s, s + w)
        return best
    bw_h = _best(h_vals)
    bw_a = _best([-v for v in a_vals])
    insight = (
        f"{leader} created {diff:.2f} more xT over 90 minutes. "
        f"{hn}'s hottest 5-min: {bw_h[1]:02d}'–{bw_h[2]:02d}' "
        f"({bw_h[0]:.2f} xT). {an}'s hottest 5-min: "
        f"{bw_a[1]:02d}'–{bw_a[2]:02d}' ({bw_a[0]:.2f} xT)."
    )
    key_insight(fig, 0.69, 0.30, 0.27, 0.52, text=insight, wrap=34)

    cards = [
        (f"{hn[:14]} xT", f"{ht:.2f}",          hc),
        ("Diff",          f"{'+' if ht-at >= 0 else ''}{ht-at:.2f}", C_GOLD),
        (f"{an[:14]} xT", f"{at:.2f}",          ac),
        ("Hottest H 5'",  f"{bw_h[0]:.2f}",     C_GOLD),
        ("Hottest A 5'",  f"{bw_a[0]:.2f}",     C_GOLD),
    ]
    metric_strip(fig, cards=cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  GOALKEEPER SAVES v2  (fig 12)
# ═════════════════════════════════════════════════════════════════════════
def make_gk_saves_v2(events, info):
    """Plots every shot each keeper faced + outcome."""
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    hid = info.get("home_id"); aid = info.get("away_id")
    score = info.get("score") or "—"
    hc, ac = _match_colors(info)

    fig = plt.figure(figsize=(16, 10), facecolor=BG_DARK)
    chrome(fig, section="GOALKEEPER SAVES",
           title=f"{hn} vs {an} — Keeper Saves",
           subtitle="Each dot is a shot the keeper faced · size = xG · "
                    "filled = goal · ringed = saved",
           hn=hn, an=an, score=str(score),
           footer_note="Save quality scales with the xG of shots faced")

    # Two pitches side-by-side
    for i, (team_id, opp_id, team_name, team_color, x0, w) in enumerate([
        (aid, hid, hn + "'s keeper", hc, 0.04, 0.45),
        (hid, aid, an + "'s keeper", ac, 0.51, 0.45),
    ]):
        ax = fig.add_axes([x0, 0.20, w, 0.62])
        themed_pitch(ax, attacking_only=True)
        ax.set_xlim(48, 102)
        # Shots THE OPPONENT took (= shots THIS keeper faced)
        sub = events[(events["team_id"] == opp_id) &
                     (events["is_shot"] == True)]
        n_total = len(sub); n_goal = 0; n_save = 0
        xg_faced = 0.0
        for _, r in sub.iterrows():
            x = float(_safe(r.get("x"), 80))
            y = float(_safe(r.get("y"), 50))
            xg = float(_safe(r.get("xG"), 0) or 0)
            xg_faced += xg
            stype = (r.get("shot_whoscored_type") or r.get("type") or "")
            if bool(r.get("is_goal")):
                ax.scatter([x], [y], s=80 + xg * 1500, marker="*",
                           color=team_color, edgecolor="white",
                           lw=1.5, alpha=0.95, zorder=5)
                n_goal += 1
            elif stype == "SavedShot":
                ax.scatter([x], [y], s=60 + xg * 1300, facecolor="none",
                           edgecolor=team_color, lw=2.0, alpha=0.85,
                           zorder=4)
                n_save += 1
            else:
                ax.scatter([x], [y], s=40 + xg * 1100, facecolor="none",
                           edgecolor="#64748b", lw=1.0, alpha=0.5,
                           zorder=3)
        # Per-keeper title
        fig.text(x0 + w/2, 0.84, team_name, ha="center", va="center",
                 color=team_color, fontsize=12, fontweight="bold")
        fig.text(x0 + w/2, 0.815,
                 f"Faced {n_total} shots ({xg_faced:.2f} xG) · "
                 f"saved {n_save} · conceded {n_goal}",
                 ha="center", va="center", color=TEXT_DIM, fontsize=9)

    # Bottom strip
    h_faced = events[(events["team_id"] == aid) &
                     (events["is_shot"] == True)]
    a_faced = events[(events["team_id"] == hid) &
                     (events["is_shot"] == True)]
    h_xg_faced = float(h_faced["xG"].fillna(0).sum()
                        if not h_faced.empty else 0)
    a_xg_faced = float(a_faced["xG"].fillna(0).sum()
                        if not a_faced.empty else 0)
    h_saves = int(((h_faced.get("shot_whoscored_type") == "SavedShot")
                   if "shot_whoscored_type" in h_faced.columns
                   else (h_faced["type"] == "SavedShot")).sum())
    a_saves = int(((a_faced.get("shot_whoscored_type") == "SavedShot")
                   if "shot_whoscored_type" in a_faced.columns
                   else (a_faced["type"] == "SavedShot")).sum())
    cards = [
        (f"{hn[:14]} xG faced", f"{h_xg_faced:.2f}", hc),
        (f"{hn[:14]} Saves",    str(h_saves),         C_GOLD),
        ("Total Shots",         str(len(h_faced) + len(a_faced)), C_GOLD),
        (f"{an[:14]} Saves",    str(a_saves),         C_GOLD),
        (f"{an[:14]} xG faced", f"{a_xg_faced:.2f}", ac),
    ]
    metric_strip(fig, cards=cards)
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  MATCH STATISTICS v2  (fig 16) — wraps the existing extension page
# ═════════════════════════════════════════════════════════════════════════
def render_legacy_chart_v2(
    *, section, title, subtitle, hn, an, score, footer_note,
    team_color, draw_legacy,
    sidebar_title, sidebar_headers, sidebar_rows,
    insight_text, metric_cards,
):
    """
    Hybrid layout: v2 chrome + sidebar + bottom strip, but the central chart
    is drawn by a LEGACY panel function (the user-preferred legacy look for
    Pass Target Zones, GK Saves, Ball Touches, High Turnovers).

    `draw_legacy(fig, ax)` is a callback that paints the chart on `ax`.
    """
    fig = plt.figure(figsize=(16, 10), facecolor=BG_DARK)
    chrome(fig, section=section, title=title, subtitle=subtitle,
           hn=hn, an=an, score=score, footer_note=footer_note)

    # Reserve the same left area as render_pitch_overlay_v2, so the sidebar
    # + insight + metric strip fall into the familiar v2 grid.
    ax = fig.add_axes([0.04, 0.18, 0.62, 0.66])
    ax.set_facecolor(BG_MID)
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL); s.set_linewidth(0.5)
    try:
        draw_legacy(fig, ax)
    except Exception:
        ax.text(0.5, 0.5, "Chart unavailable", ha="center", va="center",
                color=TEXT_DIM, fontsize=14, transform=ax.transAxes)

    # Sidebar
    ax2 = panel_card(fig, 0.69, 0.50, 0.27, 0.34,
                     title=sidebar_title, accent=team_color)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    n_cols = len(sidebar_headers)
    if n_cols == 1:
        xs = [0.05]
    elif n_cols == 2:
        xs = [0.05, 0.93]
    else:
        xs = [0.05] + [0.05 + (i * 0.88 / (n_cols - 1))
                       for i in range(1, n_cols)]
    for i, (lbl, x) in enumerate(zip(sidebar_headers, xs)):
        ha = "left" if i == 0 else ("right" if i == n_cols - 1 else "center")
        ax2.text(x, 0.83, lbl, ha=ha, va="center",
                 color=TEXT_DIM, fontsize=9, fontweight="bold",
                 transform=ax2.transAxes)
    ax2.plot([0.04, 0.96], [0.79, 0.79], color=GRID_COL, lw=0.6,
             transform=ax2.transAxes)
    if sidebar_rows:
        n = max(len(sidebar_rows), 1)
        rh = 0.70 / n
        for i, row in enumerate(sidebar_rows):
            cy = 0.74 - (i + 0.5) * rh
            if i % 2 == 0:
                ax2.add_patch(mpatches.Rectangle(
                    (0.04, cy - rh*0.42), 0.92, rh*0.84,
                    facecolor=ROW_BG, lw=0,
                    transform=ax2.transAxes, zorder=1))
            for j, (val, x) in enumerate(zip(row, xs)):
                ha = "left" if j == 0 else ("right" if j == n_cols - 1
                                              else "center")
                col = TEXT_BR if j == 0 else (
                    team_color if j == n_cols - 1 else TEXT_MAIN)
                ax2.text(x, cy, str(val), ha=ha, va="center",
                         color=col, fontsize=10.5 if j != n_cols - 1 else 11,
                         fontweight="bold",
                         transform=ax2.transAxes, zorder=2)

    key_insight(fig, 0.69, 0.16, 0.27, 0.28, text=insight_text, wrap=34)
    metric_strip(fig, cards=metric_cards)
    return fig


def make_match_stats_v2(events, info, ppda):
    """
    Reuses the polished _draw_team_stats_compare_page from match_extensions
    (which already uses the unified identity + 'Reading this page' panel).
    Captures the figure it builds via a tiny Pdf-like shim and prevents the
    inner plt.close from disposing it before we can save it later.
    """
    from match_extensions import _draw_team_stats_compare_page

    captured = []

    class _Capture:
        def savefig(self, fig, **kw):
            captured.append(fig)

    # Temporarily neutralise plt.close so the captured fig stays alive.
    _orig_close = plt.close
    plt.close = lambda *a, **k: None
    try:
        _draw_team_stats_compare_page(_Capture(), info, events, ppda)
    finally:
        plt.close = _orig_close
    return captured[-1] if captured else plt.figure(facecolor=BG_DARK)

