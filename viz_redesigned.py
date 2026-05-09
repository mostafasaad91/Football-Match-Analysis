"""
viz_redesigned.py
═════════════════════════════════════════════════════════════════════════════
Redesigned versions of the original visuals using viz_design_system.

Each function takes the same inputs as the original function in
Match_Analysis_Dark.py (events DataFrame, info dict, etc.) but returns a figure
designed with the new visual identity.

Redesigned visuals (first batch):
  1. shot_map_v2(events, info, team_id, team_color, team_name, accent)
  2. xg_flow_v2(events, info, xg_data)
  3. pass_network_v2(events, info, team_id, team_color, team_name, accent)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

from viz_design_system import (
    BG_DARK, BG_MID, BG_PANEL, BG_PITCH, GRID_COL,
    TEXT_MAIN, TEXT_BRIGHT, TEXT_DIM, TEXT_FADED,
    C_HOME, C_AWAY, C_GOLD, C_GREEN, C_PURPLE, OG_COLOR,
    apply_unified_frame, make_themed_figure,
    themed_pitch, themed_panel, metric_strip,
    legend_chips, side_panel, _shadow,
)


def _short(name: str, n: int = 18) -> str:
    if not name:
        return ""
    if len(name) <= n:
        return name
    parts = str(name).split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[-1]}"[:n]
    return str(name)[:n - 1] + "…"


# ═════════════════════════════════════════════════════════════════════════════
# 1) SHOT MAP — Re-designed
# ═════════════════════════════════════════════════════════════════════════════
def shot_map_v2(events: pd.DataFrame, info: dict, team_id: int,
                team_color: str, team_name: str,
                accent: str = C_GOLD,
                save_path: str | None = None):
    """
    Redesigned Shot Map:
      • Unified header (colour bar + section name + title + subtitle)
      • Attacking-half-only pitch (half pitch + 18-yard area)
      • Distinct markers per shot outcome (Goal/Saved/Missed/Blocked/Post)
      • Metric strip beneath the pitch: Total / Goals / xG / SoT / Big Ch
      • Side panel listing the top 5 shooters
      • Unified legend chips
    """
    fig = make_themed_figure(15, 9.5)
    apply_unified_frame(
        fig,
        section="SHOT MAP",
        title=f"{team_name} — Shot Map & Quality",
        subtitle="Marker shape = shot result   ·   marker size scales with xG",
        accent=accent,
        home_name=info.get("home_name"),
        away_name=info.get("away_name"),
        score=info.get("score"),
        footer_note="Goal=★ · SavedShot=● · MissedShots=✕ · BlockedShot=■ · ShotOnPost=◆",
    )

    # ── Pitch axes () ──
    pitch_ax = fig.add_axes([0.04, 0.16, 0.66, 0.70])
    themed_pitch(pitch_ax, attacking_only=False)

    shots = events[(events["is_shot"] == True) &
                   (events["team_id"] == team_id)].copy()  # noqa: E712

    # (18-yard box)
    pitch_ax.add_patch(mpatches.Rectangle(
        (83.5, 21.1), 16.5, 57.8,
        facecolor=accent, alpha=0.06, lw=0, zorder=1))

    SHOT_STYLE = {
        "Goal":         dict(marker="*", face=accent,    size=460, label="Goal"),
        "SavedShot":    dict(marker="o", face=C_GREEN,   size=180, label="SavedShot"),
        "MissedShots":  dict(marker="X", face="#f87171", size=160, label="MissedShots"),
        "BlockedShot":  dict(marker="s", face="#fbbf24", size=160, label="BlockedShot"),
        "ShotOnPost":   dict(marker="D", face=C_PURPLE,  size=170, label="ShotOnPost"),
    }

    counts = {k: 0 for k in SHOT_STYLE}
    if not shots.empty:
        raw_col = ("shot_whoscored_type" if "shot_whoscored_type"
                   in shots.columns else "shot_category")
        for raw, style in SHOT_STYLE.items():
            sub = shots[shots[raw_col] == raw]
            counts[raw] = len(sub)
            for _, r in sub.iterrows():
                xg = float(r.get("xG") or 0)
                size = style["size"] + xg * (1300 if raw == "Goal" else 800)
                pitch_ax.scatter(
                    r["x"], r["y"],
                    s=size, marker=style["marker"],
                    c=style["face"], edgecolor="white", linewidths=1.4,
                    alpha=0.92, zorder=5,
                )
                if raw == "Goal":
                    is_og = bool(r.get("is_own_goal", False))
                    lbl_col = OG_COLOR if is_og else accent
                    label = _short(str(r.get("player") or ""), 16)
                    if is_og:
                        label += "  (OG)"
                    pitch_ax.text(
                        r["x"], r["y"] + 5.5, label,
                        ha="center", va="bottom",
                        color=lbl_col, fontsize=8.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.28",
                                  facecolor="#000000", edgecolor=lbl_col,
                                  alpha=0.85, lw=0.8),
                        zorder=6,
                    )
                    pitch_ax.text(
                        r["x"], r["y"] - 6.5, f"xG {xg:.2f}",
                        ha="center", va="top",
                        color=accent, fontsize=8, fontweight="bold",
                        path_effects=_shadow(2),
                        zorder=6,
                    )

    # direction arrow
    pitch_ax.annotate("", xy=(95, 5), xytext=(60, 5),
                      arrowprops=dict(arrowstyle="->", color=team_color,
                                      lw=2, alpha=0.6))
    pitch_ax.text(77.5, 7.5, f"{team_name} attacking →",
                  color=team_color, fontsize=9, fontweight="bold",
                  alpha=0.7)

    # legend chips
    leg_items = [(SHOT_STYLE[k]["label"] + f"  ({counts[k]})",
                  SHOT_STYLE[k]["face"], SHOT_STYLE[k]["marker"])
                 for k in ["Goal", "SavedShot", "MissedShots",
                           "BlockedShot", "ShotOnPost"]]
    legend_chips(pitch_ax, leg_items, y=-0.04)

    # ── Metric strip ( side panel) ──
    total = len(shots) if not shots.empty else 0
    goals = counts.get("Goal", 0)
    sot = counts.get("Goal", 0) + counts.get("SavedShot", 0)
    tot_xg = float(shots["xG"].fillna(0).sum()) if not shots.empty else 0.0
    big_ch = (int(shots["big_chance"].sum())
              if not shots.empty and "big_chance" in shots.columns else 0)

    metric_strip(
        fig, x=0.73, y=0.78, w=0.24, h=0.10,
        metrics=[
            ("Shots", total, TEXT_BRIGHT),
            ("Goals", goals, accent),
            ("On Target", sot, C_GREEN),
        ],
    )
    metric_strip(
        fig, x=0.73, y=0.66, w=0.24, h=0.10,
        metrics=[
            ("Total xG", f"{tot_xg:.2f}", accent),
            ("Big Ch.", big_ch, C_PURPLE),
        ],
    )

    # ── Side panel: top scorers/shooters ──
    rows: list[tuple[str, str, str]] = []
    if not shots.empty:
        agg = (shots.groupby("player")
               .agg(shots=("is_shot", "sum"),
                    xg=("xG", lambda v: float(pd.Series(v).fillna(0).sum())),
                    goals=("is_goal", "sum"))
               .sort_values(["goals", "xg", "shots"], ascending=False)
               .head(6))
        for player, row in agg.iterrows():
            label = _short(str(player), 16)
            val = (f"{int(row['shots'])} sh  ·  "
                   f"{row['xg']:.2f} xG  ·  {int(row['goals'])} G")
            color = accent if row["goals"] > 0 else TEXT_MAIN
            rows.append((label, val, color))
    if not rows:
        rows = [("—", "no shots", TEXT_FADED)]

    side_panel(fig, x=0.73, y=0.16, w=0.24, h=0.46,
               title="Top Shooters", rows=rows, accent=accent)

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight",
                    facecolor=BG_DARK)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 2) xG FLOW — Re-designed
# ═════════════════════════════════════════════════════════════════════════════
def xg_flow_v2(events: pd.DataFrame, info: dict,
               xg_data: dict | None = None,
               save_path: str | None = None):
    """
    xG Flow timeline:
      •
      • Cumulative xG curve per team (step plot)
      • shot markers on the curve (coloured goals)
      • shaded area under each curve
      • metric strip for the final score
      • verdict below
    """
    fig = make_themed_figure(15, 8.5)
    home = info.get("home_name") or "Home"
    away = info.get("away_name") or "Away"

    apply_unified_frame(
        fig,
        section="xG FLOW",
        title="Expected Goals — Match Timeline",
        subtitle="Cumulative xG over match minutes  ·  steps mark each shot",
        accent=C_GOLD,
        home_name=home, away_name=away, score=info.get("score"),
        footer_note="Stars = goals  ·  shaded area = territory under each curve",
    )

    # ── Plot axes ──
    ax = fig.add_axes([0.07, 0.18, 0.88, 0.62])
    themed_panel(ax)
    ax.grid(True, color=GRID_COL, lw=0.5, alpha=0.5, axis="both")
    ax.set_axisbelow(True)

    shots = events[events["is_shot"] == True].copy()  # noqa: E712
    if shots.empty:
        ax.text(0.5, 0.5, "No shots recorded", ha="center", va="center",
                color=TEXT_DIM, fontsize=14, style="italic",
                transform=ax.transAxes)
        if save_path:
            fig.savefig(save_path, dpi=160, bbox_inches="tight",
                        facecolor=BG_DARK)
        return fig

    shots["xG"] = shots["xG"].fillna(0).astype(float)
    shots = shots.sort_values("minute")

    home_id = info.get("home_id")
    away_id = info.get("away_id")
    max_minute = max(int(shots["minute"].max() or 90), 90)

    def _cum(team_id):
        d = shots[shots["team_id"] == team_id]
        xs = [0] + list(d["minute"].astype(float))
        ys = np.cumsum([0] + list(d["xG"]))
        return xs + [max_minute], list(ys) + [ys[-1] if len(ys) else 0]

    hx, hy = _cum(home_id)
    ax_xs, ay_s = _cum(away_id)

    # + shaded area
    ax.fill_between(hx, 0, hy, step="post", color=C_HOME, alpha=0.18, zorder=2)
    ax.fill_between(ax_xs, 0, ay_s, step="post",
                    color=C_AWAY, alpha=0.18, zorder=2)
    ax.step(hx, hy, where="post", color=C_HOME, lw=2.4,
            label=home, zorder=4, path_effects=_shadow(2))
    ax.step(ax_xs, ay_s, where="post", color=C_AWAY, lw=2.4,
            label=away, zorder=4, path_effects=_shadow(2))

    
    for _, r in shots.iterrows():
        is_goal = bool(r.get("is_goal", False))
        is_og = bool(r.get("is_own_goal", False))
        team = r["team_id"]
        # cumulative xG
        if team == home_id:
            cum_y = sum(shots[(shots["team_id"] == home_id) &
                              (shots["minute"] <= r["minute"])]["xG"])
            color = C_HOME
        else:
            cum_y = sum(shots[(shots["team_id"] == away_id) &
                              (shots["minute"] <= r["minute"])]["xG"])
            color = C_AWAY
        if is_goal:
            ec = OG_COLOR if is_og else color
            ax.scatter([r["minute"]], [cum_y],
                       marker="*", s=320, c=C_GOLD,
                       edgecolor=ec, linewidths=1.8, zorder=8)
            ax.text(r["minute"], cum_y + 0.05,
                    _short(str(r.get("player") or ""), 14),
                    ha="center", va="bottom",
                    color=C_GOLD, fontsize=8.5, fontweight="bold",
                    path_effects=_shadow(2), zorder=9)

    ax.set_xlim(0, max_minute)
    ax.set_ylim(0, max(max(hy), max(ay_s)) * 1.18 + 0.1)
    ax.set_xlabel("Minute", color=TEXT_DIM, fontsize=10)
    ax.set_ylabel("Cumulative xG", color=TEXT_DIM, fontsize=10)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)

    
    for mark, lbl in [(45, "HT"), (90, "FT")]:
        if mark <= max_minute:
            ax.axvline(mark, color=TEXT_FADED, ls="--", lw=0.8, alpha=0.6,
                       zorder=3)
            ax.text(mark, ax.get_ylim()[1] * 0.97, lbl,
                    ha="center", va="top", color=TEXT_FADED, fontsize=8.5,
                    fontweight="bold")

    # legend
    leg = ax.legend(loc="upper left", facecolor=BG_MID, edgecolor=GRID_COL,
                    labelcolor=TEXT_MAIN, fontsize=10)
    leg.get_frame().set_linewidth(0.6)

    # ── Metric strip ──
    h_total = float(hy[-1]) if hy else 0.0
    a_total = float(ay_s[-1]) if ay_s else 0.0
    h_goals = int(shots[(shots["team_id"] == home_id) &
                        (shots["is_goal"] == True)].shape[0])  # noqa: E712
    a_goals = int(shots[(shots["team_id"] == away_id) &
                        (shots["is_goal"] == True)].shape[0])  # noqa: E712

    metric_strip(
        fig, x=0.07, y=0.06, w=0.88, h=0.08,
        metrics=[
            (f"{home} xG", f"{h_total:.2f}", C_HOME),
            (f"{home} Goals", h_goals, C_HOME),
            ("xG diff",
             f"{(h_total - a_total):+.2f}",
             C_HOME if h_total > a_total else C_AWAY),
            (f"{away} Goals", a_goals, C_AWAY),
            (f"{away} xG", f"{a_total:.2f}", C_AWAY),
        ],
    )

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight",
                    facecolor=BG_DARK)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 3) PASS NETWORK — Re-designed
# ═════════════════════════════════════════════════════════════════════════════
def pass_network_v2(events: pd.DataFrame, info: dict, team_id: int,
                    team_color: str, team_name: str,
                    accent: str = C_GOLD,
                    min_passes: int = 3,
                    save_path: str | None = None):
    """
    Pass Network on a unified pitch:
      • Nodes: each player's mean position — size scales with pass volume
      • Edges: lines between player pairs, weight by pass volume
      • Side panel listing top connections
    """
    fig = make_themed_figure(15, 9.5)
    apply_unified_frame(
        fig,
        section="PASS NETWORK",
        title=f"{team_name} — Passing Network",
        subtitle=f"Average player positions  ·  edges show ≥{min_passes} passes between pairs",
        accent=accent,
        home_name=info.get("home_name"),
        away_name=info.get("away_name"),
        score=info.get("score"),
        footer_note="Node size scales with player's passes",
    )

    pitch_ax = fig.add_axes([0.04, 0.13, 0.66, 0.74])
    themed_pitch(pitch_ax)

    # ── aggregate successful passes between player pairs ──
    passes = events[(events["is_pass"] == True) &  # noqa: E712
                    (events["team_id"] == team_id) &
                    (events["outcome"] == "Successful")].copy()

    if passes.empty:
        pitch_ax.text(50, 50, "No passes recorded",
                      ha="center", va="center",
                      color=TEXT_DIM, fontsize=14, style="italic")
        if save_path:
            fig.savefig(save_path, dpi=160, bbox_inches="tight",
                        facecolor=BG_DARK)
        return fig

    # each player's mean position
    pos = (passes.groupby(["player_id", "player"])
           .agg(x=("x", "mean"), y=("y", "mean"),
                passes=("is_pass", "sum"))
           .reset_index())

    # Need the receiver — not directly exposed by WhoScored, so build pairs between
    # a pass and the very next event for the same team (common heuristic)
    passes_sorted = passes.sort_values(["minute", "second"]).reset_index(drop=True)
    pairs_df = passes_sorted.copy()
    pairs_df["receiver_id"] = passes_sorted["player_id"].shift(-1)
    pairs_df = pairs_df.dropna(subset=["player_id", "receiver_id"])
    pairs_df["receiver_id"] = pairs_df["receiver_id"].astype(int)
    pairs_df = pairs_df[pairs_df["player_id"] != pairs_df["receiver_id"]]

    # count passes per pair (undirected: a->b == b->a)
    pairs_df["pair_key"] = pairs_df.apply(
        lambda r: tuple(sorted([int(r["player_id"]), int(r["receiver_id"])])),
        axis=1,
    )
    pair_counts = pairs_df.groupby("pair_key").size().reset_index(name="n")
    pair_counts = pair_counts[pair_counts["n"] >= min_passes]

    # ── draw the edges ──
    pos_lookup = pos.set_index("player_id")
    max_n = max(int(pair_counts["n"].max()) if len(pair_counts) else 1, 1)
    top_pairs: list[tuple[str, int]] = []
    for _, r in pair_counts.sort_values("n").iterrows():
        a, b = r["pair_key"]
        if a not in pos_lookup.index or b not in pos_lookup.index:
            continue
        x1, y1 = pos_lookup.loc[a, ["x", "y"]]
        x2, y2 = pos_lookup.loc[b, ["x", "y"]]
        intensity = r["n"] / max_n
        lw = 0.8 + 4.5 * intensity
        alpha = 0.25 + 0.7 * intensity
        pitch_ax.plot([x1, x2], [y1, y2],
                      color=team_color, lw=lw, alpha=alpha, zorder=3,
                      solid_capstyle="round")

    # ── draw the nodes ──
    if not pos.empty:
        max_p = max(pos["passes"].max(), 1)
        for _, r in pos.iterrows():
            sz = 250 + 950 * (r["passes"] / max_p)
            pitch_ax.scatter([r["x"]], [r["y"]], s=sz,
                             color=team_color,
                             edgecolor="white", linewidths=2,
                             zorder=6, alpha=0.92)
            # player number/name
            pitch_ax.text(r["x"], r["y"], _short(str(r["player"]), 10),
                          ha="center", va="center",
                          color=TEXT_BRIGHT, fontsize=7.8, fontweight="bold",
                          path_effects=_shadow(2.2), zorder=7)

    # direction arrow
    pitch_ax.annotate("", xy=(95, 5), xytext=(60, 5),
                      arrowprops=dict(arrowstyle="->", color=team_color,
                                      lw=2, alpha=0.6))
    pitch_ax.text(77.5, 7.5, f"{team_name} attacking →",
                  color=team_color, fontsize=9, fontweight="bold",
                  alpha=0.7)

    # ── Metric strip ──
    total_passes = int(passes.shape[0])
    avg_passes = int(round(pos["passes"].mean())) if len(pos) else 0
    total_pairs = int(pair_counts.shape[0])

    metric_strip(
        fig, x=0.73, y=0.78, w=0.24, h=0.10,
        metrics=[
            ("Total Pass", total_passes, TEXT_BRIGHT),
            ("Avg/Player", avg_passes, accent),
            ("Pair Links", total_pairs, C_GREEN),
        ],
    )

    # ── Side panel: top connections ──
    rows: list[tuple[str, str, str]] = []
    pid_to_name = pos.set_index("player_id")["player"].to_dict()
    top = pair_counts.sort_values("n", ascending=False).head(7)
    for _, r in top.iterrows():
        a, b = r["pair_key"]
        na = _short(str(pid_to_name.get(a, "?")), 12)
        nb = _short(str(pid_to_name.get(b, "?")), 12)
        rows.append((f"{na} ⇄ {nb}", int(r["n"]), team_color))
    if not rows:
        rows = [("—", "no pairs", TEXT_FADED)]
    side_panel(fig, x=0.73, y=0.13, w=0.24, h=0.62,
               title="Top Pass Pairs", rows=rows, accent=accent)

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight",
                    facecolor=BG_DARK)
    return fig
