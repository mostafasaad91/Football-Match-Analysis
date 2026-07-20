from __future__ import annotations

import gc
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Rectangle, Wedge
import numpy as np
import pandas as pd
from PIL import Image

import visual_redesign_preview as base
from match_metrics import (
    advanced_metrics_frames,
    build_possessions,
    box_entry_mask,
    cross_mask,
    deep_completion_mask,
    final_third_entry_mask,
    high_regain_events,
    progressive_pass_mask,
    touch_mask,
)
from match_report import compute_ppda_both


ROOT = Path(__file__).resolve().parent
MATCH_KEY = "France_vs_England_4-6"
OUT = ROOT / "output" / MATCH_KEY
DATA = ROOT / "sample_data" / MATCH_KEY

BG = base.BG
PANEL = base.PANEL
PANEL_2 = base.PANEL_2
TEXT = base.TEXT
MUTED = base.MUTED
GRID = base.GRID
HOME = base.HOME
AWAY = base.AWAY
VALUE = base.VALUE
FOCUS = base.FOCUS
NEUTRAL = base.NEUTRAL

HOME_ID = base.HOME_ID
AWAY_ID = base.AWAY_ID
HOME_NAME = base.HOME_NAME
AWAY_NAME = base.AWAY_NAME
TEAM_COLOR = base.TEAM_COLOR
TEAM_NAME = base.TEAM_NAME

PITCH_LENGTH = 105.0
PITCH_WIDTH = 58.0


def as_bool(series: pd.Series) -> pd.Series:
    return base._bool(series)


def load_all():
    return base.load_data()


def attack_xy(x, y):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    display_x = (y_arr - 50.0) * (PITCH_WIDTH / 100.0)
    display_y = x_arr * (PITCH_LENGTH / 100.0)
    return display_x, display_y


def player_position_xy(x, y):
    """Display positional maps with the provider's lateral axis corrected."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    display_x = (50.0 - y_arr) * (PITCH_WIDTH / 100.0)
    display_y = x_arr * (PITCH_LENGTH / 100.0)
    return display_x, display_y


def pitch_axes(title: str, subtitle: str):
    fig = plt.figure(figsize=(12, 9), facecolor=BG)
    active_team = HOME_NAME if HOME_NAME.lower() in title.lower() else (AWAY_NAME if AWAY_NAME.lower() in title.lower() else None)
    base.amoled_header(fig, title, subtitle, active_team=active_team)
    pitch = fig.add_axes([0.075, 0.105, 0.48, 0.72])
    side = fig.add_axes([0.615, 0.145, 0.325, 0.62])
    side.set_facecolor(PANEL)
    for spine in side.spines.values():
        spine.set_color(GRID)
    side.set_xticks([])
    side.set_yticks([])
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    fig.text(0.945, 0.035, "FULL VISUAL REDESIGN · REAL MATCH DATA", ha="right", fontsize=8, color=NEUTRAL)
    return fig, pitch, side


def draw_long_pitch(ax, line_color="#5C6470", lw=1.15):
    half_w = PITCH_WIDTH / 2
    ax.set_xlim(-half_w - 5, half_w + 5)
    ax.set_ylim(-3, PITCH_LENGTH + 3)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((-half_w, 0), PITCH_WIDTH, PITCH_LENGTH, fill=False, ec=line_color, lw=lw))
    ax.plot([-half_w, half_w], [PITCH_LENGTH / 2, PITCH_LENGTH / 2], color=line_color, lw=lw)
    ax.add_patch(Circle((0, PITCH_LENGTH / 2), PITCH_LENGTH * 0.0915, fill=False, ec=line_color, lw=lw))
    penalty_w = PITCH_WIDTH * 0.595
    six_w = PITCH_WIDTH * 0.265
    box_l = PITCH_LENGTH * 0.157
    six_l = PITCH_LENGTH * 0.052
    for y0, direction in [(0, 1), (PITCH_LENGTH, -1)]:
        ax.add_patch(Rectangle((-penalty_w / 2, y0 if direction > 0 else y0 - box_l), penalty_w, box_l, fill=False, ec=line_color, lw=lw))
        ax.add_patch(Rectangle((-six_w / 2, y0 if direction > 0 else y0 - six_l), six_w, six_l, fill=False, ec=line_color, lw=lw))
        spot_y = y0 + direction * PITCH_LENGTH * 0.105
        ax.scatter([0], [spot_y], s=7, color=line_color)
        arc_center = y0 + direction * box_l
        if direction > 0:
            ax.add_patch(Arc((0, spot_y), 18.3, 18.3, theta1=37, theta2=143, ec=line_color, lw=lw))
        else:
            ax.add_patch(Arc((0, spot_y), 18.3, 18.3, theta1=217, theta2=323, ec=line_color, lw=lw))
    ax.annotate("ATTACK", xy=(0, PITCH_LENGTH + 1.5), ha="center", va="bottom", color=FOCUS, fontsize=8, fontweight="bold")
    ax.axis("off")


def side_title(ax, text: str):
    ax.text(0.07, 0.94, text.upper(), color=MUTED, fontsize=9, fontweight="bold", va="top")
    ax.plot([0.07, 0.93], [0.89, 0.89], color=GRID, lw=1)


def side_kpis(ax, items: list[tuple[str, str]], start=0.82, gap=0.14):
    for idx, (label, value) in enumerate(items):
        y = start - idx * gap
        if y < 0.06:
            break
        ax.text(0.08, y, label.upper(), color=MUTED, fontsize=7.5, fontweight="bold", va="top")
        ax.text(0.08, y - 0.055, str(value), color=TEXT, fontsize=16, fontweight="bold", va="top")


def side_rows(ax, rows: list[tuple[str, str]], start=0.82, gap=0.085, value_color=FOCUS):
    for idx, (label, value) in enumerate(rows):
        y = start - idx * gap
        if y < 0.04:
            break
        ax.text(0.08, y, str(label), color=TEXT, fontsize=8.5, va="center")
        ax.text(0.92, y, str(value), color=value_color, fontsize=8.5, fontweight="bold", ha="right", va="center")
        ax.plot([0.08, 0.92], [y - gap * 0.45, y - gap * 0.45], color=GRID, lw=0.55, alpha=0.7)


def save(fig, filename: str) -> Path:
    if not getattr(fig, "_amoled_header_applied", False):
        candidates = []
        for item in fig.texts:
            try:
                if item.get_visible() and item.get_position()[1] >= 0.84:
                    candidates.append(item)
            except Exception:
                continue
        title_item = max(candidates, key=lambda item: float(item.get_fontsize()), default=None)
        title = title_item.get_text() if title_item is not None else filename.rsplit(".", 1)[0].replace("_", " ").title()
        subtitle_items = [item for item in candidates if item is not title_item and item.get_text().strip()]
        subtitle = subtitle_items[0].get_text() if subtitle_items else "France vs England · real match data"
        for item in candidates:
            item.set_visible(False)
        active_team = HOME_NAME if HOME_NAME.lower() in title.lower() else (AWAY_NAME if AWAY_NAME.lower() in title.lower() else None)
        base.amoled_header(fig, title, subtitle, active_team=active_team)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.16, facecolor=BG)
    plt.close(fig)
    return path


def label_layout(points: list[tuple[float, float, str]], min_gap=5.4):
    groups = {"left": [], "right": []}
    for x, y, name in points:
        groups["left" if x <= 0 else "right"].append([x, y, name, y])
    placed = []
    for side, items in groups.items():
        items.sort(key=lambda item: item[1])
        for idx in range(1, len(items)):
            if items[idx][3] - items[idx - 1][3] < min_gap:
                items[idx][3] = items[idx - 1][3] + min_gap
        overflow = items[-1][3] - (PITCH_LENGTH - 3) if items else 0
        if overflow > 0:
            for item in items:
                item[3] -= overflow
        for idx in range(len(items) - 2, -1, -1):
            if items[idx + 1][3] - items[idx][3] < min_gap:
                items[idx][3] = items[idx + 1][3] - min_gap
        under = 3 - items[0][3] if items else 0
        if under > 0:
            for item in items:
                item[3] += under
        for x, y, name, label_y in items:
            label_x = -PITCH_WIDTH / 2 - 2.5 if side == "left" else PITCH_WIDTH / 2 + 2.5
            placed.append((x, y, name, label_x, label_y, "right" if side == "left" else "left"))
    return placed


def place_player_labels(ax, points):
    for x, y, name, lx, ly, ha in label_layout(points):
        ax.plot([x, lx * 0.94], [y, ly], color=GRID, lw=0.7, zorder=4)
        ax.text(lx, ly, name, color=TEXT, fontsize=7.5, ha=ha, va="center", zorder=5)


def compact_player_label(name: str) -> str:
    """Return a readable surname that still fits inside a network node."""
    surname = str(name).strip().split()[-1] if str(name).strip() else "?"
    return surname if len(surname) <= 7 else f"{surname[:6]}…"


def draw_node_label(ax, x: float, y: float, name: str, touches: float, max_touch: float):
    label = compact_player_label(name)
    ratio = float(touches) / max(float(max_touch), 1.0)
    size = (5.2 if len(label) <= 7 else 4.2) if ratio >= 0.25 else (4.4 if len(label) <= 7 else 3.6)
    text = ax.text(x, y, label, color=TEXT, fontsize=size, fontweight="bold",
                   ha="center", va="center", zorder=6, clip_on=True)
    text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground=BG, alpha=0.9)])


def _role_fallback_position(position: str) -> tuple[float, float]:
    role = str(position or "").upper()
    x = 50.0
    if "GK" in role:
        x = 8.0
    elif role in {"DC", "DL", "DR", "DLC", "DRC"} or role.startswith("D"):
        x = 30.0
    elif "DMC" in role:
        x = 43.0
    elif role in {"MC", "ML", "MR"} or role.startswith("M"):
        x = 55.0
    elif "AM" in role:
        x = 68.0
    elif role in {"FW", "ST", "CF"} or "FW" in role:
        x = 80.0
    y = 50.0
    if "L" in role and "LC" not in role:
        y = 22.0
    elif "R" in role and "RC" not in role:
        y = 78.0
    return x, y


def _separate_network_positions(display: dict[str, tuple[float, float, float]], min_gap=5.3):
    """Apply small collision-only nudges while preserving each player's anchor."""
    names = list(display)
    if len(names) < 2:
        return display
    anchors = np.array([[display[name][0], display[name][1]] for name in names], dtype=float)
    coords = anchors.copy()
    for _ in range(90):
        shift = np.zeros_like(coords)
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                vector = coords[i] - coords[j]
                distance = float(np.hypot(vector[0], vector[1]))
                if distance >= min_gap:
                    continue
                if distance < 1e-6:
                    angle = (i * 37 + j * 19) % 360
                    vector = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                    distance = 1.0
                push = (min_gap - distance) * 0.22 * vector / distance
                shift[i] += push
                shift[j] -= push
        coords += shift
        coords += (anchors - coords) * 0.025
        coords[:, 0] = np.clip(coords[:, 0], -PITCH_WIDTH / 2 + 2.4, PITCH_WIDTH / 2 - 2.4)
        coords[:, 1] = np.clip(coords[:, 1], 2.5, PITCH_LENGTH - 2.5)
    return {
        name: (float(coords[idx, 0]), float(coords[idx, 1]), display[name][2])
        for idx, name in enumerate(names)
    }


def team_event_counts(events, team_id):
    team = events[events["team_id"].eq(team_id)]
    types = team["type"].astype(str)
    return {
        "Tackles": int(types.eq("Tackle").sum()),
        "Interceptions": int(types.eq("Interception").sum()),
        "Recoveries": int(types.eq("BallRecovery").sum()),
        "Clearances": int(types.eq("Clearance").sum()),
        "Blocks": int(types.eq("BlockedShot").sum()),
        "Fouls": int(types.eq("Foul").sum()),
    }


def xg_row(xg, team_name):
    row = xg[xg["team"].astype(str).str.lower().eq(team_name.lower())]
    return row.iloc[0] if not row.empty else pd.Series(dtype=float)


def shot_map(events, xg, team_id, number):
    pso = as_bool(events.get("is_penalty_shootout", pd.Series(False, index=events.index)))
    shots = events[events["team_id"].eq(team_id) & as_bool(events["is_shot"]) & ~pso].copy().dropna(subset=["x", "y"])
    shots["xG"] = pd.to_numeric(shots["xG"], errors="coerce").fillna(0).clip(lower=0)
    fig, pitch, side = pitch_axes(f"Shot Map · {TEAM_NAME[team_id]}", "Shot location, outcome and chance quality · marker size = xG")
    draw_long_pitch(pitch)
    markers = {"Goal": ("*", FOCUS, True), "SavedShot": ("o", VALUE, True), "BlockedShot": ("s", NEUTRAL, True), "MissedShots": ("o", MUTED, False), "ShotOnPost": ("D", FOCUS, False)}
    for event_type, (marker, color, filled) in markers.items():
        subset = shots[shots["type"].astype(str).eq(event_type)]
        if subset.empty:
            continue
        px, py = attack_xy(subset["x"], subset["y"])
        sizes = 45 + subset["xG"].to_numpy() * 520
        pitch.scatter(px, py, s=sizes, marker=marker, facecolors=color if filled else "none", edgecolors=TEXT if filled else color, linewidths=1.15, alpha=0.93, label=f"{event_type} ({len(subset)})", zorder=4)
    pitch.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, labelcolor=TEXT, fontsize=7.5)
    xr = xg_row(xg, TEAM_NAME[team_id])
    side_title(side, "SHOT OUTPUT")
    side_kpis(side, [("Shots", f"{len(shots)}"), ("xG", f"{float(xr.get('xG', 0)):.2f}"), ("xG / shot", f"{float(xr.get('xG_per_shot', 0)):.3f}"), ("On target", f"{int(float(xr.get('on_target', 0)))}")])
    return save(fig, f"{number:02d}_shot_map_{TEAM_NAME[team_id].lower()}.png")


def goals_breakdown(events):
    pso = as_bool(events.get("is_penalty_shootout", pd.Series(False, index=events.index)))
    goals = events[as_bool(events["is_goal"]) & ~pso].copy()
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.text(0.055, 0.94, "Goal Breakdown", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.898, "Scoring timeline and goal profile · penalty shootout excluded", fontsize=11, color=MUTED)
    fig.add_artist(Line2D([0.055, 0.945], [0.86, 0.86], transform=fig.transFigure, color=GRID, lw=1))
    ax = fig.add_axes([0.08, 0.18, 0.84, 0.56])
    base.clean_ax(ax)
    max_min = max(90, int(pd.to_numeric(events["minute"], errors="coerce").max()) + 2)
    ax.set_xlim(0, max_min)
    ax.set_ylim(-1.4, 1.4)
    ax.axhline(0, color=GRID, lw=1.5)
    ax.set_yticks([0.75, -0.75]); ax.set_yticklabels([HOME_NAME, AWAY_NAME], fontsize=11)
    ax.set_xlabel("Match minute")
    ordered = events.sort_values(["minute", "second", "event_id"], kind="stable").copy()
    ordered["_clock"] = (
        pd.to_numeric(ordered["minute"], errors="coerce").fillna(0) * 60
        + pd.to_numeric(ordered["second"], errors="coerce").fillna(0)
    )

    def assist_for(goal):
        explicit = goal.get("assist_player", np.nan)
        if pd.notna(explicit) and str(explicit).strip():
            return str(explicit)
        goal_clock = float(goal["_clock"])
        key_pass = as_bool(ordered.get("is_key_pass", pd.Series(False, index=ordered.index)))
        candidates = ordered[
            ordered["team_id"].eq(goal["team_id"])
            & ordered["type"].astype(str).eq("Pass")
            & ordered["outcome"].astype(str).str.lower().eq("successful")
            & key_pass
            & ordered["_clock"].between(goal_clock - 15, goal_clock, inclusive="left")
            & ordered["player"].notna()
        ]
        return str(candidates.iloc[-1]["player"]) if not candidates.empty else "UNASSISTED"

    goals = ordered.loc[goals.index].sort_values(["minute", "second"], kind="stable")
    team_goal_count = {HOME_ID: 0, AWAY_ID: 0}
    for _, goal in goals.iterrows():
        tid = int(goal["team_id"])
        y = 0.75 if tid == HOME_ID else -0.75
        minute = float(goal["minute"])
        team_goal_count[tid] += 1
        ax.vlines(minute, 0, y, color=TEAM_COLOR[tid], lw=2)
        ax.scatter(minute, y, s=150, marker="*", color=FOCUS, edgecolor=BG, linewidth=1.2, zorder=4)
        player = str(goal.get("player", "Goal")).split()[-1]
        assist = assist_for(goal)
        assist_label = "UNASSISTED" if assist == "UNASSISTED" else f"ASSIST · {assist.split()[-1]}"
        label = f"{int(minute)}′  {player}\n{assist_label}"
        horizontal_nudge = -10 if team_goal_count[tid] % 2 else 10
        ax.annotate(
            label, (minute, y),
            xytext=(horizontal_nudge, 18 if y > 0 else -18), textcoords="offset points",
            ha="center", va="bottom" if y > 0 else "top", color=TEXT, fontsize=7.2,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=PANEL, edgecolor=GRID, linewidth=0.65),
        )
    ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.65)
    return save(fig, "04_goals_breakdown.png")


def _half_network_data(events, players, team_id, half):
    period_code = "1h" if half == 1 else "2h"
    frame = events[events["period_code"].astype(str).str.lower().eq(period_code)].copy()
    frame = frame.sort_values(["minute", "second", "event_id"], kind="stable")
    team_frame = frame[frame["team_id"].eq(team_id)].copy()

    work = frame.copy()
    work["clock"] = pd.to_numeric(work["minute"], errors="coerce").fillna(0) * 60 + pd.to_numeric(work["second"], errors="coerce").fillna(0)
    work["next_team"] = work["team_id"].shift(-1)
    work["next_player"] = work["player"].shift(-1)
    work["next_clock"] = work["clock"].shift(-1)
    passes = work[
        work["team_id"].eq(team_id)
        & work["type"].astype(str).eq("Pass")
        & work["outcome"].astype(str).str.lower().eq("successful")
        & work["next_team"].eq(team_id)
        & work["player"].notna()
        & work["next_player"].notna()
        & work["next_clock"].sub(work["clock"]).between(0, 20)
    ].copy()
    passes = passes[passes["player"].astype(str).ne(passes["next_player"].astype(str))]

    touches = team_frame[touch_mask(team_frame)].dropna(subset=["player", "x", "y"]).copy()
    sub_events = team_frame[team_frame["type"].astype(str).isin(["SubstitutionOn", "SubstitutionOff"])].copy()
    log_sub_events = sub_events.copy()
    sub_on = set(sub_events[sub_events["type"].astype(str).eq("SubstitutionOn")]["player"].dropna().astype(str))
    sub_off = set(sub_events[sub_events["type"].astype(str).eq("SubstitutionOff")]["player"].dropna().astype(str))
    if half == 1:
        interval_events = events[
            events["team_id"].eq(team_id)
            & events["period_code"].astype(str).str.lower().eq("2h")
            & events["type"].astype(str).isin(["SubstitutionOn", "SubstitutionOff"])
            & (pd.to_numeric(events["minute"], errors="coerce").fillna(999) <= 45)
        ].copy()
        interval_off = interval_events[interval_events["type"].astype(str).eq("SubstitutionOff")]["player"]
        sub_off |= set(interval_off.dropna().astype(str))
        # The player introduced at the interval belongs to the second-half
        # visual only. Keep the outgoing starter marked, but do not name the
        # incoming player anywhere on the first-half card.
        log_sub_events = pd.concat(
            [sub_events, interval_events[interval_events["type"].astype(str).eq("SubstitutionOff")]],
            ignore_index=False,
        )
    participants = set(touches["player"].dropna().astype(str)) | sub_on
    if half == 1:
        starters = players[players["team_id"].eq(team_id) & players["is_first_xi"].astype(str).str.lower().isin(["true", "1", "yes"])]["name"]
        participants |= set(starters.dropna().astype(str))

    player_info = players[players["team_id"].eq(team_id)].set_index("name")
    all_touches = events[touch_mask(events) & events["team_id"].eq(team_id)].dropna(subset=["player", "x", "y"])
    position_rows = []
    for name in sorted(participants):
        player_touches = touches[touches["player"].astype(str).eq(name)]
        coords = player_touches[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna()
        if coords.empty:
            event_coords = team_frame[team_frame["player"].astype(str).eq(name)][["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna()
            coords = event_coords
        if coords.empty:
            whole_coords = all_touches[all_touches["player"].astype(str).eq(name)][["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna()
            coords = whole_coords
        if coords.empty:
            role = player_info.at[name, "position"] if name in player_info.index else ""
            x, y = _role_fallback_position(role)
        else:
            x, y = float(coords["x"].mean()), float(coords["y"].mean())
        position_rows.append({"player": name, "x": x, "y": y, "touches": int(len(player_touches))})
    positions = pd.DataFrame(position_rows).set_index("player") if position_rows else pd.DataFrame(columns=["x", "y", "touches"])
    names = set(positions.index.astype(str))
    passes = passes[passes["player"].astype(str).isin(names) & passes["next_player"].astype(str).isin(names)]
    edges = passes.groupby(["player", "next_player"]).size().reset_index(name="passes").sort_values("passes", ascending=False).head(22)

    substitutions = []
    events_sorted = log_sub_events.sort_values(["minute", "second", "event_id"], kind="stable")
    pending_off = []
    for _, row in events_sorted.iterrows():
        minute = int(float(row.get("minute", 0) or 0))
        name = compact_player_label(str(row.get("player", "")))
        if str(row.get("type")) == "SubstitutionOff":
            pending_off.append((minute, name))
        else:
            match_idx = next((idx for idx, item in enumerate(pending_off) if item[0] == minute), None)
            off_name = pending_off.pop(match_idx)[1] if match_idx is not None else "—"
            substitutions.append((minute, name, off_name))
    for minute, off_name in pending_off:
        substitutions.append((minute, "—", off_name))
    return positions, edges, sub_on, sub_off, substitutions, int(len(passes))


def pass_network(events, players, team_id, number, half):
    positions, edges, sub_on, sub_off, substitutions, completed_links = _half_network_data(events, players, team_id, half)
    half_label = "First Half" if half == 1 else "Second Half"
    fig, pitch, side = pitch_axes(
        f"Pass Network · {TEAM_NAME[team_id]} · {half_label}",
        f"All {len(positions)} participants shown · node size = touches · square = came on · gold outline = went off",
    )
    draw_long_pitch(pitch)
    display = {}
    for name, row in positions.iterrows():
        px, py = player_position_xy([row["x"]], [row["y"]])
        display[str(name)] = (float(px[0]), float(py[0]), float(row["touches"]))
    display = _separate_network_positions(display, min_gap=6.3)
    max_edge = max(float(edges["passes"].max()) if not edges.empty else 1, 1)
    for _, edge in edges.iterrows():
        a, b = str(edge["player"]), str(edge["next_player"])
        if a not in display or b not in display:
            continue
        ax, ay, _ = display[a]; bx, by, _ = display[b]
        pitch.plot([ax, bx], [ay, by], color=TEAM_COLOR[team_id], alpha=0.34,
                   lw=0.75 + 4.4 * float(edge["passes"]) / max_edge, zorder=2)
    max_touch = max([value[2] for value in display.values()] or [1])
    for name, (px, py, touches) in display.items():
        entered = name in sub_on
        left = name in sub_off
        pitch.scatter(px, py, s=260 + 640 * touches / max_touch, marker="s" if entered else "o",
                      color=TEAM_COLOR[team_id], edgecolor=FOCUS if left else TEXT,
                      linewidth=2.3 if left else 1.15, zorder=4)
        draw_node_label(pitch, px, py, name, touches, max_touch)

    side_title(side, "TOP HALF CONNECTIONS")
    side.text(0.92, 0.94, f"{len(positions)} players", color=FOCUS, fontsize=8,
              fontweight="bold", ha="right", va="top")
    side_rows(side, [(f"{compact_player_label(r.player)} → {compact_player_label(r.next_player)}", str(int(r.passes))) for r in edges.head(5).itertuples()], start=0.81, gap=0.075)
    side.text(0.08, 0.40, "SUBSTITUTIONS", color=MUTED, fontsize=7.5, fontweight="bold")
    if substitutions:
        for idx, (minute, on_name, off_name) in enumerate(substitutions[:5]):
            y = 0.35 - idx * 0.052
            side.text(0.08, y, f"{minute}′", color=FOCUS, fontsize=7.5, fontweight="bold", va="center")
            change = f"{off_name} OFF AT INTERVAL" if on_name == "—" else f"{on_name} IN  ·  {off_name} OFF"
            side.text(0.19, y, change, color=TEXT, fontsize=7.2, va="center")
    else:
        side.text(0.08, 0.34, "No in-half changes", color=MUTED, fontsize=8)
    side.text(0.08, 0.105, f"Completed pass links: {completed_links}", color=VALUE, fontsize=8, fontweight="bold")
    side.scatter([0.12, 0.31], [0.055, 0.055], s=[65, 65], marker="o", color=TEAM_COLOR[team_id], edgecolor=[TEXT, FOCUS], linewidth=[1.0, 2.1])
    side.scatter([0.50], [0.055], s=65, marker="s", color=TEAM_COLOR[team_id], edgecolor=TEXT, linewidth=1.0)
    side.text(0.16, 0.055, "Began half", color=TEXT, fontsize=6.8, va="center")
    side.text(0.35, 0.055, "Went off", color=TEXT, fontsize=6.8, va="center")
    side.text(0.54, 0.055, "Came on", color=TEXT, fontsize=6.8, va="center")
    suffix = "1h" if half == 1 else "2h"
    return save(fig, f"{number:02d}{'a' if half == 1 else 'b'}_pass_network_{TEAM_NAME[team_id].lower()}_{suffix}.png")


def xt_map(events, team_id, number):
    team = events[events["team_id"].eq(team_id) & events["type"].astype(str).eq("Pass")].copy().dropna(subset=["x", "y", "end_x", "end_y"])
    team["xT"] = pd.to_numeric(team["xT"], errors="coerce").fillna(0).clip(lower=0)
    heat, _, _ = np.histogram2d(
        team["y"], team["x"], bins=[7, 12], range=[[0, 100], [0, 100]], weights=team["xT"]
    )
    fig, pitch, side = pitch_axes(
        f"xT Heatmap · {TEAM_NAME[team_id]}",
        "Full-pitch 7 × 12 square grid · every cell sums threat added from pass origins",
    )
    cmap = LinearSegmentedColormap.from_list("xt_full_grid", [BG, "#12101F", "#2D2359", VALUE, FOCUS])
    nonzero = heat[heat > 0]
    vmax = max(float(np.percentile(nonzero, 92)) if nonzero.size else 0.0, 0.001)
    x_grid = np.linspace(-PITCH_WIDTH / 2, PITCH_WIDTH / 2, 8)
    y_grid = np.linspace(0, PITCH_LENGTH, 13)
    image = pitch.pcolormesh(
        x_grid, y_grid, heat.T, cmap=cmap, vmin=0, vmax=vmax, shading="flat",
        edgecolors=GRID, linewidth=0.58, alpha=0.98, zorder=1,
    )
    draw_long_pitch(pitch)
    for ix in range(7):
        for iy in range(12):
            value = float(heat[ix, iy])
            if value <= 0:
                continue
            intensity = min(value / vmax, 1.0)
            if intensity >= 0.68:
                number_color = BG
            elif intensity >= 0.18:
                number_color = TEXT
            else:
                number_color = MUTED
            pitch.text(
                (x_grid[ix] + x_grid[ix + 1]) / 2,
                (y_grid[iy] + y_grid[iy + 1]) / 2,
                f"{value:.2f}", color=number_color,
                fontsize=5.0, fontweight="bold", ha="center", va="center", zorder=3,
            )
    top = team.nlargest(10, "xT")
    arrow_color = "#4C6FFF"
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        sx, sy = attack_xy([row["x"]], [row["y"]]); ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        arrow = pitch.annotate(
            "", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]),
            arrowprops=dict(
                arrowstyle="-|>", color=arrow_color,
                lw=1.9 if rank <= 3 else 1.05,
                alpha=0.96 if rank <= 3 else 0.58,
                mutation_scale=11 if rank <= 3 else 8,
            ),
        )
        if arrow.arrow_patch is not None:
            arrow.arrow_patch.set_path_effects([
                path_effects.Stroke(linewidth=3.2 if rank <= 3 else 2.0, foreground=BG),
                path_effects.Normal(),
            ])
    cbar = fig.colorbar(image, ax=pitch, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.outline.set_edgecolor(GRID)
    cbar.set_label("xT added per grid square", color=MUTED, fontsize=8)
    side_title(side, "TOP 10 xT PASSES")
    side_rows(
        side,
        [(f"{rank}. {str(row['player']).split()[-1]}", f"{float(row['xT']):.3f}") for rank, (_, row) in enumerate(top.iterrows(), start=1)],
        start=0.835,
        gap=0.063,
        value_color=arrow_color,
    )
    side.text(
        0.08, 0.075,
        "Indigo arrows show the top 10 threat-adding passes; the top three use stronger strokes.",
        color=MUTED, fontsize=7.2, wrap=True,
    )
    return save(fig, f"{number:02d}_xt_map_{TEAM_NAME[team_id].lower()}.png")


def pass_map(events, team_id, number):
    frame = events[events["team_id"].eq(team_id) & events["type"].astype(str).eq("Pass")].copy()
    frame = frame.dropna(subset=["x", "y", "end_x", "end_y"])
    completed = frame["outcome"].astype(str).str.lower().eq("successful")
    key_pass = as_bool(frame.get("is_key_pass", pd.Series(False, index=frame.index)))
    fig, pitch, side = pitch_axes(
        f"Pass Map · {TEAM_NAME[team_id]}",
        "Every pass in the match · completed, incomplete and key passes are explicitly distinguished",
    )
    draw_long_pitch(pitch)
    for idx, row in frame.iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]])
        ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        if bool(key_pass.loc[idx]):
            color, alpha, width, style = FOCUS, 0.95, 1.8, "-"
        elif bool(completed.loc[idx]):
            color, alpha, width, style = VALUE, 0.22, 0.65, "-"
        else:
            color, alpha, width, style = MUTED, 0.32, 0.65, (0, (3, 3))
        pitch.plot([sx[0], ex[0]], [sy[0], ey[0]], color=color, alpha=alpha,
                   lw=width, ls=style, zorder=2)
        if bool(key_pass.loc[idx]):
            pitch.scatter(ex[0], ey[0], s=22, marker="*", color=FOCUS,
                          edgecolor=TEXT, linewidth=0.45, zorder=4)
    attempts = len(frame)
    complete_count = int(completed.sum())
    forward = int((pd.to_numeric(frame["end_x"], errors="coerce") > pd.to_numeric(frame["x"], errors="coerce")).sum())
    side_title(side, "PASSING OUTPUT")
    side_kpis(side, [
        ("Attempts", attempts),
        ("Completed", complete_count),
        ("Completion", f"{100 * complete_count / max(attempts, 1):.1f}%"),
        ("Forward passes", forward),
    ], start=0.82, gap=0.13)
    legend_y = [0.235, 0.165, 0.095]
    legend_items = [
        ("Completed pass", VALUE, "-", "o"),
        ("Incomplete pass", MUTED, (0, (3, 3)), "o"),
        (f"Key pass ({int(key_pass.sum())})", FOCUS, "-", "*"),
    ]
    for y, (label, color, style, marker) in zip(legend_y, legend_items):
        side.plot([0.09, 0.25], [y, y], color=color, lw=2.0, ls=style)
        side.scatter([0.25], [y], s=38 if marker == "*" else 20, marker=marker,
                     color=color, edgecolor=TEXT, linewidth=0.45, zorder=4)
        side.text(0.31, y, label, color=TEXT, fontsize=8, va="center")
    return save(fig, f"{number:02d}_pass_map_{TEAM_NAME[team_id].lower()}.png")


def danger_creation(events, team_metrics, team_id, number):
    team = events[events["team_id"].eq(team_id)].copy()
    entries = events[box_entry_mask(events) & events["team_id"].eq(team_id)].dropna(subset=["end_x", "end_y"])
    zone14 = team[pd.to_numeric(team["end_x"], errors="coerce").between(70, 83) & pd.to_numeric(team["end_y"], errors="coerce").between(35, 65)].dropna(subset=["end_x", "end_y"])
    key = team[as_bool(team.get("is_key_pass", pd.Series(False, index=team.index)))].dropna(subset=["x", "y"])
    fig, pitch, side = pitch_axes(f"Danger Creation · {TEAM_NAME[team_id]}", "Box entries, Zone 14 access and key passes · shape-coded on one long pitch")
    draw_long_pitch(pitch)
    for frame, coord_cols, marker, label, size in [(entries, ("end_x", "end_y"), "o", "Box entry", 46), (zone14, ("end_x", "end_y"), "D", "Zone 14 action", 42), (key, ("x", "y"), "*", "Key pass", 85)]:
        if frame.empty: continue
        px, py = attack_xy(frame[coord_cols[0]], frame[coord_cols[1]])
        pitch.scatter(px, py, s=size, marker=marker, color=VALUE if marker != "*" else FOCUS, edgecolor=TEXT, linewidth=0.75, alpha=0.85, label=f"{label} ({len(frame)})")
    pitch.legend(loc="lower center", bbox_to_anchor=(0.5, -0.085), ncol=2, frameon=False, labelcolor=TEXT, fontsize=7.5)
    side_title(side, "CREATION OUTPUT")
    side_kpis(side, [("Box entries", len(entries)), ("Zone 14 actions", len(zone14)), ("Key passes", len(key)), ("Deep completions", int(base.metric_lookup(team_metrics, "home" if team_id == HOME_ID else "away", "deep_completions")))])
    return save(fig, f"{number:02d}_danger_creation_{TEAM_NAME[team_id].lower()}.png")


def gk_saves(events, xg):
    home_xg, away_xg = xg_row(xg, HOME_NAME), xg_row(xg, AWAY_NAME)
    home_saves = float(away_xg.get("saved", 0)); away_saves = float(home_xg.get("saved", 0))
    home_faced = float(away_xg.get("xGoT", 0)); away_faced = float(home_xg.get("xGoT", 0))
    home_on_target = float(away_xg.get("on_target", 0)); away_on_target = float(home_xg.get("on_target", 0))
    home_save_rate = 100 * home_saves / max(home_on_target, 1); away_save_rate = 100 * away_saves / max(away_on_target, 1)
    rows = [("Saves", home_saves, away_saves, "{:.0f}"), ("Shots on target faced", home_on_target, away_on_target, "{:.0f}"), ("xGoT faced", home_faced, away_faced, "{:.2f}"), ("Save rate", home_save_rate, away_save_rate, "{:.1f}%")]
    fig, ax = base.page("Goalkeeper Saves", "Goalkeeper comparison from opponent shots on target · exact bilateral values")
    base.row_dot_plot(ax, rows)
    return save(fig, "14_goalkeeper_saves.png")


def xg_summary(xg):
    h, a = xg_row(xg, HOME_NAME), xg_row(xg, AWAY_NAME)
    rows = [("xG", float(h.get("xG", 0)), float(a.get("xG", 0)), "{:.2f}"), ("xG on target", float(h.get("xGoT", 0)), float(a.get("xGoT", 0)), "{:.2f}"), ("xG per shot", float(h.get("xG_per_shot", 0)), float(a.get("xG_per_shot", 0)), "{:.3f}"), ("Goals", float(h.get("goals", 0)), float(a.get("goals", 0)), "{:.0f}")]
    fig, ax = base.page("Expected Goals Summary", "Chance quality, post-shot placement and finishing output")
    base.row_dot_plot(ax, rows)
    return save(fig, "15_xg_summary.png")


def zone14(events, team_id, number):
    team = events[events["team_id"].eq(team_id)].copy()
    successful = team["outcome"].astype(str).str.lower().eq("successful")
    zone = successful & pd.to_numeric(team["end_x"], errors="coerce").between(70, 83) & pd.to_numeric(team["end_y"], errors="coerce").between(35, 65)
    actions = team[zone].dropna(subset=["x", "y", "end_x", "end_y"])
    final_third = team[successful & (pd.to_numeric(team["end_x"], errors="coerce") >= 66.7)].copy()
    final_third["end_y_num"] = pd.to_numeric(final_third["end_y"], errors="coerce")
    final_third = final_third.dropna(subset=["end_y_num"])
    lane_defs = [
        ("Left wing", 0, 20),
        ("Left half-space", 20, 40),
        ("Central lane", 40, 60),
        ("Right half-space", 60, 80),
        ("Right wing", 80, 100.0001),
    ]
    lane_colors = [HOME, VALUE, "#8E7CE8", "#FF9A8A", AWAY]
    lane_counts = [int(final_third["end_y_num"].between(lo, hi, inclusive="left").sum()) for _, lo, hi in lane_defs]
    fig, pitch, side = pitch_axes(
        f"Zone 14 & Five Lanes · {TEAM_NAME[team_id]}",
        "Lane numbers = completed actions ending in the final third · arrows = completed Zone 14 access",
    )
    draw_long_pitch(pitch)
    third_y = 66.7 * PITCH_LENGTH / 100
    for (label, lo, hi), color in zip(lane_defs, lane_colors):
        x1, _ = attack_xy([66.7], [lo])
        x2, _ = attack_xy([66.7], [min(hi, 100)])
        pitch.add_patch(Rectangle(
            (min(x1[0], x2[0]), third_y), abs(x2[0] - x1[0]), PITCH_LENGTH - third_y,
            facecolor=color, edgecolor=color, lw=0.9, alpha=0.16, zorder=0,
        ))
    for idx, ((_, lo, hi), count, color) in enumerate(zip(lane_defs, lane_counts, lane_colors)):
        center_y = (lo + min(hi, 100)) / 2
        cx, cy = attack_xy([68.8], [center_y])
        pitch.text(cx[0], cy[0], str(count), ha="center", va="center", color=color,
                   fontsize=8, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.28", fc=PANEL, ec=color, lw=1.15, alpha=0.97), zorder=6)
    zx1, zy1 = attack_xy([70], [35]); zx2, zy2 = attack_xy([83], [65])
    pitch.add_patch(Rectangle(
        (min(zx1[0], zx2[0]), min(zy1[0], zy2[0])),
        abs(zx2[0] - zx1[0]), abs(zy2[0] - zy1[0]),
        facecolor=FOCUS, alpha=0.13, edgecolor=FOCUS, lw=1.8, hatch="//", zorder=1,
    ))
    pitch.text(0, min(zy1[0], zy2[0]) + 0.9, "ZONE 14", color=FOCUS, fontsize=6.5,
               fontweight="bold", ha="center", va="bottom", zorder=5)
    for _, row in actions.iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]]); ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        pitch.annotate("", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]), arrowprops=dict(arrowstyle="-|>", color=VALUE, alpha=0.45, lw=1.0, mutation_scale=8))
    top = actions.groupby("player").size().sort_values(ascending=False).head(3)
    side_title(side, "FIVE ATTACKING LANES")
    for idx, ((label, _, _), value, color) in enumerate(zip(lane_defs, lane_counts, lane_colors)):
        y = 0.81 - idx * 0.083
        side.add_patch(Rectangle((0.08, y - 0.016), 0.035, 0.032, facecolor=color,
                                 edgecolor=TEXT, lw=0.4, alpha=0.9))
        side.text(0.15, y, label, color=TEXT, fontsize=8.5, va="center")
        side.text(0.92, y, str(value), color=color, fontsize=8.8, fontweight="bold",
                  ha="right", va="center")
        side.plot([0.08, 0.92], [y - 0.037, y - 0.037], color=GRID, lw=0.55, alpha=0.7)
    side.text(0.08, 0.34, "ZONE 14 CONTRIBUTORS", color=MUTED, fontsize=7.5, fontweight="bold")
    for idx, (name, value) in enumerate(top.items()):
        y = 0.285 - idx * 0.06
        side.text(0.08, y, str(name).split()[-1], color=TEXT, fontsize=8, va="center")
        side.text(0.92, y, str(int(value)), color=FOCUS, fontsize=8, fontweight="bold", ha="right", va="center")
    side.text(0.08, 0.075, f"Zone 14 entries: {len(actions)}", color=FOCUS, fontsize=9.5, fontweight="bold")
    return save(fig, f"{number:02d}_zone14_{TEAM_NAME[team_id].lower()}.png")


def match_stats(events, xg, team_metrics):
    h, a = xg_row(xg, HOME_NAME), xg_row(xg, AWAY_NAME)
    big = as_bool(events.get("big_chance", pd.Series(False, index=events.index))) & as_bool(events["is_shot"])
    rows_left = [("Shots", float(h.get("shots", 0)), float(a.get("shots", 0)), "{:.0f}"), ("On target", float(h.get("on_target", 0)), float(a.get("on_target", 0)), "{:.0f}"), ("Big chances", float((big & events["team_id"].eq(HOME_ID)).sum()), float((big & events["team_id"].eq(AWAY_ID)).sum()), "{:.0f}"), ("xG", float(h.get("xG", 0)), float(a.get("xG", 0)), "{:.2f}")]
    rows_right = [("Final-third entries", base.metric_lookup(team_metrics, "home", "final_third_entries"), base.metric_lookup(team_metrics, "away", "final_third_entries"), "{:.0f}"), ("Progressive passes", base.metric_lookup(team_metrics, "home", "progressive_passes"), base.metric_lookup(team_metrics, "away", "progressive_passes"), "{:.0f}"), ("Possession regains", base.metric_lookup(team_metrics, "home", "possession_regains"), base.metric_lookup(team_metrics, "away", "possession_regains"), "{:.0f}"), ("High regains", base.metric_lookup(team_metrics, "home", "high_regains"), base.metric_lookup(team_metrics, "away", "high_regains"), "{:.0f}")]
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.text(0.055, 0.94, "Match Statistics", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.898, "Attacking output and territorial process · values left, metric centre, values right", fontsize=11, color=MUTED)
    axes = fig.subplots(1, 2); fig.subplots_adjust(left=0.08, right=0.95, top=0.78, bottom=0.11, wspace=0.32)
    base.row_dot_plot(axes[0], rows_left, "ATTACKING OUTPUT")
    base.row_dot_plot(axes[1], rows_right, "PROCESS & TERRITORY")
    fig.text(0.945, 0.035, "● FRANCE   ◆ ENGLAND · REAL MATCH DATA", ha="right", fontsize=8, color=NEUTRAL)
    return save(fig, "18_match_stats.png")


def _post_match_metric_panel(ax, title, rows):
    """Use the shared Opta-style comparison contract inside dashboard panels."""
    display_rows = [
        (f"{label} ↓" if lower_better else label, home_value, away_value, fmt)
        for label, home_value, away_value, fmt, lower_better in rows
    ]
    base.row_dot_plot(ax, display_rows, title.upper())


def _derived_expected_assists(events: pd.DataFrame, team_id: int) -> float:
    """Team xA proxy: xG of the shot following each provider-tagged key pass."""
    work = events.copy()
    work["minute_num"] = pd.to_numeric(work.get("minute"), errors="coerce").fillna(0)
    work["second_num"] = pd.to_numeric(work.get("second"), errors="coerce").fillna(0)
    work["period_num"] = pd.to_numeric(work.get("period"), errors="coerce").fillna(0)
    work = work.sort_values(["period_num", "minute_num", "second_num", "event_id"]).reset_index(drop=True)
    live = ~as_bool(work.get("is_penalty_shootout", pd.Series(False, index=work.index)))
    key_pass = as_bool(work.get("is_key_pass", pd.Series(False, index=work.index))) & live
    shot = as_bool(work.get("is_shot", pd.Series(False, index=work.index))) & live
    total = 0.0
    for idx in work.index[key_pass & work["team_id"].eq(team_id)]:
        source = work.loc[idx]
        source_time = float(source["minute_num"]) * 60 + float(source["second_num"])
        for next_idx in range(idx + 1, len(work)):
            candidate = work.loc[next_idx]
            if candidate["period_num"] != source["period_num"]:
                break
            elapsed = float(candidate["minute_num"]) * 60 + float(candidate["second_num"]) - source_time
            if elapsed > 20:
                break
            if candidate["team_id"] == team_id and bool(shot.loc[next_idx]):
                total += float(pd.to_numeric(candidate.get("xG"), errors="coerce") or 0.0)
                break
    return total


def post_match_advanced_dashboard(events, xg, team_metrics):
    """One share-ready 24-metric dashboard joining attack, creation and defence."""
    home_xg, away_xg = xg_row(xg, HOME_NAME), xg_row(xg, AWAY_NAME)
    info = {"home_id": HOME_ID, "away_id": AWAY_ID, "home_name": HOME_NAME, "away_name": AWAY_NAME}
    try:
        ppda = compute_ppda_both(info, events)
        home_ppda = float(ppda["home"]["ppda"] or 0)
        away_ppda = float(ppda["away"]["ppda"] or 0)
    except Exception:
        home_ppda = away_ppda = 0.0

    metric = lambda side, key: float(base.metric_lookup(team_metrics, side, key))
    home_counts, away_counts = team_event_counts(events, HOME_ID), team_event_counts(events, AWAY_ID)
    home_xa = _derived_expected_assists(events, HOME_ID)
    away_xa = _derived_expected_assists(events, AWAY_ID)
    attack_rows = [
        ("Shots", float(home_xg.get("shots", 0)), float(away_xg.get("shots", 0)), "{:.0f}", False),
        ("Shots on target", float(home_xg.get("on_target", 0)), float(away_xg.get("on_target", 0)), "{:.0f}", False),
        ("Big chances", float(home_xg.get("big_chances", 0)), float(away_xg.get("big_chances", 0)), "{:.0f}", False),
        ("Expected goals (xG)", float(home_xg.get("xG", 0)), float(away_xg.get("xG", 0)), "{:.2f}", False),
        ("xG on target (xGoT)", float(home_xg.get("xGoT", 0)), float(away_xg.get("xGoT", 0)), "{:.2f}", False),
        ("xG per shot", float(home_xg.get("xG_per_shot", 0)), float(away_xg.get("xG_per_shot", 0)), "{:.3f}", False),
        ("Transition xG", metric("home", "transition_xG"), metric("away", "transition_xG"), "{:.2f}", False),
        ("Transition shot rate", metric("home", "transition_shot_rate"), metric("away", "transition_shot_rate"), "{:.1f}%", False),
    ]
    creation_rows = [
        ("Possession share", metric("home", "possession_share"), metric("away", "possession_share"), "{:.1f}%", False),
        ("Expected assists (xA)", home_xa, away_xa, "{:.2f}", False),
        ("Open-play xT", float(home_xg.get("xT", 0)), float(away_xg.get("xT", 0)), "{:.2f}", False),
        ("Field tilt", metric("home", "field_tilt"), metric("away", "field_tilt"), "{:.1f}%", False),
        ("Progressive passes", metric("home", "progressive_passes"), metric("away", "progressive_passes"), "{:.0f}", False),
        ("Final-third entries", metric("home", "final_third_entries"), metric("away", "final_third_entries"), "{:.0f}", False),
        ("Deep completions", metric("home", "deep_completions"), metric("away", "deep_completions"), "{:.0f}", False),
        ("Box entries", metric("home", "box_entries"), metric("away", "box_entries"), "{:.0f}", False),
    ]
    defence_rows = [
        ("PPDA", home_ppda, away_ppda, "{:.2f}", True),
        ("Tackles", float(home_counts["Tackles"]), float(away_counts["Tackles"]), "{:.0f}", False),
        ("Interceptions", float(home_counts["Interceptions"]), float(away_counts["Interceptions"]), "{:.0f}", False),
        ("Possession regains", metric("home", "possession_regains"), metric("away", "possession_regains"), "{:.0f}", False),
        ("High regains", metric("home", "high_regains"), metric("away", "high_regains"), "{:.0f}", False),
        ("Counterpress success", metric("home", "counterpress_success_rate"), metric("away", "counterpress_success_rate"), "{:.1f}%", False),
        ("Dangerous counters allowed", metric("home", "rest_defence_dangerous_counters"), metric("away", "rest_defence_dangerous_counters"), "{:.0f}", True),
        ("Rest-defence vulnerability", metric("home", "rest_defence_vulnerability"), metric("away", "rest_defence_vulnerability"), "{:.1f}%", True),
    ]

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    base.amoled_header(
        fig,
        "Post-Match Advanced Dashboard",
        "24 advanced indicators · attack, creation, control and defence · full match · shootout excluded",
        active_team=None,
    )

    panels = [
        ("Attacking output", attack_rows),
        ("Creation & control", creation_rows),
        ("Defensive control", defence_rows),
    ]
    for left, (title, rows) in zip([0.04, 0.355, 0.67], panels):
        ax = fig.add_axes([left, 0.18, 0.29, 0.61])
        _post_match_metric_panel(ax, title, rows)

    territory_team = HOME_NAME if metric("home", "field_tilt") > metric("away", "field_tilt") else AWAY_NAME
    quality_team = HOME_NAME if float(home_xg.get("xG_per_shot", 0)) > float(away_xg.get("xG_per_shot", 0)) else AWAY_NAME
    secure_team = HOME_NAME if metric("home", "rest_defence_vulnerability") < metric("away", "rest_defence_vulnerability") else AWAY_NAME
    fig.text(0.04, 0.112, "MATCH READ", color=FOCUS, fontsize=7.5, fontweight="bold")
    fig.text(
        0.108,
        0.112,
        f"{territory_team} controlled more territory; {quality_team} created the cleaner average shot and {secure_team} protected attacking possessions more securely.",
        color=TEXT,
        fontsize=8.5,
    )
    fig.text(0.04, 0.066, "xA = xG of the shot following each provider-tagged key pass (within 20 seconds).", color=MUTED, fontsize=7)
    fig.text(0.96, 0.066, "↓ LOWER IS BETTER   ·   REAL MATCH EVENTS", color=NEUTRAL, fontsize=7, ha="right")
    return save(fig, "19_post_match_advanced_dashboard.png")


def pass_thirds(events, team_id, number):
    team = events[events["team_id"].eq(team_id) & events["type"].astype(str).eq("Pass")].copy()
    completed = team[team["outcome"].astype(str).str.lower().eq("successful")]
    bins = [0, 33.333, 66.667, 100.001]
    labels = ["Defensive third", "Middle third", "Final third"]
    started = pd.cut(pd.to_numeric(completed["x"], errors="coerce"), bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels, fill_value=0)
    ended = pd.cut(pd.to_numeric(completed["end_x"], errors="coerce"), bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels, fill_value=0)
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.text(0.055, 0.94, f"Pass Distribution by Third · {TEAM_NAME[team_id]}", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.898, "Completed passes by origin and destination third", fontsize=11, color=MUTED)
    ax = fig.add_axes([0.12, 0.18, 0.76, 0.55]); base.clean_ax(ax)
    y = np.arange(3); maxv = max(started.max(), ended.max(), 1)
    ax.barh(y + 0.14, started.values, height=0.26, color=TEAM_COLOR[team_id], alpha=0.95, label="Pass origins")
    ax.barh(y - 0.14, ended.values, height=0.26, color=VALUE, alpha=0.75, label="Pass destinations")
    ax.set_yticks(y); ax.set_yticklabels(labels, color=TEXT); ax.invert_yaxis(); ax.set_xlim(0, maxv * 1.25)
    ax.grid(axis="x", color=GRID, lw=0.7)
    for idx, value in enumerate(started.values): ax.text(value + maxv * 0.02, idx + 0.14, str(int(value)), color=TEAM_COLOR[team_id], va="center", fontweight="bold")
    for idx, value in enumerate(ended.values): ax.text(value + maxv * 0.02, idx - 0.14, str(int(value)), color=VALUE, va="center", fontweight="bold")
    ax.legend(frameon=False, labelcolor=TEXT, loc="lower right")
    return save(fig, f"{number:02d}_pass_thirds_{TEAM_NAME[team_id].lower()}.png")


def progressive(events, team_id, number):
    prog = events[progressive_pass_mask(events) & events["team_id"].eq(team_id)].copy().dropna(subset=["x", "y", "end_x", "end_y"])
    prog["xT"] = pd.to_numeric(prog["xT"], errors="coerce").fillna(0)
    fig, pitch, side = pitch_axes(f"Progressive Passes · {TEAM_NAME[team_id]}", "Canonical zone-aware thresholds · strongest ten actions highlighted by xT added")
    draw_long_pitch(pitch)
    for _, row in prog.iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]]); ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        pitch.annotate("", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]), arrowprops=dict(arrowstyle="-|>", color=VALUE, alpha=0.18, lw=0.65, mutation_scale=7))
    for _, row in prog.nlargest(10, "xT").iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]]); ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        pitch.annotate("", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]), arrowprops=dict(arrowstyle="-|>", color=FOCUS, alpha=0.95, lw=1.9, mutation_scale=11))
    top = prog.groupby("player").size().sort_values(ascending=False).head(7)
    side_title(side, "TOP PROGRESSORS")
    side_rows(side, [(str(name).split()[-1], str(int(value))) for name, value in top.items()])
    side.text(0.08, 0.14, f"All progressive passes: {len(prog)}", color=VALUE, fontsize=9)
    side.text(0.08, 0.09, "Gold = top 10 by xT added", color=FOCUS, fontsize=9)
    return save(fig, f"{number:02d}_progressive_{TEAM_NAME[team_id].lower()}.png")


def crosses(events, team_id, number):
    mask = cross_mask(events)
    frame = events[mask & events["team_id"].eq(team_id)].copy().dropna(subset=["x", "y", "end_x", "end_y"])
    success = frame["outcome"].astype(str).str.lower().eq("successful")
    fig, pitch, side = pitch_axes(f"Crosses · {TEAM_NAME[team_id]}", "Cross origins and targets · completed deliveries use filled arrowheads")
    draw_long_pitch(pitch)
    for idx, row in frame.iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]]); ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        good = bool(success.loc[idx])
        pitch.annotate("", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]), arrowprops=dict(arrowstyle="-|>" if good else "->", color=VALUE if good else MUTED, alpha=0.85 if good else 0.35, lw=1.3 if good else 0.75, mutation_scale=9))
    completed = int(success.sum()); rate = 100 * completed / max(len(frame), 1)
    side_title(side, "CROSSING OUTPUT")
    side_kpis(side, [("Crosses", len(frame)), ("Completed", completed), ("Completion", f"{rate:.1f}%"), ("Open-play", int((~frame["qualifier_names"].astype(str).str.lower().str.contains("corner")).sum()))])
    return save(fig, f"{number:02d}_crosses_{TEAM_NAME[team_id].lower()}.png")


def defensive_activity(events, team_id, number):
    actions = events[events["team_id"].eq(team_id) & events["type"].astype(str).isin(["Tackle", "Interception", "BallRecovery", "Clearance", "BlockedShot", "Foul"])].dropna(subset=["x", "y"]).copy()
    fig, pitch, side = pitch_axes(f"Defensive Activity · {TEAM_NAME[team_id]}", "One sequential density scale; action types use shapes rather than competing colours")
    hx, hy = attack_xy(actions["x"], actions["y"])
    heat, _, _ = np.histogram2d(hx, hy, bins=[7, 12], range=[[-PITCH_WIDTH / 2, PITCH_WIDTH / 2], [0, PITCH_LENGTH]])
    cmap = LinearSegmentedColormap.from_list("def_full", [BG, PANEL_2, VALUE])
    pitch.imshow(heat.T, extent=[-PITCH_WIDTH / 2, PITCH_WIDTH / 2, 0, PITCH_LENGTH], origin="lower", cmap=cmap, aspect="equal", alpha=0.82)
    draw_long_pitch(pitch)
    marker_map = {"Tackle": "o", "Interception": "D", "BallRecovery": "s", "Clearance": "^", "BlockedShot": "P", "Foul": "X"}
    for event_type, marker in marker_map.items():
        subset = actions[actions["type"].astype(str).eq(event_type)]
        if subset.empty: continue
        px, py = attack_xy(subset["x"], subset["y"])
        pitch.scatter(px, py, marker=marker, s=28, facecolors=BG, edgecolors=TEXT, linewidth=0.75, alpha=0.82)
    counts = team_event_counts(events, team_id)
    side_title(side, "ACTION TYPE LEGEND")
    label_to_event = {
        "Tackles": "Tackle", "Interceptions": "Interception", "Recoveries": "BallRecovery",
        "Clearances": "Clearance", "Blocks": "BlockedShot", "Fouls": "Foul",
    }
    for idx, (label, value) in enumerate(counts.items()):
        y = 0.81 - idx * 0.095
        marker = marker_map[label_to_event[label]]
        side.scatter([0.13], [y], s=48, marker=marker, facecolors=BG, edgecolors=TEXT, linewidth=0.9)
        side.text(0.21, y, label, color=TEXT, fontsize=8.5, va="center")
        side.text(0.90, y, str(value), color=FOCUS, fontsize=8.5, fontweight="bold", ha="right", va="center")
        side.plot([0.08, 0.92], [y - 0.043, y - 0.043], color=GRID, lw=0.55, alpha=0.7)
    side.add_patch(Rectangle((0.09, 0.105), 0.08, 0.055, facecolor=VALUE, edgecolor=GRID, alpha=0.75))
    side.text(0.21, 0.132, "Teal intensity = action density", color=TEXT, fontsize=8, va="center")
    return save(fig, f"{number:02d}_defensive_activity_{TEAM_NAME[team_id].lower()}.png")


def defensive_summary(events, team_metrics):
    hc, ac = team_event_counts(events, HOME_ID), team_event_counts(events, AWAY_ID)
    labels = list(hc.keys()) + ["Possession regains"]
    rows = [(label, float(hc.get(label, base.metric_lookup(team_metrics, "home", "possession_regains"))), float(ac.get(label, base.metric_lookup(team_metrics, "away", "possession_regains"))), "{:.0f}") for label in labels]
    fig, ax = base.page("Defensive Summary", "Provider defensive actions plus inferred possession regains · exact values retained")
    base.row_dot_plot(ax, rows)
    return save(fig, "30_defensive_summary.png")


def average_positions(events, players, team_id, number, half):
    positions, _edges, sub_on, sub_off, substitutions, _completed_links = _half_network_data(events, players, team_id, half)
    half_label = "First Half" if half == 1 else "Second Half"
    fig, pitch, side = pitch_axes(
        f"Average Positions · {TEAM_NAME[team_id]} · {half_label}",
        f"All {len(positions)} participants shown · corrected left/right orientation · square = came on · gold outline = went off",
    )
    draw_long_pitch(pitch)
    display = {}
    for name, row in positions.iterrows():
        px, py = player_position_xy([row["x"]], [row["y"]])
        display[str(name)] = (float(px[0]), float(py[0]), float(row["touches"]))
    display = _separate_network_positions(display, min_gap=6.3)
    max_touch = max([value[2] for value in display.values()] or [1])
    for name, (px, py, touches) in display.items():
        entered = name in sub_on
        left = name in sub_off
        pitch.scatter(px, py, s=260 + 640 * touches / max_touch, marker="s" if entered else "o",
                      color=TEAM_COLOR[team_id], edgecolor=FOCUS if left else TEXT,
                      linewidth=2.3 if left else 1.15, zorder=4)
        draw_node_label(pitch, px, py, name, touches, max_touch)

    side_title(side, "HALF PARTICIPATION")
    side.text(0.92, 0.94, f"{len(positions)} players", color=FOCUS, fontsize=8,
              fontweight="bold", ha="right", va="top")
    active = positions.sort_values("touches", ascending=False).head(5)
    side_rows(side, [(compact_player_label(name), str(int(row["touches"]))) for name, row in active.iterrows()], start=0.81, gap=0.075)
    side.text(0.08, 0.40, "SUBSTITUTIONS", color=MUTED, fontsize=7.5, fontweight="bold")
    if substitutions:
        for idx, (minute, on_name, off_name) in enumerate(substitutions[:5]):
            y = 0.35 - idx * 0.052
            side.text(0.08, y, f"{minute}′", color=FOCUS, fontsize=7.5, fontweight="bold", va="center")
            change = f"{off_name} OFF AT INTERVAL" if on_name == "—" else f"{on_name} IN  ·  {off_name} OFF"
            side.text(0.19, y, change, color=TEXT, fontsize=7.2, va="center")
    else:
        side.text(0.08, 0.34, "No in-half changes", color=MUTED, fontsize=8)
    side.scatter([0.12, 0.31], [0.075, 0.075], s=[65, 65], marker="o", color=TEAM_COLOR[team_id], edgecolor=[TEXT, FOCUS], linewidth=[1.0, 2.1])
    side.scatter([0.50], [0.075], s=65, marker="s", color=TEAM_COLOR[team_id], edgecolor=TEXT, linewidth=1.0)
    side.text(0.16, 0.075, "Began half", color=TEXT, fontsize=6.8, va="center")
    side.text(0.35, 0.075, "Went off", color=TEXT, fontsize=6.8, va="center")
    side.text(0.54, 0.075, "Came on", color=TEXT, fontsize=6.8, va="center")
    suffix = "1h" if half == 1 else "2h"
    return save(fig, f"{number:02d}{'a' if half == 1 else 'b'}_average_positions_{TEAM_NAME[team_id].lower()}_{suffix}.png")


def dominating_zones(events):
    tmask = touch_mask(events)
    home = events[tmask & events["team_id"].eq(HOME_ID)].dropna(subset=["x", "y"])
    away = events[tmask & events["team_id"].eq(AWAY_ID)].dropna(subset=["x", "y"])
    hh, _, _ = np.histogram2d(home["y"], home["x"], bins=[5, 7], range=[[0, 100], [0, 100]])
    ah, _, _ = np.histogram2d(away["y"], away["x"], bins=[5, 7], range=[[0, 100], [0, 100]])
    total = hh + ah
    diff = np.divide(hh - ah, total, out=np.zeros_like(total), where=total > 0)
    counts = hh - ah
    fig, pitch, side = pitch_axes("Dominating Zones", "Touch-share difference · diverging scale centred on an even 50/50 split")
    cmap = LinearSegmentedColormap.from_list("dom_full", [AWAY, PANEL_2, HOME])
    image = pitch.imshow(diff.T, extent=[-PITCH_WIDTH / 2, PITCH_WIDTH / 2, 0, PITCH_LENGTH], origin="lower", cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    draw_long_pitch(pitch)
    for ix in range(5):
        for iy in range(7):
            px = -PITCH_WIDTH / 2 + (ix + 0.5) * PITCH_WIDTH / 5
            py = (iy + 0.5) * PITCH_LENGTH / 7
            pitch.text(px, py, f"{int(counts[ix, iy]):+d}", ha="center", va="center", color=TEXT, fontsize=6.5, fontweight="bold")
    cbar = fig.colorbar(image, ax=pitch, fraction=0.035, pad=0.02, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels([AWAY_NAME, "Balanced", HOME_NAME]); cbar.ax.tick_params(colors=MUTED, labelsize=7); cbar.outline.set_edgecolor(GRID)
    side_title(side, "TERRITORY TOTALS")
    side_kpis(side, [("France touches", len(home)), ("England touches", len(away)), ("Difference", f"{len(home)-len(away):+d}"), ("Cell label", "France − England")])
    return save(fig, "33_dominating_zones.png")


def box_entries(events, team_id, number):
    frame = events[box_entry_mask(events) & events["team_id"].eq(team_id)].copy().dropna(subset=["x", "y", "end_x", "end_y"])
    fig, pitch, side = pitch_axes(f"Box Entries · {TEAM_NAME[team_id]}", "Completed actions entering the penalty area · entry method encoded by shape")
    draw_long_pitch(pitch)
    for _, row in frame.iterrows():
        sx, sy = attack_xy([row["x"]], [row["y"]]); ex, ey = attack_xy([row["end_x"]], [row["end_y"]])
        event_type = str(row["type"])
        marker = "o" if event_type == "Pass" else "D"
        entry_color = VALUE if marker == "o" else FOCUS
        pitch.annotate("", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]), arrowprops=dict(arrowstyle="-|>", color=entry_color, alpha=0.58, lw=1.0, mutation_scale=8))
        pitch.scatter(ex[0], ey[0], s=32, marker=marker, color=entry_color, edgecolor=TEXT, linewidth=0.65)
    top = frame.groupby("player").size().sort_values(ascending=False).head(5)
    side_title(side, "ENTRY CONTRIBUTORS")
    side_rows(side, [(str(name).split()[-1], str(int(value))) for name, value in top.items()], start=0.81, gap=0.083)
    pass_count = int(frame["type"].astype(str).eq("Pass").sum())
    carry_count = len(frame) - pass_count
    side.text(0.08, 0.35, "ENTRY METHOD LEGEND", color=MUTED, fontsize=7.5, fontweight="bold")
    side.scatter([0.13], [0.285], s=45, marker="o", color=VALUE, edgecolor=TEXT, linewidth=0.6)
    side.text(0.21, 0.285, f"Pass entry ({pass_count})", color=TEXT, fontsize=8, va="center")
    side.scatter([0.13], [0.22], s=45, marker="D", color=FOCUS, edgecolor=TEXT, linewidth=0.6)
    side.text(0.21, 0.22, f"Carry / take-on ({carry_count})", color=TEXT, fontsize=8, va="center")
    side.annotate("", xy=(0.18, 0.15), xytext=(0.08, 0.15), arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.1))
    side.text(0.21, 0.15, "Arrow = entry path", color=TEXT, fontsize=8, va="center")
    side.text(0.08, 0.075, f"Total entries: {len(frame)}", color=FOCUS, fontsize=9.5, fontweight="bold")
    return save(fig, f"{number:02d}_box_entries_{TEAM_NAME[team_id].lower()}.png")


def high_regains(events, team_id, number):
    frame = high_regain_events(events, team_id).dropna(subset=["x", "y"]).copy()
    fig, pitch, side = pitch_axes(f"High Regains · {TEAM_NAME[team_id]}", "Open-play possession regains at x ≥ 60 · type encoded by shape")
    draw_long_pitch(pitch)
    threshold_y = 60 * PITCH_LENGTH / 100
    pitch.axhspan(threshold_y, PITCH_LENGTH, color=FOCUS, alpha=0.055)
    pitch.axhline(threshold_y, color=FOCUS, lw=1.0, ls=(0, (5, 4)))
    marker_map = {"Tackle": "o", "Interception": "D", "BallRecovery": "s"}
    for event_type, marker in marker_map.items():
        subset = frame[frame["type"].astype(str).eq(event_type)]
        if subset.empty: continue
        px, py = attack_xy(subset["x"], subset["y"])
        pitch.scatter(px, py, s=65, marker=marker, color=VALUE, edgecolor=TEXT, linewidth=0.8, label=f"{event_type} ({len(subset)})")
    other = frame[~frame["type"].astype(str).isin(marker_map)]
    if not other.empty:
        px, py = attack_xy(other["x"], other["y"]); pitch.scatter(px, py, s=65, marker="^", color=VALUE, edgecolor=TEXT, linewidth=0.8, label=f"Other ({len(other)})")
    pitch.legend(loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, labelcolor=TEXT, fontsize=7)
    top = frame.groupby("player").size().sort_values(ascending=False).head(7) if "player" in frame else pd.Series(dtype=int)
    side_title(side, "HIGH-REGAIN LEADERS")
    side_rows(side, [(str(name).split()[-1], str(int(value))) for name, value in top.items()])
    side.text(0.08, 0.14, f"Total high regains: {len(frame)}", color=FOCUS, fontsize=9, fontweight="bold")
    return save(fig, f"{number:02d}_high_regains_{TEAM_NAME[team_id].lower()}.png")


def pass_targets(events, team_id, number):
    frame = events[events["team_id"].eq(team_id) & events["type"].astype(str).eq("Pass") & events["outcome"].astype(str).str.lower().eq("successful")].dropna(subset=["end_x", "end_y"]).copy()
    heat, _, _ = np.histogram2d(frame["end_y"], frame["end_x"], bins=[7, 12], range=[[0, 100], [0, 100]])
    fig, pitch, side = pitch_axes(f"Pass Target Zones · {TEAM_NAME[team_id]}", "Completed-pass destinations · one sequential density scale")
    cmap = LinearSegmentedColormap.from_list("targets", [BG, PANEL_2, VALUE])
    image = pitch.imshow(heat.T, extent=[-PITCH_WIDTH / 2, PITCH_WIDTH / 2, 0, PITCH_LENGTH], origin="lower", cmap=cmap, aspect="equal", alpha=0.95)
    draw_long_pitch(pitch)
    max_cell = max(float(heat.max()), 1.0)
    for ix in range(7):
        for iy in range(12):
            x0 = -PITCH_WIDTH / 2 + ix * PITCH_WIDTH / 7
            y0 = iy * PITCH_LENGTH / 12
            pitch.add_patch(Rectangle((x0, y0), PITCH_WIDTH / 7, PITCH_LENGTH / 12,
                                      fill=False, edgecolor=GRID, lw=0.42, alpha=0.55, zorder=3))
            value = int(heat[ix, iy])
            color = TEXT if value >= max_cell * 0.30 else MUTED
            pitch.text(x0 + PITCH_WIDTH / 14, y0 + PITCH_LENGTH / 24, str(value),
                       color=color, fontsize=5.6, fontweight="bold", ha="center", va="center", zorder=5)
    cbar = fig.colorbar(image, ax=pitch, fraction=0.035, pad=0.02); cbar.ax.tick_params(colors=MUTED, labelsize=7); cbar.outline.set_edgecolor(GRID); cbar.set_label("Completed-pass targets", color=MUTED, fontsize=8)
    top = frame.groupby("player").size().sort_values(ascending=False).head(7)
    side_title(side, "TOP PASSERS")
    side_rows(side, [(str(name).split()[-1], str(int(value))) for name, value in top.items()])
    side.text(0.08, 0.14, f"Completed passes: {len(frame)}", color=VALUE, fontsize=9, fontweight="bold")
    return save(fig, f"{number:02d}_pass_targets_{TEAM_NAME[team_id].lower()}.png")


def ppda(events):
    info = {"home_id": HOME_ID, "away_id": AWAY_ID, "home_name": HOME_NAME, "away_name": AWAY_NAME}
    data = compute_ppda_both(info, events)
    hp = float(data["home"]["ppda"] or 0); ap = float(data["away"]["ppda"] or 0)
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.text(0.055, 0.95, "PPDA & Pressing Intensity", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.91, "Opponent passes allowed per defensive action · lower PPDA means more aggressive pressure", fontsize=10.5, color=MUTED)
    fig.add_artist(Line2D([0.055, 0.945], [0.875, 0.875], transform=fig.transFigure, color=GRID, lw=1))

    def gauge(rect, team_name, value, color, passes, actions):
        ax = fig.add_axes(rect)
        ax.set_facecolor(BG); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.72, 1.18); ax.set_aspect("equal"); ax.axis("off")
        bounds = [(5, 10, VALUE), (10, 15, "#8E7CE8"), (15, 20, FOCUS), (20, 25, "#B86A4B")]
        for low, high, band_color in bounds:
            theta_high = 180 - (low - 5) / 20 * 180
            theta_low = 180 - (high - 5) / 20 * 180
            ax.add_patch(Wedge((0, 0), 1.0, theta_low, theta_high, width=0.18,
                               facecolor=band_color, edgecolor=BG, lw=1.0, alpha=0.86))
        for tick in [5, 10, 15, 20, 25]:
            angle = np.deg2rad(180 - (tick - 5) / 20 * 180)
            ax.text(1.09 * np.cos(angle), 1.09 * np.sin(angle), str(tick), color=MUTED,
                    fontsize=7.5, ha="center", va="center")
        clipped = float(np.clip(value, 5, 25))
        angle = np.deg2rad(180 - (clipped - 5) / 20 * 180)
        ax.plot([0, 0.82 * np.cos(angle)], [0, 0.82 * np.sin(angle)], color=color, lw=4.2, solid_capstyle="round", zorder=5)
        ax.scatter([0.82 * np.cos(angle)], [0.82 * np.sin(angle)], s=65, color=TEXT, edgecolor=color, linewidth=2.0, zorder=6)
        ax.scatter([0], [0], s=62, color=color, edgecolor=TEXT, linewidth=0.8, zorder=6)
        ax.text(0, 1.12, team_name, color=TEXT, fontsize=14, fontweight="bold", ha="center")
        ax.text(0, -0.18, f"{value:.2f}", color=color, fontsize=27, fontweight="bold", ha="center", va="center")
        level = "ELITE PRESS" if value < 7.5 else "HIGH PRESS" if value < 10 else "MID PRESS" if value < 14 else "LOW PRESS"
        ax.text(0, -0.38, level, color=FOCUS, fontsize=9, fontweight="bold", ha="center")
        ax.text(-0.48, -0.56, f"{passes}", color=color, fontsize=13, fontweight="bold", ha="center")
        ax.text(-0.48, -0.67, "OPP PASSES", color=MUTED, fontsize=6.5, fontweight="bold", ha="center")
        ax.text(0.48, -0.56, f"{actions}", color=color, fontsize=13, fontweight="bold", ha="center")
        ax.text(0.48, -0.67, "DEF ACTIONS", color=MUTED, fontsize=6.5, fontweight="bold", ha="center")

    gauge([0.08, 0.42, 0.38, 0.41], HOME_NAME, hp, HOME, data["home"]["passes_allowed"], data["home"]["defensive_actions"])
    gauge([0.54, 0.42, 0.38, 0.41], AWAY_NAME, ap, AWAY, data["away"]["passes_allowed"], data["away"]["defensive_actions"])
    leader = HOME_NAME if hp < ap else AWAY_NAME
    leader_color = HOME if hp < ap else AWAY
    fig.add_artist(Rectangle((0.285, 0.302), 0.43, 0.076, transform=fig.transFigure,
                             facecolor=PANEL, edgecolor=GRID, linewidth=0.8))
    fig.text(0.5, 0.349, f"{leader} pressed more aggressively", color=leader_color,
             fontsize=13, fontweight="bold", ha="center")
    fig.text(0.5, 0.318, f"PPDA differential: {abs(hp - ap):.2f}", color=MUTED, fontsize=9, ha="center")
    zone = fig.add_axes([0.10, 0.12, 0.80, 0.14]); zone.set_xlim(0, 100); zone.set_ylim(0, 1); zone.axis("off")
    zone.add_patch(Rectangle((0, 0.12), 40, 0.66, facecolor=PANEL, edgecolor=GRID, lw=1.0))
    zone.add_patch(Rectangle((40, 0.12), 60, 0.66, facecolor=FOCUS, edgecolor=GRID, lw=1.0, alpha=0.16))
    zone.axvline(40, ymin=0.12, ymax=0.78, color=FOCUS, lw=1.2, ls=(0, (4, 3)))
    zone.text(20, 0.45, "OWN 40%", color=MUTED, fontsize=10, fontweight="bold", ha="center")
    zone.text(70, 0.45, "PPDA PRESSING ZONE · OPPONENT 60%", color=FOCUS, fontsize=10, fontweight="bold", ha="center")
    zone.text(0, 0.89, "← own goal", color=MUTED, fontsize=7, ha="left")
    zone.text(100, 0.89, "opponent goal →", color=MUTED, fontsize=7, ha="right")
    fig.text(0.945, 0.035, "METHOD: OPPONENT PASSES ÷ TACKLES + INTERCEPTIONS + FOULS + CHALLENGES + RECOVERIES", ha="right", fontsize=7.5, color=NEUTRAL)
    return save(fig, "40_ppda_pressing.png")


def transition_outcomes(events):
    annotated, possessions = build_possessions(events)
    team_data = {}
    for team_id in [HOME_ID, AWAY_ID]:
        transitions = possessions[
            possessions["team_id"].eq(team_id)
            & possessions["is_transition"].astype(bool)
        ].copy()
        total = len(transitions)
        chances = int((pd.to_numeric(transitions["transition_shots"], errors="coerce").fillna(0) > 0).sum()) if total else 0
        shots = int(pd.to_numeric(transitions["transition_shots"], errors="coerce").fillna(0).sum()) if total else 0
        goals = int(pd.to_numeric(transitions["transition_goals"], errors="coerce").fillna(0).sum()) if total else 0
        paths = []
        for transition in transitions.itertuples():
            window = annotated[
                annotated["possession_id"].eq(int(transition.possession_id))
                & annotated["team_id"].eq(team_id)
                & (pd.to_numeric(annotated["_clock_seconds"], errors="coerce") <= float(transition.start_time) + 12.0)
            ].copy()
            points = [(float(transition.start_x), float(transition.start_y))]
            for _, row in window.iterrows():
                for x_value, y_value in [(row.get("_x", np.nan), row.get("_y", np.nan)),
                                         (row.get("_end_x", np.nan), row.get("_end_y", np.nan))]:
                    if pd.notna(x_value) and pd.notna(y_value):
                        points.append((float(x_value), float(y_value)))
            end_x, end_y = max(points, key=lambda point: point[0])
            paths.append({
                "start_x": float(transition.start_x),
                "start_y": float(transition.start_y),
                "end_x": end_x,
                "end_y": end_y,
                "shot": int(transition.transition_shots) > 0,
                "goal": int(transition.transition_goals) > 0,
                "minute": int(float(transition.start_time) // 60),
            })
        team_data[team_id] = {
            "total": total,
            "chances": chances,
            "shots": shots,
            "goals": goals,
            "xg": float(pd.to_numeric(transitions["transition_xG"], errors="coerce").fillna(0).sum()) if total else 0.0,
            "box_entries": int(pd.to_numeric(transitions["transition_box_entries"], errors="coerce").fillna(0).sum()) if total else 0,
            "paths": paths,
        }

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.text(0.055, 0.95, "Transition Map & Outcomes", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.908, "Each line runs from the regain/turnover location to the most advanced point reached inside 12 seconds", fontsize=10.5, color=MUTED)
    fig.add_artist(Line2D([0.055, 0.945], [0.872, 0.872], transform=fig.transFigure, color=GRID, lw=1))

    layouts = [
        (HOME_ID, [0.055, 0.145, 0.265, 0.675], [0.335, 0.215, 0.135, 0.515], 0.19),
        (AWAY_ID, [0.525, 0.145, 0.265, 0.675], [0.805, 0.215, 0.135, 0.515], 0.66),
    ]
    for team_id, pitch_rect, card_rect, title_x in layouts:
        data = team_data[team_id]
        fig.text(title_x, 0.835, TEAM_NAME[team_id], color=TEXT, fontsize=14,
                 fontweight="bold", ha="center")
        pitch = fig.add_axes(pitch_rect)
        pitch.axhspan(66.7 * PITCH_LENGTH / 100, PITCH_LENGTH, color=FOCUS, alpha=0.035, zorder=0)
        draw_long_pitch(pitch)
        ordered_paths = sorted(data["paths"], key=lambda item: (item["goal"], item["shot"]))
        for path in ordered_paths:
            sx, sy = player_position_xy([path["start_x"]], [path["start_y"]])
            ex, ey = player_position_xy([path["end_x"]], [path["end_y"]])
            if path["goal"]:
                color, alpha, width, marker, size = FOCUS, 0.95, 2.25, "*", 75
            elif path["shot"]:
                color, alpha, width, marker, size = VALUE, 0.82, 1.45, "D", 25
            else:
                color, alpha, width, marker, size = TEAM_COLOR[team_id], 0.15, 0.65, None, 0
            pitch.annotate("", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]),
                           arrowprops=dict(arrowstyle="-|>", color=color, alpha=alpha,
                                           lw=width, mutation_scale=7 if not path["goal"] else 10), zorder=3)
            pitch.scatter(sx[0], sy[0], s=8, facecolors=BG, edgecolors=color,
                          linewidth=0.55, alpha=max(alpha, 0.35), zorder=4)
            if marker:
                pitch.scatter(ex[0], ey[0], s=size, marker=marker, color=color,
                              edgecolor=TEXT, linewidth=0.55, zorder=5)
            if path["goal"]:
                pitch.text(ex[0], ey[0] + 2.0, f"{path['minute']}′", color=FOCUS,
                           fontsize=6.5, fontweight="bold", ha="center", zorder=6)

        card = fig.add_axes(card_rect)
        card.set_facecolor(PANEL); card.set_xlim(0, 1); card.set_ylim(0, 1); card.set_xticks([]); card.set_yticks([])
        for spine in card.spines.values():
            spine.set_color(GRID)
        side_title(card, "OUTCOMES")
        side_kpis(card, [
            ("Transitions", data["total"]),
            ("Created chance", data["chances"]),
            ("Goals", data["goals"]),
            ("Transition xG", f"{data['xg']:.2f}"),
            ("Box entries", data["box_entries"]),
        ], start=0.82, gap=0.135)
        chance_rate = 100 * data["chances"] / max(data["total"], 1)
        card.text(0.08, 0.065, f"Chance rate: {chance_rate:.1f}%", color=TEAM_COLOR[team_id],
                  fontsize=7.5, fontweight="bold")

    legend = [
        Line2D([0], [0], color=HOME, lw=2, alpha=0.35, label="Transition without shot"),
        Line2D([0], [0], color=VALUE, lw=2, marker="D", markersize=5, label="Created a chance"),
        Line2D([0], [0], color=FOCUS, lw=2.5, marker="*", markersize=8, label="Ended in a goal"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.067), ncol=3,
               frameon=False, labelcolor=TEXT, fontsize=8)
    fig.text(0.055, 0.032,
             "DEFINITION  Open-play regain/turnover that within 12s progresses ≥20m, reaches the final third/box, or produces a shot · restarts excluded.",
             color=NEUTRAL, fontsize=7.5)
    fig.text(0.945, 0.032, "ARROW = START → MOST ADVANCED POINT", ha="right", fontsize=7.5, color=NEUTRAL)
    return save(fig, "41_transition_outcomes.png")


def _game_state_durations(events):
    period = events.get("period_code", pd.Series("", index=events.index)).astype(str).str.lower()
    live = events[~period.isin(["pre", "prematch", "post", "postgame", "pso", "penaltyshootout"])].copy()
    live["_clock"] = pd.to_numeric(live["minute"], errors="coerce").fillna(0) * 60 + pd.to_numeric(live["second"], errors="coerce").fillna(0)
    shootout = as_bool(live.get("is_penalty_shootout", pd.Series(False, index=live.index)))
    goals = live[as_bool(live.get("is_goal", pd.Series(False, index=live.index))) & ~shootout].sort_values(["_clock", "event_id"], kind="stable")
    end_time = float(live["_clock"].max()) if not live.empty else 0.0
    durations = {"drawing": 0.0, "home_ahead": 0.0, "away_ahead": 0.0}
    home_score = away_score = 0
    previous = 0.0

    def current_state():
        if home_score == away_score:
            return "drawing"
        return "home_ahead" if home_score > away_score else "away_ahead"

    for _, goal in goals.iterrows():
        clock = float(goal["_clock"])
        durations[current_state()] += max(clock - previous, 0.0)
        if int(float(goal.get("team_id", 0) or 0)) == HOME_ID:
            home_score += 1
        elif int(float(goal.get("team_id", 0) or 0)) == AWAY_ID:
            away_score += 1
        previous = clock
    durations[current_state()] += max(end_time - previous, 0.0)
    return durations, end_time


def game_state(events, team_metrics):
    durations, match_seconds = _game_state_durations(events)
    scenarios = [
        {
            "title": "SCORE LEVEL",
            "subtitle": "France level · England level",
            "duration_key": "drawing",
            "home_state": "drawing",
            "away_state": "drawing",
            "color": NEUTRAL,
        },
        {
            "title": "FRANCE AHEAD",
            "subtitle": "France leading · England trailing",
            "duration_key": "home_ahead",
            "home_state": "leading",
            "away_state": "trailing",
            "color": HOME,
        },
        {
            "title": "ENGLAND AHEAD",
            "subtitle": "France trailing · England leading",
            "duration_key": "away_ahead",
            "home_state": "trailing",
            "away_state": "leading",
            "color": AWAY,
        },
    ]
    specs = [("Shots", "shots", "{:.0f}"), ("xG", "xG", "{:.2f}"), ("Transitions", "transitions", "{:.0f}"), ("Box entries", "box_entries", "{:.0f}")]
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.text(0.055, 0.95, "Game-State Output by Match Situation", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.908, "Each card is one shared scoreboard situation · values are totals from possessions starting in that situation", fontsize=10.5, color=MUTED)
    fig.add_artist(Line2D([0.055, 0.945], [0.872, 0.872], transform=fig.transFigure, color=GRID, lw=1))

    timeline = fig.add_axes([0.075, 0.775, 0.85, 0.065])
    timeline.set_xlim(0, max(match_seconds, 1)); timeline.set_ylim(0, 1); timeline.axis("off")
    left = 0.0
    for scenario in scenarios:
        duration = durations[scenario["duration_key"]]
        if duration > 0:
            timeline.barh(0.56, duration, left=left, height=0.36, color=scenario["color"], alpha=0.82, edgecolor=BG, linewidth=1.0)
            if duration / max(match_seconds, 1) > 0.10:
                timeline.text(left + duration / 2, 0.56, scenario["title"], color=TEXT, fontsize=7.5, fontweight="bold", ha="center", va="center")
        left += duration
    timeline.text(0, 0.03, "MATCH TIME SHARE", color=MUTED, fontsize=6.8, ha="left")
    timeline.text(max(match_seconds, 1), 0.03, f"TOTAL · {match_seconds / 60:.1f} min", color=MUTED, fontsize=6.8, ha="right")
    legend_x = [0.15, 0.42, 0.70]
    for x, scenario in zip(legend_x, scenarios):
        minutes = durations[scenario["duration_key"]] / 60
        fig.add_artist(Rectangle((x, 0.735), 0.014, 0.014, transform=fig.transFigure,
                                 facecolor=scenario["color"], edgecolor=TEXT, lw=0.45, alpha=0.9))
        fig.text(x + 0.020, 0.742, f"{scenario['title'].title()} · {minutes:.1f} min", color=TEXT, fontsize=8, va="center")

    active_scenarios = [scenario for scenario in scenarios if durations[scenario["duration_key"]] >= 3.0]
    inactive_scenarios = [scenario for scenario in scenarios if durations[scenario["duration_key"]] < 3.0]
    if len(active_scenarios) == 3:
        card_positions = [[0.055, 0.16, 0.285, 0.535], [0.3575, 0.16, 0.285, 0.535], [0.66, 0.16, 0.285, 0.535]]
    elif len(active_scenarios) == 2:
        card_positions = [[0.06, 0.16, 0.415, 0.535], [0.525, 0.16, 0.415, 0.535]]
    else:
        card_positions = [[0.20, 0.16, 0.60, 0.535]]
    if inactive_scenarios:
        inactive_labels = {
            "FRANCE AHEAD": "France never led",
            "ENGLAND AHEAD": "England never led",
            "SCORE LEVEL": "The score was never level",
        }
        note = " · ".join(inactive_labels[scenario["title"]] for scenario in inactive_scenarios)
        fig.text(0.50, 0.705, f"NOT PLAYED · {note}", color=FOCUS, fontsize=8.2,
                 fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.38", fc=PANEL, ec=GRID, lw=0.7))
    for rect, scenario in zip(card_positions, active_scenarios):
        ax = fig.add_axes(rect)
        ax.set_facecolor(PANEL); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(scenario["color"]); spine.set_linewidth(1.2)
        duration_minutes = durations[scenario["duration_key"]] / 60
        ax.text(0.06, 0.93, scenario["title"], color=TEXT, fontsize=12, fontweight="bold", va="top")
        ax.text(0.06, 0.865, scenario["subtitle"], color=MUTED, fontsize=7.4, va="top")
        ax.text(0.94, 0.93, f"{duration_minutes:.1f} min", color=scenario["color"], fontsize=9,
                fontweight="bold", ha="right", va="top")
        ax.plot([0.06, 0.94], [0.81, 0.81], color=GRID, lw=0.8)
        ax.text(0.57, 0.755, HOME_NAME.upper(), color=HOME, fontsize=7.5, fontweight="bold", ha="center")
        ax.text(0.84, 0.755, AWAY_NAME.upper(), color=AWAY, fontsize=7.5, fontweight="bold", ha="center")
        for idx, (label, key, fmt) in enumerate(specs):
            y = 0.66 - idx * 0.135
            home_value = base.metric_lookup(team_metrics, "home", f"game_state_{scenario['home_state']}_{key}")
            away_value = base.metric_lookup(team_metrics, "away", f"game_state_{scenario['away_state']}_{key}")
            ax.text(0.07, y, label, color=TEXT, fontsize=8.5, va="center")
            ax.text(0.57, y, fmt.format(home_value), color=HOME, fontsize=12, fontweight="bold", ha="center", va="center")
            ax.text(0.84, y, fmt.format(away_value), color=AWAY, fontsize=12, fontweight="bold", ha="center", va="center")
            ax.plot([0.07, 0.93], [y - 0.064, y - 0.064], color=GRID, lw=0.55, alpha=0.75)
        ax.text(0.07, 0.075, "Values are totals, not per-90 rates.", color=NEUTRAL, fontsize=6.7)

    fig.text(0.055, 0.095, "HOW TO READ  France ahead pairs France's leading output with England's trailing output; England ahead does the reverse.", color=MUTED, fontsize=8.2)
    fig.text(0.055, 0.066, "Game state is assigned at possession start · Timeline duration is reconstructed from goal times.", color=NEUTRAL, fontsize=7.5)
    fig.text(0.945, 0.035, "BLUE = FRANCE · ORANGE = ENGLAND · REAL MATCH DATA", ha="right", fontsize=7.5, color=NEUTRAL)
    return save(fig, "43_game_state_splits.png")


def player_sequence(player_metrics):
    metrics = [("xGChain", "xGChain"), ("xGBuildup", "xGBuildup"), ("Sequence xT", "sequence_xT")]
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.text(0.055, 0.95, "Player Sequence Contribution", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.91, "Top players by xGChain, xGBuildup and sequence xT involvement", fontsize=11, color=MUTED)
    axes = fig.subplots(1, 3); fig.subplots_adjust(left=0.075, right=0.96, top=0.82, bottom=0.10, wspace=0.42)
    for ax, (title, column) in zip(axes, metrics):
        top = player_metrics.sort_values(column, ascending=False).head(8).sort_values(column)
        colors = [HOME if str(team).lower() == HOME_NAME.lower() else AWAY for team in top["team"]]
        ax.barh(top["player"].astype(str).str.split().str[-1], top[column], color=colors, alpha=0.9)
        base.clean_ax(ax); ax.grid(axis="x", color=GRID, lw=0.65); ax.set_title(title.upper(), loc="left", color=MUTED, fontsize=10, fontweight="bold")
        for idx, value in enumerate(top[column]): ax.text(value + max(top[column].max(), 0.01) * 0.025, idx, f"{value:.2f}", color=TEXT, va="center", fontsize=8)
    fig.text(0.945, 0.035, "BLUE = FRANCE · ORANGE = ENGLAND", ha="right", fontsize=8, color=NEUTRAL)
    return save(fig, "44_player_sequence_leaders.png")


def non_pitch_pages(events, xg, team_metrics):
    base.OUT_DIR = OUT
    paths = {}
    paths[1] = base.xg_flow(events)
    paths[11] = base.shot_profile(xg, events)
    paths[20] = base.touches(team_metrics)
    paths[23] = base.xt_per_minute(events)
    paths[41] = base.advanced_metrics(team_metrics)
    paths[42] = paths[41]
    renamed = {
        1: "01_xg_flow.png",
        11: "11_shot_profile.png",
        20: "20_ball_touches.png",
        23: "23_xt_per_minute.png",
        42: "42_advanced_metrics.png",
    }
    result = {}
    for number, filename in renamed.items():
        src = paths[number]
        dst = OUT / filename
        if src != dst:
            shutil.copy2(src, dst)
            src.unlink(missing_ok=True)
        result[number] = dst
    return result


def build_pdf(
    paths: list[Path],
    events: pd.DataFrame | None = None,
    xg: pd.DataFrame | None = None,
    team_metrics: pd.DataFrame | None = None,
    player_metrics: pd.DataFrame | None = None,
):
    """Build one connected tactical report followed by the full player appendix."""
    if any(value is None for value in [events, xg, team_metrics, player_metrics]):
        loaded_events, _players, loaded_xg, loaded_team_metrics, loaded_player_metrics = load_all()
        events = loaded_events if events is None else events
        xg = loaded_xg if xg is None else xg
        team_metrics = loaded_team_metrics if team_metrics is None else team_metrics
        player_metrics = loaded_player_metrics if player_metrics is None else player_metrics

    from tactical_pdf_report import build_tactical_pdf

    return build_tactical_pdf(
        paths,
        OUT / "full_visual_redesign_real_data.pdf",
        events,
        xg,
        team_metrics,
        player_metrics,
        {
            "home_id": HOME_ID,
            "away_id": AWAY_ID,
            "home_name": HOME_NAME,
            "away_name": AWAY_NAME,
        },
    )


def build_catalog(paths: list[Path]):
    rows = []
    for path in paths:
        stem = path.stem
        number = stem.split("_", 1)[0]
        try:
            file_name = path.relative_to(OUT.resolve()).as_posix()
        except ValueError:
            file_name = path.name
        rows.append({"number": number, "title": stem.split("_", 1)[1].replace("_", " ").title(), "file": file_name, "has_pitch": any(token in stem for token in ["shot_map", "pass_network", "pass_map", "xt_map", "danger", "zone14", "progressive", "crosses", "defensive_activity", "average_positions", "dominating", "box_entries", "high_regains", "pass_targets", "transition_outcomes"])})
    catalog = pd.DataFrame(rows).sort_values("number")
    catalog.to_csv(OUT / "visual_catalog.csv", index=False, encoding="utf-8-sig")
    return catalog


def player_pizzas(events: pd.DataFrame) -> list[Path]:
    """Export one canonical radar per player inside the team folders."""
    from player_radar import export_player_radars

    info = {
        "home_id": HOME_ID,
        "away_id": AWAY_ID,
        "home_name": HOME_NAME,
        "away_name": AWAY_NAME,
        "score": "4-6",
    }
    export_player_radars(events, info, str(OUT), dpi=135)
    source_root = OUT / "player_radars"
    exported = []
    for team in (HOME_NAME, AWAY_NAME):
        team_dir = source_root / team
        if not team_dir.exists():
            continue
        for source in sorted(team_dir.glob("*.png")):
            exported.append(source)
    return exported


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for obsolete in [
        "05_pass_network_france.png",
        "06_pass_network_england.png",
        "31_average_positions_france.png",
        "32_average_positions_england.png",
        "41_transition_performance.png",
    ]:
        (OUT / obsolete).unlink(missing_ok=True)
    for duplicate_radar in OUT.glob("45*_player_pizza_*.png"):
        duplicate_radar.unlink(missing_ok=True)
    base.theme()
    events, players, xg, team_metrics, player_metrics = load_all()
    team_metrics.to_csv(OUT / "team_advanced_metrics_real_data.csv", index=False, encoding="utf-8-sig")
    player_metrics.to_csv(OUT / "player_sequence_metrics_real_data.csv", index=False, encoding="utf-8-sig")
    generated = non_pitch_pages(events, xg, team_metrics)
    paths = [
        generated[1],
        shot_map(events, xg, HOME_ID, 2),
        shot_map(events, xg, AWAY_ID, 3),
        goals_breakdown(events),
        pass_network(events, players, HOME_ID, 5, 1),
        pass_network(events, players, HOME_ID, 5, 2),
        pass_network(events, players, AWAY_ID, 6, 1),
        pass_network(events, players, AWAY_ID, 6, 2),
        xt_map(events, HOME_ID, 7),
        xt_map(events, AWAY_ID, 8),
        pass_map(events, HOME_ID, 9),
        pass_map(events, AWAY_ID, 10),
        generated[11],
        danger_creation(events, team_metrics, HOME_ID, 12),
        danger_creation(events, team_metrics, AWAY_ID, 13),
        gk_saves(events, xg),
        xg_summary(xg),
        zone14(events, HOME_ID, 16),
        zone14(events, AWAY_ID, 17),
        match_stats(events, xg, team_metrics),
        post_match_advanced_dashboard(events, xg, team_metrics),
        generated[20],
        pass_thirds(events, HOME_ID, 21),
        pass_thirds(events, AWAY_ID, 22),
        generated[23],
        progressive(events, HOME_ID, 24),
        progressive(events, AWAY_ID, 25),
        crosses(events, HOME_ID, 26),
        crosses(events, AWAY_ID, 27),
        defensive_activity(events, HOME_ID, 28),
        defensive_activity(events, AWAY_ID, 29),
        defensive_summary(events, team_metrics),
        average_positions(events, players, HOME_ID, 31, 1),
        average_positions(events, players, HOME_ID, 31, 2),
        average_positions(events, players, AWAY_ID, 32, 1),
        average_positions(events, players, AWAY_ID, 32, 2),
        dominating_zones(events),
        box_entries(events, HOME_ID, 34),
        box_entries(events, AWAY_ID, 35),
        high_regains(events, HOME_ID, 36),
        high_regains(events, AWAY_ID, 37),
        pass_targets(events, HOME_ID, 38),
        pass_targets(events, AWAY_ID, 39),
        ppda(events),
        transition_outcomes(events),
        generated[42],
        game_state(events, team_metrics),
        player_sequence(player_metrics),
    ]
    paths.extend(player_pizzas(events))
    paths = sorted({path.resolve() for path in paths}, key=lambda path: path.name)
    catalog = build_catalog(paths)
    pdf = build_pdf(paths, events, xg, team_metrics, player_metrics)
    from build_qa_contact_sheets import build_qa_contact_sheets
    qa_dashboards = build_qa_contact_sheets(OUT)
    print(f"Generated {len(paths)} full redesigned visuals")
    print(f"Pitch visuals: {int(catalog['has_pitch'].sum())}")
    print(f"QA dashboards: {len(qa_dashboards)}")
    print(f"PDF: {pdf}")


if __name__ == "__main__":
    main()
