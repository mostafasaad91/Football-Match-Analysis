from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from visualization_components import (
    C_AWAY,
    C_HOME,
    EVENT_HIGHLIGHT,
    IS_LIGHT_THEME,
    PITCH_LINE,
    PITCH_LINE_ALPHA,
)

from match_metrics import (
    advanced_metrics_frames,
    high_regain_events,
    progressive_pass_mask,
    touch_mask,
)


ROOT = Path(__file__).resolve().parent
MATCH_KEY = "France_vs_England_4-6"
DATA_DIR = ROOT / "sample_data" / MATCH_KEY
BASELINE_DIR = ROOT / "baseline_visuals"
OUT_DIR = ROOT / "output" / MATCH_KEY
COMPARE_DIR = OUT_DIR / "comparisons"

# Page chrome follows the active theme. The AMOLED values stay byte-identical
# so the dark package renders exactly as before; the light branch is the white
# paper counterpart selected via MATCH_ANALYSIS_THEME=light.
if IS_LIGHT_THEME:
    BG = "#F5F5F5"
    PANEL = "#FFFFFF"
    PANEL_2 = "#FCFCFC"
    TEXT = "#333333"
    MUTED = "#666666"
    GRID = "#CCCCCC"
    VALUE = "#333333"
    FOCUS = EVENT_HIGHLIGHT
    NEUTRAL = "#888888"
else:
    BG = "#000000"
    PANEL = "#050505"
    PANEL_2 = "#080808"
    TEXT = "#F7F7F5"
    MUTED = "#A3A3A3"
    GRID = "#242424"
    VALUE = "#FFFFFF"
    FOCUS = "#FFFFFF"
    NEUTRAL = "#666666"
# Stable role colours used by every fixture and every comparison visual.
HOME = C_HOME
AWAY = C_AWAY

HOME_NAME = "France"
AWAY_NAME = "England"
HOME_ID = 341
AWAY_ID = 345

TEAM_COLOR = {HOME_ID: HOME, AWAY_ID: AWAY}
TEAM_NAME = {HOME_ID: HOME_NAME, AWAY_ID: AWAY_NAME}
MATCH_SCORE = "4 — 6"


def _bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def credited_team(events: pd.DataFrame) -> pd.Series:
    """Return the team each goal counts FOR, not the team that struck it.

    An own goal is logged on the scorer's own ``team_id``; crediting it there
    puts the goal on the wrong side of the scoreline, the wrong end of the
    timeline and the wrong colour on the chart. The parser records the real
    beneficiary in ``scoring_team``; where that is missing, an ``is_own_goal``
    flag means the other team.
    """
    team = pd.to_numeric(events.get("team_id"), errors="coerce")
    scoring = pd.to_numeric(
        events.get("scoring_team", pd.Series(np.nan, index=events.index)),
        errors="coerce",
    )
    own = _bool(events.get("is_own_goal", pd.Series(False, index=events.index)))
    flipped = team.where(~own, np.where(team.eq(HOME_ID), AWAY_ID, HOME_ID))
    return scoring.fillna(flipped)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(DATA_DIR / "events.csv", encoding="utf-8-sig")
    players = pd.read_csv(DATA_DIR / "players.csv", encoding="utf-8-sig")
    xg = pd.read_csv(DATA_DIR / "xg.csv", encoding="utf-8-sig")

    numeric_cols = [
        "minute",
        "second",
        "team_id",
        "player_id",
        "x",
        "y",
        "end_x",
        "end_y",
        "xG",
        "xT",
    ]
    for col in numeric_cols:
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce")
    for col in ["team_id", "player_id", "touches", "passes"]:
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce")

    info = {
        "home_id": HOME_ID,
        "away_id": AWAY_ID,
        "home_name": HOME_NAME,
        "away_name": AWAY_NAME,
    }
    team_metrics, player_metrics = advanced_metrics_frames(events, info)
    return events, players, xg, team_metrics, player_metrics


def theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
        }
    )


def amoled_header(
    fig: plt.Figure,
    title: str,
    subtitle: str = "",
    *,
    section: str = "MATCH VISUAL",
    active_team: str | None = None,
) -> None:
    """One neutral pure-black score strip shared by every chart."""
    strip = FancyBboxPatch(
        (0.035, 0.865), 0.93, 0.115,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        transform=fig.transFigure, facecolor=PANEL_2, edgecolor=GRID,
        linewidth=1.0, zorder=90,
    )
    fig.add_artist(strip)
    fig.add_artist(Rectangle((0.035, 0.865), 0.465, 0.004,
                             transform=fig.transFigure, color=HOME,
                             linewidth=0, zorder=92))
    fig.add_artist(Rectangle((0.500, 0.865), 0.465, 0.004,
                             transform=fig.transFigure, color=AWAY,
                             linewidth=0, zorder=92))
    glow = [path_effects.withStroke(linewidth=3.5, foreground=BG)]
    fig.text(0.055, 0.955, "●", color=FOCUS, fontsize=8.5,
             va="center", zorder=95, path_effects=glow)
    fig.text(0.070, 0.955, section.upper(), color=MUTED, fontsize=7.2,
             fontweight="bold", va="center", zorder=95)
    fig.text(0.055, 0.909, title, color=TEXT, fontsize=17.5,
             fontweight="bold", va="center", zorder=95, path_effects=glow)
    fig.text(0.055, 0.881, "STAT INFO", color=FOCUS, fontsize=6.3,
             fontweight="bold", va="center", zorder=95)
    fig.text(0.113, 0.881, subtitle[:115], color=MUTED, fontsize=6.8,
             va="center", zorder=95)

    fig.text(0.705, 0.954, HOME_NAME.upper(), color=HOME, fontsize=8.2,
             fontweight="bold", ha="right", va="center", zorder=95)
    fig.text(0.785, 0.954, MATCH_SCORE, color=TEXT, fontsize=12,
             fontweight="bold", ha="center", va="center", zorder=95,
             path_effects=glow)
    fig.text(0.865, 0.954, AWAY_NAME.upper(), color=AWAY, fontsize=8.2,
             fontweight="bold", ha="left", va="center", zorder=95)
    context = "FULL MATCH"
    if active_team in {HOME_NAME, AWAY_NAME}:
        context = f"TEAM VIEW  ·  {active_team.upper()}"
    fig.text(0.705, 0.903, context, color=NEUTRAL, fontsize=6.7,
             fontweight="bold", ha="right", va="center", zorder=95)
    fig.text(0.945, 0.903, "CREATED BY MOSTAFA SAAD", color=NEUTRAL,
             fontsize=6.1, fontweight="bold", ha="right", va="center", zorder=95)
    fig._amoled_header_applied = True


def page(title: str, subtitle: str, figsize=(14, 8)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.08, right=0.94, top=0.78, bottom=0.12)
    active_team = HOME_NAME if HOME_NAME.lower() in title.lower() else (AWAY_NAME if AWAY_NAME.lower() in title.lower() else None)
    amoled_header(fig, title, subtitle, active_team=active_team)
    fig.text(0.94, 0.035, "VISUAL REDESIGN PREVIEW · REAL MATCH DATA", ha="right", fontsize=8, color=NEUTRAL)
    return fig, ax


def save(fig: plt.Figure, name: str) -> Path:
    path = OUT_DIR / name
    fig.savefig(path, dpi=155, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return path


def clean_ax(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def draw_pitch(ax: plt.Axes, line_color=None, lw=1.25, alpha=None) -> None:
    # Both themes now draw the shared pitch line. On AMOLED the old GRID value
    # (#242424) was almost invisible against the pure-black page; white held at
    # PITCH_LINE_ALPHA outlines the pitch without covering heatmaps drawn under
    # it. Call sites that used to pass their own slate grey get the same white.
    if line_color is None or str(line_color).upper() in ("#738090", "#8290A0", GRID.upper()):
        line_color = PITCH_LINE
    a = PITCH_LINE_ALPHA if alpha is None else alpha
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), 100, 100, fill=False, ec=line_color, lw=lw, alpha=a))
    ax.plot([50, 50], [0, 100], color=line_color, lw=lw, alpha=a)
    ax.add_patch(Circle((50, 50), 10, fill=False, ec=line_color, lw=lw, alpha=a))
    ax.add_patch(Rectangle((0, 21), 16.5, 58, fill=False, ec=line_color, lw=lw, alpha=a))
    ax.add_patch(Rectangle((83.5, 21), 16.5, 58, fill=False, ec=line_color, lw=lw, alpha=a))
    ax.add_patch(Rectangle((0, 36), 5.5, 28, fill=False, ec=line_color, lw=lw, alpha=a))
    ax.add_patch(Rectangle((94.5, 36), 5.5, 28, fill=False, ec=line_color, lw=lw, alpha=a))
    ax.add_patch(Arc((11, 50), 18, 24, theta1=305, theta2=55, ec=line_color, lw=lw, alpha=a))
    ax.add_patch(Arc((89, 50), 18, 24, theta1=125, theta2=235, ec=line_color, lw=lw, alpha=a))
    ax.axis("off")


def row_dot_plot(
    ax: plt.Axes,
    rows: list[tuple[str, float, float, str]],
    title: str | None = None,
) -> None:
    """Opta-style bilateral comparison: home team | metric | away team."""
    row_count = max(len(rows), 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, row_count + 0.72)
    clean_ax(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(BG)
    if title:
        ax.set_title(title, loc="left", fontsize=11.5, color=MUTED, pad=15, fontweight="bold")

    axis_width = ax.get_position().width * ax.figure.get_figwidth()
    label_size = 8.6 if axis_width >= 5.0 else 7.4
    value_size = 9.5 if axis_width >= 5.0 else 8.2
    header_size = 7.4 if axis_width >= 5.0 else 6.7
    ax.text(0.125, row_count + 0.43, f"●  {HOME_NAME.upper()}", color=HOME,
            fontsize=header_size, fontweight="bold", ha="center", va="center")
    ax.text(0.500, row_count + 0.43, "MATCH COMPARISON", color=NEUTRAL,
            fontsize=header_size - 0.2, fontweight="bold", ha="center", va="center")
    ax.text(0.875, row_count + 0.43, f"◆  {AWAY_NAME.upper()}", color=AWAY,
            fontsize=header_size, fontweight="bold", ha="center", va="center")

    for idx, (label, home, away, fmt) in enumerate(rows):
        y = row_count - idx - 0.58
        height = 0.67
        outer = FancyBboxPatch(
            (0.012, y - height / 2), 0.976, height,
            boxstyle="round,pad=0.008,rounding_size=0.085",
            facecolor=PANEL_2, edgecolor=GRID, linewidth=0.8, zorder=1,
        )
        ax.add_patch(outer)
        # Light theme: value cells take the team colour at full strength so each
        # side is unmistakably owned (black home / orange away) — a 0.17 wash on
        # light paper collapses into indistinct grey/pink. Dark keeps the wash.
        cell_alpha = 1.0 if IS_LIGHT_THEME else 0.17
        value_color = "#FFFFFF" if IS_LIGHT_THEME else TEXT
        ax.add_patch(Rectangle((0.013, y - height / 2 + 0.01), 0.224, height - 0.02,
                               facecolor=HOME, edgecolor="none", alpha=cell_alpha, zorder=2))
        ax.add_patch(Rectangle((0.763, y - height / 2 + 0.01), 0.224, height - 0.02,
                               facecolor=AWAY, edgecolor="none", alpha=cell_alpha, zorder=2))
        ax.plot([0.237, 0.237], [y - height / 2 + 0.05, y + height / 2 - 0.05],
                color=HOME, lw=0.75, alpha=0.65, zorder=3)
        ax.plot([0.763, 0.763], [y - height / 2 + 0.05, y + height / 2 - 0.05],
                color=AWAY, lw=0.75, alpha=0.65, zorder=3)
        ax.text(0.125, y, fmt.format(home), color=value_color, fontsize=value_size,
                fontweight="bold", ha="center", va="center", zorder=4)
        ax.text(0.500, y, label, color=TEXT, fontsize=label_size,
                fontweight="bold", ha="center", va="center", zorder=4)
        ax.text(0.875, y, fmt.format(away), color=value_color, fontsize=value_size,
                fontweight="bold", ha="center", va="center", zorder=4)


def metric_lookup(team_metrics: pd.DataFrame, side: str, name: str, default=0.0) -> float:
    row = team_metrics[team_metrics["side"] == side]
    if row.empty or name not in row.columns:
        return float(default)
    value = pd.to_numeric(row.iloc[0][name], errors="coerce")
    return float(default if pd.isna(value) else value)


def shot_profile(xg: pd.DataFrame, events: pd.DataFrame) -> Path:
    def xg_value(team: str, col: str) -> float:
        row = xg[xg["team"].astype(str).str.lower() == team.lower()]
        return float(pd.to_numeric(row.iloc[0][col], errors="coerce")) if not row.empty else 0.0

    big = _bool(events.get("big_chance", pd.Series(False, index=events.index)))
    is_shot = _bool(events.get("is_shot", pd.Series(False, index=events.index)))
    pso = _bool(events.get("is_penalty_shootout", pd.Series(False, index=events.index)))
    rows = [
        ("Shots", xg_value(HOME_NAME, "shots"), xg_value(AWAY_NAME, "shots"), "{:.0f}"),
        ("Shots on target", xg_value(HOME_NAME, "on_target"), xg_value(AWAY_NAME, "on_target"), "{:.0f}"),
        (
            "Big chances",
            float((big & is_shot & ~pso & events["team_id"].eq(HOME_ID)).sum()),
            float((big & is_shot & ~pso & events["team_id"].eq(AWAY_ID)).sum()),
            "{:.0f}",
        ),
        ("xG", xg_value(HOME_NAME, "xG"), xg_value(AWAY_NAME, "xG"), "{:.2f}"),
        ("xG on target", xg_value(HOME_NAME, "xGoT"), xg_value(AWAY_NAME, "xGoT"), "{:.2f}"),
    ]
    fig, ax = page(
        "Shot Profile",
        "Opta-style bilateral comparison · exact values from full-match shots",
    )
    row_dot_plot(ax, rows)
    fig.text(0.08, 0.075, f"● {HOME_NAME}", color=HOME, fontsize=10, fontweight="bold")
    fig.text(0.18, 0.075, f"◆ {AWAY_NAME}", color=AWAY, fontsize=10, fontweight="bold")
    fig.text(0.94, 0.075, "VALUES LEFT · METRIC CENTRE · VALUES RIGHT", color=MUTED, ha="right", fontsize=8.5)
    return save(fig, "01_shot_profile_redesign.png")


def advanced_metrics(team_metrics: pd.DataFrame) -> Path:
    groups = [
        (
            "VOLUME",
            [
                ("Transitions", "transitions", "{:.0f}"),
                ("Deep completions", "deep_completions", "{:.0f}"),
                ("Final-third entries", "final_third_entries", "{:.0f}"),
                ("Box entries", "box_entries", "{:.0f}"),
            ],
        ),
        (
            "EFFICIENCY",
            [
                ("Build-up success", "build_up_success_rate", "{:.1f}%"),
                ("Entry efficiency", "final_third_entry_efficiency", "{:.1f}%"),
                ("Box entry → shot", "box_entry_to_shot_rate", "{:.1f}%"),
                ("Counterpress success", "counterpress_success_rate", "{:.1f}%"),
            ],
        ),
        (
            "VALUE",
            [
                ("Sequence xT", "sequence_xT", "{:.2f}"),
                ("Transition xT", "transition_xT", "{:.2f}"),
                ("Directness", "directness", "{:.1f}%"),
            ],
        ),
        (
            "RISK · LOWER IS BETTER",
            [
                ("Rest-defence vulnerability", "rest_defence_vulnerability", "{:.1f}%"),
                ("Dangerous counters", "rest_defence_dangerous_counters", "{:.0f}"),
                ("Transition exposure", "rest_defence_exposures", "{:.0f}"),
            ],
        ),
    ]
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    amoled_header(
        fig, "Advanced Team Metrics",
        "Volume, efficiency, value and risk separated into four exact-value comparisons",
    )
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.09, right=0.95, top=0.785, bottom=0.10, hspace=0.46, wspace=0.46)
    for ax, (group_name, specs) in zip(axes.flat, groups):
        rows = [
            (
                label,
                metric_lookup(team_metrics, "home", key),
                metric_lookup(team_metrics, "away", key),
                fmt,
            )
            for label, key, fmt in specs
        ]
        row_dot_plot(ax, rows, group_name)
    fig.text(0.06, 0.035, f"● {HOME_NAME}", color=HOME, fontsize=10, fontweight="bold")
    fig.text(0.16, 0.035, f"◆ {AWAY_NAME}", color=AWAY, fontsize=10, fontweight="bold")
    fig.text(0.94, 0.035, "REAL MATCH EVENTS · CANONICAL MATCH_METRICS", ha="right", fontsize=8, color=NEUTRAL)
    return save(fig, "02_advanced_metrics_redesign.png")


def xg_flow(events: pd.DataFrame) -> Path:
    work = events.copy()
    live = ~_bool(work.get("is_penalty_shootout", pd.Series(False, index=work.index)))
    shots = work[live & _bool(work.get("is_shot", pd.Series(False, index=work.index)))].copy()
    shots["xG"] = pd.to_numeric(shots["xG"], errors="coerce").fillna(0).clip(lower=0)
    shots["minute"] = pd.to_numeric(shots["minute"], errors="coerce").fillna(0)
    max_minute = max(90, int(shots["minute"].max()) + 3 if not shots.empty else 90)
    totals = shots.groupby("team_id")["xG"].sum()
    shot_counts = shots.groupby("team_id").size()
    home_total = float(totals.get(HOME_ID, 0.0))
    away_total = float(totals.get(AWAY_ID, 0.0))
    match_goals = shots[
        _bool(shots.get("is_goal", pd.Series(False, index=shots.index)))
    ].sort_values(["minute", "second"]).copy()
    # Own goals count for the opponent, so the scoreline, the match-state band
    # and the timeline all read from the credited team rather than team_id.
    match_goals["_credited_team"] = credited_team(match_goals)
    home_goals = int(match_goals["_credited_team"].eq(HOME_ID).sum())
    away_goals = int(match_goals["_credited_team"].eq(AWAY_ID).sum())

    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    header = FancyBboxPatch(
        (0.025, 0.855),
        0.95,
        0.125,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor=PANEL,
        edgecolor=GRID,
        linewidth=1.0,
        zorder=90,
    )
    fig.add_artist(header)
    fig.add_artist(
        Rectangle(
            (0.025, 0.855),
            0.475,
            0.004,
            transform=fig.transFigure,
            color=HOME,
            linewidth=0,
            zorder=92,
        )
    )
    fig.add_artist(
        Rectangle(
            (0.500, 0.855),
            0.475,
            0.004,
            transform=fig.transFigure,
            color=AWAY,
            linewidth=0,
            zorder=92,
        )
    )
    glow_text = [path_effects.withStroke(linewidth=3.5, foreground=BG)]
    fig.text(
        0.045,
        0.950,
        "●  XG FLOW",
        color=TEXT,
        fontsize=9.2,
        fontweight="bold",
        va="center",
        zorder=95,
    )
    fig.text(
        0.045,
        0.910,
        "CUMULATIVE EXPECTED GOALS",
        color=TEXT,
        fontsize=15.5,
        fontweight="bold",
        va="center",
        zorder=95,
        path_effects=glow_text,
    )
    fig.text(
        0.045,
        0.878,
        "STAT INFO",
        color=FOCUS,
        fontsize=6.3,
        fontweight="bold",
        va="center",
        zorder=95,
    )
    fig.text(
        0.102,
        0.878,
        "Chance quality accumulated after every shot · shootout excluded",
        color=MUTED,
        fontsize=7.0,
        va="center",
        zorder=95,
    )

    fig.text(0.385, 0.952, HOME_NAME.upper(), color=HOME, fontsize=7.2, fontweight="bold", ha="center", va="center", zorder=95)
    fig.text(0.385, 0.913, f"{home_total:.2f}", color=TEXT, fontsize=18, fontweight="bold", ha="center", va="center", zorder=95, path_effects=glow_text)
    fig.text(0.385, 0.882, f"xG  ·  {int(shot_counts.get(HOME_ID, 0))} SHOTS", color=MUTED, fontsize=6.4, ha="center", va="center", zorder=95)
    fig.text(0.520, 0.928, f"{home_goals} — {away_goals}", color=TEXT, fontsize=19, fontweight="bold", ha="center", va="center", zorder=95, path_effects=glow_text)
    fig.text(0.520, 0.888, "FULL TIME", color=MUTED, fontsize=6.2, fontweight="bold", ha="center", va="center", zorder=95)
    fig.text(0.655, 0.952, AWAY_NAME.upper(), color=AWAY, fontsize=7.2, fontweight="bold", ha="center", va="center", zorder=95)
    fig.text(0.655, 0.913, f"{away_total:.2f}", color=TEXT, fontsize=18, fontweight="bold", ha="center", va="center", zorder=95, path_effects=glow_text)
    fig.text(0.655, 0.882, f"xG  ·  {int(shot_counts.get(AWAY_ID, 0))} SHOTS", color=MUTED, fontsize=6.4, ha="center", va="center", zorder=95)
    leader_name = HOME_NAME if home_total >= away_total else AWAY_NAME
    edge = abs(home_total - away_total)
    edge_color = HOME if home_total >= away_total else AWAY
    badge = FancyBboxPatch(
        (0.800, 0.885),
        0.145,
        0.060,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        transform=fig.transFigure,
        facecolor=PANEL_2 if IS_LIGHT_THEME else "#0B0B0B",
        edgecolor=edge_color,
        linewidth=0.9,
        zorder=94,
    )
    fig.add_artist(badge)
    fig.text(0.8725, 0.925, "xG EDGE", color=MUTED, fontsize=6.0, fontweight="bold", ha="center", va="center", zorder=95)
    fig.text(0.8725, 0.901, f"{leader_name.upper()}  +{edge:.2f}", color=edge_color, fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=95)

    ax = fig.add_axes([0.07, 0.315, 0.82, 0.495])
    clean_ax(ax)
    ymax = max(0.5, max(home_total, away_total) * 1.16)
    ax.set_xlim(0, max_minute + 12)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=GRID, lw=0.75, alpha=0.85)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("MATCH MINUTE", fontsize=7, fontweight="bold", labelpad=8)
    ax.set_ylabel("CUMULATIVE xG", fontsize=7, fontweight="bold", labelpad=8)
    ax.text(0.0, 1.035, "CUMULATIVE xG", transform=ax.transAxes, color=MUTED, fontsize=6.5, fontweight="bold")
    peak = float(shots["xG"].max()) if not shots.empty else 0.0
    ax.text(0.56, 1.035, f"PEAK CHANCE  ·  {peak:.2f} xG", transform=ax.transAxes, color=FOCUS, fontsize=6.5, fontweight="bold", ha="center")
    ax.text(1.0, 1.035, "DOT SIZE = SHOT xG", transform=ax.transAxes, color=MUTED, fontsize=6.2, ha="right")
    for team_id, marker in [(HOME_ID, "o"), (AWAY_ID, "s")]:
        team = shots[shots["team_id"].eq(team_id)].sort_values(["minute", "second"])
        minutes = [0.0] + team["minute"].tolist() + [max_minute]
        cumulative = [0.0] + team["xG"].cumsum().tolist()
        cumulative += [cumulative[-1] if cumulative else 0.0]
        color = TEAM_COLOR[team_id]
        ax.step(minutes, cumulative, where="post", color=color, lw=9, alpha=0.045, solid_capstyle="round", zorder=2)
        ax.step(minutes, cumulative, where="post", color=color, lw=5.5, alpha=0.11, solid_capstyle="round", zorder=2)
        ax.step(minutes, cumulative, where="post", color=color, lw=2.8, solid_capstyle="round", zorder=3)
        shot_y = team["xG"].cumsum()
        sizes = 20 + team["xG"].to_numpy() * 72
        ax.scatter(team["minute"], shot_y, s=sizes, marker=marker, facecolor=BG, edgecolor=color, linewidth=1.5, zorder=4)
        total = cumulative[-1]
        ax.text(max_minute + 1.4, total, f"{TEAM_NAME[team_id].upper()}  ·  {total:.2f}", color=TEXT, va="center", fontsize=8.5, fontweight="bold")
        # Only goals the team actually scored belong on its own xG curve: an
        # own goal is struck by this team but counts for the opponent.
        team_goals = team[
            _bool(team.get("is_goal", pd.Series(False, index=team.index)))
            & credited_team(team).eq(team_id)
        ]
        for goal_idx, (_, goal) in enumerate(team_goals.iterrows()):
            upto = team[team["minute"].le(goal["minute"])]["xG"].sum()
            ax.scatter(goal["minute"], upto, s=130, facecolor=BG, edgecolor=FOCUS, linewidth=2.4, zorder=6)
            ax.scatter(goal["minute"], upto, s=40, facecolor=color, edgecolor=BG, linewidth=0.8, zorder=7)
            # Name the scorer on the marker — a minute alone makes the reader
            # cross-reference the timeline strip below to learn who scored.
            surname = str(goal.get("player") or "").split()[-1][:12]
            label = f"{surname} {int(goal['minute'])}′" if surname else f"{int(goal['minute'])}′"
            offset = 13 if goal_idx % 2 == 0 else 22
            ax.annotate(
                label, (goal["minute"], upto), xytext=(0, offset),
                textcoords="offset points", ha="center", color=TEXT, fontsize=6.4,
                fontweight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=BG, edgecolor=color,
                          linewidth=0.7, alpha=0.92),
            )
    ax.axvline(45, color="#3A3A3A", lw=0.9, ls=(0, (3, 4)))
    ax.text(45, ymax * 0.975, "HT", color=MUTED, fontsize=6.2, ha="center", va="top", bbox=dict(boxstyle="round,pad=0.25", fc=PANEL, ec=GRID, lw=0.6))

    state_ax = fig.add_axes([0.07, 0.205, 0.82, 0.055])
    state_ax.set_xlim(0, max_minute)
    state_ax.set_ylim(0, 1)
    state_ax.axis("off")
    state_ax.text(0, 1.16, "MATCH STATE", color=MUTED, fontsize=6.3, fontweight="bold", va="bottom")
    home_score = away_score = 0
    start = 0.0
    for _, goal in match_goals.iterrows():
        end = float(goal["minute"])
        leader_color = HOME if home_score > away_score else (AWAY if away_score > home_score else "#444444")
        state_ax.add_patch(Rectangle((start, 0.15), max(end - start, 0.3), 0.55, facecolor=leader_color, edgecolor=GRID, lw=0.65, alpha=0.22))
        if end - start >= 5:
            state_ax.text((start + end) / 2, 0.425, f"{home_score}–{away_score}", color=TEXT, fontsize=5.8, fontweight="bold", ha="center", va="center")
        if int(goal["_credited_team"]) == HOME_ID:
            home_score += 1
        else:
            away_score += 1
        start = end
    leader_color = HOME if home_score > away_score else (AWAY if away_score > home_score else "#444444")
    state_ax.add_patch(Rectangle((start, 0.15), max(max_minute - start, 0.3), 0.55, facecolor=leader_color, edgecolor=GRID, lw=0.65, alpha=0.22))
    if max_minute - start >= 4:
        state_ax.text((start + max_minute) / 2, 0.425, f"{home_score}–{away_score}", color=TEXT, fontsize=5.8, fontweight="bold", ha="center", va="center")

    goals_ax = fig.add_axes([0.07, 0.065, 0.82, 0.095])
    goals_ax.set_xlim(0, max_minute)
    goals_ax.set_ylim(0, 1)
    goals_ax.axis("off")
    goals_ax.text(0, 1.02, "GOALS", color=MUTED, fontsize=6.3, fontweight="bold", va="bottom")
    goals_ax.plot([0, max_minute], [0.50, 0.50], color=GRID, lw=0.7)
    for idx, (_, goal) in enumerate(match_goals.iterrows()):
        minute = float(goal["minute"])
        color = TEAM_COLOR.get(int(goal["_credited_team"]), TEXT)
        y = 0.78 if idx % 2 == 0 else 0.22
        goals_ax.plot([minute, minute], [0.50, y], color=color, lw=0.65, alpha=0.8)
        goals_ax.scatter([minute], [0.50], s=36, facecolor=color, edgecolor=BG, linewidth=0.7, zorder=3)
        surname = str(goal.get("player") or "Goal").split()[-1][:10].upper()
        own = " (OG)" if _bool(pd.Series([goal.get("is_own_goal", False)])).iloc[0] else ""
        goals_ax.text(minute, y, f"{int(minute)}′  {surname}{own}", color=TEXT, fontsize=5.4, fontweight="bold", ha="center", va="center")
    fig.text(0.945, 0.030, "PURE BLACK MATCH INTELLIGENCE · REAL EVENT DATA", ha="right", fontsize=6.5, color=NEUTRAL)
    fig._amoled_header_applied = True
    return save(fig, "03_xg_flow_redesign.png")


def touches(team_metrics: pd.DataFrame) -> Path:
    home_touches = metric_lookup(team_metrics, "home", "touches")
    away_touches = metric_lookup(team_metrics, "away", "touches")
    total = max(home_touches + away_touches, 1)
    home_share = 100 * home_touches / total
    away_share = 100 * away_touches / total
    rows = [
        ("Total touches", home_touches, away_touches, "{:.0f}"),
        ("Share of match touches", home_share, away_share, "{:.1f}%"),
        ("Defensive-third share", metric_lookup(team_metrics, "home", "touch_def_pct"), metric_lookup(team_metrics, "away", "touch_def_pct"), "{:.0f}%"),
        ("Middle-third share", metric_lookup(team_metrics, "home", "touch_mid_pct"), metric_lookup(team_metrics, "away", "touch_mid_pct"), "{:.0f}%"),
        ("Final-third share", metric_lookup(team_metrics, "home", "touch_att_pct"), metric_lookup(team_metrics, "away", "touch_att_pct"), "{:.0f}%"),
    ]
    fig, ax = page("Ball Touches", "Opta-style bilateral read · third shares use each team's own touches as denominator")
    row_dot_plot(ax, rows)
    return save(fig, "04_ball_touches_redesign.png")


def xt_per_minute(events: pd.DataFrame) -> Path:
    work = events.copy()
    work["minute"] = pd.to_numeric(work["minute"], errors="coerce").fillna(0).astype(int)
    work["xT"] = pd.to_numeric(work["xT"], errors="coerce").fillna(0).clip(lower=0)
    work = work[~_bool(work.get("is_penalty_shootout", pd.Series(False, index=work.index)))]
    max_min = max(90, int(work["minute"].max()) + 1)
    minutes = np.arange(0, max_min + 1)
    home = work[work["team_id"].eq(HOME_ID)].groupby("minute")["xT"].sum().reindex(minutes, fill_value=0)
    away = work[work["team_id"].eq(AWAY_ID)].groupby("minute")["xT"].sum().reindex(minutes, fill_value=0)
    home_roll = home.rolling(5, min_periods=1).mean()
    away_roll = away.rolling(5, min_periods=1).mean()
    fig, ax = page("xT per Minute", f"{HOME_NAME} above zero; {AWAY_NAME} mirrored below zero for comparison—not negative xT")
    clean_ax(ax)
    ax.axhline(0, color=GRID, lw=1.4)
    ax.bar(minutes, home.values, width=0.82, color=HOME, alpha=0.38)
    ax.bar(minutes, -away.values, width=0.82, color=AWAY, alpha=0.38)
    ax.plot(minutes, home_roll, color=HOME, lw=2.8)
    ax.plot(minutes, -away_roll, color=AWAY, lw=2.8, ls=(0, (7, 4)))
    max_abs = max(float(home.max()), float(away.max()), 0.02)
    ax.set_ylim(-max_abs * 1.35, max_abs * 1.35)
    ax.set_xlim(0, max_min + 18)
    ax.set_xlabel("Match minute")
    ax.set_yticks([])
    ax.text(max_min + 1, float(home_roll.iloc[-1]), f"{HOME_NAME} · 5-min rolling", color=HOME, va="center", fontweight="bold")
    ax.text(max_min + 1, -float(away_roll.iloc[-1]), f"{AWAY_NAME} · mirrored rolling", color=AWAY, va="center", fontweight="bold")
    ax.grid(axis="x", color=GRID, alpha=0.5, lw=0.7)
    return save(fig, "05_xt_per_minute_redesign.png")


def xt_map(events: pd.DataFrame, team_id=HOME_ID) -> Path:
    team = events[events["team_id"].eq(team_id)].copy()
    passes = team[team["type"].astype(str).eq("Pass")].copy()
    passes["xT"] = pd.to_numeric(passes["xT"], errors="coerce").fillna(0).clip(lower=0)
    valid = passes.dropna(subset=["x", "y", "end_x", "end_y"])
    heat, xedges, yedges = np.histogram2d(valid["x"], valid["y"], bins=[12, 8], range=[[0, 100], [0, 100]], weights=valid["xT"])
    cmap = LinearSegmentedColormap.from_list("xt", [BG, PANEL_2, VALUE])
    fig, ax = page(f"xT Map · {TEAM_NAME[team_id]}", "Sequential xT scale with only the five highest-value passes highlighted")
    image = ax.imshow(heat.T, extent=[0, 100, 0, 100], origin="lower", cmap=cmap, aspect="equal", alpha=0.96)
    draw_pitch(ax, line_color="#738090", lw=1.05)
    top = valid.nlargest(5, "xT")
    for _, row in top.iterrows():
        ax.annotate("", xy=(row["end_x"], row["end_y"]), xytext=(row["x"], row["y"]), arrowprops=dict(arrowstyle="-|>", color=FOCUS, lw=2.5, mutation_scale=14, connectionstyle="arc3,rad=0.03"))
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    cbar.outline.set_edgecolor(GRID)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.set_label("Accumulated xT from action origins", color=MUTED, fontsize=9)
    fig.text(0.08, 0.075, "Gold arrows = top five passes by xT added", color=FOCUS, fontsize=9)
    return save(fig, "06_xt_map_france_redesign.png")


def progressive_map(events: pd.DataFrame, team_id=HOME_ID) -> Path:
    mask = progressive_pass_mask(events)
    prog = events[mask & events["team_id"].eq(team_id)].copy().dropna(subset=["x", "y", "end_x", "end_y"])
    prog["xT"] = pd.to_numeric(prog["xT"], errors="coerce").fillna(0)
    fig, ax = page(
        f"Progressive Passes · {TEAM_NAME[team_id]}",
        "Canonical zone-aware distance thresholds · strongest actions highlighted by xT added",
    )
    draw_pitch(ax)
    for _, row in prog.iterrows():
        ax.annotate("", xy=(row["end_x"], row["end_y"]), xytext=(row["x"], row["y"]), arrowprops=dict(arrowstyle="-|>", color=VALUE, alpha=0.20, lw=0.8, mutation_scale=8))
    for _, row in prog.nlargest(10, "xT").iterrows():
        ax.annotate("", xy=(row["end_x"], row["end_y"]), xytext=(row["x"], row["y"]), arrowprops=dict(arrowstyle="-|>", color=FOCUS, alpha=0.95, lw=2.0, mutation_scale=12))
    legend = [
        Line2D([0], [0], color=VALUE, alpha=0.45, lw=2, label=f"All progressive passes ({len(prog)})"),
        Line2D([0], [0], color=FOCUS, lw=2.5, label="Top 10 by xT added"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False, labelcolor=TEXT, fontsize=9)
    return save(fig, "07_progressive_france_redesign.png")


def defensive_activity(events: pd.DataFrame, team_id=HOME_ID) -> Path:
    actions = events[
        events["team_id"].eq(team_id)
        & events["type"].astype(str).isin(["Tackle", "Interception", "BallRecovery", "Clearance", "BlockedShot", "Foul"])
    ].dropna(subset=["x", "y"]).copy()
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.text(0.06, 0.93, f"Defensive Activity · {TEAM_NAME[team_id]}", fontsize=23, fontweight="bold", color=TEXT)
    fig.text(0.06, 0.885, "Density and action type are separated so the heat layer never competes with event colours", fontsize=11, color=MUTED)
    fig.add_artist(Line2D([0.06, 0.94], [0.84, 0.84], transform=fig.transFigure, color=GRID, lw=1))
    left, right = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.05, right=0.96, top=0.79, bottom=0.13, wspace=0.08)
    cmap = LinearSegmentedColormap.from_list("def", [BG, PANEL_2, VALUE])
    heat, _, _ = np.histogram2d(actions["x"], actions["y"], bins=[12, 8], range=[[0, 100], [0, 100]])
    left.imshow(heat.T, extent=[0, 100, 0, 100], origin="lower", cmap=cmap, aspect="equal", alpha=0.95)
    draw_pitch(left, line_color="#738090", lw=1.0)
    left.set_title("DEFENSIVE DENSITY · ONE SCALE", loc="left", color=MUTED, fontsize=10, fontweight="bold")
    draw_pitch(right)
    marker_map = {"Tackle": "o", "Interception": "D", "BallRecovery": "s", "Clearance": "^", "BlockedShot": "P", "Foul": "X"}
    for event_type, marker in marker_map.items():
        subset = actions[actions["type"].astype(str).eq(event_type)]
        if not subset.empty:
            right.scatter(subset["x"], subset["y"], s=54, marker=marker, color=VALUE, edgecolor=BG, linewidth=0.8, label=f"{event_type} ({len(subset)})")
    right.set_title("ACTION TYPES · SHAPES, NOT SIX COLOURS", loc="left", color=MUTED, fontsize=10, fontweight="bold")
    right.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, labelcolor=TEXT, fontsize=8)
    fig.text(0.94, 0.035, "VISUAL REDESIGN PREVIEW · REAL MATCH DATA", ha="right", fontsize=8, color=NEUTRAL)
    return save(fig, "08_defensive_activity_france_redesign.png")


def dominance(events: pd.DataFrame) -> Path:
    tmask = touch_mask(events)
    home = events[tmask & events["team_id"].eq(HOME_ID)].dropna(subset=["x", "y"])
    away = events[tmask & events["team_id"].eq(AWAY_ID)].dropna(subset=["x", "y"])
    hh, _, _ = np.histogram2d(home["x"], home["y"], bins=[5, 3], range=[[0, 100], [0, 100]])
    ah, _, _ = np.histogram2d(away["x"], away["y"], bins=[5, 3], range=[[0, 100], [0, 100]])
    total = hh + ah
    share_diff = np.divide(hh - ah, total, out=np.zeros_like(total), where=total > 0)
    cmap = LinearSegmentedColormap.from_list("dom", [AWAY, PANEL_2, HOME])
    fig, ax = page("Dominating Zones", "Touch-share difference by zone · diverging scale centred on an even 50/50 split")
    image = ax.imshow(share_diff.T, extent=[0, 100, 0, 100], origin="lower", cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    draw_pitch(ax, line_color="#8290A0", lw=1.1)
    for ix in range(5):
        for iy in range(3):
            diff = int(hh[ix, iy] - ah[ix, iy])
            ax.text(ix * 20 + 10, iy * (100 / 3) + 100 / 6, f"{diff:+d}", ha="center", va="center", color=TEXT, fontsize=10, fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.03, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels([AWAY_NAME, "Balanced", HOME_NAME])
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(GRID)
    fig.text(0.08, 0.075, f"Cell labels = {HOME_NAME} touches minus {AWAY_NAME} touches", color=MUTED, fontsize=9)
    return save(fig, "09_dominating_zones_redesign.png")


def _pass_network_data(events: pd.DataFrame, players: pd.DataFrame, team_id: int):
    work = events.sort_values(["period", "minute", "second", "event_id"], kind="stable").copy()
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
    touches = events[touch_mask(events) & events["team_id"].eq(team_id)].dropna(subset=["player", "x", "y"])
    positions = touches.groupby("player").agg(x=("x", "mean"), y=("y", "mean"), touches=("event_id", "count")).sort_values("touches", ascending=False).head(11)
    names = set(positions.index.astype(str))
    passes = passes[passes["player"].astype(str).isin(names) & passes["next_player"].astype(str).isin(names)]
    edges = passes.groupby(["player", "next_player"]).size().reset_index(name="passes").sort_values("passes", ascending=False).head(18)
    starter_lookup = players[players["team_id"].eq(team_id)].set_index("name")["is_first_xi"].to_dict()
    return positions, edges, starter_lookup


def pass_network(events: pd.DataFrame, players: pd.DataFrame, team_id=HOME_ID) -> Path:
    positions, edges, starters = _pass_network_data(events, players, team_id)
    fig, ax = page(f"Pass Network · {TEAM_NAME[team_id]}", "Strongest 18 links only · node size = touches · line width = pass volume")
    draw_pitch(ax)
    max_edge = max(float(edges["passes"].max()) if not edges.empty else 1, 1)
    for _, edge in edges.iterrows():
        if edge["player"] not in positions.index or edge["next_player"] not in positions.index:
            continue
        a, b = positions.loc[edge["player"]], positions.loc[edge["next_player"]]
        width = 0.8 + 5.2 * float(edge["passes"]) / max_edge
        ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color=HOME, lw=width, alpha=0.36, zorder=2)
    max_touch = max(float(positions["touches"].max()) if not positions.empty else 1, 1)
    for name, row in positions.iterrows():
        size = 130 + 560 * float(row["touches"]) / max_touch
        starter = str(starters.get(name, "True")).lower() in {"true", "1", "yes"}
        marker = "o" if starter else "s"
        ax.scatter(row["x"], row["y"], s=size, marker=marker, color=HOME, edgecolor=TEXT, linewidth=1.3, zorder=4)
        short = str(name).split()[-1]
        xoff = 8 if row["x"] < 80 else -8
        ha = "left" if xoff > 0 else "right"
        ax.annotate(short, (row["x"], row["y"]), xytext=(xoff, 9), textcoords="offset points", ha=ha, va="bottom", color=TEXT, fontsize=8, arrowprops=dict(arrowstyle="-", color=GRID, lw=0.7), zorder=5)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HOME, markeredgecolor=TEXT, markersize=8, label="Starter"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=HOME, markeredgecolor=TEXT, markersize=8, label="Substitute"),
        Line2D([0], [0], color=HOME, lw=2, alpha=0.5, label="Fewer passes"),
        Line2D([0], [0], color=HOME, lw=6, alpha=0.5, label="More passes"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, labelcolor=TEXT, fontsize=8)
    return save(fig, "10_pass_network_france_redesign.png")


def average_positions(events: pd.DataFrame, players: pd.DataFrame, team_id=HOME_ID) -> Path:
    positions, _edges, starters = _pass_network_data(events, players, team_id)
    fig, ax = page(f"Average Positions · {TEAM_NAME[team_id]}", "No unexplained links · offset labels with short leaders · node size = touches")
    draw_pitch(ax)
    max_touch = max(float(positions["touches"].max()) if not positions.empty else 1, 1)
    offsets = [(8, 10), (-8, 12), (10, -14), (-10, -14)]
    for idx, (name, row) in enumerate(positions.iterrows()):
        size = 120 + 600 * float(row["touches"]) / max_touch
        starter = str(starters.get(name, "True")).lower() in {"true", "1", "yes"}
        marker = "o" if starter else "s"
        ax.scatter(row["x"], row["y"], s=size, marker=marker, color=HOME, edgecolor=TEXT, linewidth=1.3, zorder=4)
        dx, dy = offsets[idx % len(offsets)]
        ha = "left" if dx > 0 else "right"
        ax.annotate(str(name).split()[-1], (row["x"], row["y"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", color=TEXT, fontsize=8, arrowprops=dict(arrowstyle="-", color=GRID, lw=0.7))
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HOME, markeredgecolor=TEXT, markersize=8, label="Starter"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=HOME, markeredgecolor=TEXT, markersize=8, label="Substitute"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HOME, markersize=5, label="Fewer touches"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HOME, markersize=11, label="More touches"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, labelcolor=TEXT, fontsize=8)
    return save(fig, "11_average_positions_france_redesign.png")


def high_regains(events: pd.DataFrame, team_id=HOME_ID) -> Path:
    regains = high_regain_events(events, team_id).dropna(subset=["x", "y"]).copy()
    fig, ax = page(f"High Regains · {TEAM_NAME[team_id]}", "Open-play possession regains at x ≥ 60 · event type encoded by shape")
    draw_pitch(ax)
    ax.axvspan(60, 100, color=FOCUS, alpha=0.055)
    ax.axvline(60, color=FOCUS, lw=1.2, ls=(0, (5, 4)))
    marker_map = {"Tackle": "o", "Interception": "D", "BallRecovery": "s"}
    for event_type, marker in marker_map.items():
        subset = regains[regains["type"].astype(str).eq(event_type)]
        if not subset.empty:
            ax.scatter(subset["x"], subset["y"], marker=marker, s=85, color=VALUE, edgecolor=TEXT, linewidth=0.9, label=f"{event_type} ({len(subset)})", zorder=4)
    other = regains[~regains["type"].astype(str).isin(marker_map)]
    if not other.empty:
        ax.scatter(other["x"], other["y"], marker="^", s=85, color=VALUE, edgecolor=TEXT, linewidth=0.9, label=f"Other controlled regain ({len(other)})", zorder=4)
    ax.text(61, 96, "High-regain zone", color=FOCUS, fontsize=9, va="top")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, labelcolor=TEXT, fontsize=8)
    return save(fig, "12_high_regains_france_redesign.png")


def summary_board(events: pd.DataFrame, team_metrics: pd.DataFrame, xg: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.text(0.055, 0.94, "Match Analysis Snapshot", fontsize=24, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.898, f"{HOME_NAME} vs {AWAY_NAME} · rebuilt for summary-board size", fontsize=11, color=MUTED)
    fig.add_artist(Line2D([0.055, 0.945], [0.86, 0.86], transform=fig.transFigure, color=GRID, lw=1))
    gs = fig.add_gridspec(3, 4, left=0.055, right=0.945, top=0.82, bottom=0.09, hspace=0.5, wspace=0.38)
    kpis = [
        ("TRANSITIONS", "transitions", "{:.0f}"),
        ("HIGH REGAINS", "high_regains", "{:.0f}"),
        ("COUNTERPRESS", "counterpress_success_rate", "{:.1f}%"),
        ("SEQUENCE xT", "sequence_xT", "{:.2f}"),
    ]
    for idx, (label, key, fmt) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor(PANEL)
        clean_ax(ax)
        ax.set_xticks([]); ax.set_yticks([])
        hv = metric_lookup(team_metrics, "home", key)
        av = metric_lookup(team_metrics, "away", key)
        ax.text(0.05, 0.78, label, transform=ax.transAxes, color=MUTED, fontsize=8, fontweight="bold")
        ax.text(0.05, 0.38, fmt.format(hv), transform=ax.transAxes, color=TEXT, fontsize=19, fontweight="bold")
        ax.text(0.55, 0.38, fmt.format(av), transform=ax.transAxes, color=TEXT, fontsize=19, fontweight="bold")
        ax.text(0.05, 0.10, HOME_NAME, transform=ax.transAxes, color=MUTED, fontsize=7)
        ax.text(0.55, 0.10, AWAY_NAME, transform=ax.transAxes, color=MUTED, fontsize=7)

    flow_ax = fig.add_subplot(gs[1:, :2])
    clean_ax(flow_ax)
    flow_ax.grid(axis="y", color=GRID, lw=0.7)
    live = ~_bool(events.get("is_penalty_shootout", pd.Series(False, index=events.index)))
    shots = events[live & _bool(events.get("is_shot", pd.Series(False, index=events.index)))].copy()
    shots["xG"] = pd.to_numeric(shots["xG"], errors="coerce").fillna(0).clip(lower=0)
    max_min = max(90, int(pd.to_numeric(shots["minute"], errors="coerce").max()) + 2)
    for team_id, ls in [(HOME_ID, "-"), (AWAY_ID, (0, (7, 4)))]:
        team = shots[shots["team_id"].eq(team_id)].sort_values(["minute", "second"])
        vals = [0.0] + team["xG"].cumsum().tolist()
        mins = [0.0] + team["minute"].tolist()
        flow_ax.step(mins + [max_min], vals + [vals[-1]], where="post", color=TEAM_COLOR[team_id], lw=2.5, ls=ls)
        flow_ax.text(max_min + 1, vals[-1], f"{TEAM_NAME[team_id]} {vals[-1]:.2f}", color=TEAM_COLOR[team_id], va="center", fontsize=9, fontweight="bold")
    flow_ax.set_title("CUMULATIVE xG", loc="left", color=MUTED, fontsize=9, fontweight="bold")
    flow_ax.set_xlim(0, max_min + 14)
    flow_ax.tick_params(labelsize=8)

    comp_ax = fig.add_subplot(gs[1:, 2:])
    rows = [
        ("Deep completions", metric_lookup(team_metrics, "home", "deep_completions"), metric_lookup(team_metrics, "away", "deep_completions"), "{:.0f}"),
        ("Entry efficiency", metric_lookup(team_metrics, "home", "final_third_entry_efficiency"), metric_lookup(team_metrics, "away", "final_third_entry_efficiency"), "{:.1f}%"),
        ("Transition xT", metric_lookup(team_metrics, "home", "transition_xT"), metric_lookup(team_metrics, "away", "transition_xT"), "{:.2f}"),
        ("Rest-defence risk", metric_lookup(team_metrics, "home", "rest_defence_vulnerability"), metric_lookup(team_metrics, "away", "rest_defence_vulnerability"), "{:.1f}%"),
    ]
    row_dot_plot(comp_ax, rows, "TACTICAL COMPARISON")
    fig.text(0.945, 0.035, "NATIVE COMPACT CHARTS · NO FULL-PAGE SCREENSHOTS", ha="right", fontsize=8, color=NEUTRAL)
    return save(fig, "13_summary_board_redesign.png")


def export_metrics(team_metrics: pd.DataFrame, player_metrics: pd.DataFrame) -> tuple[Path, Path]:
    team_path = OUT_DIR / "team_advanced_metrics_real_data.csv"
    player_path = OUT_DIR / "player_sequence_metrics_real_data.csv"
    team_metrics.to_csv(team_path, index=False, encoding="utf-8-sig")
    player_metrics.to_csv(player_path, index=False, encoding="utf-8-sig")
    return team_path, player_path


def comparisons(new_paths: list[Path]) -> list[Path]:
    mapping = [
        ("11_shot_comparison.png", new_paths[0], "Shot profile"),
        ("41_transition_summary.png", new_paths[1], "Advanced metrics"),
        ("1_xg_flow.png", new_paths[2], "xG flow"),
        ("20_possession.png", new_paths[3], "Ball touches"),
        ("23_xt_per_minute.png", new_paths[4], "xT per minute"),
        ("7_xt_map_home.png", new_paths[5], "xT map"),
        ("24_progressive_home.png", new_paths[6], "Progressive passes"),
        ("28_defensive_hm_home.png", new_paths[7], "Defensive activity"),
        ("33_dominating_zone.png", new_paths[8], "Dominating zones"),
        ("5_pass_network_home.png", new_paths[9], "Pass network"),
        ("31_avg_position_home.png", new_paths[10], "Average positions"),
        ("36_high_turnovers_home.png", new_paths[11], "High regains"),
        ("board_08_pressing_20260719_093127.png", new_paths[12], "Summary board"),
    ]
    results: list[Path] = []
    for idx, (old_name, new_path, label) in enumerate(mapping, start=1):
        old_path = BASELINE_DIR / old_name
        if not old_path.exists() or not new_path.exists():
            continue
        old_img = plt.imread(old_path)
        new_img = plt.imread(new_path)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG)
        fig.suptitle(f"{label} · Before / Preview", color=TEXT, fontsize=20, fontweight="bold", y=0.97)
        for ax, img, title in zip(axes, [old_img, new_img], ["BEFORE · CURRENT OUTPUT", "PREVIEW · REAL DATA"]):
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title, color=MUTED, fontsize=11, fontweight="bold", pad=10)
        fig.subplots_adjust(left=0.015, right=0.985, top=0.91, bottom=0.02, wspace=0.035)
        path = COMPARE_DIR / f"{idx:02d}_{label.lower().replace(' ', '_')}_before_after.png"
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        results.append(path)
    return results


def comparison_pdf(comparison_paths: list[Path]) -> Path:
    pdf_path = OUT_DIR / "visual_redesign_before_after.pdf"
    with PdfPages(pdf_path) as pdf:
        for path in comparison_paths:
            img = plt.imread(path)
            fig = plt.figure(figsize=(16, 9), facecolor=BG)
            ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, dpi=120, facecolor=BG)
            plt.close(fig)
    return pdf_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    theme()
    events, players, xg, team_metrics, player_metrics = load_data()
    export_metrics(team_metrics, player_metrics)
    new_paths = [
        shot_profile(xg, events),
        advanced_metrics(team_metrics),
        xg_flow(events),
        touches(team_metrics),
        xt_per_minute(events),
        xt_map(events),
        progressive_map(events),
        defensive_activity(events),
        dominance(events),
        pass_network(events, players),
        average_positions(events, players),
        high_regains(events),
        summary_board(events, team_metrics, xg),
    ]
    comparison_paths = comparisons(new_paths)
    pdf_path = comparison_pdf(comparison_paths)
    print(f"Generated {len(new_paths)} redesigned visuals")
    print(f"Generated {len(comparison_paths)} before/after comparisons")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
