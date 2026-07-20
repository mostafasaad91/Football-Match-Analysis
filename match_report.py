# pyright: reportMissingImports=false, reportRedeclaration=false, reportReturnType=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPrivateImportUsage=false, reportOptionalMemberAccess=false, reportPossiblyUnboundVariable=false
"""
match_report.py
═════════════════════════════════════════════════════════════════════════════
Additive upgrades — styled to match the original football_match_analysis theme.

  Upgrade 1 — PPDA (Passes per Defensive Action) per team
  Upgrade 2 — Assist names + Open Play / Set Piece goal classification
  Upgrade 3 — Unified entry point: run_analysis(match_data) -> single PDF

Visuals use the same dark palette (BG_DARK / BG_MID / TEXT_BRIGHT / GRID_COL)
and team colours (C_RED for home, C_BLUE for away) as the original report.
"""

from __future__ import annotations

import os
import re
import ast
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from match_metrics import defensive_blocks_count, team_advanced_metrics

# Unified design system (AMOLED pure-black frame, fonts, palette)
try:
    from visualization_design import apply_unified_frame, rebrand_figure, _neon_backdrop, readable_on, readable_team_text, ACCENT_TEXT  # type: ignore
except Exception:  # pragma: no cover
    apply_unified_frame = None  # graceful fallback
    rebrand_figure = None
    _neon_backdrop = None
    readable_on = None
    readable_team_text = None
    ACCENT_TEXT = "#FFFFFF"


# ── Fonts: Inter (UI text) + JetBrains Mono (numbers/labels), matching the
# reference identity. Falls back to default sans/monospace if not installed.
FONT_SANS = "Inter Variable"
FONT_MONO = "JetBrains Mono"


def _register_fonts() -> None:
    import matplotlib.font_manager as _fm

    candidate_dirs = [
        os.path.expanduser("~/.fonts"),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ]
    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for fn in files:
                if fn.lower().endswith((".ttf", ".otf")):
                    try:
                        _fm.fontManager.addfont(os.path.join(root, fn))
                    except Exception:
                        pass
    available = {f.name for f in _fm.fontManager.ttflist}
    global FONT_SANS, FONT_MONO
    if FONT_SANS not in available:
        FONT_SANS = "DejaVu Sans"
    if FONT_MONO not in available:
        FONT_MONO = "DejaVu Sans Mono"


_register_fonts()


# ─────────────────────────────────────────────────────────────────────────────
# THEME — dark by default; light scripts can set MATCH_ANALYSIS_THEME=light
# before calling this module.
# ─────────────────────────────────────────────────────────────────────────────
def configure_theme(theme: str | None = None) -> None:
    global BG_DARK, BG_MID, BG_PANEL, BG_HEADER, GRID_COL, TEXT_MAIN, TEXT_BRIGHT, TEXT_DIM, TEXT_FADED, C_GOLD, RATING_CMAP
    theme = (theme or os.environ.get("MATCH_ANALYSIS_THEME", "dark")).strip().lower()
    if theme == "light":
        BG_DARK = "#FFFFFF"
        BG_MID = "#F3F4F6"
        BG_PANEL = "#FFFFFF"
        BG_HEADER = "#E5E7EB"
        GRID_COL = "#D1D5DB"
        TEXT_MAIN = "#1F2937"
        TEXT_BRIGHT = "#111827"
        TEXT_DIM = "#4B5563"
        TEXT_FADED = "#6B7280"
        C_GOLD = "#FFC23C"
        RATING_CMAP = LinearSegmentedColormap.from_list(
            "rating",
            ["#FEE2E2", "#FDBA74", "#FACC15", "#86EFAC", "#7DD3FC", "#38BDF8"],
        )
    else:
        BG_DARK = "#000000"
        BG_MID = "#0a0a0a"
        BG_PANEL = "#0a0a0a"
        BG_HEADER = "#101010"
        GRID_COL = "#1c1c1c"
        TEXT_MAIN = "#FFFFFF"
        TEXT_BRIGHT = "#FFFFFF"
        TEXT_DIM = "#9A9A9A"
        TEXT_FADED = "#5A5A5A"
        C_GOLD = "#FFC23C"
        RATING_CMAP = LinearSegmentedColormap.from_list(
            "rating",
            ["#3a0f10", "#7a1d1f", "#f97316", "#22c55e", "#7dd3fc", "#38BDF8"],
        )


configure_theme()


# Make standalone extension visuals inherit the same AMOLED/readable defaults.
def _apply_amoled_matplotlib_defaults() -> None:
    try:
        plt.rcParams.update(
            {
                "figure.facecolor": BG_DARK,
                "axes.facecolor": BG_PANEL,
                "savefig.facecolor": BG_DARK,
                "savefig.edgecolor": BG_DARK,
                "text.color": TEXT_MAIN,
                "axes.labelcolor": TEXT_DIM,
                "xtick.color": TEXT_DIM,
                "ytick.color": TEXT_DIM,
                "axes.edgecolor": GRID_COL,
                "grid.color": GRID_COL,
                "legend.facecolor": BG_PANEL,
                "legend.edgecolor": GRID_COL,
            }
        )
    except Exception:
        pass


_apply_amoled_matplotlib_defaults()
TEXT_SHADOW = [pe.withStroke(linewidth=2.6, foreground="#000000")]
TEXT_SHADOW_STRONG = [pe.withStroke(linewidth=3.4, foreground="#000000")]


def _ui_text(col: str | None = None, bg: str | None = None) -> str:
    """Readable text colour for dark AMOLED/navey UI accents."""
    try:
        if readable_on is not None and col:
            return readable_on(col, bg or BG_PANEL)
    except Exception:
        pass
    return TEXT_BRIGHT


def _team_label_color(col: str | None, bg: str | None = None) -> str:
    try:
        if readable_team_text is not None and col:
            return readable_team_text(col, bg or BG_PANEL)
    except Exception:
        pass
    return TEXT_BRIGHT


C_HOME = "#7A3DFF"  # canonical first-listed team role
C_AWAY = "#BEEA24"  # canonical second-listed team role
C_GOLD = "#FFC23C"
C_GREEN = "#22c55e"
C_PURPLE = "#a855f7"
OG_COLOR = "#ff00ff"


# ─────────────────────────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
VISUALS_DIR = os.path.join(OUTPUT_DIR, "visuals")

# Raster embedding DPI for visuals inside the unified PDF. The matplotlib PDF
# backend holds every embedded image in memory until the file is finalised, so
# on a RAM-tight machine ~40 figures at 320 DPI can exhaust memory during
# writeImages. 260/200 keeps the report sharp while cutting peak image memory
# by roughly a third — enough to finalise reliably.
PDF_VISUAL_DPI = 140
PDF_PAGE_DPI = 140


def _ensure_output_dirs() -> None:
    """Create /output and /output/visuals when they do not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VISUALS_DIR, exist_ok=True)


def _new_dark_fig(w: float, h: float):
    """Create a figure using the established dark theme."""
    fig = plt.figure(figsize=(w, h), facecolor=BG_DARK)
    if _neon_backdrop is not None:
        try:
            _neon_backdrop(fig)
        except Exception:
            pass
    return fig


def _style_dark_axes(ax, title: str = "", subtitle: str = ""):
    """Apply the dark theme to an axes object."""
    ax.set_facecolor(BG_PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    if title:
        ax.set_title(
            title, color=TEXT_BRIGHT, fontsize=14, fontweight="bold", pad=12, loc="left"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _na(value: Any, fmt: str | None = None) -> str:
    """Format a value, returning 'N/A' when appropriate."""
    if value is None:
        return "N/A"
    try:
        if isinstance(value, float) and np.isnan(value):
            return "N/A"
    except Exception:
        pass
    if fmt and isinstance(value, (int, float)):
        try:
            return format(value, fmt)
        except Exception:
            return str(value)
    return str(value)


def _safe_int(v, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _short_name(name: str, max_len: int = 22) -> str:
    """Shorten a player name when it exceeds the requested length."""
    if not name or name == "N/A":
        return name or "N/A"
    if len(name) <= max_len:
        return name
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[-1]}"
    return name[: max_len - 1] + "…"


# ═════════════════════════════════════════════════════════════════════════════
# UPGRADE 1 — PPDA
# ═════════════════════════════════════════════════════════════════════════════
def calculate_ppda(
    events: pd.DataFrame, team_id: int, opp_id: int, threshold: float = 40.0
) -> dict:
    """
    PPDA is the opponent's passes in its own half divided by the pressing
    team's tackles, interceptions, fouls, and recoveries in the same area.

    A threshold of 40 represents the front 60% of the pitch from the pressing
    team's perspective, following Colin Trainor's classic method.
    """
    if events is None or events.empty:
        return {
            "passes_allowed": 0,
            "defensive_actions": 0,
            "ppda": None,
            "zone_label": "opp own half",
        }

    opp_threshold = 100.0 - threshold

    opp_passes_mask = (
        (events["team_id"] == opp_id)
        & (events.get("is_pass", False) == True)  # noqa: E712
        & (events["x"].astype(float) < opp_threshold)
    )
    passes_allowed = int(opp_passes_mask.sum())

    DEF_TYPES = {"Tackle", "Interception", "Foul", "Challenge", "BallRecovery"}
    def_mask = (
        (events["team_id"] == team_id)
        & (events["type"].isin(DEF_TYPES))
        & (events["x"].astype(float) > threshold)
    )
    defensive_actions = int(def_mask.sum())

    ppda = (passes_allowed / defensive_actions) if defensive_actions > 0 else None

    return {
        "passes_allowed": passes_allowed,
        "defensive_actions": defensive_actions,
        "ppda": ppda,
        "zone_label": f"opp 60% (x<{opp_threshold:.0f} for opp)",
    }


def compute_ppda_both(info: dict, events: pd.DataFrame) -> dict:
    """Calculate PPDA for both teams."""
    return {
        "home": calculate_ppda(events, info.get("home_id"), info.get("away_id")),
        "away": calculate_ppda(events, info.get("away_id"), info.get("home_id")),
    }


def _ppda_intensity_label(ppda: float | None) -> tuple[str, str]:
    """Classify PPDA as Elite, High, Medium, or Low and return its color."""
    if ppda is None:
        return "N/A", TEXT_FADED
    if ppda < 8.0:
        return "ELITE PRESS", "#22c55e"
    if ppda < 11.0:
        return "HIGH PRESS", "#84cc16"
    if ppda < 14.0:
        return "MEDIUM BLOCK", "#facc15"
    return "LOW BLOCK", "#f97316"


def draw_ppda_gauge(ppda_data: dict, info: dict, save_path: str | None = None):
    """
    Draw a complete PPDA analysis with a dial and numeric breakdown for each
    team, plus a small pitch-area guide and an intensity classification.
    """
    fig = _new_dark_fig(14, 8)
    fig.patch.set_facecolor(BG_DARK)
    if _neon_backdrop is not None:
        try:
            _neon_backdrop(fig)
        except Exception:
            pass

    home_name = info.get("home_name") or "Home"
    away_name = info.get("away_name") or "Away"
    h = ppda_data.get("home", {}).get("ppda")
    a = ppda_data.get("away", {}).get("ppda")
    h_passes = ppda_data.get("home", {}).get("passes_allowed", 0)
    h_def = ppda_data.get("home", {}).get("defensive_actions", 0)
    a_passes = ppda_data.get("away", {}).get("passes_allowed", 0)
    a_def = ppda_data.get("away", {}).get("defensive_actions", 0)

    # ── Unified frame (yellow top + side bars, section label, title) ──
    score = info.get("score") or "—"
    if apply_unified_frame is not None:
        apply_unified_frame(
            fig,
            section="PRESSING · PPDA",
            title="Pressing Analysis — Passes Per Defensive Action",
            subtitle="Lower PPDA = more aggressive press in the opponent's "
            "60% of the pitch",
            accent=C_GOLD,
            home_name=home_name,
            away_name=away_name,
            score=str(score),
            footer_note="Method: Colin Trainor (2014)",
        )
    else:
        fig.text(
            0.5,
            0.95,
            "PRESSING ANALYSIS — PPDA",
            ha="center",
            color=TEXT_BRIGHT,
            fontsize=22,
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)],
        )
        fig.text(
            0.5,
            0.91,
            "Passes Per Defensive Action  •  lower = more aggressive press",
            ha="center",
            color=TEXT_DIM,
            fontsize=11,
            style="italic",
        )

    # ── Two semi-circular dials ──
    def _draw_dial(ax_pos, name, value, passes, def_acts, color):
        ax = fig.add_axes(ax_pos, projection="polar")
        ax.set_facecolor(BG_DARK)

        ax.set_theta_zero_location("W")
        ax.set_theta_direction(-1)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_ylim(0, 1)

        v = value if value is not None else 0
        vmin, vmax = 5.0, 25.0

        ratio = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
        angle = ratio * np.pi

        n_seg = 60
        thetas = np.linspace(0, np.pi, n_seg + 1)
        zone_colors = []
        for i in range(n_seg):
            t = i / n_seg
            if t < 0.15:
                zone_colors.append("#22c55e")
            elif t < 0.30:
                zone_colors.append("#84cc16")
            elif t < 0.45:
                zone_colors.append("#facc15")
            else:
                zone_colors.append("#f97316")
        for i in range(n_seg):
            ax.bar(
                (thetas[i] + thetas[i + 1]) / 2,
                0.18,
                bottom=0.78,
                width=(thetas[i + 1] - thetas[i]) * 0.95,
                color=zone_colors[i],
                edgecolor="none",
                alpha=0.75,
            )

        if value is not None:
            ax.plot(
                [angle, angle],
                [0, 0.92],
                color=color,
                lw=4,
                solid_capstyle="round",
                zorder=5,
            )
            ax.scatter(
                [angle],
                [0],
                s=140,
                color=color,
                edgecolor=TEXT_BRIGHT,
                linewidth=1.5,
                zorder=6,
            )
            ax.scatter(
                [angle],
                [0.92],
                s=70,
                color=TEXT_BRIGHT,
                edgecolor=color,
                linewidth=2,
                zorder=7,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["polar"].set_visible(False)

        for tick_v, tick_label in [
            (5, "5"),
            (10, "10"),
            (15, "15"),
            (20, "20"),
            (25, "25"),
        ]:
            r = (tick_v - vmin) / (vmax - vmin)
            theta_t = r * np.pi
            ax.text(
                theta_t,
                1.05,
                tick_label,
                ha="center",
                va="center",
                color=TEXT_DIM,
                fontsize=8.5,
            )

        cx, cy = ax_pos[0] + ax_pos[2] / 2, ax_pos[1] + 0.06
        val_str = f"{value:.2f}" if value is not None else "N/A"
        fig.text(
            cx,
            cy + 0.04,
            val_str,
            ha="center",
            color=color,
            fontsize=36,
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)],
        )

        intensity_lbl, intensity_col = _ppda_intensity_label(value)
        fig.text(
            cx,
            cy,
            intensity_lbl,
            ha="center",
            color=intensity_col,
            fontsize=11,
            fontweight="bold",
        )

        fig.text(
            cx,
            ax_pos[1] + ax_pos[3] - 0.02,
            name,
            ha="center",
            color=TEXT_BRIGHT,
            fontsize=15,
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground=BG_DARK)],
        )

        fig.text(
            cx - 0.06,
            cy - 0.05,
            str(passes),
            ha="center",
            color=color,
            fontsize=18,
            fontweight="bold",
        )
        fig.text(
            cx - 0.06,
            cy - 0.085,
            "OPP PASSES",
            ha="center",
            color=TEXT_DIM,
            fontsize=8,
            fontweight="bold",
        )
        fig.text(
            cx + 0.06,
            cy - 0.05,
            str(def_acts),
            ha="center",
            color=color,
            fontsize=18,
            fontweight="bold",
        )
        fig.text(
            cx + 0.06,
            cy - 0.085,
            "DEF ACTIONS",
            ha="center",
            color=TEXT_DIM,
            fontsize=8,
            fontweight="bold",
        )

    _draw_dial([0.05, 0.42, 0.40, 0.45], home_name, h, h_passes, h_def, C_HOME)
    _draw_dial([0.55, 0.42, 0.40, 0.45], away_name, a, a_passes, a_def, C_AWAY)

    if h is not None and a is not None:
        if h < a:
            verdict = f"{home_name} pressed more aggressively"
            v_color = C_HOME
            diff = a - h
        elif a < h:
            verdict = f"{away_name} pressed more aggressively"
            v_color = C_AWAY
            diff = h - a
        else:
            verdict = "Both teams pressed equally"
            v_color = TEXT_BRIGHT
            diff = 0
        fig.text(
            0.5,
            0.34,
            verdict,
            ha="center",
            color=v_color,
            fontsize=14,
            fontweight="bold",
        )
        if diff > 0:
            fig.text(
                0.5,
                0.305,
                f"PPDA differential: {diff:.2f}",
                ha="center",
                color=TEXT_DIM,
                fontsize=10,
            )

    pitch_ax = fig.add_axes([0.10, 0.06, 0.80, 0.18])
    pitch_ax.set_facecolor("#040c04")
    pitch_ax.set_xlim(0, 100)
    pitch_ax.set_ylim(0, 30)
    pitch_ax.set_xticks([])
    pitch_ax.set_yticks([])
    for s in pitch_ax.spines.values():
        s.set_edgecolor(GRID_COL)

    pitch_ax.plot([50, 50], [0, 30], color=TEXT_DIM, lw=1, ls="--", alpha=0.6)

    pitch_ax.axvline(60, color="#facc15", lw=1.2, alpha=0.7, ls="--")

    pitch_ax.add_patch(
        mpatches.Rectangle((60, 0), 40, 30, facecolor="#facc15", alpha=0.15, lw=0)
    )
    pitch_ax.text(
        80,
        15,
        "PRESSING ZONE\n(opp 60% of pitch)",
        ha="center",
        va="center",
        color="#facc15",
        fontsize=10,
        fontweight="bold",
    )
    pitch_ax.text(
        30,
        15,
        "OWN HALF",
        ha="center",
        va="center",
        color=TEXT_FADED,
        fontsize=9,
        style="italic",
    )
    pitch_ax.text(2, 27, "← own goal", color=TEXT_FADED, fontsize=8)
    pitch_ax.text(98, 27, "opp goal →", color=TEXT_FADED, fontsize=8, ha="right")

    fig.text(
        0.5,
        0.025,
        "Method: opponent passes attempted in their own 60% "
        "÷ tackles + interceptions + fouls + challenges + recoveries  "
        "(Colin Trainor, 2014)",
        ha="center",
        color=TEXT_FADED,
        fontsize=8.5,
        style="italic",
    )

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=BG_DARK)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# UPGRADE 2 — Goals classification
# ═════════════════════════════════════════════════════════════════════════════
def _goal_qualifiers(row: pd.Series) -> set[str]:
    quals = row.get("qualifier_names") or []
    if isinstance(quals, str):
        text = quals.strip()
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                quals = parsed
            else:
                quals = re.split(r"[,|;]", text)
        except Exception:
            quals = re.split(r"[,|;\\[\\]'\\\"]+", text)
    elif not isinstance(quals, (list, tuple, set)):
        quals = []
    return {str(q) for q in quals if q is not None}


def _truthy_flag(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _event_type_text(row: pd.Series) -> str:
    parts = []
    for key in ("event_type", "type", "type_display", "displayName", "outcome"):
        val = row.get(key)
        if val is not None:
            parts.append(str(val))
    parts.extend(_goal_qualifiers(row))
    return " ".join(parts).lower()


def _set_piece_subtype_from_event(row: pd.Series) -> str | None:
    text = _event_type_text(row)
    if "corner" in text:
        return "Corner"
    if "throw" in text:
        return "Throw-In"
    if "directfreekick" in text or "direct free" in text:
        return "Direct Free Kick"
    if "freekick" in text or "free kick" in text or "foul" in text:
        return "Free Kick"
    return None


def _previous_restart_subtype(
    row: pd.Series, events: pd.DataFrame | None = None
) -> str | None:
    if events is None or getattr(events, "empty", True):
        return None
    try:
        idx = row.name
        if idx in events.index:
            pos = events.index.get_loc(idx)
            if isinstance(pos, slice):
                pos = pos.start
            prior = events.iloc[: int(pos)]
        else:
            minute = row.get("minute")
            if minute is None:
                prior = events
            else:
                prior = events[events.get("minute", -1) <= minute]
    except Exception:
        prior = events

    if prior is None or prior.empty:
        return None
    team_id = row.get("team_id")
    same_team = (
        prior[prior.get("team_id") == team_id] if "team_id" in prior.columns else prior
    )
    if same_team.empty:
        same_team = prior

    for _, prev in same_team.tail(4).iloc[::-1].iterrows():
        subtype = _set_piece_subtype_from_event(prev)
        if subtype:
            return subtype
        # Stop after the immediately preceding non-empty action by the scoring side.
        return None
    return None


def goal_body_part_label(row: pd.Series) -> str:
    body = row.get("body_part")
    body_txt = "" if body is None else str(body).strip()
    low = body_txt.lower().replace("_", " ")
    if _truthy_flag(row.get("is_header", False)) or "head" in low:
        return "Header"
    if "right" in low and "foot" in low:
        return "Right Foot"
    if "left" in low and "foot" in low:
        return "Left Foot"
    if "foot" in low:
        return "Foot"
    if low:
        return body_txt.replace("_", " ").title()
    return "Unknown Body Part"


def classify_goal_type(
    row: pd.Series, events: pd.DataFrame | None = None
) -> tuple[str, str]:
    """Classify a goal as Penalty, Set Piece or Open Play."""
    qset = _goal_qualifiers(row)
    if _truthy_flag(row.get("is_own_goal", False)):
        return "Open Play", "Own Goal"
    if _truthy_flag(row.get("is_penalty", False)) or "Penalty" in qset:
        return "Penalty", "Penalty"
    if _truthy_flag(row.get("is_direct_fk", False)) or "DirectFreekick" in qset:
        return "Set Piece", "Direct Free Kick"
    direct_subtype = _set_piece_subtype_from_event(row)
    if direct_subtype:
        return "Set Piece", direct_subtype
    previous_subtype = _previous_restart_subtype(row, events)
    if previous_subtype:
        return "Set Piece", previous_subtype
    return "Open Play", "Open Play"


def build_goals_log(events: pd.DataFrame, info: dict) -> pd.DataFrame:
    """Build a goal table with assists and classifications."""
    cols = [
        "minute",
        "team",
        "scorer",
        "assist",
        "category",
        "subtype",
        "xG",
        "is_own_goal",
    ]
    if events is None or events.empty:
        return pd.DataFrame(columns=cols)

    gdf = events[events["is_goal"] == True].copy()  # noqa: E712
    if gdf.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, r in gdf.sort_values("minute").iterrows():
        category, subtype = classify_goal_type(r, events)
        scoring_team_id = r.get("scoring_team", r.get("team_id"))
        team_name = (
            info.get("home_name")
            if scoring_team_id == info.get("home_id")
            else info.get("away_name")
        )
        assist = r.get("assist_player") or ""
        rows.append(
            {
                "minute": _safe_int(r.get("minute")),
                "team": team_name or "N/A",
                "scorer": r.get("player") or "N/A",
                "assist": assist if assist else "N/A",
                "category": category,
                "subtype": subtype,
                "body_part": goal_body_part_label(r),
                "xG": r.get("xG"),
                "is_own_goal": bool(r.get("is_own_goal", False)),
                "scoring_team_id": scoring_team_id,
            }
        )
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# UPGRADE 3 — Player stats extraction & polished tables
# ═════════════════════════════════════════════════════════════════════════════
def _flatten_stat(value: Any) -> Any:
    """Convert a WhoScored statistics dictionary to a scalar value."""
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, dict):
        if "total" in value:
            return value["total"]
        nums = [v for v in value.values() if isinstance(v, (int, float))]
        if nums:
            return sum(nums)
    return None


STAT_GROUPS = [
    (
        "Identity",
        [
            ("name", "Player"),
            ("position", "Pos"),
            ("shirt_no", "#"),
            ("minutesPlayed", "Min"),
        ],
    ),
    (
        "Attack",
        [
            ("goals", "G"),
            ("assists", "A"),
            ("shotsTotal", "Sh"),
            ("shotsOnTarget", "SoT"),
            ("passesKey", "KP"),
            ("dribblesWon", "Drb"),
        ],
    ),
    (
        "Passing",
        [("passesTotal", "Pass"), ("passesAccurate", "Acc"), ("touches", "Tch")],
    ),
    (
        "Defense",
        [
            ("tacklesTotal", "Tkl"),
            ("interceptions", "Int"),
            ("aerialsWon", "Aer"),
            ("foulsCommited", "Fls"),
            ("wasFouled", "Fld"),
        ],
    ),
    ("Score", [("ratings", "Rating")]),
]


GROUP_HEADER_COLORS = {
    "Identity": "#151515",
    "Attack": "#3b1f2f",
    "Passing": "#1f3a2f",
    "Defense": "#3a2f1f",
    "Score": "#2a1f3a",
}

GROUP_HEADER_COLORS_LIGHT = {
    "Identity": "#E0F2FE",
    "Attack": "#FCE7F3",
    "Passing": "#DCFCE7",
    "Defense": "#FEF3C7",
    "Score": "#EDE9FE",
}


def extract_player_stats(md: dict) -> dict:
    """Extract all player statistics for both teams."""
    out: dict = {}

    for side in ("home", "away"):
        team = md.get(side, {}) or {}
        rows = []
        for p in team.get("players", []) or []:
            stats_raw = p.get("stats", {}) or {}
            row = {
                "name": p.get("name") or "N/A",
                "position": p.get("position") or "N/A",
                "shirt_no": p.get("shirtNo"),
                "is_first_xi": bool(p.get("isFirstEleven", False)),
            }
            for k, v in stats_raw.items():
                row[k] = _flatten_stat(v)
            if row.get("ratings") is None and p.get("playerScore") is not None:
                row["ratings"] = p.get("playerScore")
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            sort_cols = ["is_first_xi"]
            ascending = [False]
            if "ratings" in df.columns:
                sort_cols.append("ratings")
                ascending.append(False)
            df = df.sort_values(
                sort_cols, ascending=ascending, na_position="last"
            ).reset_index(drop=True)
        out[side] = df

    return out


def _rating_color(rating: Any) -> str:
    """Map a rating from 5.0 through 8.5+ to a color."""
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return BG_PANEL
    # normalize 5..9 → 0..1
    t = max(0.0, min(1.0, (r - 5.0) / 4.0))
    rgba = RATING_CMAP(t)
    return f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"


def _format_cell(key: str, value: Any) -> str:
    """Format a value according to its column type."""
    if value is None:
        return "—"
    try:
        if isinstance(value, float) and np.isnan(value):
            return "—"
    except Exception:
        pass
    if key == "name":
        return _short_name(str(value), 20)
    if key == "ratings":
        try:
            return f"{float(value):.2f}"
        except Exception:
            return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.0f}"
    return str(value)


def draw_player_stats_table(
    df: pd.DataFrame,
    team_name: str,
    team_color: str = C_HOME,
    save_path: str | None = None,
):
    """
    Draw a dark-themed player-statistics table with variable-width grouped
    columns and a rating heatmap. The Player column is wider than the others.
    """

    COL_W = {
        "name": 3.6,
        "position": 1.05,
        "shirt_no": 0.55,
        "minutesPlayed": 0.7,
        "ratings": 1.15,
    }
    DEFAULT_W = 0.85

    # ── Build flat columns from groups ──
    flat_keys: list[str] = []
    flat_labels: list[str] = []
    flat_widths: list[float] = []
    group_spans: list[tuple[str, float, float]] = []  # (gname, x_start, x_end)
    x_cursor = 0.0
    for gname, items in STAT_GROUPS:
        actual = []
        for k, lbl in items:
            if k in df.columns or k in ("name", "position", "shirt_no"):
                actual.append((k, lbl))
        if not actual:
            continue
        x_start = x_cursor
        for k, lbl in actual:
            w = COL_W.get(k, DEFAULT_W)
            flat_keys.append(k)
            flat_labels.append(lbl)
            flat_widths.append(w)
            x_cursor += w
        group_spans.append((gname, x_start, x_cursor))

    total_w = x_cursor if x_cursor > 0 else 1.0

    x_lefts = []
    acc = 0.0
    for w in flat_widths:
        x_lefts.append(acc)
        acc += w

    n_rows = max(len(df), 1)

    fig = _new_dark_fig(max(15, total_w * 0.95), max(7, 0.42 * n_rows + 3.0))
    fig.patch.set_facecolor(BG_DARK)

    # ── Title bar ──
    fig.text(
        0.04,
        0.965,
        f"PLAYER STATISTICS — {team_name.upper()}",
        color=TEXT_BRIGHT,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.935,
        "Starters first  •  ratings coloured by performance  •  "
        "G / A highlighted  •  '—' = unavailable",
        color=TEXT_DIM,
        fontsize=10,
        style="italic",
    )

    bar_ax = fig.add_axes([0.04, 0.91, 0.92, 0.012])
    bar_ax.set_facecolor(team_color)
    bar_ax.set_xticks([])
    bar_ax.set_yticks([])
    for s in bar_ax.spines.values():
        s.set_visible(False)

    # ── Main table axes ──
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.83])
    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, n_rows + 2.4)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL)

    if df.empty:
        ax.text(
            total_w / 2,
            n_rows / 2 + 1.2,
            "No player data available",
            ha="center",
            va="center",
            color=TEXT_DIM,
            fontsize=14,
            style="italic",
        )
        if save_path:
            fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=BG_DARK)
        return fig

    # ── Group header row (y=0..1) ──
    is_light_theme = BG_DARK.upper() in {"#FFFFFF", "WHITE"}
    group_header_colors = (
        GROUP_HEADER_COLORS_LIGHT if is_light_theme else GROUP_HEADER_COLORS
    )
    for gname, x0, x1 in group_spans:
        ax.add_patch(
            mpatches.Rectangle(
                (x0, 0),
                x1 - x0,
                1.0,
                facecolor=group_header_colors.get(gname, BG_MID),
                edgecolor=GRID_COL,
                lw=0.6,
            )
        )
        ax.text(
            (x0 + x1) / 2,
            0.5,
            gname.upper(),
            ha="center",
            va="center",
            color=TEXT_BRIGHT,
            fontsize=10.5,
            fontweight="bold",
            path_effects=(
                []
                if is_light_theme
                else [pe.withStroke(linewidth=2, foreground=BG_DARK)]
            ),
        )

    # ── Column header row (y=1..2.2) ──
    for j, lbl in enumerate(flat_labels):
        x0 = x_lefts[j]
        w = flat_widths[j]
        ax.add_patch(
            mpatches.Rectangle(
                (x0, 1.0),
                w,
                1.2,
                facecolor=BG_MID,
                edgecolor=GRID_COL,
                lw=0.5,
            )
        )
        if flat_keys[j] == "name":
            ha = "left"
            tx = x0 + 0.55
        else:
            ha = "center"
            tx = x0 + w / 2
        ax.text(
            tx,
            1.6,
            lbl,
            ha=ha,
            va="center",
            color=TEXT_DIM,
            fontsize=9.5,
            fontweight="bold",
        )

    # ── Data rows ──
    y0 = 2.2
    row_h = 1.0
    for i, (_, r) in enumerate(df.iterrows()):
        y = y0 + i * row_h
        is_starter = bool(r.get("is_first_xi", False))

        base_color = (
            BG_PANEL if i % 2 == 0 else ("#FFFFFF" if is_light_theme else "#0A0A0A")
        )
        if not is_starter:
            base_color = "#F1F5F9" if is_light_theme else "#080808"
        ax.add_patch(
            mpatches.Rectangle(
                (0, y),
                total_w,
                row_h,
                facecolor=base_color,
                edgecolor=GRID_COL,
                lw=0.3,
                alpha=0.95,
            )
        )

        starter_col = team_color if is_starter else TEXT_FADED
        ax.add_patch(
            mpatches.Rectangle(
                (0, y),
                0.10,
                row_h,
                facecolor=starter_col,
                lw=0,
                alpha=0.9 if is_starter else 0.35,
            )
        )

        for j, key in enumerate(flat_keys):
            x0 = x_lefts[j]
            w = flat_widths[j]
            raw = r.get(key)
            text = _format_cell(key, raw)
            text_color = TEXT_MAIN
            fontweight = "normal"

            if key == "name":

                max_chars = int(w * 7)
                disp = _short_name(str(raw or "N/A"), max_chars)
                ax.text(
                    x0 + 0.20,
                    y + row_h / 2,
                    disp,
                    ha="left",
                    va="center",
                    color=TEXT_BRIGHT if is_starter else TEXT_DIM,
                    fontsize=10,
                    fontweight="bold" if is_starter else "normal",
                )
                continue

            if key == "position":
                if text and text != "—":
                    pad_x = 0.12
                    ax.add_patch(
                        mpatches.FancyBboxPatch(
                            (x0 + pad_x, y + 0.22),
                            w - 2 * pad_x,
                            row_h - 0.44,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            facecolor="#F0F4FF" if is_light_theme else GRID_COL,
                            edgecolor=team_color,
                            lw=0.7,
                        )
                    )
                ax.text(
                    x0 + w / 2,
                    y + row_h / 2,
                    text,
                    ha="center",
                    va="center",
                    color=TEXT_BRIGHT,
                    fontsize=9,
                    fontweight="bold",
                )
                continue

            if key == "ratings" and raw is not None:
                rc = _rating_color(raw)
                pad_x = 0.10
                ax.add_patch(
                    mpatches.FancyBboxPatch(
                        (x0 + pad_x, y + 0.18),
                        w - 2 * pad_x,
                        row_h - 0.36,
                        boxstyle="round,pad=0.02,rounding_size=0.12",
                        facecolor=rc,
                        edgecolor="white",
                        lw=0.7,
                    )
                )
                ax.text(
                    x0 + w / 2,
                    y + row_h / 2,
                    text,
                    ha="center",
                    va="center",
                    color="#0a0a0a" if _is_light(rc) else TEXT_BRIGHT,
                    fontsize=10,
                    fontweight="bold",
                )
                continue

            if key == "goals" and raw and float(raw) > 0:
                text_color = C_GOLD
                fontweight = "bold"
            elif key == "assists" and raw and float(raw) > 0:
                text_color = C_GREEN
                fontweight = "bold"
            elif text in ("—", "N/A"):
                text_color = TEXT_FADED

            ax.text(
                x0 + w / 2,
                y + row_h / 2,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9.2,
                fontweight=fontweight,
            )

    # ── Footer legend ──
    fig.text(
        0.04,
        0.025,
        "Starters: bold + coloured side-bar    •    Substitutes/unused: dimmed    "
        "•    Rating cell: red→yellow→green→gold (5.0 → 9.0)",
        color=TEXT_FADED,
        fontsize=8.5,
        style="italic",
    )

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=BG_DARK)
    return fig


def _is_light(hex_color: str) -> bool:
    """Return whether a color is light enough to require dark text."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
        return lum > 0.55
    except Exception:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# UPGRADE 4 — Unified PDF + dark-themed pages
# ═════════════════════════════════════════════════════════════════════════════
def _draw_section_divider(
    pdf, num: str, title: str, subtitle: str, accent: str = C_GOLD
):
    """Create a divider page for a report section."""
    fig = _new_dark_fig(11.7, 8.27)
    fig.patch.set_facecolor(BG_DARK)

    bar_ax = fig.add_axes([0.06, 0.20, 0.008, 0.60])
    bar_ax.set_facecolor(accent)
    bar_ax.set_xticks([])
    bar_ax.set_yticks([])
    for s in bar_ax.spines.values():
        s.set_visible(False)

    fig.text(
        0.10,
        0.55,
        num,
        color=accent,
        fontsize=140,
        fontweight="bold",
        alpha=0.18,
        family="serif",
    )

    fig.text(
        0.10,
        0.62,
        f"SECTION {num}",
        color=accent,
        fontsize=12,
        fontweight="bold",
        family="sans-serif",
    )
    fig.text(
        0.10,
        0.55,
        title,
        color=TEXT_BRIGHT,
        fontsize=36,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)],
    )
    import textwrap as _tw

    fig.text(
        0.10,
        0.49,
        "\n".join(_tw.wrap(subtitle, width=74)),
        color=TEXT_DIM,
        fontsize=12.5,
        style="italic",
        va="top",
        linespacing=1.4,
    )

    line_ax = fig.add_axes([0.10, 0.40, 0.40, 0.002])
    line_ax.set_facecolor(accent)
    line_ax.set_xticks([])
    line_ax.set_yticks([])
    for s in line_ax.spines.values():
        s.set_visible(False)

    fig.text(
        0.10,
        0.20,
        "M A T C H   A N A L Y S I S   R E P O R T",
        color=TEXT_FADED,
        fontsize=8,
        fontweight="bold",
    )

    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _draw_player_radar_page(
    pdf, player, pizza_fig, accent, page_no=None, team_name="", role="", commentary=""
):
    """Portrait report page: player-pizza on top, professional tactical read below."""
    import io
    import textwrap as _tw
    import matplotlib.image as _mpimg

    buf = io.BytesIO()
    pizza_fig.savefig(buf, format="png", dpi=160, facecolor=BG_DARK)
    plt.close(pizza_fig)
    buf.seek(0)
    img = _mpimg.imread(buf)

    page_w, page_h = 8.27, 11.69
    fig = plt.figure(figsize=(page_w, page_h), facecolor=BG_DARK)
    fig.patch.set_facecolor(BG_DARK)

    # header
    fig.text(
        0.07,
        0.958,
        "Player Radar",
        ha="left",
        va="center",
        color=TEXT_BRIGHT,
        fontsize=18,
        fontweight="bold",
        family="serif",
    )
    fig.text(
        0.93,
        0.958,
        str(team_name),
        ha="right",
        va="center",
        color=accent,
        fontsize=12,
        fontweight="bold",
        family="monospace",
    )
    line_ax = fig.add_axes((0.07, 0.936, 0.86, 0.002))
    line_ax.set_facecolor(accent)
    line_ax.set_xticks([])
    line_ax.set_yticks([])
    for s in line_ax.spines.values():
        s.set_visible(False)

    # pizza — large, top ~58% of page, aspect preserved (inch-correct)
    ih, iw = img.shape[0], img.shape[1]
    img_ar = iw / ih
    fx, fy, fw, fh = 0.05, 0.365, 0.90, 0.55
    W_in, H_in = fw * page_w, fh * page_h
    if img_ar >= W_in / H_in:
        w_in = W_in
        h_in = W_in / img_ar
    else:
        h_in = H_in
        w_in = H_in * img_ar
    dw, dh = w_in / page_w, h_in / page_h
    dx = fx + (fw - dw) / 2
    dy = fy + (fh - dh) / 2
    ax = fig.add_axes([dx, dy, dw, dh])
    ax.imshow(img, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # commentary block
    ax_t = fig.add_axes((0.07, 0.07, 0.86, 0.28))
    ax_t.set_axis_off()
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.text(
        0.0,
        1.0,
        "TACTICAL READ",
        ha="left",
        va="top",
        color=accent,
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    wrapped = "\n\n".join(
        _tw.fill(p.strip(), width=104)
        for p in str(commentary).split("\n\n")
        if p.strip()
    )
    ax_t.text(
        0.0,
        0.90,
        wrapped,
        ha="left",
        va="top",
        color=TEXT_MAIN,
        fontsize=9.3,
        family="serif",
        linespacing=1.45,
    )

    _page_rail(fig, accent, label=f"Player Radar · {player}", page_no=page_no)
    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _draw_closing_page(pdf, info, events=None, ppda=None):
    """Closing page — a bookend to the cover: the same faded tactical artwork,
    gold frame, the report wordmark, a final result recap and the credits."""
    import matplotlib.image as _mpimg

    fig = _new_dark_fig(11.7, 8.27)
    home = info.get("home_name") or "Home"
    away = info.get("away_name") or "Away"
    score = str(info.get("score") or "").replace("*", "").strip()
    comp = str(info.get("competition") or "").strip()
    mdate = str(info.get("date") or "").strip()

    # Faded artwork background + veil (matches the cover) ----------------
    try:
        img = _mpimg.imread(_COVER_BG_PATH)
        bgimg = fig.add_axes([0.146, 0.0, 0.708, 1.0])
        bgimg.set_axis_off()
        bgimg.imshow(img, alpha=0.32, zorder=0, aspect="auto")
    except Exception:
        pass
    veil = fig.add_axes([0, 0, 1, 1])
    veil.set_axis_off()
    veil.set_xlim(0, 1)
    veil.set_ylim(0, 1)
    veil.add_patch(
        mpatches.Rectangle((0, 0), 1, 1, facecolor=BG_DARK, alpha=0.42, lw=0, zorder=1)
    )
    veil.add_patch(
        mpatches.Rectangle(
            (0.02, 0.026),
            0.96,
            0.948,
            facecolor="none",
            edgecolor=C_GOLD,
            lw=1.0,
            alpha=0.4,
            zorder=2,
        )
    )
    for x, y, dx, dy in [
        (0.035, 0.958, 1, -1),
        (0.965, 0.958, -1, -1),
        (0.035, 0.042, 1, 1),
        (0.965, 0.042, -1, 1),
    ]:
        veil.plot(
            [x, x + dx * 0.028], [y, y], color=C_GOLD, lw=1.8, alpha=0.8, zorder=2
        )
        veil.plot([x, x], [y, y + dy * 0.04], color=C_GOLD, lw=1.8, alpha=0.8, zorder=2)

    # Result recap ------------------------------------------------------
    recap = score
    try:
        _, pens = _extra_time_and_pens(events, info)
        if pens is not None:
            winner = home if pens[0] > pens[1] else away
            loser = away if winner == home else home
            recap = f"{winner} edge {loser} {max(pens)}-{min(pens)} on penalties (AET)"
        elif events is not None and ppda is not None:
            k = _match_kpis(events, info, ppda)
            if k["goals"][0] != k["goals"][1]:
                w = home if k["goals"][0] > k["goals"][1] else away
                recap = f"{w} win it, {score}"
    except Exception:
        pass

    fig.text(0.5, 0.66, "✦", ha="center", color=C_GOLD, fontsize=30, zorder=5)
    fig.text(
        0.5,
        0.57,
        "MATCH ANALYSIS REPORT",
        ha="center",
        color=TEXT_BRIGHT,
        fontsize=26,
        fontweight="bold",
        family="monospace",
        zorder=5,
    )
    aln = fig.add_axes([0.36, 0.545, 0.28, 0.0018])
    aln.set_facecolor(C_GOLD)
    aln.set_xticks([])
    aln.set_yticks([])
    for s in aln.spines.values():
        s.set_visible(False)
    if recap:
        fig.text(
            0.5,
            0.50,
            recap,
            ha="center",
            color=C_GOLD,
            fontsize=12.5,
            fontweight="bold",
            style="italic",
            zorder=5,
        )
    _sub = " · ".join([b for b in [f"{home} vs {away}", comp, mdate] if b])
    fig.text(0.5, 0.455, _sub, ha="center", color=TEXT_DIM, fontsize=10.5, zorder=5)
    fig.text(
        0.5,
        0.36,
        "Created by Mostafa Saad",
        ha="center",
        color=TEXT_BRIGHT,
        fontsize=13,
        family="serif",
        zorder=5,
    )
    fig.text(
        0.5,
        0.30,
        "Data: WhoScored / Opta   ·   Built with Python & Matplotlib   ·   "
        "Analysis by Mostafa Saad",
        ha="center",
        color=TEXT_FADED,
        fontsize=8.5,
        family="monospace",
        zorder=5,
    )

    _page_rail(fig, C_GOLD, label="End of Report")
    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _extra_time_and_pens(events, info):
    """Detect extra time and a penalty shootout from the event periods.

    Returns (went_to_et, pens) where pens is (home_kicks, away_kicks) scored
    in the shootout, or None if the match never reached penalties. ET goals
    are already included in the normal score (they are real is_goal events);
    this only adds the "AET" context and, if needed, the shootout score.
    """
    if events is None or events.empty or "period_code" not in events.columns:
        return False, None
    periods_seen = set(events["period_code"].dropna().astype(str).str.lower().unique())
    # Feed inconsistency: some exports use short codes (et1/etht/et2), others
    # the full WhoScored names (firstperiodofextratime/secondperiodofextratime).
    # Match either by looking for the "extratime" fragment as well.
    went_to_et = bool(periods_seen & {"et1", "etht", "et2"}) or any(
        "extratime" in pc or "extra time" in pc for pc in periods_seen
    )

    has_pso = ("pso" in periods_seen) or ("penaltyshootout" in periods_seen)
    if not has_pso or "is_penalty_shootout" not in events.columns:
        return went_to_et, None
    pso = events[events["is_penalty_shootout"].fillna(False)]
    if pso.empty:
        return went_to_et, None
    # A scored shootout kick is a row of type == "Goal" (WhoScored does NOT set
    # is_goal on shootout kicks — that flag is reserved for in-play goals). Fall
    # back to is_goal only if the type column is missing.
    if "type" in pso.columns:
        scored = pso[pso["type"].astype(str).str.lower() == "goal"]
    else:
        scored = pso[pso.get("is_goal", False).fillna(False)]
    if scored.empty:
        return went_to_et, None
    hid, aid = info.get("home_id"), info.get("away_id")
    # For a scored kick, team_id is the shooter's team; scoring_team agrees.
    side_col = "team_id" if "team_id" in scored.columns else "scoring_team"
    h_pens = int((scored[side_col] == hid).sum())
    a_pens = int((scored[side_col] == aid).sum())
    return True, (h_pens, a_pens)


def _page_rail(fig, accent, *, label="", page_no=None):
    """Consistent professional chrome on a report page: a thin colour rail down
    the left edge (so flipping the report shows which section you're in) and a
    running footer with the report name, an optional section label and a page
    number. Applied to the key narrative pages."""
    rail = fig.add_axes([0.0, 0.0, 0.012, 1.0])
    rail.set_facecolor(accent)
    rail.set_xticks([])
    rail.set_yticks([])
    for s in rail.spines.values():
        s.set_visible(False)
    fig.text(
        0.035,
        0.028,
        "MATCH ANALYSIS REPORT",
        color=TEXT_FADED,
        fontsize=7.5,
        fontweight="bold",
        family="monospace",
    )
    if label:
        fig.text(
            0.5,
            0.028,
            label.upper(),
            ha="center",
            color=TEXT_FADED,
            fontsize=7.5,
            fontweight="bold",
            family="monospace",
        )
    if page_no is not None:
        fig.text(
            0.965,
            0.028,
            f"{page_no:02d}",
            ha="right",
            color=TEXT_FADED,
            fontsize=7.5,
            fontweight="bold",
            family="monospace",
        )


def _match_kpis(events, info, ppda):
    """Headline per-team numbers used by the executive-summary / verdict pages.
    Returns a dict of two-element [home, away] lists."""
    hid, aid = info.get("home_id"), info.get("away_id")
    out = {
        "xg": [0.0, 0.0],
        "xt": [0.0, 0.0],
        "shots": [0, 0],
        "sot": [0, 0],
        "big": [0, 0],
        "poss": [50.0, 50.0],
        "goals": [0, 0],
        "ppda": [None, None],
    }
    try:
        for j, tid in enumerate((hid, aid)):
            te = events[events["team_id"] == tid]
            if "xG" in te.columns:
                out["xg"][j] = round(float(te["xG"].fillna(0).sum()), 2)
            if "xT" in te.columns:
                out["xt"][j] = round(float(te["xT"].fillna(0).sum()), 2)
            if "is_shot" in te.columns:
                sh = te[te["is_shot"].fillna(False) == True]  # noqa: E712
                # exclude own goals (logged on scorer's own team)
                if "is_goal" in sh.columns and "scoring_team" in sh.columns:
                    sh = sh[
                        ~(sh["is_goal"].fillna(False) & (sh["scoring_team"] != tid))
                    ]
                out["shots"][j] = int(len(sh))
                if "big_chance" in sh.columns:
                    out["big"][j] = int(
                        sh["big_chance"].fillna(False).astype(bool).sum()
                    )
            if "shot_whoscored_type" in te.columns:
                out["sot"][j] = int(
                    te["shot_whoscored_type"].isin(["Goal", "SavedShot"]).sum()
                )
        # goals by scoring_team (own goals credited correctly, shootout excluded)
        if "is_goal" in events.columns and "scoring_team" in events.columns:
            g = events[events["is_goal"].fillna(False)]
            if "is_penalty_shootout" in g.columns:
                g = g[~g["is_penalty_shootout"].fillna(False)]
            out["goals"][0] = int((g["scoring_team"] == hid).sum())
            out["goals"][1] = int((g["scoring_team"] == aid).sum())
        if "is_pass" in events.columns:
            ph = int(
                ((events["team_id"] == hid) & (events["is_pass"] == True)).sum()
            )  # noqa: E712
            pa = int(
                ((events["team_id"] == aid) & (events["is_pass"] == True)).sum()
            )  # noqa: E712
            if ph + pa:
                out["poss"] = [
                    round(100 * ph / (ph + pa), 1),
                    round(100 * pa / (ph + pa), 1),
                ]
        out["ppda"] = [
            ppda.get("home", {}).get("ppda"),
            ppda.get("away", {}).get("ppda"),
        ]
    except Exception:
        pass
    return out


def _key_moments(events, info, goals_df):
    """Ordered list of key moments (goals with scorer/type) for the summary."""
    moments = []
    try:
        if goals_df is not None and not goals_df.empty:
            for _, r in goals_df.sort_values("minute").iterrows():
                mn = r.get("minute")
                scorer = str(r.get("scorer") or r.get("player") or "—").split()[-1]
                team = str(r.get("scored_for") or r.get("team") or "")
                cat = str(r.get("category") or r.get("subtype") or "").strip()
                if bool(r.get("is_own_goal", False)) or "own" in cat.lower():
                    tag = "OWN GOAL"
                else:
                    tag = cat or "Goal"
                moments.append((int(mn) if pd.notna(mn) else 0, scorer, team, tag))
    except Exception:
        pass
    return moments


def _draw_executive_summary_page(pdf, info, events, ppda, goals_df):
    """One-page executive summary: the verdict, the six numbers that decided the
    match, the key moments and a two-sentence story — the front page a coach or
    analyst reads first."""
    fig = _new_dark_fig(11.7, 8.27)
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = str(info.get("score") or "? - ?").replace("*", "").strip()
    k = _match_kpis(events, info, ppda)
    went_to_et, pens = _extra_time_and_pens(events, info)

    # ── Header: title + scoreline ──────────────────────────────────────
    fig.text(
        0.06,
        0.945,
        "EXECUTIVE SUMMARY",
        color=C_GOLD,
        fontsize=12,
        fontweight="bold",
        family="monospace",
    )
    fig.text(
        0.06,
        0.90,
        f"{hn}  {score}  {an}",
        color=TEXT_BRIGHT,
        fontsize=27,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)],
    )
    ctx = ""
    if pens is not None:
        winner = hn if pens[0] > pens[1] else an
        hi, lo = max(pens), min(pens)
        ctx = f"After extra time · {winner} won {hi}-{lo} on penalties"
    elif went_to_et:
        ctx = "After extra time"
    if ctx:
        fig.text(0.06, 0.862, ctx, color=C_GOLD, fontsize=11, fontweight="bold")

    # ── Verdict line (who deserved it, on the underlying numbers) ───────
    xh, xa = k["xg"]
    if abs(xh - xa) < 0.15:
        verdict = f"An even contest on chance quality ({xh:.2f} vs {xa:.2f} xG)."
    else:
        better = hn if xh > xa else an
        verdict = (
            f"{better} carried the stronger chance quality "
            f"({max(xh, xa):.2f} vs {min(xh, xa):.2f} xG)."
        )
    gh, ga = k["goals"]
    over = ""
    if gh + ga > 0:
        # note finishing over/under-performance for the side that scored
        if gh - xh >= 0.6 or ga - xa >= 0.6:
            over = " Finishing outran the chances."
        elif (gh and gh - xh <= -0.6) or (ga and ga - xa <= -0.6):
            over = " Chances went begging in front of goal."
    fig.text(
        0.06,
        0.80,
        "VERDICT",
        color=TEXT_DIM,
        fontsize=10,
        fontweight="bold",
        family="monospace",
    )
    fig.text(
        0.06,
        0.755,
        _wrap_text_simple(verdict + over, 82),
        color=TEXT_BRIGHT,
        fontsize=13.5,
        va="top",
        linespacing=1.35,
    )

    # ── KPI cards: xG · xT · Big chances · Shots · Possession · PPDA ────
    def _fmt(v, pct=False, dec=False):
        if v is None:
            return "—"
        return f"{v:.0f}%" if pct else (f"{v:.2f}" if dec else str(v))

    cards = [
        ("xG", _fmt(xh, dec=True), _fmt(xa, dec=True), xh, xa, False),
        (
            "xT",
            _fmt(k["xt"][0], dec=True),
            _fmt(k["xt"][1], dec=True),
            k["xt"][0],
            k["xt"][1],
            False,
        ),
        (
            "Big Chances",
            str(k["big"][0]),
            str(k["big"][1]),
            k["big"][0],
            k["big"][1],
            False,
        ),
        (
            "Shots",
            str(k["shots"][0]),
            str(k["shots"][1]),
            k["shots"][0],
            k["shots"][1],
            False,
        ),
        (
            "Possession",
            _fmt(k["poss"][0], pct=True),
            _fmt(k["poss"][1], pct=True),
            k["poss"][0],
            k["poss"][1],
            False,
        ),
        (
            "PPDA",
            _fmt(k["ppda"][0], dec=True),
            _fmt(k["ppda"][1], dec=True),
            k["ppda"][0],
            k["ppda"][1],
            True,
        ),
    ]
    cw, gap = 0.135, 0.018
    x0 = 0.06
    cy, ch = 0.50, 0.16
    for i, (lbl, hv, av, hh, aa, lower_better) in enumerate(cards):
        cx = x0 + i * (cw + gap)
        ax = fig.add_axes([cx, cy, cw, ch])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (0.02, 0.02),
                0.96,
                0.96,
                boxstyle="round,pad=0.0,rounding_size=0.05",
                facecolor=BG_PANEL,
                edgecolor=GRID_COL,
                lw=1.0,
            )
        )
        ax.text(
            0.5,
            0.86,
            lbl.upper(),
            ha="center",
            color=TEXT_DIM,
            fontsize=8.5,
            fontweight="bold",
            family="monospace",
        )
        # winner highlighted in gold
        try:
            if hh is not None and aa is not None and hh != aa:
                h_win = (hh < aa) if lower_better else (hh > aa)
            else:
                h_win = None
        except TypeError:
            h_win = None
        ax.text(
            0.29,
            0.44,
            hv,
            ha="center",
            color=C_GOLD if h_win is True else TEXT_BRIGHT,
            fontsize=14,
            fontweight="bold",
            family="monospace",
        )
        ax.text(
            0.71,
            0.44,
            av,
            ha="center",
            color=C_GOLD if h_win is False else TEXT_BRIGHT,
            fontsize=14,
            fontweight="bold",
            family="monospace",
        )
        ax.text(
            0.29,
            0.14,
            hn[:9],
            ha="center",
            color=C_HOME,
            fontsize=6.8,
            fontweight="bold",
        )
        ax.text(
            0.71,
            0.14,
            an[:9],
            ha="center",
            color=C_AWAY,
            fontsize=6.8,
            fontweight="bold",
        )

    # ── Key moments ────────────────────────────────────────────────────
    moments = _key_moments(events, info, goals_df)
    fig.text(
        0.06,
        0.40,
        "KEY MOMENTS",
        color=TEXT_DIM,
        fontsize=10,
        fontweight="bold",
        family="monospace",
    )
    my = 0.355
    if moments:
        for mn, scorer, team, tag in moments[:6]:
            col = C_HOME if team == hn else (C_AWAY if team == an else TEXT_BRIGHT)
            fig.text(
                0.06,
                my,
                f"{mn}'",
                color=C_GOLD,
                fontsize=11,
                fontweight="bold",
                family="monospace",
            )
            fig.text(
                0.11, my, scorer, color=TEXT_BRIGHT, fontsize=11, fontweight="bold"
            )
            fig.text(0.27, my, f"{team}", color=col, fontsize=10, fontweight="bold")
            fig.text(0.45, my, tag, color=TEXT_DIM, fontsize=9, style="italic")
            my -= 0.038
    else:
        fig.text(
            0.06,
            my,
            "No goals in normal/extra time.",
            color=TEXT_DIM,
            fontsize=10,
            style="italic",
        )

    # ── The story (momentum) ───────────────────────────────────────────
    poss_leader = hn if k["poss"][0] > k["poss"][1] else an
    xt_leader = hn if k["xt"][0] >= k["xt"][1] else an
    story = (
        f"{poss_leader} saw more of the ball, while {xt_leader} moved it "
        f"into threatening areas more effectively. The following pages break "
        f"down how each side created and denied danger."
    )
    fig.text(
        0.62,
        0.40,
        "THE STORY",
        color=TEXT_DIM,
        fontsize=10,
        fontweight="bold",
        family="monospace",
    )
    fig.text(
        0.62,
        0.355,
        _wrap_text_simple(story, 44),
        color=TEXT_MAIN,
        fontsize=11,
        va="top",
        linespacing=1.55,
    )

    _page_rail(fig, C_GOLD, label="Executive Summary", page_no=2)
    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _draw_verdict_page(pdf, info, events, ppda, goals_df, page_no=None):
    """Closing verdict — a deeper, data-driven tactical conclusion: deserved
    result on xG, finishing efficiency, control vs penetration, the pressing
    battle and how the game was decided, laid out in two analyst columns with a
    numeric match ledger."""
    fig = _new_dark_fig(11.7, 8.27)
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = str(info.get("score") or "? - ?").replace("*", "").strip()
    k = _match_kpis(events, info, ppda)
    went_to_et, pens = _extra_time_and_pens(events, info)
    xh, xa = k["xg"]
    gh, ga = k["goals"]
    bh, ba = k["big"]
    ph, pa = k["poss"]
    xth, xta = k["xt"]
    shh, sha = k["shots"]
    oth, ota = k["sot"]
    pph, ppa = k["ppda"]

    def _fin_word(diff):
        if diff >= 0.4:
            return "a clinical, above-expectation return"
        if diff >= 0.15:
            return "slightly better than the chances warranted"
        if diff <= -0.4:
            return "a wasteful afternoon in front of goal"
        if diff <= -0.15:
            return "a shade below what the chances merited"
        return "a return in line with the chance quality"

    fig.text(
        0.06,
        0.935,
        "THE VERDICT",
        color=C_GOLD,
        fontsize=12,
        fontweight="bold",
        family="monospace",
    )
    fig.text(
        0.06,
        0.875,
        f"{hn}  {score}  {an}",
        color=TEXT_BRIGHT,
        fontsize=25,
        fontweight="bold",
    )

    # ── Deserved result ──
    if abs(xh - xa) < 0.20:
        deserved = (
            f"Expected goals finished near-level ({xh:.2f} vs {xa:.2f}). On the "
            f"balance of chances neither side did enough to claim the result was "
            f"theirs by right — this was a game decided in the fine margins."
        )
    else:
        better = hn if xh > xa else an
        deserved = (
            f"{better} generated the better body of chances ({max(xh, xa):.2f} vs "
            f"{min(xh, xa):.2f} xG) and, on the run of play, did more to earn the "
            f"result."
        )
    who_big = hn if bh > ba else (an if ba > bh else None)
    if who_big:
        deserved += f" {who_big} also fashioned the clearer openings ({max(bh,ba)} vs {min(bh,ba)} big chances)."
    if pens is not None:
        pen_win = hn if pens[0] > pens[1] else an
        deserved += (
            f" With normal play level, {pen_win} held their nerve from the spot."
        )

    # ── Finishing efficiency ──
    finishing = (
        f"{hn} took {shh} shots ({oth} on target) and scored {gh} from {xh:.2f} xG — "
        f"{_fin_word(gh - xh)}. {an} managed {ga} from {xa:.2f} xG off {sha} shots "
        f"({ota} on target) — {_fin_word(ga - xa)}."
    )

    # ── Control vs penetration ──
    poss_leader = hn if ph > pa else an
    xt_leader = hn if xth >= xta else an
    if poss_leader == xt_leader:
        control = (
            f"{poss_leader} controlled the ball ({max(ph,pa):.0f}%) and converted that "
            f"control into territory and threat (xT {max(xth,xta):.2f} vs {min(xth,xta):.2f}) — "
            f"dominance that actually reached the opponent's goal."
        )
    else:
        control = (
            f"{poss_leader} saw more of the ball ({max(ph,pa):.0f}%), but it was "
            f"{xt_leader} who moved possession into dangerous areas more efficiently "
            f"(xT {max(xth,xta):.2f} vs {min(xth,xta):.2f}) — possession without penetration "
            f"for the side on top."
        )

    # ── Pressing battle ──
    if pph is not None and ppa is not None:
        press_leader = hn if pph < ppa else an
        press = (
            f"Out of possession, {press_leader} pressed more aggressively "
            f"(PPDA {min(pph,ppa):.1f} vs {max(pph,ppa):.1f}), engaging earlier and "
            f"forcing the play back sooner."
        )
    else:
        press = (
            f"Both sides picked their pressing moments rather than committing to a "
            f"sustained high press."
        )

    # ── How it was decided ──
    moments = _key_moments(events, info, goals_df)
    if pens is not None:
        decided = (
            "Level after 120 minutes, the tie was ultimately settled on penalties."
        )
    elif gh == ga:
        decided = (
            f"The sides could not be separated, the scoreline reflecting how evenly the "
            f"key phases were shared."
        )
    else:
        win = hn if gh > ga else an
        margin = abs(gh - ga)
        first = moments[0] if moments else None
        decided = f"{win} won by a {margin}-goal margin"
        if first:
            decided += f", the game turning on the {first[0]}' goal ({first[1]}, {first[3].lower()})"
        decided += "."

    # ── Layout: two columns ──
    left = [("DESERVED RESULT", deserved), ("FINISHING", finishing)]
    right = [
        ("CONTROL vs PENETRATION", control),
        ("THE PRESSING BATTLE", press),
        ("HOW IT WAS DECIDED", decided),
    ]

    def _col(items, x, y0):
        y = y0
        for head, body in items:
            fig.text(
                x,
                y,
                head,
                color=C_GOLD,
                fontsize=10,
                fontweight="bold",
                family="monospace",
            )
            fig.text(
                x,
                y - 0.038,
                _wrap_text_simple(body, 52),
                color=TEXT_BRIGHT,
                fontsize=11.5,
                va="top",
                linespacing=1.42,
            )
            n = len(_wrap_text_simple(body, 52).split("\n"))
            y -= 0.052 + n * 0.033

    _col(left, 0.06, 0.76)
    _col(right, 0.545, 0.76)

    # ── Numeric ledger strip ──
    led_y = 0.135
    fig.add_artist(
        mpatches.Rectangle(
            (0.06, led_y - 0.01),
            0.88,
            0.075,
            transform=fig.transFigure,
            facecolor=BG_PANEL,
            edgecolor=GRID_COL,
            lw=1.0,
            zorder=1,
        )
    )
    cols = [
        ("xG", f"{xh:.2f}", f"{xa:.2f}"),
        ("SHOTS", str(shh), str(sha)),
        ("ON TARGET", str(oth), str(ota)),
        ("BIG CH.", str(bh), str(ba)),
        ("POSS %", f"{ph:.0f}", f"{pa:.0f}"),
        ("xT", f"{xth:.2f}", f"{xta:.2f}"),
        ("PPDA", f"{pph:.1f}" if pph else "—", f"{ppa:.1f}" if ppa else "—"),
    ]
    fig.text(
        0.075,
        led_y + 0.045,
        hn[:14],
        color=C_HOME,
        fontsize=8.5,
        fontweight="bold",
        family="monospace",
    )
    fig.text(
        0.075,
        led_y + 0.012,
        an[:14],
        color=C_AWAY,
        fontsize=8.5,
        fontweight="bold",
        family="monospace",
    )
    n = len(cols)
    x0, xw = 0.24, 0.70
    for i, (lbl, hv, av) in enumerate(cols):
        cx = x0 + (i + 0.5) * (xw / n)
        fig.text(
            cx,
            led_y + 0.052,
            lbl,
            ha="center",
            color=TEXT_DIM,
            fontsize=7.5,
            fontweight="bold",
            family="monospace",
        )
        fig.text(
            cx,
            led_y + 0.030,
            hv,
            ha="center",
            color=TEXT_BRIGHT,
            fontsize=11,
            fontweight="bold",
            family="monospace",
        )
        fig.text(
            cx,
            led_y + 0.008,
            av,
            ha="center",
            color=TEXT_BRIGHT,
            fontsize=11,
            fontweight="bold",
            family="monospace",
        )

    fig.text(
        0.06,
        0.065,
        "End of report · Mostafa Saad",
        color=TEXT_DIM,
        fontsize=9.5,
        family="serif",
    )
    _page_rail(fig, C_GOLD, label="Verdict", page_no=page_no)
    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _wrap_text_simple(text, width):
    import textwrap

    return "\n".join(textwrap.wrap(str(text), width=width)) or str(text)


def _draw_glance_page(pdf, info, events, ppda, goals_df):
    """One-screen 'Match at a Glance' dashboard: score + home/away split bars
    for xG, shots, on-target and possession."""
    fig = _new_dark_fig(11.7, 8.27)
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "? - ?"
    hid, aid = info.get("home_id"), info.get("away_id")

    sh = [0, 0]
    sot = [0, 0]
    xg = [0.0, 0.0]
    poss = [50.0, 50.0]
    try:
        for j, tid in enumerate((hid, aid)):
            te = events[events["team_id"] == tid]
            if "is_shot" in te.columns:
                sh[j] = int((te["is_shot"] == True).sum())
            if "xG" in te.columns:
                xg[j] = float(te["xG"].fillna(0).sum())
            if "shot_whoscored_type" in te.columns:
                sot[j] = int(
                    te["shot_whoscored_type"].isin(["Goal", "SavedShot"]).sum()
                )
        if "is_pass" in events.columns:
            ph = int(((events["team_id"] == hid) & (events["is_pass"] == True)).sum())
            pa = int(((events["team_id"] == aid) & (events["is_pass"] == True)).sum())
            if ph + pa:
                poss = [round(100 * ph / (ph + pa), 1), round(100 * pa / (ph + pa), 1)]
    except Exception:
        pass

    fig.text(
        0.5,
        0.93,
        "MATCH AT A GLANCE",
        ha="center",
        color=TEXT_BRIGHT,
        fontsize=22,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)],
    )
    fig.text(0.27, 0.85, hn, ha="right", color=C_HOME, fontsize=16, fontweight="bold")
    fig.text(
        0.50,
        0.85,
        score,
        ha="center",
        color=TEXT_BRIGHT,
        fontsize=26,
        fontweight="bold",
    )
    fig.text(0.73, 0.85, an, ha="left", color=C_AWAY, fontsize=16, fontweight="bold")

    # Extra time / penalty shootout context — a match that went past 90
    # minutes should say so, and a shootout result belongs right under the
    # normal score, not buried in the stats.
    went_to_et, pens = _extra_time_and_pens(events, info)
    if pens is not None:
        fig.text(
            0.5,
            0.815,
            f"AET · ({pens[0]} - {pens[1]} pens)",
            ha="center",
            color=C_GOLD,
            fontsize=11,
            fontweight="bold",
        )
    elif went_to_et:
        fig.text(
            0.5,
            0.815,
            "AET (After Extra Time)",
            ha="center",
            color=C_GOLD,
            fontsize=11,
            fontweight="bold",
        )

    rows = [
        ("xG", f"{xg[0]:.2f}", xg[0], xg[1], f"{xg[1]:.2f}"),
        ("Shots", str(sh[0]), sh[0], sh[1], str(sh[1])),
        ("On Target", str(sot[0]), sot[0], sot[1], str(sot[1])),
        ("Possession %", f"{poss[0]:.0f}%", poss[0], poss[1], f"{poss[1]:.0f}%"),
    ]
    y0 = 0.66
    rh = 0.135
    for i, (lbl, lv, hvv, avv, rv) in enumerate(rows):
        y = y0 - i * rh
        fig.text(
            0.5,
            y + 0.052,
            lbl,
            ha="center",
            color=TEXT_DIM,
            fontsize=11,
            fontweight="bold",
        )
        frac = (hvv / (hvv + avv)) if (hvv + avv) else 0.5
        ax = fig.add_axes([0.20, y, 0.60, 0.032])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.barh(0.5, frac, height=1.0, color=C_HOME, align="center")
        ax.barh(0.5, 1 - frac, left=frac, height=1.0, color=C_AWAY, align="center")
        fig.text(
            0.18,
            y + 0.016,
            lv,
            ha="right",
            va="center",
            color=C_HOME,
            fontsize=12,
            fontweight="bold",
        )
        fig.text(
            0.82,
            y + 0.016,
            rv,
            ha="left",
            va="center",
            color=C_AWAY,
            fontsize=12,
            fontweight="bold",
        )

    _page_rail(fig, C_GOLD, label="Match at a Glance", page_no=3)
    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _draw_toc_page(pdf, entries):
    """Contents page: list of (title, page_number, accent_color)."""
    fig = _new_dark_fig(11.7, 8.27)
    fig.text(
        0.08,
        0.90,
        "CONTENTS",
        color=C_GOLD,
        fontsize=28,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)],
    )
    fig.text(
        0.08,
        0.855,
        "Match Analysis Report",
        color=TEXT_DIM,
        fontsize=12,
        style="italic",
    )
    y = 0.76
    for title, page, col in entries:
        fig.text(0.10, y, "●", color=col, fontsize=10, va="center")
        fig.text(0.135, y, title, color=TEXT_BRIGHT, fontsize=13.5, va="center")
        fig.text(
            0.90,
            y,
            str(page),
            color=TEXT_DIM,
            fontsize=12,
            va="center",
            ha="right",
            family="monospace",
        )
        fig.text(
            0.50, y, "." * 70, color=TEXT_FADED, fontsize=8, va="center", ha="center"
        )
        y -= 0.062
    _page_rail(fig, C_GOLD, label="Contents", page_no=4)
    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _draw_glossary_page(pdf):
    """Glossary & methodology — definitions of the analytics terms used."""
    import textwrap as _tw

    fig = _new_dark_fig(11.7, 8.27)
    fig.text(
        0.08,
        0.92,
        "GLOSSARY & METHODOLOGY",
        color=C_GOLD,
        fontsize=23,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)],
    )
    terms = [
        (
            "xG — Expected Goals",
            "Probability a shot becomes a goal (0–1), from location, angle, body part and assist type. Summed, it measures a team's chance quality.",
        ),
        (
            "xGoT — Expected Goals on Target",
            "xG recomputed from on-target shot placement — measures finishing quality after contact.",
        ),
        (
            "xT — Expected Threat",
            "Value added by moving the ball into more dangerous zones through passes and carries, before any shot is taken.",
        ),
        ("Big Chance", "A shot from a clear scoring situation — typically xG ≥ 0.30."),
        (
            "PPDA — Passes per Defensive Action",
            "Opponent passes allowed per defensive action; a lower number means more intense pressing.",
        ),
        (
            "Box Entry",
            "Ball entering the opponent's penalty area by pass or carry — a key penetration measure.",
        ),
        (
            "Zone 14",
            "Central zone just outside the box; the prime pre-assist / creative area.",
        ),
        (
            "Progressive Pass",
            "A pass that moves the ball significantly closer to goal, usually breaking a defensive line.",
        ),
        (
            "Pitch markers",
            "On shape maps, starters are drawn as circles and substitutes (bench players) as squares; node size = touches/passes.",
        ),
    ]
    y = 0.84
    for t, desc in terms:
        fig.text(0.08, y, t, color=TEXT_BRIGHT, fontsize=12.5, fontweight="bold")
        wrapped = "\n".join(_tw.wrap(desc, width=108))
        fig.text(
            0.08,
            y - 0.026,
            wrapped,
            color=TEXT_DIM,
            fontsize=10,
            va="top",
            linespacing=1.35,
        )
        y -= 0.026 + 0.022 * (wrapped.count("\n") + 1) + 0.020
    _page_rail(fig, C_GOLD, label="Glossary & Methodology")
    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _add_pdf_bookmarks(pdf_path, bookmarks):
    """Add a clickable outline (bookmarks) to a finished PDF via pypdf."""
    try:
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        writer.append(reader)
        n = len(reader.pages)
        for title, idx in bookmarks:
            if 0 <= idx < n:
                writer.add_outline_item(str(title), idx)
        tmp = pdf_path + ".tmp"
        with open(tmp, "wb") as fh:
            writer.write(fh)
        os.replace(tmp, pdf_path)
    except Exception:
        pass


_COVER_BG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "cover_bg_clean.png"
)


def _draw_match_summary_page(pdf, info, goals_df, ppda, events=None):
    """Cover / hero page: a faded tactical-analytics artwork background with the
    scoreline, result hook and an xG · Shots · Possession strip overlaid in a
    neutral gold/white palette (no team-colour split, no inline charts)."""
    import matplotlib.image as _mpimg

    fig = _new_dark_fig(11.7, 8.27)
    home = info.get("home_name") or "Home"
    away = info.get("away_name") or "Away"
    score = str(info.get("score") or "? - ?").replace("*", "").strip()
    venue = str(info.get("venue") or "").strip()
    comp = str(info.get("competition") or "").strip()
    # Match kick-off date (never the report-generation date). Empty if the feed
    # didn't provide one — we simply omit it rather than show today's date.
    mdate = str(info.get("date") or "").strip()

    # ── Faded tactical artwork background (aspect kept, centred) + veil ──
    try:
        img = _mpimg.imread(_COVER_BG_PATH)
        bgimg = fig.add_axes([0.146, 0.0, 0.708, 1.0])
        bgimg.set_axis_off()
        bgimg.imshow(img, alpha=0.42, zorder=0, aspect="auto")
    except Exception:
        pass
    veil = fig.add_axes([0, 0, 1, 1])
    veil.set_axis_off()
    veil.set_xlim(0, 1)
    veil.set_ylim(0, 1)
    veil.add_patch(
        mpatches.Rectangle((0, 0), 1, 1, facecolor=BG_DARK, alpha=0.30, lw=0, zorder=1)
    )
    # gold poster frame + corner brackets
    veil.add_patch(
        mpatches.Rectangle(
            (0.02, 0.026),
            0.96,
            0.948,
            facecolor="none",
            edgecolor=C_GOLD,
            lw=1.0,
            alpha=0.4,
            zorder=2,
        )
    )
    for x, y, dx, dy in [
        (0.035, 0.958, 1, -1),
        (0.965, 0.958, -1, -1),
        (0.035, 0.042, 1, 1),
        (0.965, 0.042, -1, 1),
    ]:
        veil.plot(
            [x, x + dx * 0.028], [y, y], color=C_GOLD, lw=1.8, alpha=0.8, zorder=2
        )
        veil.plot([x, x], [y, y + dy * 0.04], color=C_GOLD, lw=1.8, alpha=0.8, zorder=2)

    # ── Eyebrow ─────────────────────────────────────────────────────────
    fig.text(
        0.5,
        0.925,
        comp.upper() or "MATCH ANALYSIS REPORT",
        ha="center",
        color=C_GOLD,
        fontsize=12.5,
        fontweight="bold",
        family="monospace",
        zorder=5,
    )
    fig.text(
        0.5,
        0.893,
        " · ".join([b for b in [venue, mdate] if b]),
        ha="center",
        color=TEXT_DIM,
        fontsize=10.5,
        style="italic",
        zorder=5,
    )

    # ── Hero score + team names (neutral white, gold accents) ──────────
    _st = dict(zorder=5, path_effects=[pe.withStroke(linewidth=5, foreground=BG_DARK)])
    fig.text(
        0.5,
        0.70,
        score,
        ha="center",
        color=TEXT_BRIGHT,
        fontsize=72,
        fontweight="bold",
        va="center",
        **_st,
    )
    _stn = dict(zorder=5, path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)])
    fig.text(
        0.30,
        0.70,
        home.upper(),
        ha="right",
        color=TEXT_BRIGHT,
        fontsize=19,
        fontweight="bold",
        va="center",
        **_stn,
    )
    fig.text(
        0.70,
        0.70,
        away.upper(),
        ha="left",
        color=TEXT_BRIGHT,
        fontsize=19,
        fontweight="bold",
        va="center",
        **_stn,
    )

    went_to_et, pens = _extra_time_and_pens(events, info)
    k = _match_kpis(events, info, ppda) if events is not None else None
    if pens is not None:
        winner = home if pens[0] > pens[1] else away
        hi, lo = max(pens), min(pens)
        hook = f"{winner} edge {home if winner == away else away} {hi}-{lo} on penalties (AET)"
        wx = 0.30 if winner == home else 0.70
        fig.text(
            wx,
            0.648,
            "★ WINNERS",
            ha=("right" if winner == home else "left"),
            color=C_GOLD,
            fontsize=9,
            fontweight="bold",
            family="monospace",
            zorder=5,
        )
    elif went_to_et:
        hook = "After extra time"
    elif k and k["goals"][0] != k["goals"][1]:
        w = home if k["goals"][0] > k["goals"][1] else away
        hook = f"{w} take the win"
    else:
        hook = "Honours even"
    fig.text(
        0.5,
        0.60,
        hook,
        ha="center",
        color=C_GOLD,
        fontsize=13.5,
        fontweight="bold",
        style="italic",
        zorder=5,
    )

    # ── xG · Shots · Possession strip (gold leader, no colour bg) ──────
    if k is not None:
        strip = [
            ("xG", f"{k['xg'][0]:.2f}", f"{k['xg'][1]:.2f}", k["xg"][0], k["xg"][1]),
            (
                "Shots",
                str(k["shots"][0]),
                str(k["shots"][1]),
                k["shots"][0],
                k["shots"][1],
            ),
            (
                "Possession",
                f"{k['poss'][0]:.0f}%",
                f"{k['poss'][1]:.0f}%",
                k["poss"][0],
                k["poss"][1],
            ),
        ]
        cw, gap = 0.235, 0.03
        total = len(strip) * cw + (len(strip) - 1) * gap
        x0 = (1 - total) / 2
        for i, (lbl, hv, av, hh, aa) in enumerate(strip):
            cx = x0 + i * (cw + gap)
            ax = fig.add_axes([cx, 0.20, cw, 0.15], zorder=5)
            ax.set_axis_off()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (0.02, 0.02),
                    0.96,
                    0.96,
                    boxstyle="round,pad=0.0,rounding_size=0.05",
                    facecolor=BG_DARK,
                    edgecolor=C_GOLD,
                    lw=1.0,
                    alpha=0.92,
                )
            )
            ax.text(
                0.5,
                0.80,
                lbl.upper(),
                ha="center",
                color=TEXT_DIM,
                fontsize=9,
                fontweight="bold",
                family="monospace",
            )
            h_win = hh > aa if hh != aa else None
            ax.text(
                0.30,
                0.40,
                hv,
                ha="center",
                color=C_GOLD if h_win is True else TEXT_BRIGHT,
                fontsize=19,
                fontweight="bold",
                family="monospace",
            )
            ax.text(
                0.70,
                0.40,
                av,
                ha="center",
                color=C_GOLD if h_win is False else TEXT_BRIGHT,
                fontsize=19,
                fontweight="bold",
                family="monospace",
            )
            ax.text(
                0.30,
                0.13,
                home[:3].upper(),
                ha="center",
                color=TEXT_DIM,
                fontsize=7,
                fontweight="bold",
                family="monospace",
            )
            ax.text(
                0.70,
                0.13,
                away[:3].upper(),
                ha="center",
                color=TEXT_DIM,
                fontsize=7,
                fontweight="bold",
                family="monospace",
            )

    # ── Formations + contents teaser ───────────────────────────────────
    fig.text(
        0.5,
        0.145,
        f"Formations   —   {home}: {_na(info.get('home_form'))}    "
        f"|    {away}: {_na(info.get('away_form'))}",
        ha="center",
        color=TEXT_DIM,
        fontsize=10,
        zorder=5,
    )
    fig.text(
        0.5,
        0.075,
        "Executive Summary · The Match Story · Chance Creation · "
        "Build-up · Defence · The Verdict",
        ha="center",
        color=TEXT_FADED,
        fontsize=8.5,
        style="italic",
        zorder=5,
    )

    _page_rail(fig, C_GOLD, label="Cover", page_no=1)
    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _draw_goals_log_page(pdf, goals_df, info):
    """Goals log with the unified visual identity."""
    fig = _new_dark_fig(14, 9)
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "—"
    home_col = info.get("home_color") or info.get("HOME_COLOR") or C_HOME
    away_col = info.get("away_color") or info.get("AWAY_COLOR") or C_AWAY
    home_text_col = _team_label_color(home_col, BG_PANEL)
    away_text_col = _team_label_color(away_col, BG_PANEL)
    if apply_unified_frame is not None:
        apply_unified_frame(
            fig,
            section="GOALS LOG",
            title=f"Goals Log — {hn} vs {an}",
            subtitle="Every goal with scorer, assist, goal category and "
            "shot xG · own goals shown in magenta",
            accent=C_GOLD,
            home_name=hn,
            away_name=an,
            score=str(score),
            footer_note="Open Play (green) · Set Piece (gold) · Own Goal " "(magenta)",
        )
    else:
        fig.text(
            0.04, 0.94, "GOALS LOG", color=TEXT_BRIGHT, fontsize=20, fontweight="bold"
        )
        fig.text(
            0.04,
            0.91,
            "Scorer  ·  Assist  ·  Goal type",
            color=TEXT_DIM,
            fontsize=10,
            style="italic",
        )

    if goals_df.empty:
        fig.text(
            0.5,
            0.5,
            "No goals recorded.",
            ha="center",
            color=TEXT_DIM,
            fontsize=14,
            style="italic",
        )
        pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
        plt.close(fig)
        return

    headers = [
        ("MIN", 0.05, "center"),
        ("TEAM", 0.13, "left"),
        ("SCORER", 0.29, "left"),
        ("ASSIST", 0.49, "left"),
        ("CATEGORY", 0.68, "left"),
        ("DETAIL", 0.80, "left"),
        ("BODY", 0.91, "left"),
        ("xG", 0.98, "right"),
    ]
    ax = fig.add_axes([0.0, 0.05, 1.0, 0.83])
    ax.set_facecolor(BG_DARK)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    y = 0.94
    for lbl, x, ha in headers:
        ax.text(
            x,
            y,
            lbl,
            ha=ha,
            va="center",
            color=TEXT_DIM,
            fontsize=9,
            fontweight="bold",
            transform=ax.transAxes,
        )
    y -= 0.018
    ax.plot([0.03, 0.99], [y, y], color=GRID_COL, lw=0.8, transform=ax.transAxes)

    n = len(goals_df)
    row_h = min(0.78 / max(n, 1), 0.07)
    y -= 0.012

    for _, r in goals_df.iterrows():
        is_og = bool(r.get("is_own_goal", False))
        team_id = r.get("scoring_team_id")
        col = (
            OG_COLOR
            if is_og
            else (C_HOME if team_id == info.get("home_id") else C_AWAY)
        )

        bg = "#1a0a0a" if col == C_HOME else ("#090909" if col == C_AWAY else "#1e0a2e")
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (0.02, y - row_h * 0.92),
                0.96,
                row_h * 0.85,
                boxstyle="round,pad=0.005,rounding_size=0.005",
                facecolor=bg,
                edgecolor=col,
                lw=0.8,
                alpha=0.92,
                transform=ax.transAxes,
            )
        )
        cy = y - row_h * 0.5

        ax.text(
            0.05,
            cy,
            f"{_safe_int(r['minute'])}'",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.30", facecolor=col, edgecolor="none"),
        )

        ax.text(
            0.13,
            cy,
            _short_name(str(r["team"]), 16),
            ha="left",
            va="center",
            color=col,
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )
        scorer_label = _short_name(str(r["scorer"]), 20)
        if is_og:
            scorer_label += "  (OG)"
        ax.text(
            0.29,
            cy,
            scorer_label,
            ha="left",
            va="center",
            color=TEXT_BRIGHT,
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.49,
            cy,
            _short_name(str(r["assist"]), 18),
            ha="left",
            va="center",
            color=TEXT_DIM,
            fontsize=9.5,
            transform=ax.transAxes,
        )

        cat_col = C_GREEN if r["category"] == "Open Play" else C_GOLD
        ax.text(
            0.68,
            cy,
            r["category"],
            ha="left",
            va="center",
            color=cat_col,
            fontsize=9.5,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.80,
            cy,
            _short_name(str(r["subtype"]), 14),
            ha="left",
            va="center",
            color=TEXT_MAIN,
            fontsize=9.5,
            transform=ax.transAxes,
        )
        ax.text(
            0.91,
            cy,
            _short_name(str(r.get("body_part", "Unknown")), 12),
            ha="left",
            va="center",
            color=TEXT_MAIN,
            fontsize=9.5,
            transform=ax.transAxes,
        )
        xg_txt = (
            f"{r['xG']:.2f}" if isinstance(r["xG"], (int, float)) and r["xG"] else "—"
        )
        ax.text(
            0.98,
            cy,
            xg_txt,
            ha="right",
            va="center",
            color=_ui_text(C_GOLD),
            fontsize=9.5,
            transform=ax.transAxes,
        )

        y -= row_h

    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _draw_ppda_page(pdf, ppda, info, visuals_dir):
    """Create the PPDA page without saving a PNG; Dark.py creates figure 40."""
    fig = draw_ppda_gauge(ppda, info, save_path=None)
    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _filename_team_side(fname: str) -> str | None:
    """Return home/away from visual filenames such as *_home.png or *_away_01.png."""
    base = os.path.basename(str(fname or "")).lower()
    if re.search(r"(^|_)home($|[_.-])", base):
        return "home"
    if re.search(r"(^|_)away($|[_.-])", base):
        return "away"
    return None


def _shape_read(events, tid):
    """A qualitative read of a team's overall SHAPE from average touch positions
    and passing spread — height up the pitch, vertical compactness, width, and
    which channel the play leaned through. Not a formation number; a description
    of how the side actually set up across the match."""
    try:
        d = events[
            (events["team_id"] == tid) & events["x"].notna() & events["y"].notna()
        ]
        if len(d) < 40:
            return ""
        mx = float(d["x"].mean())
        sx = float(d["x"].std())
        sy = float(d["y"].std())
        yv = d["y"]
        left = float((yv >= 66.6).mean())
        right = float((yv < 33.4).mean())
        height = (
            "high up the pitch"
            if mx >= 55
            else "deep in its own half" if mx <= 44 else "around the middle third"
        )
        comp = (
            "vertically compact between its lines"
            if sx <= 17
            else (
                "stretched vertically"
                if sx >= 25
                else "moderately spread from back to front"
            )
        )
        width = "narrow" if sy <= 22 else "wide" if sy >= 30 else "balanced in width"
        if abs(left - right) < 0.06:
            flank = "circulating fairly evenly across the pitch"
        elif left > right:
            flank = "leaning noticeably to the left"
        else:
            flank = "leaning noticeably to the right"
        return f"averaged its touches {height}, {comp} and {width}, {flank}"
    except Exception:
        return ""


def _infer_formation(events, tid):
    """Approximate outfield formation string (e.g. '4-3-3') from players' mean
    on-ball x positions. Uses the most-involved XI, drops the deepest node as the
    keeper, and bins the ten outfielders into defence / midfield / attack by pitch
    thirds. It is a coarse estimate from average touch position, not a lineup feed."""
    try:
        d = events[(events["team_id"] == tid) & events["x"].notna()]
        if d.empty:
            return ""
        agg = d.groupby("player").agg(x=("x", "mean"), n=("x", "size"))
        agg = agg[agg["n"] >= 6].sort_values("n", ascending=False).head(11)
        if len(agg) < 7:
            return ""
        gk = agg["x"].idxmin()  # deepest node ≈ goalkeeper
        of = agg.drop(index=gk).sort_values("x")
        xs = of["x"].to_numpy()
        lo, hi = float(xs.min()), float(xs.max())
        rng = max(hi - lo, 1e-6)
        # relative thirds of the team's own outfield spread — adapts to a side
        # camped high or pinned deep, so full-backs and forwards land in the
        # right band instead of bloating midfield.
        d_cut = lo + rng / 3.0
        a_cut = hi - rng / 3.0
        d_n = int((xs < d_cut).sum())
        a_n = int((xs >= a_cut).sum())
        m_n = int(len(xs) - d_n - a_n)
        # Average touch position skews in lopsided games (a dominant side pushes
        # everyone high, a deep side's counter-attackers spike forward), which
        # produces nonsense like 3-2-5. Only report a genuinely realistic outfield
        # shape; otherwise stay silent rather than print a misleading number.
        if len(xs) != 10:
            return ""
        if not (3 <= d_n <= 5 and 2 <= m_n <= 5 and 1 <= a_n <= 3 and d_n >= a_n):
            return ""
        return f"{d_n}-{m_n}-{a_n}"
    except Exception:
        return ""


def _ctx_for(events, info, ppda):
    """Compute the per-match tactical numbers the analyst commentary reads from.
    Returns a dict of two-element [home, away] lists plus names. Robust to gaps."""
    try:
        k = _match_kpis(events, info, ppda)
    except Exception:
        k = {}
    out = dict(k) if isinstance(k, dict) else {}
    hid, aid = info.get("home_id"), info.get("away_id")
    passes = [0, 0]
    passpct = [0, 0]
    box = [0, 0]
    topxt = [("", 0.0), ("", 0.0)]
    xgsh = [0.0, 0.0]
    otp = [0, 0]
    topcreator = [("", 0), ("", 0)]
    topshooter = [("", 0, 0.0), ("", 0, 0.0)]
    topdef = [("", 0), ("", 0)]
    clr = [0, 0]
    tkl = [0, 0]
    intc = [0, 0]
    rec = [0, 0]
    keyp = [0, 0]

    def _last(x):
        return str(x).split()[-1] if x and str(x) != "nan" else ""

    try:
        for j, tid in enumerate((hid, aid)):
            te = events[events["team_id"] == tid]
            ty = te.get("type", pd.Series("", index=te.index)).astype(str)
            oc = te.get("outcome", pd.Series("", index=te.index)).astype(str)
            isp = (
                te.get("is_pass", pd.Series(False, index=te.index)).fillna(False)
                == True
            )
            tot = int(isp.sum())
            comp = int((isp & (oc == "Successful")).sum())
            passes[j] = tot
            passpct[j] = round(100 * comp / max(tot, 1))
            if {"end_x", "end_y", "x", "y"}.issubset(te.columns):
                pp = te[isp & (oc == "Successful")]
                inbox = (pp["end_x"] >= 83) & (pp["end_y"].between(21, 79))
                outside = ~((pp["x"] >= 83) & (pp["y"].between(21, 79)))
                box[j] = int((inbox & outside).sum())
            if "xT" in te.columns:
                pxt = te[isp & (oc == "Successful") & (te["xT"].fillna(0) > 0)]
                if len(pxt):
                    gg = pxt.groupby("player")["xT"].sum().sort_values(ascending=False)
                    topxt[j] = (_last(gg.index[0]), round(float(gg.iloc[0]), 2))
            # creator (key passes), shooter, defender, defensive counts
            kp = (
                te.get("is_key_pass", pd.Series(False, index=te.index)).fillna(False)
                == True
            )
            keyp[j] = int(kp.sum())
            if kp.sum():
                gc = te[kp].groupby("player").size().sort_values(ascending=False)
                topcreator[j] = (_last(gc.index[0]), int(gc.iloc[0]))
            issh = (
                te.get("is_shot", pd.Series(False, index=te.index)).fillna(False)
                == True
            )
            if issh.sum():
                sg = (
                    te[issh]
                    .groupby("player")
                    .agg(sh=("is_shot", "size"), xg=("xG", "sum"))
                    if "xG" in te
                    else None
                )
                if sg is not None and len(sg):
                    sg = sg.sort_values("xg", ascending=False)
                    topshooter[j] = (
                        _last(sg.index[0]),
                        int(sg.iloc[0]["sh"]),
                        round(float(sg.iloc[0]["xg"]), 2),
                    )
            defmask = ty.isin(
                [
                    "Tackle",
                    "Interception",
                    "BallRecovery",
                    "Clearance",
                    "Block",
                    "BlockedPass",
                ]
            )
            if defmask.sum():
                dg = te[defmask].groupby("player").size().sort_values(ascending=False)
                topdef[j] = (_last(dg.index[0]), int(dg.iloc[0]))
            clr[j] = int((ty == "Clearance").sum())
            tkl[j] = int((ty == "Tackle").sum())
            intc[j] = int((ty == "Interception").sum())
            rec[j] = int((ty == "BallRecovery").sum())
            sh = (out.get("shots") or [0, 0])[j]
            xg = (out.get("xg") or [0, 0])[j]
            sot = (out.get("sot") or [0, 0])[j]
            xgsh[j] = round(xg / sh, 2) if sh else 0.0
            otp[j] = round(100 * sot / max(sh, 1))
    except Exception:
        pass
    formation = ["", ""]
    shape = ["", ""]
    try:
        formation = [_infer_formation(events, hid), _infer_formation(events, aid)]
        shape = [_shape_read(events, hid), _shape_read(events, aid)]
    except Exception:
        pass
    out.update(
        {
            "passes": passes,
            "passpct": passpct,
            "box": box,
            "topxt": topxt,
            "xgsh": xgsh,
            "otp": otp,
            "topcreator": topcreator,
            "topshooter": topshooter,
            "topdef": topdef,
            "clr": clr,
            "tkl": tkl,
            "intc": intc,
            "rec": rec,
            "keyp": keyp,
            "formation": formation,
            "shape": shape,
            "hn": info.get("home_name") or "Home",
            "an": info.get("away_name") or "Away",
        }
    )
    return out


def _analyst_commentary(fname, hn, an, ctx):
    """Data-driven, connected tactical commentary in the analyst voice used for
    the written match threads — reads THIS match's numbers and interprets the
    mechanism, rather than describing what the chart type is. Returns
    (heading, body) or (None, None) when no data-aware version applies."""
    f = (fname or "").lower()
    g = ctx or {}

    def col(key, default=(0, 0)):
        v = g.get(key)
        return v if isinstance(v, (list, tuple)) and len(v) >= 2 else list(default)

    gh, ga = col("goals")
    xgh, xga = col("xg")
    shh, sha = col("shots")
    oth, ota = col("sot")
    bh, ba = col("big")
    ph, pa = col("poss")
    xth, xta = col("xt")
    pph, ppa = col("ppda")
    pah, paa = col("passes")
    pcth, pcta = col("passpct")
    bxh, bxa = col("box")
    xsh_h, xsh_a = col("xgsh")
    oph, opa = col("otp")
    txh = (g.get("topxt") or [("", 0)])[0]
    txa = (g.get("topxt") or [("", 0), ("", 0)])[1]

    side = _filename_team_side(fname)
    if side == "home":
        i, team, opp = 0, hn, an
    elif side == "away":
        i, team, opp = 1, an, hn
    else:
        i, team, opp = None, None, None

    def s(pair):  # side value / opponent value
        return (pair[i], pair[1 - i]) if i is not None else (pair[0], pair[1])

    def leader(hv, av):
        return hn if hv > av else (an if av > hv else None)

    tc_l = g.get("topcreator") or [("", 0), ("", 0)]
    tsh_l = g.get("topshooter") or [("", 0, 0.0), ("", 0, 0.0)]
    td_l = g.get("topdef") or [("", 0), ("", 0)]
    tx_l = g.get("topxt") or [("", 0.0), ("", 0.0)]
    clr = col("clr")
    tkl = col("tkl")
    intc = col("intc")
    rec = col("rec")
    keyp = col("keyp")

    def named(pair):  # side entity / opponent entity
        return (pair[i], pair[1 - i]) if i is not None else (pair[0], pair[1])

    # ── The match story / xG flow ──
    if "xg_flow" in f:
        lead = leader(xgh, xga)
        real = leader(gh, ga)
        sh_l = tsh_l[0] if (xgh >= xga) else tsh_l[1]
        who = (
            f"{lead} shaped the stronger body of chances ({max(xgh,xga):.2f} to {min(xgh,xga):.2f} xG)"
            if lead
            else f"the expected-goals split finished level ({xgh:.2f} to {xga:.2f})"
        )
        star = (
            f", led at the sharp end by {sh_l[0]} ({sh_l[1]} shots, {sh_l[2]:.2f} xG)"
            if sh_l and sh_l[0]
            else ""
        )
        if lead and real and lead != real:
            flip = (
                f" Yet {real} carried the lead — a scoreline running ahead of the play, which almost always means a "
                f"handful of moments finished ruthlessly or a lead defended rather than extended."
            )
        elif lead and real and lead == real:
            flip = f" And {real} took the points too — the result and the run of play agreed."
        else:
            flip = ""
        return (
            "How the danger actually accumulated",
            f"Over the ninety minutes {who}{star}. The shape of the climb matters as much as the total: a steep, "
            f"early stretch is a side front-loading its threat and then managing the game, while value that arrives "
            f"only in isolated late steps is a team chasing rather than controlling.{flip} Where one curve pulls "
            f"clear and stays clear the performance was sustained; where it rises in a couple of jumps, the danger "
            f"came from a few situations rather than steady pressure.",
        )

    # ── Shot map (side) ──
    if "shot_map" in f and i is not None:
        sh, osh = s([shh, sha])
        xg, oxg = s([xgh, xga])
        big, obig = s([bh, ba])
        xps, _ = s([xsh_h, xsh_a])
        otpc, _ = s([oph, opa])
        shooter, _ = named(tsh_l)
        lead_by = (
            f"{shooter[0]} the focal point ({shooter[1]} shots, {shooter[2]:.2f} xG)"
            if shooter and shooter[0]
            else "no single dominant shooter"
        )
        verdict = (
            "worked the ball into clean, central positions"
            if xps >= 0.13
            else "was pushed onto lower-value efforts from distance or tight angles against a set defence"
        )
        return (
            f"How {team} manufactured its shots",
            f"{team}'s {sh} shots were worth {xg:.2f} xG — about {xps:.2f} per attempt, {otpc}% on target — with "
            f"{lead_by}. That per-shot number is the real tell: at {xps:.2f} it says {team} {verdict}. Set against "
            f"{opp}'s {oxg:.2f} xG and {obig} big chance{'s' if obig != 1 else ''}, the {big} big chance"
            f"{'s' if big != 1 else ''} here show whether the volume became genuine quality or simply arrived in "
            f"front of a compact block — reaching the edge of the area and shooting is not the same as getting "
            f"behind the line.",
        )

    # ── Pass network / build-up (side) ──
    if "pass_network" in f and i is not None:
        pas, _ = s([pah, paa])
        pct, _ = s([pcth, pcta])
        pos, _ = s([ph, pa])
        carrier, _ = named(tx_l)
        shape_l = g.get("shape") or ["", ""]
        shp = shape_l[i] if i is not None and i < len(shape_l) else ""
        shape = f" Across the match {team} {shp}. " if shp else " "
        hub = (
            f"{carrier[0]} its most threatening distributor ({carrier[1]:.2f} xT)"
            if carrier and carrier[0]
            else "no single dominant distributor"
        )
        return (
            f"How {team} built and circulated",
            f"{team} played {pas} passes at {pct}% with {pos:.0f}% of the ball, and the network's shape says how that "
            f"possession was used rather than merely how much there was — with {hub} at the heart of it.{shape}A low, wide "
            f"structure whose heaviest links sit between the defenders and the deepest midfielder is circulation in "
            f"front of the opponent; links climbing centrally toward the forwards are possession that actually "
            f"progressed. The node heights and the strongest partnerships mark the axis — left, right or straight "
            f"through the middle — {team} chose to advance through.",
        )

    # ── xT map / progression (side) ──
    if ("xt_map" in f or "progressive" in f) and i is not None:
        xt, oxt = s([xth, xta])
        carrier, _ = named(tx_l)
        share = (
            f", {carrier[0]} carrying the most of it ({carrier[1]:.2f})"
            if carrier and carrier[0]
            else ""
        )
        eff = (
            "far more" if oxt and xt > oxt * 1.4 else ("more" if xt >= oxt else "less")
        )
        conc = (
            "a team leaning on one progressor to unlock the block"
            if (carrier and carrier[1] and xt and carrier[1] > 0.2 * xt)
            else "threat shared across several carriers"
        )
        return (
            f"Who moved {team} into danger",
            f"{team} generated {xt:.2f} in Expected Threat, {eff} than {opp}'s {oxt:.2f}, and the map shows where it "
            f"came from{share} — {conc}. What matters is not just how much a side progressed but how: the gold arrows "
            f"mark the passes that broke lines, and their origin and angle tell you which channel carried the danger "
            f"and whether it was worked patiently or delivered direct. Progression is only half the job, though — the "
            f"next question is whether all this territory became shots, or died at the edge of the block.",
        )

    # ── Danger creation / box entries / zone14 (side) ──
    if any(x in f for x in ("danger", "box_entries", "zone14")) and i is not None:
        box_, _ = s([bxh, bxa])
        big, obig = s([bh, ba])
        kp, _ = s(keyp)
        creator, _ = named(tc_l)
        lead_by = (
            f"{creator[0]} leading the creation ({creator[1]} key passes)"
            if creator and creator[0]
            else "no single dominant creator"
        )
        return (
            f"How {team} turned territory into openings",
            f"With {lead_by}, {team} produced {kp} key pass{'es' if kp != 1 else ''} and {big} big chance"
            f"{'s' if big != 1 else ''} from its entries into the area. Reaching the box and hurting a defence inside "
            f"it are different competencies: repeated entries that convert into few big chances mean the final ball "
            f"or the cut-back kept breaking down against numbers inside, while entries that become big chances mean "
            f"the free receiver behind the block was found. Set against {opp}'s {obig}, this is the line between "
            f"manufacturing genuine danger and simply arriving at the edge of it.",
        )

    # ── Defensive heatmap (side) ──
    if "defensive" in f and i is not None:
        cl, _ = s(clr)
        it, _ = s(intc)
        tk, _ = s(tkl)
        defender, _ = named(td_l)
        led = (
            f"{defender[0]} carrying the load ({defender[1]} actions)"
            if defender and defender[0]
            else "the workload spread across the unit"
        )
        nature = (
            "a block clearing its lines rather than winning the ball to keep it — and against a counter-pressing "
            "opponent, clearing without retaining tends to invite the very next wave"
            if cl >= max(it + tk, 1)
            else "a side winning the ball to keep it rather than simply clearing, the sign of front-foot, proactive defending"
        )
        return (
            f"How {team} defended, and where",
            f"{team}'s defending, with {led}, breaks down as {cl} clearance{'s' if cl != 1 else ''}, {it} "
            f"interception{'s' if it != 1 else ''} and {tk} tackle{'s' if tk != 1 else ''} — and the balance is the "
            f"message. That composition points to {nature}. Read it with where the heat concentrates: dense, deep "
            f"zones in front of goal are a low block absorbing pressure; higher, more spread activity is defending "
            f"in {opp}'s half.",
        )

    # ── Pressing / PPDA / high turnovers ──
    if "ppda" in f or "high_turnover" in f or "turnover" in f:
        press = leader(-(pph or 99), -(ppa or 99)) if (pph and ppa) else None
        pline = (
            (
                f"{press} pressed the more aggressively (PPDA {min(pph,ppa):.1f} to {max(pph,ppa):.1f}), engaging after "
                f"fewer opponent passes. "
            )
            if (pph and ppa and press)
            else ""
        )
        return (
            "The pressing battle, and what it produced",
            f"{pline}But intensity is only half the read. A high press that still concedes clean chances is being "
            f"played through, not failing to engage; a higher PPDA can be a deliberate mid-block. The tangible test "
            f"is the reward — winning the ball high only matters if the regain becomes a shot. Regains that die in a "
            f"static shape mean the ball was recovered without runners to attack the moment; regains that turn "
            f"straight into efforts are a working rest-attack. That distinction is often the whole difference between "
            f"a press that merely suppresses the opponent and one that actually scores from them.",
        )

    # ── Territory / possession / touches ──
    if any(x in f for x in ("territor", "possession", "ball_touches", "dominating")):
        lead = leader(ph, pa)
        xlead = leader(xth, xta)
        pl = (
            (f"{lead} owned the ball ({max(ph,pa):.0f}% to {min(ph,pa):.0f}%)")
            if lead
            else f"possession finished near-level ({ph:.0f}% to {pa:.0f}%)"
        )
        tie = (
            f" And it translated: {xlead} also led the ball-progression value ({max(xth,xta):.2f} to {min(xth,xta):.2f} xT)."
            if (lead and xlead and lead == xlead)
            else (
                f" But it did not fully translate — {xlead} generated the greater threat ({max(xth,xta):.2f} to "
                f"{min(xth,xta):.2f} xT), a sign of possession without penetration for the side on top."
                if (lead and xlead and lead != xlead)
                else ""
            )
        )
        return (
            "Who owned the space, and where it mattered",
            f"{pl}, but ownership only counts where it happens. Dominating the attacking and middle thirds is "
            f"pressure applied; dominating your own defensive third is a team pinned back.{tie} The map separates "
            f"control that reached the opponent's goal from control that merely held the ball in safe areas.",
        )

    # ── Overall match statistics ──
    if "match_stats" in f:
        xlead = leader(xgh, xga)
        plead = leader(ph, pa)
        tension = (
            "point the same way — a reliable, one-directional story"
            if (xlead and plead and xlead == plead)
            else "pull apart: possession one way, chance quality the other — the classic signature of control without "
            "penetration, the ball and the space owned by a team that could not turn either into the cleaner openings"
        )
        return (
            "The control panel — reading the numbers together",
            f"No single row decides a match; the read comes from how they align. Chance quality ({xgh:.2f} vs "
            f"{xga:.2f} xG), big chances ({bh} vs {ba}), possession ({ph:.0f}% vs {pa:.0f}%), ball-progression "
            f"({xth:.2f} vs {xta:.2f} xT) and pressing intensity (PPDA {pph or 0:.1f} vs {ppa or 0:.1f}) here "
            f"{tension}. That is where a scoreline is either confirmed by the underlying play or exposed as running "
            f"ahead of it.",
        )

    # ── xT per minute (momentum) ──
    if "xt_per_minute" in f or "xt_minute" in f:
        return (
            "Where the momentum actually swung",
            "Threat per minute is the rhythm track of the match: sustained blocks of value mean a team repeatedly "
            "arriving in shooting positions and forcing the opponent to defend phases rather than isolated attacks. "
            "Flat stretches after a strong spell are just as telling — they mark the moment control stopped turning "
            "into chances. Line the peaks up against the goals and substitutions and the game's turning points "
            "usually announce themselves.",
        )

    return (None, None)


def _commentary_for_filename(fname: str, hn: str, an: str):
    """
    Look up tactical commentary for a saved-fig filename. Returns
    (heading, body). Falls back to a neutral message if no match.
    """
    f = (fname or "").lower()

    def _t(side):
        return hn if side == "home" else an

    # Identify side ("home"/"away"/"shared") from filename
    detected_side = _filename_team_side(fname)
    if detected_side == "home":
        side = "home"
        team = hn
    elif detected_side == "away":
        side = "away"
        team = an
    else:
        side, team = "shared", None

    if "xg_flow" in f:
        return (
            "When did the chances actually come?",
            f"The xG Flow plots cumulative Expected Goals minute by minute "
            f"as a staircase, with every step marking a shot whose height "
            f"equals its chance quality. Stars sit on goals; the shaded "
            f"territory under each curve is total chance creation. A team "
            f"that pulls clearly above the other built the stronger shot "
            f"profile, even if the scoreline says otherwise.",
        )
    if "shot_map" in f:
        return (
            f"Where did {team} create its shots?",
            f"Each dot is a single shot, located at where it was struck. "
            f"Marker size scales with xG, so big circles are big chances. "
            f"Filled circles are goals, hollow ones are misses. Cluster "
            f"density inside the box shows where {team} manufactured looks; "
            f"wide-area dots usually indicate speculative efforts that "
            f"rarely trouble keepers.",
        )
    if "breakdown_goals" in f:
        return (
            "How did the shots break down?",
            "The bar group counts every shot by outcome — total, woodwork, "
            "on target, off target, blocked. On-target conversion (goals "
            "divided by shots on target) reflects finishing efficiency. "
            "The Goals & Assists table below records the scorer, the play "
            "type (Open Play / Set Piece / Penalty), the assister and the "
            "chance's xG.",
        )
    if "pass_network" in f:
        return (
            f"How did {team} build its play?",
            f"Each node is a player placed at their average pass position. "
            f"Node size scales with passes attempted; line width between "
            f"two players scales with the volume of completed passes "
            f"between them. Heavy lines reveal {team}'s preferred "
            f"partnerships and which axis the build-up flowed through. "
            f"Top-8 partnerships also carry their pass count for quick "
            f"reading.",
        )
    if "xt_map" in f:
        return (
            f"Where did {team} generate threat?",
            f"The grid colours each pitch zone by its xT (Expected Threat) "
            f"value — the probability that owning the ball there leads to "
            f"a shot in the next few seconds. White arrows are positive-xT "
            f"passes (gained threat); red arrows are negative-xT passes "
            f"(gave threat back). The five gold arrows highlight {team}'s "
            f"five highest-xT progressive passes — usually the moments "
            f"that broke a defensive line.",
        )
    if "shot_comparison" in f:
        return (
            "Who was the more dangerous side?",
            "A side-by-side bar chart of the headline shooting numbers — "
            "total shots, on target, big chances, xG, xGoT. The fastest "
            "single view to answer: who was the more dangerous attacking "
            "side? Gold-coloured numbers mark the metric leader.",
        )
    if "danger" in f and "home" in f or "danger" in f and "away" in f or "danger_" in f:
        return (
            f"How did {team} create danger?",
            f"Every action that ended in a shot, key pass or box entry "
            f"is plotted to show which channels generated {team}'s "
            f"high-value moments. Concentrate on the warm zones — those "
            f"are {team}'s true danger lanes. Diamonds are key passes; "
            f"circles are shots; faint arrows are box entries underneath.",
        )
    if "gk_saves" in f:
        return (
            "How hard did the keepers work?",
            "Plots the location of every shot each keeper faced, with "
            "marker size scaled to the chance's xG. Filled stars are goals "
            "conceded; rings are saves. Save quality is a function of the "
            "xG of the shots faced — saving a high-xG strike beats "
            "stopping easy long-range efforts.",
        )
    if "xg_tiles" in f or "xg_summary" in f:
        return (
            "How clinical was the finishing?",
            "xG measures pre-shot chance quality; xGoT (Expected Goals on "
            "Target) measures post-shot quality once placement and power "
            "are known. A team with xGoT well above xG executed their "
            "finishing better than average; below xG points to wasteful "
            "conversion.",
        )
    if "zone14" in f:
        return (
            f"Did {team} reach the dangerous central zones?",
            f"Zone 14 is the central pocket just outside the box — "
            f"historically the richest zone for chance creation. The "
            f"half-spaces flank it. This map counts every action {team} "
            f"completed in those zones; volume here is a proxy for how "
            f"often they reached the most dangerous central real estate.",
        )
    if "match_stats" in f:
        return (
            "What do the headline numbers say?",
            "A consolidated head-to-head: Attack (goals, shots, passes, "
            "key passes), Defense (tackles, interceptions, blocks, "
            "clearances, recoveries, fouls) and Pressing (PPDA mini-dials "
            "with intensity verdict). Read it as the single-page summary "
            "of the entire match.",
        )
    if "territorial" in f or "possession" in f or "ball_touches" in f:
        return (
            "Who controlled the territory?",
            "Splits the pitch into zones and reads which side had more "
            "touches in each. Red dominance signals home-team control of "
            "that area; gold signals away. The donut totals beneath show "
            "the overall share of touches — whoever owned the territory "
            "owned the game.",
        )
    if "pass_thirds" in f:
        return (
            f"Where did {team} play its passes?",
            f"Splits passes into defensive, middle and attacking-third "
            f"buckets and visualises each on the pitch. Final-third pass "
            f"density and completion rate are the two quickest reads of "
            f"{team}'s break-down activity in the opposition area.",
        )
    if "xt_per_minute" in f:
        return (
            "When did each side surge in threat?",
            "A diverging bar chart: home xT bars rise above the zero line, "
            "away xT drops below it. Tall spikes mark the moments each "
            "side surged in threat creation. The 5-minute rolling average "
            "overlay smooths the noise and shows momentum windows.",
        )
    if "progressive" in f:
        return (
            f"How did {team} move the ball forward?",
            f"Plots every pass that closed at least 25% of the distance "
            f"to goal (or any pass into the final third). The five gold "
            f"arrows highlight {team}'s top forward gains by raw distance. "
            f"Wide-spread arrows mean the progression load was shared; "
            f"concentration around one source marks a single play-maker.",
        )
    if "crosses" in f:
        return (
            f"How did {team} attack from wide?",
            f"Every cross into the box plotted with origin and end point. "
            f"Solid arrows are successful, faded ones are unsuccessful. "
            f"Cross volume and the flank split (left vs right) reveal "
            f"{team}'s wide-attack channel and how much of {team}'s box "
            f"entry came through wide play versus central combinations.",
        )
    if "defensive_hm" in f:
        return (
            f"Where did {team} do its defending?",
            f"A density map of every defensive action {team} completed — "
            f"tackles, interceptions, clearances, blocks, recoveries, "
            f"fouls. Hot zones reveal {team}'s defensive line height: "
            f"high up the pitch (press) or deep in their own block.",
        )
    if "defensive_summary" in f:
        return (
            "Who did more defensive work?",
            "Six defensive-action types — head-to-head counts. Tackles "
            "and interceptions describe ground duels; blocks count shots "
            "stopped by a body in the way; recoveries and clearances mark "
            "how each side escaped pressure; fouls show where containment "
            "broke down. Gold labels mark the side leading on each metric.",
        )
    if "avg_position" in f:
        return (
            f"What shape did {team} hold?",
            f"Each player placed at their mean touch position, with node "
            f"size scaled to total touches. Faint lines connect every "
            f"node to the team centroid so the overall shape pops out: a "
            f"high, narrow shape signals an aggressive pressing block; a "
            f"deep, wide shape points to a low-block defensive setup.",
        )
    if "dominating_zone" in f:
        return (
            "Which side owned which zones?",
            "Each grid cell is coloured by the team that had more touches "
            "there. Block colour reveals territorial dominance at a glance "
            "— large contiguous regions in one team's colour mark areas "
            "they controlled outright. Touch counts on every cell make "
            "the heatmap quantitative.",
        )
    if "box_entries" in f:
        return (
            f"How did {team} get into the box?",
            f"Every successful entry into the opposition penalty area — "
            f"pass (gold) or carry (green). Clusters near the byline mean "
            f"wide-and-cut-back access; clusters at the D mean central, "
            f"through-ball access. The total count is a direct measure "
            f"of {team}'s break-down volume.",
        )
    if "high_turnovers" in f:
        return (
            f"Did {team} win the ball high up the pitch?",
            f"Marks every regain of possession in the final 40 metres. "
            f"Frequent high turnovers are a hallmark of a successful "
            f"press — the more concentrated the dots near the opposition "
            f"box, the more often {team} won the ball in dangerous areas.",
        )
    if "pass_target" in f:
        return (
            f"Where did {team}'s passes land?",
            f"Heatmap of where {team}'s successful passes landed — i.e. "
            f"their preferred receiving zones. Compare with the pass "
            f"network to see whether the receiving pattern matches the "
            f"network's shape, or whether passes were aiming for "
            f"under-served outlets.",
        )
    if "ppda" in f:
        return (
            "How aggressively did they press?",
            "PPDA (Passes per Defensive Action) measures pressing "
            "intensity. Low PPDA = aggressive press (fewer opponent "
            "passes allowed before forcing a defensive action). High PPDA "
            "= deeper block. The dial reads green for an aggressive "
            "press and slides toward orange for a low-block setup.",
        )
    if "player_stats" in f:
        return (
            f"How did the players rate?",
            f"Per-player totals across the match — minutes, touches, "
            f"shots, passes attempted/completed, key passes, defensive "
            f"actions, and a colour-coded performance rating. Starters "
            f"appear first, substitutes follow.",
        )
    return (
        "What does this visual add?",
        "This chart adds context to the tactical story. Read it alongside "
        "the rest of the report — chance quality, territory, pressing "
        "and ball progression all reinforce the same narrative when the "
        "numbers line up.",
    )


def _professional_tactical_commentary(
    fname: str, heading: str, body: str, hn: str, an: str
) -> str:
    """Turn a chart description into a fuller human tactical read."""
    f = (fname or "").lower()
    detected_side = _filename_team_side(fname)
    if detected_side == "home":
        team = hn
        opponent = an
    elif detected_side == "away":
        team = an
        opponent = hn
    else:
        team = "the stronger side"
        opponent = "the opponent"

    def _join(*parts: str) -> str:
        return "\n\n".join(p.strip() for p in parts if p and p.strip())

    opening = body
    if "xg_flow" in f:
        return _join(
            opening,
            "Tactically, this is the match rhythm page. A steady climb usually means sustained pressure: the team is repeatedly arriving in shooting positions, forcing the opponent to defend phases rather than isolated attacks. A flat line after a strong spell is just as important because it shows when control stopped turning into chances.",
            f"The key coaching read is the timing of the separations. If {hn} or {an} created distance early, the game state may have allowed them to manage risk afterwards. If the curve changes late, it points to substitutions, fatigue, or a structural adjustment that finally opened access to goal.",
        )
    if "shot_map" in f:
        return _join(
            opening,
            f"For {team}, the quality of the shot locations matters more than the count. Central shots inside the box suggest the attack is breaking the defensive line or finding cut-backs. Shots from wide or long range usually mean {opponent} protected the middle and forced lower-value decisions.",
            "The professional read is whether the team created repeatable chances. One large dot can come from a single transition; several good locations from similar zones suggest a deliberate route to goal. That is the difference between attacking noise and a real chance-creation pattern.",
        )
    if "breakdown_goals" in f:
        return _join(
            opening,
            "This page separates volume from efficiency. A team can shoot often without attacking well if most attempts are blocked or off target. By contrast, fewer shots with a high on-target share usually point to cleaner entries, better final actions, and calmer finishing decisions.",
            "The goals and assists table gives the human layer of the story: who finished, who supplied the final pass, and whether the goal came from open play, a set piece, or a penalty. That makes it easier to distinguish a repeatable attacking mechanism from a one-off event.",
        )
    if "pass_network" in f:
        return _join(
            opening,
            f"For {team}, the network shows the build-up skeleton. A strong triangle around centre-back, pivot and full-back usually means controlled circulation. A heavy line into one wide player shows a clear outlet. If the striker or advanced midfielders are disconnected, possession may have looked stable without really threatening {opponent}.",
            "Substitutes are useful here because they show how the structure changed after the starting shape broke. A late player appearing high and wide may indicate a chase phase; a substitute close to the midfield line may show an attempt to regain control.",
        )
    if "xt_map" in f:
        return _join(
            opening,
            f"xT is the best bridge between possession and danger. For {team}, the important question is not simply how many passes were completed, but whether those passes moved the ball into zones that changed {opponent}'s defensive problem.",
            "Look for repeated arrows into the half-space, the box edge, or the far-side channel. Those actions normally force the back line to turn, narrow, or step out. When the highest-threat arrows come from deep or wide zones, it often reveals the team's main progression weapon.",
        )
    if "shot_comparison" in f or "xg_tiles" in f or "xg_summary" in f:
        return _join(
            opening,
            "This is the efficiency check on the attacking story. The side leading shots did not necessarily create the better game; the side leading xG usually created the clearer chances. xGoT then tells us whether the finishing improved or reduced those chances after contact.",
            "A gap between xG and goals should be read carefully. It can indicate elite finishing, poor finishing, goalkeeping impact, or simply a small sample. The value of this page is that it tells you where to look next: shot map for locations, xG flow for timing, and goals table for the final actions.",
        )
    if "danger" in f:
        return _join(
            opening,
            f"This is the best page for reading where {team}'s attacks actually hurt {opponent}. Warm zones near the box or half-spaces suggest a clean attacking route. Warmth stuck near the touchline can still be useful, but it often means the next action had to be excellent to become a real chance.",
            "The coaching point is repeatability. If shots, key passes and entries all come from the same lane, the team found a reliable pattern. If they are scattered, the attack may have depended more on individual actions than on a stable structure.",
        )
    if "gk_saves" in f:
        return _join(
            opening,
            "This page gives context to the scoreline. A goalkeeper with many saves from low-value shots may simply have done the routine work. A goalkeeper saving one or two high-xG chances has changed the game state.",
            "The tactical read is also defensive: if most shots faced came from central close-range areas, the defensive block allowed access to premium zones. If the saves came from distance or angles, the outfield structure protected the most valuable space.",
        )
    if "zone14" in f:
        return _join(
            opening,
            f"Zone 14 and the half-spaces are where possession becomes creative pressure. When {team} receives or combines there, {opponent}'s centre-backs and midfield line are forced to make decisions: step out, hold shape, or pass runners on.",
            "High volume in these lanes usually explains why a side looked dangerous even before the final shot. Low volume suggests the opponent closed the central door and pushed play toward safer wide areas.",
        )
    if "match_stats" in f:
        return _join(
            opening,
            "This is the report's control panel. The attacking rows explain output, the defensive rows explain resistance, and the PPDA view explains how aggressively each side tried to win the ball back.",
            "Read the categories together rather than separately. High passes with low box threat can mean sterile possession. High defensive actions with low possession can mean a team spent too long reacting. The strongest performances usually connect territory, pressure and chance quality.",
        )
    if (
        "territorial" in f
        or "possession" in f
        or "ball_touches" in f
        or "dominating_zone" in f
    ):
        return _join(
            opening,
            "Territory is not the same as possession, but it tells us where the match was played. A team controlling advanced zones forced the opponent to defend closer to goal. A team with touches mostly in its own half may have had the ball without changing the opponent's shape.",
            "The tactical value comes from connecting this page to chance creation. Territorial control is meaningful when it feeds entries, key passes and shots. If the territorial map looks dominant but the shot pages do not, the opponent probably defended the box well.",
        )
    if "pass_thirds" in f:
        return _join(
            opening,
            f"For {team}, this page shows whether possession travelled through the pitch or got stuck. Defensive-third volume is not a problem by itself; it becomes a problem when the middle and attacking-third numbers do not grow from it.",
            "The best read is the balance between middle-third circulation and final-third penetration. A strong team normally has enough middle-third security to progress, then enough final-third quality to turn that progression into pressure.",
        )
    if "xt_per_minute" in f:
        return _join(
            opening,
            "This is the momentum page. Short spikes show individual threat moments; longer clusters show sustained tactical pressure. A team that repeatedly creates spikes after regains is probably dangerous in transition, while a team building smoother waves is likely progressing through possession.",
            "Use the timing to understand coaching interventions. A change after half-time or substitutions can reveal whether the structure improved access to dangerous zones or whether the opponent lost control through fatigue.",
        )
    if "progressive" in f:
        return _join(
            opening,
            f"Progressive passes tell us who advanced the game for {team}. Vertical arrows through the middle usually break lines directly. Diagonal arrows from centre-back to winger or full-back can be just as valuable because they move the defensive block sideways before the next action.",
            "When progression comes from multiple players, the team is harder to press. When it depends on one player, the opponent has a clear target for adjustment.",
        )
    if "crosses" in f:
        return _join(
            opening,
            f"Crosses show the final shape of {team}'s wide attacks. Byline crosses and cut-backs usually indicate penetration behind the full-back. Early crosses suggest the team reached wide areas but could not always enter the box with combinations.",
            "The important detail is not just volume but delivery context. Crosses with runners in the box are a plan; crosses under pressure into a set defence are often a symptom of blocked central access.",
        )
    if "defensive_hm" in f or "defensive_summary" in f:
        return _join(
            opening,
            f"This is the defensive personality of the match. For {team}, actions high up the pitch point to pressing and counter-pressing; actions around the box point to protection, recovery defending and emergency defending.",
            "Blocks and clearances should be read as pressure indicators. They can show commitment, but they can also show that the team spent long spells defending its own penalty area. Tackles and interceptions higher up usually suggest cleaner control.",
        )
    if "avg_position" in f:
        return _join(
            opening,
            f"Average positions give the structural picture of {team}'s match. A compact shape helps counter-press and combine. A stretched shape can create width, but it may also leave the midfield exposed if possession is lost.",
            "Substitutes matter because they reveal the second game state. Late average positions can show whether the team protected a lead, chased the game, or changed the route of attack.",
        )
    if "box_entries" in f:
        return _join(
            opening,
            f"Box entries are one of the cleanest attacking indicators for {team}. They show whether the team actually breached {opponent}'s penalty-area shell, not just whether it had the ball around it.",
            "The entry type matters. Carries often mean a player beat pressure; passes often mean the structure created a free receiver. A healthy attack usually has both.",
        )
    if "high_turnovers" in f:
        return _join(
            opening,
            f"High turnovers measure how much pressure {team} turned into immediate attacking opportunity. Winning the ball high compresses the distance to goal and often catches {opponent} before their defensive shape is rebuilt.",
            "The best high-turnover sides do not just regain possession; they convert the regain into a shot, key pass or box entry quickly. If the regain count is high but chance quality is low, the counter-press worked but the next action lacked clarity.",
        )
    if "pass_target" in f:
        return _join(
            opening,
            f"Pass target zones show where {team} wanted the next receiver to be. This is different from pass origin: it tells us the intended destination of possession and therefore the spaces the team believed were available.",
            "If targets collect between the lines, the team found pockets. If they collect wide and deep, the opponent probably blocked central access. The strongest attacking maps usually combine wide outlets with central receiving zones.",
        )
    if "ppda" in f:
        return _join(
            opening,
            "PPDA should be read as behaviour, not just a number. A low PPDA means the team allowed few passes before engaging; that usually reflects a higher line, stronger counter-press, or a deliberate plan to trap the opponent.",
            "A higher PPDA is not automatically poor. It can reflect a controlled mid-block or game-state management. The key is whether the deeper approach still protected the box and limited high-quality shots.",
        )
    if "player_stats" in f:
        return _join(
            opening,
            "This page gives the individual layer underneath the team story. Minutes and touches explain involvement; passing and key passes explain influence on possession; defensive actions explain workload without the ball.",
            "Use it to identify roles rather than just praise totals. A full-back with heavy touches may have been the outlet. A midfielder with fewer touches but high key passes may have been the connector. Substitutes show how the match plan changed late.",
        )
    return _join(
        opening,
        "The tactical value of this visual is in how it connects with the pages around it. One chart rarely tells the whole match; the stronger read comes when the same theme appears in chance quality, territory, passing direction and defensive pressure.",
        f"When those layers point in the same direction, the match story becomes reliable: which side controlled space, which side created the cleaner chances, and which team forced {opponent} to play in uncomfortable areas.",
    )


def _pdf_page_with_commentary(pdf, fig, heading: str, body: str):
    """
    Compose one reference-style PDF page: portrait page, visual on top,
    explanatory report text underneath, and no legacy commentary card.
    """
    import io as _io
    import textwrap as _tw

    is_disk_visual = isinstance(fig, (str, os.PathLike))
    if is_disk_visual:
        from PIL import Image as _Image

        with _Image.open(os.fspath(fig)) as source_image:
            source_image = source_image.convert("RGB")
            source_image.thumbnail((1800, 1200), _Image.Resampling.LANCZOS)
            img = np.asarray(source_image).copy()
    else:
        buf = _io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=PDF_VISUAL_DPI,
            bbox_inches="tight",
            facecolor=BG_DARK,
        )
        buf.seek(0)
        try:
            from PIL import Image as _Image

            with _Image.open(buf) as img_arr:
                img = np.asarray(img_arr).copy()
        except Exception:
            # Fallback: matplotlib-only path
            import matplotlib.image as _mpimg

            buf.seek(0)
            img = _mpimg.imread(buf, format="png")

    img_h, img_w = img.shape[:2]
    aspect = img_w / max(img_h, 1)

    page_w, page_h = 8.27, 11.69
    new_fig = plt.figure(figsize=(page_w, page_h), facecolor=BG_DARK)
    new_fig.patch.set_facecolor(BG_DARK)
    if _neon_backdrop is not None:
        try:
            _neon_backdrop(new_fig)
        except Exception:
            pass

    is_light_theme = str(BG_DARK).upper() in {"#FFFFFF", "WHITE"}
    subtitle_color = TEXT_DIM
    visual_bg = "#FFFFFF" if is_light_theme else BG_DARK

    new_fig.text(
        0.07,
        0.945,
        "Tactical Commentary",
        ha="left",
        va="center",
        color=TEXT_BRIGHT,
        fontsize=18,
        fontweight="bold",
        family="serif",
    )
    line_ax = new_fig.add_axes((0.07, 0.922, 0.86, 0.002))
    line_ax.set_facecolor(C_GOLD)
    line_ax.set_xticks([])
    line_ax.set_yticks([])
    for s in line_ax.spines.values():
        s.set_visible(False)

    # Large visual: span most of the upper page, aspect preserved in INCHES
    # (not page-fractions) so landscape charts are not squashed or shrunk.
    frame_x, frame_y, frame_w, frame_h = 0.05, 0.42, 0.90, 0.48
    W_in, H_in = frame_w * page_w, frame_h * page_h
    if aspect >= (W_in / H_in):
        w_in = W_in
        h_in = W_in / aspect
    else:
        h_in = H_in
        w_in = H_in * aspect
    draw_w, draw_h = w_in / page_w, h_in / page_h
    draw_x = frame_x + (frame_w - draw_w) / 2
    draw_y = frame_y + (frame_h - draw_h) / 2

    ax_img = new_fig.add_axes((draw_x, draw_y, draw_w, draw_h))
    ax_img.set_facecolor(visual_bg)
    ax_img.imshow(img, aspect="auto")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for s in ax_img.spines.values():
        s.set_visible(False)

    ax_txt = new_fig.add_axes((0.07, 0.085, 0.86, 0.315))
    ax_txt.set_facecolor(BG_DARK)
    ax_txt.set_xticks([])
    ax_txt.set_yticks([])
    for s in ax_txt.spines.values():
        s.set_visible(False)
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.text(
        0.0,
        0.98,
        heading,
        ha="left",
        va="top",
        color=TEXT_BRIGHT,
        fontsize=14,
        fontweight="bold",
        family="serif",
        transform=ax_txt.transAxes,
    )
    wrapped = "\n\n".join(
        _tw.fill(p.strip(), width=100) for p in str(body).split("\n\n") if p.strip()
    )
    ax_txt.text(
        0.0,
        0.83,
        wrapped,
        ha="left",
        va="top",
        color=TEXT_MAIN,
        fontsize=8.9,
        family="serif",
        transform=ax_txt.transAxes,
        linespacing=1.20,
    )
    new_fig.text(
        0.07,
        0.055,
        "Match Analysis Report",
        ha="left",
        va="center",
        color=subtitle_color,
        fontsize=8.5,
        family="sans-serif",
    )
    new_fig.text(
        0.93,
        0.055,
        "What does this visual add?",
        ha="right",
        va="center",
        color=subtitle_color,
        fontsize=8.5,
        family="sans-serif",
    )

    pdf.savefig(new_fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(new_fig)
    if not is_disk_visual:
        plt.close(fig)


def _draw_visual_commentary(pdf, heading: str, body: str):
    """
    Companion 'What does this visual add?' page placed right after each tactical
    visual. Carries the unified visual identity (yellow accent bars + footer)
    so the explanation feels part of the same report.
    """
    fig = _new_dark_fig(14, 9)
    if apply_unified_frame is not None:
        apply_unified_frame(
            fig,
            section="READING THIS VISUAL",
            title=heading,
            subtitle="Tactical commentary — what the chart on the previous "
            "page is telling you",
            accent=C_GOLD,
            footer_note="Commentary is generated from the metric definitions "
            "applied to the event stream",
        )
    # Body text panel
    ax = fig.add_axes([0.06, 0.20, 0.88, 0.62])
    ax.set_facecolor(BG_PANEL)
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL)
        s.set_linewidth(0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Soft accent stripe at the top of the panel
    ax.add_patch(
        mpatches.Rectangle(
            (0, 0.94),
            1,
            0.06,
            facecolor=C_GOLD,
            alpha=0.14,
            lw=0,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.04,
        0.97,
        "KEY TAKEAWAYS",
        ha="left",
        va="center",
        color=_ui_text(C_GOLD),
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
        path_effects=[pe.withStroke(linewidth=2, foreground=BG_DARK)],
    )
    # Wrap body text into the panel
    import textwrap as _tw

    wrapped = "\n\n".join(
        _tw.fill(p.strip(), width=110) for p in body.split("\n\n") if p.strip()
    )
    ax.text(
        0.04,
        0.88,
        wrapped,
        ha="left",
        va="top",
        color=TEXT_MAIN,
        fontsize=11,
        transform=ax.transAxes,
        linespacing=1.55,
        wrap=True,
    )
    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


# ── Commentary catalogue: position-keyed text for the 39 tactical visuals ──
# Order must match football_match_analysis._build_visual_catalog (idx 1..N).
_VISUAL_COMMENTARY = [
    # 1. xG Flow
    (
        "When did the chances actually come?",
        "The xG Flow plots cumulative xG minute by minute as a staircase, with "
        "every step marking a shot whose height equals its chance quality. "
        "Stars sit on goals; the shaded territory under each curve is total "
        "chance creation. A team that pulls clearly above the other built the "
        "stronger shot profile, even if the scoreline says otherwise.",
    ),
    # 2. Home Shot Map
    (
        "Reading the Home Shot Map",
        "Each dot is a single shot, located at where it was struck. Marker "
        "size scales with xG, so big circles are big chances. Filled circles "
        "are goals, hollow ones are misses. Cluster density inside the box "
        "shows where the team manufactured looks — wide-area dots usually "
        "indicate speculative efforts.",
    ),
    # 3. Away Shot Map
    (
        "Reading the Away Shot Map",
        "Same encoding as the home shot map: dot location is where the shot "
        "was taken, dot size is xG, fill = goal. Compare both maps side by "
        "side to read which side accessed central, high-value zones and which "
        "had to settle for low-percentage outside-the-box attempts.",
    ),
    # 4. Shot Breakdown & Goals
    (
        "How did the shots break down?",
        "The bar group counts every shot by outcome — total, woodwork, on "
        "target, off target, blocked. On-target conversion (goals divided by "
        "shots on target) reflects finishing efficiency. The Goals & Assists "
        "table below records the scorer, the play type (Open Play / Set "
        "Piece / Penalty), the assister and the chance's xG.",
    ),
    # 5. Home Pass Network
    (
        "Reading the Home Pass Network",
        "Each node is a player placed at their average pass position. Node "
        "size scales with passes attempted; line width between two players "
        "scales with the volume of completed passes between them. Heavy lines "
        "reveal the side's preferred passing partnerships and which axis the "
        "build-up flowed through.",
    ),
    # 6. Away Pass Network
    (
        "Reading the Away Pass Network",
        "Same encoding as the home network. Read each side's shape: a flat, "
        "wide network indicates a pass-through-the-thirds approach; a tall, "
        "narrow one points to vertical, central build-up. Isolated nodes far "
        "from the cluster usually mark a wide outlet who barely received.",
    ),
    # 7. Home xT Map
    (
        "Reading the Home xT Map",
        "The grid colours each pitch zone by its xT (expected threat) value — "
        "the probability that owning the ball there leads to a shot in the "
        "next few seconds. White arrows are positive-xT passes (gained "
        "threat); red arrows are negative-xT passes (gave threat back). "
        "Dense white traffic into the warm zones is the team's progression "
        "signature.",
    ),
    # 8. Away xT Map
    (
        "Reading the Away xT Map",
        "Same grid + arrow encoding for the away side. The total xT in the "
        "side panel sums all positive-xT passes; compare totals to read who "
        "carried the threat-creation load and where on the pitch each side's "
        "progression actually happened.",
    ),
    # 9. Shot Comparison
    (
        "Who was the more dangerous side?",
        "A side-by-side bar chart of the headline shooting numbers — total "
        "shots, shots on target, big chances, xG. It is the fastest single "
        "view to answer 'who was the more dangerous attacking side?' before "
        "diving into the maps.",
    ),
    # 10. Home Danger Creation
    (
        "Reading the Home Danger Creation",
        "Every event that ended in a shot, key pass or box entry is plotted "
        "to show which channels generated the high-value moments. Concentrate "
        "on the warm zones — those are the side's true danger lanes.",
    ),
    # 11. Away Danger Creation
    (
        "Reading the Away Danger Creation",
        "Same encoding for the away side: shots, key passes and box entries "
        "overlaid on the pitch. Compare the two danger maps to see which "
        "side's attack lived in central, high-quality areas vs. the flanks.",
    ),
    # 12. Goalkeeper Saves
    (
        "How hard did the keepers work?",
        "Plots the location of every shot the keeper faced, colour-coded by "
        "outcome (saved / goal / off-target). Save quality is a function of "
        "the xG of the shots faced — saving a high-xG strike beats stopping "
        "easy long-range efforts.",
    ),
    # 13. xG / xGoT Summary
    (
        "How clinical was the finishing?",
        "xG measures pre-shot chance quality; xGoT (expected goals on target) "
        "measures the post-shot quality once placement and power are known. "
        "A team with xGoT well above xG executed their finishing better than "
        "the average; below xG points to wasteful conversion.",
    ),
    # 14. Home Zone 14 and Half-Spaces
    (
        "Reading Home Zone 14 & Half-Spaces",
        "Zone 14 is the central pocket just outside the box — historically the "
        "richest zone for chance creation. The half-spaces flank it. This map "
        "counts every action the side completed in those zones; volume here "
        "is a proxy for how often they reached the most dangerous central "
        "real estate.",
    ),
    # 15. Away Zone 14 and Half-Spaces
    (
        "Reading Away Zone 14 & Half-Spaces",
        "Same encoding for the away side. Compare central-pocket access — "
        "the side that worked the ball into Zone 14 more often usually had "
        "the better creative platform, even before chance quality is "
        "measured.",
    ),
    # 16. Match Statistics (legacy table)
    (
        "What do the headline numbers say?",
        "A consolidated table of headline numbers — possession, passes, "
        "shots, on-target, xG, xT, key passes, big chances. Each row is a "
        "head-to-head with the leader bolded. Use it as the single-page "
        "summary of the entire match.",
    ),
    # 17. Territorial Control
    (
        "Reading Territorial Control",
        "Splits the pitch into bands and reads which team had the ball more "
        "often in each. A side with most of its colour in the opponent's "
        "third was camped high and aggressive; deep colour bands mean the "
        "side defended low and tried to play out from the back.",
    ),
    # 18. Ball Touches
    (
        "Who controlled the territory?",
        "Plots every touch on the pitch as a heatmap. Hot zones reveal the "
        "side's centre of gravity — where the game actually got played. "
        "Compare the two heatmaps to see which territory each side owned.",
    ),
    # 19. Home Pass Map by Third
    (
        "Reading the Home Pass Map by Third",
        "Splits passes into defensive, middle and attacking third buckets "
        "and visualises each on the pitch. Final-third pass density and "
        "completion rate are the two quickest reads of break-down activity "
        "in the opposition area.",
    ),
    # 20. Away Pass Map by Third
    (
        "Reading the Away Pass Map by Third",
        "Same per-third pass map for the away side. A team that had heavy "
        "middle-third volume but thin attacking-third volume struggled to "
        "convert build-up into break-down; that is the classic 'sideways "
        "possession' pattern.",
    ),
    # 21. xT per Minute
    (
        "When did each side surge in threat?",
        "A diverging bar chart: home xT bars rise above the zero line, away "
        "xT drops below it. Tall spikes mark the moments each side surged in "
        "threat creation. The 5-minute rolling average overlay smooths the "
        "noise and shows momentum windows.",
    ),
    # 22. Home Progressive Passes
    (
        "Reading Home Progressive Passes",
        "Plots every pass that moved the ball at least 25% closer to goal "
        "(or any pass into the final third). Wide-spread arrows mean the "
        "progression load was shared; concentration around one player marks "
        "a single source of forward play.",
    ),
    # 23. Away Progressive Passes
    (
        "Reading Away Progressive Passes",
        "Same encoding for the away side. The total count and the spatial "
        "distribution together tell you whether the side moved the ball "
        "forward by volume, by quality, or both.",
    ),
    # 24. Home Crosses
    (
        "Reading Home Crosses",
        "Every cross into the box plotted with its origin and end point. "
        "Successful crosses are colour-coded distinctly. A cluster from the "
        "byline indicates a touchline-and-cut-back attack; deeper origins "
        "mark whipped, second-phase deliveries.",
    ),
    # 25. Away Crosses
    (
        "Reading Away Crosses",
        "Same cross map for the away side. Compare cross volume and accuracy "
        "to understand how much of each side's box entry came through wide "
        "play versus central combinations.",
    ),
    # 26. Home Defensive Heatmap
    (
        "Reading the Home Defensive Heatmap",
        "A density map of every defensive action the side completed — "
        "tackles, interceptions, clearances, blocks, recoveries. The hot "
        "zones reveal where the team chose to defend: high up the pitch "
        "(press) or deep in their own block.",
    ),
    # 27. Away Defensive Heatmap
    (
        "Reading the Away Defensive Heatmap",
        "Same defensive density map for the away side. Compare the heat "
        "centres to read each team's defensive line height — a side whose "
        "actions cluster near the halfway line was pressing; one whose "
        "actions sit near their own box was sitting deep.",
    ),
    # 28. Defensive Summary
    (
        "Who did more defensive work?",
        "A headline panel of defensive metrics for both sides — tackles, "
        "interceptions, blocks, clearances, recoveries, aerial duels. "
        "Tackles show direct duels, interceptions show anticipation and "
        "cover-shadow play, recoveries show control of loose-ball moments "
        "after pressure or clearances.",
    ),
    # 29. Home Average Positions
    (
        "Reading Home Average Positions",
        "Each player's average XY position across all their touches, scaled "
        "and connected to show team shape. A high, narrow shape signals an "
        "aggressive, compact pressing block; a deep, wide shape points to a "
        "low-block defensive setup.",
    ),
    # 30. Away Average Positions
    (
        "Reading Away Average Positions",
        "Same average-position shape for the away side. Compare the two "
        "shapes to read the structural matchup — overlap on the centre line "
        "indicates a battle for the middle, separation indicates one side "
        "controlled the territory.",
    ),
    # 31. Dominating Zone
    (
        "Which side owned which zones?",
        "The pitch is divided into a grid; each cell is coloured by the side "
        "that had more ball-touches there. Block colour reveals territorial "
        "dominance at a glance — large contiguous regions in one team's "
        "colour mark areas they controlled outright.",
    ),
    # 32. Home Box Entries
    (
        "Reading Home Box Entries",
        "Plots every entry into the opposition penalty area — through pass, "
        "carry, cross or set piece. Clusters near the byline mean wide-and-"
        "cut-back access; clusters at the D mean central, through-ball "
        "access. The total count is a direct measure of break-down volume.",
    ),
    # 33. Away Box Entries
    (
        "Reading Away Box Entries",
        "Same encoding for the away side. Compare both maps to see whose "
        "attack actually arrived inside the 18-yard box, and through which "
        "channel the bulk of the entries came.",
    ),
    # 34. Home High Regains
    (
        "Reading Home High Regains",
        "Marks inferred open-play changes of control beginning at x >= 60. "
        "Restarts are excluded, and each dot represents the first event of "
        "the new possession rather than every defensive action in the zone.",
    ),
    # 35. Away High Regains
    (
        "Reading Away High Regains",
        "Use the count with transition shots and regain-to-shot rate. More "
        "high regains show where control was won, but only same-possession "
        "outcomes show whether the press created an immediate attack.",
    ),
    # 36. Home Pass Target Zones
    (
        "Reading Home Pass Target Zones",
        "Heatmap of where on the pitch the side's passes landed — i.e. their "
        "preferred receiving zones. Compare with the pass network to see "
        "whether the receiving pattern matches the network's shape, or "
        "whether passes were aiming for under-served outlets.",
    ),
    # 37. Away Pass Target Zones
    (
        "Reading Away Pass Target Zones",
        "Same pass-target heatmap for the away side. A side whose target "
        "zones cluster high and central had build-up reaching dangerous "
        "areas; clusters in their own half indicate possession recycled but "
        "rarely advanced.",
    ),
]


def _commentary_for_visual(idx_zero_based: int, hn: str, an: str):
    """
    Look up the (heading, body) commentary for a tactical visual by its
    position in extra_figs. Falls back to a generic message for any
    out-of-range index so the PDF never breaks.
    """
    if 0 <= idx_zero_based < len(_VISUAL_COMMENTARY):
        return _VISUAL_COMMENTARY[idx_zero_based]
    return (
        "What does this visual add?",
        "This chart adds context to the tactical story above. Read it "
        "alongside the previous pages — chance quality, territory, "
        "pressing and ball progression all reinforce the same story when "
        "the numbers line up.",
    )


def _draw_team_stats_compare_page(pdf, info, events, ppda):
    """
    Team comparison page with the unified visual identity.
    Layout (top → bottom):
        • Yellow accent frame + section header
        • Attacking & passing stats bars (top)
        • PPDA mini-gauges + verdict EMBEDDED in the middle
        • Defensive stats bars (bottom) — including correct Blocks count
        • Detailed English commentary footer
    """
    fig = _new_dark_fig(14, 9)
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "—"
    home_col = info.get("home_color") or info.get("HOME_COLOR") or C_HOME
    away_col = info.get("away_color") or info.get("AWAY_COLOR") or C_AWAY
    home_text_col = _team_label_color(home_col, BG_PANEL)
    away_text_col = _team_label_color(away_col, BG_PANEL)

    went_to_et, pens = _extra_time_and_pens(events, info)
    et_note = ""
    if pens is not None:
        et_note = (
            f" · AET, {pens[0]}-{pens[1]} pens — every row below includes extra time"
        )
    elif went_to_et:
        et_note = " · AET — every row below includes extra time"

    if apply_unified_frame is not None:
        apply_unified_frame(
            fig,
            section="TEAM STATISTICS",
            title=f"{hn} vs {an} — Match Statistics",
            subtitle="Attack · Passing · Pressing (PPDA) · Defense — "
            "every metric is computed directly from the event stream" + et_note,
            accent=C_GOLD,
            home_name=hn,
            away_name=an,
            score=str(score),
            footer_note="Bars scaled to the higher value · bold = leader",
        )
    else:
        fig.text(
            0.04,
            0.94,
            "TEAM STATISTICS — COMPARISON",
            color=TEXT_BRIGHT,
            fontsize=20,
            fontweight="bold",
        )
        fig.text(
            0.04,
            0.91,
            "Side-by-side counts from event data",
            color=TEXT_DIM,
            fontsize=10,
            style="italic",
        )

    home_id = info.get("home_id")
    away_id = info.get("away_id")
    advanced = team_advanced_metrics(events, info)
    home_advanced = advanced["home"]
    away_advanced = advanced["away"]

    def _count(team_id, mask_fn) -> int:
        if events is None or events.empty:
            return 0
        try:
            return int(mask_fn(events[events["team_id"] == team_id]).sum())
        except Exception:
            return 0

    # Goals must be counted by scoring_team WITHOUT pre-filtering on the event
    # row's team_id: an own goal is logged on the scorer's own team_id but
    # credited (scoring_team) to the opponent. Pre-filtering by team_id — as
    # _count does — would drop that goal from the beneficiary's tally (the bug
    # that showed Australia 0 despite their 1-1 own-goal equaliser). Shootout
    # kicks (is_penalty_shootout) are excluded so the real scoreline stands.
    def _goals_for(team_id) -> int:
        if events is None or events.empty or "is_goal" not in events.columns:
            return 0
        try:
            g = events[events["is_goal"].fillna(False)]
            if "is_penalty_shootout" in g.columns:
                g = g[~g["is_penalty_shootout"].fillna(False)]
            side_col = "scoring_team" if "scoring_team" in g.columns else "team_id"
            return int((g[side_col] == team_id).sum())
        except Exception:
            return 0

    # Shots exclude own goals: an own goal is flagged is_shot on the scorer's
    # own team_id, but it is not a shot AT the opponent's goal, so counting it
    # would inflate that team's shot tally (and disagree with the shot map,
    # which already drops it).
    def _shots_for(team_id) -> int:
        if events is None or events.empty or "is_shot" not in events.columns:
            return 0
        try:
            s = events[
                (events["team_id"] == team_id)
                & (events["is_shot"].fillna(False) == True)
            ]  # noqa: E712
            if "is_goal" in s.columns and "scoring_team" in s.columns:
                og = s["is_goal"].fillna(False) & (s["scoring_team"] != team_id)
                s = s[~og]
            return int(len(s))
        except Exception:
            return 0

    # xG / xT are summed straight from the event columns (same convention the
    # rest of the report uses); big chances = shots flagged as a big chance by
    # the provider. All three are chance-quality metrics, so they sit with the
    # attacking stats.
    def _xg_for(team_id):
        if events is None or events.empty or "xG" not in events.columns:
            return 0.0
        return round(
            float(events.loc[events["team_id"] == team_id, "xG"].fillna(0).sum()), 2
        )

    def _xt_for(team_id):
        # Match the xT map's "total xT": threat CREATED by positive successful
        # passes only (not the raw net of every event's xT delta), so the two
        # visuals agree.
        if events is None or events.empty or "xT" not in events.columns:
            return 0.0
        d = events[events["team_id"] == team_id]
        if "is_pass" in d.columns:
            d = d[(d["is_pass"].fillna(False) == True)]  # noqa: E712
            if "outcome" in d.columns:
                d = d[d["outcome"] == "Successful"]
        xt = d["xT"].fillna(0)
        return round(float(xt[xt > 0].sum()), 2)

    def _big_chances_for(team_id):
        if events is None or events.empty or "big_chance" not in events.columns:
            return 0
        s = events[(events["team_id"] == team_id)]
        if "is_shot" in s.columns:
            s = s[s["is_shot"].fillna(False) == True]  # noqa: E712
        return int(s["big_chance"].fillna(False).astype(bool).sum())

    # ── Stats grouped: attacking/passing on top, defensive on bottom ──
    attack_rows = [
        ("Goals", _goals_for(home_id), _goals_for(away_id)),
        ("xG", _xg_for(home_id), _xg_for(away_id)),
        ("Shots", _shots_for(home_id), _shots_for(away_id)),
        ("Big chances", _big_chances_for(home_id), _big_chances_for(away_id)),
        ("xT", _xt_for(home_id), _xt_for(away_id)),
        (
            "Passes attempted",
            _count(home_id, lambda d: d.get("is_pass", False) == True),  # noqa: E712
            _count(away_id, lambda d: d.get("is_pass", False) == True),
        ),  # noqa: E712
        (
            "Key passes",
            _count(
                home_id, lambda d: d.get("is_key_pass", False) == True
            ),  # noqa: E712
            _count(away_id, lambda d: d.get("is_key_pass", False) == True),
        ),  # noqa: E712
        (
            "Progressive passes",
            home_advanced["progressive_passes"],
            away_advanced["progressive_passes"],
        ),
        (
            "Box entries",
            home_advanced["box_entries"],
            away_advanced["box_entries"],
        ),
        (
            "Field tilt %",
            home_advanced["field_tilt"],
            away_advanced["field_tilt"],
        ),
    ]

    def _defensive_blocks_for(team_id) -> int:
        if events is None or events.empty:
            return 0
        opp_id = away_id if team_id == home_id else home_id
        opp_blocks = defensive_blocks_count(events, team_id, opp_id)
        if opp_blocks == 0:
            opp_side = "away" if team_id == home_id else "home"
            mc = (info.get("matchcentre_stats", {}) or {}).get(opp_side, {}) or {}
            try:
                opp_blocks = int(mc.get("blocked") or 0)
            except Exception:
                opp_blocks = 0
        return opp_blocks

    # Fouls: a "Foul" event is logged for BOTH the player who committed it
    # (outcome Unsuccessful) and the one who won it (Successful). Count only
    # fouls COMMITTED when the feed distinguishes them, else all Foul rows.
    _foul_rows = (
        events[events["type"] == "Foul"]
        if (events is not None and not events.empty and "type" in events.columns)
        else None
    )
    _splits_fouls = (
        _foul_rows is not None
        and "outcome" in events.columns
        and _foul_rows["outcome"].astype(str).str.lower().eq("unsuccessful").any()
        and _foul_rows["outcome"].astype(str).str.lower().eq("successful").any()
    )

    def _fouls_committed(team_id):
        if _foul_rows is None:
            return 0
        f = _foul_rows[_foul_rows["team_id"] == team_id]
        if _splits_fouls:
            return int(f["outcome"].astype(str).str.lower().eq("unsuccessful").sum())
        return int(len(f))

    # Duels (won/contested). Aerial: a header duel is logged for both players —
    # winner Successful, loser Unsuccessful. Ground: won = tackles + take-ons
    # won; contested = the team's tackles + take-ons + challenges (a challenge
    # is a lost ground duel). Returns a (won, "won/total") pair for each side.
    def _duels(team_id):
        # Ground duels modelled as dribble contests (TakeOn) so both sides share
        # the same contested total, like aerials: dribbler's team wins a
        # Successful TakeOn, defending team wins an Unsuccessful one.
        if events is None or events.empty or "type" not in events.columns:
            return (0, "0/0"), (0, "0/0")
        ty = events["type"].astype(str)
        has_out = "outcome" in events.columns
        out = events["outcome"].astype(str) if has_out else None
        tm = events["team_id"] == team_id
        _others = [t for t in events["team_id"].dropna().unique() if t != team_id]
        opp_id = (
            max(_others, key=lambda t: int((events["team_id"] == t).sum()))
            if _others
            else None
        )
        om = events["team_id"] == opp_id if opp_id is not None else (tm & False)

        def _c(mask, tp, success=None):
            m = mask & (ty == tp)
            if success is not None and has_out:
                m = m & (out == ("Successful" if success else "Unsuccessful"))
            return int(m.sum())

        aw, at = (_c(tm, "Aerial", True) if has_out else 0), _c(tm, "Aerial")
        to_self = _c(tm, "TakeOn")
        to_opp = _c(om, "TakeOn")
        gt = to_self + to_opp
        gw = (
            (_c(tm, "TakeOn", True) + (to_opp - _c(om, "TakeOn", True)))
            if has_out
            else 0
        )
        return (aw, f"{aw}/{at}"), (gw, f"{gw}/{gt}")

    _h_aer, _h_grd = _duels(home_id)
    _a_aer, _a_grd = _duels(away_id)

    # ── Defensive stats: a blocked shot belongs to the defending team.
    defensive_rows = [
        (
            "Tackles",
            _count(home_id, lambda d: d["type"] == "Tackle"),
            _count(away_id, lambda d: d["type"] == "Tackle"),
        ),
        (
            "Interceptions",
            _count(home_id, lambda d: d["type"] == "Interception"),
            _count(away_id, lambda d: d["type"] == "Interception"),
        ),
        ("Blocks", _defensive_blocks_for(home_id), _defensive_blocks_for(away_id)),
        (
            "Clearances",
            _count(home_id, lambda d: d["type"] == "Clearance"),
            _count(away_id, lambda d: d["type"] == "Clearance"),
        ),
        (
            "Provider recoveries",
            home_advanced["provider_recoveries"],
            away_advanced["provider_recoveries"],
        ),
        (
            "Possession regains",
            home_advanced["possession_regains"],
            away_advanced["possession_regains"],
        ),
        (
            "High regains",
            home_advanced["high_regains"],
            away_advanced["high_regains"],
        ),
        ("Fouls", _fouls_committed(home_id), _fouls_committed(away_id)),
        ("Aerial duels", _h_aer, _a_aer),
        ("Ground duels", _h_grd, _a_grd),
    ]

    h_ppda = ppda.get("home", {}).get("ppda")
    a_ppda = ppda.get("away", {}).get("ppda")

    # ── Helper to draw a stats panel with bar comparison ──
    def _draw_stats_panel(panel_xywh, title, rows_list, accent, home_color, away_color):
        x, y, w, h = panel_xywh
        ax = fig.add_axes([x, y, w, h])
        ax.set_facecolor(BG_PANEL)
        for s in ax.spines.values():
            s.set_edgecolor(accent)
            s.set_linewidth(1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # ── Panel header strip — flat, matching panel_card()'s `.panel-head` ──
        ax.add_patch(
            mpatches.Rectangle(
                (0, 0.93), 1, 0.07, facecolor=BG_HEADER, lw=0, transform=ax.transAxes
            )
        )
        ax.add_patch(
            mpatches.Circle(
                (0.018, 0.965),
                0.011,
                facecolor=accent,
                edgecolor="none",
                transform=ax.transAxes,
            )
        )
        ax.text(
            0.04,
            0.965,
            title.upper(),
            ha="left",
            va="center",
            color=TEXT_BRIGHT,
            fontsize=10.5,
            fontweight="bold",
            family=FONT_MONO,
            transform=ax.transAxes,
        )

        # ── Team labels row (just below the title strip) ──
        ax.text(
            0.02,
            0.885,
            hn,
            ha="left",
            va="center",
            color=home_color,
            fontsize=9.5,
            fontweight="bold",
            family=FONT_MONO,
            transform=ax.transAxes,
        )
        ax.text(
            0.98,
            0.885,
            an,
            ha="right",
            va="center",
            color=away_color,
            fontsize=9.5,
            fontweight="bold",
            family=FONT_MONO,
            transform=ax.transAxes,
        )
        # subtle divider under team labels
        ax.plot(
            [0.02, 0.98], [0.85, 0.85], color=GRID_COL, lw=1.0, transform=ax.transAxes
        )

        n = len(rows_list)
        if n == 0:
            return
        # Layout zones: fixed value columns, mirrored bars, and a clean
        # centre label chip so long labels never sit on top of the bars.
        top, bot = 0.83, 0.05
        spacing = (top - bot) / n
        for i, (label, hv, av) in enumerate(rows_list):
            cy = top - (i + 0.5) * spacing
            # A value may be a (bar_number, display_string) tuple — used by duel
            # rows that scale the bar by "won" but print "won/total".
            h_disp = hv[1] if isinstance(hv, (tuple, list)) else _na(hv)
            a_disp = av[1] if isinstance(av, (tuple, list)) else _na(av)
            hv_num = hv[0] if isinstance(hv, (tuple, list)) else hv
            av_num = av[0] if isinstance(av, (tuple, list)) else av
            try:
                hh, aa = float(hv_num), float(av_num)
                mx = max(hh, aa, 1)
                h_ratio, a_ratio = hh / mx, aa / mx
            except (TypeError, ValueError):
                h_ratio = a_ratio = 0
                hh = aa = None

            bar_h = spacing * 0.40
            # home bar (anchored at 0.38, grows leftwards)
            if h_ratio:
                bw = 0.27 * h_ratio
                ax.add_patch(
                    mpatches.Rectangle(
                        (0.38 - bw, cy - bar_h / 2),
                        bw,
                        bar_h,
                        facecolor=home_color,
                        alpha=0.88,
                        lw=0,
                        transform=ax.transAxes,
                    )
                )
            # away bar (anchored at 0.62, grows rightwards)
            if a_ratio:
                bw = 0.27 * a_ratio
                ax.add_patch(
                    mpatches.Rectangle(
                        (0.62, cy - bar_h / 2),
                        bw,
                        bar_h,
                        facecolor=away_color,
                        alpha=0.88,
                        lw=0,
                        transform=ax.transAxes,
                    )
                )

            h_better = hh is not None and aa is not None and hh > aa
            a_better = aa is not None and hh is not None and aa > hh

            # Numbers — placed OUTSIDE the bars so nothing overlaps them.
            # Leader gets gold + bold for instant scan; loser stays bright.
            leader_home_col = C_GOLD if h_better else TEXT_BRIGHT
            leader_away_col = C_GOLD if a_better else TEXT_BRIGHT
            # Long "won/total" values need a smaller font and a slightly inset
            # anchor, or a 5-char string overruns the panel's left/right edge.
            _long = max(len(str(h_disp)), len(str(a_disp))) > 3
            _vfs = 8.5 if _long else 11.5
            _hx = 0.085 if _long else 0.06
            _ax = 0.915 if _long else 0.94
            ax.text(
                _hx,
                cy,
                h_disp,
                ha="right",
                va="center",
                color=leader_home_col,
                fontsize=_vfs,
                fontweight="bold",
                family=FONT_MONO,
                transform=ax.transAxes,
            )
            ax.text(
                _ax,
                cy,
                a_disp,
                ha="left",
                va="center",
                color=leader_away_col,
                fontsize=_vfs,
                fontweight="bold",
                family=FONT_MONO,
                transform=ax.transAxes,
            )
            # Stat label — centered on a chip, never over the bars.
            ax.text(
                0.50,
                cy,
                label,
                ha="center",
                va="center",
                color=TEXT_DIM,
                fontsize=8.7,
                fontweight="bold",
                family=FONT_SANS,
                transform=ax.transAxes,
                bbox=dict(
                    boxstyle="round,pad=0.18,rounding_size=0.02",
                    facecolor=BG_MID,
                    edgecolor="none",
                    alpha=0.96,
                ),
            )

    # Left column: attacking/passing (top) · defensive (bottom).
    _draw_stats_panel(
        (0.035, 0.50, 0.40, 0.34),
        "Attack & Passing",
        attack_rows,
        home_col,
        home_col,
        away_col,
    )
    _draw_stats_panel(
        (0.035, 0.06, 0.40, 0.40),
        "Defensive Actions",
        defensive_rows,
        away_col,
        home_col,
        away_col,
    )

    # ── Result banner (left half, above the panels) — states the final score
    # AND the shootout outcome clearly, so a 1-1 that was actually decided on
    # penalties never reads as a plain draw. Sits opposite the PPDA header.
    _clean_score = str(score).replace("*", "").strip()
    if pens is not None:
        if pens[0] > pens[1]:
            _win, _wa, _wb = hn, pens[0], pens[1]
        else:
            _win, _wa, _wb = an, pens[1], pens[0]
        result_line = f"FT {_clean_score} (AET) · {_win} won {_wa}-{_wb} on penalties"
    elif went_to_et:
        result_line = f"FT {_clean_score} — after extra time"
    else:
        result_line = ""
    if result_line:
        fig.text(
            0.235,
            0.862,
            result_line,
            ha="center",
            va="center",
            color=C_GOLD,
            fontsize=10.5,
            fontweight="bold",
            family=FONT_MONO,
            path_effects=[pe.withStroke(linewidth=2.5, foreground=BG_DARK)],
        )

    # ── Centre column: PPDA mini-gauges (the requested embedded PPDA) ──
    def _mini_dial(ax_pos, name, value, color):
        ax = fig.add_axes(ax_pos, projection="polar")
        ax.set_facecolor(BG_DARK)
        ax.set_theta_zero_location("W")
        ax.set_theta_direction(-1)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_ylim(0, 1)

        v = value if value is not None else 0
        vmin, vmax = 5.0, 25.0
        ratio = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
        angle = ratio * np.pi

        n_seg = 48
        thetas = np.linspace(0, np.pi, n_seg + 1)
        for i in range(n_seg):
            t = i / n_seg
            if t < 0.15:
                c = "#22c55e"
            elif t < 0.30:
                c = "#84cc16"
            elif t < 0.45:
                c = "#facc15"
            else:
                c = "#f97316"
            ax.bar(
                (thetas[i] + thetas[i + 1]) / 2,
                0.18,
                bottom=0.78,
                width=(thetas[i + 1] - thetas[i]) * 0.95,
                color=c,
                edgecolor="none",
                alpha=0.80,
            )
        if value is not None:
            ax.plot(
                [angle, angle],
                [0, 0.92],
                color=color,
                lw=3.5,
                solid_capstyle="round",
                zorder=5,
            )
            ax.scatter(
                [angle],
                [0.92],
                s=55,
                color=TEXT_BRIGHT,
                edgecolor=color,
                linewidth=1.8,
                zorder=7,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["polar"].set_visible(False)

        cx = ax_pos[0] + ax_pos[2] / 2
        # Team name ABOVE the dial; value + intensity clear BELOW the gauge's
        # flat diameter so the big number never sits on the coloured arc. The
        # semicircle's baseline sits roughly at the box's vertical centre, so
        # the value is dropped below the box bottom to stay clear.
        fig.text(
            cx,
            ax_pos[1] + ax_pos[3] + 0.012,
            name,
            ha="center",
            color=TEXT_BRIGHT,
            fontsize=10.5,
            fontweight="bold",
            family=FONT_SANS,
        )
        val_str = f"{value:.2f}" if value is not None else "N/A"
        fig.text(
            cx,
            ax_pos[1] - 0.008,
            val_str,
            ha="center",
            color=color,
            fontsize=15,
            fontweight="bold",
            family=FONT_MONO,
        )
        lbl, lcol = _ppda_intensity_label(value)
        fig.text(
            cx,
            ax_pos[1] - 0.032,
            lbl,
            ha="center",
            color=lcol,
            fontsize=7.5,
            fontweight="bold",
            family=FONT_MONO,
        )

    # ── Right half TOP: PPDA gauges + verdict (fills what used to be dead
    # centre space). Header dot + mono label matches panel_card's .panel-head.
    fig.add_artist(
        mpatches.Circle(
            (0.475, 0.862),
            0.0035,
            transform=fig.transFigure,
            facecolor=C_GOLD,
            edgecolor="none",
            zorder=2,
        )
    )
    fig.text(
        0.487,
        0.862,
        "PRESSING · PPDA",
        ha="left",
        color=TEXT_BRIGHT,
        fontsize=10.5,
        fontweight="bold",
        family=FONT_MONO,
    )
    fig.text(
        0.717,
        0.828,
        "Passes per Defensive Action — lower = more aggressive press",
        ha="center",
        color=TEXT_DIM,
        fontsize=8.5,
        family=FONT_SANS,
    )

    _mini_dial([0.545, 0.64, 0.15, 0.15], hn, h_ppda, home_col)
    _mini_dial([0.740, 0.64, 0.15, 0.15], an, a_ppda, away_col)

    # PPDA verdict line
    if h_ppda is not None and a_ppda is not None:
        if h_ppda < a_ppda:
            verdict = f"{hn} pressed more aggressively"
            vcol = home_text_col
            diff = a_ppda - h_ppda
        elif a_ppda < h_ppda:
            verdict = f"{an} pressed more aggressively"
            vcol = away_text_col
            diff = h_ppda - a_ppda
        else:
            verdict = "Both teams pressed equally"
            vcol = TEXT_BRIGHT
            diff = 0.0
        fig.text(
            0.717,
            0.585,
            verdict,
            ha="center",
            color=vcol,
            fontsize=12,
            fontweight="bold",
        )
        if diff > 0:
            fig.text(
                0.717,
                0.562,
                f"PPDA differential: {diff:.2f}",
                ha="center",
                color=TEXT_DIM,
                fontsize=9,
            )

    # ── Right half BOTTOM: structured commentary in a wide panel laid out as
    # three columns. The old narrow 0.16-wide column forced the body text to
    # overflow its box; a wide panel with manual word-wrap fixes that and
    # fills the previously-empty lower-centre space.
    import textwrap

    com_ax = fig.add_axes([0.47, 0.06, 0.495, 0.42])
    com_ax.set_facecolor(BG_PANEL)
    for s in com_ax.spines.values():
        s.set_edgecolor(GRID_COL)
        s.set_linewidth(1.0)
    com_ax.set_xticks([])
    com_ax.set_yticks([])
    com_ax.set_xlim(0, 1)
    com_ax.set_ylim(0, 1)

    # Header strip — flat, matching panel_card()'s `.panel-head`.
    com_ax.add_patch(
        mpatches.Rectangle(
            (0, 0.90), 1, 0.10, facecolor=BG_HEADER, transform=com_ax.transAxes, lw=0
        )
    )
    com_ax.add_patch(
        mpatches.Circle(
            (0.022, 0.95),
            0.011,
            facecolor=C_GOLD,
            edgecolor="none",
            transform=com_ax.transAxes,
        )
    )
    com_ax.text(
        0.045,
        0.95,
        "READING THIS PAGE",
        ha="left",
        va="center",
        color=TEXT_BRIGHT,
        fontsize=9.5,
        fontweight="bold",
        family=FONT_MONO,
        transform=com_ax.transAxes,
    )
    com_ax.plot(
        [0, 1], [0.90, 0.90], color=GRID_COL, lw=1.0, transform=com_ax.transAxes
    )

    # Three structured sections — heading + body, side by side in columns.
    sections = [
        (
            home_text_col,
            "ATTACK & PASSING",
            "Shots and key passes show how often each side created looks at "
            "goal. Pass volume reflects how long the ball was kept circulating.",
        ),
        (
            C_GOLD,
            "PRESSING (PPDA)",
            "Lower PPDA = more aggressive press. Green means an opponent action "
            "was forced every few passes; orange means a deeper block. The "
            "verdict names the more aggressive presser.",
        ),
        (
            away_text_col,
            "DEFENSIVE ACTIONS",
            "Tackles and interceptions describe ground duels. Blocks stop shots "
            "with a body. Recoveries and clearances show how pressure was "
            "escaped; fouls mark where containment broke down.",
        ),
    ]

    col_cx = [0.185, 0.505, 0.825]
    for (color, heading, body), cx in zip(sections, col_cx):
        com_ax.add_patch(
            mpatches.Circle(
                (cx - 0.145, 0.82),
                0.010,
                facecolor=color,
                lw=0,
                transform=com_ax.transAxes,
                zorder=4,
            )
        )
        com_ax.text(
            cx - 0.125,
            0.82,
            heading,
            ha="left",
            va="center",
            color=color,
            fontsize=8.3,
            fontweight="bold",
            transform=com_ax.transAxes,
            path_effects=[pe.withStroke(linewidth=2, foreground=BG_DARK)],
        )
        wrapped = textwrap.fill(body, width=30)
        com_ax.text(
            cx - 0.145,
            0.74,
            wrapped,
            ha="left",
            va="top",
            color=TEXT_MAIN,
            fontsize=7.8,
            transform=com_ax.transAxes,
            linespacing=1.5,
        )

    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _draw_player_stats_pages(pdf, player_stats, info, visuals_dir):
    """Create player-statistics pages without saving PNG files.

    Dark.py creates figures 41 and 42 and orders them with the other visuals in
    SAVE_DIR.
    """
    for side, color in (("home", C_HOME), ("away", C_AWAY)):
        df = player_stats.get(side, pd.DataFrame())
        team_name = info.get(f"{side}_name") or side.title()
        fig = draw_player_stats_table(df, team_name, team_color=color, save_path=None)
        pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
        plt.close(fig)


def export_pdf_pages(pdf_path: str, out_dir: str, dpi: int = 200) -> list[str]:
    """Render every page of the report PDF to a separate PNG for social posting.
    Saves into <out_dir>/report_pages/page_01.png … . Returns the paths written.
    Uses PyMuPDF (fitz) if available; silently no-ops if it isn't."""
    if not pdf_path or not os.path.exists(pdf_path):
        return []
    pages_dir = os.path.join(out_dir, "report_pages")
    os.makedirs(pages_dir, exist_ok=True)
    written = []
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        for idx in range(doc.page_count):
            pix = doc[idx].get_pixmap(dpi=dpi)
            p = os.path.join(pages_dir, f"page_{idx + 1:02d}.png")
            pix.save(p)
            written.append(p)
        doc.close()
    except ImportError:
        pass
    except Exception:
        pass
    return written


def _merge_pdfs(output_path: str, pdf_paths: list[str]) -> bool:
    """
    Merge multiple PDFs into one file. Prefer pypdf, fall back to PyPDF2, and
    return False when neither package is available.
    """
    pdf_paths = [p for p in pdf_paths if p and os.path.exists(p)]
    if not pdf_paths:
        return False

    try:
        from pypdf import PdfWriter

        writer = PdfWriter()
        for p in pdf_paths:
            writer.append(p)
        with open(output_path, "wb") as f:
            writer.write(f)
        writer.close()
        return True
    except ImportError:
        pass
    try:
        from PyPDF2 import PdfMerger

        merger = PdfMerger()
        for p in pdf_paths:
            merger.append(p)
        merger.write(output_path)
        merger.close()
        return True
    except ImportError:
        return False


def run_analysis(
    match_data: dict,
    parse_all_fn=None,
    extra_figs: list | None = None,
    extra_figs_filenames: list | None = None,
    merge_with_pdfs: list[str] | None = None,
    final_pdf_name: str | None = None,
) -> dict:
    configure_theme()
    _apply_amoled_matplotlib_defaults()
    """
    Unified entry point that creates one organized dark-themed PDF:
        1. Match summary
        2. Goals log
        3. PPDA gauge
        4. Team stats comparison
        5. Embedded extra_figs (the original 39 visuals, when supplied)
        6. Optional merging with source PDFs in merge_with_pdfs

    Args:
        match_data: raw matchCentreData dict.
        parse_all_fn: The project's original parse_all() function.
        extra_figs: The original 39 matplotlib figures, embedded after the new
                    pages in the same PDF.
        merge_with_pdfs: External PDF paths, such as the source tactical PDF,
                         to merge into the final PDF.
        final_pdf_name: Final PDF name when merging. Default:
                        full_match_report_<ts>.pdf

    Every visual and the final PDF are saved in the match directory.
    """
    if parse_all_fn is None:
        raise ValueError(
            "run_analysis requires parse_all_fn (e.g. parse_all_fn=parse_all)."
        )

    global OUTPUT_DIR, VISUALS_DIR
    output_dir = os.environ.get("MATCH_ANALYSIS_OUTPUT_DIR")
    if output_dir:
        OUTPUT_DIR = output_dir
        VISUALS_DIR = output_dir
    _ensure_output_dirs()

    info, events, _players_df = parse_all_fn(match_data)

    global C_HOME, C_AWAY
    C_HOME = "#7A3DFF"
    C_AWAY = "#BEEA24"
    info["home_color"] = C_HOME
    info["away_color"] = C_AWAY

    ppda = compute_ppda_both(info, events)
    goals_df = build_goals_log(events, info)
    player_stats = {"home": pd.DataFrame(), "away": pd.DataFrame()}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        goals_df.to_csv(
            os.path.join(OUTPUT_DIR, f"goals_log_{ts}.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    except Exception:
        pass

    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "—"

    # ── Group extra_figs by side (home / away / shared) ──
    # Group the tactical visuals by PHASE OF PLAY rather than by team, so the
    # report reads as a story — the match, then territory, then how each side
    # created, built up and defended — with home and away sitting side by side
    # within each phase for direct comparison.
    def _visual_phase(fname: str) -> int:
        f = (fname or "").lower()
        if any(
            k in f
            for k in (
                "xg_flow",
                "breakdown_goals",
                "shot_comparison",
                "xg_tiles",
                "xg_summary",
                "gk_saves",
                "xt_per_minute",
                "match_stats",
            )
        ):
            return 1  # the match story
        if any(
            k in f for k in ("territorial", "possession", "ball_touches", "dominating")
        ):
            return 2  # territory & control
        if any(
            k in f for k in ("shot_map", "danger", "zone14", "box_entries", "crosses")
        ):
            return 3  # chance creation
        if any(
            k in f
            for k in (
                "pass_network",
                "xt_map",
                "pass_thirds",
                "progressive",
                "avg_position",
                "pass_target",
            )
        ):
            return 4  # build-up & progression
        if any(
            k in f
            for k in ("defensive_hm", "defensive_summary", "high_turnovers", "ppda")
        ):
            return 5  # defence & pressing
        return 1

    PHASE_INFO = {
        1: (
            "THE MATCH STORY",
            "How the game actually unfolded — the xG rhythm, the shots, the "
            "finishing and the headline numbers that framed the result.",
            "#FFC23C",
        ),
        2: (
            "TERRITORY & CONTROL",
            "Who owned the ball and the space, and where on the pitch the "
            "game was really played.",
            "#A78BFA",
        ),
        3: (
            "CHANCE CREATION",
            "How each side manufactured danger — shots, the dangerous central "
            "zones, box entries and wide play, home and away side by side.",
            "#34D399",
        ),
        4: (
            "BUILD-UP & PROGRESSION",
            "How each side circulated possession and moved the ball forward "
            "into threatening areas.",
            "#60A5FA",
        ),
        5: (
            "DEFENCE & PRESSING",
            "How each side defended its goal, pressed the opponent and won "
            "the ball back.",
            "#F87171",
        ),
    }

    phases = {i: [] for i in range(1, 6)}
    if extra_figs:
        names = list(extra_figs_filenames or [])
        while len(names) < len(extra_figs):
            names.append("")
        for fig, fname in zip(extra_figs, names):
            if "player_stats" in (fname or "").lower():
                continue
            phases[_visual_phase(fname)].append((fig, fname))

    # Flat reading order across the whole report so each page can hand the
    # reader off to the next visual — one connected tactical argument.
    _flat_order = [fn for i in range(1, 6) for _fg, fn in phases[i]]
    _next_of = {
        fn: (_flat_order[i + 1] if i + 1 < len(_flat_order) else None)
        for i, fn in enumerate(_flat_order)
    }

    def _bridge(next_fname):
        if not next_fname:
            return (
                "This is the final analytical page — read it back against the whole "
                "sequence above: chance quality, field position, ball progression and "
                "pressure after loss should all tell one story."
            )
        nf = (next_fname or "").lower()
        reason = "to add the next layer of the tactical picture"
        for key, why in (
            ("xg_flow", "to see when these patterns actually moved the scoreline"),
            (
                "breakdown_goals",
                "to separate shot volume from finishing and see how the goals arrived",
            ),
            ("shot_map", "to check whether this turned into clean shooting positions"),
            (
                "shot_comparison",
                "to weigh shot quantity against genuine chance quality",
            ),
            (
                "xg_tiles",
                "to compare chance quality before the shot with quality after contact",
            ),
            (
                "xg_summary",
                "to compare chance quality before the shot with quality after contact",
            ),
            ("gk_saves", "to see how much real danger the goalkeeper faced"),
            ("pass_network", "to see the passing structure underneath it"),
            ("xt_minute", "to follow attacking momentum minute by minute"),
            ("xt_map", "to see which ball-progression routes carried the threat"),
            ("danger", "to see where the attacks actually hurt the opponent"),
            ("zone14", "to see how the central connection zones were used"),
            ("box_entries", "to see how often the attack reached the penalty area"),
            ("crosses", "to judge the wide-delivery plan against box occupation"),
            ("progressive", "to see how the ball was driven up the pitch"),
            ("pass_thirds", "to see where possession actually settled"),
            ("target_zones", "to see where the next receiver was wanted"),
            ("average_position", "to see the shape that underpinned it"),
            ("avg_position", "to see the shape that underpinned it"),
            ("territorial", "to see where on the pitch this was happening"),
            ("dominating_zone", "to turn this control into pitch geography"),
            ("ball_touches", "to see where touches concentrated"),
            ("possession", "to see where possession was concentrated"),
            ("turnover", "to see whether the press created attacking value"),
            ("ppda", "to see how aggressively the ball was pressed"),
            ("def", "to see how the same side defended without the ball"),
            ("match_stats", "to anchor it in the overall control panel"),
        ):
            if key in nf:
                reason = why
                break
        try:
            nh, _ = _commentary_for_filename(next_fname, hn, an)
        except Exception:
            nh = "the next visual"
        return f"Read next alongside “{nh}” {reason}."

    def _emit_visual(fig, fname):
        """Apply unified chrome (idempotent for v2 figs) + emit a single
        composite PDF page with visual on top and commentary below."""
        is_disk_visual = isinstance(fig, (str, os.PathLike))
        if not is_disk_visual and rebrand_figure is not None:
            try:
                rebrand_figure(
                    fig, home_name=hn, away_name=an, score=str(score), accent=C_GOLD
                )
            except Exception:
                pass
        # Prefer the data-driven, connected analyst read; fall back to the
        # generic descriptive commentary only when no data-aware version applies.
        try:
            _ctx = events.attrs.get("_report_ctx")
        except Exception:
            _ctx = None
        if _ctx is None:
            _ctx = _ctx_for(events, info, ppda)
            try:
                events.attrs["_report_ctx"] = _ctx
            except Exception:
                pass
        a_head, a_body = _analyst_commentary(fname or "", hn, an, _ctx)
        if a_body:
            heading, body = a_head, a_body
        else:
            heading, body = _commentary_for_filename(fname or "", hn, an)
            body = _professional_tactical_commentary(fname or "", heading, body, hn, an)
        body = body + "\n\n" + _bridge(_next_of.get(fname))
        try:
            _pdf_page_with_commentary(pdf, fig, heading, body)
        except Exception:
            if not is_disk_visual:
                try:
                    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
                except Exception:
                    pass

    pdf_path = os.path.join(OUTPUT_DIR, f"match_report_{ts}.pdf")

    # Page numbers for the contents list / bookmarks. Page order below:
    #   1 Summary · 2 Glance · 3 Contents · 4 Shared-divider · shared… ·
    #   Home-divider · home… · Away-divider · away… · Glossary · Closing
    # Page order: 1 Cover · 2 Executive Summary · 3 Glance · 4 Contents ·
    # then one divider + visuals per non-empty PHASE · Glossary · Verdict · Closing
    p_summary, p_exec, p_glance, p_contents = 1, 2, 3, 4
    active_phases = [i for i in range(1, 6) if phases[i]]
    _pg = p_contents
    phase_pages = {}  # phase_idx -> divider page number
    for i in active_phases:
        _pg += 1
        phase_pages[i] = _pg
        _pg += len(phases[i])

    # ── Player Radars section (top-5 statistical performers per team) ──
    try:
        import player_radar as _pr

        radar_top = _pr.top_players_per_team(events, info, n=5)
    except Exception:
        _pr, radar_top = None, {"home": [], "away": []}
    n_radar = len(radar_top.get("home", [])) + len(radar_top.get("away", []))
    radar_enabled = bool(_pr) and n_radar > 0
    radar_divider_page = None
    if radar_enabled:
        radar_divider_page = _pg + 1
        _pg = radar_divider_page + n_radar

    p_glossary = _pg + 1
    p_verdict = p_glossary + 1

    toc = [
        ("Match Summary", p_summary, C_GOLD),
        ("Executive Summary", p_exec, C_GOLD),
        ("Match at a Glance", p_glance, C_GOLD),
    ]
    for i in active_phases:
        name, _sub, col = PHASE_INFO[i]
        toc.append((name.title(), phase_pages[i], col))
    if radar_enabled:
        toc.append(("Player Radars", radar_divider_page, C_GOLD))
    toc.append(("Glossary & Methodology", p_glossary, C_GOLD))
    toc.append(("The Verdict", p_verdict, C_GOLD))

    bookmarks = [
        ("Match Summary", p_summary - 1),
        ("Executive Summary", p_exec - 1),
        ("Match at a Glance", p_glance - 1),
        ("Contents", p_contents - 1),
    ]
    for i in active_phases:
        bookmarks.append((PHASE_INFO[i][0].title(), phase_pages[i] - 1))
    if radar_enabled:
        bookmarks.append(("Player Radars", radar_divider_page - 1))
    bookmarks.append(("Glossary & Methodology", p_glossary - 1))
    bookmarks.append(("The Verdict", p_verdict - 1))

    with PdfPages(pdf_path) as pdf:
        # ── Cover page ─────────────────────────────────────────────
        _draw_match_summary_page(pdf, info, goals_df, ppda, events=events)

        # ── Executive Summary (verdict + key numbers + moments) ────
        _draw_executive_summary_page(pdf, info, events, ppda, goals_df)

        # ── Match at a Glance (one-screen dashboard) ───────────────
        _draw_glance_page(pdf, info, events, ppda, goals_df)

        # ── Contents ───────────────────────────────────────────────
        _draw_toc_page(pdf, toc)

        # ── Tactical phases (story → territory → creation → build-up
        #    → defence), each introduced by its own section divider ──
        for k, i in enumerate(active_phases, start=1):
            name, subtitle, col = PHASE_INFO[i]
            _draw_section_divider(pdf, f"{k:02d}", name, subtitle, col)
            for fig, fname in phases[i]:
                _emit_visual(fig, fname)

        # ── Player Radars (top-5 statistical performers per team) ──
        if radar_enabled:
            try:
                _draw_section_divider(
                    pdf,
                    f"{len(active_phases) + 1:02d}",
                    "PLAYER RADARS",
                    "The standout statistical performers of each side — every "
                    "metric scaled against all players on the pitch, grouped "
                    "into attack, passing and defence.",
                    C_GOLD,
                )
                radar_data = _pr.build_report_radars(events, info, OUTPUT_DIR, top_n=5)
                pg = radar_divider_page
                for side in ("home", "away"):
                    accent = C_HOME if side == "home" else C_AWAY
                    tname = radar_data.get(side, {}).get("name", "")
                    for player, pfig, prole, note in radar_data.get(side, {}).get(
                        "figs", []
                    ):
                        pg += 1
                        _draw_player_radar_page(
                            pdf,
                            player,
                            pfig,
                            accent,
                            page_no=pg,
                            team_name=tname,
                            role=prole,
                            commentary=note,
                        )
            except Exception:
                pass

        # ── Glossary & Methodology ─────────────────────────────────
        _draw_glossary_page(pdf)

        # ── The Verdict (report ends on a conclusion) ──────────────
        _draw_verdict_page(pdf, info, events, ppda, goals_df, page_no=p_verdict)

        # ── Closing page (bookend to the cover) ────────────────────
        _draw_closing_page(pdf, info, events=events, ppda=ppda)

    # Clickable bookmarks / outline for navigation.
    _add_pdf_bookmarks(pdf_path, bookmarks)

    # One PNG per report page, in <match folder>/report_pages/ — ready to post.
    report_pages = export_pdf_pages(pdf_path, OUTPUT_DIR, dpi=200)

    return {
        "pdf": pdf_path,
        "extension_pdf": pdf_path,
        "visuals_dir": VISUALS_DIR,
        "report_pages": report_pages,
        "ppda": ppda,
        "goals": goals_df,
        "player_stats": player_stats,
    }
