# pyright: reportMissingImports=false, reportRedeclaration=false, reportReturnType=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPrivateImportUsage=false, reportOptionalMemberAccess=false, reportPossiblyUnboundVariable=false
"""
match_extensions.py
═════════════════════════════════════════════════════════════════════════════
Additive upgrades — styled to match the original Match_Analysis_Dark theme.

  Upgrade 1 — PPDA (Passes per Defensive Action) per team
  Upgrade 2 — Assist names + Open Play / Set Piece goal classification
  Upgrade 3 — Full per-player stats + polished per-team visual tables
  Upgrade 4 — Unified entry point: run_analysis(match_data) -> single PDF

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

# Unified design system (yellow top/side bars, fonts, palette)
try:
    from viz_design_system import apply_unified_frame, rebrand_figure  # type: ignore
except Exception:  # pragma: no cover
    apply_unified_frame = None  # graceful fallback
    rebrand_figure = None


# ─────────────────────────────────────────────────────────────────────────────
# THEME — dark by default; light scripts can set MATCH_ANALYSIS_THEME=light
# before calling this module.
# ─────────────────────────────────────────────────────────────────────────────
def configure_theme(theme: str | None = None) -> None:
    global BG_DARK, BG_MID, BG_PANEL, GRID_COL, TEXT_MAIN, TEXT_BRIGHT, TEXT_DIM, TEXT_FADED, C_GOLD, RATING_CMAP
    theme = (theme or os.environ.get("MATCH_ANALYSIS_THEME", "dark")).strip().lower()
    if theme == "light":
        BG_DARK     = "#FFFFFF"
        BG_MID      = "#F3F4F6"
        BG_PANEL    = "#FFFFFF"
        GRID_COL    = "#D1D5DB"
        TEXT_MAIN   = "#1F2937"
        TEXT_BRIGHT = "#111827"
        TEXT_DIM    = "#4B5563"
        TEXT_FADED  = "#6B7280"
        C_GOLD      = "#D97706"
        RATING_CMAP = LinearSegmentedColormap.from_list(
            "rating",
            ["#FEE2E2", "#FDBA74", "#FACC15", "#86EFAC", "#38BDF8", "#2563EB"],
        )
    else:
        BG_DARK     = "#050508"
        BG_MID      = "#0d1117"
        BG_PANEL    = "#0a0e16"
        GRID_COL    = "#1e2836"
        TEXT_MAIN   = "#f0f4ff"
        TEXT_BRIGHT = "#ffffff"
        TEXT_DIM    = "#94a3b8"
        TEXT_FADED  = "#64748b"
        C_GOLD      = "#facc15"
        RATING_CMAP = LinearSegmentedColormap.from_list(
            "rating",
            ["#3a0f10", "#7a1d1f", "#b45309", "#facc15", "#22c55e", "#0ea5e9"],
        )


configure_theme()

C_HOME      = "#e63946"   # المضيف
C_AWAY      = "#1e90ff"   # الضيف
C_GOLD      = "#D97706" if os.environ.get("MATCH_ANALYSIS_THEME", "dark").strip().lower() == "light" else "#facc15"
C_GREEN     = "#22c55e"
C_PURPLE    = "#a855f7"
OG_COLOR    = "#ff00ff"

# heatmap للتقييمات
# ─────────────────────────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
VISUALS_DIR = os.path.join(OUTPUT_DIR, "visuals")

# Higher-quality raster embedding for visuals inside the unified PDF.
PDF_VISUAL_DPI = 320
PDF_PAGE_DPI = 240


def _ensure_output_dirs() -> None:
    """ينشئ /output و /output/visuals لو مش موجودين."""
    os.makedirs(VISUALS_DIR, exist_ok=True)


def _new_dark_fig(w: float, h: float):
    """ينشئ figure بالـ dark theme المطابق للأصلي."""
    fig = plt.figure(figsize=(w, h), facecolor=BG_DARK)
    return fig


def _style_dark_axes(ax, title: str = "", subtitle: str = ""):
    """يضبط axes على الـ dark theme."""
    ax.set_facecolor(BG_MID)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    if title:
        ax.set_title(title, color=TEXT_BRIGHT, fontsize=14,
                     fontweight="bold", pad=12, loc="left")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _na(value: Any, fmt: str | None = None) -> str:
    """يطبع القيمة أو 'N/A'."""
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
    """يقصّر اسم اللاعب لو طويل."""
    if not name or name == "N/A":
        return name or "N/A"
    if len(name) <= max_len:
        return name
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[-1]}"
    return name[:max_len - 1] + "…"


# ═════════════════════════════════════════════════════════════════════════════
# UPGRADE 1 — PPDA
# ═════════════════════════════════════════════════════════════════════════════
def calculate_ppda(events: pd.DataFrame, team_id: int, opp_id: int,
                   threshold: float = 40.0) -> dict:
    """
    PPDA = (تمريرات الخصم في نص ملعبه) ÷
           (تدخلات + قطع + أخطاء + استخلاصات الفريق الضاغط في نفس المنطقة).

    threshold=40 ⇒ 60% الأمامية للملعب من منظور الفريق الضاغط
    (طريقة Colin Trainor الكلاسيكية).
    """
    if events is None or events.empty:
        return {"passes_allowed": 0, "defensive_actions": 0,
                "ppda": None, "zone_label": "opp own half"}

    opp_threshold = 100.0 - threshold

    # خطوة 1: تمريرات الخصم في نص ملعبه (x < 60 من منظوره)
    opp_passes_mask = (
        (events["team_id"] == opp_id)
        & (events.get("is_pass", False) == True)  # noqa: E712
        & (events["x"].astype(float) < opp_threshold)
    )
    passes_allowed = int(opp_passes_mask.sum())

    # خطوة 2: الأكشنز الدفاعية للفريق الضاغط في نفس المنطقة (x > 40 من منظوره)
    DEF_TYPES = {"Tackle", "Interception", "Foul", "Challenge", "BallRecovery"}
    def_mask = (
        (events["team_id"] == team_id)
        & (events["type"].isin(DEF_TYPES))
        & (events["x"].astype(float) > threshold)
    )
    defensive_actions = int(def_mask.sum())

    # خطوة 3: القسمة (لو 0 أكشنز ⇒ None لتجنّب القسمة على صفر)
    ppda = (passes_allowed / defensive_actions) if defensive_actions > 0 else None

    return {
        "passes_allowed": passes_allowed,
        "defensive_actions": defensive_actions,
        "ppda": ppda,
        "zone_label": f"opp 60% (x<{opp_threshold:.0f} for opp)",
    }


def compute_ppda_both(info: dict, events: pd.DataFrame) -> dict:
    """يحسب PPDA للفريقين معًا."""
    return {
        "home": calculate_ppda(events, info.get("home_id"), info.get("away_id")),
        "away": calculate_ppda(events, info.get("away_id"), info.get("home_id")),
    }


def _ppda_intensity_label(ppda: float | None) -> tuple[str, str]:
    """يصنّف PPDA لمستوى ضغط: Elite/High/Medium/Low + لون."""
    if ppda is None:
        return "N/A", TEXT_FADED
    if ppda < 8.0:
        return "ELITE PRESS", "#22c55e"
    if ppda < 11.0:
        return "HIGH PRESS", "#84cc16"
    if ppda < 14.0:
        return "MEDIUM BLOCK", "#facc15"
    return "LOW BLOCK", "#f97316"


def draw_ppda_gauge(ppda_data: dict, info: dict,
                    save_path: str | None = None):
    """
    يرسم تحليل PPDA كاملًا: لكل فريق dial + breakdown أرقام
    + شرح المنطقة على ملعب صغير + تصنيف intensity.
    """
    fig = _new_dark_fig(14, 8)
    fig.patch.set_facecolor(BG_DARK)

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
        fig.text(0.5, 0.95, "PRESSING ANALYSIS — PPDA",
                 ha="center", color=TEXT_BRIGHT, fontsize=22, fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)])
        fig.text(0.5, 0.91,
                 "Passes Per Defensive Action  •  lower = more aggressive press",
                 ha="center", color=TEXT_DIM, fontsize=11, style="italic")

    # ── Two semi-circular dials ──
    def _draw_dial(ax_pos, name, value, passes, def_acts, color):
        ax = fig.add_axes(ax_pos, projection="polar")
        ax.set_facecolor(BG_DARK)
        # نص الدائرة من 0 لـ π (نصف دائرة)
        ax.set_theta_zero_location("W")
        ax.set_theta_direction(-1)
        ax.set_thetamin(0); ax.set_thetamax(180)
        ax.set_ylim(0, 1)

        # الـ scale من 5 (ضغط ممتاز) لـ 25 (مفيش ضغط)
        v = value if value is not None else 0
        vmin, vmax = 5.0, 25.0
        # mapping: ppda=5 ⇒ 0° (شمال), ppda=25 ⇒ 180° (يمين)
        ratio = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
        angle = ratio * np.pi

        # خلفية الـ arc — متدرّج
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
            ax.bar((thetas[i] + thetas[i+1]) / 2, 0.18, bottom=0.78,
                   width=(thetas[i+1] - thetas[i]) * 0.95,
                   color=zone_colors[i], edgecolor="none",
                   alpha=0.75)

        # المؤشر (الإبرة)
        if value is not None:
            ax.plot([angle, angle], [0, 0.92],
                    color=color, lw=4,
                    solid_capstyle="round", zorder=5)
            ax.scatter([angle], [0], s=140, color=color,
                       edgecolor=TEXT_BRIGHT, linewidth=1.5, zorder=6)
            ax.scatter([angle], [0.92], s=70, color=TEXT_BRIGHT,
                       edgecolor=color, linewidth=2, zorder=7)

        # شيل الـ ticks/labels الافتراضية
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["polar"].set_visible(False)

        # علامات رقمية
        for tick_v, tick_label in [(5, "5"), (10, "10"), (15, "15"),
                                    (20, "20"), (25, "25")]:
            r = (tick_v - vmin) / (vmax - vmin)
            theta_t = r * np.pi
            ax.text(theta_t, 1.05, tick_label,
                    ha="center", va="center",
                    color=TEXT_DIM, fontsize=8.5)

        # القيمة الكبيرة في النص
        cx, cy = ax_pos[0] + ax_pos[2] / 2, ax_pos[1] + 0.06
        val_str = f"{value:.2f}" if value is not None else "N/A"
        fig.text(cx, cy + 0.04, val_str,
                 ha="center", color=color, fontsize=36, fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)])

        intensity_lbl, intensity_col = _ppda_intensity_label(value)
        fig.text(cx, cy, intensity_lbl,
                 ha="center", color=intensity_col, fontsize=11,
                 fontweight="bold")

        # اسم الفريق فوق
        fig.text(cx, ax_pos[1] + ax_pos[3] - 0.02, name,
                 ha="center", color=TEXT_BRIGHT,
                 fontsize=15, fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=2, foreground=BG_DARK)])

        # breakdown أرقام تحت
        fig.text(cx - 0.06, cy - 0.05, str(passes),
                 ha="center", color=color, fontsize=18, fontweight="bold")
        fig.text(cx - 0.06, cy - 0.085, "OPP PASSES",
                 ha="center", color=TEXT_DIM, fontsize=8, fontweight="bold")
        fig.text(cx + 0.06, cy - 0.05, str(def_acts),
                 ha="center", color=color, fontsize=18, fontweight="bold")
        fig.text(cx + 0.06, cy - 0.085, "DEF ACTIONS",
                 ha="center", color=TEXT_DIM, fontsize=8, fontweight="bold")

    _draw_dial([0.05, 0.42, 0.40, 0.45], home_name, h, h_passes, h_def, C_HOME)
    _draw_dial([0.55, 0.42, 0.40, 0.45], away_name, a, a_passes, a_def, C_AWAY)

    # ── Comparison strip في النص ──
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
        fig.text(0.5, 0.34, verdict,
                 ha="center", color=v_color, fontsize=14, fontweight="bold")
        if diff > 0:
            fig.text(0.5, 0.305, f"PPDA differential: {diff:.2f}",
                     ha="center", color=TEXT_DIM, fontsize=10)

    # ── Legend زون pitch بسيط ──
    pitch_ax = fig.add_axes([0.10, 0.06, 0.80, 0.18])
    pitch_ax.set_facecolor("#040c04")
    pitch_ax.set_xlim(0, 100); pitch_ax.set_ylim(0, 30)
    pitch_ax.set_xticks([]); pitch_ax.set_yticks([])
    for s in pitch_ax.spines.values():
        s.set_edgecolor(GRID_COL)
    # خط النص
    pitch_ax.plot([50, 50], [0, 30], color=TEXT_DIM, lw=1, ls="--", alpha=0.6)
    # خط 60
    pitch_ax.axvline(60, color="#facc15", lw=1.2, alpha=0.7, ls="--")
    # المنطقة الدفاعية للخصم (60%)
    pitch_ax.add_patch(mpatches.Rectangle(
        (60, 0), 40, 30, facecolor="#facc15", alpha=0.15, lw=0))
    pitch_ax.text(80, 15, "PRESSING ZONE\n(opp 60% of pitch)",
                  ha="center", va="center", color="#facc15",
                  fontsize=10, fontweight="bold")
    pitch_ax.text(30, 15, "OWN HALF",
                  ha="center", va="center", color=TEXT_FADED,
                  fontsize=9, style="italic")
    pitch_ax.text(2, 27, "← own goal", color=TEXT_FADED, fontsize=8)
    pitch_ax.text(98, 27, "opp goal →", color=TEXT_FADED, fontsize=8, ha="right")

    fig.text(0.5, 0.025,
             "Method: opponent passes attempted in their own 60% "
             "÷ tackles + interceptions + fouls + challenges + recoveries  "
             "(Colin Trainor, 2014)",
             ha="center", color=TEXT_FADED, fontsize=8.5, style="italic")

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight",
                    facecolor=BG_DARK)
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


def _previous_restart_subtype(row: pd.Series, events: pd.DataFrame | None = None) -> str | None:
    if events is None or getattr(events, "empty", True):
        return None
    try:
        idx = row.name
        if idx in events.index:
            pos = events.index.get_loc(idx)
            if isinstance(pos, slice):
                pos = pos.start
            prior = events.iloc[:int(pos)]
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
    same_team = prior[prior.get("team_id") == team_id] if "team_id" in prior.columns else prior
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


def classify_goal_type(row: pd.Series, events: pd.DataFrame | None = None) -> tuple[str, str]:
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
    """يبني جدول الأهداف بالصانعين والتصنيف."""
    cols = ["minute", "team", "scorer", "assist", "category", "subtype",
            "xG", "is_own_goal"]
    if events is None or events.empty:
        return pd.DataFrame(columns=cols)

    gdf = events[events["is_goal"] == True].copy()  # noqa: E712
    if gdf.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, r in gdf.sort_values("minute").iterrows():
        category, subtype = classify_goal_type(r, events)
        scoring_team_id = r.get("scoring_team", r.get("team_id"))
        team_name = (info.get("home_name") if scoring_team_id == info.get("home_id")
                     else info.get("away_name"))
        assist = r.get("assist_player") or ""
        rows.append({
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
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# UPGRADE 3 — Player stats extraction & polished tables
# ═════════════════════════════════════════════════════════════════════════════
def _flatten_stat(value: Any) -> Any:
    """يحوّل WhoScored stat dict لقيمة واحدة."""
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


# تجميع الإحصائيات لمجموعات منطقية — بيظهر في رؤوس الجدول
STAT_GROUPS = [
    ("Identity",  [("name",          "Player"),
                   ("position",      "Pos"),
                   ("shirt_no",      "#"),
                   ("minutesPlayed", "Min")]),
    ("Attack",    [("goals",         "G"),
                   ("assists",       "A"),
                   ("shotsTotal",    "Sh"),
                   ("shotsOnTarget", "SoT"),
                   ("passesKey",     "KP"),
                   ("dribblesWon",   "Drb")]),
    ("Passing",   [("passesTotal",   "Pass"),
                   ("passesAccurate","Acc"),
                   ("touches",       "Tch")]),
    ("Defense",   [("tacklesTotal",  "Tkl"),
                   ("interceptions", "Int"),
                   ("aerialsWon",    "Aer"),
                   ("foulsCommited", "Fls"),
                   ("wasFouled",     "Fld")]),
    ("Score",     [("ratings",       "Rating")]),
]

# ألوان رؤوس المجموعات
GROUP_HEADER_COLORS = {
    "Identity": "#1f2a3a",
    "Attack":   "#3b1f2f",
    "Passing":  "#1f3a2f",
    "Defense":  "#3a2f1f",
    "Score":    "#2a1f3a",
}

GROUP_HEADER_COLORS_LIGHT = {
    "Identity": "#E0F2FE",
    "Attack":   "#FCE7F3",
    "Passing":  "#DCFCE7",
    "Defense":  "#FEF3C7",
    "Score":    "#EDE9FE",
}


def extract_player_stats(md: dict) -> dict:
    """يستخرج كل إحصائيات اللاعبين للفريقين."""
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
            df = df.sort_values(sort_cols, ascending=ascending,
                                na_position="last").reset_index(drop=True)
        out[side] = df

    return out


def _rating_color(rating: Any) -> str:
    """خريطة لون من التقييم (5.0 → 8.5+)."""
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return BG_PANEL
    # normalize 5..9 → 0..1
    t = max(0.0, min(1.0, (r - 5.0) / 4.0))
    rgba = RATING_CMAP(t)
    return f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"


def _format_cell(key: str, value: Any) -> str:
    """يصيغ القيمة حسب نوع العمود."""
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


def draw_player_stats_table(df: pd.DataFrame, team_name: str,
                            team_color: str = C_HOME,
                            save_path: str | None = None):
    """
    يرسم جدول إحصائيات لاعبين بأعمدة بعرض متغيّر (Player أعرض من بقية
    الأعمدة)، مجمّع بمجموعات، dark theme، مع heatmap للتقييم.
    """
    # عرض كل عمود بالـ "units" — Player أعرض، Pos متوسط، الباقي ضيق
    COL_W = {
        "name": 3.6, "position": 1.05, "shirt_no": 0.55,
        "minutesPlayed": 0.7, "ratings": 1.15,
    }
    DEFAULT_W = 0.85  # لباقي الإحصائيات

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
            flat_keys.append(k); flat_labels.append(lbl); flat_widths.append(w)
            x_cursor += w
        group_spans.append((gname, x_start, x_cursor))

    total_w = x_cursor if x_cursor > 0 else 1.0
    # x positions (left edges) لكل عمود
    x_lefts = []
    acc = 0.0
    for w in flat_widths:
        x_lefts.append(acc); acc += w

    n_rows = max(len(df), 1)

    # حجم الـ figure متناسب مع total_w
    fig = _new_dark_fig(max(15, total_w * 0.95),
                        max(7, 0.42 * n_rows + 3.0))
    fig.patch.set_facecolor(BG_DARK)

    # ── Title bar ──
    fig.text(0.04, 0.965, f"PLAYER STATISTICS — {team_name.upper()}",
             color=TEXT_BRIGHT, fontsize=18, fontweight="bold")
    fig.text(0.04, 0.935,
             "Starters first  •  ratings coloured by performance  •  "
             "G / A highlighted  •  '—' = unavailable",
             color=TEXT_DIM, fontsize=10, style="italic")

    bar_ax = fig.add_axes([0.04, 0.91, 0.92, 0.012])
    bar_ax.set_facecolor(team_color)
    bar_ax.set_xticks([]); bar_ax.set_yticks([])
    for s in bar_ax.spines.values():
        s.set_visible(False)

    # ── Main table axes ──
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.83])
    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, n_rows + 2.4)
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL)

    if df.empty:
        ax.text(total_w / 2, n_rows / 2 + 1.2, "No player data available",
                ha="center", va="center", color=TEXT_DIM, fontsize=14,
                style="italic")
        if save_path:
            fig.savefig(save_path, dpi=160, bbox_inches="tight",
                        facecolor=BG_DARK)
        return fig

    # ── Group header row (y=0..1) ──
    is_light_theme = BG_DARK.upper() in {"#FFFFFF", "WHITE"}
    group_header_colors = GROUP_HEADER_COLORS_LIGHT if is_light_theme else GROUP_HEADER_COLORS
    for gname, x0, x1 in group_spans:
        ax.add_patch(mpatches.Rectangle(
            (x0, 0), x1 - x0, 1.0,
            facecolor=group_header_colors.get(gname, BG_MID),
            edgecolor=GRID_COL, lw=0.6,
        ))
        ax.text((x0 + x1) / 2, 0.5, gname.upper(),
                ha="center", va="center",
                color=TEXT_BRIGHT, fontsize=10.5, fontweight="bold",
                path_effects=[] if is_light_theme else [pe.withStroke(linewidth=2, foreground=BG_DARK)])

    # ── Column header row (y=1..2.2) ──
    for j, lbl in enumerate(flat_labels):
        x0 = x_lefts[j]; w = flat_widths[j]
        ax.add_patch(mpatches.Rectangle(
            (x0, 1.0), w, 1.2, facecolor=BG_MID,
            edgecolor=GRID_COL, lw=0.5,
        ))
        if flat_keys[j] == "name":
            ha = "left"; tx = x0 + 0.55
        else:
            ha = "center"; tx = x0 + w / 2
        ax.text(tx, 1.6, lbl, ha=ha, va="center",
                color=TEXT_DIM, fontsize=9.5, fontweight="bold")

    # ── Data rows ──
    y0 = 2.2
    row_h = 1.0
    for i, (_, r) in enumerate(df.iterrows()):
        y = y0 + i * row_h
        is_starter = bool(r.get("is_first_xi", False))

        base_color = BG_PANEL if i % 2 == 0 else ("#F8FAFC" if is_light_theme else "#0f1520")
        if not is_starter:
            base_color = "#F1F5F9" if is_light_theme else "#080a10"
        ax.add_patch(mpatches.Rectangle(
            (0, y), total_w, row_h, facecolor=base_color,
            edgecolor=GRID_COL, lw=0.3, alpha=0.95,
        ))

        # شريط جانبي للأساسيين
        starter_col = team_color if is_starter else TEXT_FADED
        ax.add_patch(mpatches.Rectangle(
            (0, y), 0.10, row_h, facecolor=starter_col, lw=0,
            alpha=0.9 if is_starter else 0.35,
        ))

        for j, key in enumerate(flat_keys):
            x0 = x_lefts[j]; w = flat_widths[j]
            raw = r.get(key)
            text = _format_cell(key, raw)
            text_color = TEXT_MAIN
            fontweight = "normal"

            if key == "name":
                # اسم بمحاذاة يسار، يقص بحسب عرض العمود
                max_chars = int(w * 7)
                disp = _short_name(str(raw or "N/A"), max_chars)
                ax.text(x0 + 0.20, y + row_h / 2, disp,
                        ha="left", va="center",
                        color=TEXT_BRIGHT if is_starter else TEXT_DIM,
                        fontsize=10,
                        fontweight="bold" if is_starter else "normal")
                continue

            if key == "position":
                if text and text != "—":
                    pad_x = 0.12
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (x0 + pad_x, y + 0.22), w - 2 * pad_x, row_h - 0.44,
                        boxstyle="round,pad=0.02,rounding_size=0.10",
                        facecolor="#E5E7EB" if is_light_theme else GRID_COL,
                        edgecolor=team_color, lw=0.7,
                    ))
                ax.text(x0 + w / 2, y + row_h / 2, text,
                        ha="center", va="center",
                        color=TEXT_BRIGHT, fontsize=9, fontweight="bold")
                continue

            if key == "ratings" and raw is not None:
                rc = _rating_color(raw)
                pad_x = 0.10
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x0 + pad_x, y + 0.18), w - 2 * pad_x, row_h - 0.36,
                    boxstyle="round,pad=0.02,rounding_size=0.12",
                    facecolor=rc, edgecolor="white", lw=0.7,
                ))
                ax.text(x0 + w / 2, y + row_h / 2, text,
                        ha="center", va="center",
                        color="#0a0a0a" if _is_light(rc) else TEXT_BRIGHT,
                        fontsize=10, fontweight="bold")
                continue

            if key == "goals" and raw and float(raw) > 0:
                text_color = C_GOLD; fontweight = "bold"
            elif key == "assists" and raw and float(raw) > 0:
                text_color = C_GREEN; fontweight = "bold"
            elif text in ("—", "N/A"):
                text_color = TEXT_FADED

            ax.text(x0 + w / 2, y + row_h / 2, text,
                    ha="center", va="center",
                    color=text_color, fontsize=9.2, fontweight=fontweight)

    # ── Footer legend ──
    fig.text(
        0.04, 0.025,
        "Starters: bold + coloured side-bar    •    Substitutes/unused: dimmed    "
        "•    Rating cell: red→yellow→green→blue (5.0 → 9.0)",
        color=TEXT_FADED, fontsize=8.5, style="italic",
    )

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight",
                    facecolor=BG_DARK)
    return fig


def _is_light(hex_color: str) -> bool:
    """يحدد لو اللون فاتح عشان يختار لون نص مناسب."""
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
def _draw_section_divider(pdf, num: str, title: str, subtitle: str,
                          accent: str = C_GOLD):
    """صفحة فاصلة لكل قسم في التقرير."""
    fig = _new_dark_fig(11.7, 8.27)
    fig.patch.set_facecolor(BG_DARK)

    # خط ملوّن طولي على الشمال
    bar_ax = fig.add_axes([0.06, 0.20, 0.008, 0.60])
    bar_ax.set_facecolor(accent)
    bar_ax.set_xticks([]); bar_ax.set_yticks([])
    for s in bar_ax.spines.values():
        s.set_visible(False)

    # رقم القسم — كبير جدًا خافت
    fig.text(0.10, 0.55, num,
             color=accent, fontsize=140, fontweight="bold",
             alpha=0.18, family="serif")

    # عنوان القسم
    fig.text(0.10, 0.62, f"SECTION {num}",
             color=accent, fontsize=12, fontweight="bold",
             family="sans-serif")
    fig.text(0.10, 0.55, title,
             color=TEXT_BRIGHT, fontsize=36, fontweight="bold",
             path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)])
    fig.text(0.10, 0.49, subtitle,
             color=TEXT_DIM, fontsize=12.5, style="italic")

    # خط فاصل
    line_ax = fig.add_axes([0.10, 0.45, 0.40, 0.002])
    line_ax.set_facecolor(accent)
    line_ax.set_xticks([]); line_ax.set_yticks([])
    for s in line_ax.spines.values():
        s.set_visible(False)

    # توقيع
    fig.text(0.10, 0.20, "M A T C H   A N A L Y S I S   R E P O R T",
             color=TEXT_FADED, fontsize=8, fontweight="bold")

    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _draw_closing_page(pdf, info):
    """Minimal closing page."""
    fig = _new_dark_fig(8.27, 11.69)
    fig.patch.set_facecolor(BG_DARK)

    fig.text(
        0.5, 0.50,
        "End of report by Mostafa Saad",
        ha="center", va="center",
        color=TEXT_DIM, fontsize=10,
        family="serif",
    )

    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _draw_match_summary_page(pdf, info, goals_df, ppda):
    """صفحة كوفر بنفس الـ dark theme."""
    fig = _new_dark_fig(11.7, 8.27)
    home = info.get("home_name") or "Home"
    away = info.get("away_name") or "Away"
    score = info.get("score") or "? - ?"
    venue = info.get("venue") or "N/A"
    date = datetime.now().strftime("%Y-%m-%d %H:%M")

    fig.text(0.5, 0.92, "MATCH ANALYSIS REPORT",
             ha="center", color=TEXT_BRIGHT,
             fontsize=24, fontweight="bold",
             path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)])
    fig.text(0.5, 0.875, "Extended Tactical Pack",
             ha="center", color=TEXT_DIM, fontsize=11,
             style="italic")

    # السكور
    fig.text(0.27, 0.74, home, ha="right", color=C_HOME,
             fontsize=22, fontweight="bold",
             path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)])
    fig.text(0.50, 0.74, score, ha="center", color=TEXT_BRIGHT,
             fontsize=30, fontweight="bold")
    fig.text(0.73, 0.74, away, ha="left", color=C_AWAY,
             fontsize=22, fontweight="bold",
             path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)])

    fig.text(0.5, 0.66, f"Venue: {venue}    •    Generated: {date}",
             ha="center", color=TEXT_DIM, fontsize=11)

    # كروت أرقام
    h_ppda = ppda.get("home", {}).get("ppda")
    a_ppda = ppda.get("away", {}).get("ppda")
    n_goals = len(goals_df)
    sp = int((goals_df["category"] == "Set Piece").sum()) if not goals_df.empty else 0
    op = n_goals - sp

    cards = [
        ("GOALS", str(n_goals), TEXT_BRIGHT),
        ("OPEN PLAY", str(op), C_GREEN),
        ("SET PIECE", str(sp), C_GOLD),
        (f"PPDA · {home[:14]}", _na(h_ppda, ".2f"), C_HOME),
        (f"PPDA · {away[:14]}", _na(a_ppda, ".2f"), C_AWAY),
    ]
    n = len(cards)
    card_w = 0.16
    gap = (1 - n * card_w) / (n + 1)
    for i, (label, val, col) in enumerate(cards):
        x0 = gap + i * (card_w + gap)
        ax = fig.add_axes([x0, 0.30, card_w, 0.20])
        ax.set_facecolor(BG_MID)
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.65, val, ha="center", va="center",
                color=col, fontsize=22, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.22, label, ha="center", va="center",
                color=TEXT_DIM, fontsize=8.5, fontweight="bold",
                transform=ax.transAxes)

    fig.text(0.5, 0.18,
             f"Formations — {home}: {_na(info.get('home_form'))}    "
             f"|    {away}: {_na(info.get('away_form'))}",
             ha="center", color=TEXT_DIM, fontsize=10.5)

    fig.text(0.5, 0.05,
             f"01 Player Ratings   ·   02 {info.get('home_name','Home')} "
             f"Analysis   ·   03 {info.get('away_name','Away')} Analysis"
             f"   ·   04 Shared Insights",
             ha="center", color=TEXT_FADED, fontsize=9, style="italic")

    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _draw_goals_log_page(pdf, goals_df, info):
    """Goals log with the unified visual identity."""
    fig = _new_dark_fig(14, 9)
    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "—"
    if apply_unified_frame is not None:
        apply_unified_frame(
            fig,
            section="GOALS LOG",
            title=f"Goals Log — {hn} vs {an}",
            subtitle="Every goal with scorer, assist, goal category and "
                     "shot xG · own goals shown in magenta",
            accent=C_GOLD,
            home_name=hn, away_name=an, score=str(score),
            footer_note="Open Play (green) · Set Piece (gold) · Own Goal "
                        "(magenta)",
        )
    else:
        fig.text(0.04, 0.94, "GOALS LOG",
                 color=TEXT_BRIGHT, fontsize=20, fontweight="bold")
        fig.text(0.04, 0.91, "Scorer  ·  Assist  ·  Goal type",
                 color=TEXT_DIM, fontsize=10, style="italic")

    if goals_df.empty:
        fig.text(0.5, 0.5, "No goals recorded.",
                 ha="center", color=TEXT_DIM, fontsize=14, style="italic")
        pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
        plt.close(fig)
        return

    headers = [("MIN", 0.05, "center"),
               ("TEAM", 0.13, "left"),
               ("SCORER", 0.29, "left"),
               ("ASSIST", 0.49, "left"),
               ("CATEGORY", 0.68, "left"),
               ("DETAIL", 0.80, "left"),
               ("BODY", 0.91, "left"),
               ("xG", 0.98, "right")]
    ax = fig.add_axes([0.0, 0.05, 1.0, 0.83])
    ax.set_facecolor(BG_DARK); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    y = 0.94
    for lbl, x, ha in headers:
        ax.text(x, y, lbl, ha=ha, va="center",
                color=TEXT_DIM, fontsize=9, fontweight="bold",
                transform=ax.transAxes)
    y -= 0.018
    ax.plot([0.03, 0.99], [y, y], color=GRID_COL, lw=0.8,
            transform=ax.transAxes)

    n = len(goals_df)
    row_h = min(0.78 / max(n, 1), 0.07)
    y -= 0.012

    for _, r in goals_df.iterrows():
        is_og = bool(r.get("is_own_goal", False))
        team_id = r.get("scoring_team_id")
        col = (OG_COLOR if is_og
               else (C_HOME if team_id == info.get("home_id") else C_AWAY))

        bg = "#1a0a0a" if col == C_HOME else ("#060f1e" if col == C_AWAY else "#1e0a2e")
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, y - row_h * 0.92), 0.96, row_h * 0.85,
            boxstyle="round,pad=0.005,rounding_size=0.005",
            facecolor=bg, edgecolor=col, lw=0.8, alpha=0.92,
            transform=ax.transAxes,
        ))
        cy = y - row_h * 0.5

        # شارة الدقيقة
        ax.text(0.05, cy, f"{_safe_int(r['minute'])}'",
                ha="center", va="center",
                color="white", fontsize=10, fontweight="bold",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.30",
                          facecolor=col, edgecolor="none"))

        ax.text(0.13, cy, _short_name(str(r["team"]), 16),
                ha="left", va="center",
                color=col, fontsize=10, fontweight="bold",
                transform=ax.transAxes)
        scorer_label = _short_name(str(r["scorer"]), 20)
        if is_og:
            scorer_label += "  (OG)"
        ax.text(0.29, cy, scorer_label,
                ha="left", va="center",
                color=TEXT_BRIGHT, fontsize=10, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.49, cy, _short_name(str(r["assist"]), 18),
                ha="left", va="center",
                color=TEXT_DIM, fontsize=9.5, transform=ax.transAxes)

        cat_col = C_GREEN if r["category"] == "Open Play" else C_GOLD
        ax.text(0.68, cy, r["category"],
                ha="left", va="center",
                color=cat_col, fontsize=9.5, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.80, cy, _short_name(str(r["subtype"]), 14),
                ha="left", va="center",
                color=TEXT_MAIN, fontsize=9.5, transform=ax.transAxes)
        ax.text(0.91, cy, _short_name(str(r.get("body_part", "Unknown")), 12),
                ha="left", va="center",
                color=TEXT_MAIN, fontsize=9.5, transform=ax.transAxes)
        xg_txt = f"{r['xG']:.2f}" if isinstance(r["xG"], (int, float)) and r["xG"] else "—"
        ax.text(0.98, cy, xg_txt,
                ha="right", va="center",
                color=C_GOLD, fontsize=9.5, transform=ax.transAxes)

        y -= row_h

    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _draw_ppda_page(pdf, ppda, info, visuals_dir):
    """صفحة PPDA — لا تحفظ PNG (الفيجوال بيتولّد كـ fig 40 من Dark.py)."""
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
        side = "home"; team = hn
    elif detected_side == "away":
        side = "away"; team = an
    else:
        side, team = "shared", None

    if "xg_flow" in f:
        return (
            "Reading the xG Flow",
            f"The xG Flow plots cumulative Expected Goals minute by minute "
            f"as a staircase, with every step marking a shot whose height "
            f"equals its chance quality. Stars sit on goals; the shaded "
            f"territory under each curve is total chance creation. A team "
            f"that pulls clearly above the other built the stronger shot "
            f"profile, even if the scoreline says otherwise."
        )
    if "shot_map" in f:
        return (
            f"Reading {team}'s Shot Map",
            f"Each dot is a single shot, located at where it was struck. "
            f"Marker size scales with xG, so big circles are big chances. "
            f"Filled circles are goals, hollow ones are misses. Cluster "
            f"density inside the box shows where {team} manufactured looks; "
            f"wide-area dots usually indicate speculative efforts that "
            f"rarely trouble keepers."
        )
    if "breakdown_goals" in f:
        return (
            "Reading the Shot Breakdown",
            "The bar group counts every shot by outcome — total, woodwork, "
            "on target, off target, blocked. On-target conversion (goals "
            "divided by shots on target) reflects finishing efficiency. "
            "The Goals & Assists table below records the scorer, the play "
            "type (Open Play / Set Piece / Penalty), the assister and the "
            "chance's xG."
        )
    if "pass_network" in f:
        return (
            f"Reading {team}'s Pass Network",
            f"Each node is a player placed at their average pass position. "
            f"Node size scales with passes attempted; line width between "
            f"two players scales with the volume of completed passes "
            f"between them. Heavy lines reveal {team}'s preferred "
            f"partnerships and which axis the build-up flowed through. "
            f"Top-8 partnerships also carry their pass count for quick "
            f"reading."
        )
    if "xt_map" in f:
        return (
            f"Reading {team}'s xT Map",
            f"The grid colours each pitch zone by its xT (Expected Threat) "
            f"value — the probability that owning the ball there leads to "
            f"a shot in the next few seconds. White arrows are positive-xT "
            f"passes (gained threat); red arrows are negative-xT passes "
            f"(gave threat back). The five gold arrows highlight {team}'s "
            f"five highest-xT progressive passes — usually the moments "
            f"that broke a defensive line."
        )
    if "shot_comparison" in f:
        return (
            "Reading the Shot Comparison",
            "A side-by-side bar chart of the headline shooting numbers — "
            "total shots, on target, big chances, xG, xGoT. The fastest "
            "single view to answer: who was the more dangerous attacking "
            "side? Gold-coloured numbers mark the metric leader."
        )
    if "danger" in f and "home" in f or "danger" in f and "away" in f or \
       "danger_" in f:
        return (
            f"Reading {team}'s Danger Creation",
            f"Every action that ended in a shot, key pass or box entry "
            f"is plotted to show which channels generated {team}'s "
            f"high-value moments. Concentrate on the warm zones — those "
            f"are {team}'s true danger lanes. Diamonds are key passes; "
            f"circles are shots; faint arrows are box entries underneath."
        )
    if "gk_saves" in f:
        return (
            "Reading the Goalkeeper Saves",
            "Plots the location of every shot each keeper faced, with "
            "marker size scaled to the chance's xG. Filled stars are goals "
            "conceded; rings are saves. Save quality is a function of the "
            "xG of the shots faced — saving a high-xG strike beats "
            "stopping easy long-range efforts."
        )
    if "xg_tiles" in f or "xg_summary" in f:
        return (
            "Reading the xG and xGoT Summary",
            "xG measures pre-shot chance quality; xGoT (Expected Goals on "
            "Target) measures post-shot quality once placement and power "
            "are known. A team with xGoT well above xG executed their "
            "finishing better than average; below xG points to wasteful "
            "conversion."
        )
    if "zone14" in f:
        return (
            f"Reading {team}'s Zone 14 & Half-Spaces",
            f"Zone 14 is the central pocket just outside the box — "
            f"historically the richest zone for chance creation. The "
            f"half-spaces flank it. This map counts every action {team} "
            f"completed in those zones; volume here is a proxy for how "
            f"often they reached the most dangerous central real estate."
        )
    if "match_stats" in f:
        return (
            "Reading the Match Statistics",
            "A consolidated head-to-head: Attack (goals, shots, passes, "
            "key passes), Defense (tackles, interceptions, blocks, "
            "clearances, recoveries, fouls) and Pressing (PPDA mini-dials "
            "with intensity verdict). Read it as the single-page summary "
            "of the entire match."
        )
    if "territorial" in f or "possession" in f or "ball_touches" in f:
        return (
            "Reading the Ball Touches",
            "Splits the pitch into zones and reads which side had more "
            "touches in each. Red dominance signals home-team control of "
            "that area; blue signals away. The donut totals beneath show "
            "the overall share of touches — whoever owned the territory "
            "owned the game."
        )
    if "pass_thirds" in f:
        return (
            f"Reading {team}'s Pass Map by Third",
            f"Splits passes into defensive, middle and attacking-third "
            f"buckets and visualises each on the pitch. Final-third pass "
            f"density and completion rate are the two quickest reads of "
            f"{team}'s break-down activity in the opposition area."
        )
    if "xt_per_minute" in f:
        return (
            "Reading xT per Minute",
            "A diverging bar chart: home xT bars rise above the zero line, "
            "away xT drops below it. Tall spikes mark the moments each "
            "side surged in threat creation. The 5-minute rolling average "
            "overlay smooths the noise and shows momentum windows."
        )
    if "progressive" in f:
        return (
            f"Reading {team}'s Progressive Passes",
            f"Plots every pass that closed at least 25% of the distance "
            f"to goal (or any pass into the final third). The five gold "
            f"arrows highlight {team}'s top forward gains by raw distance. "
            f"Wide-spread arrows mean the progression load was shared; "
            f"concentration around one source marks a single play-maker."
        )
    if "crosses" in f:
        return (
            f"Reading {team}'s Crosses",
            f"Every cross into the box plotted with origin and end point. "
            f"Solid arrows are successful, faded ones are unsuccessful. "
            f"Cross volume and the flank split (left vs right) reveal "
            f"{team}'s wide-attack channel and how much of {team}'s box "
            f"entry came through wide play versus central combinations."
        )
    if "defensive_hm" in f:
        return (
            f"Reading {team}'s Defensive Heatmap",
            f"A density map of every defensive action {team} completed — "
            f"tackles, interceptions, clearances, blocks, recoveries, "
            f"fouls. Hot zones reveal {team}'s defensive line height: "
            f"high up the pitch (press) or deep in their own block."
        )
    if "defensive_summary" in f:
        return (
            "Reading the Defensive Summary",
            "Six defensive-action types — head-to-head counts. Tackles "
            "and interceptions describe ground duels; blocks count shots "
            "stopped by a body in the way; recoveries and clearances mark "
            "how each side escaped pressure; fouls show where containment "
            "broke down. Gold labels mark the side leading on each metric."
        )
    if "avg_position" in f:
        return (
            f"Reading {team}'s Average Positions",
            f"Each player placed at their mean touch position, with node "
            f"size scaled to total touches. Faint lines connect every "
            f"node to the team centroid so the overall shape pops out: a "
            f"high, narrow shape signals an aggressive pressing block; a "
            f"deep, wide shape points to a low-block defensive setup."
        )
    if "dominating_zone" in f:
        return (
            "Reading the Dominating Zone",
            "Each grid cell is coloured by the team that had more touches "
            "there. Block colour reveals territorial dominance at a glance "
            "— large contiguous regions in one team's colour mark areas "
            "they controlled outright. Touch counts on every cell make "
            "the heatmap quantitative."
        )
    if "box_entries" in f:
        return (
            f"Reading {team}'s Box Entries",
            f"Every successful entry into the opposition penalty area — "
            f"pass (gold) or carry (green). Clusters near the byline mean "
            f"wide-and-cut-back access; clusters at the D mean central, "
            f"through-ball access. The total count is a direct measure "
            f"of {team}'s break-down volume."
        )
    if "high_turnovers" in f:
        return (
            f"Reading {team}'s High Turnovers",
            f"Marks every regain of possession in the final 40 metres. "
            f"Frequent high turnovers are a hallmark of a successful "
            f"press — the more concentrated the dots near the opposition "
            f"box, the more often {team} won the ball in dangerous areas."
        )
    if "pass_target" in f:
        return (
            f"Reading {team}'s Pass Target Zones",
            f"Heatmap of where {team}'s successful passes landed — i.e. "
            f"their preferred receiving zones. Compare with the pass "
            f"network to see whether the receiving pattern matches the "
            f"network's shape, or whether passes were aiming for "
            f"under-served outlets."
        )
    if "ppda" in f:
        return (
            "Reading the PPDA Gauge",
            "PPDA (Passes per Defensive Action) measures pressing "
            "intensity. Low PPDA = aggressive press (fewer opponent "
            "passes allowed before forcing a defensive action). High PPDA "
            "= deeper block. The dial reads green for an aggressive "
            "press and slides toward orange for a low-block setup."
        )
    if "player_stats" in f:
        return (
            f"Reading {team or 'the squad'}'s Player Stats",
            f"Per-player totals across the match — minutes, touches, "
            f"shots, passes attempted/completed, key passes, defensive "
            f"actions, and a colour-coded performance rating. Starters "
            f"appear first, substitutes follow."
        )
    return (
        "Reading this visual",
        "This chart adds context to the tactical story. Read it alongside "
        "the rest of the report — chance quality, territory, pressing "
        "and ball progression all reinforce the same narrative when the "
        "numbers line up."
    )


def _professional_tactical_commentary(fname: str, heading: str, body: str,
                                      hn: str, an: str) -> str:
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
            f"The key coaching read is the timing of the separations. If {hn} or {an} created distance early, the game state may have allowed them to manage risk afterwards. If the curve changes late, it points to substitutions, fatigue, or a structural adjustment that finally opened access to goal."
        )
    if "shot_map" in f:
        return _join(
            opening,
            f"For {team}, the quality of the shot locations matters more than the count. Central shots inside the box suggest the attack is breaking the defensive line or finding cut-backs. Shots from wide or long range usually mean {opponent} protected the middle and forced lower-value decisions.",
            "The professional read is whether the team created repeatable chances. One large dot can come from a single transition; several good locations from similar zones suggest a deliberate route to goal. That is the difference between attacking noise and a real chance-creation pattern."
        )
    if "breakdown_goals" in f:
        return _join(
            opening,
            "This page separates volume from efficiency. A team can shoot often without attacking well if most attempts are blocked or off target. By contrast, fewer shots with a high on-target share usually point to cleaner entries, better final actions, and calmer finishing decisions.",
            "The goals and assists table gives the human layer of the story: who finished, who supplied the final pass, and whether the goal came from open play, a set piece, or a penalty. That makes it easier to distinguish a repeatable attacking mechanism from a one-off event."
        )
    if "pass_network" in f:
        return _join(
            opening,
            f"For {team}, the network shows the build-up skeleton. A strong triangle around centre-back, pivot and full-back usually means controlled circulation. A heavy line into one wide player shows a clear outlet. If the striker or advanced midfielders are disconnected, possession may have looked stable without really threatening {opponent}.",
            "Substitutes are useful here because they show how the structure changed after the starting shape broke. A late player appearing high and wide may indicate a chase phase; a substitute close to the midfield line may show an attempt to regain control."
        )
    if "xt_map" in f:
        return _join(
            opening,
            f"xT is the best bridge between possession and danger. For {team}, the important question is not simply how many passes were completed, but whether those passes moved the ball into zones that changed {opponent}'s defensive problem.",
            "Look for repeated arrows into the half-space, the box edge, or the far-side channel. Those actions normally force the back line to turn, narrow, or step out. When the highest-threat arrows come from deep or wide zones, it often reveals the team's main progression weapon."
        )
    if "shot_comparison" in f or "xg_tiles" in f or "xg_summary" in f:
        return _join(
            opening,
            "This is the efficiency check on the attacking story. The side leading shots did not necessarily create the better game; the side leading xG usually created the clearer chances. xGoT then tells us whether the finishing improved or reduced those chances after contact.",
            "A gap between xG and goals should be read carefully. It can indicate elite finishing, poor finishing, goalkeeping impact, or simply a small sample. The value of this page is that it tells you where to look next: shot map for locations, xG flow for timing, and goals table for the final actions."
        )
    if "danger" in f:
        return _join(
            opening,
            f"This is the best page for reading where {team}'s attacks actually hurt {opponent}. Warm zones near the box or half-spaces suggest a clean attacking route. Warmth stuck near the touchline can still be useful, but it often means the next action had to be excellent to become a real chance.",
            "The coaching point is repeatability. If shots, key passes and entries all come from the same lane, the team found a reliable pattern. If they are scattered, the attack may have depended more on individual actions than on a stable structure."
        )
    if "gk_saves" in f:
        return _join(
            opening,
            "This page gives context to the scoreline. A goalkeeper with many saves from low-value shots may simply have done the routine work. A goalkeeper saving one or two high-xG chances has changed the game state.",
            "The tactical read is also defensive: if most shots faced came from central close-range areas, the defensive block allowed access to premium zones. If the saves came from distance or angles, the outfield structure protected the most valuable space."
        )
    if "zone14" in f:
        return _join(
            opening,
            f"Zone 14 and the half-spaces are where possession becomes creative pressure. When {team} receives or combines there, {opponent}'s centre-backs and midfield line are forced to make decisions: step out, hold shape, or pass runners on.",
            "High volume in these lanes usually explains why a side looked dangerous even before the final shot. Low volume suggests the opponent closed the central door and pushed play toward safer wide areas."
        )
    if "match_stats" in f:
        return _join(
            opening,
            "This is the report's control panel. The attacking rows explain output, the defensive rows explain resistance, and the PPDA view explains how aggressively each side tried to win the ball back.",
            "Read the categories together rather than separately. High passes with low box threat can mean sterile possession. High defensive actions with low possession can mean a team spent too long reacting. The strongest performances usually connect territory, pressure and chance quality."
        )
    if "territorial" in f or "possession" in f or "ball_touches" in f or "dominating_zone" in f:
        return _join(
            opening,
            "Territory is not the same as possession, but it tells us where the match was played. A team controlling advanced zones forced the opponent to defend closer to goal. A team with touches mostly in its own half may have had the ball without changing the opponent's shape.",
            "The tactical value comes from connecting this page to chance creation. Territorial control is meaningful when it feeds entries, key passes and shots. If the territorial map looks dominant but the shot pages do not, the opponent probably defended the box well."
        )
    if "pass_thirds" in f:
        return _join(
            opening,
            f"For {team}, this page shows whether possession travelled through the pitch or got stuck. Defensive-third volume is not a problem by itself; it becomes a problem when the middle and attacking-third numbers do not grow from it.",
            "The best read is the balance between middle-third circulation and final-third penetration. A strong team normally has enough middle-third security to progress, then enough final-third quality to turn that progression into pressure."
        )
    if "xt_per_minute" in f:
        return _join(
            opening,
            "This is the momentum page. Short spikes show individual threat moments; longer clusters show sustained tactical pressure. A team that repeatedly creates spikes after regains is probably dangerous in transition, while a team building smoother waves is likely progressing through possession.",
            "Use the timing to understand coaching interventions. A change after half-time or substitutions can reveal whether the structure improved access to dangerous zones or whether the opponent lost control through fatigue."
        )
    if "progressive" in f:
        return _join(
            opening,
            f"Progressive passes tell us who advanced the game for {team}. Vertical arrows through the middle usually break lines directly. Diagonal arrows from centre-back to winger or full-back can be just as valuable because they move the defensive block sideways before the next action.",
            "When progression comes from multiple players, the team is harder to press. When it depends on one player, the opponent has a clear target for adjustment."
        )
    if "crosses" in f:
        return _join(
            opening,
            f"Crosses show the final shape of {team}'s wide attacks. Byline crosses and cut-backs usually indicate penetration behind the full-back. Early crosses suggest the team reached wide areas but could not always enter the box with combinations.",
            "The important detail is not just volume but delivery context. Crosses with runners in the box are a plan; crosses under pressure into a set defence are often a symptom of blocked central access."
        )
    if "defensive_hm" in f or "defensive_summary" in f:
        return _join(
            opening,
            f"This is the defensive personality of the match. For {team}, actions high up the pitch point to pressing and counter-pressing; actions around the box point to protection, recovery defending and emergency defending.",
            "Blocks and clearances should be read as pressure indicators. They can show commitment, but they can also show that the team spent long spells defending its own penalty area. Tackles and interceptions higher up usually suggest cleaner control."
        )
    if "avg_position" in f:
        return _join(
            opening,
            f"Average positions give the structural picture of {team}'s match. A compact shape helps counter-press and combine. A stretched shape can create width, but it may also leave the midfield exposed if possession is lost.",
            "Substitutes matter because they reveal the second game state. Late average positions can show whether the team protected a lead, chased the game, or changed the route of attack."
        )
    if "box_entries" in f:
        return _join(
            opening,
            f"Box entries are one of the cleanest attacking indicators for {team}. They show whether the team actually breached {opponent}'s penalty-area shell, not just whether it had the ball around it.",
            "The entry type matters. Carries often mean a player beat pressure; passes often mean the structure created a free receiver. A healthy attack usually has both."
        )
    if "high_turnovers" in f:
        return _join(
            opening,
            f"High turnovers measure how much pressure {team} turned into immediate attacking opportunity. Winning the ball high compresses the distance to goal and often catches {opponent} before their defensive shape is rebuilt.",
            "The best high-turnover sides do not just regain possession; they convert the regain into a shot, key pass or box entry quickly. If the regain count is high but chance quality is low, the counter-press worked but the next action lacked clarity."
        )
    if "pass_target" in f:
        return _join(
            opening,
            f"Pass target zones show where {team} wanted the next receiver to be. This is different from pass origin: it tells us the intended destination of possession and therefore the spaces the team believed were available.",
            "If targets collect between the lines, the team found pockets. If they collect wide and deep, the opponent probably blocked central access. The strongest attacking maps usually combine wide outlets with central receiving zones."
        )
    if "ppda" in f:
        return _join(
            opening,
            "PPDA should be read as behaviour, not just a number. A low PPDA means the team allowed few passes before engaging; that usually reflects a higher line, stronger counter-press, or a deliberate plan to trap the opponent.",
            "A higher PPDA is not automatically poor. It can reflect a controlled mid-block or game-state management. The key is whether the deeper approach still protected the box and limited high-quality shots."
        )
    if "player_stats" in f:
        return _join(
            opening,
            "This page gives the individual layer underneath the team story. Minutes and touches explain involvement; passing and key passes explain influence on possession; defensive actions explain workload without the ball.",
            "Use it to identify roles rather than just praise totals. A full-back with heavy touches may have been the outlet. A midfielder with fewer touches but high key passes may have been the connector. Substitutes show how the match plan changed late."
        )
    return _join(
        opening,
        "The tactical value of this visual is in how it connects with the pages around it. One chart rarely tells the whole match; the stronger read comes when the same theme appears in chance quality, territory, passing direction and defensive pressure.",
        f"When those layers point in the same direction, the match story becomes reliable: which side controlled space, which side created the cleaner chances, and which team forced {opponent} to play in uncomfortable areas."
    )


def _pdf_page_with_commentary(pdf, fig, heading: str, body: str):
    """
    Compose one reference-style PDF page: portrait page, visual on top,
    explanatory report text underneath, and no legacy commentary card.
    """
    import io as _io
    import textwrap as _tw
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=PDF_VISUAL_DPI, bbox_inches="tight",
                facecolor=BG_DARK)
    buf.seek(0)
    try:
        from PIL import Image as _Image
        img_arr = _Image.open(buf)
        img = np.asarray(img_arr)
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

    is_light_theme = str(BG_DARK).upper() in {"#FFFFFF", "WHITE"}
    subtitle_color = TEXT_DIM
    visual_bg = "#FFFFFF" if is_light_theme else BG_DARK

    new_fig.text(
        0.07, 0.945, "Tactical Commentary",
        ha="left", va="center", color=TEXT_BRIGHT,
        fontsize=18, fontweight="bold", family="serif",
    )
    line_ax = new_fig.add_axes((0.07, 0.922, 0.86, 0.002))
    line_ax.set_facecolor(C_GOLD)
    line_ax.set_xticks([]); line_ax.set_yticks([])
    for s in line_ax.spines.values():
        s.set_visible(False)

    frame_x, frame_y, frame_w, frame_h = 0.07, 0.56, 0.86, 0.30
    frame_aspect = frame_w / frame_h
    if aspect >= frame_aspect:
        draw_w = frame_w
        draw_h = frame_w / aspect
    else:
        draw_h = frame_h
        draw_w = frame_h * aspect
    draw_x = frame_x + (frame_w - draw_w) / 2
    draw_y = frame_y + (frame_h - draw_h) / 2

    ax_img = new_fig.add_axes((draw_x, draw_y, draw_w, draw_h))
    ax_img.set_facecolor(visual_bg)
    ax_img.imshow(img, aspect="auto")
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for s in ax_img.spines.values():
        s.set_visible(False)

    ax_txt = new_fig.add_axes((0.07, 0.095, 0.86, 0.405))
    ax_txt.set_facecolor(BG_DARK)
    ax_txt.set_xticks([]); ax_txt.set_yticks([])
    for s in ax_txt.spines.values():
        s.set_visible(False)
    ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)
    ax_txt.text(
        0.0, 0.98, heading,
        ha="left", va="top", color=TEXT_BRIGHT,
        fontsize=14, fontweight="bold", family="serif",
        transform=ax_txt.transAxes,
    )
    wrapped = "\n\n".join(
        _tw.fill(p.strip(), width=100)
        for p in str(body).split("\n\n")
        if p.strip()
    )
    ax_txt.text(
        0.0, 0.83, wrapped,
        ha="left", va="top", color=TEXT_MAIN,
        fontsize=8.9, family="serif",
        transform=ax_txt.transAxes, linespacing=1.20,
    )
    new_fig.text(
        0.07, 0.055, "Match Analysis Report",
        ha="left", va="center", color=subtitle_color,
        fontsize=8.5, family="sans-serif",
    )
    new_fig.text(
        0.93, 0.055, "Reading this visual",
        ha="right", va="center", color=subtitle_color,
        fontsize=8.5, family="sans-serif",
    )

    pdf.savefig(new_fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(new_fig)
    plt.close(fig)


def _draw_visual_commentary(pdf, heading: str, body: str):
    """
    Companion 'Reading this visual' page placed right after each tactical
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
        s.set_edgecolor(GRID_COL); s.set_linewidth(0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # Soft accent stripe at the top of the panel
    ax.add_patch(mpatches.Rectangle((0, 0.94), 1, 0.06,
                                     facecolor=C_GOLD, alpha=0.18, lw=0,
                                     transform=ax.transAxes))
    ax.text(0.04, 0.97, "KEY TAKEAWAYS", ha="left", va="center",
            color=C_GOLD, fontsize=10, fontweight="bold",
            transform=ax.transAxes,
            path_effects=[pe.withStroke(linewidth=2, foreground=BG_DARK)])
    # Wrap body text into the panel
    import textwrap as _tw
    wrapped = "\n\n".join(
        _tw.fill(p.strip(), width=110) for p in body.split("\n\n") if p.strip()
    )
    ax.text(0.04, 0.88, wrapped, ha="left", va="top",
            color=TEXT_MAIN, fontsize=11, transform=ax.transAxes,
            linespacing=1.55, wrap=True)
    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


# ── Commentary catalogue: position-keyed text for the 39 tactical visuals ──
# Order must match Match_Analysis_Dark._build_visual_catalog (idx 1..N).
_VISUAL_COMMENTARY = [
    # 1. xG Flow
    ("Reading the xG Flow",
     "The xG Flow plots cumulative xG minute by minute as a staircase, with "
     "every step marking a shot whose height equals its chance quality. "
     "Stars sit on goals; the shaded territory under each curve is total "
     "chance creation. A team that pulls clearly above the other built the "
     "stronger shot profile, even if the scoreline says otherwise."),
    # 2. Home Shot Map
    ("Reading the Home Shot Map",
     "Each dot is a single shot, located at where it was struck. Marker "
     "size scales with xG, so big circles are big chances. Filled circles "
     "are goals, hollow ones are misses. Cluster density inside the box "
     "shows where the team manufactured looks — wide-area dots usually "
     "indicate speculative efforts."),
    # 3. Away Shot Map
    ("Reading the Away Shot Map",
     "Same encoding as the home shot map: dot location is where the shot "
     "was taken, dot size is xG, fill = goal. Compare both maps side by "
     "side to read which side accessed central, high-value zones and which "
     "had to settle for low-percentage outside-the-box attempts."),
    # 4. Shot Breakdown & Goals
    ("Reading the Shot Breakdown",
     "The bar group counts every shot by outcome — total, woodwork, on "
     "target, off target, blocked. On-target conversion (goals divided by "
     "shots on target) reflects finishing efficiency. The Goals & Assists "
     "table below records the scorer, the play type (Open Play / Set "
     "Piece / Penalty), the assister and the chance's xG."),
    # 5. Home Pass Network
    ("Reading the Home Pass Network",
     "Each node is a player placed at their average pass position. Node "
     "size scales with passes attempted; line width between two players "
     "scales with the volume of completed passes between them. Heavy lines "
     "reveal the side's preferred passing partnerships and which axis the "
     "build-up flowed through."),
    # 6. Away Pass Network
    ("Reading the Away Pass Network",
     "Same encoding as the home network. Read each side's shape: a flat, "
     "wide network indicates a pass-through-the-thirds approach; a tall, "
     "narrow one points to vertical, central build-up. Isolated nodes far "
     "from the cluster usually mark a wide outlet who barely received."),
    # 7. Home xT Map
    ("Reading the Home xT Map",
     "The grid colours each pitch zone by its xT (expected threat) value — "
     "the probability that owning the ball there leads to a shot in the "
     "next few seconds. White arrows are positive-xT passes (gained "
     "threat); red arrows are negative-xT passes (gave threat back). "
     "Dense white traffic into the warm zones is the team's progression "
     "signature."),
    # 8. Away xT Map
    ("Reading the Away xT Map",
     "Same grid + arrow encoding for the away side. The total xT in the "
     "side panel sums all positive-xT passes; compare totals to read who "
     "carried the threat-creation load and where on the pitch each side's "
     "progression actually happened."),
    # 9. Shot Comparison
    ("Reading the Shot Comparison",
     "A side-by-side bar chart of the headline shooting numbers — total "
     "shots, shots on target, big chances, xG. It is the fastest single "
     "view to answer 'who was the more dangerous attacking side?' before "
     "diving into the maps."),
    # 10. Home Danger Creation
    ("Reading the Home Danger Creation",
     "Every event that ended in a shot, key pass or box entry is plotted "
     "to show which channels generated the high-value moments. Concentrate "
     "on the warm zones — those are the side's true danger lanes."),
    # 11. Away Danger Creation
    ("Reading the Away Danger Creation",
     "Same encoding for the away side: shots, key passes and box entries "
     "overlaid on the pitch. Compare the two danger maps to see which "
     "side's attack lived in central, high-quality areas vs. the flanks."),
    # 12. Goalkeeper Saves
    ("Reading the Goalkeeper Saves",
     "Plots the location of every shot the keeper faced, colour-coded by "
     "outcome (saved / goal / off-target). Save quality is a function of "
     "the xG of the shots faced — saving a high-xG strike beats stopping "
     "easy long-range efforts."),
    # 13. xG / xGoT Summary
    ("Reading the xG and xGoT Summary",
     "xG measures pre-shot chance quality; xGoT (expected goals on target) "
     "measures the post-shot quality once placement and power are known. "
     "A team with xGoT well above xG executed their finishing better than "
     "the average; below xG points to wasteful conversion."),
    # 14. Home Zone 14 and Half-Spaces
    ("Reading Home Zone 14 & Half-Spaces",
     "Zone 14 is the central pocket just outside the box — historically the "
     "richest zone for chance creation. The half-spaces flank it. This map "
     "counts every action the side completed in those zones; volume here "
     "is a proxy for how often they reached the most dangerous central "
     "real estate."),
    # 15. Away Zone 14 and Half-Spaces
    ("Reading Away Zone 14 & Half-Spaces",
     "Same encoding for the away side. Compare central-pocket access — "
     "the side that worked the ball into Zone 14 more often usually had "
     "the better creative platform, even before chance quality is "
     "measured."),
    # 16. Match Statistics (legacy table)
    ("Reading the Match Statistics",
     "A consolidated table of headline numbers — possession, passes, "
     "shots, on-target, xG, xT, key passes, big chances. Each row is a "
     "head-to-head with the leader bolded. Use it as the single-page "
     "summary of the entire match."),
    # 17. Territorial Control
    ("Reading Territorial Control",
     "Splits the pitch into bands and reads which team had the ball more "
     "often in each. A side with most of its colour in the opponent's "
     "third was camped high and aggressive; deep colour bands mean the "
     "side defended low and tried to play out from the back."),
    # 18. Ball Touches
    ("Reading the Ball Touches",
     "Plots every touch on the pitch as a heatmap. Hot zones reveal the "
     "side's centre of gravity — where the game actually got played. "
     "Compare the two heatmaps to see which territory each side owned."),
    # 19. Home Pass Map by Third
    ("Reading the Home Pass Map by Third",
     "Splits passes into defensive, middle and attacking third buckets "
     "and visualises each on the pitch. Final-third pass density and "
     "completion rate are the two quickest reads of break-down activity "
     "in the opposition area."),
    # 20. Away Pass Map by Third
    ("Reading the Away Pass Map by Third",
     "Same per-third pass map for the away side. A team that had heavy "
     "middle-third volume but thin attacking-third volume struggled to "
     "convert build-up into break-down; that is the classic 'sideways "
     "possession' pattern."),
    # 21. xT per Minute
    ("Reading xT per Minute",
     "A diverging bar chart: home xT bars rise above the zero line, away "
     "xT drops below it. Tall spikes mark the moments each side surged in "
     "threat creation. The 5-minute rolling average overlay smooths the "
     "noise and shows momentum windows."),
    # 22. Home Progressive Passes
    ("Reading Home Progressive Passes",
     "Plots every pass that moved the ball at least 25% closer to goal "
     "(or any pass into the final third). Wide-spread arrows mean the "
     "progression load was shared; concentration around one player marks "
     "a single source of forward play."),
    # 23. Away Progressive Passes
    ("Reading Away Progressive Passes",
     "Same encoding for the away side. The total count and the spatial "
     "distribution together tell you whether the side moved the ball "
     "forward by volume, by quality, or both."),
    # 24. Home Crosses
    ("Reading Home Crosses",
     "Every cross into the box plotted with its origin and end point. "
     "Successful crosses are colour-coded distinctly. A cluster from the "
     "byline indicates a touchline-and-cut-back attack; deeper origins "
     "mark whipped, second-phase deliveries."),
    # 25. Away Crosses
    ("Reading Away Crosses",
     "Same cross map for the away side. Compare cross volume and accuracy "
     "to understand how much of each side's box entry came through wide "
     "play versus central combinations."),
    # 26. Home Defensive Heatmap
    ("Reading the Home Defensive Heatmap",
     "A density map of every defensive action the side completed — "
     "tackles, interceptions, clearances, blocks, recoveries. The hot "
     "zones reveal where the team chose to defend: high up the pitch "
     "(press) or deep in their own block."),
    # 27. Away Defensive Heatmap
    ("Reading the Away Defensive Heatmap",
     "Same defensive density map for the away side. Compare the heat "
     "centres to read each team's defensive line height — a side whose "
     "actions cluster near the halfway line was pressing; one whose "
     "actions sit near their own box was sitting deep."),
    # 28. Defensive Summary
    ("Reading the Defensive Summary",
     "A headline panel of defensive metrics for both sides — tackles, "
     "interceptions, blocks, clearances, recoveries, aerial duels. "
     "Tackles show direct duels, interceptions show anticipation and "
     "cover-shadow play, recoveries show control of loose-ball moments "
     "after pressure or clearances."),
    # 29. Home Average Positions
    ("Reading Home Average Positions",
     "Each player's average XY position across all their touches, scaled "
     "and connected to show team shape. A high, narrow shape signals an "
     "aggressive, compact pressing block; a deep, wide shape points to a "
     "low-block defensive setup."),
    # 30. Away Average Positions
    ("Reading Away Average Positions",
     "Same average-position shape for the away side. Compare the two "
     "shapes to read the structural matchup — overlap on the centre line "
     "indicates a battle for the middle, separation indicates one side "
     "controlled the territory."),
    # 31. Dominating Zone
    ("Reading the Dominating Zone",
     "The pitch is divided into a grid; each cell is coloured by the side "
     "that had more ball-touches there. Block colour reveals territorial "
     "dominance at a glance — large contiguous regions in one team's "
     "colour mark areas they controlled outright."),
    # 32. Home Box Entries
    ("Reading Home Box Entries",
     "Plots every entry into the opposition penalty area — through pass, "
     "carry, cross or set piece. Clusters near the byline mean wide-and-"
     "cut-back access; clusters at the D mean central, through-ball "
     "access. The total count is a direct measure of break-down volume."),
    # 33. Away Box Entries
    ("Reading Away Box Entries",
     "Same encoding for the away side. Compare both maps to see whose "
     "attack actually arrived inside the 18-yard box, and through which "
     "channel the bulk of the entries came."),
    # 34. Home High Turnovers
    ("Reading Home High Turnovers",
     "Marks every regain of possession in the attacking third. Frequent "
     "high turnovers are a hallmark of a successful press — the more "
     "concentrated the dots near the opposition box, the more often the "
     "team won the ball in dangerous areas."),
    # 35. Away High Turnovers
    ("Reading Away High Turnovers",
     "Same high-turnover map for the away side. The team with more "
     "attacking-third regains generated more counter-press shot threats. "
     "Empty heatmaps point to a team that ceded the high zone to the "
     "opposition."),
    # 36. Home Pass Target Zones
    ("Reading Home Pass Target Zones",
     "Heatmap of where on the pitch the side's passes landed — i.e. their "
     "preferred receiving zones. Compare with the pass network to see "
     "whether the receiving pattern matches the network's shape, or "
     "whether passes were aiming for under-served outlets."),
    # 37. Away Pass Target Zones
    ("Reading Away Pass Target Zones",
     "Same pass-target heatmap for the away side. A side whose target "
     "zones cluster high and central had build-up reaching dangerous "
     "areas; clusters in their own half indicate possession recycled but "
     "rarely advanced."),
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
        "Reading this visual",
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

    if apply_unified_frame is not None:
        apply_unified_frame(
            fig,
            section="TEAM STATISTICS",
            title=f"{hn} vs {an} — Match Statistics",
            subtitle="Attack · Passing · Pressing (PPDA) · Defense — "
                     "every metric is computed directly from the event stream",
            accent=C_GOLD,
            home_name=hn, away_name=an, score=str(score),
            footer_note="Bars scaled to the higher value · bold = leader",
        )
    else:
        fig.text(0.04, 0.94, "TEAM STATISTICS — COMPARISON",
                 color=TEXT_BRIGHT, fontsize=20, fontweight="bold")
        fig.text(0.04, 0.91, "Side-by-side counts from event data",
                 color=TEXT_DIM, fontsize=10, style="italic")

    home_id = info.get("home_id")
    away_id = info.get("away_id")

    def _count(team_id, mask_fn) -> int:
        if events is None or events.empty:
            return 0
        try:
            return int(mask_fn(events[events["team_id"] == team_id]).sum())
        except Exception:
            return 0

    # ── Stats grouped: attacking/passing on top, defensive on bottom ──
    attack_rows = [
        ("Goals",
         _count(home_id, lambda d: (d.get("is_goal", False) == True) &  # noqa: E712
                (d.get("scoring_team", -1) == home_id)),
         _count(away_id, lambda d: (d.get("is_goal", False) == True) &  # noqa: E712
                (d.get("scoring_team", -1) == away_id))),
        ("Shots",
         _count(home_id, lambda d: d.get("is_shot", False) == True),  # noqa: E712
         _count(away_id, lambda d: d.get("is_shot", False) == True)),  # noqa: E712
        ("Passes attempted",
         _count(home_id, lambda d: d.get("is_pass", False) == True),  # noqa: E712
         _count(away_id, lambda d: d.get("is_pass", False) == True)),  # noqa: E712
        ("Key passes",
         _count(home_id, lambda d: d.get("is_key_pass", False) == True),  # noqa: E712
         _count(away_id, lambda d: d.get("is_key_pass", False) == True)),  # noqa: E712
    ]

    def _blocked_shots_by(shooter_id) -> int:
        if events is None or events.empty or "team_id" not in events.columns:
            return 0
        sub = events[events["team_id"] == shooter_id]
        if sub.empty:
            return 0
        hit = pd.Series(False, index=sub.index)
        for col in ("type", "shot_whoscored_type", "shot_category"):
            if col in sub.columns:
                vals = sub[col].fillna("").astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)
                hit = hit | vals.isin({"blockedshot", "blocked"})
        if "qualifier_names" in sub.columns:
            hit = hit | sub["qualifier_names"].fillna("").astype(str).str.contains(r"\bBlocked\b", case=False, regex=True)
        if "is_shot" in sub.columns:
            hit = hit & ((sub["is_shot"] == True) | hit)
        return int(hit.sum())

    def _defensive_blocks_for(team_id) -> int:
        opp_id = away_id if team_id == home_id else home_id
        opp_blocks = _blocked_shots_by(opp_id)
        own_blocks = _blocked_shots_by(team_id)
        if not opp_blocks:
            opp_side = "away" if team_id == home_id else "home"
            mc = (info.get("matchcentre_stats", {}) or {}).get(opp_side, {}) or {}
            try:
                opp_blocks = int(mc.get("blocked") or 0)
            except Exception:
                opp_blocks = 0
        if not own_blocks:
            own_side = "home" if team_id == home_id else "away"
            mc = (info.get("matchcentre_stats", {}) or {}).get(own_side, {}) or {}
            try:
                own_blocks = int(mc.get("blocked") or 0)
            except Exception:
                own_blocks = 0
        return opp_blocks if opp_blocks else own_blocks

    # ── Defensive stats: a blocked shot belongs to the defending team.
    defensive_rows = [
        ("Tackles",
         _count(home_id, lambda d: d["type"] == "Tackle"),
         _count(away_id, lambda d: d["type"] == "Tackle")),
        ("Interceptions",
         _count(home_id, lambda d: d["type"] == "Interception"),
         _count(away_id, lambda d: d["type"] == "Interception")),
        ("Blocks",
         _defensive_blocks_for(home_id),
         _defensive_blocks_for(away_id)),
        ("Clearances",
         _count(home_id, lambda d: d["type"] == "Clearance"),
         _count(away_id, lambda d: d["type"] == "Clearance")),
        ("Recoveries",
         _count(home_id, lambda d: d["type"] == "BallRecovery"),
         _count(away_id, lambda d: d["type"] == "BallRecovery")),
        ("Fouls",
         _count(home_id, lambda d: d["type"] == "Foul"),
         _count(away_id, lambda d: d["type"] == "Foul")),
    ]

    h_ppda = ppda.get("home", {}).get("ppda")
    a_ppda = ppda.get("away", {}).get("ppda")

    # ── Helper to draw a stats panel with bar comparison ──
    def _draw_stats_panel(panel_xywh, title, rows_list, accent):
        x, y, w, h = panel_xywh
        ax = fig.add_axes([x, y, w, h])
        ax.set_facecolor(BG_MID)
        for s in ax.spines.values():
            s.set_edgecolor(accent); s.set_linewidth(1.0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        # ── Panel title row (top) — accent strip with title only ──
        ax.add_patch(mpatches.Rectangle((0, 0.93), 1, 0.07,
                                         facecolor=accent, alpha=0.22, lw=0,
                                         transform=ax.transAxes))
        ax.text(0.02, 0.965, title.upper(), ha="left", va="center",
                color=accent, fontsize=11, fontweight="bold",
                transform=ax.transAxes,
                path_effects=[pe.withStroke(linewidth=2, foreground=BG_DARK)])

        # ── Team labels row (just below the title strip) ──
        ax.text(0.02, 0.885, hn, ha="left", va="center",
                color=C_HOME, fontsize=9.5, fontweight="bold",
                transform=ax.transAxes,
                path_effects=[pe.withStroke(linewidth=2, foreground=BG_DARK)])
        ax.text(0.98, 0.885, an, ha="right", va="center",
                color=C_AWAY, fontsize=9.5, fontweight="bold",
                transform=ax.transAxes,
                path_effects=[pe.withStroke(linewidth=2, foreground=BG_DARK)])
        # subtle divider under team labels
        ax.plot([0.02, 0.98], [0.85, 0.85], color=accent, lw=0.5,
                alpha=0.35, transform=ax.transAxes)

        n = len(rows_list)
        if n == 0:
            return
        # Layout zones: fixed value columns, mirrored bars, and a clean
        # centre label chip so long labels never sit on top of the bars.
        top, bot = 0.83, 0.05
        spacing = (top - bot) / n
        for i, (label, hv, av) in enumerate(rows_list):
            cy = top - (i + 0.5) * spacing
            try:
                hh, aa = float(hv), float(av)
                mx = max(hh, aa, 1)
                h_ratio, a_ratio = hh / mx, aa / mx
            except (TypeError, ValueError):
                h_ratio = a_ratio = 0
                hh = aa = None

            bar_h = spacing * 0.40
            # home bar (anchored at 0.38, grows leftwards)
            if h_ratio:
                bw = 0.27 * h_ratio
                ax.add_patch(mpatches.Rectangle(
                    (0.38 - bw, cy - bar_h / 2), bw, bar_h,
                    facecolor=C_HOME, alpha=0.80, lw=0,
                    transform=ax.transAxes))
            # away bar (anchored at 0.62, grows rightwards)
            if a_ratio:
                bw = 0.27 * a_ratio
                ax.add_patch(mpatches.Rectangle(
                    (0.62, cy - bar_h / 2), bw, bar_h,
                    facecolor=C_AWAY, alpha=0.80, lw=0,
                    transform=ax.transAxes))

            h_better = hh is not None and aa is not None and hh > aa
            a_better = aa is not None and hh is not None and aa > hh

            # Numbers — placed OUTSIDE the bars so nothing overlaps them.
            # Leader gets gold + bold for instant scan; loser stays bright.
            home_col = C_GOLD if h_better else TEXT_BRIGHT
            away_col = C_GOLD if a_better else TEXT_BRIGHT
            ax.text(0.06, cy, _na(hv), ha="right", va="center",
                    color=home_col,
                    fontsize=11.5, fontweight="bold",
                    transform=ax.transAxes,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground=BG_DARK)])
            ax.text(0.94, cy, _na(av), ha="left", va="center",
                    color=away_col,
                    fontsize=11.5, fontweight="bold",
                    transform=ax.transAxes,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground=BG_DARK)])
            # Stat label — centered on a chip, never over the bars.
            ax.text(0.50, cy, label, ha="center", va="center",
                    color=TEXT_BRIGHT, fontsize=8.7, fontweight="bold",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.02",
                              facecolor=BG_MID, edgecolor="none", alpha=0.96),
                    path_effects=[pe.withStroke(linewidth=1.4, foreground=BG_DARK)])

    # Top-left: attacking/passing  ·  Bottom-left: defensive
    _draw_stats_panel((0.04, 0.55, 0.42, 0.30),
                      "Attack & Passing", attack_rows, C_HOME)
    _draw_stats_panel((0.04, 0.14, 0.42, 0.36),
                      "Defensive Actions", defensive_rows, C_AWAY)

    # ── Centre column: PPDA mini-gauges (the requested embedded PPDA) ──
    def _mini_dial(ax_pos, name, value, color):
        ax = fig.add_axes(ax_pos, projection="polar")
        ax.set_facecolor(BG_DARK)
        ax.set_theta_zero_location("W"); ax.set_theta_direction(-1)
        ax.set_thetamin(0); ax.set_thetamax(180); ax.set_ylim(0, 1)

        v = value if value is not None else 0
        vmin, vmax = 5.0, 25.0
        ratio = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
        angle = ratio * np.pi

        n_seg = 48
        thetas = np.linspace(0, np.pi, n_seg + 1)
        for i in range(n_seg):
            t = i / n_seg
            if t < 0.15:   c = "#22c55e"
            elif t < 0.30: c = "#84cc16"
            elif t < 0.45: c = "#facc15"
            else:          c = "#f97316"
            ax.bar((thetas[i] + thetas[i+1]) / 2, 0.18, bottom=0.78,
                   width=(thetas[i+1] - thetas[i]) * 0.95,
                   color=c, edgecolor="none", alpha=0.80)
        if value is not None:
            ax.plot([angle, angle], [0, 0.92], color=color, lw=3.5,
                    solid_capstyle="round", zorder=5)
            ax.scatter([angle], [0.92], s=55, color=TEXT_BRIGHT,
                       edgecolor=color, linewidth=1.8, zorder=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["polar"].set_visible(False)

        cx = ax_pos[0] + ax_pos[2] / 2
        cy = ax_pos[1] + 0.02
        val_str = f"{value:.2f}" if value is not None else "N/A"
        fig.text(cx, cy + 0.045, val_str, ha="center", color=color,
                 fontsize=22, fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=2, foreground=BG_DARK)])
        lbl, lcol = _ppda_intensity_label(value)
        fig.text(cx, cy, lbl, ha="center", color=lcol,
                 fontsize=8, fontweight="bold")
        fig.text(cx, ax_pos[1] + ax_pos[3] - 0.005, name,
                 ha="center", color=TEXT_BRIGHT, fontsize=10,
                 fontweight="bold")

    # PPDA section header (centre column)
    _hdr = fig.add_axes([0.49, 0.86, 0.30, 0.004])
    _hdr.set_facecolor(C_GOLD)
    _hdr.set_xticks([]); _hdr.set_yticks([])
    for _s in _hdr.spines.values():
        _s.set_visible(False)

    fig.text(0.64, 0.88, "PRESSING · PPDA",
             ha="center", color=C_GOLD, fontsize=11, fontweight="bold")
    fig.text(0.64, 0.855,
             "Passes per Defensive Action — lower = more aggressive press",
             ha="center", color=TEXT_DIM, fontsize=8.5, style="italic")

    _mini_dial([0.49, 0.66, 0.14, 0.16], hn, h_ppda, C_HOME)
    _mini_dial([0.65, 0.66, 0.14, 0.16], an, a_ppda, C_AWAY)

    # PPDA verdict line
    if h_ppda is not None and a_ppda is not None:
        if h_ppda < a_ppda:
            verdict = f"{hn} pressed more aggressively"; vcol = C_HOME
            diff = a_ppda - h_ppda
        elif a_ppda < h_ppda:
            verdict = f"{an} pressed more aggressively"; vcol = C_AWAY
            diff = h_ppda - a_ppda
        else:
            verdict = "Both teams pressed equally"; vcol = TEXT_BRIGHT
            diff = 0.0
        fig.text(0.64, 0.62, verdict, ha="center", color=vcol,
                 fontsize=11, fontweight="bold")
        if diff > 0:
            fig.text(0.64, 0.60, f"PPDA differential: {diff:.2f}",
                     ha="center", color=TEXT_DIM, fontsize=9)

    # ── Right column: structured commentary ─────────────────────────
    com_ax = fig.add_axes([0.81, 0.14, 0.16, 0.71])
    com_ax.set_facecolor(BG_PANEL)
    for s in com_ax.spines.values():
        s.set_edgecolor(GRID_COL)
    com_ax.set_xticks([]); com_ax.set_yticks([])
    com_ax.set_xlim(0, 1); com_ax.set_ylim(0, 1)

    # Header strip
    com_ax.add_patch(mpatches.Rectangle((0, 0.945), 1, 0.055,
                                         facecolor=C_GOLD, alpha=0.22,
                                         transform=com_ax.transAxes, lw=0))
    com_ax.text(0.06, 0.972, "READING THIS PAGE",
                ha="left", va="center", color=C_GOLD,
                fontsize=9.5, fontweight="bold",
                transform=com_ax.transAxes,
                path_effects=[pe.withStroke(linewidth=2, foreground=BG_DARK)])
    com_ax.plot([0.06, 0.94], [0.945, 0.945], color=C_GOLD, lw=0.6,
                alpha=0.45, transform=com_ax.transAxes)

    # Three structured sections — heading + body
    sections = [
        (C_HOME,
         "ATTACK & PASSING",
         "Shots and key passes show how often each side created looks at "
         "goal. Pass volume reflects how long the team kept the ball "
         "circulating between phases."),
        (C_GOLD,
         "PRESSING (PPDA)",
         "Lower PPDA = more aggressive press. The dial sits in green when "
         "an opponent action was forced every few passes; it slides toward "
         "orange when the side sat in a deeper block. The verdict names "
         "the more aggressive presser."),
        (C_AWAY,
         "DEFENSIVE ACTIONS",
         "Tackles and interceptions describe ground duels. Blocks count "
         "shots stopped by a body in the way. Recoveries and clearances "
         "show how pressure was escaped; fouls mark where containment "
         "broke down."),
    ]

    cy = 0.905
    for color, heading, body in sections:
        # accent dot
        com_ax.add_patch(mpatches.Circle((0.075, cy), 0.012,
                                          facecolor=color, lw=0,
                                          transform=com_ax.transAxes,
                                          zorder=4))
        com_ax.text(0.115, cy, heading, ha="left", va="center",
                    color=color, fontsize=8.5, fontweight="bold",
                    transform=com_ax.transAxes,
                    path_effects=[pe.withStroke(linewidth=2,
                                                foreground=BG_DARK)])
        # body — wrapped, indented under heading
        com_ax.text(0.075, cy - 0.025, body,
                    ha="left", va="top", color=TEXT_MAIN,
                    fontsize=7.8, transform=com_ax.transAxes,
                    wrap=True, linespacing=1.55)
        cy -= 0.27

    pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
    plt.close(fig)


def _draw_player_stats_pages(pdf, player_stats, info, visuals_dir):
    """صفحات إحصائيات اللاعبين — لا تحفظ PNG (الفيجوالز بتتولّد كـ
    figs 41/42 من Dark.py وبتترتّب جنب باقي الفيجوالز في SAVE_DIR)."""
    for side, color in (("home", C_HOME), ("away", C_AWAY)):
        df = player_stats.get(side, pd.DataFrame())
        team_name = info.get(f"{side}_name") or side.title()
        fig = draw_player_stats_table(df, team_name, team_color=color,
                                      save_path=None)
        pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
        plt.close(fig)


def _merge_pdfs(output_path: str, pdf_paths: list[str]) -> bool:
    """
    يدمج عدة PDFs في ملف واحد. بيستخدم pypdf لو متاح،
    أو PyPDF2 كـ fallback. لو الاتنين مش متاحين بيرجع False.
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


def run_analysis(match_data: dict,
                 parse_all_fn=None,
                 extra_figs: list | None = None,
                 extra_figs_filenames: list | None = None,
                 merge_with_pdfs: list[str] | None = None,
                 final_pdf_name: str | None = None) -> dict:
    configure_theme()
    """
    نقطة الدخول الموحدة. بتطلع PDF واحد منظم بالـ dark theme:
        1. Match summary
        2. Goals log
        3. PPDA gauge
        4. Team stats comparison
        5. Player stats — Home
        6. Player stats — Away
        7. Embedded extra_figs (الـ 39 visual الأصلية لو اتبعتت)
        8. (اختياري) دمج مع PDFs الأصلية في merge_with_pdfs

    Args:
        match_data: raw matchCentreData dict.
        parse_all_fn: parse_all() الأصلية من المشروع.
        extra_figs: الفيجوالز الأصلية (39) كـ matplotlib Figures — هتتدمج
                    داخل نفس الـ PDF بعد الصفحات الجديدة.
        merge_with_pdfs: مسارات لـ PDFs خارجية (مثلاً tactical PDF الأصلي)
                         تتدمج في النهاية في الـ final PDF.
        final_pdf_name: اسم الـ PDF النهائي (لو فيه دمج). افتراضي:
                        full_match_report_<ts>.pdf

    وكل visual بيتحفظ كـ PNG منفصل في output/visuals/.
    """
    if parse_all_fn is None:
        raise ValueError(
            "run_analysis requires parse_all_fn (e.g. parse_all_fn=parse_all)."
        )

    _ensure_output_dirs()

    info, events, _players_df = parse_all_fn(match_data)

    ppda = compute_ppda_both(info, events)
    goals_df = build_goals_log(events, info)
    player_stats = extract_player_stats(match_data)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        goals_df.to_csv(
            os.path.join(OUTPUT_DIR, f"goals_log_{ts}.csv"),
            index=False, encoding="utf-8-sig",
        )
    except Exception:
        pass

    hn = info.get("home_name") or "Home"
    an = info.get("away_name") or "Away"
    score = info.get("score") or "—"

    # ── Group extra_figs by side (home / away / shared / player_table) ──
    grouped = {"player_table": [], "home": [], "away": [], "shared": []}
    if extra_figs:
        names = list(extra_figs_filenames or [])
        # Pad with empty strings if shorter than figs
        while len(names) < len(extra_figs):
            names.append("")
        for fig, fname in zip(extra_figs, names):
            f = (fname or "").lower()
            if "player_stats" in f:
                grouped["player_table"].append((fig, fname))
            elif _filename_team_side(fname) == "home":
                grouped["home"].append((fig, fname))
            elif _filename_team_side(fname) == "away":
                grouped["away"].append((fig, fname))
            else:
                grouped["shared"].append((fig, fname))

    def _emit_visual(fig, fname):
        """Apply unified chrome (idempotent for v2 figs) + emit a single
        composite PDF page with visual on top and commentary below."""
        if rebrand_figure is not None:
            try:
                rebrand_figure(fig, home_name=hn, away_name=an,
                               score=str(score), accent=C_GOLD)
            except Exception:
                pass
        heading, body = _commentary_for_filename(fname or "", hn, an)
        body = _professional_tactical_commentary(fname or "", heading, body, hn, an)
        try:
            _pdf_page_with_commentary(pdf, fig, heading, body)
        except Exception:
            try:
                pdf.savefig(fig, dpi=PDF_PAGE_DPI, facecolor=BG_DARK)
            except Exception:
                pass

    pdf_path = os.path.join(OUTPUT_DIR, f"match_report_{ts}.pdf")
    with PdfPages(pdf_path) as pdf:
        # ── Cover page ─────────────────────────────────────────────
        _draw_match_summary_page(pdf, info, goals_df, ppda)

        # ── Section 1: PLAYER RATINGS (both squads) ────────────────
        _draw_section_divider(
            pdf, "01", "PLAYER RATINGS",
            f"Per-player performance for both {hn} and {an} — starters "
            f"first, ratings coloured by quality",
            C_GREEN,
        )
        if grouped["player_table"]:
            for fig, fname in grouped["player_table"]:
                _emit_visual(fig, fname)
        else:
            _draw_player_stats_pages(pdf, player_stats, info, VISUALS_DIR)

        # ── Section 2: SHARED INSIGHTS ─────────────────────────────
        _draw_section_divider(
            pdf, "02", "SHARED INSIGHTS",
            "Head-to-head visuals showing both sides together — chance "
            "quality, territory, pressing and ball progression",
            C_PURPLE,
        )
        for fig, fname in grouped["shared"]:
            _emit_visual(fig, fname)

        # ── Section 3: HOME-TEAM ANALYSIS ──────────────────────────
        _draw_section_divider(
            pdf, "03", f"{hn.upper()} ANALYSIS",
            f"How {hn} attacked, progressed the ball, and defended — "
            f"every {hn.lower()}-specific visual with tactical commentary",
            C_HOME,
        )
        for fig, fname in grouped["home"]:
            _emit_visual(fig, fname)

        # ── Section 4: AWAY-TEAM ANALYSIS ──────────────────────────
        _draw_section_divider(
            pdf, "04", f"{an.upper()} ANALYSIS",
            f"How {an} attacked, progressed the ball, and defended — "
            f"every {an.lower()}-specific visual with tactical commentary",
            C_AWAY,
        )
        for fig, fname in grouped["away"]:
            _emit_visual(fig, fname)

        # ── Closing page ───────────────────────────────────────────
        _draw_closing_page(pdf, info)

    return {
        "pdf": pdf_path,
        "extension_pdf": pdf_path,
        "visuals_dir": VISUALS_DIR,
        "ppda": ppda,
        "goals": goals_df,
        "player_stats": player_stats,
    }
