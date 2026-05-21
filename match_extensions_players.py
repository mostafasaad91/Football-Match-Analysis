# pyright: reportMissingImports=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportAssignmentType=false
"""SofaScore trial wrapper around match_extensions.

The normal project imports ``match_extensions`` directly.  The trial scripts
import this module instead, which patches only the player-stat extraction/table
columns used by the extended report.  All other report behaviour is delegated to
the existing module.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import match_extensions as _base
from match_extensions import *  # noqa: F401,F403 - re-export normal API
from advanced_player_stats import SOFA_STAT_GROUPS, fetch_advanced_player_stats


_WHOSCORED_EXTRACT_PLAYER_STATS = _base.extract_player_stats
_BASE_COMMENTARY_FOR_FILENAME = _base._commentary_for_filename
_BASE_PROFESSIONAL_TACTICAL_COMMENTARY = _base._professional_tactical_commentary
_CURRENT_INFO: dict[str, Any] = {}
IDENTITY_STAT_KEYS = {"name", "position", "shirt_no", "minutesPlayed", "is_first_xi"}


def export_player_stats_csvs(player_stats: dict, output_dir: str, ts: str) -> list[str]:
    """Export clean SofaScore category tables to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []
    identity = ["name", "position", "shirt_no", "minutesPlayed", "is_first_xi"]
    ordered_metric_keys: list[str] = []
    for group_name, cols in SOFA_STAT_GROUPS:
        if group_name == "Identity":
            continue
        for key, _label in cols:
            if key not in ordered_metric_keys and key not in IDENTITY_STAT_KEYS:
                ordered_metric_keys.append(key)

    for side in ("home", "away"):
        df = player_stats.get(side) if isinstance(player_stats, dict) else None
        if df is None or getattr(df, "empty", True):
            continue
        identity_cols = [c for c in identity if c in df.columns]
        full_cols = identity_cols + [
            c for c in ordered_metric_keys
            if _series_has_data(df, c) and c not in identity_cols
        ]
        if len(full_cols) <= len(identity_cols):
            continue
        full_path = os.path.join(output_dir, f"player_stats_{side}_full.csv")
        _csv_view(df, full_cols).to_csv(full_path, index=False, encoding="utf-8-sig")
        paths.append(full_path)
        for group_name, cols in SOFA_STAT_GROUPS:
            if group_name == "Identity":
                continue
            keys = [k for k, _label in cols if _series_has_data(df, k)]
            export_cols = identity_cols + [k for k in keys if k not in identity_cols]
            if len(export_cols) <= len(identity_cols):
                continue
            slug = group_name.lower().replace(" ", "_")
            path = os.path.join(output_dir, f"player_stats_{side}_{slug}.csv")
            _csv_view(df, export_cols).to_csv(path, index=False, encoding="utf-8-sig")
            paths.append(path)
    return paths


def _csv_view(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.loc[:, cols].copy()
    for col in ("expectedGoals", "expectedGoalsOnTarget", "expectedAssists", "goalsPrevented"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out


def _series_has_data(df: pd.DataFrame, key: str) -> bool:
    if key not in df.columns:
        return False
    series = df[key]
    if not bool(series.notna().any()):
        return False
    cleaned = series.dropna()
    if cleaned.empty:
        return False
    if cleaned.astype(str).str.strip().isin({"", "—", "-", "nan", "None"}).all():
        return False
    return True


def _available_group_metrics(df: pd.DataFrame, metric_cols) -> list[tuple[str, str]]:
    return [
        (k, lbl) for k, lbl in metric_cols
        if k not in IDENTITY_STAT_KEYS and _series_has_data(df, k)
    ]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def extract_player_stats(md: dict) -> dict:
    """Return SofaScore player tables when available, otherwise WhoScored."""
    if not _env_bool("SOFASCORE_PLAYER_TABLES", True):
        return _WHOSCORED_EXTRACT_PLAYER_STATS(md)

    info = dict(_CURRENT_INFO)
    if not info:
        home = md.get("home", {}) or {}
        away = md.get("away", {}) or {}
        info = {
            "home_name": home.get("name") or home.get("teamName") or "Home",
            "away_name": away.get("name") or away.get("teamName") or "Away",
        }

    result = fetch_advanced_player_stats(
        info,
        event_id=os.environ.get("SOFASCORE_EVENT_ID") or None,
        auto_search=_env_bool("SOFASCORE_AUTO_SEARCH", True),
        min_confidence=_env_float("SOFASCORE_MIN_MATCH_CONFIDENCE", 0.82),
        verbose=_env_bool("SOFASCORE_VERBOSE", False),
    )
    if result.player_stats:
        print(
            "[SofaScore] Using player tables "
            f"(event={result.event_id}, confidence={result.confidence:.2f}, source={result.source})."
        )
        return result.player_stats

    print(f"[SofaScore] Falling back to WhoScored player tables: {result.warning}")
    return _WHOSCORED_EXTRACT_PLAYER_STATS(md)


def draw_player_stats_table(df: pd.DataFrame, team_name: str,
                            team_color: str = _base.C_HOME,
                            save_path: str | None = None):
    """Draw SofaScore-style category tables as multiple readable pages."""
    return _draw_sofa_player_stats_pages(df, team_name, team_color=team_color, save_path=save_path)


def _draw_sofa_player_stats_pages(df: pd.DataFrame, team_name: str,
                                  team_color: str = _base.C_HOME,
                                  save_path: str | None = None):
    groups = [g for g in SOFA_STAT_GROUPS if g[0] != "Score"]
    metric_groups = [
        (group_name, metrics, _available_group_metrics(df, metrics))
        for group_name, metrics in groups
        if group_name != "Identity"
    ]
    metric_groups = [
        (group_name, metrics, available)
        for group_name, metrics, available in metric_groups
        if available
    ]
    identity_cols = [("name", "Player"), ("position", "Pos"), ("minutesPlayed", "Min")]

    if df.empty or not metric_groups:
        return []

    pages = []
    is_light = str(_base.BG_DARK).upper() in {"#FFFFFF", "WHITE"}
    total_pages = len(metric_groups)
    for page_idx, (group_name, metrics, available_metrics) in enumerate(metric_groups, start=1):
        fig = _base._new_dark_fig(16, 9.5)
        fig.patch.set_facecolor(_base.BG_DARK)
        fig.text(0.035, 0.965, f"SOFASCORE PLAYER STATS - {team_name.upper()}",
                 color=_base.TEXT_BRIGHT, fontsize=18, fontweight="bold")
        fig.text(0.035, 0.935,
                 f"{group_name} table ({page_idx}/{total_pages})",
                 color=_base.TEXT_DIM, fontsize=10, style="italic")
        bar_ax = fig.add_axes((0.035, 0.905, 0.93, 0.010))
        bar_ax.set_facecolor(team_color)
        bar_ax.set_xticks([]); bar_ax.set_yticks([])
        for s in bar_ax.spines.values():
            s.set_visible(False)
        _draw_sofa_group_table(
            fig, df, group_name, identity_cols, available_metrics,
            rect=(0.035, 0.070, 0.93, 0.805),
            team_color=team_color,
            is_light=is_light,
        )
        setattr(fig, "_sofa_group_name", group_name.lower().replace(" ", "_"))
        pages.append(fig)

    if save_path:
        root, ext = os.path.splitext(save_path)
        ext = ext or ".png"
        for idx, page in enumerate(pages, start=1):
            page.savefig(f"{root}_{idx:02d}{ext}", dpi=220, facecolor=_base.BG_DARK)
    return pages


def _draw_sofa_group_table(fig, df: pd.DataFrame, group_name: str,
                           identity_cols, metric_cols, *, rect, team_color: str,
                           is_light: bool):
    available_metrics = list(metric_cols)
    cols = identity_cols + [
        (k, lbl) for k, lbl in available_metrics
        if k not in {identity_key for identity_key, _identity_label in identity_cols}
    ]

    ax = fig.add_axes(rect)
    ax.set_facecolor(_base.BG_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(_base.GRID_COL)
        s.set_linewidth(0.8)

    header_bg = {
        "General": "#E0F2FE" if is_light else "#1f2a3a",
        "Attacking": "#FCE7F3" if is_light else "#3b1f2f",
        "Defending": "#FEF3C7" if is_light else "#3a2f1f",
        "Passing": "#DCFCE7" if is_light else "#1f3a2f",
        "Duels": "#EDE9FE" if is_light else "#2a1f3a",
        "Goalkeeping": "#DBEAFE" if is_light else "#13233a",
    }.get(group_name, _base.BG_MID)

    ax.add_patch(mpatches.Rectangle((0, 0.88), 1, 0.12, facecolor=header_bg, edgecolor="none"))
    ax.text(0.015, 0.94, group_name.upper(), ha="left", va="center",
            color=_base.TEXT_BRIGHT, fontsize=10.5, fontweight="bold")

    if not available_metrics:
        ax.text(
            0.5, 0.47,
            "No SofaScore data for this category",
            ha="center",
            va="center",
            color=_base.TEXT_DIM,
            fontsize=12,
            fontweight="bold",
        )
        return

    col_widths = []
    for key, _label in cols:
        if key == "name":
            col_widths.append(0.25)
        elif key == "position":
            col_widths.append(0.055)
        elif key == "minutesPlayed":
            col_widths.append(0.065)
        else:
            col_widths.append(0.62 / max(len(cols) - 3, 1))
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]

    x_positions = [0.0]
    for w in col_widths[:-1]:
        x_positions.append(x_positions[-1] + w)

    row_top = 0.82
    row_h = min(0.062, 0.78 / max(len(df) + 1, 2))
    ax.add_patch(mpatches.Rectangle((0, row_top), 1, row_h, facecolor=_base.BG_MID, edgecolor=_base.GRID_COL, lw=0.4))
    for (x, w, (_key, label)) in zip(x_positions, col_widths, cols):
        ha = "left" if _key == "name" else "center"
        tx = x + 0.01 if _key == "name" else x + w / 2
        ax.text(tx, row_top + row_h / 2, label, ha=ha, va="center",
                color=_base.TEXT_DIM, fontsize=7.6, fontweight="bold")

    max_rows = min(len(df), 24)
    for i, (_, r) in enumerate(df.head(max_rows).iterrows()):
        y = row_top - (i + 1) * row_h
        bg = _base.BG_PANEL if i % 2 == 0 else ("#F8FAFC" if is_light else "#0f1520")
        if not bool(r.get("is_first_xi", False)):
            bg = "#F1F5F9" if is_light else "#080a10"
        ax.add_patch(mpatches.Rectangle((0, y), 1, row_h, facecolor=bg, edgecolor=_base.GRID_COL, lw=0.25))
        ax.add_patch(mpatches.Rectangle((0, y), 0.005, row_h, facecolor=team_color, edgecolor="none", alpha=0.9))
        for x, w, (key, _label) in zip(x_positions, col_widths, cols):
            raw = r.get(key)
            text = _format_sofa_cell(key, raw)
            if key == "name":
                text = _base._short_name(str(raw or "N/A"), 22)
                ax.text(x + 0.012, y + row_h / 2, text, ha="left", va="center",
                        color=_base.TEXT_BRIGHT, fontsize=7.8, fontweight="bold")
            else:
                color = _base.TEXT_MAIN
                weight = "normal"
                if key in {"goals", "assists"} and raw not in (None, 0, "0", "—"):
                    color = _base.C_GOLD if key == "goals" else _base.C_GREEN
                    weight = "bold"
                if text in {"—", "N/A"}:
                    color = _base.TEXT_FADED
                ax.text(x + w / 2, y + row_h / 2, text, ha="center", va="center",
                        color=color, fontsize=7.4, fontweight=weight)

    if len(df) > max_rows:
        ax.text(0.99, 0.02, f"+{len(df) - max_rows} more players",
                ha="right", va="bottom", color=_base.TEXT_FADED, fontsize=7.5, style="italic")


def _format_sofa_cell(key: str, value: Any) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    if key in {"expectedGoals", "expectedGoalsOnTarget", "expectedAssists", "goalsPrevented"}:
        try:
            return f"{float(value):.3f}"
        except Exception:
            return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.1f}"
    return str(value)


def _remove_timestamp_from_output_names(output_dir: str | None) -> None:
    if not output_dir or not os.path.isdir(output_dir):
        return
    timestamp_re = re.compile(r"_\d{8}_\d{6}(?=\.|_|$)")
    for root, _dirs, files in os.walk(output_dir):
        for name in files:
            if not name.lower().endswith((".csv", ".png")):
                continue
            cleaned = timestamp_re.sub("", name)
            if cleaned == name:
                continue
            src = os.path.join(root, name)
            dst = os.path.join(root, cleaned)
            try:
                if os.path.exists(dst):
                    os.remove(dst)
                os.replace(src, dst)
            except Exception:
                pass


def _clean_output_name(name: str) -> str:
    root, ext = os.path.splitext(str(name))
    root = re.sub(r"_\d{8}_\d{6}(?=$|_)", "", root)
    return f"{root}{ext}"


def _visual_sort_key(name: str) -> tuple[int, int, str]:
    base = os.path.basename(name).lower()
    match = re.match(r"(\d+)_", base)
    if match:
        return (0, int(match.group(1)), base)
    board = re.match(r"board_(\d+)_", base)
    if board:
        return (1, int(board.group(1)), base)
    return (2, 9999, base)


class _PngFigureProxy:
    def __init__(self, path: str):
        self.path = path

    def savefig(self, target, *args, **kwargs):
        with open(self.path, "rb") as src:
            data = src.read()
        if hasattr(target, "write"):
            target.write(data)
            return
        with open(target, "wb") as dst:
            dst.write(data)


def _ensure_pdf_visual_inputs(extra_figs, extra_figs_filenames, output_dir: str | None):
    figs = list(extra_figs or [])
    names = [_clean_output_name(os.path.basename(n or "")) for n in list(extra_figs_filenames or [])]
    while len(names) < len(figs):
        names.append("")

    existing = {n for n in names if n}
    if not output_dir or not os.path.isdir(output_dir):
        return figs, names

    pngs = [
        os.path.join(output_dir, n)
        for n in os.listdir(output_dir)
        if n.lower().endswith(".png")
    ]
    for path in sorted(pngs, key=lambda p: _visual_sort_key(os.path.basename(p))):
        name = _clean_output_name(os.path.basename(path))
        if name in existing:
            continue
        try:
            figs.append(_PngFigureProxy(path))
            names.append(name)
            existing.add(name)
        except Exception:
            pass
    return figs, names


def _sofa_player_table_commentary(fname: str, hn: str, an: str) -> tuple[str, str] | None:
    f = (fname or "").lower()
    if "player_stats" not in f:
        return None

    if "player_stats_home" in f:
        team = hn
        opponent = an
    elif "player_stats_away" in f:
        team = an
        opponent = hn
    else:
        team = "the team"
        opponent = "the opponent"

    contexts = {
        "general": (
            f"Reading {team}'s General Player Table",
            f"This table is the broad individual overview for {team}: goals, assists, tackles won, pass accuracy, duel output and minutes played. It is designed to show who stayed involved across both phases rather than who only produced one isolated action.",
            f"The tactical read is role balance. Players combining high minutes with strong passing and duel numbers were central to the match rhythm; players with lower involvement but decisive goals or assists affected the result through moments. Against {opponent}, this page helps separate constant influence from final-action impact.",
        ),
        "attacking": (
            f"Reading {team}'s Attacking Player Table",
            f"This page focuses on chance involvement: shots, on-target efforts, blocked shots, big chances, xG, xGOT, xA and successful dribbles. It tells you which players were actually responsible for turning possession into goal threat.",
            f"For {team}, the important detail is the relationship between xG and shot volume. A player with few shots but high xG probably received the ball in premium zones; a player with many low-value attempts may have found shooting volume without clean access. xA adds the creator layer, showing who supplied the final pass before danger appeared.",
        ),
        "defending": (
            f"Reading {team}'s Defending Player Table",
            f"This table isolates defensive workload: tackles won, interceptions, blocks, clearances, ball recoveries and errors leading to shots or goals. It shows who protected space, who broke up attacks, and who had to defend under pressure.",
            f"The coaching read is where the defending happened. High clearances and blocks often mean {team} spent spells protecting the box; high interceptions and recoveries usually point to cleaner control before {opponent} could settle. Errors are included because one defensive action can change the whole shot profile.",
        ),
        "passing": (
            f"Reading {team}'s Passing Player Table",
            f"This page shows the players who controlled circulation and progression: accurate passes, key passes, long balls, own-half passes, opposition-half passes, crosses and final-third passes. It is the best individual view of build-up responsibility.",
            f"For {team}, own-half accuracy points to security under pressure, while opposition-half and final-third passing point to progression and invention. Key passes reveal who turned possession into a direct chance. Long balls and crosses explain whether the team progressed through construction or by switching play and attacking the box earlier.",
        ),
        "duels": (
            f"Reading {team}'s Duels Player Table",
            f"This table measures contact and pressure: total duels won, ground duels, aerial duels, successful dribbles, fouls won, fouls committed and possession losses. It shows who handled the physical and technical contests around the ball.",
            f"The tactical value is in the matchup. If {team}'s midfielders and wide players won ground duels, the team likely protected second balls and transitions. Aerial duel numbers show who handled direct play. Possession losses need context: they can reflect sloppy touches, but they can also reflect players receiving under heavy pressure in ambitious zones.",
        ),
        "goalkeeping": (
            f"Reading {team}'s Goalkeeping Player Table",
            f"This page is specific to goalkeeper actions: saves, saves from inside the box, sweeper actions, goals prevented, claims and punches. It gives context to the scoreline beyond the raw number of goals conceded.",
            f"For {team}, saves inside the box usually carry more value than routine stops from distance because they come from cleaner chances. Sweeper actions show whether the goalkeeper protected the space behind the back line. Goals prevented helps explain whether the goalkeeper added value relative to the quality of shots faced.",
        ),
    }

    for key, value in contexts.items():
        if key in f:
            heading, intro, tactical = value
            return heading, f"{intro}\n\n{tactical}"

    heading = f"Reading {team}'s Player Table"
    body = (
        f"This SofaScore player table gives the individual layer underneath {team}'s team performance. "
        f"Use it to connect the match visuals with the players who carried those actions.\n\n"
        f"The strongest read comes from comparing minutes, involvement and action type. A high-volume player may have controlled the rhythm, while a low-volume player may still have decided the match through one high-value attacking or defensive action."
    )
    return heading, body


def _sofa_commentary_for_filename(fname: str, hn: str, an: str):
    player_commentary = _sofa_player_table_commentary(fname, hn, an)
    if player_commentary is not None:
        return player_commentary
    return _BASE_COMMENTARY_FOR_FILENAME(fname, hn, an)


def _sofa_professional_tactical_commentary(fname: str, heading: str, body: str,
                                           hn: str, an: str) -> str:
    if "player_stats" in (fname or "").lower():
        return body
    return _BASE_PROFESSIONAL_TACTICAL_COMMENTARY(fname, heading, body, hn, an)


def run_analysis(match_data: dict,
                 parse_all_fn=None,
                 extra_figs: list | None = None,
                 extra_figs_filenames: list | None = None,
                 merge_with_pdfs: list[str] | None = None,
                 final_pdf_name: str | None = None) -> dict:
    """Run the normal extended report with SofaScore player tables patched in."""
    global _CURRENT_INFO
    if parse_all_fn is None:
        raise ValueError("run_analysis requires parse_all_fn.")

    info, events, players_df = parse_all_fn(match_data)
    _CURRENT_INFO = info or {}

    def _cached_parse_all(_md):
        return info, events, players_df

    old_extract = _base.extract_player_stats
    old_draw = _base.draw_player_stats_table
    old_groups = _base.STAT_GROUPS
    old_commentary = _base._commentary_for_filename
    old_professional_commentary = _base._professional_tactical_commentary
    old_output_dir = _base.OUTPUT_DIR
    old_visuals_dir = _base.VISUALS_DIR
    try:
        output_dir = os.environ.get("MATCH_ANALYSIS_OUTPUT_DIR")
        if output_dir:
            _base.OUTPUT_DIR = output_dir
            _base.VISUALS_DIR = os.path.join(output_dir, "visuals")
        extra_figs, extra_figs_filenames = _ensure_pdf_visual_inputs(
            extra_figs,
            extra_figs_filenames,
            output_dir or _base.OUTPUT_DIR,
        )
        _base.extract_player_stats = extract_player_stats
        _base.draw_player_stats_table = draw_player_stats_table
        _base.STAT_GROUPS = SOFA_STAT_GROUPS
        _base._commentary_for_filename = _sofa_commentary_for_filename
        _base._professional_tactical_commentary = _sofa_professional_tactical_commentary
        result = _base.run_analysis(
            match_data,
            parse_all_fn=_cached_parse_all,
            extra_figs=extra_figs,
            extra_figs_filenames=extra_figs_filenames,
            merge_with_pdfs=merge_with_pdfs,
            final_pdf_name=final_pdf_name,
        )
        _remove_timestamp_from_output_names(output_dir or _base.OUTPUT_DIR)
        return result
    finally:
        _base.extract_player_stats = old_extract
        _base.draw_player_stats_table = old_draw
        _base.STAT_GROUPS = old_groups
        _base._commentary_for_filename = old_commentary
        _base._professional_tactical_commentary = old_professional_commentary
        _base.OUTPUT_DIR = old_output_dir
        _base.VISUALS_DIR = old_visuals_dir
