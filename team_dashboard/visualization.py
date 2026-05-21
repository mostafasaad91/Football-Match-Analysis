"""Dashboard visual generation."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .models import DashboardData
from .processing import numeric, stat


VISUAL_ORDER = [
    "overview.png", "attack.png", "defense.png", "possession.png",
    "form.png", "squad_stats.png", "discipline.png",
]


def ensure_visual_dir(output_dir: str = "output/visuals") -> Path:
    """Create and return the visuals output directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_all_visuals(data: DashboardData, output_dir: str = "output/visuals") -> list[str]:
    """Generate all requested dashboard visuals and return file paths."""
    path = ensure_visual_dir(output_dir)
    outputs = [
        draw_overview(data, path / "overview.png"),
        draw_attack(data, path / "attack.png"),
        draw_defense(data, path / "defense.png"),
        draw_possession(data, path / "possession.png"),
        draw_form(data, path / "form.png"),
        draw_squad_stats(data, path / "squad_stats.png"),
        draw_discipline(data, path / "discipline.png"),
    ]
    return [str(p) for p in outputs if p]


def _figure(title: str):
    """Create a standard dashboard figure."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#f8fafc")
    ax.set_facecolor("#ffffff")
    ax.set_title(title, fontsize=18, fontweight="bold", loc="left", pad=18)
    return fig, ax


def _save(fig, path: Path) -> Path:
    """Save and close a figure."""
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def draw_overview(data: DashboardData, path: Path) -> Path:
    """Draw the season summary card."""
    fig, ax = _figure(f"{data.team_name} - Season Overview")
    ax.axis("off")
    items = [
        ("Matches", stat(data, "matches_played")), ("W-D-L", f"{stat(data,'wins')}-{stat(data,'draws')}-{stat(data,'losses')}"),
        ("GF / GA", f"{stat(data,'goals_for')} / {stat(data,'goals_against')}"), ("GD", stat(data, "goal_difference")),
        ("Points", stat(data, "points")), ("Formation", stat(data, "formation")),
    ]
    for i, (label, value) in enumerate(items):
        x = 0.08 + (i % 3) * 0.30
        y = 0.68 - (i // 3) * 0.28
        ax.text(x, y, str(value), fontsize=28, fontweight="bold", color="#0f172a", transform=ax.transAxes)
        ax.text(x, y - 0.08, label, fontsize=12, color="#64748b", transform=ax.transAxes)
    ax.text(0.08, 0.12, f"Season: {data.season_label}", fontsize=12, color="#334155", transform=ax.transAxes)
    return _save(fig, path)


def draw_attack(data: DashboardData, path: Path) -> Path:
    """Draw attacking production chart."""
    fig, ax = _figure("Attack")
    labels = ["Goals", "xG", "Shots/Game", "Shots OT"]
    values = [numeric(stat(data, k)) for k in ["goals_for", "xg_for", "shots_per_game", "shots_on_target"]]
    ax.bar(labels, values, color=["#ef4444", "#f97316", "#3b82f6", "#22c55e"])
    ax.set_ylabel("Value")
    return _save(fig, path)


def draw_defense(data: DashboardData, path: Path) -> Path:
    """Draw defensive record chart."""
    fig, ax = _figure("Defense")
    labels = ["GA", "xGA", "Clean Sheets", "PPDA"]
    values = [numeric(stat(data, k)) for k in ["goals_against", "xg_against", "clean_sheets", "ppda"]]
    ax.bar(labels, values, color=["#991b1b", "#fb7185", "#14b8a6", "#6366f1"])
    return _save(fig, path)


def draw_possession(data: DashboardData, path: Path) -> Path:
    """Draw possession and passing indicators."""
    fig, ax = _figure("Possession and Passing")
    labels = ["Possession %", "Pass Accuracy %", "Aerial Won %"]
    values = [numeric(stat(data, k)) for k in ["possession_avg", "passing_accuracy", "aerial_duels_won_pct"]]
    ax.barh(labels, values, color="#2563eb")
    ax.set_xlim(0, max(values + [100]))
    return _save(fig, path)


def draw_form(data: DashboardData, path: Path) -> Path:
    """Draw last 10 match form strip."""
    fig, ax = _figure("Last 10 Matches")
    ax.axis("off")
    matches = data.matches[:10]
    if not matches:
        ax.text(0.5, 0.5, "No match-by-match data available", ha="center", va="center", fontsize=16)
        return _save(fig, path)
    for i, match in enumerate(matches):
        result = _result_letter(match, data.team_name)
        color = {"W": "#22c55e", "D": "#facc15", "L": "#ef4444"}.get(result, "#94a3b8")
        ax.add_patch(plt.Rectangle((0.05 + i * 0.09, 0.45), 0.07, 0.16, color=color, transform=ax.transAxes))
        ax.text(0.085 + i * 0.09, 0.53, result, ha="center", va="center", color="white", fontweight="bold", transform=ax.transAxes)
    return _save(fig, path)


def draw_squad_stats(data: DashboardData, path: Path) -> Path:
    """Draw top performer table."""
    fig, ax = _figure("Top Performers")
    ax.axis("off")
    rows = data.players[:12]
    if not rows:
        ax.text(0.5, 0.5, "No squad statistics available", ha="center", va="center", fontsize=16)
        return _save(fig, path)
    df = pd.DataFrame(rows)[["name", "goals", "assists", "rating"]]
    table = ax.table(cellText=df.fillna("N/A").values, colLabels=["Player", "Goals", "Assists", "Rating"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    return _save(fig, path)


def draw_discipline(data: DashboardData, path: Path) -> Path:
    """Draw discipline and duel chart."""
    fig, ax = _figure("Discipline and Duels")
    labels = ["Yellow", "Red", "Fouls", "Aerial Won %"]
    values = [numeric(stat(data, k)) for k in ["yellow_cards", "red_cards", "fouls", "aerial_duels_won_pct"]]
    ax.bar(labels, values, color=["#facc15", "#dc2626", "#64748b", "#0ea5e9"])
    return _save(fig, path)


def _result_letter(match: dict, team_name: str) -> str:
    """Infer W/D/L from a normalised match row."""
    score = str(match.get("score") or "")
    try:
        home_goals, away_goals = [int(x) for x in score.split("-")[:2]]
        is_home = str(match.get("home", "")).lower() == team_name.lower()
        team_goals, opp_goals = (home_goals, away_goals) if is_home else (away_goals, home_goals)
        if team_goals > opp_goals:
            return "W"
        if team_goals < opp_goals:
            return "L"
        return "D"
    except Exception:
        return "N/A"

