"""Merge provider data into dashboard-ready structures."""

from __future__ import annotations

from typing import Any

from .models import DashboardData, StatValue


STAT_ORDER = [
    "matches_played", "wins", "draws", "losses", "goals_for", "goals_against",
    "goal_difference", "points", "xg_for", "xg_against", "possession_avg",
    "shots_per_game", "shots_on_target", "clean_sheets", "team_rating",
    "passing_accuracy", "aerial_duels_won_pct", "ppda", "yellow_cards",
    "red_cards", "formation", "fouls",
]


def merge_provider_data(team_name: str, sofa: dict[str, Any], whoscored: dict[str, Any]) -> DashboardData:
    """Merge SofaScore and WhoScored payloads, preserving stat sources."""
    data = DashboardData(team_name=team_name)
    data.season_label = sofa.get("season_label") or "Current season"
    for stat in STAT_ORDER:
        value, source = _first_available(
            (sofa.get("stats", {}) or {}).get(stat), "SofaScore",
            (whoscored.get("stats", {}) or {}).get(stat), "WhoScored",
        )
        data.stats[stat] = StatValue(value=value, source=source)
        data.sources[stat] = source
    data.players = _merge_players(sofa.get("players", []), [])
    data.matches = sofa.get("matches") or whoscored.get("matches") or []
    data.warnings.extend(sofa.get("warnings", []))
    data.warnings.extend(whoscored.get("warnings", []))
    return data


def _first_available(a: Any, a_source: str, b: Any, b_source: str) -> tuple[Any, str]:
    """Return the first non-empty stat value and its source."""
    if a not in (None, "", "N/A"):
        return a, a_source
    if b not in (None, "", "N/A"):
        return b, b_source
    return "N/A", "N/A"


def _merge_players(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge player stat rows, preferring the primary provider."""
    rows = primary or secondary or []
    return sorted(rows, key=lambda r: (float(r.get("goals") or 0), float(r.get("assists") or 0)), reverse=True)[:20]


def stat(data: DashboardData, key: str, default: Any = "N/A") -> Any:
    """Convenience getter for dashboard stat values."""
    return data.stats.get(key, StatValue(default)).value


def numeric(value: Any, default: float = 0.0) -> float:
    """Convert a stat value to float, returning default on failure."""
    try:
        if value in (None, "N/A", ""):
            return default
        return float(str(value).replace("%", ""))
    except Exception:
        return default

