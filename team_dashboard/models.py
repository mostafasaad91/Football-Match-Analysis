"""Data models for the team season dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TeamCandidate:
    """A possible team match from a data provider search."""

    provider: str
    name: str
    team_id: str | int | None = None
    url: str | None = None
    country: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StatValue:
    """A single statistic with its source provider."""

    value: Any = "N/A"
    source: str = "N/A"


@dataclass
class DashboardData:
    """Merged team-season dashboard data ready for charts and reporting."""

    team_name: str
    season_label: str = "Current season"
    stats: dict[str, StatValue] = field(default_factory=dict)
    players: list[dict[str, Any]] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

