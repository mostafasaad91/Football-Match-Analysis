"""SofaScore scraping using unofficial public endpoints with graceful fallbacks."""

from __future__ import annotations

from typing import Any

from .http_client import SafeHttpClient
from .models import TeamCandidate


BASE = "https://www.sofascore.com"


def _api(client: SafeHttpClient, path: str) -> dict[str, Any] | None:
    """Fetch a SofaScore API path if allowed and return JSON."""
    result = client.get(f"{BASE}{path}", json_expected=True)
    return result.json_data if result.ok and isinstance(result.json_data, dict) else None


def _latest_competition(team_id: str | int, client: SafeHttpClient) -> tuple[int | None, int | None, str]:
    """Infer the latest unique tournament and season IDs for a team."""
    data = _api(client, f"/api/v1/team/{team_id}/unique-tournaments")
    tournaments = data.get("uniqueTournaments", []) if data else []
    for tournament in tournaments:
        tid = tournament.get("id")
        seasons = _api(client, f"/api/v1/team/{team_id}/unique-tournament/{tid}/seasons")
        season_items = seasons.get("seasons", []) if seasons else []
        if season_items:
            season = season_items[0]
            return tid, season.get("id"), f"{tournament.get('name', 'Current season')} {season.get('year', '')}".strip()
    return None, None, "Current season"


def scrape_sofascore(candidate: TeamCandidate, client: SafeHttpClient) -> dict[str, Any]:
    """Scrape SofaScore team season statistics and top player data."""
    output: dict[str, Any] = {"stats": {}, "players": [], "matches": [], "warnings": []}
    if not candidate.team_id:
        output["warnings"].append("SofaScore team id unavailable.")
        return output

    team_id = candidate.team_id
    unique_tournament_id, season_id, season_label = _latest_competition(team_id, client)
    output["season_label"] = season_label
    if unique_tournament_id and season_id:
        stats = _api(
            client,
            f"/api/v1/team/{team_id}/unique-tournament/{unique_tournament_id}/season/{season_id}/statistics/overall",
        )
        if stats:
            data = stats.get("statistics") or stats
            output["stats"].update(_normalise_team_stats(data))
    else:
        output["warnings"].append("Could not infer SofaScore current season.")

    players = _api(client, f"/api/v1/team/{team_id}/players")
    if players:
        output["players"] = _normalise_players(players.get("players", []))

    events = _api(client, f"/api/v1/team/{team_id}/events/last/0")
    if events:
        output["matches"] = _normalise_matches(events.get("events", []))
    return output


def _normalise_team_stats(data: dict[str, Any]) -> dict[str, Any]:
    """Map SofaScore response fields to dashboard stat names."""
    return {
        "matches_played": data.get("matches") or data.get("matchesPlayed"),
        "wins": data.get("wins"),
        "draws": data.get("draws"),
        "losses": data.get("losses"),
        "goals_for": data.get("goalsScored") or data.get("goalsFor"),
        "goals_against": data.get("goalsConceded") or data.get("goalsAgainst"),
        "goal_difference": data.get("goalDifference"),
        "points": data.get("points"),
        "xg_for": data.get("expectedGoals") or data.get("xgFor"),
        "xg_against": data.get("expectedGoalsAgainst") or data.get("xgAgainst"),
        "possession_avg": data.get("averageBallPossession") or data.get("ballPossession"),
        "shots_per_game": data.get("shotsPerGame") or data.get("totalShots"),
        "shots_on_target": data.get("shotsOnTarget") or data.get("shotsOnTargetPerGame"),
        "clean_sheets": data.get("cleanSheets"),
        "fouls": data.get("fouls"),
        "yellow_cards": data.get("yellowCards"),
        "red_cards": data.get("redCards"),
    }


def _normalise_players(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise SofaScore player rows."""
    rows = []
    for item in players:
        player = item.get("player") or item
        stats = item.get("statistics") or {}
        rows.append(
            {
                "name": player.get("shortName") or player.get("name"),
                "goals": stats.get("goals") or item.get("goals"),
                "assists": stats.get("goalAssist") or stats.get("assists") or item.get("assists"),
                "rating": stats.get("rating") or item.get("rating"),
            }
        )
    return rows


def _normalise_matches(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise SofaScore events into match rows."""
    rows = []
    for event in events:
        home = event.get("homeTeam", {}).get("name")
        away = event.get("awayTeam", {}).get("name")
        hs = (event.get("homeScore") or {}).get("current")
        away_score = (event.get("awayScore") or {}).get("current")
        rows.append(
            {
                "date": event.get("startTimestamp"),
                "home": home,
                "away": away,
                "score": f"{hs}-{away_score}" if hs is not None and away_score is not None else "N/A",
                "competition": (event.get("tournament") or {}).get("name"),
            }
        )
    return rows

