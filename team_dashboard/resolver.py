"""Team search and selection across SofaScore and WhoScored."""

from __future__ import annotations

from bs4 import BeautifulSoup

from .http_client import SafeHttpClient
from .models import TeamCandidate


SOFASCORE_BASE = "https://www.sofascore.com"
WHOSCORED_BASE = "https://www.whoscored.com"


def search_sofascore_team(team_name: str, client: SafeHttpClient) -> list[TeamCandidate]:
    """Search SofaScore for team candidates."""
    url = f"{SOFASCORE_BASE}/api/v1/search/all?q={team_name}"
    result = client.get(url, json_expected=True)
    if not result.ok or not isinstance(result.json_data, dict):
        return []
    candidates: list[TeamCandidate] = []
    for item in result.json_data.get("results", []):
        entity = item.get("entity") or item
        if (entity.get("type") or item.get("type")) not in {None, "team"}:
            continue
        if "team" in str(item.get("type", "team")).lower() or entity.get("name"):
            candidates.append(
                TeamCandidate(
                    provider="sofascore",
                    name=entity.get("name") or entity.get("shortName") or team_name,
                    team_id=entity.get("id"),
                    url=f"{SOFASCORE_BASE}/team/football/{entity.get('slug', '')}/{entity.get('id')}",
                    country=(entity.get("country") or {}).get("name") if isinstance(entity.get("country"), dict) else None,
                    extra=entity,
                )
            )
    return [c for c in candidates if c.team_id]


def search_whoscored_team(team_name: str, client: SafeHttpClient) -> list[TeamCandidate]:
    """Search WhoScored for team candidates by parsing the public search page."""
    url = f"{WHOSCORED_BASE}/Search/?t={team_name}"
    result = client.get(url)
    if not result.ok or not result.text:
        return []
    soup = BeautifulSoup(result.text, "html.parser")
    candidates: list[TeamCandidate] = []
    for link in soup.select("a[href*='/Teams/']"):
        href = link.get("href") or ""
        label = " ".join(link.get_text(" ", strip=True).split())
        if not label:
            continue
        parts = href.split("/")
        team_id = next((p for p in parts if p.isdigit()), None)
        candidates.append(
            TeamCandidate(
                provider="whoscored",
                name=label,
                team_id=team_id,
                url=href if href.startswith("http") else f"{WHOSCORED_BASE}{href}",
            )
        )
    seen = set()
    unique = []
    for candidate in candidates:
        key = (candidate.provider, candidate.team_id, candidate.name.lower())
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def choose_candidate(candidates: list[TeamCandidate], provider: str) -> TeamCandidate | None:
    """Choose a candidate, asking the user when there are multiple matches."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    print(f"\nMultiple {provider} teams found:")
    for idx, candidate in enumerate(candidates, start=1):
        country = f" ({candidate.country})" if candidate.country else ""
        print(f"{idx}. {candidate.name}{country} - {candidate.url or candidate.team_id}")
    while True:
        choice = input(f"Select {provider} team number, or press Enter to skip: ").strip()
        if not choice:
            return None
        try:
            selected = int(choice)
            if 1 <= selected <= len(candidates):
                return candidates[selected - 1]
        except ValueError:
            pass
        print("Invalid selection. Try again.")

