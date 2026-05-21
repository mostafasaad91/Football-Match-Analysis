"""Public entrypoint for building a team season dashboard."""

from __future__ import annotations

import re

from .http_client import SafeHttpClient
from .models import TeamCandidate
from .processing import merge_provider_data
from .report import build_pdf_report
from .resolver import choose_candidate, search_sofascore_team, search_whoscored_team
from .sofascore import scrape_sofascore
from .visualization import generate_all_visuals
from .whoscored import scrape_whoscored


def get_team_stats(
    team_name: str,
    sofascore_team_id: int | str | None = None,
    sofascore_team_url: str | None = None,
    whoscored_team_url: str | None = None,
):
    """Build a full season dashboard for a team name and return the merged data object."""
    client = SafeHttpClient()
    team_name = _clean_team_name(team_name)
    sofa_team = _manual_sofascore_candidate(team_name, sofascore_team_id, sofascore_team_url)
    whoscored_team = _manual_whoscored_candidate(team_name, whoscored_team_url)

    if sofa_team is None:
        sofa_candidates = search_sofascore_team(team_name, client)
        sofa_team = choose_candidate(sofa_candidates, "SofaScore")
    if whoscored_team is None:
        whoscored_candidates = search_whoscored_team(team_name, client)
        whoscored_team = choose_candidate(whoscored_candidates, "WhoScored")

    if sofa_team is None and whoscored_team is None:
        print(f"No team found for '{team_name}' on SofaScore or WhoScored.")
        print("If robots.txt blocks search, set SOFASCORE_TEAM_ID or WHOSCORED_TEAM_URL in team_season_dashboard.py.")
        return None

    sofa_data = scrape_sofascore(sofa_team, client) if sofa_team else {"stats": {}, "players": [], "matches": [], "warnings": ["SofaScore team not selected."]}
    whoscored_data = scrape_whoscored(whoscored_team, client) if whoscored_team else {"stats": {}, "matches": [], "warnings": ["WhoScored team not selected."]}
    display_name = (sofa_team.name if sofa_team else whoscored_team.name) if (sofa_team or whoscored_team) else team_name
    data = merge_provider_data(display_name, sofa_data, whoscored_data)

    visual_paths = generate_all_visuals(data, "output/visuals")
    pdf_path = build_pdf_report(visual_paths, "output/team_report.pdf")

    print("\nTeam dashboard complete")
    print(f"Team: {data.team_name}")
    print(f"Visuals: output/visuals/")
    print(f"PDF: {pdf_path}")
    print("\nStat sources:")
    for key, source in data.sources.items():
        print(f"  {key}: {source}")
    if data.warnings:
        print("\nWarnings:")
        for warning in data.warnings:
            print(f"  - {warning}")
    return data


def _clean_team_name(team_name: str) -> str:
    """Remove accidental wrapping quotes from a manually entered team name."""
    return team_name.strip().strip("\"'").strip()


def _manual_sofascore_candidate(team_name: str, team_id: int | str | None, team_url: str | None) -> TeamCandidate | None:
    """Build a SofaScore candidate from a manually supplied team id or URL."""
    resolved_id = str(team_id).strip() if team_id not in (None, "") else None
    url = (team_url or "").strip()
    if not resolved_id and url:
        match = re.search(r"/(\d+)(?:[/?#]|$)", url)
        resolved_id = match.group(1) if match else None
    if not resolved_id:
        return None
    slug = re.sub(r"[^a-z0-9]+", "-", team_name.lower()).strip("-") or "team"
    return TeamCandidate(
        provider="sofascore",
        name=team_name,
        team_id=resolved_id,
        url=url or f"https://www.sofascore.com/team/football/{slug}/{resolved_id}",
    )


def _manual_whoscored_candidate(team_name: str, team_url: str | None) -> TeamCandidate | None:
    """Build a WhoScored candidate from a manually supplied team URL."""
    url = (team_url or "").strip()
    if not url:
        return None
    match = re.search(r"/Teams/(\d+)", url)
    return TeamCandidate(
        provider="whoscored",
        name=team_name,
        team_id=match.group(1) if match else None,
        url=url,
    )
