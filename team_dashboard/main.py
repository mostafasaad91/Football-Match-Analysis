"""Public entrypoint for building a team season dashboard."""

from __future__ import annotations

from .http_client import SafeHttpClient
from .processing import merge_provider_data
from .report import build_pdf_report
from .resolver import choose_candidate, search_sofascore_team, search_whoscored_team
from .sofascore import scrape_sofascore
from .visualization import generate_all_visuals
from .whoscored import scrape_whoscored


def get_team_stats(team_name: str):
    """Build a full season dashboard for a team name and return the merged data object."""
    client = SafeHttpClient()
    sofa_candidates = search_sofascore_team(team_name, client)
    whoscored_candidates = search_whoscored_team(team_name, client)
    sofa_team = choose_candidate(sofa_candidates, "SofaScore")
    whoscored_team = choose_candidate(whoscored_candidates, "WhoScored")

    if sofa_team is None and whoscored_team is None:
        print(f"No team found for '{team_name}' on SofaScore or WhoScored.")
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

