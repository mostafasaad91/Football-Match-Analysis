"""WhoScored scraping helpers with static parsing and safe fallbacks."""

from __future__ import annotations

import json
import re
from typing import Any

from bs4 import BeautifulSoup

from .http_client import SafeHttpClient
from .models import TeamCandidate


def scrape_whoscored(candidate: TeamCandidate, client: SafeHttpClient) -> dict[str, Any]:
    """Scrape available WhoScored team information from the team page."""
    output: dict[str, Any] = {"stats": {}, "matches": [], "warnings": []}
    if not candidate.url:
        output["warnings"].append("WhoScored URL unavailable.")
        return output
    result = client.get(candidate.url)
    if not result.ok or not result.text:
        output["warnings"].append(f"WhoScored page unavailable: {result.error}")
        return output
    soup = BeautifulSoup(result.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    output["stats"].update(_parse_text_stats(text))
    output["matches"] = _parse_match_rows(soup)
    embedded = _extract_json_blobs(result.text)
    output["stats"].update(_parse_embedded_stats(embedded))
    return output


def _parse_text_stats(text: str) -> dict[str, Any]:
    """Extract common WhoScored stats from visible page text."""
    patterns = {
        "team_rating": r"Rating\s+([0-9.]+)",
        "passing_accuracy": r"Pass Success(?:\s+%)?\s+([0-9.]+)",
        "aerial_duels_won_pct": r"Aerial(?:s)? Won(?:\s+%)?\s+([0-9.]+)",
        "yellow_cards": r"Yellow Cards\s+([0-9]+)",
        "red_cards": r"Red Cards\s+([0-9]+)",
        "formation": r"Formation\s+([0-9\-]+)",
        "ppda": r"PPDA\s+([0-9.]+)",
    }
    stats: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.I)
        if match:
            stats[key] = match.group(1)
    return stats


def _extract_json_blobs(html: str) -> list[Any]:
    """Extract JSON-looking blobs from script tags."""
    blobs = []
    for match in re.finditer(r"({\"[^<]{100,}})", html):
        raw = match.group(1)
        try:
            blobs.append(json.loads(raw))
        except Exception:
            continue
    return blobs


def _parse_embedded_stats(blobs: list[Any]) -> dict[str, Any]:
    """Recursively search embedded JSON for useful stat names."""
    wanted = {
        "team_rating": {"rating", "avgRating", "averageRating"},
        "passing_accuracy": {"passSuccess", "passSuccessPercentage", "passAccuracy"},
        "aerial_duels_won_pct": {"aerialWonPerGame", "aerialSuccess", "aerialWon"},
        "yellow_cards": {"yellowCards"},
        "red_cards": {"redCards"},
    }
    found: dict[str, Any] = {}

    def walk(obj: Any) -> None:
        """Walk nested JSON-like objects and collect the first matching stat values."""
        if isinstance(obj, dict):
            for stat_name, keys in wanted.items():
                if stat_name not in found:
                    for key in keys:
                        if key in obj and obj[key] not in (None, ""):
                            found[stat_name] = obj[key]
                            break
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    for blob in blobs:
        walk(blob)
    return found


def _parse_match_rows(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Parse match rows from HTML tables where available."""
    rows = []
    for tr in soup.select("tr"):
        cells = [" ".join(td.get_text(" ", strip=True).split()) for td in tr.select("td")]
        if len(cells) >= 4 and any("-" in c for c in cells):
            rows.append({"raw": cells})
    return rows[:60]
