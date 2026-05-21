# pyright: reportMissingImports=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
"""Experimental SofaScore player-stat fetcher for the trial scripts.

This module is intentionally isolated from the normal WhoScored pipeline.  The
trial scripts can use it to replace only the player-stat tables, while the
original analysis scripts keep their current behaviour.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import quote_plus

import pandas as pd


SOFA_STAT_GROUPS = [
    ("Identity", [
        ("name", "Player"),
        ("position", "Pos"),
        ("shirt_no", "#"),
        ("minutesPlayed", "Min"),
    ]),
    ("General", [
        ("goals", "Goals"),
        ("assists", "Assists"),
        ("tacklesWon", "Tackles (won)"),
        ("accurate_passes_fmt", "Accurate passes"),
        ("duels_won_fmt", "Duels (won)"),
        ("ground_duels_won_fmt", "Ground duels (won)"),
        ("aerial_duels_won_fmt", "Aerial duels (won)"),
        ("minutesPlayed", "Minutes played"),
    ]),
    ("Attacking", [
        ("goals", "Goals"),
        ("assists", "Assists"),
        ("totalShots", "Total shots"),
        ("onTargetScoringAttempt", "On target"),
        ("shotOffTarget", "Off target"),
        ("blockedScoringAttempt", "Blocked shots"),
        ("bigChanceCreated", "Big chances created"),
        ("bigChanceMissed", "Big chances missed"),
        ("expectedGoals", "Expected goals"),
        ("expectedGoalsOnTarget", "xGOT"),
        ("expectedAssists", "Expected assists"),
        ("successfulDribbles_fmt", "Successful dribbles"),
    ]),
    ("Defending", [
        ("tacklesWon", "Tackles (won)"),
        ("interceptions", "Interceptions"),
        ("outfielderBlock", "Blocks"),
        ("clearances", "Clearances"),
        ("ballRecovery", "Ball recovery"),
        ("dribbledPast", "Dribbled past"),
        ("errorLeadToShot", "Error led to shot"),
        ("errorLeadToGoal", "Error led to goal"),
    ]),
    ("Passing", [
        ("accurate_passes_fmt", "Accurate passes"),
        ("passesKey", "Key passes"),
        ("accurateLongBalls_fmt", "Accurate long balls"),
        ("accurateOwnHalfPasses_fmt", "Own half passes"),
        ("accurateOppositionHalfPasses_fmt", "Opp. half passes"),
        ("accurateCross_fmt", "Accurate crosses"),
        ("accurateFinalThirdPasses_fmt", "Final third passes"),
        ("accurateThroughBall_fmt", "Through balls"),
    ]),
    ("Duels", [
        ("duels_won_fmt", "Duels (won)"),
        ("ground_duels_won_fmt", "Ground duels (won)"),
        ("aerial_duels_won_fmt", "Aerial duels (won)"),
        ("successfulDribbles_fmt", "Successful dribbles"),
        ("possessionLostCtrl", "Possession lost"),
        ("wasFouled", "Was fouled"),
        ("foulsCommited", "Fouls"),
        ("challengeLost", "Challenge lost"),
    ]),
    ("Goalkeeping", [
        ("saves", "Saves"),
        ("savedShotsFromInsideTheBox", "Saves inside box"),
        ("accurateKeeperSweeper_fmt", "Keeper sweeper"),
        ("goalsPrevented", "Goals prevented"),
        ("highClaims", "High claims"),
        ("punches", "Punches"),
    ]),
]


@dataclass
class SofaFetchResult:
    event_id: int | None
    confidence: float
    source: str
    player_stats: dict[str, pd.DataFrame] | None
    warning: str | None = None


def fetch_advanced_player_stats(
    info: dict[str, Any],
    *,
    event_id: int | str | None = None,
    auto_search: bool = True,
    min_confidence: float = 0.82,
    verbose: bool = False,
) -> SofaFetchResult:
    """Find a SofaScore event and return normalized home/away player tables."""
    explicit_id = _coerce_event_id(event_id or os.environ.get("SOFASCORE_EVENT_ID"))
    if explicit_id is not None:
        lineups = _fetch_lineups(explicit_id, verbose=verbose)
        if lineups:
            return SofaFetchResult(
                event_id=explicit_id,
                confidence=1.0,
                source="manual_event_id",
                player_stats=_normalize_lineups(lineups),
            )
        return SofaFetchResult(
            event_id=explicit_id,
            confidence=0.0,
            source="manual_event_id",
            player_stats=None,
            warning=f"SofaScore event {explicit_id} did not return lineups.",
        )

    if not auto_search:
        return SofaFetchResult(
            event_id=None,
            confidence=0.0,
            source="disabled",
            player_stats=None,
            warning="SofaScore auto-search is disabled and no event id was provided.",
        )

    match = find_sofascore_event(info, min_confidence=min_confidence, verbose=verbose)
    if not match:
        return SofaFetchResult(
            event_id=None,
            confidence=0.0,
            source="auto_search",
            player_stats=None,
            warning="No SofaScore event matched the WhoScored teams with enough confidence.",
        )

    lineups = _fetch_lineups(match["id"], verbose=verbose)
    if not lineups:
        return SofaFetchResult(
            event_id=match["id"],
            confidence=match["confidence"],
            source="auto_search",
            player_stats=None,
            warning=f"SofaScore event {match['id']} matched but lineups were unavailable.",
        )

    return SofaFetchResult(
        event_id=match["id"],
        confidence=match["confidence"],
        source="auto_search",
        player_stats=_normalize_lineups(lineups),
    )


def find_sofascore_event(
    info: dict[str, Any],
    *,
    min_confidence: float = 0.82,
    verbose: bool = False,
) -> dict[str, Any] | None:
    home = str(info.get("home_name") or info.get("homeTeam") or "").strip()
    away = str(info.get("away_name") or info.get("awayTeam") or "").strip()
    if not home or not away:
        return None

    queries = [
        f"{home} {away}",
        f"{away} {home}",
        f"{home} vs {away}",
    ]
    candidates: list[dict[str, Any]] = []
    for query in queries:
        for endpoint in _search_urls(query):
            payload = _fetch_json(endpoint, verbose=verbose)
            if payload:
                candidates.extend(_extract_event_candidates(payload))

    ranked = []
    for cand in candidates:
        score = _score_candidate(cand, home, away, info)
        if score >= min_confidence:
            item = dict(cand)
            item["confidence"] = score
            ranked.append(item)

    if not ranked:
        return None
    ranked.sort(key=lambda c: c.get("confidence", 0), reverse=True)
    return ranked[0]


def _search_urls(query: str) -> list[str]:
    q = quote_plus(query)
    return [
        f"https://www.sofascore.com/api/v1/search/all?q={q}&page=0",
        f"https://api.sofascore.com/api/v1/search/all?q={q}&page=0",
        f"https://www.sofascore.com/api/v1/search/events?q={q}&page=0",
        f"https://api.sofascore.com/api/v1/search/events?q={q}&page=0",
    ]


def _lineups_urls(event_id: int) -> list[str]:
    return [
        f"https://www.sofascore.com/api/v1/event/{event_id}/lineups",
        f"https://api.sofascore.com/api/v1/event/{event_id}/lineups",
    ]


def _fetch_lineups(event_id: int, *, verbose: bool = False) -> dict[str, Any] | None:
    for url in _lineups_urls(event_id):
        payload = _fetch_json(url, verbose=verbose)
        if payload and ("home" in payload or "away" in payload):
            return payload
    return None


def _fetch_json(url: str, *, verbose: bool = False) -> Any | None:
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.sofascore.com",
        "Referer": "https://www.sofascore.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    }

    # First try normal requests/cloudscraper.  Some machines/accounts can read
    # SofaScore this way; others get 403 and need the browser fallback.
    try:
        try:
            import cloudscraper  # type: ignore
            session = cloudscraper.create_scraper()
            resp = session.get(url, headers=headers, timeout=18)
        except Exception:
            import requests
            resp = requests.get(url, headers=headers, timeout=18)
        if resp.status_code == 200:
            return resp.json()
        if verbose:
            print(f"[SofaScore] HTTP {resp.status_code} for {url}")
    except Exception as exc:
        if verbose:
            print(f"[SofaScore] requests failed for {url}: {exc}")

    return _fetch_json_with_browser(url, verbose=verbose)


def _fetch_json_with_browser(url: str, *, verbose: bool = False) -> Any | None:
    try:
        import undetected_chromedriver as uc  # type: ignore
        from selenium.webdriver.common.by import By  # type: ignore
    except Exception as exc:
        if verbose:
            print(f"[SofaScore] browser fallback unavailable: {exc}")
        return None

    driver = None
    try:
        opts = uc.ChromeOptions()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--lang=en-US")
        driver = uc.Chrome(options=opts)
        driver.set_page_load_timeout(35)
        driver.get(url)
        time.sleep(1.5)
        text = driver.find_element(By.TAG_NAME, "body").text
        if not text:
            return None
        return json.loads(text)
    except Exception as exc:
        if verbose:
            print(f"[SofaScore] browser fetch failed for {url}: {exc}")
        return None
    finally:
        try:
            if driver is not None:
                driver.quit()
        except Exception:
            pass


def _extract_event_candidates(payload: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for item in _walk_dicts(payload):
        event = item.get("event") if isinstance(item.get("event"), dict) else item
        if not isinstance(event, dict):
            continue
        if not {"homeTeam", "awayTeam"}.issubset(event.keys()):
            continue
        event_id = _coerce_event_id(event.get("id"))
        if event_id is None:
            continue
        found.append({
            "id": event_id,
            "home": _team_name(event.get("homeTeam")),
            "away": _team_name(event.get("awayTeam")),
            "startTimestamp": event.get("startTimestamp"),
            "status": event.get("status"),
            "homeScore": _score_value(event.get("homeScore")),
            "awayScore": _score_value(event.get("awayScore")),
        })

    unique: dict[int, dict[str, Any]] = {}
    for item in found:
        unique[item["id"]] = item
    return list(unique.values())


def _walk_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_dicts(value)


def _score_candidate(cand: dict[str, Any], home: str, away: str, info: dict[str, Any]) -> float:
    direct = (_similar(cand.get("home"), home) + _similar(cand.get("away"), away)) / 2
    swapped = (_similar(cand.get("home"), away) + _similar(cand.get("away"), home)) / 2
    base = max(direct, swapped)

    # Small boost when the known score also matches.
    try:
        h_score = int(info.get("home_score", info.get("homeScore", -999)))
        a_score = int(info.get("away_score", info.get("awayScore", -999)))
        if cand.get("homeScore") == h_score and cand.get("awayScore") == a_score:
            base += 0.06
    except Exception:
        pass
    return min(1.0, base)


def _normalize_lineups(payload: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {
        "home": _normalize_team_players(payload.get("home") or {}),
        "away": _normalize_team_players(payload.get("away") or {}),
    }


def _normalize_team_players(team_payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for entry in team_payload.get("players", []) or []:
        player = entry.get("player") or entry
        stats = entry.get("statistics") or entry.get("stats") or {}
        row = {
            "name": player.get("shortName") or player.get("name") or entry.get("name") or "N/A",
            "position": entry.get("position") or player.get("position") or "N/A",
            "shirt_no": entry.get("shirtNumber") or entry.get("jerseyNumber") or player.get("jerseyNumber"),
            "is_first_xi": not bool(entry.get("substitute", False)),
            "minutesPlayed": _pick(stats, "minutesPlayed", "minutes", "playedMinutes"),
            "touches": _pick(stats, "touches"),
            "possessionLostCtrl": _pick(stats, "possessionLostCtrl", "possessionLost"),
            "foulsCommited": _pick(stats, "fouls", "foulsCommited", "foulsCommitted"),
            "wasFouled": _pick(stats, "wasFouled"),
            "goals": _pick(stats, "goals", "goal") or 0,
            "assists": _pick(stats, "goalAssist", "assists", "assist") or 0,
            "totalShots": _pick(stats, "totalShots", "totalShot", "shotsTotal"),
            "onTargetScoringAttempt": _pick(stats, "onTargetScoringAttempt", "shotsOnTarget"),
            "shotOffTarget": _pick(stats, "shotOffTarget", "shotsOffTarget"),
            "blockedScoringAttempt": _pick(stats, "blockedScoringAttempt"),
            "bigChanceCreated": _pick(stats, "bigChanceCreated"),
            "bigChanceMissed": _pick(stats, "bigChanceMissed"),
            "expectedGoals": _pick(stats, "expectedGoals", "xg"),
            "expectedGoalsOnTarget": _pick(stats, "expectedGoalsOnTarget", "xgot"),
            "expectedAssists": _pick(stats, "expectedAssists", "xa"),
            "passesKey": _pick(stats, "keyPass", "keyPasses", "passesKey"),
            "interceptions": _pick(stats, "interceptionWon", "interceptions"),
            "outfielderBlock": _pick(stats, "outfielderBlock"),
            "clearances": _pick(stats, "totalClearance", "clearances"),
            "ballRecovery": _pick(stats, "ballRecovery", "possessionWon"),
            "dribbledPast": _pick(stats, "dribbledPast"),
            "errorLeadToShot": _pick(stats, "errorLeadToAShot", "errorLeadToShot"),
            "errorLeadToGoal": _pick(stats, "errorLeadToGoal"),
            "challengeLost": _pick(stats, "challengeLost"),
            "saves": _pick(stats, "saves"),
            "savedShotsFromInsideTheBox": _pick(stats, "savedShotsFromInsideTheBox"),
            "highClaims": _pick(stats, "goodHighClaim", "highClaims"),
            "punches": _pick(stats, "punches"),
            "goalsPrevented": _pick(stats, "goalsPrevented"),
        }

        row["tacklesWon"] = _pick(stats, "totalTackle", "wonTackle", "tacklesWon")
        row["accurate_passes_fmt"] = _made_total_pct(
            _pick(stats, "accuratePass", "accuratePasses"),
            _pick(stats, "totalPass", "passesTotal"),
        )
        row["accurateLongBalls_fmt"] = _made_total_pct(
            _pick(stats, "accurateLongBalls"),
            _pick(stats, "totalLongBalls", "longBalls"),
        )
        row["accurateCross_fmt"] = _made_total_pct(
            _pick(stats, "accurateCross", "accurateCrosses"),
            _pick(stats, "totalCross", "totalCrosses"),
        )
        row["accurateOwnHalfPasses_fmt"] = _made_total_pct(
            _pick(stats, "accurateOwnHalfPasses"),
            _pick(stats, "totalOwnHalfPasses"),
        )
        row["accurateOppositionHalfPasses_fmt"] = _made_total_pct(
            _pick(stats, "accurateOppositionHalfPasses"),
            _pick(stats, "totalOppositionHalfPasses"),
        )
        row["accurateFinalThirdPasses_fmt"] = _made_total_pct(
            _pick(stats, "accurateFinalThirdPasses"),
            _pick(stats, "totalFinalThirdPasses"),
        )
        row["accurateThroughBall_fmt"] = _made_total_pct(
            _pick(stats, "accurateThroughBall"),
            _pick(stats, "totalThroughBall"),
        )
        row["successfulDribbles_fmt"] = _made_total_pct(
            _pick(stats, "wonContest", "successfulDribbles"),
            _pick(stats, "totalContest", "attemptedDribbles"),
        )
        duel_won = _pick(stats, "duelWon", "duelsWon")
        duel_lost = _pick(stats, "duelLost", "duelsLost")
        aerial_won = _pick(stats, "aerialWon", "aerialsWon", "aerialDuelsWon")
        aerial_lost = _pick(stats, "aerialLost", "aerialsLost", "aerialDuelsLost")
        ground_won = _pick(stats, "groundDuelsWon", "groundDuelWon")
        ground_lost = _pick(stats, "groundDuelsLost", "groundDuelLost")
        if ground_won is None:
            ground_won = _subtract_nonnegative(duel_won, aerial_won)
        if ground_lost is None:
            ground_lost = _subtract_nonnegative(duel_lost, aerial_lost)

        row["duels_won_fmt"] = _won_total(duel_won, duel_lost)
        row["ground_duels_won_fmt"] = _won_total(
            ground_won,
            ground_lost,
        )
        row["aerial_duels_won_fmt"] = _won_total(
            aerial_won,
            aerial_lost,
        )
        row["accurateKeeperSweeper_fmt"] = _made_total_pct(
            _pick(stats, "accurateKeeperSweeper"),
            _pick(stats, "totalKeeperSweeper"),
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.reset_index(drop=True)
    return df


def _pick(stats: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = stats.get(key)
        if value is not None:
            return _flatten(value)
    return None


def _flatten(value: Any) -> Any:
    if isinstance(value, dict):
        if "total" in value:
            return value["total"]
        nums = [v for v in value.values() if isinstance(v, (int, float))]
        if nums:
            return sum(nums)
    return value


def _made_total_pct(made: Any, total: Any) -> str | None:
    made_i = _to_int(made)
    total_i = _to_int(total)
    if made_i is None and total_i is None:
        return None
    if made_i is None:
        made_i = 0
    if total_i is None:
        return str(made_i)
    pct = round((made_i / total_i) * 100) if total_i else 0
    return f"{made_i}/{total_i} ({pct}%)"


def _won_total(won: Any, lost: Any) -> str | None:
    won_i = _to_int(won)
    lost_i = _to_int(lost)
    if won_i is None and lost_i is None:
        return None
    if won_i is None:
        won_i = 0
    if lost_i is None:
        lost_i = 0
    return f"{won_i + lost_i} ({won_i})"


def _subtract_nonnegative(total_part: Any, known_part: Any) -> int | None:
    total_i = _to_int(total_part)
    known_i = _to_int(known_part)
    if total_i is None:
        return None
    if known_i is None:
        known_i = 0
    return max(total_i - known_i, 0)


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value))
    except Exception:
        return None


def _team_name(team_obj: Any) -> str:
    if isinstance(team_obj, dict):
        return str(team_obj.get("name") or team_obj.get("shortName") or "").strip()
    return str(team_obj or "").strip()


def _score_value(score_obj: Any) -> int | None:
    if isinstance(score_obj, dict):
        return _to_int(score_obj.get("current") or score_obj.get("display") or score_obj.get("normaltime"))
    return _to_int(score_obj)


def _normalize_name(name: Any) -> str:
    text = str(name or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _similar(a: Any, b: Any) -> float:
    aa = _normalize_name(a)
    bb = _normalize_name(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0
    if aa in bb or bb in aa:
        return 0.93
    return SequenceMatcher(None, aa, bb).ratio()


def _coerce_event_id(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(str(value).strip())
    except Exception:
        return None
