"""Opta's xG for every shot, from FotMob, for the fixture being rendered.

The local engine prices a shot from what the event feed carries. Opta prices it
from that plus where the goalkeeper and the defenders stood, which the feed
does not publish -- and on big chances that is most of the answer. Fitted as
closely as the feed allows (``fit_xg_reference.py``), the engine still misses a
flagged chance by 0.16 on average, and a match turns on three or four of them:
Brighton 3-0 Arsenal came out 1.75-1.04 locally against Opta's 1.34-1.58, the
other way round from every source the audience compares against.

So when FotMob publishes a fixture's shot map, each shot is given Opta's own
value, paired to the event by side, half, minute, player and place. Anything
left unpaired -- a shot FotMob does not list, a penalty, an own goal, a fixture
it has no map for -- keeps the engine's value, and the render never waits on
this: every failure returns the events untouched with a line saying why.

Set ``MATCH_ANALYSIS_REFERENCE_XG=0`` to render the engine's numbers alone.
"""

from __future__ import annotations

import difflib
import gzip
import json
import math
import os
import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE = "https://www.fotmob.com"
STORE = Path(__file__).resolve().parent / "output" / "reference_xg" / "fotmob"
SOURCE = "opta_via_fotmob"

# Our competition name -> FotMob league id and slug.
LEAGUES = {
    "Premier League": (47, "premier-league"),
    "LaLiga": (87, "laliga"),
    "Serie A": (55, "serie"),
    "Bundesliga": (54, "bundesliga"),
    "Ligue 1": (53, "ligue-1"),
    "Champions League": (42, "champions-league"),
    "Europa League": (73, "europa-league"),
    "League Cup": (133, "efl-cup"),
}

# Words that name a club's form rather than the club, so "Brighton & Hove
# Albion" and "Brighton" or "Borussia M.Gladbach" and "Borussia
# Mönchengladbach" compare on what they share.
_NOISE = {"fc", "cf", "ac", "as", "sc", "ss", "ssc", "afc", "club", "de", "and",
          "calcio", "1913", "1907", "hove", "albion", "united", "city", "town"}

# Short names the packages carry that share no word with the long form. The two
# Manchester clubs get tokens of their own, because "united" and "city" are
# noise above and both would otherwise read as "manchester".
_ALIASES = {
    "man utd": "manutd", "manchester united": "manutd",
    "man city": "mancity", "manchester city": "mancity",
    "psg": "paris saint germain",
    "rbl": "rb leipzig",
}
_ALIAS_NAMES = set(_ALIASES.values())

MAX_MINUTES_APART = 2
MAX_METRES_APART = 12.0


def _norm(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    text = _ALIASES.get(text, text)
    words = text.split()
    return " ".join(w for w in words if w not in _NOISE) or text


def similar(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    # An alias is a decided identity, not a spelling to be near: "manutd" and
    # "mancity" share enough letters to pass a fuzzy match.
    if a in _ALIAS_NAMES or b in _ALIAS_NAMES:
        return 1.0 if a == b else 0.0
    if a == b or a in b or b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def surname(name: str) -> str:
    plain = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    parts = re.sub(r"[^a-z ]", " ", plain.lower()).split()
    return parts[-1] if parts else ""


def _goals(score: str) -> tuple[int, int] | None:
    found = re.findall(r"\d+", str(score or ""))
    return (int(found[0]), int(found[1])) if len(found) >= 2 else None


def _next_data(session, url: str, timeout: float = 40) -> dict | None:
    response = session.get(url, impersonate="chrome", timeout=timeout)
    if response.status_code != 200:
        return None
    found = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', response.text, re.S)
    return json.loads(found.group(1)) if found else None


def league_fixtures(session, competition: str) -> list[dict]:
    """Every finished fixture FotMob lists for the competition this season."""
    if competition not in LEAGUES:
        return []
    league_id, slug = LEAGUES[competition]
    data = _next_data(session, f"{BASE}/leagues/{league_id}/fixtures/{slug}")
    if not data:
        return []
    matches = (data["props"]["pageProps"].get("fixtures") or {}).get("allMatches") or []
    return [{"id": str(m["id"]), "url": m["pageUrl"].split("#")[0],
             "home": m["home"]["name"], "away": m["away"]["name"],
             "utc": (m.get("status") or {}).get("utcTime", "")[:10],
             "score": (m.get("status") or {}).get("scoreStr", "")}
            for m in matches if (m.get("status") or {}).get("finished")]


def match_fixture(ours: dict, fixtures: list[dict]) -> dict | None:
    """The FotMob fixture for ours: same day (a day either side for kick-offs
    near midnight UTC), both clubs recognisably the same, and the same score
    when both sides know it. ``ours`` needs date, home, away and score."""
    try:
        day = date.fromisoformat(str(ours["date"])[:10])
    except ValueError:
        return None
    window = {(day + timedelta(days=d)).isoformat() for d in (-1, 0, 1)}
    best, best_score = None, 0.0
    for fixture in fixtures:
        if fixture["utc"] not in window:
            continue
        score = min(similar(ours["home"], fixture["home"]), similar(ours["away"], fixture["away"]))
        mine, theirs = _goals(ours.get("score")), _goals(fixture["score"])
        if mine and theirs and mine != theirs:
            continue
        if score > best_score:
            best, best_score = fixture, score
    return best if best_score >= 0.6 else None


def fetch_shotmap(session, fixture: dict) -> dict | None:
    """Every shot FotMob shows for one fixture, checked to be that fixture.

    A FotMob match URL names the pairing, not the game, so the page can show
    the other meeting of the same clubs; the id on the page decides."""
    for url in (BASE + fixture["url"], f"{BASE}/match/{fixture['id']}"):
        data = _next_data(session, url)
        if not data:
            continue
        props = data["props"]["pageProps"]
        if str((props.get("general") or {}).get("matchId")) != fixture["id"]:
            continue
        shots = ((props.get("content") or {}).get("shotmap") or {}).get("shots")
        teams = [{"id": t.get("id"), "name": t.get("name"), "score": t.get("score")}
                 for t in (props.get("header") or {}).get("teams", [])]
        return {"fixture": fixture, "teams": teams, "shots": shots or []}
    return None


def their_shots(payload: dict) -> list[dict]:
    """Opta's non-penalty, non-own-goal shots, keyed to home or away."""
    teams = payload.get("teams") or []
    side = {t.get("id"): ("home" if i == 0 else "away") for i, t in enumerate(teams)}
    out = []
    for shot in payload.get("shots") or []:
        if shot.get("isOwnGoal") or str(shot.get("situation")) == "Penalty":
            continue
        if shot.get("expectedGoals") is None:
            continue
        # Opta prices a shot again once it is struck, from where it crossed the
        # line, how hard, and where the keeper was. Only a shot that reached
        # the keeper or the net has that value; a block or a miss has none.
        framed = shot.get("isOnTarget") and not shot.get("isBlocked")
        xgot = shot.get("expectedGoalsOnTarget") if framed else None
        out.append({"side": side.get(shot.get("teamId")),
                    "minute": int(shot.get("min") or 0) + int(shot.get("minAdded") or 0),
                    "period": str(shot.get("period") or ""),
                    "sur": surname(shot.get("playerName") or shot.get("fullName") or ""),
                    "x": float(shot.get("x") or 0), "y": float(shot.get("y") or 0),
                    "xg": float(shot["expectedGoals"]),
                    "xgot": None if xgot is None else float(xgot)})
    return out


def our_shots(events: pd.DataFrame, info: dict) -> list[dict]:
    """Our non-penalty, non-own-goal shots in the same shape, with their index."""
    side_of = {info.get("home_id"): "home", info.get("away_id"): "away"}
    flag = lambda column: (events[column].fillna(False).astype(bool)  # noqa: E731
                           if column in events else pd.Series(False, index=events.index))
    mask = flag("is_shot") & ~flag("is_own_goal") & ~flag("is_penalty") & ~flag("is_penalty_shootout")
    out = []
    for index, shot in events[mask].iterrows():
        out.append({"index": index, "side": side_of.get(shot.get("team_id")),
                    "minute": int(shot.get("minute") or 0),
                    "period": str(shot.get("period") or ""),
                    "sur": surname(shot.get("player") or ""),
                    "x": float(shot.get("x") or 0) * 1.05,
                    "y": float(shot.get("y") or 0) * 0.68})
    return out


def pair(ours: list[dict], theirs: list[dict]) -> list[tuple[int, int]]:
    """One-to-one pairs of shots, closest first, on side, half, minute, name, place.

    WhoScored runs added time on from 45 and 90, FotMob writes 90+5, so the
    minutes compare once FotMob's are added up. Coordinates are both in metres
    towards the goal being attacked once WhoScored's percentages are scaled.
    """
    candidates = []
    for i, a in enumerate(ours):
        for j, b in enumerate(theirs):
            if a["side"] != b["side"] or a["side"] is None:
                continue
            if a["period"] and b["period"] and a["period"] != b["period"]:
                continue
            apart = abs(a["minute"] - b["minute"])
            metres = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
            if apart > MAX_MINUTES_APART or metres > MAX_METRES_APART:
                continue
            same_name = a["sur"] and a["sur"] == b["sur"]
            candidates.append((apart + (0 if same_name else 2.0) + metres / 6.0, i, j))
    used_a, used_b, pairs = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
    return pairs


def cached(fotmob_id) -> dict | None:
    path = STORE / f"{fotmob_id}.json.gz"
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


# The fixture header of every stored map, read once per process: a re-price of
# every package would otherwise open every stored map once per package.
_INDEX: list[dict] | None = None


def _index() -> list[dict]:
    global _INDEX
    if _INDEX is None:
        _INDEX = []
        for path in STORE.glob("*.json.gz") if STORE.is_dir() else ():
            fixture = (cached(path.name.split(".")[0]) or {}).get("fixture")
            if fixture:
                _INDEX.append(fixture)
    return _INDEX


def save(payload: dict, package: str | None = None) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    target = STORE / f"{payload['fixture']['id']}.json.gz"
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        json.dump({**payload, "package": package}, handle)
    if _INDEX is not None:
        _INDEX.append(payload["fixture"])


def _find_cached(info: dict) -> dict | None:
    """A stored shot map for this fixture, found without the network."""
    ours = {"date": info.get("date"), "home": info.get("home_name"),
            "away": info.get("away_name"), "score": info.get("score")}
    fixture = match_fixture(ours, _index())
    return cached(fixture["id"]) if fixture else None


def reference_payload(info: dict, session=None) -> tuple[dict | None, str]:
    """This fixture's shot map: from the store if it is there, else from FotMob."""
    payload = _find_cached(info)
    if payload is not None:
        return payload, "stored shot map"
    competition = info.get("competition")
    if competition not in LEAGUES:
        return None, f"no FotMob league for {competition!r}"
    if session is None:
        from curl_cffi import requests as cr

        session = cr.Session()
    fixture = match_fixture({"date": info.get("date"), "home": info.get("home_name"),
                             "away": info.get("away_name"), "score": info.get("score")},
                            league_fixtures(session, competition))
    if fixture is None:
        return None, "fixture not found on FotMob"
    payload = fetch_shotmap(session, fixture)
    if payload is None:
        return None, "FotMob page had no shot map"
    return payload, "fetched from FotMob"


def apply_reference_xg(events: pd.DataFrame, info: dict, payload: dict | None = None,
                       package: str | None = None) -> tuple[pd.DataFrame, str]:
    """Events with Opta's value on every shot it can be paired to.

    Never raises. Returns the events untouched, and why, when anything is
    missing. ``payload`` skips the lookup (tests, and callers that hold one).
    """
    if os.environ.get("MATCH_ANALYSIS_REFERENCE_XG", "1").strip().lower() in {"0", "false", "no", "off"}:
        return events, "reference xG switched off; engine values kept"
    try:
        how = "supplied shot map"
        if payload is None:
            payload, how = reference_payload(info)
            if payload is None:
                return events, f"{how}; engine values kept"
            if how == "fetched from FotMob":
                save(payload, package)
        ours, theirs = our_shots(events, info), their_shots(payload)
        pairs = pair(ours, theirs)
        if not pairs:
            return events, "no shot could be paired; engine values kept"
        out = events.copy()
        if "xg_source" not in out:
            out["xg_source"] = ""
        out["xg_source"] = out["xg_source"].astype(object)
        # Post-shot xG rides along: match_metrics.post_shot_xg reads it before
        # falling back to the local placement estimate.
        out["xgot_reference"] = float("nan")
        for i, j in pairs:
            out.at[ours[i]["index"], "xG"] = round(theirs[j]["xg"], 4)
            out.at[ours[i]["index"], "xg_source"] = SOURCE
            if theirs[j]["xgot"] is not None:
                out.at[ours[i]["index"], "xgot_reference"] = round(theirs[j]["xgot"], 4)
        info["xg_reference_source"] = (f"Opta via FotMob for {len(pairs)} of {len(ours)} shots, "
                                       "internal model for the rest")
        return out, (f"Opta xG on {len(pairs)} of {len(ours)} shots ({how}); "
                     f"{len(theirs) - len(pairs)} FotMob shot(s) not in the event feed")
    except Exception as error:  # the render must never wait on this
        return events, f"reference xG failed ({type(error).__name__}: {error}); engine values kept"
