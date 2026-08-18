"""Does the parsed match describe a real, coherent fixture?

Everything downstream — the report, the posters, the article — reads the frames
and writes about them with complete confidence. None of it asks whether the
frames make sense. A fixture collected from the wrong URL, or two matches
merged, produces a well-typeset, fully-illustrated document about a game that
did not happen, and nothing in the pipeline notices.

The checks here are deliberately the kind that cannot be wrong about a genuine
match: a player belongs to one side, the goals add up to the score, the teams
named are the teams that played. Each returns a Problem naming what it found
and what it expected, so a failure is a sentence rather than a stack trace.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class Problem:
    check: str
    detail: str

    def __str__(self) -> str:
        return f"{self.check}: {self.detail}"


def _bool(series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _goals(xg: pd.DataFrame, team: str) -> int | None:
    row = xg[xg["team"].astype(str).str.lower().eq(str(team).lower())]
    if row.empty or "goals" not in row.columns:
        return None
    try:
        return int(float(row.iloc[0]["goals"]))
    except (TypeError, ValueError):
        return None


def check_no_player_appears_for_both_sides(events, players, info) -> list[Problem]:
    """A player belongs to one team. Two means the frames were merged wrong."""
    if players is None or players.empty or "name" not in players.columns:
        return []
    named = players.dropna(subset=["name", "team_id"])
    counts = named.groupby("name")["team_id"].nunique()
    shared = sorted(counts[counts > 1].index.astype(str))
    if not shared:
        return []
    listed = ", ".join(shared[:6]) + (" …" if len(shared) > 6 else "")
    return [Problem(
        "player on two teams",
        f"{len(shared)} player(s) are listed for both sides: {listed}",
    )]


def check_event_players_belong_to_their_team(events, players, info) -> list[Problem]:
    """Every player acting for a team is in that team's squad."""
    if players is None or players.empty or "name" not in players.columns:
        return []
    squads: dict[int, set[str]] = {}
    for team_id, group in players.dropna(subset=["name"]).groupby("team_id"):
        try:
            squads[int(team_id)] = set(group["name"].astype(str))
        except (TypeError, ValueError):
            continue
    if len(squads) < 2:
        return []

    acting = events.dropna(subset=["player", "team_id"])
    strays: dict[int, set[str]] = {}
    for team_id, group in acting.groupby("team_id"):
        try:
            squad = squads.get(int(team_id))
        except (TypeError, ValueError):
            continue
        if squad is None:
            continue
        loose = set(group["player"].astype(str)) - squad
        if loose:
            strays[int(team_id)] = loose

    problems = []
    names = {int(info["home_id"]): str(info["home_name"]),
             int(info["away_id"]): str(info["away_name"])}
    for team_id, loose in strays.items():
        listed = ", ".join(sorted(loose)[:6]) + (" …" if len(loose) > 6 else "")
        problems.append(Problem(
            "player outside the squad",
            f"{len(loose)} player(s) act for {names.get(team_id, team_id)} "
            f"without appearing in its squad: {listed}",
        ))
    return problems


def check_goals_match_the_score(events, xg, info) -> list[Problem]:
    """The goals in the events add up to the score the package prints."""
    problems = []
    counted = {}
    goal_rows = events[_bool(events.get("is_goal"))] if "is_goal" in events else events.iloc[0:0]
    for side in ("home", "away"):
        team_id = info.get(f"{side}_id")
        name = str(info.get(f"{side}_name") or side)
        try:
            team_id = int(team_id)
        except (TypeError, ValueError):
            continue
        # An own goal is recorded against the scorer's own team, so credit the
        # opponent when the column says so.
        own = _bool(goal_rows.get("is_own_goal")) if "is_own_goal" in goal_rows else None
        for_team = goal_rows["team_id"].eq(team_id)
        if own is not None and len(own):
            scored = int((for_team & ~own).sum() + (~for_team & own).sum())
        else:
            scored = int(for_team.sum())
        counted[name] = scored
        stated = _goals(xg, name)
        if stated is not None and stated != scored:
            problems.append(Problem(
                "score does not match the events",
                f"{name}: the export says {stated} goal(s), the events contain {scored}",
            ))
    return problems


def check_the_published_shots_match_the_events(events, xg, info) -> list[Problem]:
    """The team's shot totals and the shot events tell the same story.

    The exported totals prefer the provider's own published figures when it has
    them, and the events are what everything per-player is built from. A gap of
    one is normal — WhoScored counts a shot the keeper turned round the post as
    on target while the event is coded as blocked — and that gap is why adding
    up the players' radars can land a shot short of the team card.

    A wide gap is different: it means the two files were not produced from the
    same match, which nothing else here would notice.
    """
    if "shot_whoscored_type" not in events.columns:
        return []
    shots = events[_bool(events.get("is_shot"))]
    problems = []
    for side in ("home", "away"):
        try:
            team_id = int(info[f"{side}_id"])
        except (KeyError, TypeError, ValueError):
            continue
        name = str(info.get(f"{side}_name") or side)
        rows = xg[xg["team"].astype(str).str.lower().eq(name.lower())]
        if rows.empty:
            continue
        row, mine = rows.iloc[0], shots[shots["team_id"].eq(team_id)]
        for column, counted in (
            ("shots", len(mine)),
            ("on_target", int(mine["shot_whoscored_type"].isin(["Goal", "SavedShot"]).sum())),
        ):
            if column not in row.index:
                continue
            try:
                stated = int(float(row[column]))
            except (TypeError, ValueError):
                continue
            gap = abs(stated - counted)
            if gap > max(2, 0.2 * max(stated, counted, 1)):
                problems.append(Problem(
                    "shot totals disagree with the events",
                    f"{name}: the export says {stated} {column.replace('_', ' ')}, "
                    f"the events contain {counted}",
                ))
    return problems


SQUADS_FILE = "squads.json"


def _known_squads(root=None) -> dict[str, set[str]]:
    """Rosters the user maintains, keyed by team name. Empty when absent.

    The pipeline has no idea who plays for whom: it reads the squads out of the
    provider's own match feed, so a fixture whose feed lists the wrong side for
    a player is internally consistent and passes every other check here. This
    is the only place an outside opinion can enter, and it is opt-in because
    the project cannot keep a league's transfers up to date on the user's
    behalf.
    """
    from pathlib import Path

    base = Path(root) if root else Path(__file__).resolve().parent
    path = base / SQUADS_FILE
    if not path.exists():
        return {}
    try:
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    squads: dict[str, set[str]] = {}
    for team, names in (raw or {}).items():
        if isinstance(names, list):
            squads[str(team).strip().lower()] = {str(n).strip() for n in names if str(n).strip()}
    return squads


def check_players_belong_to_the_squad_you_named(events, players, info) -> list[Problem]:
    """Every player is in the roster the user keeps for that team, if any."""
    squads = _known_squads()
    if not squads or players is None or players.empty or "name" not in players.columns:
        return []

    problems = []
    for side in ("home", "away"):
        name = str(info.get(f"{side}_name") or "").strip()
        known = squads.get(name.lower())
        if not known:
            continue
        try:
            team_id = int(info[f"{side}_id"])
        except (KeyError, TypeError, ValueError):
            continue
        listed = players[players["team_id"].eq(team_id)]["name"].dropna().astype(str)
        # Compare on a loose form so an accent or a middle name does not
        # produce a false alarm about a player who is in the roster.
        folded = {_fold(n): n for n in known}
        strangers = sorted({n for n in listed if _fold(n) not in folded})
        if strangers:
            shown = ", ".join(strangers[:6]) + (" …" if len(strangers) > 6 else "")
            problems.append(Problem(
                "player not in the squad you listed",
                f"{name}: {len(strangers)} player(s) are not in your {SQUADS_FILE} "
                f"roster: {shown}",
            ))
    return problems


# Letters that are not an accented Latin letter underneath, so decomposition
# leaves nothing behind: "Ødegaard" folded to "degaard" and a roster typed as
# "Odegaard" was reported as a player who does not exist.
_LETTER_ALIASES = {
    "ø": "o", "æ": "ae", "œ": "oe", "å": "a", "ð": "d", "þ": "th",
    "ß": "ss", "ł": "l", "đ": "d", "ħ": "h", "ı": "i", "ŋ": "n",
}


def _fold(name: str) -> str:
    """A name reduced to letters, for comparing rosters written by hand."""
    import re
    import unicodedata

    lowered = str(name).lower()
    for letter, plain in _LETTER_ALIASES.items():
        lowered = lowered.replace(letter, plain)
    stripped = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", stripped)


def check_no_player_changed_team_since_a_stored_match(events, players, info) -> list[Problem]:
    """A player the history has seen for another team is worth a second look.

    No configuration and no external source: the project's own match history
    already records who played for whom. A real transfer trips this once and
    the user accepts it; a fixture attributing a player to the wrong side trips
    it too, which is the case nothing else here can see.
    """
    if players is None or players.empty or "name" not in players.columns:
        return []
    try:
        import sqlite3
        from pathlib import Path

        db = Path(__file__).resolve().parent / "output" / "match_history.db"
        if not db.exists():
            return []
        connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            seen: dict[str, set[str]] = {}
            for name, team in connection.execute(
                "SELECT player, team FROM player_match_stats WHERE team IS NOT NULL"
            ):
                seen.setdefault(_fold(name), set()).add(str(team))
        finally:
            connection.close()
    except Exception:
        return []
    if not seen:
        return []

    problems = []
    for side in ("home", "away"):
        name = str(info.get(f"{side}_name") or "").strip()
        try:
            team_id = int(info[f"{side}_id"])
        except (KeyError, TypeError, ValueError):
            continue
        moved = []
        for player in players[players["team_id"].eq(team_id)]["name"].dropna().astype(str):
            previous = seen.get(_fold(player))
            if previous and name and name not in previous:
                moved.append(f"{player} (last seen for {sorted(previous)[0]})")
        if moved:
            shown = ", ".join(moved[:5]) + (" …" if len(moved) > 5 else "")
            problems.append(Problem(
                "player listed for a different team than the history has",
                f"{name}: {shown}",
            ))
    return problems


def check_the_teams_are_two_and_named(events, info) -> list[Problem]:
    """Exactly two team ids act in the match, and both are the ones named."""
    ids = {int(v) for v in events["team_id"].dropna().unique()
           if str(v).strip() not in ("", "nan")}
    try:
        expected = {int(info["home_id"]), int(info["away_id"])}
    except (KeyError, TypeError, ValueError):
        return [Problem("fixture identity", "the match info names no team ids")]
    if ids == expected:
        return []
    extra, absent = sorted(ids - expected), sorted(expected - ids)
    detail = []
    if extra:
        detail.append(f"events contain team id(s) {extra} that the fixture does not name")
    if absent:
        detail.append(f"the fixture names team id(s) {absent} that never act")
    return [Problem("fixture identity", "; ".join(detail))]


def check_the_match_has_enough_events(events, info) -> list[Problem]:
    """A parsed match with almost no events is a collection failure."""
    if len(events) >= 400:
        return []
    return [Problem(
        "too few events",
        f"{len(events)} events parsed; a full match is normally well over a thousand",
    )]


CHECKS = (
    check_the_teams_are_two_and_named,
    check_the_published_shots_match_the_events,
    check_the_match_has_enough_events,
    check_no_player_appears_for_both_sides,
    check_players_belong_to_the_squad_you_named,
    check_no_player_changed_team_since_a_stored_match,
    check_event_players_belong_to_their_team,
    check_goals_match_the_score,
)


def inspect(events: pd.DataFrame, players: pd.DataFrame, xg: pd.DataFrame,
            info: dict) -> list[Problem]:
    """Run every check. An empty list means the fixture looks coherent."""
    problems: list[Problem] = []
    for check in CHECKS:
        try:
            if check in (check_goals_match_the_score,
                         check_the_published_shots_match_the_events):
                problems.extend(check(events, xg, info))
            elif check in (check_the_teams_are_two_and_named,
                           check_the_match_has_enough_events):
                problems.extend(check(events, info))
            else:
                problems.extend(check(events, players, info))
        except Exception as error:  # a broken check must not mask the others
            problems.append(Problem(check.__name__, f"check failed: {error!r}"))
    return problems


def describe(problems: Iterable[Problem]) -> str:
    return "\n".join(f"  - {problem}" for problem in problems)
