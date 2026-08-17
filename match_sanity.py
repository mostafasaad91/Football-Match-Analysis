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
    check_the_match_has_enough_events,
    check_no_player_appears_for_both_sides,
    check_event_players_belong_to_their_team,
    check_goals_match_the_score,
)


def inspect(events: pd.DataFrame, players: pd.DataFrame, xg: pd.DataFrame,
            info: dict) -> list[Problem]:
    """Run every check. An empty list means the fixture looks coherent."""
    problems: list[Problem] = []
    for check in CHECKS:
        try:
            if check is check_goals_match_the_score:
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
