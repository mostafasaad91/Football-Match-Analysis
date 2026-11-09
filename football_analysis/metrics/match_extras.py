"""Match measures added after the November 2026 review.

Each one answers a question a reader actually asks, from columns the parse
already keeps:

    expected_points     what the chances were worth in points
    xg_by_situation     where each side's chances came from
    ppda_timeline       when each side pressed, in 15-minute bins
    game_control        how much of the match each side was the likelier winner
    set_piece_summary   corners and dead balls, for and against
    cutbacks            low balls pulled back from the byline
    through_balls       passes played in behind, per player
    keeper_actions      sweeping, claims, punches and diving saves
    shot_profile        headers, first-time shots, big chances missed
    tactical_fouls      fouls that stop a counter straight after losing the ball
    offsides            the trap sprung, and runs mistimed

``match_extras`` gathers them for one fixture into a plain dict that the
boards, the posters and the article read.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _truthy(series: pd.Series | None, index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index)
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _q(events: pd.DataFrame) -> pd.Series:
    return events.get("qualifier_names", pd.Series("", index=events.index)).astype(str)


def _has(events: pd.DataFrame, name: str) -> pd.Series:
    return _q(events).str.contains(name, regex=False)


def _shots(events: pd.DataFrame) -> pd.DataFrame:
    shots = events[_truthy(events.get("is_shot"), events.index)]
    return shots[~_truthy(shots.get("is_penalty_shootout"), shots.index)]


def _ok(events: pd.DataFrame) -> pd.Series:
    return events.get("outcome", pd.Series("", index=events.index)).astype(str).str.lower().eq("successful")


def _clock(events: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(events["minute"], errors="coerce").fillna(0) * 60
            + pd.to_numeric(events.get("second", 0), errors="coerce").fillna(0))


# ── Expected points ────────────────────────────────────────────────────
def _goal_distribution(xgs: np.ndarray, max_goals: int = 12) -> np.ndarray:
    """Probability of 0..max goals from independent shots (Poisson-binomial)."""
    dist = np.zeros(max_goals + 1)
    dist[0] = 1.0
    for p in np.clip(xgs, 0, 0.99):
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return dist


def expected_points(events: pd.DataFrame, home_id: Any, away_id: Any) -> dict[str, float]:
    """Win/draw/loss chances and expected points if the match's shots were replayed."""
    shots = _shots(events)
    xg = pd.to_numeric(shots.get("xG"), errors="coerce").fillna(0.0)
    h = _goal_distribution(xg[shots["team_id"] == home_id].to_numpy())
    a = _goal_distribution(xg[shots["team_id"] == away_id].to_numpy())
    joint = np.outer(h, a)
    home_win = float(np.tril(joint, -1).sum())
    away_win = float(np.triu(joint, 1).sum())
    draw = float(np.trace(joint))
    return {
        "home_win": round(home_win, 3), "draw": round(draw, 3), "away_win": round(away_win, 3),
        "home_xpts": round(3 * home_win + draw, 2), "away_xpts": round(3 * away_win + draw, 2),
    }


# ── Where the chances came from ────────────────────────────────────────
SITUATIONS = ("Open play", "Counter", "Corner", "Free kick", "Throw-in", "Penalty")


def xg_by_situation(events: pd.DataFrame, team_id: Any) -> dict[str, dict[str, float]]:
    """Shots and xG by the situation WhoScored attaches to each attempt."""
    shots = _shots(events)
    shots = shots[shots["team_id"] == team_id]
    q = _q(shots)
    pen = _truthy(shots.get("is_penalty"), shots.index) | q.str.contains("Penalty", regex=False)
    label = np.select(
        [pen, q.str.contains("FastBreak", regex=False), q.str.contains("FromCorner", regex=False),
         q.str.contains("ThrowinSetPiece", regex=False),
         q.str.contains("DirectFreekick", regex=False) | q.str.contains("SetPiece", regex=False)],
        ["Penalty", "Counter", "Corner", "Throw-in", "Free kick"], default="Open play")
    xg = pd.to_numeric(shots.get("xG"), errors="coerce").fillna(0.0)
    goals = _truthy(shots.get("is_goal"), shots.index)
    out = {}
    for name in SITUATIONS:
        m = label == name
        out[name] = {"shots": int(m.sum()), "xG": round(float(xg[m].sum()), 2), "goals": int(goals[m].sum())}
    return out


# ── Pressing through the match ─────────────────────────────────────────
_DEF_ACTIONS = {"Tackle", "Interception", "Challenge", "Foul", "BlockedPass"}


def ppda_timeline(events: pd.DataFrame, team_id: Any, opponent_id: Any, step: int = 15) -> list[dict]:
    """PPDA per ``step`` minutes, by the same definition as the match figure.

    Each window is counted with ``calculate_ppda`` -- the function the article
    and the match facts use -- so the bins add up to the full-match PPDA the
    reader sees elsewhere. A separate definition here (other action types,
    other zone edges) put 6.9 on the poster beside 5.0 in the article.
    """
    from football_analysis.reports.match_report import calculate_ppda

    minute = pd.to_numeric(events["minute"], errors="coerce").fillna(0)
    end = int(max(90, minute.max()))
    starts = list(range(0, end, step))
    # Stoppage time shorter than half a bin is folded into the last full bin:
    # four minutes on their own produced a PPDA of 2.0 from two passes.
    if len(starts) > 1 and end - starts[-1] < step / 2:
        starts.pop()
    rows = []
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else end + 1
        window = (minute >= start) & (minute < stop)
        counted = calculate_ppda(events[window], team_id, opponent_id)
        passes, actions = int(counted["passes_allowed"]), int(counted["defensive_actions"])
        rows.append({"start": start, "end": min(stop, end), "passes": passes,
                     "actions": actions, "ppda": round(passes / actions, 1) if actions else None})
    return rows


# ── Control ────────────────────────────────────────────────────────────
def game_control(events: pd.DataFrame, home_id: Any, away_id: Any) -> dict[str, float]:
    """Share of match minutes each side was the likelier winner, from the win-probability curve."""
    from football_analysis.metrics.match_metrics import win_probability

    curve = win_probability(events, home_id, away_id)
    if curve.empty:
        return {"home": 0.0, "away": 0.0, "level": 100.0}
    home = curve["home_win"] > curve["away_win"] + 0.02
    away = curve["away_win"] > curve["home_win"] + 0.02
    n = len(curve)
    return {"home": round(100 * home.sum() / n, 1), "away": round(100 * away.sum() / n, 1),
            "level": round(100 * (n - home.sum() - away.sum()) / n, 1)}


# ── Dead balls ─────────────────────────────────────────────────────────
def set_piece_summary(events: pd.DataFrame, team_id: Any) -> dict[str, float]:
    """Corners taken, how many brought a shot within ten seconds, and set-piece xG."""
    own = events[events["team_id"] == team_id]
    corners = own[_has(own, "CornerTaken")]
    shots = _shots(events)
    shots_t = _clock(shots)
    shot_team = shots["team_id"]
    corner_shot = 0
    for t in _clock(corners):
        corner_shot += int(((shot_team == team_id) & (shots_t >= t) & (shots_t <= t + 10)).any())
    situations = xg_by_situation(events, team_id)
    dead = ("Corner", "Free kick", "Throw-in")
    return {
        "corners": int(len(corners)),
        "corners_to_shot": corner_shot,
        "set_piece_shots": sum(situations[s]["shots"] for s in dead),
        "set_piece_xg": round(sum(situations[s]["xG"] for s in dead), 2),
        "set_piece_goals": sum(situations[s]["goals"] for s in dead),
        "long_throws": int((_has(own, "ThrowIn") & (pd.to_numeric(own.get("pass_length"), errors="coerce") >= 25)
                            & (pd.to_numeric(own.get("end_x"), errors="coerce") >= 83)).sum()),
    }


# ── Chance creation types ──────────────────────────────────────────────
def cutbacks(events: pd.DataFrame, team_id: Any) -> pd.DataFrame:
    """Low balls pulled back from near the byline into the area around the penalty spot."""
    own = events[(events["team_id"] == team_id) & _truthy(events.get("is_pass"), events.index)]
    x, y = pd.to_numeric(own["x"], errors="coerce"), pd.to_numeric(own["y"], errors="coerce")
    ex, ey = pd.to_numeric(own["end_x"], errors="coerce"), pd.to_numeric(own["end_y"], errors="coerce")
    high = _has(own, "Chipped") | _has(own, "HeadPass")
    mask = (x >= 88) & ((y <= 37) | (y >= 63)) & ex.between(78, 94) & ey.between(33, 67) & (ex < x) & ~high
    cols = ["minute", "player", "x", "y", "end_x", "end_y", "outcome"]
    return own.loc[mask, [c for c in cols if c in own.columns]]


def through_balls(events: pd.DataFrame, team_id: Any) -> pd.DataFrame:
    """Through balls per player: attempted and completed."""
    own = events[(events["team_id"] == team_id) & _has(events, "Throughball")]
    if own.empty:
        return pd.DataFrame(columns=["player", "attempted", "completed"])
    own = own.assign(_ok=_ok(own).astype(int))
    return (own.groupby("player").agg(attempted=("_ok", "size"), completed=("_ok", "sum"))
            .reset_index().sort_values(["completed", "attempted"], ascending=False))


# ── Goalkeepers ────────────────────────────────────────────────────────
def keeper_actions(events: pd.DataFrame, team_id: Any) -> dict[str, int]:
    own = events[events["team_id"] == team_id]
    saves = own[own["type"].eq("Save")]
    return {
        "sweeper_actions": int(own["type"].eq("KeeperSweeper").sum()),
        "claims": int(own["type"].eq("Claim").sum()),
        "punches": int(own["type"].eq("Punch").sum()),
        "diving_saves": int(_has(saves, "DivingSave").sum()),
        "saves_in_box": int(_has(saves, "KeeperSaveInTheBox").sum()),
    }


# ── Shooting ───────────────────────────────────────────────────────────
def shot_profile(events: pd.DataFrame, team_id: Any) -> dict[str, float]:
    shots = _shots(events)
    shots = shots[shots["team_id"] == team_id]
    xg = pd.to_numeric(shots.get("xG"), errors="coerce").fillna(0.0)
    goal = _truthy(shots.get("is_goal"), shots.index)
    head = _truthy(shots.get("is_header"), shots.index) | _has(shots, "Head")
    first = _has(shots, "FirstTouch")
    big = _truthy(shots.get("big_chance"), shots.index) | _has(shots, "BigChance")
    return {
        "headers": int(head.sum()), "header_xg": round(float(xg[head].sum()), 2),
        "first_time": int(first.sum()), "first_time_xg": round(float(xg[first].sum()), 2),
        "big_chances": int(big.sum()), "big_chances_scored": int((big & goal).sum()),
        "big_chances_missed": int((big & ~goal).sum()),
    }


# ── Discipline and the line ────────────────────────────────────────────
def tactical_fouls(events: pd.DataFrame, team_id: Any, window: float = 6.0) -> int:
    """Fouls committed within ``window`` s of the side's own failed action in the opponent half."""
    ordered = events.sort_values(["minute", "second"], kind="stable")
    t = _clock(ordered)
    count = 0
    last_loss = -1e9
    for (_, row), clock in zip(ordered.iterrows(), t):
        if row["team_id"] != team_id:
            continue
        if row["type"] == "Foul" and str(row.get("outcome", "")).lower() != "successful":
            if clock - last_loss <= window:
                count += 1
        elif str(row.get("outcome", "")).lower() == "unsuccessful" and float(row.get("x") or 0) >= 50:
            last_loss = clock
    return count


def offsides(events: pd.DataFrame, team_id: Any) -> dict[str, int]:
    own = events[events["team_id"] == team_id]
    # OffsideProvoked is logged on the defending side that sprang the trap;
    # OffsidePass on the side whose pass found a player offside.
    return {"trap_sprung": int(own["type"].eq("OffsideProvoked").sum()),
            "caught": int(own["type"].eq("OffsidePass").sum())}


# ── One fixture ────────────────────────────────────────────────────────
def match_extras(events: pd.DataFrame, info: dict[str, Any]) -> dict[str, Any]:
    home, away = info.get("home_id"), info.get("away_id")
    sides = {"home": (home, away), "away": (away, home)}
    out: dict[str, Any] = {"expected_points": expected_points(events, home, away),
                           "game_control": game_control(events, home, away)}
    for side, (tid, oid) in sides.items():
        out[side] = {
            "xg_by_situation": xg_by_situation(events, tid),
            "ppda_timeline": ppda_timeline(events, tid, oid),
            "set_pieces": set_piece_summary(events, tid),
            "cutbacks": int(len(cutbacks(events, tid))),
            "cutbacks_completed": int(_ok(cutbacks(events, tid)).sum()) if len(cutbacks(events, tid)) else 0,
            "through_balls": through_balls(events, tid).to_dict("records"),
            "keeper": keeper_actions(events, tid),
            "shots": shot_profile(events, tid),
            "tactical_fouls": tactical_fouls(events, tid),
            "offsides": offsides(events, tid),
        }
    return out
