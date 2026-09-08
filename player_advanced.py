"""Per-player measures the raw event counts cannot give on their own.

Three faults in ranking players by what the feed records directly:

**Volume is not intensity.** A defensive action can only happen while the
opponent has the ball, so a side pinned in its own half records more of them
for being worse. In Chelsea 4-3 Brighton the counts say Chelsea defended more
(125 actions to 103) and the rate says the opposite by a distance: 16.4 actions
per 100 opponent touches against 33.6. Ranking on the count rewards a player
for his team not having the ball.

**Minutes are not equal.** A substitute with twenty-five minutes is compared
with a starter who played ninety-eight, and every count is a count of a longer
match.

**Positions are not comparable.** A centre-back ranked on expected goals is
being measured against a job he was not doing. Comparing him with the other
defenders on the pitch asks whether he did *his* job well.

Everything here is computed from columns already in the event frame. Three of
them — Aerial duels, Dispossessed and Challenge — were in the data and read by
nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Actions that can only occur while the other side has the ball. They are the
# numerator of the possession adjustment; opponent touches are the denominator.
_DEFENSIVE_TYPES = ("Tackle", "Interception", "BallRecovery", "Clearance",
                    "Challenge", "BlockedPass")

# A ball into the box counts as final third for reception purposes.
_FINAL_THIRD_X = 66.7


def _flag(frame, column):
    if column not in frame:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "1.0"})


def _numeric(frame, column):
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def possession_adjusted(events, players_frame):
    """Defensive actions per 100 opponent touches, per player.

    The standard correction for the fact that a defensive action requires the
    opponent to have the ball. Without it the metric measures how little of the
    ball a player's team had.
    """
    from match_metrics import touch_mask

    touches = events[touch_mask(events)]
    by_team = touches.groupby("team_id").size().to_dict()
    total = sum(by_team.values())

    out = {}
    for name, rows in events.dropna(subset=["player"]).groupby("player"):
        team = rows["team_id"].mode()
        if team.empty:
            continue
        team_id = team.iloc[0]
        opponent_touches = total - by_team.get(team_id, 0)
        actions = int(rows["type"].isin(_DEFENSIVE_TYPES).sum())
        out[str(name)] = {
            "defensive_actions": actions,
            "padj_defensive_actions": (
                100.0 * actions / opponent_touches if opponent_touches else 0.0),
        }
    return out


def progression_distance(events):
    """Metres gained toward the opponent goal, per player.

    Ten short passes and one forty-metre switch both count as progression by
    the usual definition; only one of them moved the team up the pitch. This
    sums the actual advance of successful passes and carries, so distance is
    what it is measured in.
    """
    moves = events[(events["type"].isin(["Pass", "Carry"]))
                   & (events.get("outcome", "").astype(str).str.lower() == "successful")]
    if moves.empty:
        return {}
    start = _numeric(moves, "x")
    end = _numeric(moves, "end_x")
    # Only forward movement is progression; a pass backwards is not negative
    # progression, it is a different action, so it contributes nothing.
    gained = (end - start).clip(lower=0)
    frame = moves.assign(_gained=gained * 1.05)  # x is 0-100, the pitch is 105 m
    return {
        str(name): {"progression_metres": float(rows["_gained"].sum())}
        for name, rows in frame.dropna(subset=["player"]).groupby("player")
    }


def duels_and_losses(events):
    """Aerials, dispossessions and challenges — three fields nothing read.

    An aerial duel is in the feed 62 times a match and appeared in no radar,
    no profile and no ranking. Dispossessed is how often a player lost the ball
    while carrying it, which is the cost side of the progression numbers that
    were already being counted.
    """
    out = {}
    for name, rows in events.dropna(subset=["player"]).groupby("player"):
        kinds = rows["type"].astype(str)
        outcome = rows.get("outcome", pd.Series("", index=rows.index)).astype(str).str.lower()
        aerials = kinds.eq("Aerial")
        aerials_won = int((aerials & outcome.eq("successful")).sum())
        out[str(name)] = {
            "aerials": int(aerials.sum()),
            "aerials_won": aerials_won,
            "aerial_pct": 100.0 * aerials_won / max(int(aerials.sum()), 1),
            "dispossessed": int(kinds.eq("Dispossessed").sum()),
            "dribbled_past": int(kinds.eq("Challenge").sum()),
        }
    return out


def defensive_height(events):
    """The average x of a player's defensive actions.

    Two defenders with identical tackle counts are doing different jobs if one
    of them wins the ball on halfway and the other on his own eighteen-yard
    line. Nothing in the package distinguished them.
    """
    defending = events[events["type"].isin(_DEFENSIVE_TYPES)].dropna(subset=["player", "x"])
    if defending.empty:
        return {}
    return {
        str(name): {"defensive_height": float(_numeric(rows, "x").mean())}
        for name, rows in defending.groupby("player")
        if len(rows) >= 3  # below three actions the mean is one tackle's location
    }


def line_breaking(events):
    """Passes that travel forward far enough to take a line of pressure out.

    Uses the recorded pass angle, so "forward" is the feed's own measure and
    not a guess from coordinates. Twenty metres is roughly the gap between two
    banks of players.
    """
    passes = events[_flag(events, "is_pass")].copy()
    if passes.empty:
        return {}
    angle = _numeric(passes, "pass_angle")
    length = _numeric(passes, "pass_length")
    complete = passes.get("outcome", pd.Series("", index=passes.index)).astype(str).str.lower().eq("successful")
    # Angle is radians with 0 as straight ahead, so forward is the half circle
    # around zero.
    forward = (angle < np.pi / 2) | (angle > 3 * np.pi / 2)
    breaking = forward & (length >= 20) & complete
    passes = passes.assign(_breaking=breaking, _complete=complete)
    out = {}
    for name, rows in passes.dropna(subset=["player"]).groupby("player"):
        out[str(name)] = {
            "line_breaking_passes": int(rows["_breaking"].sum()),
        }
    return out


def final_third_receptions(events):
    """How often a player received the ball in the final third.

    Where a player gets on the ball is a different question from what he does
    with it, and only the second was being measured. Inferred from the pass
    that reached him, so it carries the same caveat as every reception figure
    in this package: the feed names the passer, not the receiver.
    """
    passes = events[_flag(events, "is_pass")].copy()
    complete = passes.get("outcome", pd.Series("", index=passes.index)).astype(str).str.lower().eq("successful")
    landed = _numeric(passes, "end_x") >= _FINAL_THIRD_X
    arrivals = passes[complete & landed]
    if arrivals.empty:
        return {}
    ordered = events.sort_values(["period_code", "minute", "second"], kind="stable")
    positions = {index: order for order, index in enumerate(ordered.index)}
    receivers = {}
    order_list = list(ordered.index)
    for index in arrivals.index:
        seat = positions.get(index)
        if seat is None or seat + 1 >= len(order_list):
            continue
        following = ordered.loc[order_list[seat + 1]]
        if following.get("team_id") != arrivals.loc[index, "team_id"]:
            continue
        name = following.get("player")
        if isinstance(name, str) and name.strip():
            receivers[name] = receivers.get(name, 0) + 1
    return {name: {"final_third_receptions": count} for name, count in receivers.items()}


def enrich(observations, events, players_frame=None):
    """Add every advanced measure above to a player-observation frame."""
    if observations is None or observations.empty:
        return observations
    frame = observations.copy()
    sources = [
        possession_adjusted(events, players_frame),
        progression_distance(events),
        duels_and_losses(events),
        defensive_height(events),
        line_breaking(events),
        final_third_receptions(events),
    ]
    combined = {}
    for source in sources:
        for name, values in source.items():
            combined.setdefault(name, {}).update(values)

    # A player with no reading is not a player with a reading of zero. That is
    # only true of counts: nought aerials won is a fact, whereas an average
    # defensive height of nought would place him on his own goal line, and a
    # player under the three-action floor has no height at all.
    absent = {"defensive_height": np.nan}

    columns = sorted({key for values in combined.values() for key in values})
    for column in columns:
        default = absent.get(column, 0.0)
        frame[column] = [
            combined.get(str(name), {}).get(column, default)
            for name in frame["player"]
        ]

    # Per-90 versions of the volume measures, so a substitute is compared on
    # rate rather than on having played less of the match.
    minutes = pd.to_numeric(frame.get("minutes"), errors="coerce").fillna(0)
    safe = minutes.where(minutes >= 1, np.nan)
    for column in ("progression_metres", "line_breaking_passes",
                   "final_third_receptions", "defensive_actions"):
        if column in frame:
            frame[column + "_p90"] = (frame[column] * 90 / safe).fillna(0.0)
    return frame


# ---------------------------------------------------------------------------
# comparison pools
# ---------------------------------------------------------------------------

_LINES = {
    "GK": "Goalkeeper",
    "D": "Defence",
    "M": "Midfield",
    "F": "Attack",
}


def line_of(role):
    """Which broad line a role belongs to, for a like-for-like comparison.

    A centre-back ranked on expected goals is being measured against a job he
    was not doing. WhoScored's role codes lead with the line -- DMR, MC, AMC,
    FW -- so the first letter carries it, with the attacking-midfield codes
    pulled forward because their job is chance creation, not screening.
    """
    code = str(role or "").upper().strip()
    if not code or code in {"SUB", "UNKNOWN", "NAN", "NONE"}:
        return "Unknown"
    if code in {"GK", "GOALKEEPER"}:
        return "Goalkeeper"
    if code.startswith("AM") or code in {"FW", "FWL", "FWR", "ST", "SS", "CF"}:
        return "Attack"
    if code.startswith("D"):
        return "Defence"
    if code.startswith("M") or code.startswith("DM"):
        return "Midfield"
    if code.startswith("F"):
        return "Attack"
    return "Unknown"
