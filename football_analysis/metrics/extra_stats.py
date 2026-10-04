"""Counting statistics for one match that the advanced-metric tables do not carry.

The per-match team and player tables hold the model-based figures (xG, xT, line
breaks, progression). Adding a block of rounds up also needs the plain counts that
sit under them -- passes, key passes, assists, tackles, cards, corners, duels, the
two halves of PPDA -- and they are read here straight off the event stream, with
the same helpers the match report uses so a number agrees with the page that
prints it.

Each function returns raw counts only. Ratios are worked out after adding up
(see ``aggregate.py``), because a pass-completion rate is not the mean of five rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_analysis.metrics.match_metrics import (
    _outcome_is_successful,
    cross_mask,
    defensive_blocks_count,
    fouls_committed_mask,
    live_event_mask,
    set_piece_breakdown,
)
from football_analysis.reports.match_report import calculate_ppda

LONG_BALL_METRES = 32.0
BOX_X, BOX_Y_LOW, BOX_Y_HIGH = 83.0, 21.1, 78.9
FINAL_THIRD_X = 66.7


def _flag(events: pd.DataFrame, column: str) -> pd.Series:
    if column not in events:
        return pd.Series(False, index=events.index)
    return events[column].astype(str).str.lower().isin(["true", "1", "1.0"])


def _pass_features(events: pd.DataFrame) -> pd.DataFrame:
    """Per-event booleans for passes, on the live events only."""
    live = live_event_mask(events)
    is_pass = _flag(events, "is_pass") & live
    done = events.get("outcome", pd.Series("", index=events.index)).map(_outcome_is_successful)
    x = pd.to_numeric(events.get("x"), errors="coerce")
    end_x = pd.to_numeric(events.get("end_x"), errors="coerce")
    end_y = pd.to_numeric(events.get("end_y"), errors="coerce")
    length = pd.to_numeric(events.get("pass_length"), errors="coerce")
    cross = cross_mask(events)
    return pd.DataFrame(
        {
            "pass": is_pass,
            "completed": is_pass & done,
            # Counted on any row the feed flags, as the statistics page does.
            "key": _flag(events, "is_key_pass"),
            "long": is_pass & (length >= LONG_BALL_METRES),
            "long_done": is_pass & done & (length >= LONG_BALL_METRES),
            "cross": cross & is_pass,
            "cross_done": cross & is_pass & done,
            "final_third": is_pass & done & (end_x >= FINAL_THIRD_X) & (x < FINAL_THIRD_X),
            "into_box": is_pass
            & done
            & (end_x >= BOX_X)
            & end_y.between(BOX_Y_LOW, BOX_Y_HIGH)
            & ~(
                (x >= BOX_X)
                & pd.to_numeric(events.get("y"), errors="coerce").between(BOX_Y_LOW, BOX_Y_HIGH)
            ),
        },
        index=events.index,
    )


def _card_kinds(events: pd.DataFrame) -> pd.DataFrame:
    """Yellow and red cards, a second yellow counting as a yellow and a red."""
    is_card = events.get("type", pd.Series("", index=events.index)).astype(str).eq("Card")
    tokens = (
        events.get("qualifier_names", pd.Series("", index=events.index)).astype(str).str.casefold()
    )
    second = tokens.str.contains("secondyellow") | tokens.str.contains("second yellow")
    red = tokens.str.contains("'red'") | tokens.str.contains('"red"') | second
    yellow = tokens.str.contains("yellow")
    return pd.DataFrame({"yellow": is_card & yellow, "red": is_card & red}, index=events.index)


def _fouls_won_mask(events: pd.DataFrame, committed: pd.Series) -> pd.Series:
    """Foul rows that belong to the player who was fouled.

    The feed writes two rows per foul when it is paired: the offender's and the
    player fouled. Where it writes only one, nobody can be said to have won it.
    """
    foul_rows = events.get("type", pd.Series("", index=events.index)).astype(str).eq("Foul")
    foul_rows &= live_event_mask(events)
    paired = bool((foul_rows & ~committed).any())
    return foul_rows & ~committed if paired else pd.Series(False, index=events.index)


def team_extra(events: pd.DataFrame, info: dict) -> pd.DataFrame:
    """One row per side with the plain counts and the pieces of PPDA."""
    features = _pass_features(events)
    cards = _card_kinds(events)
    types = events.get("type", pd.Series("", index=events.index)).astype(str)
    fouls = fouls_committed_mask(events)
    sides = (
        (info["home_name"], info["home_id"], info["away_id"]),
        (info["away_name"], info["away_id"], info["home_id"]),
    )
    rows = []
    for name, team_id, opponent_id in sides:
        mine = events["team_id"].eq(team_id)
        theirs = events["team_id"].eq(opponent_id)
        won_row = (
            events.get("outcome", pd.Series("", index=events.index)).astype(str).eq("Successful")
        )
        # The same definitions as the statistics page. An aerial is logged for both
        # players, the winner Successful. A ground duel is a take-on: the dribbler's
        # side wins a Successful one and the defending side wins an Unsuccessful one,
        # so both sides share one contested total.
        aerial_rows = types.eq("Aerial") & mine
        own_take_ons = types.eq("TakeOn") & mine
        their_take_ons = types.eq("TakeOn") & theirs
        ground_won = int((own_take_ons & won_row).sum() + (their_take_ons & ~won_row).sum())
        ppda = calculate_ppda(events, team_id, opponent_id)
        row = {
            "team": name,
            "passes": int((features["pass"] & mine).sum()),
            "passes_completed": int((features["completed"] & mine).sum()),
            "key_passes": int((features["key"] & mine).sum()),
            "long_balls": int((features["long"] & mine).sum()),
            "long_balls_completed": int((features["long_done"] & mine).sum()),
            "passes_into_final_third": int((features["final_third"] & mine).sum()),
            "passes_into_box": int((features["into_box"] & mine).sum()),
            "tackles": int((types.eq("Tackle") & mine).sum()),
            "interceptions": int((types.eq("Interception") & mine).sum()),
            "clearances": int((types.eq("Clearance") & mine).sum()),
            "blocks": int(defensive_blocks_count(events, team_id, opponent_id)),
            "ball_recoveries": int((types.eq("BallRecovery") & mine).sum()),
            "fouls_committed": int((fouls & mine).sum()),
            "fouls_won": int((fouls & events["team_id"].eq(opponent_id)).sum()),
            "yellow_cards": int((cards["yellow"] & mine).sum()),
            "red_cards": int((cards["red"] & mine).sum()),
            "corners": int((types.eq("CornerAwarded") & mine).sum()),
            "aerials": int(aerial_rows.sum()),
            "aerials_won": int((aerial_rows & won_row).sum()),
            "ground_duels": int(own_take_ons.sum() + their_take_ons.sum()),
            "ground_duels_won": ground_won,
            "ppda_passes_allowed": int(ppda["passes_allowed"]),
            "ppda_defensive_actions": int(ppda["defensive_actions"]),
        }
        breakdown = set_piece_breakdown(events, team_id)
        for key, label in (
            ("open_play", "open_play"),
            ("corner", "corner"),
            ("free_kick", "free_kick"),
            ("throw_in", "throw_in"),
            ("penalty", "penalty"),
        ):
            row[f"{label}_shots"] = int(breakdown[key]["shots"])
            row[f"{label}_xG"] = float(breakdown[key]["xG"])
            row[f"{label}_goals"] = int(breakdown[key]["goals"])
        rows.append(row)
    return pd.DataFrame(rows)


def player_extra(events: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, team) with the counts the per-match player frame lacks."""
    features = _pass_features(events)
    cards = _card_kinds(events)
    live = live_event_mask(events)
    types = events.get("type", pd.Series("", index=events.index)).astype(str)
    fouls = fouls_committed_mask(events)
    shot = _flag(events, "is_shot") & live & ~_flag(events, "is_penalty_shootout")
    shot_type = events.get("shot_whoscored_type", pd.Series("", index=events.index)).astype(str)
    on_target = shot & shot_type.isin(["Goal", "SavedShot"])
    own_goal = _flag(events, "is_own_goal")

    frame = pd.DataFrame(
        {
            "player": events["player"],
            "team_id": events["team_id"],
            "key_passes": features["key"],
            "crosses": features["cross"],
            "crosses_completed": features["cross_done"],
            "long_balls": features["long"],
            "long_balls_completed": features["long_done"],
            "passes_into_final_third": features["final_third"],
            "passes_into_box": features["into_box"],
            "tackles": types.eq("Tackle") & live,
            "shots_on_target": on_target & ~own_goal,
            "fouls_committed": fouls,
            "fouls_won": _fouls_won_mask(events, fouls),
            "yellow_cards": cards["yellow"],
            "red_cards": cards["red"],
        }
    ).dropna(subset=["player"])
    grouped = frame.groupby(["player", "team_id"], sort=False).sum().reset_index()

    # Assists are written on the goal row, naming the passer.
    goal = _flag(events, "is_goal") & ~own_goal
    assists = (
        events[goal & events["assist_player"].notna()]
        .groupby(["assist_player", "team_id"])
        .size()
        .rename("assists")
        .reset_index()
        .rename(columns={"assist_player": "player"})
    )
    grouped = grouped.merge(assists, on=["player", "team_id"], how="outer")
    for column in grouped.columns:
        if column not in {"player", "team_id"}:
            grouped[column] = pd.to_numeric(grouped[column], errors="coerce").fillna(0).astype(int)
    return grouped


def starts_by_player(players: pd.DataFrame) -> pd.DataFrame:
    """Whether each player was in the starting eleven, from players.csv."""
    if players is None or players.empty:
        return pd.DataFrame(columns=["player", "team_id", "started"])
    out = pd.DataFrame(
        {
            "player": players["name"],
            "team_id": players["team_id"],
            "started": players.get("is_first_xi", pd.Series(False, index=players.index))
            .astype(str)
            .str.lower()
            .isin(["true", "1", "1.0"])
            .astype(int),
        }
    )
    return out.replace([np.inf, -np.inf], np.nan)
