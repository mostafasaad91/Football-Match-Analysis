"""Canonical event, possession, regain, transition, and team metric logic.

The project historically calculated the same metric in several renderers with
slightly different thresholds.  This module is the single source of truth for
metrics that can be derived from the WhoScored/Opta-style event stream.

The possession model is intentionally provider-agnostic and deterministic.  It
is an event-data approximation, not tracking-data ground truth.  Raw provider
``BallRecovery`` counts are preserved separately from inferred possession
regains so the two concepts are never presented as the same statistic.
"""

from __future__ import annotations

import ast
import math
import weakref
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

PERIOD_ORDER = {
    "pre": -1,
    "1h": 1,
    "firsthalf": 1,
    "2h": 2,
    "secondhalf": 2,
    "et1": 3,
    "firstperiodofextratime": 3,
    "et2": 4,
    "secondperiodofextratime": 4,
    "pso": 5,
    "penaltyshootout": 5,
    "postgame": 6,
}

NON_LIVE_PERIODS = {"pre", "pso", "penaltyshootout", "postgame"}
MARKER_TYPES = {
    "Start",
    "End",
    "FormationSet",
    "FormationChange",
    "SubstitutionOn",
    "SubstitutionOff",
    "Card",
}
SHOT_TYPES = {"Goal", "SavedShot", "MissedShots", "BlockedShot", "ShotOnPost"}
REGAIN_TYPES = {"BallRecovery", "Interception", "Tackle"}
KEEPER_CONTROL_TYPES = {"KeeperPickup", "KeeperSweeper", "Save", "Claim", "Smother"}
TOUCH_TYPES = {
    "Pass",
    "OffsidePass",
    "BallTouch",
    "TakeOn",
    "Carry",
    "Goal",
    "SavedShot",
    "MissedShots",
    "BlockedShot",
    "ShotOnPost",
}
RESTART_QUALIFIERS = {
    "cornertaken",
    "freekicktaken",
    "goalkick",
    "throwin",
    "penalty",
    "kickoff",
}
DEAD_BALL_TYPES = {
    "CornerAwarded",
    "Foul",
    "OffsideGiven",
    "OffsidePass",
    "OffsideProvoked",
}

TRANSITION_WINDOW_SECONDS = 12.0
TRANSITION_MIN_PROGRESS = 20.0
DANGEROUS_COUNTER_MIN_PROGRESS = 40.0
HIGH_REGAIN_X = 60.0
FINAL_THIRD_X = 66.7
DEEP_COMPLETION_X = 80.0
BOX_X = 83.0
BOX_Y_MIN = 21.0
BOX_Y_MAX = 79.0

_POSSESSION_CACHE: dict[
    int,
    tuple[
        weakref.ReferenceType[pd.DataFrame],
        tuple[Any, ...],
        pd.DataFrame,
        pd.DataFrame,
    ],
] = {}
_TEAM_METRIC_CACHE: dict[
    int,
    tuple[
        weakref.ReferenceType[pd.DataFrame],
        tuple[Any, ...],
        dict[str, dict[str, Any]],
    ],
] = {}
_PLAYER_SEQUENCE_CACHE: dict[
    int,
    tuple[
        weakref.ReferenceType[pd.DataFrame],
        tuple[Any, ...],
        dict[str, dict[str, float]],
    ],
] = {}


def _cache_reference(events: pd.DataFrame, cache: dict) -> weakref.ReferenceType:
    identity = id(events)

    def _discard(_reference) -> None:
        cache.pop(identity, None)

    return weakref.ref(events, _discard)


def _normalise_period(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "")


def _qualifier_tokens(value: Any) -> set[str]:
    """Return normalized qualifier tokens from lists or CSV string values."""
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("[") or raw.startswith("(") or raw.startswith("{"):
            try:
                value = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                value = [raw]
        else:
            value = [raw]
    if isinstance(value, dict):
        value = value.values()
    if not isinstance(value, Iterable) or isinstance(value, (bytes, bytearray)):
        value = [value]
    return {
        str(item).strip().lower().replace(" ", "")
        for item in value
        if item is not None and str(item).strip()
    }


def _bool_series(events: pd.DataFrame, name: str) -> pd.Series:
    if name not in events.columns:
        return pd.Series(False, index=events.index, dtype=bool)
    values = events[name]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _numeric_series(events: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in events.columns:
        return pd.Series(default, index=events.index, dtype=float)
    return pd.to_numeric(events[name], errors="coerce").fillna(default).astype(float)


def _outcome_is_successful(value: Any) -> bool:
    return str(value or "").strip().lower() in {"successful", "success", "true", "1"}


def _is_true(value: Any) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def is_restart_event(row: pd.Series | dict[str, Any]) -> bool:
    qualifiers = _qualifier_tokens(row.get("qualifier_names", []))
    return bool(qualifiers & RESTART_QUALIFIERS)


def live_event_mask(events: pd.DataFrame) -> pd.Series:
    periods = events.get("period_code", pd.Series("", index=events.index)).map(
        _normalise_period
    )
    types = events.get("type", pd.Series("", index=events.index)).fillna("").astype(str)
    shootout = _bool_series(events, "is_penalty_shootout")
    return ~periods.isin(NON_LIVE_PERIODS) & ~types.isin(MARKER_TYPES) & ~shootout


def fouls_committed_mask(events: pd.DataFrame) -> pd.Series:
    """Return provider foul rows attributed to the team committing the foul.

    WhoScored-style feeds can emit two ``Foul`` rows per incident: an
    ``Unsuccessful`` row for the offender and a ``Successful`` row for the
    player who won the foul. When both outcomes are present, only the former is
    counted. Single-sided feeds retain all foul rows as a safe fallback.
    """
    types = (
        events.get("type", pd.Series("", index=events.index))
        .fillna("")
        .astype(str)
        .str.casefold()
    )
    foul_rows = types.eq("foul") & live_event_mask(events)
    outcomes = (
        events.get("outcome", pd.Series("", index=events.index))
        .fillna("")
        .astype(str)
        .str.casefold()
    )
    foul_outcomes = outcomes[foul_rows]
    paired_feed = foul_outcomes.eq("unsuccessful").any() and foul_outcomes.eq(
        "successful"
    ).any()
    return foul_rows & outcomes.eq("unsuccessful") if paired_feed else foul_rows


def fouls_committed_count(events: pd.DataFrame, team_id: Any) -> int:
    """Count committed fouls for one team using the canonical provider rule."""
    team_ids = events.get("team_id", pd.Series(np.nan, index=events.index))
    return int((fouls_committed_mask(events) & team_ids.eq(team_id)).sum())


def blocked_shot_mask(events: pd.DataFrame) -> pd.Series:
    """Return shot rows the provider classified as blocked.

    The normalized event ``type`` can be ``SavedShot`` even when WhoScored's
    original shot type is ``BlockedShot``.  Preserve the provider meaning by
    checking the canonical shot fields and the exact ``Blocked`` qualifier.
    ``BlockedX``/``BlockedY`` alone do not make a shot a block.
    """
    blocked = pd.Series(False, index=events.index, dtype=bool)
    for column in ("type", "shot_whoscored_type", "shot_category"):
        if column not in events.columns:
            continue
        values = (
            events[column]
            .fillna("")
            .astype(str)
            .str.casefold()
            .str.replace(r"[^a-z]", "", regex=True)
        )
        blocked |= values.isin({"blockedshot", "blocked"})
    if "qualifier_names" in events.columns:
        blocked |= events["qualifier_names"].map(
            lambda value: "blocked" in _qualifier_tokens(value)
        )

    types = events.get("type", pd.Series("", index=events.index)).fillna("").astype(str)
    is_shot = _bool_series(events, "is_shot") | types.isin(SHOT_TYPES)
    return blocked & is_shot & live_event_mask(events)


def defensive_block_events(
    events: pd.DataFrame,
    defending_team_id: Any,
    opponent_team_id: Any,
) -> pd.DataFrame:
    """Map the opponent's blocked shots onto the defending team's perspective."""
    team_ids = events.get("team_id", pd.Series(np.nan, index=events.index))
    blocks = events[blocked_shot_mask(events) & team_ids.eq(opponent_team_id)].copy()
    if blocks.empty:
        return blocks

    # WhoScored normalizes both teams to their own left-to-right attack. Rotate
    # the shooter's coordinates so a block appears in the defender's own half.
    for column in ("x", "y", "end_x", "end_y"):
        if column in blocks.columns:
            values = pd.to_numeric(blocks[column], errors="coerce")
            blocks[column] = 100.0 - values
    blocks["team_id"] = defending_team_id
    blocks["type"] = "BlockedShot"
    if "player" in blocks.columns:
        blocks["player"] = "Team block"
    return blocks


def defensive_blocks_count(
    events: pd.DataFrame,
    defending_team_id: Any,
    opponent_team_id: Any,
) -> int:
    """Count shots blocked by a defence, not shots blocked against its attack."""
    return int(len(defensive_block_events(events, defending_team_id, opponent_team_id)))


def cross_mask(events: pd.DataFrame, successful_only: bool = False) -> pd.Series:
    """Canonical cross mask: passes carrying the provider Cross qualifier."""
    mask = _bool_series(events, "is_cross")
    if "qualifier_names" in events.columns:
        mask |= events["qualifier_names"].map(
            lambda value: "cross" in _qualifier_tokens(value)
        )
    if "is_cross" not in events.columns and "qualifier_names" not in events.columns:
        x = _numeric_series(events, "x", np.nan)
        y = _numeric_series(events, "y", np.nan)
        end_x = _numeric_series(events, "end_x", np.nan)
        mask = (
            _bool_series(events, "is_pass")
            & (x >= 60)
            & ((y <= 22) | (y >= 78))
            & (end_x >= 80)
        )
    if successful_only:
        successful = events.get("outcome", pd.Series("", index=events.index)).map(
            _outcome_is_successful
        )
        mask &= successful
    return mask & live_event_mask(events)


def touch_mask(events: pd.DataFrame) -> pd.Series:
    """Return events that represent an intentional on-ball touch."""
    types = events.get("type", pd.Series("", index=events.index)).fillna("").astype(str)
    return types.isin(TOUCH_TYPES) & live_event_mask(events)


def progressive_pass_mask(events: pd.DataFrame) -> pd.Series:
    """Return successful open-play passes meeting one canonical threshold.

    Thresholds approximate the widely used distance-to-goal definition on a
    normalized 0-100 pitch:

    * own half to own half: at least 28.6 units (roughly 30 metres)
    * own half to opposition half: at least 14.3 units (roughly 15 metres)
    * opposition half to opposition half: at least 9.5 units (roughly 10 metres)
    """
    is_pass = _bool_series(events, "is_pass")
    successful = events.get("outcome", pd.Series("", index=events.index)).map(
        _outcome_is_successful
    )
    x = _numeric_series(events, "x", np.nan)
    end_x = _numeric_series(events, "end_x", np.nan)
    gain = end_x - x
    thresholds = np.select(
        [(x < 50) & (end_x < 50), (x < 50) & (end_x >= 50), x >= 50],
        [28.6, 14.3, 9.5],
        default=np.inf,
    )
    restart = events.apply(is_restart_event, axis=1)
    return (
        is_pass
        & successful
        & x.notna()
        & end_x.notna()
        & (gain >= thresholds)
        & ~restart
        & live_event_mask(events)
    )


def box_entry_mask(events: pd.DataFrame) -> pd.Series:
    x = _numeric_series(events, "x", np.nan)
    y = _numeric_series(events, "y", np.nan)
    end_x = _numeric_series(events, "end_x", np.nan)
    end_y = _numeric_series(events, "end_y", np.nan)
    action = _bool_series(events, "is_pass") | events.get(
        "type", pd.Series("", index=events.index)
    ).eq("Carry")
    successful = events.get("outcome", pd.Series("", index=events.index)).map(
        _outcome_is_successful
    )
    starts_in_box = (x >= BOX_X) & y.between(BOX_Y_MIN, BOX_Y_MAX)
    ends_in_box = (end_x >= BOX_X) & end_y.between(BOX_Y_MIN, BOX_Y_MAX)
    return action & successful & ~starts_in_box & ends_in_box & live_event_mask(events)


def final_third_entry_mask(events: pd.DataFrame) -> pd.Series:
    x = _numeric_series(events, "x", np.nan)
    end_x = _numeric_series(events, "end_x", np.nan)
    action = _bool_series(events, "is_pass") | events.get(
        "type", pd.Series("", index=events.index)
    ).eq("Carry")
    successful = events.get("outcome", pd.Series("", index=events.index)).map(
        _outcome_is_successful
    )
    return (
        action
        & successful
        & (x < FINAL_THIRD_X)
        & (end_x >= FINAL_THIRD_X)
        & live_event_mask(events)
    )


def deep_completion_mask(events: pd.DataFrame) -> pd.Series:
    x = _numeric_series(events, "x", np.nan)
    end_x = _numeric_series(events, "end_x", np.nan)
    end_y = _numeric_series(events, "end_y", np.nan)
    successful = events.get("outcome", pd.Series("", index=events.index)).map(
        _outcome_is_successful
    )
    return (
        _bool_series(events, "is_pass")
        & successful
        & (x < DEEP_COMPLETION_X)
        & (end_x >= DEEP_COMPLETION_X)
        & end_y.between(15, 85)
        & ~events.apply(is_restart_event, axis=1)
        & live_event_mask(events)
    )


def _control_team(row: pd.Series) -> Any:
    """Return the team demonstrating controlled possession on this event."""
    team = row.get("team_id")
    if pd.isna(team):
        return None
    event_type = str(row.get("type") or "")
    successful = _outcome_is_successful(row.get("outcome"))
    if event_type in SHOT_TYPES:
        return team
    if event_type in {"BallRecovery", "Interception"}:
        return team
    if event_type == "Tackle" and successful:
        return team
    if event_type in KEEPER_CONTROL_TYPES and successful:
        return team
    if event_type in {"Pass", "KeyPass", "Carry", "TakeOn", "BallTouch", "Aerial"}:
        return team if successful else None
    return None


def _event_xy(row: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(pd.Series([row.get("end_x")]), errors="coerce").iloc[0]
    y = pd.to_numeric(pd.Series([row.get("end_y")]), errors="coerce").iloc[0]
    if pd.isna(x):
        x = pd.to_numeric(pd.Series([row.get("x")]), errors="coerce").iloc[0]
    if pd.isna(y):
        y = pd.to_numeric(pd.Series([row.get("y")]), errors="coerce").iloc[0]
    return (
        float(x) if pd.notna(x) else math.nan,
        float(y) if pd.notna(y) else math.nan,
    )


def build_possessions(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate events with possession IDs and return a possession summary.

    Possessions are inferred only from explicit control actions. Defensive and
    administrative events do not flip possession on their own. A restart always
    creates a new possession and is excluded from transition classification.
    """
    if events is None or events.empty:
        return pd.DataFrame() if events is None else events.copy(), pd.DataFrame()

    cache_key = (
        len(events),
        tuple(events.index[:1]),
        tuple(events.index[-1:]),
        (
            str(events.get("event_id", pd.Series(dtype=object)).iloc[-1])
            if "event_id" in events.columns and len(events)
            else ""
        ),
    )
    cached = _POSSESSION_CACHE.get(id(events))
    if cached and cached[0]() is events and cached[1] == cache_key:
        return cached[2], cached[3]

    work = events.copy()
    work["_source_index"] = np.arange(len(work))
    period_codes = work.get("period_code", pd.Series("", index=work.index)).map(
        _normalise_period
    )
    work["_period_code"] = period_codes
    work["_period_order"] = period_codes.map(PERIOD_ORDER).fillna(99).astype(int)
    work["_minute"] = _numeric_series(work, "minute")
    work["_second"] = _numeric_series(work, "second").clip(0, 59.999)
    work["_clock_seconds"] = work["_minute"] * 60.0 + work["_second"]
    work["_is_shot"] = _bool_series(work, "is_shot")
    work["_is_goal"] = _bool_series(work, "is_goal")
    work["_is_penalty"] = _bool_series(work, "is_penalty")
    work["_is_pass"] = _bool_series(work, "is_pass")
    work["_successful"] = work.get("outcome", pd.Series("", index=work.index)).map(
        _outcome_is_successful
    )
    work["_x"] = _numeric_series(work, "x", np.nan)
    work["_y"] = _numeric_series(work, "y", np.nan)
    work["_end_x"] = _numeric_series(work, "end_x", np.nan)
    work["_end_y"] = _numeric_series(work, "end_y", np.nan)
    work["_xT"] = _numeric_series(work, "xT")
    work["_xG"] = _numeric_series(work, "xG")
    work["_is_restart"] = work.apply(is_restart_event, axis=1)
    work["_provider_fastbreak"] = work.get(
        "qualifier_names", pd.Series("", index=work.index)
    ).map(lambda value: bool(_qualifier_tokens(value) & {"fastbreak", "counterattack"}))
    starts_in_box = (work["_x"] >= BOX_X) & work["_y"].between(BOX_Y_MIN, BOX_Y_MAX)
    ends_in_box = (work["_end_x"] >= BOX_X) & work["_end_y"].between(
        BOX_Y_MIN, BOX_Y_MAX
    )
    controlled_move = (
        work["_is_pass"] | work.get("type", pd.Series("", index=work.index)).eq("Carry")
    ) & work["_successful"]
    work["_box_entry"] = controlled_move & ~starts_in_box & ends_in_box
    work["_final_third_entry"] = (
        controlled_move
        & (work["_x"] < FINAL_THIRD_X)
        & (work["_end_x"] >= FINAL_THIRD_X)
    )
    work = work.sort_values(
        ["_period_order", "_clock_seconds", "_source_index"], kind="stable"
    ).reset_index(drop=True)
    work["possession_id"] = pd.Series(pd.NA, index=work.index, dtype="Int64")
    work["possession_team"] = np.nan
    work["possession_start_reason"] = ""

    current_team = None
    current_period = None
    possession_id = 0
    last_control_team = None

    for idx, row in work.iterrows():
        period = row["_period_code"]
        event_type = str(row.get("type") or "")
        if period in NON_LIVE_PERIODS or _is_true(
            row.get("is_penalty_shootout", False)
        ):
            continue
        if period != current_period:
            current_period = period
            current_team = None
            last_control_team = None

        candidate = _control_team(row)
        restart = bool(row["_is_restart"])
        starts_new = False
        reason = ""
        if candidate is not None:
            if restart:
                starts_new = True
                reason = "restart"
            elif current_team is None:
                starts_new = True
                reason = (
                    "period_start" if last_control_team is None else "opponent_turnover"
                )
            elif candidate != current_team:
                starts_new = True
                reason = (
                    "recovery" if event_type in REGAIN_TYPES else "opponent_turnover"
                )

        if starts_new:
            possession_id += 1
            current_team = candidate
            work.at[idx, "possession_start_reason"] = reason
        if candidate is not None:
            current_team = candidate
            last_control_team = candidate

        if current_team is not None and event_type not in MARKER_TYPES:
            work.at[idx, "possession_id"] = possession_id
            work.at[idx, "possession_team"] = current_team

        if event_type == "Goal":
            current_team = None

    summaries: list[dict[str, Any]] = []
    assigned = work[work["possession_id"].notna()].copy()
    for possession_value, group in assigned.groupby("possession_id", sort=True):
        group = group.sort_values(["_clock_seconds", "_source_index"], kind="stable")
        team = group["possession_team"].dropna().iloc[0]
        team_events = group[group.get("team_id") == team].copy()
        if team_events.empty:
            continue
        first = team_events.iloc[0]
        last = team_events.iloc[-1]
        start_time = float(first["_clock_seconds"])
        end_time = float(last["_clock_seconds"])
        duration = max(end_time - start_time, 0.0)
        start_x = pd.to_numeric(pd.Series([first.get("x")]), errors="coerce").iloc[0]
        start_y = pd.to_numeric(pd.Series([first.get("y")]), errors="coerce").iloc[0]
        if pd.isna(start_x):
            start_x = 50.0
        if pd.isna(start_y):
            start_y = 50.0
        end_x, end_y = _event_xy(last)

        x_values = pd.concat(
            [team_events["_x"], team_events["_end_x"]], ignore_index=True
        ).dropna()
        max_x = float(x_values.max()) if not x_values.empty else float(start_x)
        window = team_events[
            team_events["_clock_seconds"] <= start_time + TRANSITION_WINDOW_SECONDS
        ]
        window_x = pd.concat(
            [window["_x"], window["_end_x"]], ignore_index=True
        ).dropna()
        transition_max_x = (
            float(window_x.max()) if not window_x.empty else float(start_x)
        )
        transition_progress = max(transition_max_x - float(start_x), 0.0)
        shot_mask = team_events["_is_shot"]
        goal_mask = team_events["_is_goal"]
        window_shots = window["_is_shot"]
        window_goals = window["_is_goal"]
        provider_fastbreak = team_events["_provider_fastbreak"]
        start_reason = str(first.get("possession_start_reason") or "")
        if not start_reason:
            reasons = group["possession_start_reason"]
            start_reason = next((str(v) for v in reasons if str(v)), "")
        transition_candidate = start_reason in {"recovery", "opponent_turnover"}
        reached_final_third_early = transition_max_x >= FINAL_THIRD_X
        reached_box_early = transition_max_x >= BOX_X
        is_transition = start_reason != "restart" and bool(
            provider_fastbreak.any()
            or (
                transition_candidate
                and (
                    transition_progress >= TRANSITION_MIN_PROGRESS
                    or reached_final_third_early
                    or reached_box_early
                    or window_shots.any()
                )
            )
        )
        xt = team_events["_xT"]
        xg = team_events["_xG"]
        passes = team_events["_is_pass"]
        successful = team_events["_successful"]
        movement_mask = (
            (
                passes
                | team_events.get("type", pd.Series("", index=team_events.index)).eq(
                    "Carry"
                )
            )
            & successful
            & team_events[["_x", "_y", "_end_x", "_end_y"]].notna().all(axis=1)
        )
        movement_distance = np.hypot(
            team_events.loc[movement_mask, "_end_x"]
            - team_events.loc[movement_mask, "_x"],
            team_events.loc[movement_mask, "_end_y"]
            - team_events.loc[movement_mask, "_y"],
        )
        summaries.append(
            {
                "possession_id": int(possession_value),
                "team_id": team,
                "period_code": first["_period_code"],
                "period_order": int(first["_period_order"]),
                "start_source_index": int(first["_source_index"]),
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "start_reason": start_reason,
                "start_event_type": str(first.get("type") or ""),
                "start_x": float(start_x),
                "start_y": float(start_y),
                "end_x": end_x,
                "end_y": end_y,
                "max_x": max_x,
                "net_progress": max(max_x - float(start_x), 0.0),
                "movement_distance": float(movement_distance.sum()),
                "passes": int(passes.sum()),
                "completed_passes": int((passes & successful).sum()),
                "shots": int(shot_mask.sum()),
                "goals": int(goal_mask.sum()),
                "xG": float(xg.sum()),
                "npxG": float(xg[~team_events["_is_penalty"]].sum()),
                "xT": float(xt[xt > 0].sum()),
                "box_entries": int(team_events["_box_entry"].sum()),
                "final_third_entries": int(team_events["_final_third_entry"].sum()),
                "provider_fastbreak": bool(provider_fastbreak.any()),
                "is_transition": is_transition,
                "transition_progress": transition_progress,
                "transition_shots": int(window_shots.sum()),
                "transition_goals": int(window_goals.sum()),
                "transition_xG": float(window["_xG"].sum()),
                "transition_xT": float(window.loc[window["_xT"] > 0, "_xT"].sum()),
                "transition_box_entries": int(window["_box_entry"].sum()),
                "transition_duration": min(duration, TRANSITION_WINDOW_SECONDS),
                "is_high_regain": bool(
                    transition_candidate and float(start_x) >= HIGH_REGAIN_X
                ),
                "counterpress_regain": False,
            }
        )

    possessions = pd.DataFrame(summaries)
    if not possessions.empty:
        possessions = possessions.sort_values("possession_id").reset_index(drop=True)
        # A counterpress regain is a team winning the ball back after the
        # opponent controlled it for no more than five seconds, near the loss.
        for idx in range(2, len(possessions)):
            before = possessions.iloc[idx - 2]
            opponent = possessions.iloc[idx - 1]
            regained = possessions.iloc[idx]
            if before["team_id"] != regained["team_id"]:
                continue
            if opponent["team_id"] == regained["team_id"] or opponent["duration"] > 5.0:
                continue
            coords = [
                before["end_x"],
                before["end_y"],
                regained["start_x"],
                regained["start_y"],
            ]
            if any(pd.isna(value) for value in coords):
                continue
            distance = math.hypot(
                float(before["end_x"]) - float(regained["start_x"]),
                float(before["end_y"]) - float(regained["start_y"]),
            )
            if distance <= 15.0 and regained["start_reason"] != "restart":
                possessions.at[idx, "counterpress_regain"] = True

    _POSSESSION_CACHE[id(events)] = (
        _cache_reference(events, _POSSESSION_CACHE),
        cache_key,
        work,
        possessions,
    )
    return work, possessions


def high_regain_events(events: pd.DataFrame, team_id: Any) -> pd.DataFrame:
    annotated, possessions = build_possessions(events)
    if possessions.empty:
        return events.iloc[0:0].copy()
    ids = possessions[
        (possessions["team_id"] == team_id) & possessions["is_high_regain"]
    ]["possession_id"]
    return annotated[
        annotated["possession_id"].isin(ids)
        & annotated["possession_start_reason"].ne("")
    ].copy()


def _possessions_with_game_state(
    possessions: pd.DataFrame,
    annotated: pd.DataFrame,
    info: dict[str, Any],
) -> pd.DataFrame:
    """Attach leading/drawing/trailing state at each possession's start."""
    if possessions.empty:
        return possessions.copy()

    home_id = info.get("home_id")
    away_id = info.get("away_id")
    scored_for = annotated.get(
        "scoring_team", annotated.get("team_id", pd.Series(index=annotated.index))
    )
    goals = annotated[
        annotated["_is_goal"]
        & ~_bool_series(annotated, "is_penalty_shootout")
        & scored_for.notna()
    ].copy()
    goals["_scoring_team"] = scored_for.loc[goals.index]
    goal_rows = [
        (
            int(row["_period_order"]),
            float(row["_clock_seconds"]),
            int(row["_source_index"]),
            row["_scoring_team"],
        )
        for _, row in goals.iterrows()
    ]

    enriched = possessions.copy()
    states: list[str] = []
    score_for: list[int] = []
    score_against: list[int] = []
    for _, possession in enriched.iterrows():
        start_key = (
            int(possession["period_order"]),
            float(possession["start_time"]),
            int(possession["start_source_index"]),
        )
        home_score = sum(
            1 for *key, team in goal_rows if tuple(key) < start_key and team == home_id
        )
        away_score = sum(
            1 for *key, team in goal_rows if tuple(key) < start_key and team == away_id
        )
        team_id = possession["team_id"]
        own_score = home_score if team_id == home_id else away_score
        opponent_score = away_score if team_id == home_id else home_score
        states.append(
            "leading"
            if own_score > opponent_score
            else "trailing" if own_score < opponent_score else "drawing"
        )
        score_for.append(own_score)
        score_against.append(opponent_score)
    enriched["game_state"] = states
    enriched["score_for"] = score_for
    enriched["score_against"] = score_against
    return enriched


def player_sequence_metrics(events: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return player xGChain, xGBuildup, and sequence-xT involvement.

    xGChain credits each participant in a possession with its non-penalty xG.
    xGBuildup applies the same credit after excluding the shot taker and key-pass
    provider. Sequence xT credits participants with the possession's positive xT.
    """
    cache_key = (
        len(events),
        (
            str(events.get("event_id", pd.Series(dtype=object)).iloc[-1])
            if "event_id" in events.columns and len(events)
            else ""
        ),
    )
    cached = _PLAYER_SEQUENCE_CACHE.get(id(events))
    if cached and cached[0]() is events and cached[1] == cache_key:
        return cached[2]

    annotated, possessions = build_possessions(events)
    credits: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {
            "xGChain": 0.0,
            "xGBuildup": 0.0,
            "sequence_xT": 0.0,
            "sequences": 0.0,
        }
    )
    sequence_groups = {
        int(possession_id): group
        for possession_id, group in annotated[
            annotated["possession_id"].notna()
        ].groupby("possession_id", sort=False)
    }
    for _, possession in possessions.iterrows():
        possession_id = int(possession["possession_id"])
        team_id = possession["team_id"]
        sequence_events = sequence_groups.get(possession_id, annotated.iloc[0:0])
        sequence_events = sequence_events[sequence_events.get("team_id") == team_id]
        if sequence_events.empty or "player" not in sequence_events.columns:
            continue
        player_names = sequence_events["player"].dropna().astype(str).str.strip()
        participants = {
            player for player in player_names if player and player.lower() != "nan"
        }
        shot_players = set(
            sequence_events.loc[sequence_events["_is_shot"], "player"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        key_pass_mask = _bool_series(sequence_events, "is_key_pass")
        key_pass_players = set(
            sequence_events.loc[key_pass_mask, "player"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        sequence_xg = float(possession.get("npxG", 0.0) or 0.0)
        sequence_xt = float(possession.get("xT", 0.0) or 0.0)
        for player in participants:
            credits[player]["xGChain"] += sequence_xg
            credits[player]["sequence_xT"] += sequence_xt
            credits[player]["sequences"] += 1.0
            if player not in shot_players and player not in key_pass_players:
                credits[player]["xGBuildup"] += sequence_xg

    result = {
        player: {key: round(value, 3) for key, value in values.items()}
        for player, values in credits.items()
    }
    _PLAYER_SEQUENCE_CACHE[id(events)] = (
        _cache_reference(events, _PLAYER_SEQUENCE_CACHE),
        cache_key,
        result,
    )
    return result


def team_advanced_metrics(
    events: pd.DataFrame, info: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return canonical team metrics for report and visualization consumers."""
    cache_key = (
        len(events),
        info.get("home_id"),
        info.get("away_id"),
        (
            str(events.get("event_id", pd.Series(dtype=object)).iloc[-1])
            if "event_id" in events.columns and len(events)
            else ""
        ),
    )
    cached = _TEAM_METRIC_CACHE.get(id(events))
    if cached and cached[0]() is events and cached[1] == cache_key:
        return cached[2]
    annotated, possessions = build_possessions(events)
    possessions = _possessions_with_game_state(possessions, annotated, info)
    result: dict[str, dict[str, Any]] = {}
    team_ids = {"home": info.get("home_id"), "away": info.get("away_id")}

    passes = _bool_series(events, "is_pass") & live_event_mask(events)
    successful = events.get("outcome", pd.Series("", index=events.index)).map(
        _outcome_is_successful
    )
    final_third_passes: dict[str, int] = {}
    for side, team_id in team_ids.items():
        team_pass_mask = passes & (events.get("team_id") == team_id)
        end_x = _numeric_series(events, "end_x", np.nan)
        final_third_passes[side] = int(
            (team_pass_mask & successful & (end_x >= FINAL_THIRD_X)).sum()
        )
    field_tilt_total = sum(final_third_passes.values())
    possession_duration_total = (
        float(possessions["duration"].sum()) if not possessions.empty else 0.0
    )

    for side, team_id in team_ids.items():
        team_events = events[events.get("team_id") == team_id].copy()
        team_pass_mask = passes & (events.get("team_id") == team_id)
        team_possessions = (
            possessions[possessions["team_id"] == team_id].copy()
            if not possessions.empty
            else pd.DataFrame()
        )
        transitions = (
            team_possessions[team_possessions["is_transition"]]
            if not team_possessions.empty
            else pd.DataFrame()
        )
        regains = (
            team_possessions[
                team_possessions["start_reason"].isin({"recovery", "opponent_turnover"})
            ]
            if not team_possessions.empty
            else pd.DataFrame()
        )
        high_regains = (
            team_possessions[team_possessions["is_high_regain"]]
            if not team_possessions.empty
            else pd.DataFrame()
        )
        build_ups = (
            team_possessions[team_possessions["start_x"] < 33]
            if not team_possessions.empty
            else pd.DataFrame()
        )
        successful_build_ups = (
            int((build_ups["max_x"] >= FINAL_THIRD_X).sum())
            if not build_ups.empty
            else 0
        )
        box_possessions = (
            team_possessions[team_possessions["box_entries"] > 0]
            if not team_possessions.empty
            else pd.DataFrame()
        )
        final_third_possessions = (
            team_possessions[team_possessions["final_third_entries"] > 0]
            if not team_possessions.empty
            else pd.DataFrame()
        )
        counterpress_attempts = 0
        counterpress_successes = 0
        rest_defence_exposures = 0
        rest_defence_dangerous_counters = 0
        ordered_possessions = possessions.sort_values("possession_id").reset_index(
            drop=True
        )
        for possession_index in range(max(len(ordered_possessions) - 1, 0)):
            lost = ordered_possessions.iloc[possession_index]
            opponent = ordered_possessions.iloc[possession_index + 1]
            if lost["team_id"] != team_id or opponent["team_id"] == team_id:
                continue
            if (
                lost["period_order"] != opponent["period_order"]
                or opponent["start_reason"] == "restart"
            ):
                continue
            counterpress_attempts += 1
            if possession_index + 2 < len(ordered_possessions):
                regained = ordered_possessions.iloc[possession_index + 2]
                if regained["team_id"] == team_id and bool(
                    regained["counterpress_regain"]
                ):
                    counterpress_successes += 1
            if float(lost["max_x"]) >= FINAL_THIRD_X:
                rest_defence_exposures += 1
                reached_final_third_in_window = (
                    float(opponent["start_x"])
                    + float(opponent["transition_progress"])
                    >= FINAL_THIRD_X
                )
                dangerous_counter = bool(opponent["is_transition"]) and (
                    int(opponent["transition_shots"]) > 0
                    or int(opponent["transition_box_entries"]) > 0
                    or (
                        float(opponent["transition_progress"])
                        >= DANGEROUS_COUNTER_MIN_PROGRESS
                        and reached_final_third_in_window
                    )
                )
                if dangerous_counter:
                    rest_defence_dangerous_counters += 1

        game_state_splits: dict[str, dict[str, float | int]] = {}
        for state in ("leading", "drawing", "trailing"):
            state_possessions = (
                team_possessions[team_possessions["game_state"] == state]
                if not team_possessions.empty
                else pd.DataFrame()
            )
            game_state_splits[state] = {
                "possessions": len(state_possessions),
                "completed_passes": (
                    int(state_possessions["completed_passes"].sum())
                    if not state_possessions.empty
                    else 0
                ),
                "shots": (
                    int(state_possessions["shots"].sum())
                    if not state_possessions.empty
                    else 0
                ),
                "xG": (
                    round(float(state_possessions["xG"].sum()), 2)
                    if not state_possessions.empty
                    else 0.0
                ),
                "sequence_xT": (
                    round(float(state_possessions["xT"].sum()), 2)
                    if not state_possessions.empty
                    else 0.0
                ),
                "transitions": (
                    int(state_possessions["is_transition"].sum())
                    if not state_possessions.empty
                    else 0
                ),
                "box_entries": (
                    int(state_possessions["box_entries"].sum())
                    if not state_possessions.empty
                    else 0
                ),
            }
        transition_count = len(transitions)
        provider_recoveries = int((team_events.get("type") == "BallRecovery").sum())
        team_touch_mask = touch_mask(team_events)
        touch_x = _numeric_series(team_events, "x", np.nan)
        touches = int(team_touch_mask.sum())
        result[side] = {
            "provider_recoveries": provider_recoveries,
            "possession_regains": len(regains),
            "high_regains": len(high_regains),
            "regain_to_shot_rate": round(
                100
                * int((regains["transition_shots"] > 0).sum())
                / max(len(regains), 1),
                1,
            ),
            "regain_xT": (
                round(float(regains["transition_xT"].sum()), 2)
                if not regains.empty
                else 0.0
            ),
            "regain_xG": (
                round(float(regains["transition_xG"].sum()), 2)
                if not regains.empty
                else 0.0
            ),
            "transitions": transition_count,
            "transition_shots": (
                int(transitions["transition_shots"].sum()) if transition_count else 0
            ),
            "transition_goals": (
                int(transitions["transition_goals"].sum()) if transition_count else 0
            ),
            "transition_xG": (
                round(float(transitions["transition_xG"].sum()), 2)
                if transition_count
                else 0.0
            ),
            "transition_xT": (
                round(float(transitions["transition_xT"].sum()), 2)
                if transition_count
                else 0.0
            ),
            "transition_box_entries": (
                int(transitions["transition_box_entries"].sum())
                if transition_count
                else 0
            ),
            "transition_shot_rate": round(
                100
                * int((transitions["transition_shots"] > 0).sum())
                / max(transition_count, 1),
                1,
            ),
            "avg_transition_duration": (
                round(float(transitions["transition_duration"].mean()), 1)
                if transition_count
                else 0.0
            ),
            "avg_transition_progress": (
                round(float(transitions["transition_progress"].mean()), 1)
                if transition_count
                else 0.0
            ),
            "counterpress_regains": (counterpress_successes),
            "counterpress_attempts": counterpress_attempts,
            "counterpress_success_rate": round(
                100 * counterpress_successes / max(counterpress_attempts, 1), 1
            ),
            "progressive_passes": int(progressive_pass_mask(team_events).sum()),
            "crosses": int(cross_mask(team_events).sum()),
            "completed_crosses": int(
                cross_mask(team_events, successful_only=True).sum()
            ),
            "touches": touches,
            "touch_def_pct": round(
                100 * int((team_touch_mask & (touch_x < 33)).sum()) / max(touches, 1)
            ),
            "touch_mid_pct": round(
                100
                * int((team_touch_mask & touch_x.between(33, 67)).sum())
                / max(touches, 1)
            ),
            "touch_att_pct": round(
                100 * int((team_touch_mask & (touch_x > 67)).sum()) / max(touches, 1)
            ),
            "field_tilt": round(
                100 * final_third_passes[side] / max(field_tilt_total, 1), 1
            ),
            "deep_completions": int(deep_completion_mask(team_events).sum()),
            "final_third_entries": int(final_third_entry_mask(team_events).sum()),
            "final_third_entry_possessions": len(final_third_possessions),
            "final_third_entry_efficiency": round(
                100
                * (
                    int((final_third_possessions["box_entries"] > 0).sum())
                    if not final_third_possessions.empty
                    else 0
                )
                / max(len(final_third_possessions), 1),
                1,
            ),
            "box_entries": int(box_entry_mask(team_events).sum()),
            "box_entry_to_shot_rate": round(
                100
                * int((box_possessions["shots"] > 0).sum())
                / max(len(box_possessions), 1),
                1,
            ),
            "build_up_attempts": len(build_ups),
            "build_up_successes": successful_build_ups,
            "build_up_success_rate": round(
                100 * successful_build_ups / max(len(build_ups), 1), 1
            ),
            "sequence_xT": (
                round(float(team_possessions["xT"].sum()), 2)
                if not team_possessions.empty
                else 0.0
            ),
            "sequence_xT_per_possession": (
                round(
                    float(team_possessions["xT"].sum()) / max(len(team_possessions), 1),
                    3,
                )
                if not team_possessions.empty
                else 0.0
            ),
            "xT_per_possession": (
                round(
                    float(team_possessions["xT"].sum()) / max(len(team_possessions), 1),
                    3,
                )
                if not team_possessions.empty
                else 0.0
            ),
            "directness": (
                round(
                    min(
                        100
                        * float(team_possessions["net_progress"].sum())
                        / max(float(team_possessions["movement_distance"].sum()), 1.0),
                        100.0,
                    ),
                    1,
                )
                if not team_possessions.empty
                else 0.0
            ),
            "rest_defence_exposures": rest_defence_exposures,
            "rest_defence_dangerous_counters": rest_defence_dangerous_counters,
            "rest_defence_vulnerability": round(
                100 * rest_defence_dangerous_counters / max(rest_defence_exposures, 1),
                1,
            ),
            "game_state_splits": game_state_splits,
            "possession_count": len(team_possessions),
            "possession_share": round(
                100
                * (
                    float(team_possessions["duration"].sum())
                    if not team_possessions.empty
                    else 0.0
                )
                / max(possession_duration_total, 1.0),
                1,
            ),
            "pass_share": round(
                100 * int(team_pass_mask.sum()) / max(int(passes.sum()), 1), 1
            ),
        }

    _TEAM_METRIC_CACHE[id(events)] = (
        _cache_reference(events, _TEAM_METRIC_CACHE),
        cache_key,
        result,
    )
    return result


def advanced_metrics_frames(
    events: pd.DataFrame, info: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return flat team and player frames ready for CSV export."""
    advanced = team_advanced_metrics(events, info)
    team_rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        row: dict[str, Any] = {
            "side": side,
            "team_id": info.get(f"{side}_id"),
            "team": info.get(f"{side}_name"),
        }
        for metric_name, metric_value in advanced[side].items():
            if metric_name == "game_state_splits":
                for state, state_metrics in metric_value.items():
                    for state_metric, state_value in state_metrics.items():
                        row[f"game_state_{state}_{state_metric}"] = state_value
            else:
                row[metric_name] = metric_value
        team_rows.append(row)

    player_teams: dict[str, Any] = {}
    if {"player", "team_id"}.issubset(events.columns):
        valid_players = events.dropna(subset=["player", "team_id"])
        for player, group in valid_players.groupby("player", sort=False):
            modes = group["team_id"].mode()
            if not modes.empty:
                player_teams[str(player)] = modes.iloc[0]

    player_rows: list[dict[str, Any]] = []
    for player_name, metrics in player_sequence_metrics(events).items():
        team_id = player_teams.get(str(player_name))
        team_name = (
            info.get("home_name")
            if team_id == info.get("home_id")
            else info.get("away_name") if team_id == info.get("away_id") else None
        )
        player_rows.append(
            {
                "player": player_name,
                "team_id": team_id,
                "team": team_name,
                **metrics,
            }
        )
    player_frame = pd.DataFrame(player_rows)
    if not player_frame.empty:
        player_frame = player_frame.sort_values(
            "xGChain", ascending=False, ignore_index=True
        )
    return pd.DataFrame(team_rows), player_frame


__all__ = [
    "BOX_X",
    "FINAL_THIRD_X",
    "HIGH_REGAIN_X",
    "advanced_metrics_frames",
    "blocked_shot_mask",
    "build_possessions",
    "box_entry_mask",
    "cross_mask",
    "deep_completion_mask",
    "defensive_block_events",
    "defensive_blocks_count",
    "final_third_entry_mask",
    "fouls_committed_count",
    "fouls_committed_mask",
    "high_regain_events",
    "live_event_mask",
    "player_sequence_metrics",
    "progressive_pass_mask",
    "team_advanced_metrics",
    "touch_mask",
]
