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



# ═════════════════════════════════════════════════════════════════════════════
# Shot placement and post-shot expected goals
# ═════════════════════════════════════════════════════════════════════════════
# Opta reports where a shot crossed the goal line in the same 0-100 scale as
# pitch width: the posts sit at 45.2 and 54.8, and the crossbar at a height
# of 38.
OPTA_POST_LEFT = 45.2
OPTA_POST_RIGHT = 54.8
OPTA_CROSSBAR = 38.0

# Placement qualifiers, used when the provider records the zone but not the
# exact crossing point. Values are fractions of the goal: x in -1..1 across the
# width, y in 0..1 up the height. Off-target zones sit outside that range.
PLACEMENT_ZONES: dict[str, tuple[float, float]] = {
    "lowleft": (-0.62, 0.18),
    "lowcentre": (0.00, 0.16),
    "lowright": (0.62, 0.18),
    "highleft": (-0.62, 0.76),
    "highcentre": (0.00, 0.80),
    "highright": (0.62, 0.76),
    "missleft": (-1.45, 0.42),
    "missright": (1.45, 0.42),
    "misshigh": (0.00, 1.20),
    "missleftandhigh": (-1.30, 1.14),
    "missrightandhigh": (1.30, 1.14),
    "missleftandlow": (-1.45, 0.14),
    "missrightandlow": (1.45, 0.14),
}

ON_TARGET_SHOT_TYPES = {"Goal", "SavedShot"}


def shot_placement(row: Any) -> tuple[float, float] | None:
    """Return where a shot crossed the goal line, in goal-frame fractions.

    ``(x, y)`` with x in -1..1 across the goal width and y in 0..1 up its
    height; values outside that box are off target. Prefers the provider's
    exact ``goal_mouth_y`` / ``goal_mouth_z``, and falls back to the placement
    qualifier when only the zone was recorded. Returns None when neither the
    coordinates nor a placement token are present — a blocked shot, or a wild
    miss the provider did not track.
    """
    getter = row.get if hasattr(row, "get") else (lambda key, default=None: default)
    gy = pd.to_numeric(pd.Series([getter("goal_mouth_y")]), errors="coerce").iloc[0]
    gz = pd.to_numeric(pd.Series([getter("goal_mouth_z")]), errors="coerce").iloc[0]
    if pd.notna(gy) and pd.notna(gz):
        span = (OPTA_POST_RIGHT - OPTA_POST_LEFT) / 2.0
        return (float(gy) - (OPTA_POST_LEFT + span)) / span, float(gz) / OPTA_CROSSBAR

    tokens = _qualifier_tokens(getter("qualifier_names"))
    for token in tokens:
        if token in PLACEMENT_ZONES:
            return PLACEMENT_ZONES[token]
    return None


def placement_difficulty(px: float, py: float) -> float:
    """Return how hard a placement is to save, from 0 (at the keeper) to 1.

    Reaching a shot costs the keeper both lateral travel and height, and the
    two are not equally expensive: getting down to a low ball near the post is
    quicker than getting up to the same lateral position under the bar. The
    lateral and vertical terms are combined with that asymmetry, then clamped —
    anything outside the frame is not a save situation at all.
    """
    lateral = min(abs(float(px)), 1.0)
    vertical = min(max(float(py), 0.0), 1.0)
    # Keeper's reachable envelope is widest at mid height; corners cost most.
    return float(min((lateral ** 1.35) * 0.62 + (vertical ** 1.25) * 0.38, 1.0))


def post_shot_xg(events: pd.DataFrame) -> pd.Series:
    """Return post-shot xG (PSxG) per event, zero for anything off target.

    Plain xG answers "how good was the chance". PSxG answers "how likely was
    this attempt, struck exactly where it was struck, to beat the keeper" — so
    it only exists for shots that reached the target, and it rewards placement
    the shot-quality model cannot see.

    The report previously used the sum of xG over on-target shots under the
    name xGoT, which ignores placement entirely: a shot rolled at the keeper
    and one in the top corner scored identically. This weights the base chance
    by how far the placement pulled the ball away from the goalkeeper.

    Heuristic, not a fitted model. The multiplier runs from 0.55 for a shot
    straight at the keeper to 2.45 in a corner, and the result is capped at
    0.97 because no placement is a certain goal.
    """
    if events is None or events.empty:
        return pd.Series(dtype=float)

    base = _numeric_series(events, "xG", 0.0).fillna(0.0).clip(lower=0.0)
    shot_type = events.get("shot_whoscored_type", pd.Series("", index=events.index)).astype(str)
    on_target = shot_type.isin(ON_TARGET_SHOT_TYPES)

    values = pd.Series(0.0, index=events.index, dtype=float)
    for idx in events.index[on_target]:
        point = shot_placement(events.loc[idx])
        if point is None:
            # On target but no placement recorded: fall back to the base chance.
            values.at[idx] = float(min(base.at[idx], 0.97))
            continue
        px, py = point
        difficulty = placement_difficulty(px, py)
        multiplier = 0.55 + 1.90 * difficulty
        values.at[idx] = float(min(base.at[idx] * multiplier, 0.97))

    # Goals keep their MODELLED value, they are not set to 1. Scoring the
    # attempt is the outcome PSxG is trying to predict; hard-coding it to
    # certainty would make "PSxG minus goals conceded" positive for every
    # keeper in every match and destroy the whole point of the measure.
    return values


def team_post_shot_xg(events: pd.DataFrame, team_id: Any) -> float:
    """Return one team's total PSxG."""
    if events is None or events.empty:
        return 0.0
    shots = events[_bool_series(events, "is_shot") & (events.get("team_id") == team_id)]
    if shots.empty:
        return 0.0
    return round(float(post_shot_xg(shots).sum()), 2)



# ═════════════════════════════════════════════════════════════════════════════
# Set pieces
# ═════════════════════════════════════════════════════════════════════════════
SET_PIECE_SOURCES = {
    "penalty": ("penalty",),
    "corner": ("fromcorner", "cornertaken"),
    "free_kick": ("freekicktaken", "directfreekick", "fromfreekick"),
    "throw_in": ("throwin", "fromthrowin"),
}


def shot_set_piece_source(row: Any) -> str:
    """Classify how a shot originated: penalty, corner, free kick, throw or open play.

    Read in that order deliberately — a penalty is also tagged as a free kick
    in some feeds, and a header from a corner routine that followed a free kick
    carries both tokens. First match wins so a shot is only counted once.
    """
    getter = row.get if hasattr(row, "get") else (lambda key, default=None: default)
    tokens = _qualifier_tokens(getter("qualifier_names"))
    if _is_true(getter("is_penalty")):
        return "penalty"
    for source, markers in SET_PIECE_SOURCES.items():
        if tokens.intersection(markers):
            return source
    return "open_play"


SET_PIECE_LOOKBACK_EVENTS = 3


def shot_origin(events: pd.DataFrame, index: Any, lookback: int = SET_PIECE_LOOKBACK_EVENTS) -> str:
    """Classify a shot's origin, looking back over the events that produced it.

    This feed does not tag the shot itself. A header from a corner carries only
    ``BoxCentre`` and ``Head``; the ``CornerTaken`` / ``FromCorner`` tokens sit
    on the delivery a couple of events earlier. Reading the shot row alone
    reports every set-piece goal as open play, so the preceding events are
    folded in.
    """
    own = shot_set_piece_source(events.loc[index])
    if own != "open_play":
        return own

    positions = events.index.get_indexer([index])
    if len(positions) == 0 or positions[0] < 0:
        return "open_play"
    position = int(positions[0])
    window = events.iloc[max(position - lookback, 0) : position + 1]

    tokens: set[str] = set()
    for _, row in window.iterrows():
        tokens |= _qualifier_tokens(row.get("qualifier_names"))
    if "penalty" in tokens:
        return "penalty"
    for source, markers in SET_PIECE_SOURCES.items():
        if tokens.intersection(markers):
            return source
    return "open_play"


def set_piece_breakdown(events: pd.DataFrame, team_id: Any) -> dict[str, dict[str, float]]:
    """Return shots, goals and xG for one team split by how the shot originated."""
    empty = {"shots": 0, "goals": 0, "xG": 0.0, "xG_per_shot": 0.0}
    result = {key: dict(empty) for key in ("open_play", "corner", "free_kick", "throw_in", "penalty")}
    if events is None or events.empty:
        return result

    ordered = events.sort_values(["minute", "second"], kind="stable")
    is_team_shot = (
        _bool_series(ordered, "is_shot")
        & (ordered.get("team_id") == team_id)
        & ~_bool_series(ordered, "is_penalty_shootout")
    )
    if not bool(is_team_shot.any()):
        return result

    xg = _numeric_series(ordered, "xG", 0.0).fillna(0.0).clip(lower=0.0)
    own_goal = _bool_series(ordered, "is_own_goal")
    goal = _bool_series(ordered, "is_goal") & ~own_goal
    for idx in ordered.index[is_team_shot]:
        bucket = result[shot_origin(ordered, idx)]
        bucket["shots"] += 1
        bucket["goals"] += int(bool(goal.at[idx]))
        bucket["xG"] += float(xg.at[idx])
    for bucket in result.values():
        bucket["xG"] = round(bucket["xG"], 2)
        bucket["xG_per_shot"] = round(bucket["xG"] / max(bucket["shots"], 1), 3)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Momentum, line height and compactness — all sampled in equal time windows
# ═════════════════════════════════════════════════════════════════════════════
DEFENSIVE_ACTION_TYPES = {
    "Tackle", "Interception", "Clearance", "BallRecovery", "Challenge",
    "BlockedPass", "Foul", "Aerial",
}


def _window_index(events: pd.DataFrame, window: int) -> pd.Series:
    minute = _numeric_series(events, "minute", 0.0).fillna(0.0)
    return (minute // max(window, 1)).astype(int)


def xg_momentum(events: pd.DataFrame, home_id: Any, away_id: Any, window: int = 5) -> pd.DataFrame:
    """Return per-window xG for each side and the home-minus-away differential.

    The cumulative xG curve answers who finished ahead; it cannot show who was
    on top at minute 60. A windowed differential can — it crosses zero every
    time control changes hands.
    """
    columns = ["window_start", "home_xG", "away_xG", "differential"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    shots = events[_bool_series(events, "is_shot") & ~_bool_series(events, "is_penalty_shootout")].copy()
    if shots.empty:
        return pd.DataFrame(columns=columns)

    shots["_xg"] = _numeric_series(shots, "xG", 0.0).fillna(0.0).clip(lower=0.0)
    shots["_window"] = _window_index(shots, window)
    last = int(shots["_window"].max())
    rows = []
    for w in range(last + 1):
        chunk = shots[shots["_window"].eq(w)]
        home = float(chunk.loc[chunk["team_id"].eq(home_id), "_xg"].sum())
        away = float(chunk.loc[chunk["team_id"].eq(away_id), "_xg"].sum())
        # Round the components first and subtract those, so the differential
        # always equals the two numbers printed beside it. Rounding each
        # independently leaves rows where 0.213 - 0.042 reads as 0.172.
        home_rounded = round(home, 3)
        away_rounded = round(away, 3)
        rows.append(
            {
                "window_start": w * window,
                "home_xG": home_rounded,
                "away_xG": away_rounded,
                "differential": round(home_rounded - away_rounded, 3),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def defensive_line_height(events: pd.DataFrame, team_id: Any, window: int = 5) -> pd.DataFrame:
    """Return the mean x of a team's defensive actions per time window.

    A single average hides the story. Tracking it over the match shows the
    minute a side stopped engaging high and dropped toward its own box.
    """
    columns = ["window_start", "height", "actions"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    types = events.get("type", pd.Series("", index=events.index)).astype(str)
    frame = events[
        types.isin(DEFENSIVE_ACTION_TYPES)
        & (events.get("team_id") == team_id)
        & ~_bool_series(events, "is_penalty_shootout")
    ].copy()
    frame = frame.dropna(subset=["x"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["_window"] = _window_index(frame, window)
    frame["_x"] = _numeric_series(frame, "x", np.nan)
    grouped = frame.groupby("_window")["_x"].agg(["mean", "count"]).reset_index()
    return pd.DataFrame(
        {
            "window_start": grouped["_window"] * window,
            "height": grouped["mean"].round(1),
            "actions": grouped["count"].astype(int),
        }
    )


def team_compactness(events: pd.DataFrame, team_id: Any, window: int = 5) -> pd.DataFrame:
    """Return vertical and horizontal spread of a team's touches per window.

    Spread is the interquartile range rather than min-to-max, so one full-back
    pushed on alone does not read as the whole side stretching.
    """
    columns = ["window_start", "vertical_spread", "horizontal_spread", "touches"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    frame = events[touch_mask(events) & (events.get("team_id") == team_id)].copy()
    frame = frame.dropna(subset=["x", "y"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["_window"] = _window_index(frame, window)
    frame["_x"] = _numeric_series(frame, "x", np.nan)
    frame["_y"] = _numeric_series(frame, "y", np.nan)
    rows = []
    for w, chunk in frame.groupby("_window"):
        if len(chunk) < 4:
            continue
        rows.append(
            {
                "window_start": int(w) * window,
                "vertical_spread": round(
                    float(chunk["_x"].quantile(0.75) - chunk["_x"].quantile(0.25)), 1
                ),
                "horizontal_spread": round(
                    float(chunk["_y"].quantile(0.75) - chunk["_y"].quantile(0.25)), 1
                ),
                "touches": int(len(chunk)),
            }
        )
    return pd.DataFrame(rows, columns=columns)



# ═════════════════════════════════════════════════════════════════════════════
# Pass-network centrality
# ═════════════════════════════════════════════════════════════════════════════
def pass_links(events: pd.DataFrame, team_id: Any, max_gap_seconds: float = 20.0) -> dict[tuple[str, str], int]:
    """Return completed pass counts between team-mates as {(from, to): count}.

    A pass is credited to a pair when the next event belongs to the same team,
    a different player, and follows closely enough to be the reception.
    """
    if events is None or events.empty:
        return {}

    work = events.sort_values(["minute", "second"], kind="stable").copy()
    work["_clock"] = _numeric_series(work, "minute", 0.0) * 60 + _numeric_series(work, "second", 0.0)
    work["_next_team"] = work.get("team_id").shift(-1)
    work["_next_player"] = work.get("player").shift(-1)
    work["_next_clock"] = work["_clock"].shift(-1)

    successful = work.get("outcome", pd.Series("", index=work.index)).map(_outcome_is_successful)
    is_pass = work.get("type", pd.Series("", index=work.index)).astype(str).eq("Pass")
    candidates = work[
        is_pass
        & successful
        & work.get("team_id").eq(team_id)
        & work["_next_team"].eq(team_id)
        & work.get("player").notna()
        & work["_next_player"].notna()
        & work["_next_clock"].sub(work["_clock"]).between(0, max_gap_seconds)
    ]

    links: dict[tuple[str, str], int] = defaultdict(int)
    for _, row in candidates.iterrows():
        source = str(row.get("player"))
        target = str(row.get("_next_player"))
        if source and target and source != target:
            links[(source, target)] += 1
    return dict(links)


def _brandes_betweenness(nodes: list[str], adjacency: dict[str, set[str]]) -> dict[str, float]:
    """Unweighted Brandes betweenness, normalized to 0..1.

    Implemented directly rather than pulling in networkx: the graph is at most
    a couple of dozen players, and the dependency is not otherwise required.
    """
    from collections import deque

    scores = {node: 0.0 for node in nodes}
    for source in nodes:
        stack: list[str] = []
        predecessors: dict[str, list[str]] = {node: [] for node in nodes}
        sigma = {node: 0.0 for node in nodes}
        distance = {node: -1 for node in nodes}
        sigma[source] = 1.0
        distance[source] = 0
        queue = deque([source])
        while queue:
            current = queue.popleft()
            stack.append(current)
            for neighbour in adjacency.get(current, ()):
                if distance[neighbour] < 0:
                    distance[neighbour] = distance[current] + 1
                    queue.append(neighbour)
                if distance[neighbour] == distance[current] + 1:
                    sigma[neighbour] += sigma[current]
                    predecessors[neighbour].append(current)
        delta = {node: 0.0 for node in nodes}
        while stack:
            node = stack.pop()
            for predecessor in predecessors[node]:
                if sigma[node]:
                    delta[predecessor] += (sigma[predecessor] / sigma[node]) * (1 + delta[node])
            if node != source:
                scores[node] += delta[node]

    count = len(nodes)
    if count > 2:
        # Undirected normalisation: every pair is counted from both ends.
        scale = 2.0 / ((count - 1) * (count - 2))
        scores = {node: value * scale for node, value in scores.items()}
    return scores


def network_centrality(events: pd.DataFrame, team_id: Any) -> pd.DataFrame:
    """Return per-player connection volume and betweenness for one team.

    The pass network already shows who passed to whom; it does not say who the
    side depended on to connect. Betweenness answers that — a high score means
    the player sits on the routes between team-mates, and removing them splits
    the network. Degree alone would just rank the busiest passer.
    """
    columns = ["player", "passes_made", "passes_received", "degree", "betweenness"]
    links = pass_links(events, team_id)
    if not links:
        return pd.DataFrame(columns=columns)

    nodes = sorted({name for pair in links for name in pair})
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    made: dict[str, int] = defaultdict(int)
    received: dict[str, int] = defaultdict(int)
    for (source, target), count in links.items():
        adjacency[source].add(target)
        adjacency[target].add(source)
        made[source] += count
        received[target] += count

    betweenness = _brandes_betweenness(nodes, adjacency)
    rows = [
        {
            "player": node,
            "passes_made": int(made.get(node, 0)),
            "passes_received": int(received.get(node, 0)),
            "degree": int(len(adjacency[node])),
            "betweenness": round(float(betweenness.get(node, 0.0)), 4),
        }
        for node in nodes
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("betweenness", ascending=False, kind="stable").reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# Turnovers
# ═════════════════════════════════════════════════════════════════════════════
TURNOVER_PUNISH_SECONDS = 15.0


def turnover_events(events: pd.DataFrame, team_id: Any) -> pd.DataFrame:
    """Return where a team lost the ball, and whether the loss was punished.

    A turnover count says little on its own — a side that passes more loses the
    ball more. What matters is where it was lost and whether the opponent
    turned the loss into a shot quickly, which is what ``punished`` marks.
    """
    columns = ["minute", "x", "y", "punished", "conceded_xG", "seconds_to_shot"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    _annotated, possessions = build_possessions(events)
    if possessions.empty:
        return pd.DataFrame(columns=columns)

    ordered = possessions.sort_values("start_time", kind="stable").reset_index(drop=True)
    rows = []
    for position in range(len(ordered) - 1):
        current = ordered.iloc[position]
        following = ordered.iloc[position + 1]
        if current["team_id"] != team_id or following["team_id"] == team_id:
            continue
        if str(following["start_reason"]) not in {"opponent_turnover", "recovery"}:
            continue
        gap = float(following["end_time"]) - float(following["start_time"])
        punished = bool(following["shots"] > 0 and gap <= TURNOVER_PUNISH_SECONDS)
        rows.append(
            {
                "minute": int(float(current["end_time"]) // 60),
                "x": float(current["end_x"]),
                "y": float(current["end_y"]),
                "punished": punished,
                "conceded_xG": round(float(following["xG"]), 3) if punished else 0.0,
                "seconds_to_shot": round(gap, 1) if punished else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


# ═════════════════════════════════════════════════════════════════════════════
# Duels and shot placement zones
# ═════════════════════════════════════════════════════════════════════════════
def duel_map(events: pd.DataFrame, team_id: Any) -> pd.DataFrame:
    """Return aerial and ground duels for one team with location and result.

    A duel total is a flat number, and the same 50% win rate means very
    different things in a team own box and in the opposition half, so location
    and duel type are kept alongside the result.
    """
    columns = ["minute", "x", "y", "kind", "won"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    types = events.get("type", pd.Series("", index=events.index)).astype(str)
    frame = events[
        types.isin({"Aerial", "Tackle", "Challenge", "TakeOn"})
        & (events.get("team_id") == team_id)
        & live_event_mask(events)
    ].dropna(subset=["x", "y"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    successful = frame.get("outcome", pd.Series("", index=frame.index)).map(_outcome_is_successful)
    kinds = types.loc[frame.index].map(lambda value: "aerial" if value == "Aerial" else "ground")
    return pd.DataFrame(
        {
            "minute": _numeric_series(frame, "minute", 0.0).astype(int),
            "x": _numeric_series(frame, "x", np.nan),
            "y": _numeric_series(frame, "y", np.nan),
            "kind": kinds.values,
            "won": successful.values,
        },
        columns=columns,
    ).reset_index(drop=True)


def shot_placement_zones(events: pd.DataFrame, team_id: Any) -> dict[str, int]:
    """Return how a team on-target shots were distributed across the goal.

    A 3x3 read of the frame: which parts of the goal a side actually attacks.
    """
    zones = {
        f"{vertical}_{horizontal}": 0
        for vertical in ("high", "mid", "low")
        for horizontal in ("left", "centre", "right")
    }
    if events is None or events.empty:
        return zones

    shots = events[
        _bool_series(events, "is_shot")
        & (events.get("team_id") == team_id)
        & ~_bool_series(events, "is_penalty_shootout")
    ]
    shot_type = shots.get("shot_whoscored_type", pd.Series("", index=shots.index)).astype(str)
    for idx in shots.index[shot_type.isin(ON_TARGET_SHOT_TYPES)]:
        point = shot_placement(shots.loc[idx])
        if point is None:
            continue
        px, py = point
        if abs(px) > 1.0 or not (0.0 <= py <= 1.0):
            continue
        horizontal = "left" if px < -0.28 else ("right" if px > 0.28 else "centre")
        vertical = "low" if py < 0.34 else ("high" if py > 0.66 else "mid")
        zones[f"{vertical}_{horizontal}"] += 1
    return zones



# ═════════════════════════════════════════════════════════════════════════════
# Pass geometry and goalkeeper distribution
# ═════════════════════════════════════════════════════════════════════════════
# Opta pitch coordinates are a 0-100 grid over a nominal 105x68 m pitch.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
LONG_BALL_METRES = 32.0


def pass_geometry(events: pd.DataFrame) -> pd.DataFrame:
    """Return length in metres and direction in degrees for every pass.

    The provider records ``Length`` and ``Angle`` as qualifier values, but this
    feed drops the values and keeps only the names, so both are derived from
    the coordinates instead. Where the parser did capture the provider values,
    those win — they are the source of truth and the geometry is only a
    reconstruction.

    Direction is 0 degrees straight up the pitch, positive to the right,
    negative to the left, so a lateral pass sits near +/-90 and a backward pass
    beyond +/-90.
    """
    columns = ["length_m", "direction_deg", "is_forward", "is_long"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    x = _numeric_series(events, "x", np.nan)
    y = _numeric_series(events, "y", np.nan)
    end_x = _numeric_series(events, "end_x", np.nan)
    end_y = _numeric_series(events, "end_y", np.nan)

    dx_m = (end_x - x) * (PITCH_LENGTH_M / 100.0)
    dy_m = (end_y - y) * (PITCH_WIDTH_M / 100.0)
    derived_length = np.sqrt(dx_m**2 + dy_m**2)

    provider = _numeric_series(events, "pass_length", np.nan)
    length = provider.where(provider > 0, derived_length)

    direction = np.degrees(np.arctan2(dy_m, dx_m))
    return pd.DataFrame(
        {
            "length_m": length.round(1),
            "direction_deg": pd.Series(direction, index=events.index).round(1),
            "is_forward": (end_x - x) > 2.0,
            "is_long": length >= LONG_BALL_METRES,
        },
        index=events.index,
        columns=columns,
    )


def pass_length_profile(events: pd.DataFrame, team_id: Any) -> dict[str, float]:
    """Return how one team distributed its passing by length and direction."""
    empty = {
        "passes": 0,
        "avg_length_m": 0.0,
        "long_ball_share": 0.0,
        "forward_share": 0.0,
        "completion": 0.0,
        "long_ball_completion": 0.0,
    }
    if events is None or events.empty:
        return empty

    passes = events[
        _bool_series(events, "is_pass")
        & (events.get("team_id") == team_id)
        & live_event_mask(events)
    ].dropna(subset=["x", "y", "end_x", "end_y"])
    if passes.empty:
        return empty

    geometry = pass_geometry(passes)
    successful = passes.get("outcome", pd.Series("", index=passes.index)).map(_outcome_is_successful)
    long_balls = geometry["is_long"]
    return {
        "passes": int(len(passes)),
        "avg_length_m": round(float(geometry["length_m"].mean()), 1),
        "long_ball_share": round(100 * float(long_balls.mean()), 1),
        "forward_share": round(100 * float(geometry["is_forward"].mean()), 1),
        "completion": round(100 * float(successful.mean()), 1),
        "long_ball_completion": round(
            100 * float(successful[long_balls].mean()) if bool(long_balls.any()) else 0.0, 1
        ),
    }


GK_LAUNCH_METRES = 40.0


def goalkeeper_distribution(events: pd.DataFrame, team_id: Any, keeper: str | None = None) -> dict[str, float]:
    """Return how a goalkeeper used the ball: launch rate, length, completion.

    A keeper who plays 40 short passes and one hopeful clearance is doing a
    different job from one who launches half of them, and neither shows up in
    a save count. ``launch_share`` is the fraction of distribution sent beyond
    ``GK_LAUNCH_METRES``.
    """
    empty = {
        "distributions": 0,
        "avg_length_m": 0.0,
        "launch_share": 0.0,
        "completion": 0.0,
        "launch_completion": 0.0,
        "goal_kicks": 0,
    }
    if events is None or events.empty:
        return empty

    frame = events[
        _bool_series(events, "is_pass")
        & (events.get("team_id") == team_id)
        & live_event_mask(events)
    ].dropna(subset=["x", "y", "end_x", "end_y"])
    if frame.empty:
        return empty

    if keeper:
        frame = frame[frame.get("player").astype(str) == str(keeper)]
    else:
        # No keeper named: fall back to passes started inside the six-yard area,
        # which in practice is the goalkeeper distributing.
        frame = frame[_numeric_series(frame, "x", np.nan) <= 6.0]
    if frame.empty:
        return empty

    geometry = pass_geometry(frame)
    successful = frame.get("outcome", pd.Series("", index=frame.index)).map(_outcome_is_successful)
    launched = geometry["length_m"] >= GK_LAUNCH_METRES
    goal_kicks = int(
        sum(1 for _, row in frame.iterrows() if "goalkick" in _qualifier_tokens(row.get("qualifier_names")))
    )
    return {
        "distributions": int(len(frame)),
        "avg_length_m": round(float(geometry["length_m"].mean()), 1),
        "launch_share": round(100 * float(launched.mean()), 1),
        "completion": round(100 * float(successful.mean()), 1),
        "launch_completion": round(
            100 * float(successful[launched].mean()) if bool(launched.any()) else 0.0, 1
        ),
        "goal_kicks": goal_kicks,
    }



# ═════════════════════════════════════════════════════════════════════════════
# Press resistance
# ═════════════════════════════════════════════════════════════════════════════
PRESSURE_WINDOW_SECONDS = 4.0
PRESSURE_RADIUS = 14.0

PRESSURE_ACTION_TYPES = {
    "Tackle", "Challenge", "Interception", "BallRecovery", "Foul", "Aerial", "BlockedPass",
}


def pressure_mask(events: pd.DataFrame, team_id: Any) -> pd.Series:
    """Flag a team's actions that happened with an opponent closing them down.

    Event data has no defender coordinates, so pressure is inferred: an
    opposition defensive action logged within a few seconds and a short
    distance of the same spot. That is an approximation of pressure, not a
    tracking-grade measurement, and it will miss a defender who arrived without
    registering an event.
    """
    empty = pd.Series(False, index=events.index, dtype=bool)
    if events is None or events.empty:
        return empty

    work = events.copy()
    work["_clock"] = _numeric_series(work, "minute", 0.0) * 60 + _numeric_series(work, "second", 0.0)
    work["_x"] = _numeric_series(work, "x", np.nan)
    work["_y"] = _numeric_series(work, "y", np.nan)

    types = work.get("type", pd.Series("", index=work.index)).astype(str)
    opponents = work[
        types.isin(PRESSURE_ACTION_TYPES)
        & work.get("team_id").ne(team_id)
        & work["_x"].notna()
        & work["_y"].notna()
    ]
    if opponents.empty:
        return empty

    # Opponent actions mirror onto the acting team's coordinate frame.
    opponent_clock = opponents["_clock"].to_numpy()
    opponent_x = 100.0 - opponents["_x"].to_numpy()
    opponent_y = 100.0 - opponents["_y"].to_numpy()

    own = work.get("team_id").eq(team_id) & work["_x"].notna() & work["_y"].notna()
    flags = empty.copy()
    for idx in work.index[own]:
        clock = float(work.at[idx, "_clock"])
        near_in_time = np.abs(opponent_clock - clock) <= PRESSURE_WINDOW_SECONDS
        if not near_in_time.any():
            continue
        dx = opponent_x[near_in_time] - float(work.at[idx, "_x"])
        dy = opponent_y[near_in_time] - float(work.at[idx, "_y"])
        if bool((np.sqrt(dx**2 + dy**2) <= PRESSURE_RADIUS).any()):
            flags.at[idx] = True
    return flags


def press_resistance(events: pd.DataFrame, team_id: Any) -> dict[str, float]:
    """Return how well a team kept the ball when it was being pressed.

    The gap between the pressed and unpressed completion rate is the number
    that matters: a high overall completion built entirely on unpressed
    circulation is a different quality from one held under pressure.
    """
    empty = {
        "passes_under_pressure": 0,
        "pressed_share": 0.0,
        "pressed_completion": 0.0,
        "free_completion": 0.0,
        "resistance_gap": 0.0,
    }
    if events is None or events.empty:
        return empty

    passes = events[
        _bool_series(events, "is_pass")
        & (events.get("team_id") == team_id)
        & live_event_mask(events)
    ]
    if passes.empty:
        return empty

    pressed = pressure_mask(events, team_id).reindex(passes.index).fillna(False)
    successful = passes.get("outcome", pd.Series("", index=passes.index)).map(_outcome_is_successful)
    pressed_completion = 100 * float(successful[pressed].mean()) if bool(pressed.any()) else 0.0
    free_completion = 100 * float(successful[~pressed].mean()) if bool((~pressed).any()) else 0.0
    return {
        "passes_under_pressure": int(pressed.sum()),
        "pressed_share": round(100 * float(pressed.mean()), 1),
        "pressed_completion": round(pressed_completion, 1),
        "free_completion": round(free_completion, 1),
        "resistance_gap": round(pressed_completion - free_completion, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Line-breaking passes
# ═════════════════════════════════════════════════════════════════════════════
def line_breaking_passes(events: pd.DataFrame, team_id: Any, opponent_id: Any, window: int = 5) -> pd.DataFrame:
    """Return passes that started behind the opponent's line and ended beyond it.

    Without tracking data the opponent's defensive line is estimated from where
    they were actually defending in that five-minute window, so the threshold
    moves with the game rather than sitting at a fixed x. A pass counts when it
    starts behind that line and finishes at least a few metres past it.
    """
    columns = ["minute", "x", "y", "end_x", "end_y", "line_height", "successful"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    heights = defensive_line_height(events, opponent_id, window=window)
    if heights.empty:
        return pd.DataFrame(columns=columns)
    # The opponent's own x needs mirroring into this team's attacking frame.
    height_by_window = {
        int(row.window_start): 100.0 - float(row.height) for row in heights.itertuples()
    }
    fallback = float(np.mean(list(height_by_window.values())))

    passes = events[
        _bool_series(events, "is_pass")
        & (events.get("team_id") == team_id)
        & live_event_mask(events)
    ].dropna(subset=["x", "end_x"])
    if passes.empty:
        return pd.DataFrame(columns=columns)

    minute = _numeric_series(passes, "minute", 0.0)
    start_x = _numeric_series(passes, "x", np.nan)
    end_x = _numeric_series(passes, "end_x", np.nan)
    successful = passes.get("outcome", pd.Series("", index=passes.index)).map(_outcome_is_successful)

    rows = []
    for idx in passes.index:
        window_start = int(minute.at[idx] // window) * window
        line = height_by_window.get(window_start, fallback)
        if start_x.at[idx] < line <= end_x.at[idx] - 4.0:
            rows.append(
                {
                    "minute": int(minute.at[idx]),
                    "x": float(start_x.at[idx]),
                    "y": float(_numeric_series(passes, "y", np.nan).at[idx]),
                    "end_x": float(end_x.at[idx]),
                    "end_y": float(_numeric_series(passes, "end_y", np.nan).at[idx]),
                    "line_height": round(line, 1),
                    "successful": bool(successful.at[idx]),
                }
            )
    return pd.DataFrame(rows, columns=columns)


# ═════════════════════════════════════════════════════════════════════════════
# Win probability
# ═════════════════════════════════════════════════════════════════════════════
def win_probability(events: pd.DataFrame, home_id: Any, away_id: Any, window: int = 5) -> pd.DataFrame:
    """Return a home win / draw / away win curve across the match.

    A logistic on the goal difference and the time still to play, nudged by the
    xG each side has been generating. It is a transparent heuristic, not a
    fitted market model — the point is to show how much a goal at minute 20 was
    worth compared with the same goal at minute 85.
    """
    columns = ["minute", "goal_difference", "home_win", "draw", "away_win"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    live = events[live_event_mask(events)].copy()
    if live.empty:
        return pd.DataFrame(columns=columns)

    live["_minute"] = _numeric_series(live, "minute", 0.0)
    live["_xg"] = _numeric_series(live, "xG", 0.0).fillna(0.0).clip(lower=0.0)
    goals = live[_bool_series(live, "is_goal")].copy()
    goals["_credited"] = (
        pd.to_numeric(goals.get("scoring_team"), errors="coerce")
        if "scoring_team" in goals.columns
        else pd.Series(np.nan, index=goals.index)
    ).fillna(goals.get("team_id"))

    full_time = max(90, int(live["_minute"].max()))
    rows = []
    for minute in range(0, full_time + 1, window):
        scored = goals[goals["_minute"] <= minute]
        home_goals = int((scored["_credited"] == home_id).sum())
        away_goals = int((scored["_credited"] == away_id).sum())
        difference = home_goals - away_goals

        played = max(minute, 1)
        remaining = max(full_time - minute, 0)
        shots = live[live["_minute"] <= minute]
        home_rate = float(shots.loc[shots["team_id"] == home_id, "_xg"].sum()) / played
        away_rate = float(shots.loc[shots["team_id"] == away_id, "_xg"].sum()) / played
        # Expected remaining goals feed the edge alongside the current score.
        #
        # Dividing by the minutes played makes the rate explode early — one
        # shot in the opening five minutes extrapolated to a two-goal edge and
        # sent the curve to 99% before anybody had scored. The rate is trusted
        # in proportion to how much of the match it is based on, and the whole
        # term is capped: a scoreline is evidence, a shot rate is a hint.
        confidence = min(played / 30.0, 1.0)
        expected_swing = float(
            np.clip((home_rate - away_rate) * remaining * confidence, -1.25, 1.25)
        )

        # A lead is worth more the less time is left to overturn it.
        certainty = 0.6 + 2.4 * (1 - remaining / max(full_time, 1))
        edge = difference * certainty + expected_swing
        draw_weight = math.exp(-abs(edge) * 0.9) * (0.42 + 0.38 * (remaining / max(full_time, 1)))
        home_raw = math.exp(edge * 0.85)
        away_raw = math.exp(-edge * 0.85)
        total = home_raw + away_raw + draw_weight * (home_raw + away_raw)
        rows.append(
            {
                "minute": minute,
                "goal_difference": difference,
                "home_win": round(home_raw / total, 3),
                "draw": round(draw_weight * (home_raw + away_raw) / total, 3),
                "away_win": round(away_raw / total, 3),
            }
        )
    return pd.DataFrame(rows, columns=columns)



# ═════════════════════════════════════════════════════════════════════════════
# Unified action value
# ═════════════════════════════════════════════════════════════════════════════
# This is a zone-value model, not a fitted VAEP. Real VAEP trains two
# classifiers on labelled sequences to estimate P(score) and P(concede) in the
# next N actions; that needs a season of data this project does not hold. What
# follows keeps VAEP's shape — every action, offensive and defensive, priced on
# one scale — using a transparent value surface instead of learned weights.
#
# The surface answers "how much is possession at this spot worth", rising
# toward the opposition goal and falling toward the touchlines.
ZONE_VALUE_MAX = 0.155
ZONE_VALUE_MIN = 0.002
TURNOVER_RISK = 0.55

# A defensive action does not bank the whole threat it interrupts — the
# opponent often recovers the ball and rebuilds. Crediting the full mirrored
# zone value put four centre-backs at the top of the ranking, which says more
# about the model than about the match. A regain that gives the ball away ends
# the attack; a clearance usually only delays it, so it earns less.
REGAIN_CREDIT = 0.35
CLEARANCE_CREDIT = 0.15


def zone_value(x: Any, y: Any) -> float:
    """Return the attacking value of holding the ball at (x, y), 0..~0.155.

    ``x`` runs 0-100 toward the opponent's goal, ``y`` 0-100 across the pitch.
    Depth dominates and is applied on a steep curve, because the difference
    between the halfway line and the edge of the box is far larger than the
    difference between two spots in one's own half. Width is a milder penalty
    for being pinned against a touchline.
    """
    try:
        depth = min(max(float(x), 0.0), 100.0) / 100.0
        width = min(max(float(y), 0.0), 100.0) / 100.0
    except (TypeError, ValueError):
        return ZONE_VALUE_MIN

    # Central band keeps full value; the wings lose up to a third of it.
    centrality = 1.0 - 0.34 * min(abs(width - 0.5) / 0.5, 1.0) ** 1.6
    span = ZONE_VALUE_MAX - ZONE_VALUE_MIN
    return float(ZONE_VALUE_MIN + span * (depth**3.2) * centrality)


def action_values(events: pd.DataFrame) -> pd.Series:
    """Return a value in goals for every action, offensive and defensive.

    Four cases:

    * a successful move (pass, carry, dribble) is worth the change in zone
      value between where it started and where it ended;
    * a failed move costs a fraction of the value it was holding, because
      losing the ball on the edge of the opponent's box is not the same
      mistake as losing it on your own goal line;
    * a shot is a terminal action worth its xG less the value it gave up;
    * a defensive action that regains the ball is worth the zone value it
      denies the opponent, mirrored into their attacking frame.

    Summing this per player gives one ranking that a centre-back and a winger
    can both appear in, which is the point of the exercise.
    """
    if events is None or events.empty:
        return pd.Series(dtype=float)

    x = _numeric_series(events, "x", np.nan)
    y = _numeric_series(events, "y", np.nan)
    end_x = _numeric_series(events, "end_x", np.nan)
    end_y = _numeric_series(events, "end_y", np.nan)
    xg = _numeric_series(events, "xG", 0.0).fillna(0.0).clip(lower=0.0)
    types = events.get("type", pd.Series("", index=events.index)).astype(str)
    successful = events.get("outcome", pd.Series("", index=events.index)).map(_outcome_is_successful)
    is_shot = _bool_series(events, "is_shot")
    live = live_event_mask(events)

    move_types = {"Pass", "Carry", "TakeOn", "BallTouch"}
    values = pd.Series(0.0, index=events.index, dtype=float)

    for idx in events.index:
        if not bool(live.at[idx]):
            continue
        start = zone_value(x.at[idx], y.at[idx]) if pd.notna(x.at[idx]) else ZONE_VALUE_MIN

        if bool(is_shot.at[idx]):
            values.at[idx] = float(xg.at[idx]) - start
            continue

        action = types.at[idx]
        if action in move_types:
            if not bool(successful.at[idx]):
                values.at[idx] = -start * TURNOVER_RISK
            elif pd.notna(end_x.at[idx]) and pd.notna(end_y.at[idx]):
                values.at[idx] = zone_value(end_x.at[idx], end_y.at[idx]) - start
            continue

        if action in REGAIN_TYPES or action in {"Clearance", "BlockedPass"}:
            if bool(successful.at[idx]) and pd.notna(x.at[idx]):
                # The opponent was attacking the other way, so the value denied
                # is read at the mirrored point, then discounted by how much of
                # the attack the action actually ended.
                denied = zone_value(100.0 - float(x.at[idx]), 100.0 - float(y.at[idx]))
                credit = REGAIN_CREDIT if action in REGAIN_TYPES else CLEARANCE_CREDIT
                values.at[idx] = denied * credit
    return values


def player_action_value(events: pd.DataFrame, team_id: Any | None = None) -> pd.DataFrame:
    """Return per-player action value, split into on-ball and defensive work."""
    columns = ["player", "team_id", "actions", "offensive_value", "defensive_value", "total_value"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    frame = events if team_id is None else events[events.get("team_id") == team_id]
    frame = frame[frame.get("player").notna()]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    values = action_values(frame)
    types = frame.get("type", pd.Series("", index=frame.index)).astype(str)
    defensive = types.isin(REGAIN_TYPES | {"Clearance", "BlockedPass"})

    work = pd.DataFrame(
        {
            "player": frame.get("player").astype(str),
            "team_id": frame.get("team_id"),
            "value": values,
            "defensive": defensive,
        }
    )
    grouped = work.groupby(["player", "team_id"], dropna=False)
    rows = []
    for (player, team), chunk in grouped:
        rows.append(
            {
                "player": player,
                "team_id": team,
                "actions": int(len(chunk)),
                "offensive_value": round(float(chunk.loc[~chunk["defensive"], "value"].sum()), 3),
                "defensive_value": round(float(chunk.loc[chunk["defensive"], "value"].sum()), 3),
                "total_value": round(float(chunk["value"].sum()), 3),
            }
        )
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values("total_value", ascending=False, kind="stable").reset_index(drop=True)



# ═════════════════════════════════════════════════════════════════════════════
# Pitch control
# ═════════════════════════════════════════════════════════════════════════════
CONTROL_DECAY = 11.0


def average_positions(events: pd.DataFrame, team_id: Any) -> pd.DataFrame:
    """Return each player's average touch position and touch count."""
    columns = ["player", "x", "y", "touches"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    frame = events[touch_mask(events) & (events.get("team_id") == team_id)]
    frame = frame.dropna(subset=["x", "y", "player"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    grouped = frame.groupby(frame["player"].astype(str)).agg(
        x=("x", "mean"), y=("y", "mean"), touches=("x", "size")
    ).reset_index().rename(columns={"player": "player"})
    grouped.columns = columns
    return grouped[grouped["touches"] >= 3].reset_index(drop=True)


def pitch_control(
    events: pd.DataFrame, home_id: Any, away_id: Any, cells_x: int = 60, cells_y: int = 40
) -> tuple[np.ndarray, dict[str, float]]:
    """Return a control surface and each side's share of the pitch.

    A hard Voronoi split says a cell belongs entirely to whoever is nearest,
    which is a poor description of football: two players a metre apart do not
    share a boundary, they contest the same space. Influence decays with
    distance instead, so a cell can be strongly held, weakly held or genuinely
    contested, and the surface stays smooth.

    Returns ``(grid, shares)`` where grid values run 0 (away control) to 1
    (home control), with 0.5 contested. Built from average positions, so it
    describes the shape a side held on average, not any single moment.
    """
    grid = np.full((cells_y, cells_x), 0.5, dtype=float)
    shares = {"home": 50.0, "away": 50.0, "contested": 0.0}

    home = average_positions(events, home_id)
    away = average_positions(events, away_id)
    if home.empty or away.empty:
        return grid, shares

    xs = np.linspace(0, 100, cells_x)
    ys = np.linspace(0, 100, cells_y)
    mesh_x, mesh_y = np.meshgrid(xs, ys)

    def influence(frame: pd.DataFrame, mirror: bool) -> np.ndarray:
        total = np.zeros_like(mesh_x, dtype=float)
        for row in frame.itertuples():
            px = 100.0 - float(row.x) if mirror else float(row.x)
            py = 100.0 - float(row.y) if mirror else float(row.y)
            distance = np.sqrt((mesh_x - px) ** 2 + (mesh_y - py) ** 2)
            total += np.exp(-distance / CONTROL_DECAY)
        return total

    # Both sides are placed in the home team's attacking frame.
    home_influence = influence(home, mirror=False)
    away_influence = influence(away, mirror=True)
    denominator = home_influence + away_influence
    grid = np.where(denominator > 0, home_influence / np.maximum(denominator, 1e-9), 0.5)

    contested = float(np.mean((grid > 0.45) & (grid < 0.55))) * 100
    shares = {
        "home": round(float(np.mean(grid > 0.55)) * 100, 1),
        "away": round(float(np.mean(grid < 0.45)) * 100, 1),
        "contested": round(contested, 1),
    }
    return grid, shares



# ═════════════════════════════════════════════════════════════════════════════
# Sequence typology
# ═════════════════════════════════════════════════════════════════════════════
SEQUENCE_TYPES = ("build_up", "sustained", "direct", "counter", "set_piece", "other")


def classify_sequence(possession: Any) -> str:
    """Return how one possession was built.

    Order matters. A counter that began at a throw-in is a restart first, and a
    long sequence that started deep is build-up rather than sustained pressure,
    so the more specific tests run before the general ones.
    """
    get = possession.get if hasattr(possession, "get") else (lambda key, default=None: default)
    passes = int(get("passes", 0) or 0)
    start_x = float(get("start_x", 50.0) or 50.0)
    progress = float(get("net_progress", 0.0) or 0.0)
    reason = str(get("start_reason", "") or "")
    duration = float(get("duration", 0.0) or 0.0)

    if reason == "restart":
        return "set_piece"
    if bool(get("is_transition", False)) and duration <= TRANSITION_WINDOW_SECONDS:
        return "counter"
    if start_x < 40.0 and passes >= 6:
        return "build_up"
    if passes >= 8 and start_x >= 40.0:
        return "sustained"
    if passes <= 3 and progress >= 40.0:
        return "direct"
    return "other"


def sequence_typology(events: pd.DataFrame, team_id: Any) -> pd.DataFrame:
    """Return how a team built its possessions, and what each kind was worth.

    Totals say how much a side created. This says which way of playing created
    it — the difference between a team that scores from sustained pressure and
    one that scores on the counter, which no aggregate can show.
    """
    columns = ["type", "sequences", "shots", "goals", "xG", "xG_per_sequence", "share_of_xG"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    _annotated, possessions = build_possessions(events)
    team = possessions[possessions["team_id"] == team_id] if not possessions.empty else pd.DataFrame()
    if team.empty:
        return pd.DataFrame(columns=columns)

    buckets = {key: {"sequences": 0, "shots": 0, "goals": 0, "xG": 0.0} for key in SEQUENCE_TYPES}
    for row in team.to_dict("records"):
        bucket = buckets[classify_sequence(row)]
        bucket["sequences"] += 1
        bucket["shots"] += int(row.get("shots", 0) or 0)
        bucket["goals"] += int(row.get("goals", 0) or 0)
        bucket["xG"] += float(row.get("xG", 0.0) or 0.0)

    total_xg = sum(bucket["xG"] for bucket in buckets.values())
    rows = []
    for key in SEQUENCE_TYPES:
        bucket = buckets[key]
        if not bucket["sequences"]:
            continue
        rows.append(
            {
                "type": key,
                "sequences": bucket["sequences"],
                "shots": bucket["shots"],
                "goals": bucket["goals"],
                "xG": round(bucket["xG"], 2),
                "xG_per_sequence": round(bucket["xG"] / max(bucket["sequences"], 1), 3),
                "share_of_xG": round(100 * bucket["xG"] / max(total_xg, 0.001), 1),
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("xG", ascending=False, kind="stable").reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# Receptions between the lines
# ═════════════════════════════════════════════════════════════════════════════
def receptions_between_lines(
    events: pd.DataFrame, team_id: Any, opponent_id: Any, window: int = 5, band: float = 12.0
) -> pd.DataFrame:
    """Return passes received in the pocket between the opponent's lines.

    The single most useful answer to "how did they open the block". Playing
    round a low block and playing through it produce similar possession
    numbers and completely different pictures; a reception in the band just in
    front of the defensive line is the one that hurts.

    The line is estimated per window from where the opponent actually defended,
    and the pocket is the ``band`` of pitch immediately in front of it.
    """
    columns = ["minute", "player", "x", "y", "line_height"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    heights = defensive_line_height(events, opponent_id, window=window)
    if heights.empty:
        return pd.DataFrame(columns=columns)
    height_by_window = {int(r.window_start): 100.0 - float(r.height) for r in heights.itertuples()}
    fallback = float(np.mean(list(height_by_window.values())))

    passes = events[
        _bool_series(events, "is_pass")
        & (events.get("team_id") == team_id)
        & live_event_mask(events)
    ]
    successful = passes.get("outcome", pd.Series("", index=passes.index)).map(_outcome_is_successful)
    passes = passes[successful].dropna(subset=["end_x", "end_y"])
    if passes.empty:
        return pd.DataFrame(columns=columns)

    ordered = events.sort_values(["minute", "second"], kind="stable")
    next_player = ordered.get("player").shift(-1)
    minute = _numeric_series(passes, "minute", 0.0)
    end_x = _numeric_series(passes, "end_x", np.nan)
    end_y = _numeric_series(passes, "end_y", np.nan)

    rows = []
    for idx in passes.index:
        line = height_by_window.get(int(minute.at[idx] // window) * window, fallback)
        # The pocket sits just in front of the line, on the defenders' side.
        if line - band <= end_x.at[idx] <= line:
            receiver = next_player.get(idx)
            rows.append(
                {
                    "minute": int(minute.at[idx]),
                    "player": str(receiver) if pd.notna(receiver) else "",
                    "x": float(end_x.at[idx]),
                    "y": float(end_y.at[idx]),
                    "line_height": round(line, 1),
                }
            )
    return pd.DataFrame(rows, columns=columns)


# ═════════════════════════════════════════════════════════════════════════════
# Switches, tempo and pressing triggers
# ═════════════════════════════════════════════════════════════════════════════
SWITCH_WIDTH = 40.0


def switches_of_play(events: pd.DataFrame, team_id: Any) -> pd.DataFrame:
    """Return passes that moved the ball a long way across the pitch.

    A switch is the direct evidence that a side was trying to stretch a block
    rather than force it centrally.
    """
    columns = ["minute", "player", "x", "y", "end_x", "end_y", "width", "successful"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    passes = events[
        _bool_series(events, "is_pass")
        & (events.get("team_id") == team_id)
        & live_event_mask(events)
    ].dropna(subset=["x", "y", "end_x", "end_y"])
    if passes.empty:
        return pd.DataFrame(columns=columns)

    y = _numeric_series(passes, "y", np.nan)
    end_y = _numeric_series(passes, "end_y", np.nan)
    width = (end_y - y).abs()
    wide = width >= SWITCH_WIDTH
    if not bool(wide.any()):
        return pd.DataFrame(columns=columns)

    selected = passes[wide]
    return pd.DataFrame(
        {
            "minute": _numeric_series(selected, "minute", 0.0).astype(int),
            "player": selected.get("player").astype(str),
            "x": _numeric_series(selected, "x", np.nan),
            "y": _numeric_series(selected, "y", np.nan),
            "end_x": _numeric_series(selected, "end_x", np.nan),
            "end_y": _numeric_series(selected, "end_y", np.nan),
            "width": width[wide].round(1),
            "successful": selected.get("outcome", pd.Series("", index=selected.index)).map(_outcome_is_successful),
        },
        columns=columns,
    ).reset_index(drop=True)


def time_to_progress(events: pd.DataFrame, team_id: Any) -> dict[str, float]:
    """Return how quickly a team carried a regain into the final third."""
    empty = {"possessions": 0, "reached_final_third": 0, "reach_rate": 0.0, "median_seconds": 0.0}
    if events is None or events.empty:
        return empty

    annotated, possessions = build_possessions(events)
    team = possessions[possessions["team_id"] == team_id] if not possessions.empty else pd.DataFrame()
    if team.empty:
        return empty

    clock = pd.to_numeric(annotated.get("_clock_seconds"), errors="coerce")
    reached = []
    for row in team.itertuples():
        if float(row.max_x) < FINAL_THIRD_X:
            continue
        window = annotated[
            annotated["possession_id"].eq(int(row.possession_id))
            & (pd.to_numeric(annotated.get("_x"), errors="coerce") >= FINAL_THIRD_X)
        ]
        if window.empty:
            continue
        first = float(clock.loc[window.index].min())
        reached.append(max(first - float(row.start_time), 0.0))

    return {
        "possessions": int(len(team)),
        "reached_final_third": int(len(reached)),
        "reach_rate": round(100 * len(reached) / max(len(team), 1), 1),
        "median_seconds": round(float(np.median(reached)), 1) if reached else 0.0,
    }


TRIGGER_LABELS = {
    "Pass": "opponent pass",
    "BallTouch": "loose touch",
    "TakeOn": "dribble attempt",
    "Clearance": "clearance",
    "ThrowIn": "throw-in",
}


def pressing_triggers(events: pd.DataFrame, team_id: Any) -> pd.DataFrame:
    """Return what the opponent was doing when a high regain happened.

    "They pressed well" is not a finding. Knowing the press repeatedly fed on
    back passes to the goalkeeper, or on loose first touches, is — it names the
    trigger a coach can drill against.
    """
    columns = ["trigger", "regains", "share"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    regains = high_regain_events(events, team_id)
    if regains.empty:
        return pd.DataFrame(columns=columns)

    ordered = events.sort_values(["minute", "second"], kind="stable").reset_index(drop=True)
    positions = {str(row.event_id): index for index, row in enumerate(ordered.itertuples()) if hasattr(row, "event_id")}

    counts: defaultdict[str, int] = defaultdict(int)
    for row in regains.itertuples():
        index = positions.get(str(getattr(row, "event_id", "")))
        if index is None or index == 0:
            continue
        previous = ordered.iloc[index - 1]
        if previous.get("team_id") == team_id:
            continue
        tokens = _qualifier_tokens(previous.get("qualifier_names"))
        if "throwin" in tokens:
            label = "throw-in"
        elif "goalkick" in tokens:
            label = "goal kick"
        elif str(previous.get("type")) == "Pass" and float(_numeric_series(ordered, "end_x", np.nan).iloc[index - 1] or 50) < float(_numeric_series(ordered, "x", np.nan).iloc[index - 1] or 50):
            label = "backward pass"
        else:
            label = TRIGGER_LABELS.get(str(previous.get("type")), str(previous.get("type") or "other").lower())
        counts[label] += 1

    total = sum(counts.values())
    rows = [
        {"trigger": label, "regains": count, "share": round(100 * count / max(total, 1), 1)}
        for label, count in counts.items()
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("regains", ascending=False, kind="stable").reset_index(drop=True)


def rest_defence_structure(events: pd.DataFrame, team_id: Any) -> dict[str, float]:
    """Return how many players were behind the ball when possession was lost.

    ``rest_defence_vulnerability`` already measures the consequence — how often
    a loss became a dangerous counter. This measures the cause: the structure
    the team was actually carrying at the moment it lost the ball.
    """
    empty = {"losses": 0, "avg_players_behind": 0.0, "exposed_losses": 0, "exposed_share": 0.0}
    if events is None or events.empty:
        return empty

    losses = turnover_events(events, team_id)
    if losses.empty:
        return empty

    touches = events[touch_mask(events) & (events.get("team_id") == team_id)].dropna(subset=["x"])
    if touches.empty:
        return empty
    touch_minute = _numeric_series(touches, "minute", 0.0)
    touch_x = _numeric_series(touches, "x", np.nan)
    touch_player = touches.get("player").astype(str)

    behind_counts = []
    for row in losses.itertuples():
        # Players active in the same minute, positioned behind the loss.
        window = (touch_minute >= row.minute - 1) & (touch_minute <= row.minute + 1)
        if not bool(window.any()):
            continue
        nearby = pd.DataFrame({"player": touch_player[window], "x": touch_x[window]})
        positions = nearby.groupby("player")["x"].mean()
        behind_counts.append(int((positions < float(row.x)).sum()))

    if not behind_counts:
        return empty
    exposed = sum(1 for count in behind_counts if count <= 3)
    return {
        "losses": int(len(behind_counts)),
        "avg_players_behind": round(float(np.mean(behind_counts)), 1),
        "exposed_losses": int(exposed),
        "exposed_share": round(100 * exposed / max(len(behind_counts), 1), 1),
    }



# ═════════════════════════════════════════════════════════════════════════════
# Goal origin, substitution impact, combinations, second balls, tilt timeline
# ═════════════════════════════════════════════════════════════════════════════
def goal_origin_chains(events: pd.DataFrame, home_id: Any, away_id: Any) -> pd.DataFrame:
    """Return the sequence behind every goal: where it began and how it was built.

    "68' Foden" is a fact. "68' — seven passes, 24 seconds, from a regain on
    halfway, four players involved" is the thing an analyst actually writes.
    """
    columns = [
        "minute", "scorer", "team_id", "sequence_type", "passes",
        "duration", "start_x", "start_y", "players", "started_from",
    ]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    annotated, possessions = build_possessions(events)
    if possessions.empty:
        return pd.DataFrame(columns=columns)

    goals = annotated[_bool_series(annotated, "is_goal") & ~_bool_series(annotated, "is_own_goal")]
    by_id = {int(row.possession_id): row for row in possessions.itertuples()}

    rows = []
    for goal in goals.itertuples():
        possession_id = getattr(goal, "possession_id", None)
        if possession_id is None or pd.isna(possession_id):
            continue
        possession = by_id.get(int(possession_id))
        if possession is None:
            continue
        window = annotated[annotated["possession_id"].eq(int(possession_id))]
        players = window.get("player").dropna().astype(str).nunique()
        rows.append(
            {
                "minute": int(_numeric_series(goals, "minute", 0.0).loc[goal.Index]),
                "scorer": str(getattr(goal, "player", "") or ""),
                "team_id": getattr(goal, "team_id", None),
                "sequence_type": classify_sequence(possession._asdict()),
                "passes": int(possession.passes),
                "duration": round(float(possession.duration), 1),
                "start_x": round(float(possession.start_x), 1),
                "start_y": round(float(possession.start_y), 1),
                "players": int(players),
                "started_from": str(possession.start_reason),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("minute", kind="stable").reset_index(drop=True)


def substitution_impact(events: pd.DataFrame, home_id: Any, away_id: Any) -> pd.DataFrame:
    """Return how the match changed either side of each substitution.

    Whether a change worked is normally argued from memory. Comparing the
    twenty minutes before and after gives it a number — with the caveat that a
    substitution is never the only thing that changed.
    """
    columns = ["minute", "team_id", "player_on", "player_off", "window",
               "xg_before", "xg_after", "tilt_before", "tilt_after"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    types = events.get("type", pd.Series("", index=events.index)).astype(str)
    subs = events[types.eq("SubstitutionOn")]
    if subs.empty:
        return pd.DataFrame(columns=columns)

    minute = _numeric_series(events, "minute", 0.0)
    xg = _numeric_series(events, "xG", 0.0).fillna(0.0).clip(lower=0.0)
    end_x = _numeric_series(events, "end_x", np.nan)
    passes = _bool_series(events, "is_pass")
    successful = events.get("outcome", pd.Series("", index=events.index)).map(_outcome_is_successful)
    final_third = passes & successful & (end_x >= FINAL_THIRD_X)

    def tilt(team: Any, low: float, high: float) -> float:
        window = (minute >= low) & (minute < high)
        own = int((final_third & window & events.get("team_id").eq(team)).sum())
        total = int((final_third & window).sum())
        return round(100 * own / max(total, 1), 1)

    # Pair each arrival with a departure in order within the same minute. A
    # nearest-match lookup gave all four of a half-time quadruple change the
    # same partner, which read as one player being replaced four times.
    departures: defaultdict[tuple[Any, int], list[str]] = defaultdict(list)
    off_rows = events[types.eq("SubstitutionOff")]
    for off in off_rows.itertuples():
        key = (getattr(off, "team_id", None), int(float(_numeric_series(off_rows, "minute", 0.0).loc[off.Index])))
        departures[key].append(str(getattr(off, "player", "") or ""))
    used: defaultdict[tuple[Any, int], int] = defaultdict(int)

    rows = []
    for sub in subs.itertuples():
        at = float(_numeric_series(subs, "minute", 0.0).loc[sub.Index])
        team = getattr(sub, "team_id", None)
        span = 20.0
        before = (minute >= at - span) & (minute < at)
        after = (minute >= at) & (minute < at + span)
        key = (team, int(at))
        pool = departures.get(key, [])
        position = used[key]
        used[key] += 1
        rows.append(
            {
                "minute": int(at),
                "team_id": team,
                "player_on": str(getattr(sub, "player", "") or ""),
                "player_off": pool[position] if position < len(pool) else "",
                "window": int(span),
                "xg_before": round(float(xg[before & events.get("team_id").eq(team)].sum()), 2),
                "xg_after": round(float(xg[after & events.get("team_id").eq(team)].sum()), 2),
                "tilt_before": tilt(team, at - span, at),
                "tilt_after": tilt(team, at, at + span),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("minute", kind="stable").reset_index(drop=True)


COMBINATION_WINDOW_SECONDS = 6.0


def third_man_combinations(events: pd.DataFrame, team_id: Any) -> pd.DataFrame:
    """Return repeated A-B-C passing patterns inside a short window.

    Two-player links show who plays with whom. Three-player chains show the
    rehearsed pattern, which is what a side actually trains.
    """
    columns = ["combination", "count"]
    links = []
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    work = events.sort_values(["minute", "second"], kind="stable").reset_index(drop=True)
    work["_clock"] = _numeric_series(work, "minute", 0.0) * 60 + _numeric_series(work, "second", 0.0)
    successful = work.get("outcome", pd.Series("", index=work.index)).map(_outcome_is_successful)
    is_pass = work.get("type", pd.Series("", index=work.index)).astype(str).eq("Pass")
    own = work.get("team_id").eq(team_id)

    for index in range(len(work) - 2):
        trio = work.iloc[index : index + 3]
        if not bool(own.iloc[index : index + 3].all()):
            continue
        if not bool(is_pass.iloc[index : index + 2].all()):
            continue
        if not bool(successful.iloc[index : index + 2].all()):
            continue
        if float(trio["_clock"].iloc[-1] - trio["_clock"].iloc[0]) > COMBINATION_WINDOW_SECONDS:
            continue
        names = [str(value) for value in trio.get("player").tolist()]
        if any(name in ("", "nan") for name in names) or len(set(names)) < 3:
            continue
        links.append(" \u2192 ".join(name.split()[-1] for name in names))

    if not links:
        return pd.DataFrame(columns=columns)
    counts: defaultdict[str, int] = defaultdict(int)
    for link in links:
        counts[link] += 1
    frame = pd.DataFrame(
        [{"combination": key, "count": value} for key, value in counts.items()], columns=columns
    )
    return frame.sort_values("count", ascending=False, kind="stable").reset_index(drop=True)


def second_ball_recovery(events: pd.DataFrame, team_id: Any) -> dict[str, float]:
    """Return how often a team won the loose ball after a long ball or set piece.

    The delivery is only half a dead-ball routine; who collects the knock-down
    decides whether it becomes a second phase or a counter against.
    """
    empty = {"contests": 0, "won": 0, "win_rate": 0.0}
    if events is None or events.empty:
        return empty

    work = events.sort_values(["minute", "second"], kind="stable").reset_index(drop=True)
    geometry = pass_geometry(work)
    types = work.get("type", pd.Series("", index=work.index)).astype(str)

    contests = 0
    won = 0
    for index in range(len(work) - 1):
        tokens = _qualifier_tokens(work.iloc[index].get("qualifier_names"))
        is_long = bool(geometry["is_long"].iloc[index]) and types.iloc[index] == "Pass"
        is_dead_ball = bool(tokens & {"cornertaken", "freekicktaken", "throwin"})
        if not (is_long or is_dead_ball):
            continue
        nxt = work.iloc[index + 1]
        if str(nxt.get("type")) not in {"Aerial", "BallRecovery", "Challenge", "Tackle", "BallTouch"}:
            continue
        contests += 1
        if nxt.get("team_id") == team_id and _outcome_is_successful(nxt.get("outcome")):
            won += 1

    return {
        "contests": contests,
        "won": won,
        "win_rate": round(100 * won / max(contests, 1), 1),
    }


def field_tilt_timeline(events: pd.DataFrame, home_id: Any, away_id: Any, window: int = 5) -> pd.DataFrame:
    """Return each side's share of final-third passes per time window.

    Field tilt as one number hides the swing. Over time it shows the spells
    where territory actually changed hands.
    """
    columns = ["window_start", "home_tilt", "away_tilt"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    passes = _bool_series(events, "is_pass") & live_event_mask(events)
    successful = events.get("outcome", pd.Series("", index=events.index)).map(_outcome_is_successful)
    end_x = _numeric_series(events, "end_x", np.nan)
    final_third = passes & successful & (end_x >= FINAL_THIRD_X)
    if not bool(final_third.any()):
        return pd.DataFrame(columns=columns)

    minute = _numeric_series(events, "minute", 0.0)
    last = int(minute.max())
    rows = []
    for start in range(0, last + 1, window):
        in_window = final_third & (minute >= start) & (minute < start + window)
        home = int((in_window & events.get("team_id").eq(home_id)).sum())
        away = int((in_window & events.get("team_id").eq(away_id)).sum())
        total = home + away
        if not total:
            continue
        rows.append(
            {
                "window_start": start,
                "home_tilt": round(100 * home / total, 1),
                "away_tilt": round(100 * away / total, 1),
            }
        )
    return pd.DataFrame(rows, columns=columns)


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
    "post_shot_xg",
    "shot_placement",
    "team_post_shot_xg",
    "placement_difficulty",
    "set_piece_breakdown",
    "shot_origin",
    "shot_set_piece_source",
    "xg_momentum",
    "defensive_line_height",
    "team_compactness",
    "network_centrality",
    "pass_links",
    "turnover_events",
    "duel_map",
    "shot_placement_zones",
    "pass_geometry",
    "pass_length_profile",
    "goalkeeper_distribution",
    "press_resistance",
    "pressure_mask",
    "line_breaking_passes",
    "win_probability",
    "action_values",
    "player_action_value",
    "zone_value",
    "average_positions",
    "pitch_control",
    "classify_sequence",
    "sequence_typology",
    "receptions_between_lines",
    "switches_of_play",
    "time_to_progress",
    "pressing_triggers",
    "rest_defence_structure",
    "goal_origin_chains",
    "substitution_impact",
    "third_man_combinations",
    "second_ball_recovery",
    "field_tilt_timeline",
    "progressive_pass_mask",
    "team_advanced_metrics",
    "touch_mask",
]
