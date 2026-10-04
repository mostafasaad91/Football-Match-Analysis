"""Team and player figures added up over several matches.

Every package holds one match. A question that spans a block of rounds -- who
created the most in the first five, which side pressed hardest, who ran the
most progression metres per 90 -- needs those matches added up, and adding up
is where the mistakes are: a percentage cannot be summed, a per-90 cannot be
averaged across players with different minutes, and a rate of "x per 100 touches"
over five matches is not the mean of five rates.

So the two functions here sum what is a count, recompute what is a ratio from the
summed counts, weight the few averages that need a weight, and leave out of the
totals anything that is only meaningful per match.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# -- players ------------------------------------------------------------------
PLAYER_COUNTS = [
    "touches",
    "passes",
    "completed_passes",
    "progressive_passes",
    "progressive_carries",
    "progressions",
    "line_breaking_passes",
    "progression_metres",
    "final_third_receptions",
    "box_entries",
    "shots",
    "goals",
    "xG",
    "xA",
    "positive_xT",
    "xGChain",
    "xGBuildup",
    "takeons",
    "takeons_won",
    "dispossessed",
    "recoveries",
    "interceptions",
    "tackles_won",
    "clearances",
    "dribbled_past",
    "aerials",
    "aerials_won",
    "defensive_actions",
    "saves",
    "claims",
    "sweeps",
    "goals_prevented",
]
PLAYER_PER90 = [
    "xG",
    "xA",
    "xG_xA",
    "positive_xT",
    "xGChain",
    "xGBuildup",
    "shots",
    "touches",
    "passes",
    "progressive_passes",
    "progressions",
    "line_breaking_passes",
    "progression_metres",
    "final_third_receptions",
    "box_entries",
    "takeons_won",
    "recoveries",
    "interceptions",
    "tackles_won",
    "clearances",
    "defensive_actions",
    "dispossessed",
]


def _ratio(numerator, denominator, scale=1.0):
    """``scale * numerator / denominator``, blank where the denominator is zero."""
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return scale * pd.to_numeric(numerator, errors="coerce") / denominator


def _mode(values):
    values = values.dropna()
    return values.mode().iloc[0] if len(values) else ""


def aggregate_players(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per player and team from per-match rows (``player``, ``team``, ``minutes``, ...).

    ``rows`` is the concatenation of every match's player frame with a ``team`` name and a
    ``match_id`` column added. Counts are summed; ratios are recomputed from the sums;
    defensive height is weighted by defensive actions and the position-adjusted figure by
    minutes. Minutes are summed, and ``matches`` counts the matches the player appears in.
    """
    if rows.empty:
        return pd.DataFrame()
    frame = rows.copy()
    for column in PLAYER_COUNTS + ["minutes", "defensive_height", "padj_defensive_actions"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
    frame["_height_weight"] = frame["defensive_height"].fillna(0) * frame[
        "defensive_actions"
    ].fillna(0)
    frame["_height_actions"] = np.where(
        frame["defensive_height"].notna(), frame["defensive_actions"].fillna(0), 0
    )
    frame["_padj_weight"] = frame["padj_defensive_actions"].fillna(0) * frame["minutes"].fillna(0)
    frame["_padj_minutes"] = np.where(
        frame["padj_defensive_actions"].notna(), frame["minutes"].fillna(0), 0
    )
    grouped = frame.groupby(["player", "team"], sort=False)
    out = grouped[PLAYER_COUNTS + ["minutes"]].sum(min_count=1)
    out["matches"] = grouped["match_id"].nunique()
    out["role"] = grouped["role_group"].agg(_mode) if "role_group" in frame else ""
    sums = grouped[["_height_weight", "_height_actions", "_padj_weight", "_padj_minutes"]].sum()
    out = out.reset_index()
    out["pass_pct"] = _ratio(out["completed_passes"], out["passes"], 100).values
    out["progressive_pass_pct"] = _ratio(
        out["progressive_passes"], out["completed_passes"], 100
    ).values
    out["xG_per_shot"] = _ratio(out["xG"], out["shots"]).values
    out["xG_xA"] = out["xG"] + out["xA"]
    out["takeon_success_pct"] = _ratio(out["takeons_won"], out["takeons"], 100).values
    out["aerial_pct"] = _ratio(out["aerials_won"], out["aerials"], 100).values
    out["xT_per_100_touches"] = _ratio(out["positive_xT"], out["touches"], 100).values
    out["xA_per_100_passes"] = _ratio(out["xA"], out["passes"], 100).values
    out["defensive_height"] = _ratio(sums["_height_weight"], sums["_height_actions"]).values
    out["padj_defensive_actions"] = _ratio(sums["_padj_weight"], sums["_padj_minutes"]).values
    first = ["player", "team", "role", "matches", "minutes"]
    return out[first + [c for c in out.columns if c not in first]].sort_values(
        ["minutes", "player"], ascending=[False, True], ignore_index=True
    )


def per90(totals: pd.DataFrame, minimum_minutes: float = 90.0) -> pd.DataFrame:
    """Per-90 rates from player totals, for players with at least ``minimum_minutes``.

    A per-90 over twelve minutes is noise: the floor keeps the table to players the rate
    says something about. Goalkeepers' goals prevented is a total, not a rate, so it
    is carried over unchanged.
    """
    if totals.empty:
        return totals.copy()
    kept = totals[totals["minutes"].fillna(0) >= minimum_minutes].copy()
    out = kept[["player", "team", "role", "matches", "minutes"]].copy()
    for column in PLAYER_PER90:
        out[f"{column}_p90"] = 90.0 * kept[column] / kept["minutes"]
    for column in (
        "pass_pct",
        "progressive_pass_pct",
        "xG_per_shot",
        "takeon_success_pct",
        "aerial_pct",
        "xT_per_100_touches",
        "defensive_height",
        "padj_defensive_actions",
    ):
        out[column] = kept[column]
    return out.sort_values("minutes", ascending=False, ignore_index=True)


# -- teams --------------------------------------------------------------------
# Names that mark a figure which is a rate, a share or an average: it is averaged
# over matches and never summed.
_RATE = re.compile(
    r"(rate|pct|share|tilt|ppda|avg|average|per_|efficiency|height|spread|length|"
    r"speed|duration|progress|compact|accuracy|success|score|_per)",
    re.IGNORECASE,
)
TEAM_ID_COLUMNS = {"team_id", "side", "round"}


def is_rate(column: str) -> bool:
    return bool(_RATE.search(column))


def aggregate_teams(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Totals and per-match averages, one row per team, from per-match team rows.

    ``rows`` needs ``team``, ``match_id``, ``goals_for`` and ``goals_against`` plus any
    numeric metrics. Returns ``(totals, per_match)``. Totals leave rate-like columns
    blank, because five rates do not add up to anything; ``per_match`` is the mean of
    every numeric column, which is the right summary for a rate and a fair one for a count.
    Points, results and goal difference are worked out from the scores.
    """
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = rows.copy()
    numeric = [
        c
        for c in frame.select_dtypes(include="number").columns
        if c not in TEAM_ID_COLUMNS and c not in {"goals_for", "goals_against"}
    ]
    grouped = frame.groupby("team", sort=False)
    base = pd.DataFrame(
        {
            "matches": grouped["match_id"].nunique(),
            "goals_for": grouped["goals_for"].sum(),
            "goals_against": grouped["goals_against"].sum(),
        }
    )
    won = frame["goals_for"] > frame["goals_against"]
    drawn = frame["goals_for"] == frame["goals_against"]
    base["won"] = won.groupby(frame["team"], sort=False).sum().astype(int)
    base["drawn"] = drawn.groupby(frame["team"], sort=False).sum().astype(int)
    base["lost"] = base["matches"] - base["won"] - base["drawn"]
    base["points"] = 3 * base["won"] + base["drawn"]
    base["goal_difference"] = base["goals_for"] - base["goals_against"]
    lead = [
        "matches",
        "won",
        "drawn",
        "lost",
        "points",
        "goals_for",
        "goals_against",
        "goal_difference",
    ]

    sums = grouped[numeric].sum(min_count=1)
    for column in numeric:
        if is_rate(column):
            sums[column] = np.nan
    if {"xG", "shots"} <= set(sums.columns):
        sums["xG_per_shot"] = _ratio(sums["xG"], sums["shots"]).values
    totals = base[lead].join(sums).reset_index()
    means = grouped[numeric].mean()
    means = means.join(base["matches"]).reset_index()
    per_match = pd.concat(
        [
            means[["team", "matches"]],
            (base[["goals_for", "goals_against"]].div(base["matches"], axis=0)).reset_index(
                drop=True
            ),
            means.drop(columns=["team", "matches"]),
        ],
        axis=1,
    )
    order = ["points", "goal_difference", "goals_for"]
    totals = totals.sort_values(order, ascending=False, ignore_index=True)
    per_match = per_match.set_index("team").loc[totals["team"]].reset_index()
    return totals, per_match
