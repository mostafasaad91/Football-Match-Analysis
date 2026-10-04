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
    # from extra_stats.player_extra
    "assists",
    "key_passes",
    "shots_on_target",
    "crosses",
    "crosses_completed",
    "long_balls",
    "long_balls_completed",
    "passes_into_final_third",
    "passes_into_box",
    "tackles",
    "fouls_committed",
    "fouls_won",
    "yellow_cards",
    "red_cards",
    "started",
]
PLAYER_PER90 = [
    "assists",
    "key_passes",
    "shots_on_target",
    "crosses",
    "long_balls",
    "passes_into_final_third",
    "passes_into_box",
    "tackles",
    "fouls_committed",
    "fouls_won",
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


def _mode_role(values):
    """The role a player most often started in; ``Unknown`` only when he never started."""
    known = values[~values.astype(str).isin(["Unknown", "nan", ""])].dropna()
    return known.mode().iloc[0] if len(known) else "Unknown"


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
    for column in PLAYER_COUNTS + [
        "minutes",
        "defensive_height",
        "padj_defensive_actions",
    ]:
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
    out["role"] = grouped["role_group"].agg(_mode_role) if "role_group" in frame else ""
    sums = grouped[
        [
            "_height_weight",
            "_height_actions",
            "_padj_weight",
            "_padj_minutes",
        ]
    ].sum()
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
    out["cross_pct"] = _ratio(out["crosses_completed"], out["crosses"], 100).values
    out["long_ball_pct"] = _ratio(out["long_balls_completed"], out["long_balls"], 100).values
    out["tackle_pct"] = _ratio(out["tackles_won"], out["tackles"], 100).values
    out["shot_accuracy"] = _ratio(out["shots_on_target"], out["shots"], 100).values
    out["goal_conversion"] = _ratio(out["goals"], out["shots"], 100).values
    out["goals_minus_xG"] = out["goals"] - out["xG"]
    out["goal_contributions"] = out["goals"] + out["assists"]
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
        "cross_pct",
        "long_ball_pct",
        "tackle_pct",
        "shot_accuracy",
        "goal_conversion",
    ):
        out[column] = kept[column]
    out["goal_contributions_p90"] = 90.0 * kept["goal_contributions"] / kept["minutes"]
    return out.sort_values("minutes", ascending=False, ignore_index=True)


# -- teams --------------------------------------------------------------------
# Names that mark a figure which is a rate, a share or an average: it is averaged
# over matches and never summed.
_RATE = re.compile(
    r"(rate|pct|share|tilt|ppda|avg_|average|per_|_per|efficiency|directness|"
    r"vulnerability|accuracy)",
    re.IGNORECASE,
)
# Counts whose names would otherwise read as rates: the two halves of PPDA.
_COUNTS = {"ppda_passes_allowed", "ppda_defensive_actions"}
TEAM_ID_COLUMNS = {"team_id", "side", "round"}


def is_rate(column: str) -> bool:
    """A figure that is averaged over matches and never added up."""
    return column not in _COUNTS and bool(_RATE.search(column))


# Ratios worked out from summed counts, so a side's pass completion over five matches
# is its completed passes over its attempts and not the mean of five percentages.
TEAM_RATIOS = {
    "pass_pct": ("passes_completed", "passes", 100.0),
    "long_ball_pct": ("long_balls_completed", "long_balls", 100.0),
    "cross_pct": ("completed_crosses", "crosses", 100.0),
    "aerial_pct": ("aerials_won", "aerials", 100.0),
    "ground_duel_pct": ("ground_duels_won", "ground_duels", 100.0),
    "shot_accuracy": ("on_target", "shots", 100.0),
    "goal_conversion": ("goals_for", "shots", 100.0),
    "xG_per_shot": ("xG", "shots", 1.0),
    "ppda": ("ppda_passes_allowed", "ppda_defensive_actions", 1.0),
    "xG_against_per_shot": ("xG_against", "shots_against", 1.0),
}


def _apply_team_ratios(table: pd.DataFrame, matches: pd.Series | None = None) -> pd.DataFrame:
    """Recompute the ratio columns, and the for-minus-against figures, from the counts.

    ``table`` holds summed counts (``matches`` is None) or per-match means (``matches`` given);
    a ratio of two means equals the ratio of the two sums, so one rule serves both.
    """
    for name, (top, bottom, scale) in TEAM_RATIOS.items():
        if top in table and bottom in table:
            table[name] = _ratio(table[top], table[bottom], scale).values
    if {"xG", "xG_against"} <= set(table.columns):
        table["xG_difference"] = table["xG"] - table["xG_against"]
    if {"goals_for", "xG"} <= set(table.columns):
        table["goals_minus_xG"] = table["goals_for"] - table["xG"]
    if {"xGoT_against", "goals_against"} <= set(table.columns):
        # What the side's goalkeeping saved beyond the shots it faced: the post-shot value of
        # the attempts on target, less the goals actually conceded.
        table["goals_prevented"] = table["xGoT_against"] - table["goals_against"]
    return table


def partial_columns(rows: pd.DataFrame) -> list[str]:
    """Numeric columns that are missing for some matches but not all of them.

    A sum or a mean over such a column silently covers only the matches that had it,
    and reads as if it covered them all. Packages rendered before a metric existed are
    the usual cause; the caller should rebuild the column or say so.
    """
    numeric = rows.select_dtypes(include="number")
    missing = numeric.isna().sum()
    return [c for c in numeric.columns if 0 < missing[c] < len(numeric)]


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
    totals = base[lead].join(sums).reset_index()
    # The two goal columns sit in the base block, so give the ratio step one table.
    totals = _apply_team_ratios(totals)
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
    per_match = _apply_team_ratios(per_match)
    order = ["points", "goal_difference", "goals_for"]
    totals = totals.sort_values(order, ascending=False, ignore_index=True)
    per_match = per_match.set_index("team").loc[totals["team"]].reset_index()
    return totals, per_match
