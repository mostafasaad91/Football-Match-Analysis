from __future__ import annotations

import re
from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd

from frame_values import number as _number, text as _text, whole as _whole
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

import crests
from match_report import compute_ppda_both
from visualization_components import IS_LIGHT_THEME


PAGE_W = 14 * 72
PAGE_H = 12.0 * 72
BASE_PAGE_H = 9 * 72
VISUAL_NOTE_H = PAGE_H - BASE_PAGE_H

# Aligned to the values the rendered visuals use, so the page chrome and the
# images sitting on it are the same ground rather than two near-matches.
#
# This module carried the black values as literals and had no theme branch at
# all, so the light package produced a black report with light visuals pasted
# onto it — the one part of that package still wearing the other identity.
if IS_LIGHT_THEME:
    BG = colors.HexColor("#F5F5F5")
    PANEL = colors.HexColor("#FFFFFF")
    PANEL_2 = colors.HexColor("#EDEDED")
    GRID = colors.HexColor("#D8D8D8")
    TEXT = colors.HexColor("#1F1F1F")
    MUTED = colors.HexColor("#5C6169")
    NEUTRAL = colors.HexColor("#8A8F97")
    HOME = colors.HexColor("#0A0A0A")
    AWAY = colors.HexColor("#E76F51")
else:
    BG = colors.HexColor("#000000")
    PANEL = colors.HexColor("#0A0A0A")
    PANEL_2 = colors.HexColor("#101010")
    GRID = colors.HexColor("#1C1C1C")
    TEXT = colors.HexColor("#FFFFFF")
    MUTED = colors.HexColor("#9A9A9A")
    NEUTRAL = colors.HexColor("#5A5A5A")
    HOME = colors.HexColor("#2F5BFF")
    AWAY = colors.HexColor("#FFD400")

# Fixture colours fall back to these when the caller supplies none, so the
# fallback has to follow the page too.
_DEFAULT_HOME, _DEFAULT_AWAY = HOME, AWAY

# Structural marks — section numbers, card rules, the spine label. These name
# parts of the report, not parts of the match, so they must not wear a colour
# that competes with the two teams. The previous amber did: it ran to 1,398
# characters against 648 for both kit colours combined, which made a fixed
# accent, rather than the fixture, the loudest thing in the document.
FOCUS = colors.HexColor("#3F4650") if IS_LIGHT_THEME else colors.HexColor("#C8CDD4")

# Text sits on this margin everywhere: headers, commentary, cards and the
# embedded visuals. One number, so a page has one left edge.
TEXT_MARGIN = 42

# The report had grown 23 distinct font sizes, several within a fifth of a
# point of each other, which is what a document looks like when every element
# was sized on its own. Six steps, each clearly different from its neighbour.
#
#   DISPLAY  the cover score
#   TITLE    page and commentary headings
#   SECTION  card titles, column headings
#   BODY     running text
#   CAPTION  subtitles, table values
#   MICRO    eyebrows, footers, legends
TYPE_DISPLAY, TYPE_TITLE, TYPE_SECTION = 34, 17, 11
TYPE_BODY, TYPE_CAPTION, TYPE_MICRO = 9, 7.5, 6.5

# One deliberate exception. On the cover the two halves of the lead statistic
# are set at different sizes because the gap between them is the finding —
# 8.8 against 91.2 should look as lopsided as it reads. Named rather than
# left as bare numbers so it stays a decision instead of becoming drift.
TYPE_LEAD_MINOR, TYPE_LEAD_MAJOR = 40, 62

# The cover's thesis sentence was set at TYPE_TITLE, the same size as the club
# names beside it, so the one line carrying the whole report did not outrank
# the fixture. It gets its own step, and the fixture line drops below it.
TYPE_THESIS, TYPE_FIXTURE = 23, 15

# The comparison card on the cover: the score, each club's name, the
# figure on each side of a row, and the initials that stand in for a crest
# that never downloaded. Named rather than written into the drawing code,
# because a bare number in a setFont call is a size nothing else can find.
TYPE_COVER_SCORE, TYPE_COVER_TEAM = 44, 18
TYPE_COVER_FIGURE, TYPE_COVER_MARK = 21, 27

# The cover's frame and the card inside it. The two rules are fixed to the
# sheet; everything between them is centred, so the air above the crests and
# the air under the last figure come out equal whatever COVER_ROWS holds.
#
# RISE, GAP and SINK are how far the card's ink actually reaches from the line
# it is drawn on: the crest half-height above the badge line, the badge line
# down to the first row, and a row's figures below their own line. They are the
# measurements the centring needs, so they live next to it rather than being
# re-derived from the drawing code.
COVER_MARGIN = 56
COVER_HEAD_DROP, COVER_FOOT_LIFT = 92, 88
COVER_CREST_RISE, COVER_ROW_GAP, COVER_ROW_SINK = 46, 104, 12
COVER_ROW_STEP = 52.0

# Sampled from the publisher's mark in assets/logo.jpg. Brand elements only —
# never a value, a bar, or anything a reader could mistake for a team. It reads
# teal against Manchester City's bluer #6CABDD, but the rule is what keeps them
# apart: the brand colour is never placed next to a number.
BRAND = colors.HexColor("#1F7C8A") if IS_LIGHT_THEME else colors.HexColor("#6BCAD6")

# The publisher's badge. Absent on a fresh clone, so every use is guarded and
# the cover falls back to a typographic wordmark rather than failing.
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.jpg"


def _as_pdf_color(value, fallback):
    """Return a reportlab colour for a hex string, or the fallback if unusable."""
    try:
        return colors.HexColor(str(value).strip())
    except (ValueError, AttributeError, TypeError):
        return fallback
VALUE = colors.HexColor("#5B3FBF") if IS_LIGHT_THEME else colors.HexColor("#9A7CF2")


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"1", "true", "yes"})


def _metric(frame: pd.DataFrame, side: str, key: str, default: float = 0.0) -> float:
    row = frame[frame["side"].astype(str).str.lower().eq(side.lower())]
    if row.empty or key not in row.columns:
        return float(default)
    value = pd.to_numeric(row.iloc[0][key], errors="coerce")
    return float(default if pd.isna(value) else value)


def _xg_metric(frame: pd.DataFrame, team: str, key: str, default: float = 0.0) -> float:
    row = frame[frame["team"].astype(str).str.lower().eq(team.lower())]
    if row.empty or key not in row.columns:
        return float(default)
    value = pd.to_numeric(row.iloc[0][key], errors="coerce")
    return float(default if pd.isna(value) else value)


def _name(row: pd.Series, fallback: str) -> str:
    value = row.get("team", fallback)
    return str(value) if pd.notna(value) else fallback


def _clock(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(frame.get("minute", 0), errors="coerce").fillna(0) * 60
        + pd.to_numeric(frame.get("second", 0), errors="coerce").fillna(0)
    )


def _goal_summary(events: pd.DataFrame, team_names: dict[int, str]) -> tuple[list[dict], str]:
    pso = _bool(events.get("is_penalty_shootout", pd.Series(False, index=events.index)))
    goals = events[_bool(events.get("is_goal", pd.Series(False, index=events.index))) & ~pso].copy()
    goals = goals.sort_values(["minute", "second", "event_id"], kind="stable")
    rows: list[dict] = []
    running = {team_id: 0 for team_id in team_names}
    for _, goal in goals.iterrows():
        team_id = int(goal["team_id"])
        running[team_id] = running.get(team_id, 0) + 1
        # str(goal.get("player", "Goal")) returns "nan" when the column exists
        # and the cell is empty — the default only covers a missing key — so a
        # goal with no recorded scorer was reported as "scored through nan".
        scorer = _text(goal.get("player"))
        rows.append(
            {
                "minute": _whole(goal.get("minute")),
                # The seconds are needed to say when a first-minute goal
                # arrived; without them the report and the article described
                # the same goal differently.
                "second": _whole(goal.get("second")),
                "team_id": team_id,
                "team": team_names.get(team_id, str(team_id)),
                "player": scorer,
                "score": dict(running),
            }
        )
    timeline = " | ".join(
        f"{row['minute']}' {row['player'].split()[-1] if row['player'] else 'unknown'} ({row['team']})"
        for row in rows
    )
    return rows, timeline


def _player_leaders(events: pd.DataFrame, player_metrics: pd.DataFrame, team: str) -> dict[str, str]:
    team_ids = player_metrics[player_metrics["team"].astype(str).str.lower().eq(team.lower())]["team_id"].dropna()
    team_id = int(team_ids.iloc[0]) if not team_ids.empty else None
    frame = events[events["team_id"].eq(team_id)].copy() if team_id is not None else events.iloc[0:0].copy()
    goals = frame[_bool(frame.get("is_goal", pd.Series(False, index=frame.index)))]
    goal_counts = goals.groupby("player").size().sort_values(ascending=False)
    shots = frame[_bool(frame.get("is_shot", pd.Series(False, index=frame.index)))].copy()
    shots["xG"] = pd.to_numeric(shots.get("xG", 0), errors="coerce").fillna(0)
    shot_xg = shots.groupby("player")["xG"].sum().sort_values(ascending=False)
    passes = frame[frame.get("type", pd.Series("", index=frame.index)).astype(str).eq("Pass")].copy()
    passes["xT"] = pd.to_numeric(passes.get("xT", 0), errors="coerce").fillna(0).clip(lower=0)
    pass_xt = passes.groupby("player")["xT"].sum().sort_values(ascending=False)
    team_pm = player_metrics[player_metrics["team"].astype(str).str.lower().eq(team.lower())]
    chain = team_pm.sort_values("xGChain", ascending=False) if "xGChain" in team_pm else team_pm
    buildup = team_pm.sort_values("xGBuildup", ascending=False) if "xGBuildup" in team_pm else team_pm
    sequence = team_pm.sort_values("sequence_xT", ascending=False) if "sequence_xT" in team_pm else team_pm

    def leader(series: pd.Series, fmt: str) -> str:
        # "No qualifying player" is a database message, not a sentence. A side
        # that did not score has nobody to name, and the report should say so.
        if series.empty:
            return "nobody"
        return f"{series.index[0]} ({fmt.format(float(series.iloc[0]))})"

    def frame_leader(data: pd.DataFrame, key: str, fmt: str) -> str:
        if data.empty or key not in data:
            return "nobody"
        row = data.iloc[0]
        return f"{row['player']} ({fmt.format(float(row[key]))})"

    def goal_leader() -> str:
        """The top scorer, with the count in words a person would use."""
        if goal_counts.empty:
            return "nobody scored"
        name, count = goal_counts.index[0], int(goal_counts.iloc[0])
        return f"{name} ({count} goal{'s' if count != 1 else ''})"

    return {
        "goals": goal_leader(),
        "shot_xg": leader(shot_xg, "{:.2f} xG"),
        "pass_xt": leader(pass_xt, "{:.2f} pass xT"),
        "chain": frame_leader(chain, "xGChain", "{:.2f} xGChain"),
        "buildup": frame_leader(buildup, "xGBuildup", "{:.2f} xGBuildup"),
        "sequence": frame_leader(sequence, "sequence_xT", "{:.2f} sequence xT"),
    }


def _all_player_profiles(events: pd.DataFrame, player_metrics: pd.DataFrame) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    pso = _bool(events.get("is_penalty_shootout", pd.Series(False, index=events.index)))
    live = events[~pso].copy()
    players = sorted(set(live.get("player", pd.Series(dtype=str)).dropna().astype(str)) | set(player_metrics.get("player", pd.Series(dtype=str)).dropna().astype(str)))
    for player in players:
        frame = live[live.get("player", pd.Series("", index=live.index)).astype(str).eq(player)].copy()
        pm = player_metrics[player_metrics.get("player", pd.Series("", index=player_metrics.index)).astype(str).eq(player)]
        shots = frame[_bool(frame.get("is_shot", pd.Series(False, index=frame.index)))].copy()
        shots["xG"] = pd.to_numeric(shots.get("xG", 0), errors="coerce").fillna(0)
        passes = frame[frame.get("type", pd.Series("", index=frame.index)).astype(str).eq("Pass")].copy()
        passes["xT"] = pd.to_numeric(passes.get("xT", 0), errors="coerce").fillna(0).clip(lower=0)
        team = str(pm.iloc[0].get("team", "")) if not pm.empty else ""
        profiles[player.lower()] = {
            "player": player,
            "team": team,
            "goals": int(_bool(frame.get("is_goal", pd.Series(False, index=frame.index))).sum()),
            "shots": int(len(shots)),
            "xG": float(shots["xG"].sum()),
            "key_passes": int(_bool(frame.get("is_key_pass", pd.Series(False, index=frame.index))).sum()),
            "pass_xT": float(passes["xT"].sum()),
            "xGChain": float(pd.to_numeric(pm.iloc[0].get("xGChain", 0), errors="coerce")) if not pm.empty else 0.0,
            "xGBuildup": float(pd.to_numeric(pm.iloc[0].get("xGBuildup", 0), errors="coerce")) if not pm.empty else 0.0,
            "sequence_xT": float(pd.to_numeric(pm.iloc[0].get("sequence_xT", 0), errors="coerce")) if not pm.empty else 0.0,
        }
    return profiles


def _match_verdict(team_metrics, xg, match_info):
    """The verdict, or None when it cannot be formed."""
    try:
        from match_verdict import read_match

        return read_match(team_metrics, xg, match_info)
    except Exception:
        return None


def build_context(
    events: pd.DataFrame,
    xg: pd.DataFrame,
    team_metrics: pd.DataFrame,
    player_metrics: pd.DataFrame,
    match_info: dict,
) -> dict:
    home_name = str(match_info["home_name"])
    away_name = str(match_info["away_name"])
    home_id = int(match_info["home_id"])
    away_id = int(match_info["away_id"])
    team_names = {home_id: home_name, away_id: away_name}
    goal_rows, goal_timeline = _goal_summary(events, team_names)
    home_goals = int(_xg_metric(xg, home_name, "goals", sum(row["team_id"] == home_id for row in goal_rows)))
    away_goals = int(_xg_metric(xg, away_name, "goals", sum(row["team_id"] == away_id for row in goal_rows)))
    winner = home_name if home_goals > away_goals else away_name if away_goals > home_goals else "Neither side"
    loser = away_name if winner == home_name else home_name if winner == away_name else "the opponent"

    info = {
        "home_id": home_id,
        "away_id": away_id,
        "home_name": home_name,
        "away_name": away_name,
    }
    try:
        ppda = compute_ppda_both(info, events)
        home_ppda = float(ppda["home"]["ppda"] or 0)
        away_ppda = float(ppda["away"]["ppda"] or 0)
    except Exception:
        home_ppda = away_ppda = 0.0

    context = {
        "home": home_name,
        "away": away_name,
        "home_id": home_id,
        "away_id": away_id,
        "home_goals": home_goals,
        "away_goals": away_goals,
        # Em-dash, matching the score printed in every rendered visual. A
        # hyphen here made the same fixture look typeset by two different hands
        # depending on whether you were reading a page or an image on it.
        "score": f"{home_goals} — {away_goals}",
        # Whether each side's xG describes a performance or a deficit.
        # Shared with the article so the two documents cannot reach
        # opposite verdicts about the same two numbers.
        "verdict": _match_verdict(team_metrics, xg, match_info),
        "winner": winner,
        "loser": loser,
        "goal_rows": goal_rows,
        "goal_timeline": goal_timeline,
        "home_ppda": home_ppda,
        "away_ppda": away_ppda,
        "home_players": _player_leaders(events, player_metrics, home_name),
        "away_players": _player_leaders(events, player_metrics, away_name),
        "player_profiles": _all_player_profiles(events, player_metrics),
    }
    for side, team in [("home", home_name), ("away", away_name)]:
        for key in ["xG", "xGoT", "xG_per_shot", "shots", "on_target", "big_chances", "xT"]:
            context[f"{side}_{key}"] = _xg_metric(xg, team, key)
        for key in [
            "possession_share", "pass_share", "field_tilt", "deep_completions",
            "touches", "touch_def_pct", "touch_mid_pct", "touch_att_pct",
            "final_third_entries", "final_third_entry_efficiency", "box_entries",
            "box_entry_to_shot_rate", "build_up_success_rate", "sequence_xT",
            "build_up_attempts", "build_up_successes", "progressive_passes",
            "crosses", "completed_crosses", "directness", "high_regains",
            "regain_to_shot_rate", "regain_xG", "regain_xT",
            "transitions", "transition_shots", "transition_goals", "transition_xG",
            "transition_xT", "transition_shot_rate", "avg_transition_progress",
            "counterpress_regains", "counterpress_attempts", "counterpress_success_rate", "rest_defence_exposures",
            "rest_defence_dangerous_counters", "rest_defence_vulnerability",
        ]:
            context[f"{side}_{key}"] = _metric(team_metrics, side, key)
    return context


def _lead(home_name: str, away_name: str, home_value, away_value,
          tolerance: float = 0.0) -> tuple[str, str, bool]:
    """Return (leader, trailer, level) for one pair of values.

    Several readings named a fixed side — "Man City\'s curve finished above
    Arsenal\'s", "average chance quality favoured {away}" — and printed the two
    numbers next to the claim, so the sentence contradicted its own evidence
    whenever the home side led. Every such sentence asks this instead.
    """
    try:
        home_value, away_value = float(home_value), float(away_value)
    except (TypeError, ValueError):
        return home_name, away_name, False
    if abs(home_value - away_value) <= tolerance:
        return home_name, away_name, True
    if home_value >= away_value:
        return home_name, away_name, False
    return away_name, home_name, False


def _goal_moment(goal: dict) -> str:
    """When a goal arrived, phrased the way a report would phrase it.

    Opta counts the opening minute as minute 0, so a goal from the kick-off was
    printed as "in minute 0" — which is not a thing anyone says, and it landed
    on the one goal most worth describing precisely, since a goal that early is
    the reason the two sides never played a level match at all.
    """
    # int(nan) raises ValueError, and `nan or 0` returns the nan: a goal row
    # with an empty minute took the whole report down.
    minute = _whole(goal.get("minute"))
    second = _whole(goal.get("second"))
    if minute == 0:
        return f"after {second} seconds" if second else "straight from the kick-off"
    return f"in minute {minute + 1}"


def _section_copy(c: dict) -> dict[str, dict]:
    home, away = c["home"], c["away"]
    winner, loser = c["winner"], c["loser"]
    early = c["goal_rows"][0] if c["goal_rows"] else None
    early_text = (
        (f"{early['team']} scored through {early['player']} {_goal_moment(early)}"
         if early["player"] else f"{early['team']} scored {_goal_moment(early)}")
        + ", forcing the opponent to operate from a chasing game state."
        if early else "The score state did not create a clear early tactical constraint."
    )
    press_team = home if c["home_ppda"] < c["away_ppda"] else away
    press_ppda = min(c["home_ppda"], c["away_ppda"])
    transition_team = home if c["home_transition_shot_rate"] > c["away_transition_shot_rate"] else away
    territory_team = home if c["home_field_tilt"] > c["away_field_tilt"] else away
    territory_value = max(c["home_field_tilt"], c["away_field_tilt"])

    # Every heading and sentence below has to follow the numbers it is printed
    # beside. This card used to claim the volume was level and that the second
    # side "also attempted" whatever it had, so Arsenal's 9 against Manchester
    # City's 12 read as a match.
    home_shots, away_shots = int(c["home_shots"]), int(c["away_shots"])
    shot_gap = abs(home_shots - away_shots)
    if shot_gap == 0:
        shot_volume_head = "Shot volume was level"
        shot_volume_text = f"{home} and {away} attempted {home_shots} shots each."
    else:
        shot_leader = home if home_shots > away_shots else away
        shot_trailer = away if shot_leader == home else home
        # A flat "gap of three or fewer" called 9 against 12 close, which is a
        # third more attempts. Closeness is relative to the volume shot.
        close = shot_gap <= 2 or shot_gap / max(home_shots, away_shots) <= 0.15
        shot_volume_head = (
            "Shot volume was close" if close else f"{shot_leader} shot more often"
        )
        shot_volume_text = (
            f"{home} attempted {home_shots} shots; {away} attempted {away_shots}."
            if close else
            f"{shot_leader} attempted {max(home_shots, away_shots)} shots; "
            f"{shot_trailer} {min(home_shots, away_shots)}."
        )

    # "The game never settled" was printed over every result, including a 3-0
    # settled by a goal inside the first minute. Whether a match swung is a
    # question about who scored and when, so ask it.
    scorers = {row["team"] for row in c["goal_rows"]}
    lead_changes = 0
    running = {home: 0, away: 0}
    ahead = None
    for row in c["goal_rows"]:
        running[row["team"]] = running.get(row["team"], 0) + 1
        now = None if running[home] == running[away] else (
            home if running[home] > running[away] else away)
        if now is not None and ahead is not None and now != ahead:
            lead_changes += 1
        ahead = now
    if lead_changes:
        settled_card = ("The lead changed hands", (
            f"The {c['score']} scoreline was reached after {lead_changes} change"
            f"{'s' if lead_changes != 1 else ''} of leader, so neither side spent the "
            f"match defending the same problem."))
    elif len(scorers) == 1 and early:
        settled_card = ("One side led throughout", (
            f"Only {early['team']} scored, and did so {_goal_moment(early)}, so the "
            f"tactical conditions were set early and never reversed."))
    else:
        settled_card = ("The result held", (
            f"The {c['score']} scoreline was built without the lead changing hands, so "
            f"the trailing side spent the match solving one problem rather than several."))

    # Four cards below asserted a direction and then printed numbers that ran
    # the other way. Each now reads its own figures first.
    goals_total = c["home_goals"] + c["away_goals"]
    xg_total = c["home_xG"] + c["away_xG"]
    # A 1-0 reads "the match produced 1 goals". Six separate sentences in this
    # module formatted the total the same way and none reached for _count,
    # which has been here since the player pages were fixed for it. Nothing
    # caught it because every fixture under test had scored at least twice.
    scoreline = _count(goals_total, "goal", "goals")
    if goals_total > xg_total + 0.4:
        finishing_card = ("Finishing ran hot", (
            f"The match produced {scoreline} from {xg_total:.2f} combined xG, so "
            f"execution ran ahead of the chances created. Conversion is the noisiest part "
            f"of a match and the least likely to repeat."))
    elif goals_total < xg_total - 0.4:
        finishing_card = ("Finishing ran cold", (
            f"The match produced {scoreline} from {xg_total:.2f} combined xG, so the "
            f"chances created were not fully taken. The attacking processes were better than "
            f"the scoreline records."))
    else:
        finishing_card = ("Finishing tracked the chances", (
            f"The match produced {scoreline} from {xg_total:.2f} combined xG, so "
            f"conversion neither flattered nor hid either performance. The underlying numbers "
            f"can be read close to face value."))

    quality_team = home if c["home_xG_per_shot"] >= c["away_xG_per_shot"] else away
    quality_other = away if quality_team == home else home
    quality_best = max(c["home_xG_per_shot"], c["away_xG_per_shot"])
    quality_worst = min(c["home_xG_per_shot"], c["away_xG_per_shot"])
    volume_of_quality_team = (c["home_shots"] if quality_team == home else c["away_shots"])
    volume_of_other = (c["home_shots"] if quality_other == home else c["away_shots"])
    if abs(quality_best - quality_worst) <= 0.01:
        efficiency_card = ("Chance quality did not separate them", (
            f"Both sides averaged close to {quality_best:.3f} expected goals an attempt, so "
            f"the difference in the match was made somewhere other than in the value of the "
            f"shooting."))
    elif volume_of_quality_team < volume_of_other:
        efficiency_card = ("The decisive edge was efficiency", (
            f"{quality_team} shot less often than {quality_other} and still averaged "
            f"{quality_best:.3f} expected goals an attempt against {quality_worst:.3f}, "
            f"converting fewer attacking situations into a higher-quality return."))
    else:
        efficiency_card = ("The better chances went with the volume", (
            f"{quality_team} both shot more often and averaged the better attempt, "
            f"{quality_best:.3f} expected goals against {quality_worst:.3f}, so the shooting "
            f"advantage was not a trade-off between quantity and quality."))

    possession_team = home if c["home_possession_share"] >= c["away_possession_share"] else away
    if possession_team == territory_team:
        territory_card = ("Territory and possession pointed the same way", (
            f"{territory_team} held {territory_value:.1f}% of the final-third passing and "
            f"{max(c['home_possession_share'], c['away_possession_share']):.1f}% of possession, "
            f"so the ball and the ground were owned by the same side. The open question is "
            f"whether that control reached the penalty area or stopped in front of it."))
    else:
        territory_card = ("Territory and possession told different stories", (
            f"{territory_team} held {territory_value:.1f}% of the final-third passing while "
            f"{possession_team} held the larger share of the ball, so one side owned the "
            f"match and the other owned the ground closest to goal."))

    if transition_team == winner and winner:
        game_state_card = ("Game state amplified the pattern", (
            f"{winner}'s lead created more opportunities to attack space as {loser} committed "
            f"additional players, so the transition advantage and the scoreline reinforced "
            f"each other."))
    elif winner:
        game_state_card = ("Game state cut against the pattern", (
            f"{transition_team} took more from broken play while {'leading' if transition_team == winner else 'chasing'} "
            f"the match, which is the harder version: a trailing side wins transitions against "
            f"an opponent that no longer needs to commit players, so the advantage came from "
            f"the first forward action rather than from the space the score created."))
    else:
        game_state_card = ("Neither side owned the score state", (
            f"{transition_team} took more from broken play in a match that stayed level, so "
            f"the transition advantage was structural rather than a product of the scoreline."))

    # An xG total is a record of chances, not of a performance. A losing side
    # whose total finished higher was left to look like the better team, and a
    # reader would take that from two numbers printed side by side — even when
    # the game-state split in the same report says almost all of it arrived
    # after they went behind, against an opponent that had stopped attacking.
    quality_card = ("Chance quality separated them",
                    f"{home} produced {c['home_xG']:.2f} xG; {away} produced "
                    f"{c['away_xG']:.2f} xG.")
    verdict = c.get("verdict")
    if verdict is not None and verdict.loser_was_only_chasing:
        beaten = verdict.of(verdict.loser)
        quality_card = ("The higher xG belongs to the chase", (
            f"{beaten.team} finished on {beaten.xg:.2f} xG against "
            f"{c['home_xG'] if verdict.winner == home else c['away_xG']:.2f}, but "
            f"{beaten.chasing_xg:.2f} of it — {100 * beaten.chasing_share:.0f}% — "
            f"came while behind. Before that, {beaten.not_chasing_xg:.2f}."))

    return {
        "Match Story": {
            "subtitle": "Score state, momentum and the moments that changed the tactical problem",
            "performance": [
                ("The first goal shaped the match", early_text),
                settled_card,
                ("Chasing changed risk", f"{loser} had to increase forward numbers and accept more space behind the ball as the match developed."),
            ],
            "data": [
                (shot_volume_head, shot_volume_text),
                quality_card,
                finishing_card,
            ],
            "implication": f"Read every territorial and pressing metric through game state: {winner} protected a lead for long periods, while {loser} accumulated attacking activity under greater urgency.",
        },
        "Chance Creation": {
            "subtitle": "Shot quality, final-third access and the difference between threat and conversion",
            "performance": [
                efficiency_card,
                ("Access did not guarantee clean shots", f"{territory_team}'s territorial control still had to pass through compact central protection and crowded finishing zones."),
                ("Final actions mattered", "Use the shot and zone maps to separate useful penetration from low-value circulation around the box."),
            ],
            "data": [
                ("Average shot quality", f"{home}: {c['home_xG_per_shot']:.3f} xG/shot; {away}: {c['away_xG_per_shot']:.3f}."),
                ("Shots on target", f"{home}: {int(c['home_on_target'])}; {away}: {int(c['away_on_target'])}."),
                ("Box-entry conversion", f"{home} turned {c['home_box_entry_to_shot_rate']:.1f}% of box entries into shots; {away} reached {c['away_box_entry_to_shot_rate']:.1f}%."),
            ],
            "implication": "The key coaching question is not who reached the final third more often, but who created the cleaner last action after arriving there.",
        },
        "Possession and Progression": {
            "subtitle": "How each side moved the ball, occupied territory and connected build-up to penetration",
            "performance": [
                territory_card,
                ("Progression was not the end product", "The pass maps should be read from first-line exit through final-third reception, not as isolated completion totals."),
                ("Half-to-half structures changed", "Average positions and networks are split by half so interval substitutions do not distort the starting structure."),
            ],
            "data": [
                ("Possession share", f"{home}: {c['home_possession_share']:.1f}%; {away}: {c['away_possession_share']:.1f}%."),
                ("Final-third access", f"{home}: {int(c['home_final_third_entries'])} entries and {int(c['home_deep_completions'])} deep completions; {away}: {int(c['away_final_third_entries'])} and {int(c['away_deep_completions'])}."),
                ("Build-up success", f"{home}: {c['home_build_up_success_rate']:.1f}%; {away}: {c['away_build_up_success_rate']:.1f}%."),
                ("Sequence value", f"{home}: {c['home_sequence_xT']:.2f} sequence xT; {away}: {c['away_sequence_xT']:.2f}."),
            ],
            "implication": f"{territory_team}'s structure generated repeated access, but the report must test whether that access created central superiority, box occupation and shots rather than possession without consequence.",
        },
        "Pressing and Rest Defence": {
            "subtitle": "Pressure intensity, regain value and the protection left behind the press",
            "performance": [
                ("Aggression and security were not the same", f"{press_team} pressed at the lower PPDA of {press_ppda:.2f}, but the rest-defence pages show the cost of committing players forward."),
                ("Pressing success needs an outcome", "A regain is only tactically valuable if it prevents progression or creates a useful next action."),
                ("The opponent exploited release moments", "Dangerous counters identify when the first pressure was bypassed and the back line had to defend open space."),
            ],
            "data": [
                ("PPDA", f"{home}: {c['home_ppda']:.2f}; {away}: {c['away_ppda']:.2f}. Lower means more aggressive pressure."),
                ("High regains", f"{home}: {int(c['home_high_regains'])}; {away}: {int(c['away_high_regains'])}."),
                ("Counterpress success", f"{home}: {c['home_counterpress_success_rate']:.1f}%; {away}: {c['away_counterpress_success_rate']:.1f}%."),
                ("Rest-defence vulnerability", f"{home}: {c['home_rest_defence_vulnerability']:.1f}% with {int(c['home_rest_defence_dangerous_counters'])} dangerous counters conceded; {away}: {c['away_rest_defence_vulnerability']:.1f}% and {int(c['away_rest_defence_dangerous_counters'])}."),
            ],
            "implication": "The tactical priority is the connection between the press and the cover behind it: pressure without compact rest defence can increase territorial control while also increasing opponent shot quality.",
        },
        "Transitions and Efficiency": {
            "subtitle": "The speed, value and conversion of attacks before the opponent could reset",
            "performance": [
                ("Transitions were the clearest separator", f"{transition_team} converted open-field moments into shots more consistently."),
                ("The first forward action mattered", "Transition progress measures whether regains immediately broke a line or merely restarted possession."),
                game_state_card,
            ],
            "data": [
                ("Transition shot rate", f"{home}: {c['home_transition_shot_rate']:.1f}% ({int(c['home_transition_shots'])}/{int(c['home_transitions'])}); {away}: {c['away_transition_shot_rate']:.1f}% ({int(c['away_transition_shots'])}/{int(c['away_transitions'])})."),
                ("Transition xG", f"{home}: {c['home_transition_xG']:.2f}; {away}: {c['away_transition_xG']:.2f}."),
                ("Transition goals", f"{home}: {int(c['home_transition_goals'])}; {away}: {int(c['away_transition_goals'])}."),
                ("Average progress", f"{home}: {c['home_avg_transition_progress']:.1f}; {away}: {c['away_avg_transition_progress']:.1f} pitch units."),
            ],
            "implication": f"{transition_team}'s advantage came from turning a few seconds of disorder into clearer shots. The defensive response is to secure the ball-side rest defence before attacking numbers advance.",
        },
        "Player Impact Appendix": {
            "subtitle": "Role-specific match influence, sequence involvement and individual event profiles",
            "performance": [
                (f"{home} attacking reference", c["home_players"]["goals"] + "; " + c["home_players"]["chain"] + "."),
                (f"{away} attacking reference", c["away_players"]["goals"] + "; " + c["away_players"]["chain"] + "."),
                ("Build-up influence", f"{home}: {c['home_players']['buildup']}; {away}: {c['away_players']['buildup']}."),
            ],
            "data": [
                ("Highest pass xT", f"{home}: {c['home_players']['pass_xt']}; {away}: {c['away_players']['pass_xt']}."),
                ("Highest sequence xT", f"{home}: {c['home_players']['sequence']}; {away}: {c['away_players']['sequence']}."),
                ("Interpretation rule", "Player radars describe this match only. Minutes, role and score state must be considered before treating them as ability ratings."),
            ],
            "implication": "Use the player pages to explain team mechanisms: who progressed the ball, who connected sequences and who converted the final action. Do not rank unlike roles from one match.",
        },
    }


MATCH_STORY = {"01", "04", "14", "15", "18", "23", "43"}
CHANCE_CREATION = {"02", "03", "11", "12", "13", "16", "17", "26", "27", "34", "35"}
POSSESSION = {"05a", "05b", "06a", "06b", "07", "08", "09", "10", "20", "21", "22", "24", "25", "31a", "31b", "32a", "32b", "33", "38", "39"}
PRESSING = {"28", "29", "30", "36", "37", "40"}
TRANSITIONS = {"41", "42"}


def classify_visual(path: Path) -> str:
    if "player_radars" in path.parts or path.stem.startswith("44_"):
        return "Player Impact Appendix"
    prefix = path.stem.split("_", 1)[0]
    if prefix in MATCH_STORY:
        return "Match Story"
    if prefix in CHANCE_CREATION:
        return "Chance Creation"
    if prefix in POSSESSION:
        return "Possession and Progression"
    if prefix in PRESSING:
        return "Pressing and Rest Defence"
    if prefix in TRANSITIONS:
        return "Transitions and Efficiency"
    return "Match Story"


def tactical_lens(path: Path) -> str:
    stem = path.stem.lower()
    rules = [
        ("xg_flow", "Read score changes against cumulative chance quality; a gap between goals and xG highlights execution and variance."),
        ("goals_breakdown", "Use the scoring order to understand the game-state pressure behind every later tactical choice."),
        ("shot_map", "Compare shot location and size before judging finishing; volume alone does not describe chance quality."),
        ("shot_profile", "Separate volume, accuracy and quality to identify whether the attack failed at access or execution."),
        ("danger_creation", "Look for repeatable routes into danger, not isolated high-value events."),
        ("zone14", "Central access in front of the box is most valuable when the next action breaks the final line."),
        ("box_entries", "Judge box access by entry type, receiver support and whether it produced a shot."),
        ("crosses", "Cross volume is useful only with box occupation, target quality and second-ball structure."),
        ("pass_network", "Read connections as team structure: spacing, hubs, width and substitution effects matter more than raw pass totals."),
        ("average_positions", "Use the half-specific structure to assess width, line height, compactness and role changes."),
        ("xt_map", "The heatmap shows where passes added threat; test whether hot zones connected to box entries and shots."),
        ("xt_per_minute", "Threat spikes identify the match periods when progression became penetration."),
        ("pass_map", "Distinguish circulation from line-breaking actions and note where failed passes exposed transition risk."),
        ("pass_thirds", "Compare retention and progression through each third to locate the build-up bottleneck."),
        ("progressive", "Progressive volume matters when receivers can continue forward before the defence resets."),
        ("pass_targets", "Destination density reveals occupation; combine it with completion and next-action quality."),
        ("dominating_zones", "Territorial dominance describes location, not outcome; compare it with shot quality and game state."),
        ("ppda", "Lower PPDA signals more aggressive pressure, but success must be checked against high regains and rest-defence exposure."),
        ("high_regains", "A high regain becomes valuable when it creates a shot before the opponent reorganises."),
        ("defensive_activity", "Location and type of defensive actions reveal whether the block defended proactively or close to its own goal."),
        ("defensive_summary", "Balance ball-winning volume with the quality of protection behind the challenge."),
        ("transition_outcomes", "Compare transition frequency with shot rate and xG to measure efficiency in moments of disorder."),
        ("advanced_metrics", "Read volume, efficiency, value and risk separately; a team can lead one layer and lose another."),
        ("game_state", "Leading and trailing phases change risk appetite, field position and the meaning of possession totals."),
        ("player_sequence", "Sequence leaders identify involvement in valuable attacks, not just the final pass or shot."),
        ("goalkeeper", "Separate save volume from post-shot quality to assess intervention rather than workload alone."),
        ("xg_summary", "Treat finishing above xG as match execution, not automatically as a repeatable attacking advantage."),
        ("match_stats", "Use the overview to frame the story, then rely on phase-specific pages for tactical explanation."),
        ("post_match_advanced", "Read attack and defence as one system: territorial ambition only helps when chance quality and protection behind the ball remain connected."),
        ("ball_touches", "Touch location describes occupation and game state; it does not by itself measure control."),
    ]
    if "player_radars" in path.parts:
        return "Single-match player profile: interpret every segment through minutes, position, role and team game state."
    for token, lens in rules:
        if token in stem:
            return lens
    return "Use this page as supporting evidence inside the section narrative, not as a standalone conclusion."


def _count(value, one: str, many: str) -> str:
    """A count with the noun that agrees with it.

    Player pages carry small numbers, so the plural is wrong more often than it
    is right: three separate paragraphs printed "1 goals, 1 shots and 1 key
    passes" about the same player.
    """
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return f"{value} {many}"
    return f"{number} {one if number == 1 else many}"


def _spaced_out(text: str) -> str:
    """Letter-spacing, which reportlab has no setting for."""
    return "  ".join(str(text))


def _club_initials(name: str) -> str:
    """AVFC from "Aston Villa FC"; PSG stays PSG."""
    words = [w for w in str(name).replace("-", " ").split() if w]
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:3].upper()
    # One letter per word, up to four. Truncating the joined initials instead
    # turned "Aston Villa FC" into AVF, dropping the C off the club's own
    # abbreviation.
    return "".join(w[0] for w in words[:4]).upper()


def _slugged(text: str) -> str:
    """A name reduced to the form filenames carry it in."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _visual_team(path: Path, context: dict) -> tuple[str | None, str | None]:
    # Use the visual filename only. The output directory contains both team
    # names (for example, France_vs_England) and must not influence attribution.
    #
    # Compare on the slug, not the name. "man city" is not a substring of
    # "03_shot_map_man_city.png", so every two-word side failed to be
    # identified and its fourteen boards fell through to the generic ending —
    # while a one-word side matched and got the real reading. The defect was
    # invisible in any fixture where both names happened to be one word.
    #
    # And match the slug as the trailing token rather than anywhere in the
    # name, longest first. A substring search gave every board of a Milan
    # derby to whichever Milan was listed at home, and handed a side called
    # Cross all the crossing boards, because "cross" sits inside "crosses".
    identity = _slugged(path.stem)
    for suffix in ("_1h", "_2h"):
        if identity.endswith(suffix):
            identity = identity[: -len(suffix)]
            break

    candidates = sorted(
        (("home", _slugged(str(context["home"]))), ("away", _slugged(str(context["away"])))),
        key=lambda pair: len(pair[1]), reverse=True,
    )
    for side, slug in candidates:
        if slug and (identity == slug or identity.endswith("_" + slug)):
            return str(context[side]), side
    return None, None


def _visual_title(path: Path) -> str:
    if "player_radars" in path.parts:
        return f"Player Radar - {path.stem.replace('_', ' ')}"
    stem = path.stem
    if "_" in stem:
        stem = stem.split("_", 1)[1]
    return stem.replace("_", " ").replace(" 1h", " - First Half").replace(" 2h", " - Second Half").title()


def _legacy_visual_explanation(path: Path, context: dict) -> str:
    stem = path.stem.lower()
    team, side = _visual_team(path, context)
    home, away = context["home"], context["away"]

    if "player_radars" in path.parts:
        player = path.stem.replace("_", " ")
        profile = context.get("player_profiles", {}).get(player.lower(), {})
        if profile:
            return (
                f"{player}'s match profile combines {_count(profile['goals'], 'goal', 'goals')}, "
                f"{_count(profile['shots'], 'shot', 'shots')} ({profile['xG']:.2f} xG), "
                f"{_count(profile['key_passes'], 'key pass', 'key passes')} and {profile['pass_xT']:.2f} pass xT. The sequence layer ({profile['sequence_xT']:.2f} xT; "
                f"{profile['xGChain']:.2f} xGChain) shows involvement beyond the final action, but the shape must still be interpreted through role, minutes and score state."
            )
        return tactical_lens(path) + " The page should be used to describe role-specific contribution, not to rank unlike positions."

    if "xg_flow" in stem:
        return (
            f"The cumulative curve shows {away} finishing on {context['away_xG']:.2f} xG against {home}'s {context['home_xG']:.2f}. "
            f"Because the match produced {_count(context['home_goals'] + context['away_goals'], 'goal', 'goals')} from {context['home_xG'] + context['away_xG']:.2f} combined xG, the scoreline contains a large execution component. "
            "Read each step with the goal markers to separate sustained chance creation from finishing variance."
        )
    if "goals_breakdown" in stem:
        halftime_home = sum(row["team_id"] == context["home_id"] and row["minute"] <= 45 for row in context["goal_rows"])
        halftime_away = sum(row["team_id"] == context["away_id"] and row["minute"] <= 45 for row in context["goal_rows"])
        return (
            f"The scoring sequence reached {halftime_home}-{halftime_away} by half-time before the second-half response changed the risk profile. "
            f"{context['loser']} had to add attacking numbers and accelerate restarts, while {context['winner']} gained more space to attack after regains. "
            "The assist labels identify the final connector, but the wider mechanism should be checked in the possession and transition sections."
        )
    if "goalkeeper" in stem:
        return (
            f"The goalkeepers faced {int(context['home_on_target'])} and {int(context['away_on_target'])} on-target attempts, with post-shot quality of "
            f"{context['home_xGoT']:.2f} xGoT for {home} and {context['away_xGoT']:.2f} for {away}. Compare saves with xGoT rather than save count alone: "
            "a high workload can contain routine shots, while a smaller number of high-quality attempts can be more decisive."
        )
    if "xg_summary" in stem or "shot_profile" in stem:
        volume_leader, _volume_trailer, volume_level = _lead(
            home, away, context["home_shots"], context["away_shots"], tolerance=2)
        quality_leader, quality_trailer, quality_level = _lead(
            home, away, context["home_xG_per_shot"], context["away_xG_per_shot"],
            tolerance=0.005)
        volume_text = (
            f"Shot volume was level at {int(context['home_shots'])}-{int(context['away_shots'])}"
            if volume_level else
            f"{volume_leader} shot more often, {int(context['home_shots'])}-{int(context['away_shots'])}"
        )
        leader_per_shot = context[
            "home_xG_per_shot" if quality_leader == home else "away_xG_per_shot"]
        trailer_per_shot = context[
            "home_xG_per_shot" if quality_trailer == home else "away_xG_per_shot"]
        quality_text = (
            "average chance quality was almost identical "
            f"({leader_per_shot:.3f} against {trailer_per_shot:.3f} xG per shot)"
            if quality_level else
            f"average chance quality favoured {quality_leader} "
            f"({leader_per_shot:.3f} against {trailer_per_shot:.3f} xG per shot)"
        )
        return (
            f"{volume_text}, and {quality_text}. The analytical separation is therefore quality and execution, not volume. "
            "Use the location maps to identify which entry routes produced that difference."
        )
    if "match_stats" in stem:
        tilt_leader, _tilt_trailer, _tilt_level = _lead(
            home, away, context["home_field_tilt"], context["away_field_tilt"])
        xg_leader, _xg_trailer, xg_level = _lead(
            home, away, context["home_xG"], context["away_xG"], tolerance=0.05)
        tilt_value = context[
            "home_field_tilt" if tilt_leader == home else "away_field_tilt"]
        if xg_level or xg_leader == tilt_leader:
            # No contradiction to point at: the same side led both.
            opening = (
                f"{tilt_leader} held {tilt_value:.1f}% field tilt and more final-third "
                f"access, and the xG return followed the territory"
            )
        else:
            opening = (
                f"The overview contains the central contradiction: {tilt_leader} held "
                f"{tilt_value:.1f}% field tilt and more final-third access, while "
                f"{xg_leader} produced the stronger xG return"
            )
        return (
            f"{opening}. "
            "Treat the page as a map of questions rather than a conclusion - the following phase-specific charts explain how territory, pressure and transitions created different outcomes."
        )
    if "xt_per_minute" in stem:
        return (
            f"Threat accumulation reached {context['home_sequence_xT']:.2f} sequence xT for {home} and {context['away_sequence_xT']:.2f} for {away}. "
            "Spikes identify the periods when progression became penetration; flat periods show circulation without a meaningful change in scoring potential. "
            "Compare these windows with goals and game state before calling one side consistently dominant."
        )
    if "game_state" in stem:
        return (
            f"The early score forced {context['loser']} to operate mainly from a trailing state and allowed {context['winner']} to choose when to accelerate. "
            "Leading teams can accept less territory while protecting central access and attacking open space; trailing teams often inflate entry and pressure totals through urgency. "
            "This split is the context layer for every full-match average in the report."
        )
    if "shot_map" in stem and side:
        return (
            f"{team} generated {int(context[f'{side}_shots'])} shots worth {context[f'{side}_xG']:.2f} xG, an average of {context[f'{side}_xG_per_shot']:.3f} per attempt. "
            f"The map should be judged by centrality, distance and repeatability, not marker count. {team}'s {context[f'{side}_box_entry_to_shot_rate']:.1f}% box-entry-to-shot rate shows how often penetration became an immediate attempt."
        )
    if "danger_creation" in stem and side:
        return (
            f"{team} recorded {int(context[f'{side}_final_third_entries'])} final-third entries, {int(context[f'{side}_deep_completions'])} deep completions and {int(context[f'{side}_box_entries'])} box entries. "
            "The useful pattern is the route repeated under pressure: central combinations, half-space receptions or wide deliveries. Threat is sustainable when the same route produces support around the receiver and a controlled next action."
        )
    if "zone14" in stem and side:
        return (
            f"Zone 14 access should be read as a platform for the next action, not an achievement by itself. {team}'s {int(context[f'{side}_deep_completions'])} deep completions indicate how often possession reached advanced central areas. "
            "The tactical question is whether the receiver could turn, combine or release a runner before the block collapsed."
        )
    if "crosses" in stem and side:
        return (
            f"{team} attempted {int(context[f'{side}_crosses'])} crosses and completed {int(context[f'{side}_completed_crosses'])}. "
            "Crossing effectiveness depends on the number and timing of box runners, the weak-side occupation and the structure for second balls. A failed delivery is also a transition event if the rest defence is not set behind it."
        )
    if "box_entries" in stem and side:
        return (
            f"{team} entered the box {_count(context[f'{side}_box_entries'], 'time', 'times')} but converted {context[f'{side}_box_entry_to_shot_rate']:.1f}% of those entries into shots. "
            "Separate controlled entries with support from isolated carries or forced passes. The main improvement target is the decision immediately after entry: shoot, cut back, recycle or secure the second phase."
        )
    if "pass_network" in stem and side:
        half = "first half" if "_1h" in stem else "second half"
        return (
            f"This {half} network describes {team}'s functional structure: the largest hubs show where possession repeatedly connected, while thick links reveal preferred routes. "
            f"Read it alongside {team}'s {context[f'{side}_pass_share']:.1f}% pass share and the score state. A dense link can represent control, but it can also reveal predictable circulation if it does not connect to advanced receivers."
        )
    if "average_positions" in stem and side:
        half = "first half" if "_1h" in stem else "second half"
        return (
            f"The {half} average positions show {team}'s occupation rather than a fixed formation. Check line height, full-back width, central staggering and the distance between the attacking line and the rest defence. "
            "The half split is essential because substitutions and score-state changes would otherwise merge different tactical structures into one misleading average."
        )
    if "xt_map" in stem and side:
        return (
            f"The square grid locates where {team}'s passes increased scoring potential. High-value cells matter most when they form a route - for example build-up to half-space, then half-space to box - rather than isolated bright zones. "
            f"Compare the heat with {int(context[f'{side}_box_entries'])} box entries and the shot map to test whether threat was converted into a final action."
        )
    if "pass_map" in stem and side:
        return (
            f"{team}'s pass map should be read in layers: circulation to stabilise possession, progressive actions to break lines and failed passes that exposed transition space. "
            f"The side played a {context[f'{side}_pass_share']:.1f}% share of match passes and recorded {int(context[f'{side}_progressive_passes'])} progressive passes. The location of risk is more informative than completion percentage alone."
        )
    if "ball_touches" in stem:
        return (
            f"Touch volume favoured {away} {int(context['away_touches'])}-{int(context['home_touches'])}, yet field tilt favoured {home} {context['home_field_tilt']:.1f}-{context['away_field_tilt']:.1f}. "
            "That contrast means total involvement and attacking-territory control were not the same. Use touch zones to locate where possession occurred before judging who controlled the match."
        )
    if "pass_thirds" in stem and side:
        return (
            f"The thirds view tests whether {team}'s possession moved through a stable chain or stalled between units. Link the distribution to {int(context[f'{side}_final_third_entries'])} final-third entries and "
            f"{context[f'{side}_final_third_entry_efficiency']:.1f}% entry efficiency. A high defensive-third share can reflect build-up control or an inability to progress, depending on the next line."
        )
    if "progressive" in stem and side:
        return (
            f"{team} completed {int(context[f'{side}_progressive_passes'])} progressive passes. Value comes from receiver context: a pass that breaks a line but leaves the receiver isolated is less useful than one that enables the next forward action. "
            "Track where progress ended and whether the team retained enough players behind the ball to control a turnover."
        )
    if "dominating_zones" in stem:
        return (
            f"Territorial dominance favoured {home} through a {context['home_field_tilt']:.1f}% field tilt, but the final score and xG favoured {away}. "
            "The map therefore describes where the match was played, not who used those zones better. The next layer is conversion: entries, shot quality and protection against the counter."
        )
    if "pass_targets" in stem and side:
        return (
            f"Destination density reveals where {team} tried to place the next receiver. Repeated targets can show deliberate occupation, but they can also expose predictable routes if the opponent can lock the receiver's next action. "
            f"Read the hot zones with {int(context[f'{side}_deep_completions'])} deep completions and the xT grid to distinguish occupation from genuine threat."
        )
    if "ppda" in stem:
        return (
            f"{home}'s PPDA of {context['home_ppda']:.2f} was lower than {away}'s {context['away_ppda']:.2f}, showing more aggressive pressure. "
            f"However, pressure quality must include the outcome: high regains were {int(context['home_high_regains'])}-{int(context['away_high_regains'])}, while rest-defence vulnerability was {context['home_rest_defence_vulnerability']:.1f}% versus {context['away_rest_defence_vulnerability']:.1f}%. "
            "Aggression without cover can win territory and still concede cleaner transitions."
        )
    if "high_regains" in stem and side:
        return (
            f"{team} produced {int(context[f'{side}_high_regains'])} high regains, but only {context[f'{side}_regain_to_shot_rate']:.1f}% became shots. "
            f"The regain sequences generated {context[f'{side}_regain_xG']:.2f} xG and {context[f'{side}_regain_xT']:.2f} xT. The decisive coaching detail is the first pass after recovery: forward if the opponent is open, secure if support is not yet connected."
        )
    if "defensive_activity" in stem and side:
        return (
            f"The action locations show where {team}'s block engaged: higher interventions indicate proactive pressure, while deeper clusters indicate protection close to goal. "
            f"Pair the map with {context[f'{side}_rest_defence_exposures']:.0f} rest-defence exposures and {context[f'{side}_rest_defence_dangerous_counters']:.0f} dangerous counters to judge whether challenges were supported by cover."
        )
    if "defensive_summary" in stem:
        return (
            f"The defensive comparison separates ball-winning volume from structural security. {home} allowed {int(context['home_rest_defence_dangerous_counters'])} dangerous counters and {away} allowed {int(context['away_rest_defence_dangerous_counters'])}; "
            "that difference matters more than raw tackle volume when evaluating the protection behind attacks."
        )
    if "transition_outcomes" in stem:
        return (
            f"{away} turned {_count(context['away_transitions'], 'transition', 'transitions')} into {_count(context['away_transition_shots'], 'shot', 'shots')} ({context['away_transition_shot_rate']:.1f}%) and {context['away_transition_xG']:.2f} xG. "
            f"{home} produced {_count(context['home_transition_shots'], 'shot', 'shots')} from {int(context['home_transitions'])} transitions ({context['home_transition_shot_rate']:.1f}%). "
            "The winner's edge was not simply more transitions, but faster conversion of disorder into a clean final action."
        )
    if "advanced_metrics" in stem:
        return (
            f"The four layers explain why one global control label would be misleading: {home} led sequence value ({context['home_sequence_xT']:.2f}) and field tilt, while {away} led shot efficiency and transition conversion. "
            "Volume describes how often a phase occurred; efficiency and risk explain whether repeating it helped the team win."
        )
    if "player_sequence" in stem:
        return (
            f"Sequence leaders identify players who connected valuable attacks before the final shot. {context['home_players']['chain']} led {home}'s xGChain profile, while {context['away_players']['chain']} led {away}'s. "
            "Use xGBuildup to recognise earlier involvement and sequence xT to identify players who repeatedly moved possession into more dangerous states."
        )
    return tactical_lens(path) + " The tactical conclusion should be checked against score state, opponent behaviour and the next phase of play."


def visual_explanation(path: Path, context: dict) -> str:
    """Explain the football mechanism first and use metrics only as supporting evidence."""
    stem = path.stem.lower()
    team, side = _visual_team(path, context)
    home, away = context["home"], context["away"]
    winner, loser = context["winner"], context["loser"]

    if "player_radars" in path.parts:
        player = path.stem.replace("_", " ")
        p = context.get("player_profiles", {}).get(player.lower(), {})
        if not p:
            return (
                f"This radar should be read as a map of {player}'s role in this match, not as a rating of the player's overall level. "
                "A wide segment only matters when it connects to the team's mechanism: receiving in a useful line, progressing under pressure, creating the next advantage or finishing the attack. "
                "The uneven shape is therefore more informative than the total area because it shows where the player entered the possession chain and where the contribution stopped."
            )
        if p["goals"] or p["xG"] >= 0.35:
            role_read = "The profile is weighted toward the final action, so the key question is how the team delivered the player into finishing positions rather than how often the player touched the ball."
        elif p["key_passes"] >= 2 or p["pass_xT"] >= 0.35:
            role_read = "The profile is weighted toward connection and chance creation: the player helped turn possession into a more dangerous next action rather than merely circulating it."
        elif p["xGBuildup"] >= p["xGChain"] * 0.55 and p["xGBuildup"] > 0:
            role_read = "The strongest contribution came earlier in the sequence, suggesting value in build-up support, line access and continuity before the final pass or shot."
        else:
            role_read = "The shape points to a supporting contribution inside collective sequences rather than dominance of one decisive action."
        # "the direct output was 1 goals, 1 shots and 1 key passes" — the
        # counts are almost always small, so the plural is wrong more often
        # than it is right on a player page.
        return (
            f"{role_read} {player} was involved in attacks worth {p['xGChain']:.2f} xGChain and {p['sequence_xT']:.2f} sequence xT, while the direct output was "
            f"{_count(p['goals'], 'goal', 'goals')}, {_count(p['shots'], 'shot', 'shots')} and "
            f"{_count(p['key_passes'], 'key pass', 'key passes')}. Those figures are evidence for the role, not the story by themselves. "
            f"Read the missing or smaller segments as boundaries of the match role: they may reflect position, minutes, the score state or the team's route of attack. The radar is most useful when traced back to the team pages that show where {_text(p.get('team'), 'the team')} created space for this contribution."
        )

    if "xg_flow" in stem:
        # Which curve finishes higher is a fact about the two totals, and this
        # sentence used to name the away side whatever they were.
        leader, trailer, level = _lead(
            home, away, context["home_xG"], context["away_xG"], tolerance=0.05)
        if level:
            curve_leader, curve_clause = "The two curves", " finished level"
        else:
            curve_leader = f"{leader}'s curve"
            curve_clause = f" finished above {trailer}'s"
        return (
            f"The match developed through separate bursts of danger rather than a smooth exchange of chances. The vertical steps show when an attack reached a genuine finishing situation; the flat stretches show periods in which possession did not materially improve the chance of scoring. "
            f"{curve_leader}{curve_clause}, but the larger tactical point is the timing of the jumps: once {loser} had to chase, attacks became more direct and the spaces between the pressing line and the defensive cover grew. "
            f"The final {context['score']} score came from {context['home_xG'] + context['away_xG']:.2f} combined xG, so finishing amplified the tactical advantages rather than simply mirroring the volume of chances."
        )
    if "goals_breakdown" in stem:
        first = context["goal_rows"][0] if context["goal_rows"] else None
        first_text = f"The opening goal {_goal_moment(first)} gave {first['team']} control over the risk level" if first else "The opening phase did not establish a stable score-state advantage"
        return (
            f"{first_text}. From that point, the trailing side had to push more players beyond the ball, shorten the time spent circulating and accept more direct attacks. That changed both teams at once: the chaser gained territory but weakened the distances protecting turnovers, while the leader could defend central space and wait for open-field moments. "
            "The scorer and assist labels identify the final action, but each goal should be read as the end of a chain involving the regain or progression route, the movement that displaced the last line and the final decision in the box. The order of the goals therefore explains why later full-match averages cannot be treated as neutral."
        )
    if "goalkeeper" in stem:
        # "Greatly exceeded the pre-shot expectation" was asserted over every
        # match, including three goals from 2.96 combined xG.
        scored = context["home_goals"] + context["away_goals"]
        expected = context["home_xG"] + context["away_xG"]
        # A 1-0 printed "1 goals came from 1.36 xG". _count already exists in
        # this module for exactly this and was not reached for here, which is
        # how the same defect the player pages were fixed for reappeared on the
        # goalkeeper board.
        goals = _count(scored, "goal", "goals")
        if scored > expected + 0.75:
            conversion = f"{goals} ran well ahead of the {expected:.2f} xG the chances were worth"
        elif scored < expected - 0.75:
            conversion = f"{goals} fell short of the {expected:.2f} xG the chances were worth"
        else:
            conversion = f"{goals} came from {expected:.2f} xG, close to par"
        return (
            "This page separates goalkeeper influence from the defensive workload in front of the goalkeeper. Save count alone can reward a keeper for facing several routine attempts, whereas post-shot quality asks how difficult the shots became after placement and power were known. "
            f"The two goalkeepers faced a match in which {conversion}, so the analysis must distinguish defensive access, finishing execution and actual shot-stopping. "
            "A concession is not automatically a goalkeeper error: central close-range shots usually point first to box protection, while goals from lower-quality positions place more weight on placement, visibility, reaction and starting position."
        )
    if "xg_summary" in stem or "shot_profile" in stem:
        return (
            "The attacks were separated by the quality of the final situation, not by how often each side shot. Equal shot volume can hide very different processes: one team may arrive centrally after moving the block, while the other shoots earlier because the route into the box has closed. "
            f"Here the average attempt was more valuable for {away}, which means the decisive advantage occurred before the strike - in the entry route, receiver support and the defender's distance from the shooter. "
            "Accuracy then describes execution of those situations, but it should not be confused with repeatability. The most transferable lesson is which attacking structure repeatedly created an uncontested or well-supported final action."
        )
    if "match_stats" in stem:
        return (
            f"The overview reveals two different kinds of control. {home} spent more of the match in advanced territory and generated repeated access, yet {away} converted its attacks into the stronger chance profile and the winning score. "
            "That is not a contradiction: territorial control describes where possession was established, while attacking control describes what happened after the defensive block was engaged. The losing side's activity may have forced the opponent deeper without consistently moving the last line or creating a free receiver in the box. "
            "Treat this page as the tactical problem statement; the following visuals isolate whether the separation came from timing, occupation, progression, pressing or transition protection."
        )
    if "post_match_advanced" in stem:
        # Which side held territory and which produced the cleaner shot has to
        # come from the values. Naming home as the territorial side and away as
        # the efficient one was fixed text, so in any match where that was the
        # other way round the paragraph contradicted its own numbers.
        territory = home if context["home_field_tilt"] >= context["away_field_tilt"] else away
        quality = home if context["home_xG_per_shot"] >= context["away_xG_per_shot"] else away
        secure = (
            home
            if context["home_rest_defence_vulnerability"] <= context["away_rest_defence_vulnerability"]
            else away
        )
        return (
            f"This dashboard joins the two sides of the same tactical plan. {territory} controlled more advanced territory and reached the final third more often, but that attacking commitment also left more demanding rest-defence situations. "
            f"{quality} created the cleaner average shot, and {secure} protected its attacking possessions more securely. The useful interpretation is therefore not attack versus defence as separate departments: spacing during possession determined both the quality of the next attack and the security of the next defensive action."
        )
    if "xt_per_minute" in stem:
        return (
            "The threat timeline identifies the moments when possession changed the defensive problem rather than merely changing location. Sharp peaks normally reflect a line-breaking pass, a reception facing goal or an action that forced the back line to retreat; quiet periods indicate that the block could shift without being disorganised. "
            f"The important reading is the clustering of peaks around score changes. When {loser} increased urgency, the match produced more open possessions, but more threat did not automatically mean more control because the same attacking commitment enlarged the space available after turnovers. "
            "This visual should therefore be read as a momentum map: it locates the windows to review, while the phase maps explain the mechanism inside those windows."
        )
    if "game_state" in stem:
        return (
            f"The score divided the match into unequal tactical conditions. While leading, {winner} could prioritise central compactness, choose selective pressing moments and attack the space left by an opponent that needed the next goal. While trailing, {loser} had to increase the height and number of supporting players, which raised attacking presence but reduced security behind the ball. "
            "This is why possession, field tilt and pressure totals can rise for the losing team without proving superior control. They partly describe necessity. The useful comparison is how each side behaved under the same state - level, leading or trailing - and whether its structure remained capable of creating chances without giving away immediate transition access."
        )
    if "shot_map" in stem and side:
        quality = "relatively strong" if context[f"{side}_xG_per_shot"] >= 0.12 else "relatively modest"
        return (
            f"{team}'s shot locations show the endpoint of its attacking choices. Central attempts close to goal usually indicate that the team moved or pinned the last line before the finish; wider and longer attempts often indicate that penetration stopped and the shooter accepted the remaining option. "
            f"The average chance quality was {quality}, but the tactical value lies in whether the high-value locations came from a repeatable route - cut-backs, central combinations, runs behind or second balls - rather than one isolated event. "
            f"The side turned {context[f'{side}_box_entry_to_shot_rate']:.1f}% of box entries into shots, so the map also tests decision-making after entry: secure the receiver, create the extra pass and finish before the block can collapse."
        )
    if "danger_creation" in stem and side:
        return (
            f"This visual traces how {team} moved from access to actual danger. An entry only becomes tactically useful when the receiver has a forward body position, nearby support and a next action that forces a defender to leave the line. Repeated activity on the outside can move the block without breaking it; repeated half-space or central access is more likely to create a decision between stepping out and protecting the box. "
            f"{team}'s entries, deep completions and box arrivals should therefore be read as a funnel, not three independent totals. The drop between stages shows where attacks lost clarity - before the final line, at the first touch near the box or in the selection of the final pass."
        )
    if "zone14" in stem and side:
        return (
            f"Zone 14 is valuable because a receiver there can threaten both sides of the defensive line: shoot, combine through the centre or release a runner into either channel. The visual should be read for the conditions of the reception, not only the number of actions. "
            f"For {team}, central access was most useful when the receiver arrived between midfield and defence with the body open and a third-player run already moving the last line. If the receiver was closed from behind or received square, the opponent could compress the area and force play back outside. "
            "The key mechanism is therefore occupation around the receiver - depth ahead, width outside and protection behind - because the zone does not create danger on its own."
        )
    if "crosses" in stem and side:
        return (
            f"The crossing map describes the final choice after wide progression. A cross is structurally strong when the near-post run pins the closest centre-back, a second runner attacks the central or far-post lane and another player protects the edge for the clearance. Without those layers, delivery volume mainly returns the ball to the opponent. "
            f"For {team}, the useful question is not how many balls entered the area but whether the timing of the delivery matched the arrival of the runners and whether the weak side stayed occupied. Every blocked or cleared cross also begins a defensive phase, so the positions behind the ball are part of the attacking evaluation."
        )
    if "box_entries" in stem and side:
        return (
            f"The entry map distinguishes reaching the penalty area from controlling the action inside it. A carry or pass into the box can still be low value if the receiver is isolated, facing away from goal or immediately surrounded. The strongest entries arrive behind the full-back, between centre-back and full-back, or into a cut-back lane after the defence has been turned. "
            f"{team}'s conversion of entries into shots shows how often penetration survived the first contact. Where it did not, the likely issue was the next decision: forcing the shot, delaying until the lane closed, or lacking a second runner. The coaching focus is coordinated arrival, not entry volume."
        )
    if "pass_network" in stem and side:
        half = "first half" if "_1h" in stem else "second half"
        return (
            f"The {half} network is a picture of {team}'s functional relationships, not a formation diagram. Thick connections reveal the routes the opponent had to manage repeatedly; large hubs reveal where circulation depended on one player or zone. "
            "The main tactical test is whether the network contains vertical and diagonal links between units. Dense horizontal links can stabilise possession but also allow the block to slide without being penetrated, while a connection into a player between the lines forces a defender to step and opens the next space. "
            "Compare the two halves for changes in height, width and central access: substitutions and score state can turn the same nominal shape into a different possession structure."
        )
    if "average_positions" in stem and side:
        half = "first half" if "_1h" in stem else "second half"
        return (
            f"The {half} average positions show the spaces {team} occupied across many possessions; they should not be mistaken for fixed starting locations. The distances between the points are the key evidence. Good attacking spacing creates width, depth and at least one player between lines without disconnecting the players who must protect a turnover. "
            "If the front line is too flat, the ball carrier sees few diagonal options. If both full-backs advance without a stable central screen, the team may gain width but lose control of the first counter pass. "
            "Read this page with the network: position shows where options existed, while the links show whether the ball actually reached them."
        )
    if "xt_map" in stem and side:
        return (
            f"The heatmap locates where {team} increased the probability of creating a future chance through ball movement. The bright cells are most meaningful when they connect into a route across several zones: an exit from pressure, a reception between lines, then an action into or across the box. A single hot square can come from one exceptional pass; a connected corridor is closer to a repeatable mechanism. "
            "The map also reveals where progression stopped. Threat concentrated outside the block may indicate useful territory without central access, whereas value close to the box suggests the team was forcing defenders to turn and protect goal. The shot and entry pages test whether that threat survived into the final action."
        )
    if "pass_map" in stem and side:
        return (
            f"{team}'s pass map should be read as a sequence of tactical functions. Some passes attract pressure and stabilise the build-up; others eliminate a line; the final group attacks the space created by the earlier actions. Judging all of them by completion rate would hide this difference. "
            "The location of failed passes is equally important. A failed vertical pass with compact support may be an acceptable attacking risk, while a square loss with both full-backs high can immediately expose the centre. "
            "The map therefore explains both progression and transition vulnerability: where the team chose to accelerate, whether the receiver had continuity, and whether the rest defence was prepared for failure."
        )
    if "ball_touches" in stem:
        return (
            "The touch distribution separates involvement from territorial purpose. A team can record more touches because it circulates across the first two lines, while the opponent records fewer touches but receives more often in spaces that threaten the last line. "
            f"In this match, total involvement and field tilt pointed in different directions, showing that the sides controlled different layers of the game. The useful reading is where touches accumulated: deep build-up may indicate calm control or pressure confinement; repeated touches around the box may indicate sustained attack or an inability to find the final lane. "
            "The next pages break the pitch into phases to identify where possession actually accelerated or stalled."
        )
    if "pass_thirds" in stem and side:
        return (
            f"This view tests the continuity of {team}'s possession chain. A healthy build-up does not simply complete passes in each third; it moves the opponent, creates a free player on the next line and preserves support after the pass. A large defensive-third share can mean controlled construction, but it can also mean the first pressing line repeatedly forced play back. "
            "The tactical bottleneck is the point where the distribution changes: difficulty leaving the first third suggests an exit problem, while repeated arrival in the middle third without final-third continuity suggests insufficient positioning between lines. "
            "Entry efficiency is evidence of whether the structure converted circulation into access, not a substitute for understanding how."
        )
    if "progressive" in stem and side:
        return (
            f"The progressive-pass map isolates actions that moved {team} materially closer to goal, but distance gained is only the first layer. A progressive pass becomes tactically powerful when it breaks an opponent line and delivers the receiver in a position to continue forward before pressure arrives. "
            "The best routes are often diagonal because they change both the vertical and horizontal problem for the block. Repeated straight passes into a marked receiver may register as progress but end with a bounce pass or turnover. "
            "Read the endpoints and surrounding support: the map should reveal who received on the far side of pressure, who provided the third-man option and whether the team remained protected if the action failed."
        )
    if "dominating_zones" in stem:
        # Named home as the territorial side and away as the efficient one
        # whatever the figures said, so in this fixture it credited Arsenal
        # with the stronger field tilt on 29.2% against Manchester City's 70.8.
        zone_territory = home if context["home_field_tilt"] >= context["away_field_tilt"] else away
        zone_efficient = home if context["home_xG_per_shot"] >= context["away_xG_per_shot"] else away
        # In a match where neither side led, "X's stronger field tilt ... yet X
        # used its possessions more efficiently" named the same team twice over
        # a difference that was not there.
        tilt_level = abs(context["home_field_tilt"] - context["away_field_tilt"]) <= 2.0
        quality_level = abs(context["home_xG_per_shot"] - context["away_xG_per_shot"]) <= 0.01
        if tilt_level and quality_level:
            opening = (
                "The zone map shows where each team established more sustained influence, but "
                "territory is a platform rather than an outcome. Neither side owned the ground "
                "and neither created the better attempt, so the map is a record of two teams "
                "occupying the pitch evenly rather than of one imposing itself. ")
        elif tilt_level:
            opening = (
                f"The zone map shows where each team established more sustained influence, but "
                f"territory is a platform rather than an outcome. The ground was shared, yet "
                f"{zone_efficient} used its dangerous possessions more efficiently, so the "
                f"difference was made inside the areas both sides reached. ")
        elif quality_level:
            opening = (
                f"The zone map shows where each team established more sustained influence, but "
                f"territory is a platform rather than an outcome. {zone_territory}'s stronger "
                f"field tilt meant more play was located near the attacking end, and yet both "
                f"sides arrived at the same quality of attempt. ")
        else:
            opening = (
                f"The zone map shows where each team established more sustained influence, but territory is a platform rather than an outcome. {zone_territory}'s stronger field tilt meant more play was located near the attacking end, yet {zone_efficient} used its dangerous possessions more efficiently. ")
        return (
            opening +
            "This can happen when the deeper side protects central space, encourages circulation toward the touchline and attacks the first open pass after recovery. The territorial side then appears dominant while repeatedly restarting outside the block. "
            "The tactical judgement must join zone control to the next action: did dominance create a free receiver, a box entry and a shot, or did it increase the number of players ahead of the ball without improving the final situation?"
        )
    if "pass_targets" in stem and side:
        return (
            f"The target map shows where {team} wanted the next receiver to appear. Repeated destinations expose the occupation rules of the attack: width to stretch the line, half-space presence to connect units, or a player between lines acting as the third-man platform. "
            "A hot zone is not automatically a successful route. If the receiver repeatedly receives with the back to goal or without a runner beyond, the opponent can allow the pass and lock the next action. The most valuable destinations are those that change the defender's orientation and give the receiver at least two forward options. "
            "Compare the targets with the xT map to see whether occupation actually changed the threat level."
        )
    if "ppda" in stem:
        press_team = home if context["home_ppda"] < context["away_ppda"] else away
        return (
            f"The PPDA comparison indicates that {press_team} allowed fewer opponent passes before engaging, but intensity is not the same as pressing control. A coherent press needs a trigger, pressure on the ball, cover of the nearest inside option and a back line ready to compress the space behind. If one layer arrives late, the opponent can use the first free pass to attack an exposed defence. "
            "The visual should therefore be read as the height and frequency of the intention to press. High regains show whether that intention won the ball in useful areas; dangerous counters show whether the structure survived when the press was broken. The best press reduces both opponent progression and the team's own transition risk."
        )
    if "high_regains" in stem and side:
        return (
            f"The regain map shows where {team}'s pressure actually recovered possession, but the decisive phase begins one action later. Immediately after a high regain, the opponent is narrow around the lost ball and its back line may be unbalanced; the recovering team has a short window to play forward before the block resets. "
            "A successful sequence therefore needs the first receiver to scan before the regain, a forward option beyond the ball and support for a cut-back or second action. If those options are absent, securing possession may be better than forcing a low-control pass. "
            "The gap between regains and shots diagnoses whether the pressing structure was connected to an attacking structure."
        )
    if "defensive_activity" in stem and side:
        return (
            f"The action locations reveal the height and posture of {team}'s defending. Interventions higher up the pitch suggest an attempt to prevent progression early; deeper clusters suggest that the block prioritised box protection and accepted territory. Neither is automatically better. The question is whether the action was supported by the next defender and whether the space behind the challenge remained protected. "
            "A tackle near the touchline can be a successful pressing trap if the inside lane is closed. The same tackle becomes risky if a missed challenge opens a central transition. "
            "Read the density and location together with the rest-defence exposures to distinguish proactive control from emergency defending."
        )
    if "defensive_summary" in stem:
        return (
            "This comparison separates visible defensive work from the quality of the defensive structure. High tackle or interception volume can mean aggression, but it can also mean the opponent repeatedly reached areas that required intervention. Structural security is better reflected by the distances between the press, the midfield screen and the last line, plus the quality of attacks conceded after the first duel was lost. "
            f"The difference in dangerous counters conceded shows which side protected its attacks more reliably. In a high-scoring match, the central question is not who defended more often, but who forced the opponent into predictable, supported duels and who was repeatedly left defending open space toward goal."
        )
    if "transition_outcomes" in stem:
        transition_team = home if context["home_transition_shot_rate"] > context["away_transition_shot_rate"] else away
        return (
            f"The transition page shows how each team used the few seconds before the opponent restored its shape. {transition_team} was more effective at turning those unstable moments into shots, which points to faster recognition of the first forward option and better running beyond the ball. "
            "A strong transition is not simply a fast attack. It creates numerical or positional superiority with the first two actions: the ball carrier fixes a defender, one runner threatens depth and another offers a safer continuation. "
            f"The match state gave {winner} more opportunities to attack an opponent committing players forward, but the efficiency still depended on spacing and decision speed. The defensive lesson is to organise protection before the turnover occurs, not after the counter has started."
        )
    if "advanced_metrics" in stem:
        return (
            f"The combined metrics explain why the match cannot be reduced to one claim of dominance. {home} controlled more territory and accumulated stronger sequence value, while {away} extracted more from the phases closest to goal and from moments of transition. "
            "Volume tells us which behaviours occurred often; efficiency tells us whether those behaviours advanced the tactical objective; value tells us how much the action improved the chance of scoring; risk tells us what the team exposed while doing it. A side can therefore look superior in possession and still lose the decisive exchange. "
            "The correct conclusion is phase-specific: identify which structure created repeatable advantages, which relied on execution and which carried an unsustainable defensive cost."
        )
    if "player_sequence" in stem:
        return (
            "The sequence leaders move the analysis away from only crediting the final pass and shot. Valuable attacks are usually collective chains: one player attracts pressure, another receives beyond it, a third connects the action and the final player converts the chance. xGChain captures involvement across the chance sequence, while build-up and sequence threat help identify contributions made earlier. "
            f"For {home} and {away}, the leading names should be traced back to the team structure: did they receive because of deliberate occupation, did they carry the ball through pressure, or did their value come from repeatedly choosing the correct next action? "
            "The following radars break those collective mechanisms into role-specific match profiles without treating unlike positions as directly comparable."
        )
    # ---- boards that had no branch at all ------------------------------
    # Fifteen visuals per match reached the generic ending below because
    # nothing above matched their stem. Each now gets the same treatment as
    # the rest: the mechanism first, this match's numbers as evidence.
    if "match_momentum" in stem:
        swing, chased, level = _lead(home, away, context["home_xG"], context["away_xG"])
        return (
            "Momentum here is expected-goal difference inside five-minute windows, which is a "
            "deliberately short lens: it asks who was creating in each passage, not who ended "
            "the night ahead. A block of bars on one side is a period the opponent could not "
            "settle, and the interesting question is always what changed at its edges - a "
            "substitution, a goal, a press that started arriving earlier. "
            + ("Neither side owned the windows for long, which is the signature of a match "
               "decided by single actions rather than by a passage of control. "
               if level else
               f"{swing} finished on {context['home_xG' if swing == home else 'away_xG']:.2f} "
               f"against {chased}'s "
               f"{context['home_xG' if chased == home else 'away_xG']:.2f}, but the totals hide "
               f"when that gap was built; the bars locate it. ")
            + "Read the windows against the goal timeline: a side that dominates the ten minutes "
            "after conceding is reacting, and a side that dominates the ten minutes before "
            "scoring built something. Only the second is a mechanism worth repeating."
        )
    if "win_probability" in stem:
        return (
            f"This curve converts the chances as they arrived into the likelihood of each result, "
            f"so it measures what a spectator could reasonably have believed at each point rather "
            f"than what the final {context['score']} makes it look like in hindsight. Steep moves "
            f"belong to goals and clear chances; the long flat stretches are where the match was "
            f"being played without being decided. "
            f"The tactical value is in the shape, not the endpoint. A curve that settles early "
            f"says the losing side never assembled a sustained response and that the leader was "
            f"able to manage rather than defend; a curve that keeps moving says the result stayed "
            f"available to both. "
            f"Read it beside the game-state pages: once the probability stops moving, both teams "
            f"are playing a different match from the one the pre-match plan described, and every "
            f"later total is coloured by that."
        )
    if "sequence_types" in stem:
        return (
            "Every attack is classified by how the possession began - built from settled "
            "possession, launched from a turnover, or restarted from a set piece - because those "
            "three routes ask completely different things of a defence and are coached "
            "separately. A side whose danger is concentrated in one column has one way of hurting "
            "an opponent, which is easier to plan against than a spread. "
            f"Set that against the totals: {home} took {context['home_transition_xG']:.2f} "
            f"expected goals from broken play and {away} {context['away_transition_xG']:.2f}, "
            f"from {int(context['home_transitions'])} and "
            f"{int(context['away_transitions'])} transitions. "
            "The reading to avoid is treating one route as inherently better. Sustained "
            "possession that produces nothing is not superior to three transitions that produce "
            "a goal; what matters is whether the route was chosen or forced, and whether the "
            "side had a second one available when the first was closed."
        )
    if "goal_origins" in stem:
        # goal_timeline is a pipe-delimited machine string. Dropping it into a
        # paragraph put "0' Calafiori (Arsenal) | 27' Havertz (Arsenal)" in the
        # middle of a sentence in a document meant to be read.
        told = [f"{row['player']} {_goal_moment(row)} for {row['team']}"
                if row["player"] else f"{row['team']} {_goal_moment(row)}"
                for row in context["goal_rows"]]
        if len(told) > 1:
            listed = ", ".join(told[:-1]) + f", then {told[-1]}"
        else:
            listed = told[0] if told else ""
        origins = f"The goals came through {listed}. " if listed else ""
        return (
            f"Each goal is traced back to the moment its possession started, which usually sits "
            f"further from the finish than the highlight suggests. The origin identifies the "
            f"action that actually created the advantage - a regain in a useful area, an exit "
            f"that beat the first pressure, a restart - and that is the part a team can rehearse. "
            f"{origins}"
            f"Read origin and finish together. Goals beginning deep in a team's own half point to "
            f"progression that survived several lines and normally to an opponent caught with too "
            f"many players ahead of the ball; goals beginning high point to pressing structure "
            f"rather than possession structure. The two demand different work in the week."
        )
    if "pitch_control" in stem:
        territory = home if context["home_field_tilt"] >= context["away_field_tilt"] else away
        other = away if territory == home else home
        return (
            f"The surface models which side would reach a loose ball first across the whole "
            f"pitch, weighted by distance, so it describes space occupied rather than passes "
            f"completed. It answers a question the possession figure cannot: not who had the "
            f"ball, but who would have had it. "
            f"{territory} held the larger share of advanced territory here "
            f"({context['home_field_tilt' if territory == home else 'away_field_tilt']:.1f}% "
            f"field tilt against "
            f"{context['home_field_tilt' if other == home else 'away_field_tilt']:.1f}%), and the "
            f"map shows where that control was real and where it was conceded on purpose. "
            f"A settled block gives up the areas in front of it deliberately; the diagnostic is "
            f"whether the controlled zones touch the penalty area or stop at its edge. Territory "
            f"that ends twenty metres from goal is a platform the opponent is content to allow."
        )
    if "set_pieces" in stem:
        return (
            "Restarts are the one phase where both teams get to arrange themselves in advance, "
            "which makes them the most coachable source of chances on this page and the least "
            "excusable source of concessions. The map should be read for repetition: the same "
            "delivery to the same zone twice is a rehearsed routine, and the second attempt tells "
            "you whether the opponent adjusted. "
            f"With {int(context['home_crosses'])} crosses from {home} and "
            f"{int(context['away_crosses'])} from {away} in open play, the restarts have to be "
            f"judged separately - a team can be poor from the run of play and still own the "
            f"restarts, and the corrections are unrelated. "
            "The defensive reading is the second ball as much as the first contact. Most damage "
            "from a corner arrives after the initial header, from an edge-of-box position that "
            "was left unoccupied because everyone was marking inside."
        )
    if "ball_losses" in stem and side:
        exposure = context[f"{side}_rest_defence_vulnerability"]
        counters = int(context[f"{side}_rest_defence_dangerous_counters"])
        return (
            f"Not every turnover matters, and this map separates the ones that did. A loss is "
            f"expensive when it happens with players committed ahead of the ball and the opponent "
            f"facing forward; the same loss in a settled shape costs nothing but possession. "
            f"{team} were punished on {exposure:.1f}% of their advanced losses, conceding "
            f"{counters} dangerous counter{'s' if counters != 1 else ''}. "
            f"That figure is a property of the shape before the loss, not of the player who lost "
            f"it. Read the clusters for their height and their width: losses on the far side from "
            f"the covering midfielder are the ones that become counters, because the recovery run "
            f"has to cross the pitch before it can start. "
            f"The correction is in the possession phase - who holds the inside lane while the "
            f"ball is wide - rather than in the instruction to lose the ball less often."
        )
    if "defensive_shape" in stem:
        return (
            f"This compares the shape each side held without the ball: how high the first line "
            f"engaged, how much distance sat between the units, and whether the block stayed "
            f"connected as the ball moved across it. "
            f"{home} allowed {context['home_ppda']:.2f} opponent passes per defensive action and "
            f"{away} {context['away_ppda']:.2f}, so one side was engaging materially earlier than "
            f"the other. Height is a choice with a cost attached, and this page shows what was "
            f"bought with it. "
            f"The test is compactness rather than height. A high line with thirty metres to the "
            f"midfield is not pressing, it is two separate teams; a deep block with the same gap "
            f"is not protecting the box either. Judge the distance between the lines first, then "
            f"ask whether the space that shape conceded was the space the opponent wanted."
        )
    if "playing_through" in stem and side:
        return (
            f"These are the passes that eliminated a defensive line rather than moving around it, "
            f"which is the distinction between progress and territory. A block slides "
            f"comfortably against circulation; it has to break its own structure when a pass "
            f"arrives behind one of its lines. "
            f"{team} completed {_count(context[f'{side}_deep_completions'], 'pass', 'passes')} into the deep "
            f"attacking zone from {int(context[f'{side}_progressive_passes'])} progressive "
            f"passes, which is the ratio worth watching - it separates a team that moves the ball "
            f"forward from one that moves it through. "
            f"Read the receptions, not the passes. A line-breaking pass to a player facing his "
            f"own goal with a defender on his back has not broken anything: the defence steps, "
            f"the ball comes back, and the shape is intact. The ones that count leave the "
            f"receiver able to turn, which is a function of when the pass was played more than "
            f"where it went."
        )
    if "unlocking" in stem and side:
        return (
            f"This isolates receptions in the pocket ahead of the defensive line - the position "
            f"from which a player can shoot, slide a runner in behind, or force a centre-back to "
            f"step out and open the space he was occupying. It is the single most valuable place "
            f"to receive and the hardest to occupy repeatedly. "
            f"{team} reached the final third "
            f"{_count(context[f'{side}_final_third_entries'], 'time', 'times')} "
            f"and the penalty area {int(context[f'{side}_box_entries'])}; these receptions are "
            f"most of the explanation for the gap between those two numbers. "
            f"What makes the reception work is what is happening around it. Without a runner "
            f"threatening depth, the defender can simply step and press the receiver with no risk "
            f"behind him, and the pocket stops existing. The occupation of the pocket and the run "
            f"beyond it are one coordinated action, not two."
        )
    if "press_triggers" in stem:
        presser = home if context["home_high_regains"] >= context["away_high_regains"] else away
        return (
            f"A press is a set of conditions, not an effort level. This page asks what the "
            f"opponent was doing at the moment the ball was won high - receiving with the back to "
            f"goal, taking a touch too many, playing into a covered lane - because those are the "
            f"cues a team actually trains, and a regain without a cue behind it is an accident "
            f"rather than a mechanism. "
            f"{presser} won the ball in the opponent's territory "
            f"{int(context['home_high_regains' if presser == home else 'away_high_regains'])} "
            f"times, converting "
            f"{context['home_regain_to_shot_rate' if presser == home else 'away_regain_to_shot_rate']:.1f}% "
            f"of all regains into a shot. "
            f"Read the triggers against the rest-defence page. A press that wins the ball on a "
            f"predictable cue leaves the defence arranged for the moment it fails; a press that "
            f"wins it from individual pursuit does not, and its cost shows up as counters "
            f"conceded rather than as a lower regain count."
        )
    if "action_value" in stem:
        return (
            "Every action on the pitch is priced by how much it changed the probability of a goal "
            "at either end, which puts a recovery in a dangerous area and a pass that creates a "
            "shot on the same scale. It is the most honest single view of contribution and the "
            "easiest to over-read. "
            "The caution is sample size. One high-value action can outweigh eighty ordinary ones "
            "over ninety minutes, so the ranking says who had the largest moments, not who played "
            "best. A defender whose whole match was preventing situations from arising scores "
            "near zero here by construction. "
            "Use it to locate passages worth reviewing rather than to rank players. The question "
            "the page answers well is where the value in this match was concentrated; the "
            "question it answers badly is who deserves credit for it."
        )
    return (
        _legacy_visual_explanation(path, context)
        + " The deeper reading is the causal chain behind the pattern: opponent behaviour created a space, the team occupied or missed that space, and the next action either preserved the advantage or returned control."
    )


def visual_implication(path: Path, context: dict) -> str:
    stem = path.stem.lower()
    team, _ = _visual_team(path, context)
    if "player_radars" in path.parts:
        return "Use the profile to define the player's match function and the support that function required. Do not convert one-match shape into a general ability ranking; compare the player with the tactical demands of the role."
    rules = [
        ("xg_flow", "The coaching focus is to reproduce the possessions that generated the large steps and remove the structural conditions behind the opponent's steps. The curve identifies when to review; it does not explain the mechanism alone."),
        ("goals_breakdown", "The score state must frame every later conclusion. Separate behaviours chosen by design from behaviours forced by chasing the match."),
        ("shot_map", "Improve the route into the shot, not merely the instruction to shoot more. Review body orientation, defender distance, support around the receiver and the availability of the extra pass."),
        ("shot_profile", "Finishing may vary from match to match; the entry structure is more coachable. Protect and repeat the routes that created the cleanest attempts."),
        ("xg_summary", "Use xG as a quality check, not a verdict on performance. The next task is locating the attacking behaviours that created or prevented high-value shots."),
        ("goalkeeper", "Assign responsibility across the whole defensive chain: pressure on the shot, protection of the central lane, visibility and the goalkeeper's intervention."),
        ("match_stats", "Do not declare control from one total. Build the match conclusion by joining territory, chance quality, transition security and score state."),
        ("post_match_advanced", "Use the dashboard as the publishable match verdict: preserve the attacking behaviours that created access, but coach the occupation behind the ball that determines whether those attacks are sustainable."),
        ("xt_per_minute", "Review the peak windows on video and identify the repeated trigger: regain, overload, line-breaking reception or game-state change."),
        ("game_state", "Evaluate the game plan by state. The same possession share can mean patient control while level and harmless circulation while chasing."),
        ("danger_creation", "Coach the connection between entry and continuation: the receiver needs support ahead, beside and behind the ball before the defence collapses."),
        ("zone14", "Central occupation needs coordinated depth and width. Arriving in the zone without a runner beyond or a secure rest defence usually produces recycling, not penetration."),
        ("crosses", "Set rules for runner lanes, delivery timing and second-ball protection. Cross selection and rest defence belong to the same tactical action."),
        ("box_entries", "Judge success by the quality of the next action. Create entries that allow the receiver to face goal or find a cut-back, rather than simply counting penalty-area touches."),
        ("pass_network", "Strengthen links that connect units and reduce dependence on safe horizontal hubs. The opponent should be forced to defend more than one progression route."),
        ("average_positions", "Adjust distances before adjusting names. The team needs attacking occupation and turnover protection at the same time."),
        ("xt_map", "Turn isolated hot cells into a repeatable corridor of progression. The valuable route should continue into controlled box access and a shot."),
        ("pass_map", "Define acceptable risk by zone and support. Losing an attacking pass can be manageable; losing a square pass with poor cover can decide the match."),
        ("ball_touches", "Move the interpretation from how much the ball was used to where and why it was used. Occupation and next-action quality determine whether touches become control."),
        ("pass_thirds", "Target the transition between units where the chain breaks. The solution may be positioning, body orientation, a third-man option or earlier width rather than simply faster passing."),
        ("progressive", "Progress should leave the receiver able to continue. Coach the pass, the receiving angle and the supporting run as one action."),
        ("dominating_zones", "Territory must be converted without weakening protection behind the ball. Otherwise dominance can increase exposure while leaving shot quality unchanged."),
        ("pass_targets", "Create destinations that give the receiver two forward options. Repeating a marked target only makes the possession predictable."),
        ("ppda", "Link every pressing trigger to cover and back-line compression. More aggression is useful only if it reduces opponent access without increasing open-field exposure."),
        ("high_regains", "Rehearse the first three seconds after recovery: scan, secure a forward option and decide whether to attack immediately or stabilise possession."),
        ("defensive_activity", "Evaluate the support behind each intervention. The objective is to force predictable duels, not simply accumulate defensive actions."),
        ("defensive_summary", "Improve the connection between attack and defence. Rest-defence positioning should be established while the team has the ball."),
        ("transition_outcomes", "The attack and the protection behind it must be coached together. Good transition defence begins with occupation before possession is lost."),
        ("advanced_metrics", "Keep the verdict phase-specific. Preserve repeatable structural advantages, treat finishing as volatile and prioritise the risks that gave the opponent direct access to goal."),
        ("player_sequence", "Use the leaders to assign tactical responsibility: who initiates, who connects, who advances and who finishes. The team mechanism matters more than a flat player ranking."),
        # The fifteen boards that had no rule here all returned the same
        # closing line, so a third of the report ended on one sentence.
        ("match_momentum", "Take the two or three windows with the largest swing into the video session. The bars say when to look; only the footage says whether a change of shape, a substitution or one player's decision opened the passage."),
        ("win_probability", "Judge the plan by where the curve moved, not by where it finished. A result that was settled by the hour asks a different question of the losing side than one that stayed open."),
        ("sequence_types", "Make sure the side has a second route to danger. A team that creates only from settled possession can be shut down by a deep block; a team that creates only from turnovers needs the opponent to make a mistake."),
        ("goal_origins", "Rehearse the origin, not the finish. The action that created the advantage sits several passes before the shot and is the part that can be repeated on purpose."),
        ("pitch_control", "Ask whether the controlled space touched the box. Territory that stops at the edge of the area is a platform the opponent is content to concede, and holding more of it changes nothing on its own."),
        ("set_pieces", "Treat restarts as their own training block with their own personnel. They are the only phase where both teams arrange themselves in advance, and the second ball decides more of them than the first contact."),
        ("ball_losses", "Fix the shape before the loss rather than the loss itself. Who holds the inside lane while the ball is wide determines whether a turnover costs possession or a chance."),
        ("defensive_shape", "Coach the distance between the units before the height of the first line. A high press with a disconnected midfield concedes more than a deep block, and both are visible here as spacing rather than as effort."),
        ("playing_through", "Judge a line-breaking pass by the reception it produced. A pass behind the line to a player facing his own goal has not broken anything; timing matters more than direction."),
        ("unlocking", "Coach the occupation of the pocket and the run beyond it as one action. Without a runner threatening depth, the defender can step onto the receiver at no risk and the pocket stops existing."),
        ("press_triggers", "Define the cue, not the intensity. A regain won on a trained trigger leaves the defence arranged for the moment the press fails; one won on individual pursuit does not, and the cost appears as counters conceded."),
        ("action_value", "Use this to find the passages worth reviewing, not to rank the squad. A defender who prevented situations from arising scores near zero here by construction."),
    ]
    for token, implication in rules:
        if token in stem:
            return implication if not team else f"For {team}, {implication[0].lower() + implication[1:]}"
    return "Translate the pattern into a coachable behaviour: define the space, the trigger, the supporting positions and the safe response if the action fails."


def visual_commentary_title(path: Path, context: dict) -> str:
    stem = path.stem.lower()
    team, _ = _visual_team(path, context)
    if "player_radars" in path.parts:
        return f"How {path.stem.replace('_', ' ')} influenced the match"
    titles = [
        ("xg_flow", "How the danger actually accumulated"),
        ("goals_breakdown", "How the score changed the tactical problem"),
        ("shot_map", f"How {team or 'the team'} manufactured its shots"),
        ("shot_profile", "Why equal shot volume produced unequal danger"),
        ("xg_summary", "What separated chance volume from chance quality"),
        ("goalkeeper", "How much did shot-stopping shape the result?"),
        ("match_stats", "Two different kinds of control shaped the match"),
        ("post_match_advanced", "How attack and defence combined to shape the result"),
        ("xt_per_minute", "Where possession became genuine threat"),
        ("game_state", "Why the score changed the meaning of every total"),
        ("danger_creation", f"How {team or 'the team'} turned territory into openings"),
        ("zone14", f"How {team or 'the team'} used the space in front of the box"),
        ("crosses", f"How {team or 'the team'} attacked from wide areas"),
        ("box_entries", f"How {team or 'the team'} converted access into box control"),
        ("pass_network", f"How {team or 'the team'} built and circulated"),
        ("average_positions", f"What attacking shape did {team or 'the team'} actually hold?"),
        ("xt_map", f"Where {team or 'the team'} moved the ball into danger"),
        ("pass_map", f"How {team or 'the team'} balanced progression and risk"),
        ("ball_touches", "Where involvement became territorial influence"),
        ("pass_thirds", f"Where {team or 'the team'} progressed and where it stalled"),
        ("progressive", f"Which passes moved {team or 'the team'} beyond pressure"),
        ("dominating_zones", "Who owned the space, and where it mattered"),
        ("pass_targets", f"Where {team or 'the team'} wanted the next receiver"),
        ("ppda", "The pressing battle, and what the intensity produced"),
        ("high_regains", f"What {team or 'the team'} did after winning the ball high"),
        ("defensive_activity", f"How {team or 'the team'} defended, and where"),
        ("defensive_summary", "Who defended with greater structural control?"),
        ("transition_outcomes", "Who used the moments of disorder better?"),
        ("advanced_metrics", "Reading volume, efficiency, value and risk together"),
        ("player_sequence", "Who connected the attacks before the final action?"),
    ]
    for token, title in titles:
        if token in stem:
            return title
    return "What this visual adds to the match story"


def _join_sentences(*parts: str) -> str:
    """Join analyst sentences into continuous prose.

    Each source paragraph already ends in a full stop, so joining is mostly a
    matter of collapsing whitespace and making sure a missing terminator does
    not run two sentences together.
    """
    cleaned = []
    for part in parts:
        text = " ".join(str(part or "").split())
        if not text:
            continue
        if text[-1] not in ".!?":
            text += "."
        cleaned.append(text)
    return " ".join(cleaned)


# Connective openers for the evidence and the conclusion. Picked per visual by
# a hash of the filename so the report does not repeat the same two phrases on
# every page, but a given page always reads the same way between runs.
# Each lead ends in a colon so the sentence that follows keeps its own
# capitalisation — lower-casing the first character mangled proper nouns and
# produced lines like "put numbers to it and juventus led field tilt".
_EVIDENCE_LEADS = (
    "The numbers behind that:",
    "Against the event record:",
    "The data supports the picture:",
    "Put numbers to it:",
    "The underlying record agrees:",
)
_CONCLUSION_LEADS = (
    "What that means in practice:",
    "The practical read:",
    "Taken together:",
    "For the coaching follow-up:",
    "The takeaway:",
)


def _split_for_columns(text: str) -> tuple[str, str]:
    """Split a paragraph into two balanced columns at a sentence boundary.

    Splitting mid-sentence would leave a clause hanging at the foot of the
    first column, so the break is taken at whichever full stop sits closest to
    the halfway mark.
    """
    # Split only where a terminator is followed by a space and a capital. A
    # plain [.!?] rule broke inside every decimal the commentary quotes, so
    # "field tilt 91.2%-8.8%" became two "sentences" and the second column
    # opened mid-number with "8%, and Man City led...".
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9‘“\"'])", text.strip())
    sentences = [part for part in parts if part]
    if len(sentences) < 2:
        return text.strip(), ""

    target = len(text) / 2
    best_index, best_gap = 1, None
    running = 0
    for index, sentence in enumerate(sentences[:-1], start=1):
        running += len(sentence) + 1
        gap = abs(running - target)
        if best_gap is None or gap < best_gap:
            best_index, best_gap = index, gap
    return " ".join(sentences[:best_index]).strip(), " ".join(sentences[best_index:]).strip()


def visual_narrative(path: Path, context: dict) -> str:
    """Return one continuous analyst paragraph for a visual.

    The report used to print three labelled blocks — PERFORMANCE ANALYST,
    DATA ANALYST, INTEGRATED READ — under every page. That reads as a form
    being filled in rather than as somebody explaining the match, so the three
    strands are now written as one paragraph: what happened, what the data
    says about it, and what to do with that.
    """
    mechanism = visual_explanation(path, context)
    evidence = visual_data_read(path, context)
    conclusion = visual_implication(path, context)

    seed = sum(ord(character) for character in path.stem)
    evidence_lead = _EVIDENCE_LEADS[seed % len(_EVIDENCE_LEADS)]
    conclusion_lead = _CONCLUSION_LEADS[(seed // 7) % len(_CONCLUSION_LEADS)]

    evidence = f"{evidence_lead} {evidence}" if evidence else ""
    conclusion = f"{conclusion_lead} {conclusion}" if conclusion else ""
    return _join_sentences(mechanism, evidence, conclusion)


def visual_data_read(path: Path, context: dict) -> str:
    """A concise data-analyst paragraph that validates or qualifies the tactical read."""
    stem = path.stem.lower()
    team, side = _visual_team(path, context)
    home, away = context["home"], context["away"]
    if "player_radars" in path.parts:
        player = path.stem.replace("_", " ")
        p = context.get("player_profiles", {}).get(player.lower(), {})
        if p:
            return (
                f"The event record places {player} in sequences worth {p['xGChain']:.2f} xGChain and {p['sequence_xT']:.2f} sequence xT, with "
                f"{_count(p['shots'], 'shot', 'shots')}, {_count(p['key_passes'], 'key pass', 'key passes')} and {p['pass_xT']:.2f} threat added by passing. These are single-match contributions, so role and minutes matter more than the total radar area."
            )
        return "The radar is scaled within this match, so it describes relative involvement on the day rather than long-term player quality."
    if "xg_flow" in stem:
        # Named the away side as the one who "finished with" the xG whatever
        # the totals said, and asserted that the goals exceeded them.
        xg_leader, xg_trailer, xg_level = _lead(
            home, away, context["home_xG"], context["away_xG"], tolerance=0.05)
        top = context["home_xG"] if xg_leader == home else context["away_xG"]
        bottom = context["home_xG"] if xg_trailer == home else context["away_xG"]
        scored = context["home_goals"] + context["away_goals"]
        expected = context["home_xG"] + context["away_xG"]
        tally = _count(scored, "goal", "goals")
        if scored > expected + 0.4:
            finishing = (f"The {tally} ran ahead of the combined {expected:.2f} xG, so "
                         f"finishing widened the score gap beyond the underlying chance gap.")
        elif scored < expected - 0.4:
            finishing = (f"The {tally} fell short of the combined {expected:.2f} xG, so "
                         f"the scoreline understates what the chances were worth.")
        else:
            finishing = (f"The {tally} came from a combined {expected:.2f} xG, so the "
                         f"scoreline and the chances created say the same thing.")
        opening = (
            f"Both sides attempted {int(context['home_shots'])} and "
            f"{int(context['away_shots'])} shots, and the curves finished level at "
            f"{top:.2f} xG. " if xg_level else
            f"Both sides attempted {int(context['home_shots'])} and "
            f"{int(context['away_shots'])} shots, but {xg_leader} finished with {top:.2f} xG "
            f"against {xg_trailer}'s {bottom:.2f}. ")
        return opening + finishing
    if "goals_breakdown" in stem:
        first = context["goal_rows"][0] if context["goal_rows"] else None
        first_line = f"{first['team']} scored first {_goal_moment(first)}" if first else "The game remained level early"
        return f"{first_line}; the match then produced {_count(context['home_goals'] + context['away_goals'], 'goal', 'goals')}. The sequence and assist fields locate the decisive actions, but the score-state split is required before comparing full-match possession or pressure totals."
    if "goalkeeper" in stem:
        return (
            f"{home} and {away} produced {int(context['home_on_target'])} and {int(context['away_on_target'])} on-target attempts, with {context['home_xGoT']:.2f} and {context['away_xGoT']:.2f} xGoT. "
            "The gap between goals, pre-shot xG and post-shot xGoT separates chance access, finishing placement and goalkeeper intervention."
        )
    if "shot_profile" in stem or "xg_summary" in stem:
        return (
            f"Shot volume was {int(context['home_shots'])}-{int(context['away_shots'])}, but average quality was {context['home_xG_per_shot']:.3f} xG per shot for {home} and {context['away_xG_per_shot']:.3f} for {away}. "
            f"The comparison supports a quality-over-volume interpretation, while the final score still contains finishing variance."
        )
    if "match_stats" in stem:
        return (
            f"{home} held {context['home_possession_share']:.1f}% possession and {context['home_field_tilt']:.1f}% field tilt; {away} posted {context['away_possession_share']:.1f}% and {context['away_field_tilt']:.1f}%. "
            f"Yet xG finished {context['home_xG']:.2f}-{context['away_xG']:.2f}, showing that territorial presence and final-action efficiency pointed in different directions."
        )
    if "post_match_advanced" in stem:
        # Who "led" a metric has to be read off the values. This block used to
        # hard-code the home side as the territory leader and the away side as
        # the efficiency leader, so a match where that was reversed printed a
        # sentence that contradicted the numbers printed beside it.
        def _leader(home_value: float, away_value: float, higher_is_better: bool = True) -> tuple[str, float, float]:
            home_first = (home_value >= away_value) if higher_is_better else (home_value <= away_value)
            if home_first:
                return home, home_value, away_value
            return away, away_value, home_value

        tilt_team, tilt_top, tilt_other = _leader(
            context["home_field_tilt"], context["away_field_tilt"]
        )
        entry_team, entry_top, entry_other = _leader(
            context["home_final_third_entries"], context["away_final_third_entries"]
        )
        quality_team, quality_top, quality_other = _leader(
            context["home_xG_per_shot"], context["away_xG_per_shot"]
        )
        transition_team, transition_top, transition_other = _leader(
            context["home_transition_shot_rate"], context["away_transition_shot_rate"]
        )
        return (
            f"{tilt_team} led field tilt {tilt_top:.1f}%-{tilt_other:.1f}%, and {entry_team} led final-third entries {int(entry_top)}-{int(entry_other)}. "
            f"{quality_team} led xG per shot {quality_top:.3f}-{quality_other:.3f} and {transition_team} led transition shot rate {transition_top:.1f}%-{transition_other:.1f}%, "
            f"while rest-defence vulnerability was {context['home_rest_defence_vulnerability']:.1f}% for {home} against {context['away_rest_defence_vulnerability']:.1f}% for {away}."
        )
    if "xt_per_minute" in stem:
        return (
            f"Total expected threat was {context['home_xT']:.2f} for {home} and {context['away_xT']:.2f} for {away}. The totals describe accumulated value; the minute-by-minute peaks identify when that value arrived and whether it clustered around goals or game-state changes."
        )
    if "game_state" in stem:
        return (
            f"The final {context['score']} outcome forced the teams to spend different amounts of time level, leading and trailing. Full-match averages therefore mix behaviours produced under different incentives and should be treated as weighted summaries, not neutral tactical baselines."
        )
    if "shot_map" in stem and side:
        return (
            f"{team} generated {int(context[f'{side}_shots'])} attempts worth {context[f'{side}_xG']:.2f} xG, or {context[f'{side}_xG_per_shot']:.3f} per shot, and put {int(context[f'{side}_on_target'])} on target. "
            f"The {context[f'{side}_box_entry_to_shot_rate']:.1f}% box-entry-to-shot rate shows how often penetration became an immediate finish."
        )
    if "danger_creation" in stem and side:
        return (
            f"The event chain records {int(context[f'{side}_final_third_entries'])} final-third entries, {int(context[f'{side}_deep_completions'])} deep completions and {int(context[f'{side}_box_entries'])} box entries for {team}. "
            "The fall between stages identifies whether attacks broke down before the last line or after reaching the penalty area."
        )
    if "zone14" in stem and side:
        return (
            f"{team} recorded {int(context[f'{side}_deep_completions'])} deep completions and {int(context[f'{side}_box_entries'])} box entries. Zone 14 actions are most credible when the same possessions continue into those outcomes rather than ending with a safe recycle."
        )
    if "crosses" in stem and side:
        attempts = int(context[f"{side}_crosses"])
        completed = int(context[f"{side}_completed_crosses"])
        rate = completed / attempts * 100 if attempts else 0.0
        return f"{team} attempted {attempts} crosses and completed {completed} ({rate:.1f}%). Completion is a limited measure; shot creation and the position of the next defensive action determine whether the delivery was tactically productive."
    if "box_entries" in stem and side:
        return (
            f"{team} reached the box {_count(context[f'{side}_box_entries'], 'time', 'times')}, with {context[f'{side}_box_entry_to_shot_rate']:.1f}% becoming shots. "
            "The conversion rate is the useful denominator because it distinguishes access from a controlled final action."
        )
    if "pass_network" in stem and side:
        half = "first-half" if "_1h" in stem else "second-half"
        return (
            f"Across the full match {team} held {context[f'{side}_pass_share']:.1f}% of the pass share and registered {int(context[f'{side}_touches'])} touches. "
            f"The {half} network then shows how that volume was distributed; changes between halves should be interpreted with substitutions and score state rather than as one stable structure."
        )
    if "average_positions" in stem and side:
        half = "first-half" if "_1h" in stem else "second-half"
        return (
            f"This is a {half} average, so it compresses many possession moments into one location per player. Pair it with {team}'s {context[f'{side}_field_tilt']:.1f}% field tilt and the half-specific network before inferring line height or permanent role changes."
        )
    if "xt_map" in stem and side:
        other = "away" if side == "home" else "home"
        return (
            f"{team} accumulated {context[f'{side}_xT']:.2f} expected threat against {context[f'{other}_xT']:.2f} for the opponent. The grid identifies concentration, but a hot cell supported by only one action should not be treated as a repeatable route without the entry and shot maps."
        )
    if "pass_map" in stem and side:
        return (
            f"{team} recorded {int(context[f'{side}_progressive_passes'])} progressive passes and {int(context[f'{side}_final_third_entries'])} final-third entries. Those counts describe advancement; the failure locations and the rest-defence pages determine the cost of the chosen passing risk."
        )
    if "ball_touches" in stem:
        return (
            f"Touches finished {int(context['home_touches'])}-{int(context['away_touches'])}, while field tilt was {context['home_field_tilt']:.1f}%-{context['away_field_tilt']:.1f}%. The different directions confirm that total involvement and control of attacking territory were not the same measure."
        )
    if "pass_thirds" in stem and side:
        return (
            f"{team} produced {int(context[f'{side}_final_third_entries'])} final-third entries at {context[f'{side}_final_third_entry_efficiency']:.1f}% efficiency. The thirds distribution gives the volume base; efficiency tests how often possession crossed into the next meaningful phase."
        )
    if "progressive" in stem and side:
        return (
            f"The data records {int(context[f'{side}_progressive_passes'])} progressive passes and {context[f'{side}_sequence_xT']:.2f} sequence xT for {team}. The relationship between the two helps separate frequent advancement from advancement that materially improved the attacking state."
        )
    if "dominating_zones" in stem:
        # Both halves of this sentence named a fixed side, so in this fixture
        # it read "Field tilt favoured Arsenal 29.2%-70.8%, while xG favoured
        # Man City 1.08-1.88" — inverted twice, against the figures beside it.
        tilt_team = home if context["home_field_tilt"] >= context["away_field_tilt"] else away
        xg_team = home if context["home_xG"] >= context["away_xG"] else away
        tilt_top = max(context["home_field_tilt"], context["away_field_tilt"])
        tilt_low = min(context["home_field_tilt"], context["away_field_tilt"])
        xg_top = max(context["home_xG"], context["away_xG"])
        xg_low = min(context["home_xG"], context["away_xG"])
        closing = (
            "The opposing signals are evidence that zone ownership did not translate "
            "proportionally into chance quality."
            if tilt_team != xg_team else
            "Territory and chance quality ran the same way here, so the open question is "
            "whether the zones held were the ones that mattered or simply the ones the "
            "opponent was content to concede.")
        return (
            f"Field tilt favoured {tilt_team} {tilt_top:.1f}%-{tilt_low:.1f}%, while xG "
            f"favoured {xg_team} {xg_top:.2f}-{xg_low:.2f}. {closing}"
        )
    if "pass_targets" in stem and side:
        return (
            f"{team} registered {int(context[f'{side}_deep_completions'])} deep completions and {int(context[f'{side}_box_entries'])} box entries. Compare those outcomes with the destination density to test whether the preferred target zones actually advanced the possession."
        )
    if "ppda" in stem:
        return (
            f"PPDA was {context['home_ppda']:.2f} for {home} and {context['away_ppda']:.2f} for {away}; lower indicates earlier defensive engagement. High regains were {int(context['home_high_regains'])}-{int(context['away_high_regains'])}, while dangerous counters conceded were {int(context['home_rest_defence_dangerous_counters'])}-{int(context['away_rest_defence_dangerous_counters'])}."
        )
    if "high_regains" in stem and side:
        return (
            f"{team} made {int(context[f'{side}_high_regains'])} high regains; {context[f'{side}_regain_to_shot_rate']:.1f}% became shots, producing {context[f'{side}_regain_xG']:.2f} xG and {context[f'{side}_regain_xT']:.2f} xT. This separates pressing activity from attacking return."
        )
    if "defensive_activity" in stem and side:
        return (
            f"{team} faced {int(context[f'{side}_rest_defence_exposures'])} rest-defence exposures and conceded {int(context[f'{side}_rest_defence_dangerous_counters'])} dangerous counters. The locations of interventions show where the defence acted; those outcomes qualify whether the activity represented control or emergency response."
        )
    if "defensive_summary" in stem:
        return (
            f"Rest-defence vulnerability was {context['home_rest_defence_vulnerability']:.1f}% for {home} and {context['away_rest_defence_vulnerability']:.1f}% for {away}, with {int(context['home_rest_defence_dangerous_counters'])} and {int(context['away_rest_defence_dangerous_counters'])} dangerous counters conceded. "
            "This outcome measure is more diagnostic than raw defensive-action volume alone."
        )
    if "transition_outcomes" in stem:
        return (
            f"{home} turned {_count(context['home_transitions'], 'transition', 'transitions')} into {_count(context['home_transition_shots'], 'shot', 'shots')} ({context['home_transition_shot_rate']:.1f}%); {away} turned {int(context['away_transitions'])} into {int(context['away_transition_shots'])} ({context['away_transition_shot_rate']:.1f}%). "
            f"Transition xG was {context['home_transition_xG']:.2f}-{context['away_transition_xG']:.2f}."
        )
    if "advanced_metrics" in stem:
        return (
            f"{home} led field tilt and sequence xT ({context['home_sequence_xT']:.2f} to {context['away_sequence_xT']:.2f}), while {away} led xG per shot and transition shot rate. The split confirms that volume, value, efficiency and risk produced different leaders."
        )
    if "player_sequence" in stem:
        return (
            f"The xGChain leaders were {context['home_players']['chain']} for {home} and {context['away_players']['chain']} for {away}. xGBuildup and sequence xT add earlier involvement, preventing the analysis from assigning all credit to the final passer or shooter."
        )
    # The same fifteen boards that had no reading and no coaching note also had
    # no data line, so a third of the report carried one sentence three times.
    if "match_momentum" in stem:
        return (
            f"The windows are five minutes wide and sum to {context['home_xG']:.2f}-"
            f"{context['away_xG']:.2f} xG. A short window is a small sample by "
            f"construction, so read the run of bars rather than any single one; one "
            f"clear chance fills a window on its own."
        )
    if "win_probability" in stem:
        return (
            f"The curve is driven by the "
            f"{_count(context['home_goals'] + context['away_goals'], 'goal', 'goals')} "
            f"and the {context['home_xG'] + context['away_xG']:.2f} combined xG behind "
            f"{'it' if context['home_goals'] + context['away_goals'] == 1 else 'them'}, "
            f"priced as they arrived. It carries no information the shot record does "
            f"not, so treat it as a readable summary of sequence rather than as evidence "
            f"of its own."
        )
    if "sequence_types" in stem:
        return (
            f"Broken play produced {context['home_transition_xG']:.2f} xG for {home} from "
            f"{int(context['home_transitions'])} transitions and "
            f"{context['away_transition_xG']:.2f} from {int(context['away_transitions'])} "
            f"for {away}. The remainder came from settled possession and restarts, which is "
            f"the comparison the columns are for."
        )
    if "goal_origins" in stem:
        return (
            f"{_count(context['home_goals'] + context['away_goals'], 'goal', 'goals')} "
            f"{'is' if context['home_goals'] + context['away_goals'] == 1 else 'are'} "
            f"traced here, of "
            f"which {int(context['home_transition_goals']) + int(context['away_transition_goals'])} "
            f"began as a transition. With a handful of events the classification matters "
            f"more than the count: check each origin against the footage before treating "
            f"the split as a pattern."
        )
    if "pitch_control" in stem:
        return (
            f"The surface is a model, not a measurement: it infers reach from position and "
            f"distance rather than recording it. Field tilt puts the same question in one "
            f"number - {context['home_field_tilt']:.1f}% for {home} against "
            f"{context['away_field_tilt']:.1f}% - and the two should broadly agree. Where "
            f"they do not, trust the passing record."
        )
    if "set_pieces" in stem:
        return (
            f"Restart chances are a small sample in any single match, so the totals here "
            f"carry less weight than the repetition of a routine. {home} attempted "
            f"{int(context['home_crosses'])} crosses in all phases and {away} "
            f"{int(context['away_crosses'])}, which is the volume the delivery quality "
            f"should be judged against."
        )
    if "ball_losses" in stem and side:
        return (
            f"{team} were exposed on {context[f'{side}_rest_defence_vulnerability']:.1f}% of "
            f"{int(context[f'{side}_rest_defence_exposures'])} advanced losses, conceding "
            f"{_count(context[f'{side}_rest_defence_dangerous_counters'], 'dangerous counter', 'dangerous counters')}. "
            f"The denominator matters: a low rate over few losses is not the same evidence "
            f"as a low rate over many."
        )
    if "defensive_shape" in stem:
        return (
            f"PPDA read {context['home_ppda']:.2f} for {home} and {context['away_ppda']:.2f} "
            f"for {away}, with {int(context['home_high_regains'])} and "
            f"{int(context['away_high_regains'])} high regains. PPDA measures how early a "
            f"side engaged and says nothing about what the engagement won, which is why the "
            f"regain counts sit beside it."
        )
    if "playing_through" in stem and side:
        return (
            f"{team} completed {_count(context[f'{side}_deep_completions'], 'pass', 'passes')} into "
            f"the deep attacking zone from {int(context[f'{side}_progressive_passes'])} "
            f"progressive passes. The ratio is the useful figure; the raw progressive count "
            f"rewards a side that simply had more of the ball."
        )
    if "unlocking" in stem and side:
        return (
            f"{team} reached the final third {int(context[f'{side}_final_third_entries'])} "
            f"times and the box {int(context[f'{side}_box_entries'])}, converting "
            f"{context[f'{side}_box_entry_to_shot_rate']:.1f}% of those entries into a shot. "
            f"Receptions in the pocket are the step between the first two numbers."
        )
    if "press_triggers" in stem:
        return (
            f"{home} won {int(context['home_high_regains'])} balls in the opponent's "
            f"territory and {away} {int(context['away_high_regains'])}, converting "
            f"{context['home_regain_to_shot_rate']:.1f}% and "
            f"{context['away_regain_to_shot_rate']:.1f}% of all regains into a shot. The "
            f"trigger classification is inferred from the opponent's action, so treat it as "
            f"a description of the moment rather than proof of intent."
        )
    if "action_value" in stem:
        return (
            f"Values are in goals, so they are directly comparable across action types and "
            f"directly distorted by sample size. Over a single match the ranking is "
            f"dominated by a handful of events out of the {int(context['home_touches']) + int(context['away_touches'])} "
            f"touches recorded, which is why it locates passages rather than rating players."
        )
    return "The numerical layer should confirm the visual pattern, provide a denominator and identify uncertainty. It should not replace the football mechanism shown on the page."


def _next_purpose(path: Path) -> str:
    stem = path.stem.lower()
    purposes = [
        ("goals_breakdown", "places the data back into the scoring sequence and explains the change in game state"),
        ("shot_map", "moves from overall quality to the locations and types of attempts"),
        ("shot_profile", "compares volume, accuracy and quality on one scale"),
        ("goalkeeper", "tests how much of the outcome came from post-shot execution and saves"),
        ("xg_summary", "condenses shot volume and quality into a direct team comparison"),
        ("match_stats", "sets the broad match context before the report isolates individual mechanisms"),
        ("post_match_advanced", "compresses the main attacking and defensive verdict into one publishable comparison"),
        ("xt_per_minute", "locates the periods when possession became meaningful threat"),
        ("game_state", "shows how leading and trailing altered the meaning of the full-match totals"),
        ("danger_creation", "traces the repeated routes that carried possession into dangerous areas"),
        ("zone14", "tests central access in front of the penalty area"),
        ("crosses", "examines whether wide progression had runners, targets and second-ball support"),
        ("box_entries", "checks whether final-third access became controlled penalty-area possession"),
        ("pass_network", "reveals the structural links and hubs behind the possession pattern"),
        ("average_positions", "shows the occupation and spacing that produced those passing links"),
        ("xt_map", "locates the zones where progression added the most threat"),
        ("pass_map", "expands the view from high-value actions to the full passing risk profile"),
        ("ball_touches", "tests where the team's overall involvement occurred"),
        ("pass_thirds", "locates the phase where progression accelerated or stalled"),
        ("progressive", "isolates the line-breaking actions inside the wider possession structure"),
        ("dominating_zones", "compares territorial control across the entire pitch"),
        ("pass_targets", "shows where the next receiver was repeatedly found"),
        ("defensive_activity", "moves from possession into the locations and height of defensive engagement"),
        ("defensive_summary", "compares ball-winning activity with structural protection"),
        ("high_regains", "tests whether the press created useful attacking possessions"),
        ("ppda", "summarises pressing intensity before the report evaluates transition consequences"),
        ("transition_outcomes", "measures how efficiently each side attacked before the block could reset"),
        ("advanced_metrics", "joins volume, efficiency, value and risk into one diagnostic view"),
        ("player_sequence", "moves from team mechanisms to the players who connected valuable attacks"),
    ]
    if "player_radars" in path.parts:
        return "continues the role-by-role review inside the player appendix"
    for token, purpose in purposes:
        if token in stem:
            return purpose
    return "adds the next piece of evidence required to test this tactical reading"


def next_visual_step(next_path: Path | None) -> str:
    if next_path is None:
        return "This closes the visual appendix. Return to the Final Tactical Verdict to connect the player roles with the team-level coaching priorities."
    return f"Read next alongside '{_visual_title(next_path)}' to see how it {_next_purpose(next_path)}."


class TacticalPDF:
    def __init__(self, output: Path, context: dict):
        self.output = output
        self.context = context
        self.page = 0
        self.canvas = canvas.Canvas(str(output), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
        self.canvas.setTitle(f"{context['home']} vs {context['away']} - Detailed Tactical and Data Report")
        self.canvas.setAuthor("Mostafa Saad")
        self.canvas.setSubject("Football performance analysis and match data report")
        # The chrome carries the fixture's own colours. It used to be a fixed
        # blue/yellow pair, so every page framed images drawn in the teams'
        # real kit colours with a border belonging to neither of them.
        self.home_color = _as_pdf_color(context.get("home_color"), HOME)
        self.away_color = _as_pdf_color(context.get("away_color"), AWAY)
        self.body = ParagraphStyle("body", fontName="Helvetica", fontSize=TYPE_BODY, leading=13.2, textColor=TEXT, alignment=TA_LEFT)
        self.small = ParagraphStyle("small", fontName="Helvetica", fontSize=TYPE_CAPTION, leading=10.5, textColor=MUTED, alignment=TA_LEFT)
        self.card = ParagraphStyle("card", fontName="Helvetica", fontSize=TYPE_BODY, leading=12.3, textColor=TEXT, alignment=TA_LEFT)
        self.analysis = ParagraphStyle("analysis", fontName="Helvetica", fontSize=TYPE_CAPTION, leading=11.25, textColor=TEXT, alignment=TA_LEFT)
        self.implication = ParagraphStyle("implication", fontName="Helvetica", fontSize=TYPE_CAPTION, leading=10.7, textColor=TEXT, alignment=TA_LEFT)
        self.next_step = ParagraphStyle("next_step", fontName="Helvetica", fontSize=TYPE_CAPTION, leading=10.1, textColor=MUTED, alignment=TA_LEFT)
        # The commentary used to be set in Times while every embedded visual is
        # sans, so each page carried two unrelated type families and read as two
        # documents stapled together. One family throughout; the commentary is
        # separated from the chrome by weight and colour instead.
        self.commentary_title = ParagraphStyle("commentary_title", fontName="Helvetica-Bold", fontSize=TYPE_TITLE, leading=19.5, textColor=TEXT, alignment=TA_LEFT)
        self.commentary_body = ParagraphStyle("commentary_body", fontName="Helvetica", fontSize=TYPE_BODY, leading=13.2, textColor=TEXT, alignment=TA_LEFT)
        self.commentary_next = ParagraphStyle("commentary_next", fontName="Helvetica-Oblique", fontSize=TYPE_CAPTION, leading=11.0, textColor=MUTED, alignment=TA_LEFT)

    def _start(self, bookmark: str | None = None, outline: str | None = None, level: int = 0):
        self.page += 1
        self.canvas.setFillColor(BG)
        self.canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        if bookmark:
            self.canvas.bookmarkPage(bookmark)
            if outline:
                self.canvas.addOutlineEntry(outline, bookmark, level=level, closed=False)

    def _finish(self):
        self.canvas.setFillColor(NEUTRAL)
        self.canvas.setFont("Helvetica-Bold", TYPE_MICRO)
        self.canvas.drawRightString(PAGE_W - 24, 14, f"PAGE {self.page:02d}  |  REAL MATCH EVENTS")
        self.canvas.saveState()
        self.canvas.setStrokeColor(GRID)
        self.canvas.setLineWidth(0.5)
        self.canvas.line(24, 25, PAGE_W - 24, 25)
        self.canvas.restoreState()
        self.canvas.showPage()

    def _header(self, title: str, subtitle: str, section: str):
        c = self.canvas
        # 86pt of panel rather than 70: the section label, the title and the
        # subtitle used to be stacked so tightly that the subtitle's descenders
        # touched the team-colour rule closing the panel.
        top = PAGE_H - 24
        panel_h = 86
        base = top - panel_h
        c.setFillColor(PANEL_2)
        c.roundRect(24, base, PAGE_W - 48, panel_h, 9, fill=1, stroke=0)
        c.setFillColor(self.home_color); c.rect(28, base + 2, (PAGE_W - 56) / 2, 3, fill=1, stroke=0)
        c.setFillColor(self.away_color); c.rect(PAGE_W / 2, base + 2, (PAGE_W - 56) / 2, 3, fill=1, stroke=0)
        c.setFillColor(BRAND); c.circle(43, top - 21, 3.2, fill=1, stroke=0)
        c.setFillColor(MUTED); c.setFont("Helvetica-Bold", TYPE_MICRO); c.drawString(54, top - 24, section.upper())
        c.setFillColor(TEXT); c.setFont("Helvetica-Bold", TYPE_TITLE); c.drawString(42, top - 50, title)
        c.setFillColor(MUTED); c.setFont("Helvetica", TYPE_CAPTION); c.drawString(42, top - 68, subtitle[:125])
        c.setFillColor(self.home_color); c.setFont("Helvetica-Bold", TYPE_BODY); c.drawRightString(PAGE_W - 300, PAGE_H - 50, self.context["home"].upper())
        c.setFillColor(TEXT); c.setFont("Helvetica-Bold", TYPE_TITLE); c.drawCentredString(PAGE_W - 245, PAGE_H - 50, self.context["score"])
        c.setFillColor(self.away_color); c.setFont("Helvetica-Bold", TYPE_BODY); c.drawString(PAGE_W - 190, PAGE_H - 50, self.context["away"].upper())

    def _paragraph(self, text: str, x: float, top: float, width: float, max_height: float, style: ParagraphStyle | None = None) -> float:
        paragraph = Paragraph(text, style or self.body)
        _, height = paragraph.wrap(width, max_height)
        paragraph.drawOn(self.canvas, x, top - height)
        return height

    def _card_box(self, x: float, y: float, w: float, h: float, title: str, accent=FOCUS):
        c = self.canvas
        c.setFillColor(PANEL)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.8)
        c.roundRect(x, y, w, h, 7, fill=1, stroke=1)
        c.setFillColor(accent)
        c.rect(x, y + h - 3, w, 3, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", TYPE_CAPTION)
        c.drawString(x + 14, y + h - 22, title.upper())

    # Metrics the cover may lead with, as (context key, printed name, unit).
    # Each is a percentage split that sums to 100 across the two sides, so one
    # number states the whole balance and the bar underneath is honest.
    # Two kinds of number. A "split" is a share of one whole and the two sides
    # sum to 100, so it draws as one divided bar. A "rate" is each side's own
    # percentage of its own attempts; the two are independent and each gets its
    # own bar against a common 100 baseline. Drawing a rate as a split bar
    # would claim the two halves add up, which they do not.
    COVER_LEADS = (
        ("split", "field_tilt", "Field tilt", "share of completed passes reaching the final third"),
        ("split", "possession_share", "Possession", "share of the match in controlled possession"),
        # pass_share counts passes attempted, not completed: the mask carries no
        # outcome filter. Both descriptions of it said "completed".
        ("split", "pass_share", "Pass share", "share of all passes played"),
        ("rate", "box_entry_to_shot_rate", "Box entry to shot", "share of penalty-area entries that became a shot"),
        ("rate", "regain_to_shot_rate", "Regain to shot", "share of possession regains that became a shot"),
        ("rate", "transition_shot_rate", "Transition to shot", "share of transitions that became a shot"),
        ("rate", "build_up_success_rate", "Build-up success", "share of build-up attempts that cleared the press"),
        ("rate", "final_third_entry_efficiency", "Final-third efficiency", "share of final-third entries that became a box entry"),
    )

    def _verdict(self) -> str:
        """One sentence on how the result related to the chances created.

        This used to assert that the winner had won the execution battle and
        that the loser's activity never became shot quality. That reads well
        when the winner also created more — and contradicts the numbers printed
        beside it when they did not. Fulham lost 0-1 having led xG 1.58 to 0.81,
        shots 14 to 10 and field tilt 62.6% to 37.4%, under a sentence saying
        their activity had not become control of shot quality.
        """
        winner, loser = self.context.get("winner"), self.context.get("loser")
        home, away = self.context["home"], self.context["away"]

        def xg_for(team):
            side = "home" if team == home else "away"
            try:
                return float(self.context.get(f"{side}_xG"))
            except (TypeError, ValueError):
                return None

        if not winner or winner == loser:
            leader, trailer = home, away
            lead_xg, trail_xg = xg_for(home), xg_for(away)
            if lead_xg is not None and trail_xg is not None and trail_xg > lead_xg:
                leader, trailer = away, home
            return (f"The draw flattered neither side equally: {leader} created "
                    f"the better share of the chances {trailer} had to survive.")

        winner_xg, loser_xg = xg_for(winner), xg_for(loser)
        if winner_xg is None or loser_xg is None:
            return f"{winner} took the result; the process behind it is what the report examines."
        if winner_xg >= loser_xg:
            return (f"{winner} won the execution battle. "
                    f"{loser}'s activity never became control of shot quality.")
        # The higher total is not the better performance when almost all of it
        # arrived after going behind. This is the first line a reader sees, and
        # it was making the claim the report goes on to correct eleven pages
        # later: Manchester United's 1.97 against Hull's 1.60 was 94% chased,
        # and 0.11 while the match was level.
        verdict = self.context.get("verdict")
        if verdict is not None and verdict.loser_was_only_chasing:
            beaten = verdict.of(loser)
            return (f"{loser}'s xG is a chase, not a performance. "
                    f"{100 * beaten.chasing_share:.0f}% of it arrived behind; "
                    f"before that, {beaten.not_chasing_xg:.2f}.")
        return (f"{loser} created the better chances and lost. "
                f"{winner} needed fewer of them and took them.")

    def _cover_lead(self):
        """The match's most lopsided percentage, and how to draw it.

        Returns ``(kind, name, note, home, away)``; never None. The previous
        version needed a 25-point gap in field tilt, possession or pass share
        and returned None otherwise, which is almost always: a 59/41 possession
        match still did not qualify. The cover then fell to a single thin strip
        and 49% of the page was empty in two dead bands.

        Ranked on the *relative* gap rather than the absolute one, so a 9%
        against 3% conversion rate outranks a 54 against 46 territory split —
        which is the right way round, because it is the bigger difference.
        """
        best = None
        for kind, key, name, note in self.COVER_LEADS:
            home = self.context.get(f"home_{key}")
            away = self.context.get(f"away_{key}")
            try:
                home, away = float(home), float(away)
            except (TypeError, ValueError):
                continue
            if not (home > 0 or away > 0):
                continue
            if kind == "split" and not 95.0 <= home + away <= 105.0:
                continue  # not a two-way split; a divided bar would lie
            gap = abs(home - away) / max(home, away, 1e-6)
            if best is None or gap > best[0]:
                best = (gap, kind, name, note, home, away)
        if best is None:
            return None
        return best[1:]

    def _cover_logo(self, cx: float, top: float, size: float) -> float:
        """Draw the publisher's badge centred on ``cx``; return its bottom edge.

        Falls back to the wordmark when the file is absent, so a fresh clone
        still produces a finished cover rather than a hole where a logo was.
        """
        c = self.canvas
        if LOGO_PATH.exists():
            try:
                # The badge is a JPEG on its own black ground, so on the light
                # page it lands as a bare black square. Give it a rounded plate
                # of the same black and it reads as a deliberate badge tile
                # instead of an unmasked crop.
                if IS_LIGHT_THEME:
                    pad = size * 0.06
                    c.saveState()
                    c.setFillColor(colors.HexColor("#0A0A0A"))
                    c.roundRect(cx - size / 2 - pad, top - size - pad,
                                size + 2 * pad, size + 2 * pad, size * 0.09,
                                stroke=0, fill=1)
                    c.restoreState()
                c.drawImage(str(LOGO_PATH), cx - size / 2, top - size, size, size,
                            mask=None, preserveAspectRatio=True, anchor="c")
                return top - size
            except Exception:
                pass  # unreadable image: fall through to the wordmark
        c.setFillColor(TEXT); c.setFont("Helvetica-Bold", TYPE_DISPLAY)
        c.drawCentredString(cx, top - 34, "TACTICAL")
        c.setFillColor(BRAND); c.setFont("Helvetica-Bold", TYPE_BODY)
        c.drawCentredString(cx, top - 50, "F O O T B A L L   D A T A   &   A N A L Y S I S")
        return top - 62

    # The artwork bleeds from this height to the top of the sheet.
    COVER_ART_FLOOR = 262

    # The eight numbers a reader wants before anything else, in the order a
    # match is usually argued: who had the ball, what the chances were worth,
    # how often they tried, how good the looks were, how far they got, how far
    # inside, who held the ground, and what the possessions were worth.
    COVER_ROWS = (
        ("POSSESSION",          "possession_share",    "{:.1f}%"),
        ("EXPECTED GOALS",      "xG",                  "{:.2f}"),
        ("SHOTS  (ON TARGET)",  "shots",               ""),
        ("BIG CHANCES",         "big_chances",         "{:.0f}"),
        ("FINAL THIRD ENTRIES", "final_third_entries", "{:.0f}"),
        ("BOX ENTRIES",         "box_entries",         "{:.0f}"),
        ("PITCH CONTROL",       "field_tilt",          "{:.0f}%"),
        ("SEQUENCE THREAT  xT", "sequence_xT",         "{:.2f}"),
    )

    def cover(self):
        """The match in eight numbers, in the two clubs' colours.

        The cover was the pitch-control artwork with one sentence under it. It
        looked like the front of a document and told a reader nothing they
        could act on, and the sentence — being a single claim — was the part of
        the report most likely to be wrong.

        A comparison card is the opposite. Every row is two figures and a bar
        drawn from them, so it cannot assert anything the data does not, and
        the shape of the match arrives in one look.
        """
        self._start("cover", "Cover")
        c = self.canvas
        centre = PAGE_W / 2

        top = PAGE_H - 54
        c.setFillColor(NEUTRAL)
        c.setFont("Helvetica-Bold", TYPE_CAPTION)
        competition = self._cover_competition().upper()
        c.drawString(COVER_MARGIN, top, _spaced_out(competition or "MATCH ANALYSIS"))
        if competition:
            c.setFillColor(MUTED)
            c.setFont("Helvetica-Bold", TYPE_MICRO)
            c.drawString(COVER_MARGIN, top - 19, _spaced_out("MATCH ANALYSIS"))
        self._cover_logo(PAGE_W - 92, PAGE_H - 22, 64)

        head_rule = PAGE_H - COVER_HEAD_DROP
        foot_rule = COVER_FOOT_LIFT
        c.setStrokeColor(GRID)
        c.setLineWidth(0.8)
        c.line(COVER_MARGIN, head_rule, PAGE_W - COVER_MARGIN, head_rule)
        c.line(COVER_MARGIN, foot_rule, PAGE_W - COVER_MARGIN, foot_rule)

        # Both rules are fixed to the sheet and the card is centred between
        # them. Hanging the card from the header and letting the footer float
        # under the last row put all the slack in one place: 129pt of air
        # above the crests and 40 below the final figure, on a card that is
        # meant to read as one block. It also meant the footer's position was
        # a function of how many rows COVER_ROWS happened to hold.
        rows = self._cover_rows()
        body = (COVER_CREST_RISE + COVER_ROW_GAP
                + (len(rows) - 1) * COVER_ROW_STEP + COVER_ROW_SINK)
        badge_y = (head_rule + foot_rule + body) / 2 - COVER_CREST_RISE
        self._cover_badges(centre, badge_y)

        first = badge_y - COVER_ROW_GAP
        for index, row in enumerate(rows):
            self._cover_row(centre, first - index * COVER_ROW_STEP, *row)

        c.setFillColor(NEUTRAL)
        c.setFont("Helvetica-Bold", TYPE_MICRO)
        byline = str(self.context.get("byline") or "MOSTAFA SAAD").upper()
        c.drawString(COVER_MARGIN, foot_rule - 21, _spaced_out(byline))
        c.drawRightString(PAGE_W - COVER_MARGIN, foot_rule - 21,
                          _spaced_out("WHOSCORED / OPTA EVENT DATA"))

        # The two-colour rule every visual and poster closes on.
        c.setFillColor(self.home_color)
        c.rect(0, 0, PAGE_W / 2, 5, stroke=0, fill=1)
        c.setFillColor(self.away_color)
        c.rect(PAGE_W / 2, 0, PAGE_W / 2, 5, stroke=0, fill=1)
        self._finish()

    def _cover_competition(self) -> str:
        """The competition line, read from the fixture rather than typed."""
        supplied = str(self.context.get("competition") or "").strip()
        if supplied:
            return supplied
        try:
            from match_fixture import describe, from_url

            url = self.context.get("url") or self._stored_url()
            line = describe(from_url(url))
            if line:
                return line
        except Exception:
            pass
        return ""

    def _stored_url(self) -> str:
        """The fixture's URL from the match history, when info has none.

        The collector does not put the URL in match_info.json, so the
        competition line came out empty and the header printed its subtitle
        twice. The history has kept it all along.
        """
        import sqlite3

        try:
            database = Path(__file__).resolve().parent / "output" / "match_history.db"
            if not database.exists():
                return ""
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            try:
                row = connection.execute(
                    "SELECT url FROM matches WHERE home_team = ? AND away_team = ? "
                    "ORDER BY stored_at DESC LIMIT 1",
                    (str(self.context.get("home")), str(self.context.get("away"))),
                ).fetchone()
            finally:
                connection.close()
            return str(row[0]) if row and row[0] else ""
        except Exception:
            return ""

    def _cover_badges(self, centre: float, y: float):
        """Crest, name and score, with the crest carrying the club's colour."""
        c = self.canvas
        crest, spread = 92, 232

        for team_id, colour, name, x in (
            (self.context.get("home_id"), self.home_color,
             self.context["home"], centre - spread),
            (self.context.get("away_id"), self.away_color,
             self.context["away"], centre + spread),
        ):
            badge = self._crest_reader(team_id)
            if badge is not None:
                c.drawImage(badge, x - crest / 2, y - crest / 2, crest, crest,
                            mask="auto", preserveAspectRatio=True, anchor="c")
            else:
                # No crest cached: a disc in the club's colour with its
                # initials, which is what the crest would have carried.
                c.setFillColor(colour)
                c.circle(x, y, crest / 2, stroke=0, fill=1)
                c.setFillColor(colors.white)
                c.setFont("Helvetica-Bold", TYPE_COVER_MARK)
                c.drawCentredString(x, y - 10, _club_initials(name))
            c.setFillColor(TEXT)
            c.setFont("Helvetica-Bold", TYPE_COVER_TEAM)
            c.drawCentredString(x, y - crest / 2 - 25, str(name).upper())

        c.setFillColor(TEXT)
        c.setFont("Helvetica-Bold", TYPE_COVER_SCORE)
        c.drawCentredString(centre, y - 15, str(self.context["score"]))

    def _cover_rows(self):
        """(label, home text, away text, home share) for every row."""
        built = []
        for label, key, shape in self.COVER_ROWS:
            home, away, home_text, away_text = self._cover_pair(key, shape)
            total = abs(home) + abs(away)
            share = 0.5 if not total else abs(home) / total
            built.append((label, home_text, away_text, share))
        return built

    def _cover_pair(self, key: str, shape: str):
        """One row's figures, as numbers and as the text to print."""
        context = self.context
        if key == "shots":
            home = _number(context.get("home_shots"))
            away = _number(context.get("away_shots"))
            return (home, away,
                    f"{home:.0f} ({_number(context.get('home_on_target')):.0f})",
                    f"{away:.0f} ({_number(context.get('away_on_target')):.0f})")
        home = _number(context.get(f"home_{key}"))
        away = _number(context.get(f"away_{key}"))
        return home, away, shape.format(home), shape.format(away)

    def _cover_row(self, centre: float, y: float, label: str,
                   home_text: str, away_text: str, home_share: float):
        """A label, two figures, and one bar split between the two colours.

        Both halves grow outwards from the centre line, which is what makes a
        row read as a comparison rather than as two unrelated lengths.
        """
        c = self.canvas
        width, height = 560.0, 6.0
        left = centre - width / 2
        half = width / 2

        c.setFillColor(MUTED)
        c.setFont("Helvetica-Bold", TYPE_MICRO)
        c.drawCentredString(centre, y + 15, _spaced_out(label))

        # The track, so a short bar still reads against a measured length.
        c.setFillColor(GRID)
        c.roundRect(left, y - height / 2, width, height, height / 2,
                    stroke=0, fill=1)

        # Each half is that side's share of the pair, scaled so the larger
        # figure fills its half and the smaller is drawn in proportion to it.
        # Four lines of arithmetic here overwrote one another and every one of
        # them pinned the leading side at exactly half, so a 2.15 against 1.11
        # drew the same blue bar as a 1.11 against 2.15.
        share = min(max(float(home_share), 0.0), 1.0)
        bigger = max(share, 1.0 - share) or 1.0
        home_length = half * (share / bigger)
        away_length = half * ((1.0 - share) / bigger)

        # Both bars meet at the centre line and grow outwards, so the end each
        # one reaches is its figure and the two are read against one another.
        c.setFillColor(self.home_color)
        c.roundRect(centre - home_length, y - height / 2, home_length, height,
                    height / 2, stroke=0, fill=1)
        c.setFillColor(self.away_color)
        c.roundRect(centre, y - height / 2, away_length, height,
                    height / 2, stroke=0, fill=1)

        # The leader's figure is printed in its own colour; the other stays
        # neutral, so the winner of each row is readable without the bar.
        leads = share >= 0.5
        c.setFillColor(self.home_color if leads else NEUTRAL)
        c.setFont("Helvetica-Bold", TYPE_COVER_FIGURE)
        c.drawRightString(left - 20, y - 7, home_text)
        c.setFillColor(self.away_color if not leads else NEUTRAL)
        c.drawString(left + width + 20, y - 7, away_text)

    def _cover_thesis(self, centre: float, top: float, text: str, measure: float):
        """The report's one-sentence finding, wrapped to a readable measure.

        Set as a single centred string it ran the full width of the sheet and
        touched both margins; a line that long is a banner, not a sentence.
        """
        c = self.canvas
        c.setFillColor(TEXT)
        c.setFont("Helvetica-Bold", TYPE_THESIS)
        lines, current = [], ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if c.stringWidth(candidate, "Helvetica-Bold", TYPE_THESIS) <= measure:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        for index, line in enumerate(lines[:3]):
            c.drawCentredString(centre, top - index * (TYPE_THESIS + 8), line)
        return top - (len(lines[:3]) - 1) * (TYPE_THESIS + 8)

    def _cover_pitch(self, y: float, height: float):
        """The centre of a pitch, faint, behind the cover's hero statistic.

        Only the halfway line, the centre circle and the two touchlines. A full
        pitch was drawn here first and its penalty areas sit exactly where the
        hero's two numbers do -- at 62pt those are the largest things on the
        sheet, and the box lines ran straight through them. These three marks
        say "pitch" on their own and occupy the middle of the band, which is
        the one part of it no text uses.
        """
        c = self.canvas
        left, width = 56, PAGE_W - 112
        middle = left + width / 2
        c.saveState()
        c.setStrokeColor(GRID)
        c.setFillColor(GRID)
        c.setStrokeAlpha(0.75)
        c.setFillAlpha(0.75)
        c.setLineWidth(1.0)
        c.line(left, y, left + width, y)
        c.line(left, y + height, left + width, y + height)
        c.line(middle, y, middle, y + height)
        c.circle(middle, y + height / 2, height * 0.30, stroke=1, fill=0)
        c.circle(middle, y + height / 2, 2.0, stroke=0, fill=1)
        c.restoreState()

    def _cover_fixture(self, centre: float, baseline: float):
        """Crest, name, score, name, crest — the line the visuals also carry.

        The report was the only part of the package without club crests once
        the visuals and the posters gained them.
        """
        c = self.canvas
        home, away = self.context["home"], self.context["away"]
        score = self.context["score"]
        crest, pad, gap = 36, 13, 20

        score_w = c.stringWidth(score, "Helvetica-Bold", TYPE_DISPLAY)
        home_w = c.stringWidth(home.upper(), "Helvetica-Bold", TYPE_FIXTURE)
        away_w = c.stringWidth(away.upper(), "Helvetica-Bold", TYPE_FIXTURE)
        home_badge = self._crest_reader(self.context.get("home_id"))
        away_badge = self._crest_reader(self.context.get("away_id"))
        lead_in = crest + pad if home_badge is not None else 0
        lead_out = crest + pad if away_badge is not None else 0

        total = lead_in + home_w + gap + score_w + gap + away_w + lead_out
        left = centre - total / 2
        x = left
        if home_badge is not None:
            c.drawImage(home_badge, x, baseline - 10, crest, crest,
                        mask="auto", preserveAspectRatio=True, anchor="c")
            x += lead_in
        c.setFillColor(self.home_color); c.setFont("Helvetica-Bold", TYPE_FIXTURE)
        c.drawString(x, baseline, home.upper())
        c.setFillColor(TEXT); c.setFont("Helvetica-Bold", TYPE_DISPLAY)
        c.drawString(x + home_w + gap, baseline - 9, score)
        c.setFillColor(self.away_color); c.setFont("Helvetica-Bold", TYPE_FIXTURE)
        c.drawString(x + home_w + gap + score_w + gap, baseline, away.upper())
        if away_badge is not None:
            c.drawImage(away_badge, x + home_w + gap + score_w + gap + away_w + pad,
                        baseline - 10, crest, crest,
                        mask="auto", preserveAspectRatio=True, anchor="c")

        # The two-colour rule every visual and poster closes its header with,
        # so the cover is recognisably the front of the same document.
        rule_y = baseline - 32
        c.setFillColor(self.home_color)
        c.rect(left, rule_y, total / 2, 2.4, stroke=0, fill=1)
        c.setFillColor(self.away_color)
        c.rect(left + total / 2, rule_y, total / 2, 2.4, stroke=0, fill=1)

    @staticmethod
    def _crest_reader(team_id):
        """An ImageReader for one club's crest, or None if there isn't one."""
        if team_id is None:
            return None
        try:
            if crests.crest_image(int(team_id)) is None:
                return None
            return ImageReader(str(crests.cache_path(int(team_id))))
        except Exception:
            return None

    def _cover_lead_bar(self, kind: str, name: str, note: str,
                        home_value: float, away_value: float, y: float):
        """The match's most lopsided percentage, at full width.

        ``kind`` decides the graphic. A "split" divides one bar between the two
        sides, because the pair sums to the whole. A "rate" gives each side its
        own bar against a common 100 baseline, because the two are independent
        percentages of different denominators and one divided bar would claim
        they add up.
        """
        c = self.canvas
        left, width = 56, PAGE_W - 112

        # A split bar sits on one track at y; the rate mode stacks two, and its
        # upper track ran through the bottom of the 62pt number beside it, so
        # the numbers are lifted clear in that mode.
        stacked = kind != "split"
        label_y = y + (124 if stacked else 92)
        home_y = y + (66 if stacked else 40)
        away_y = y + (56 if stacked else 30)

        c.setFillColor(MUTED); c.setFont("Helvetica-Bold", TYPE_CAPTION)
        c.drawString(left, label_y, f"{name.upper()}  ·  {note.upper()}")

        # Deliberately mismatched: the side that lost the battle is set smaller
        # so the pair reads as lopsided before either number is parsed.
        minor, major = ((TYPE_LEAD_MINOR, TYPE_LEAD_MAJOR) if away_value >= home_value
                        else (TYPE_LEAD_MAJOR, TYPE_LEAD_MINOR))
        c.setFillColor(self.home_color); c.setFont("Helvetica-Bold", minor)
        c.drawString(left, home_y, f"{home_value:.1f}%")
        c.setFillColor(self.away_color); c.setFont("Helvetica-Bold", major)
        c.drawRightString(left + width, away_y, f"{away_value:.1f}%")

        if kind == "split":
            total = max(home_value + away_value, 1e-6)
            home_w = width * home_value / total
            c.setFillColor(self.home_color); c.rect(left, y, home_w, 14, fill=1, stroke=0)
            c.setFillColor(self.away_color)
            c.rect(left + home_w, y, width - home_w, 14, fill=1, stroke=0)
        else:
            # Two tracks, separated enough to read as two measurements. Butted
            # together they looked like one two-tone bar, which is exactly the
            # split reading this mode exists to avoid.
            ceiling = max(home_value, away_value, 1e-6)
            for offset, value, colour in ((26, home_value, self.home_color),
                                          (0, away_value, self.away_color)):
                c.setFillColor(PANEL_2)
                c.rect(left, y + offset, width, 12, fill=1, stroke=0)
                c.setFillColor(colour)
                c.rect(left, y + offset, width * value / ceiling, 12, fill=1, stroke=0)

        c.setFont("Helvetica-Bold", TYPE_SECTION)
        c.setFillColor(self.home_color)
        c.drawString(left, y - 22, self.context["home"].upper())
        c.setFillColor(self.away_color)
        c.drawRightString(left + width, y - 22, self.context["away"].upper())

    def _cover_strip(self, y: float, exclude: str | None = None):
        """Supporting splits under the lead statistic.

        ``exclude`` drops whichever metric the hero bar already carries —
        printing FIELD TILT as the headline and again in the strip below spent
        a cell restating a number the reader had just been shown.
        """
        c = self.canvas
        left, width = 56, PAGE_W - 112
        cells = [
            ("Expected goals", self.context.get("home_xG"), self.context.get("away_xG"), "{:.2f}"),
            ("Shots", self.context.get("home_shots"), self.context.get("away_shots"), "{:.0f}"),
            ("Box entries", self.context.get("home_box_entries"), self.context.get("away_box_entries"), "{:.0f}"),
            ("Field tilt", self.context.get("home_field_tilt"), self.context.get("away_field_tilt"), "{:.1f}"),
            ("Possession", self.context.get("home_possession_share"), self.context.get("away_possession_share"), "{:.1f}"),
        ]
        if exclude:
            cells = [cell for cell in cells if cell[0].lower() != exclude.lower()]
        cells = cells[:4]
        step = width / len(cells)
        for idx, (label, home_value, away_value, fmt) in enumerate(cells):
            x = left + idx * step
            c.setFillColor(MUTED); c.setFont("Helvetica-Bold", TYPE_MICRO)
            c.drawString(x, y + 34, label.upper())
            try:
                home_text, away_text = fmt.format(float(home_value)), fmt.format(float(away_value))
            except (TypeError, ValueError):
                continue
            c.setFillColor(self.home_color); c.setFont("Helvetica-Bold", TYPE_TITLE)
            c.drawString(x, y + 10, home_text)
            offset = c.stringWidth(home_text, "Helvetica-Bold", 19)
            c.setFillColor(NEUTRAL); c.setFont("Helvetica-Bold", TYPE_SECTION)
            c.drawString(x + offset + 6, y + 10, "/")
            c.setFillColor(self.away_color); c.setFont("Helvetica-Bold", TYPE_TITLE)
            c.drawString(x + offset + 18, y + 10, away_text)
        c.setStrokeColor(GRID); c.setLineWidth(0.6)
        c.line(left, y - 12, left + width, y - 12)

    def executive_summary(self, sections: dict[str, dict]):
        self._start("executive_summary", "Executive Summary")
        self._header("Executive Summary", "The result, the mechanism and the main coaching implications", "REPORT OPEN")
        c = self.canvas
        c.setFillColor(TEXT); c.setFont("Helvetica-Bold", TYPE_TITLE)
        # Same sentence as the cover, from the same numbers — the summary used
        # to assert the winner had created more regardless of whether they had.
        c.drawString(42, PAGE_H - 132, self._verdict()[:132])
        bullets = [
            sections["Match Story"]["data"][1],
            sections["Chance Creation"]["data"][0],
            sections["Possession and Progression"]["data"][1],
            sections["Pressing and Rest Defence"]["data"][3],
            sections["Transitions and Efficiency"]["data"][0],
        ]
        # Every card used to be a fixed 135pt tall for one line of text, which
        # left each one about nine tenths empty and pushed the block into the
        # bottom half of an otherwise blank page. Height now follows the text.
        column_w = 444
        wide_w = PAGE_W - 84
        top = PAGE_H - 190
        row_gap, col_gap = 18, 36

        def card_height(text: str, width: float) -> float:
            paragraph = Paragraph(escape(text), self.card)
            _, text_h = paragraph.wrap(width - 28, PAGE_H)
            return max(text_h + 52, 74)

        pairs = list(zip(bullets[:4], range(1, 5)))
        row_heights = [
            max(card_height(body, column_w) for (_t, body), _i in pairs[start:start + 2])
            for start in (0, 2)
        ]
        wide_preview = card_height(bullets[4][1], wide_w)

        # Cards sized to their text leave slack on a page this tall. Spend it on
        # the gaps between them rather than letting it pool into one hole at the
        # foot of the page — the block then reads as laid out, not as stranded.
        strip_block = 100
        content = sum(row_heights) + wide_preview + strip_block
        slack = (top - 150) - content - (2 * row_gap + 46)
        if slack > 0:
            row_gap += min(slack / 3, 34)

        y = top
        for row_start in (0, 2):
            row = pairs[row_start:row_start + 2]
            row_h = row_heights[row_start // 2]
            for column, ((title, body), idx) in enumerate(row):
                x = 42 + column * (column_w + col_gap)
                accent = self.home_color if idx % 2 else self.away_color
                self._card_box(x, y - row_h, column_w, row_h, f"{idx:02d} - {title}", accent)
                self._paragraph(escape(body), x + 14, y - 38, column_w - 28, row_h - 50, self.card)
            y -= row_h + row_gap

        title, body = bullets[4]
        wide_h = card_height(body, wide_w)
        self._card_box(42, y - wide_h, wide_w, wide_h, f"05 - {title}", FOCUS)
        self._paragraph(escape(body), 56, y - 38, wide_w - 28, wide_h - 50, self.card)
        y -= wide_h + 46

        # Sizing the cards to their text freed most of the lower half. The most
        # read page in the report should spend that on evidence rather than on
        # air, so the headline splits go underneath — the same four the cover
        # uses, which is what a reader arriving from page 01 expects to see.
        if y > 150:
            c.setFillColor(MUTED); c.setFont("Helvetica-Bold", TYPE_MICRO)
            c.drawString(42, y, "THE FOUR SPLITS BEHIND THE VERDICT")
            self._cover_strip(y=y - 52)
        self._finish()

    def toc(self, entries: list[tuple[str, int, str]]):
        self._start("contents", "Contents")
        self._header("Report Contents", "A performance-analysis reading path followed by the complete player appendix", "NAVIGATION")
        c = self.canvas
        y = PAGE_H - 150
        for idx, (title, page, subtitle) in enumerate(entries, start=1):
            # The marker used to alternate between the two team colours by row
            # number, which encoded nothing at all — section 02 is not "the away
            # team's section". One neutral rule per row instead.
            c.setFillColor(NEUTRAL)
            c.rect(56, y - 12, 2, 26, fill=1, stroke=0)
            c.setFillColor(TEXT); c.setFont("Helvetica-Bold", TYPE_SECTION)
            label = f"{idx:02d}  {title}"
            c.drawString(74, y, label)
            c.setFillColor(MUTED); c.setFont("Helvetica", TYPE_CAPTION)
            c.drawString(74, y - 15, subtitle[:106])

            # Leader rule starts where the title ends rather than at a fixed
            # x, so it joins the two sides of the row instead of floating.
            rule_start = 74 + c.stringWidth(label, "Helvetica-Bold", 11) + 12
            page_label = f"PAGE {page:02d}"
            rule_end = PAGE_W - 55 - c.stringWidth(page_label, "Helvetica-Bold", 10) - 12
            if rule_end > rule_start:
                c.setStrokeColor(GRID); c.setLineWidth(0.6)
                c.line(rule_start, y + 3, rule_end, y + 3)
            c.setFillColor(MUTED); c.setFont("Helvetica-Bold", TYPE_SECTION)
            c.drawRightString(PAGE_W - 55, y, page_label)
            y -= 57
        self._finish()

    def section_page(self, title: str, copy: dict, section_index: int):
        bookmark = "section_" + title.lower().replace(" ", "_").replace("&", "and")
        self._start(bookmark, title)
        self._header(title, copy["subtitle"], f"SECTION {section_index:02d}")
        self._card_box(42, 290, 444, 270, "Performance Analyst View", HOME)
        self._card_box(522, 290, 444, 270, "Data Analyst Evidence", AWAY)
        for x, rows in [(42, copy["performance"]), (522, copy["data"])]:
            top = 520
            for idx, (label, body) in enumerate(rows, start=1):
                self.canvas.setFillColor(FOCUS); self.canvas.setFont("Helvetica-Bold", TYPE_CAPTION)
                self.canvas.drawString(x + 14, top, f"{idx:02d}  {label.upper()}")
                height = self._paragraph(escape(body), x + 14, top - 10, 416, 55, self.small)
                top -= max(62, height + 29)
        self._card_box(42, 97, PAGE_W - 84, 140, "Tactical Implication", FOCUS)
        self._paragraph(escape(copy["implication"]), 60, 202, PAGE_W - 120, 75, self.body)
        self.canvas.setFillColor(NEUTRAL); self.canvas.setFont("Helvetica", TYPE_MICRO)
        self.canvas.drawString(60, 118, "Use the following visuals as evidence for this section. Read the explanation and next analytical step below every chart.")
        self._finish()

    def verdict(self):
        self._start("final_verdict", "Final Tactical Verdict")
        self._header("Final Tactical Verdict", "A joined performance and data conclusion", "SYNTHESIS")
        c = self.canvas
        home, away = self.context["home"], self.context["away"]
        c.setFillColor(TEXT); c.setFont("Helvetica-Bold", TYPE_TITLE)
        c.drawString(42, PAGE_H - 142, f"{self.context['winner']} controlled the decisive moments, not every phase of the match.")
        self._card_box(42, 365, 444, 170, "Why the winner won", AWAY if self.context["winner"] == away else HOME)
        winner_side = "home" if self.context["winner"] == home else "away"
        winner_text = (
            f"The winning side combined {self.context[f'{winner_side}_xG']:.2f} xG-level chance production with the stronger transition conversion and superior final execution. "
            "They did not need to dominate every territorial metric because their best attacks arrived before the opponent could restore compactness."
        )
        self._paragraph(escape(winner_text), 58, 495, 412, 105, self.body)
        self._card_box(522, 365, 444, 170, "Why the loser remained dangerous", HOME if self.context["winner"] == away else AWAY)
        loser_text = (
            f"The losing side still generated repeated final-third access, pressure and sequence value. Their problem was conversion: territory and activity did not produce the same shot quality, while greater attacking commitment increased exposure behind the ball."
        )
        self._paragraph(escape(loser_text), 538, 495, 412, 105, self.body)
        self._card_box(42, 117, PAGE_W - 84, 190, "Coaching priorities", FOCUS)
        priorities = [
            "1. Protect the first pass after losing possession: rest-defence spacing must be set before the final-third action.",
            "2. Improve box-entry selection: create a clean shot or a controlled second phase rather than forcing the first available action.",
            "3. Connect pressing triggers to cover: the nearest pressure, central screen and back-line depth must move as one unit.",
            "4. Review all conclusions by game state: leading and trailing phases created different risk and possession incentives.",
        ]
        top = 270
        for item in priorities:
            self._paragraph(escape(item), 60, top, PAGE_W - 120, 34, self.card)
            top -= 36
        self._finish()

    def methodology(self):
        self._start("methodology", "Methodology and Caveats")
        self._header("Methodology and Caveats", "Definitions and limits needed to interpret a single-match report", "TRUST LAYER")
        left = [
            ("xG", "Expected-goal value estimates chance quality before the shot outcome."),
            ("xGoT", "Post-shot expected goals evaluate the quality of attempts that reached the target."),
            ("xT", "Expected threat values ball progression by the change in scoring potential between locations."),
            ("PPDA", "Opponent passes allowed per defensive action in the pressing zone; lower values indicate more aggressive pressure."),
        ]
        right = [
            ("Single-match sample", "Finishing, transition conversion and player radar extremes can be highly volatile."),
            ("Game-state effect", "A team protecting a lead and a team chasing it face different incentives; totals are not tactically neutral."),
            ("Assists", "When the source assist field is empty, the report infers the last successful key pass within 15 seconds of the goal and should be read as a derived assist."),
            ("Player profiles", "The radar pages describe match contribution, not long-term player quality or recruitment-grade percentiles."),
        ]
        for x, title, rows, accent in [(42, "Metric definitions", left, HOME), (522, "Interpretation limits", right, AWAY)]:
            self._card_box(x, 162, 444, 390, title, accent)
            top = 512
            for label, body in rows:
                self.canvas.setFillColor(FOCUS); self.canvas.setFont("Helvetica-Bold", TYPE_CAPTION)
                self.canvas.drawString(x + 16, top, label.upper())
                self._paragraph(escape(body), x + 16, top - 12, 410, 55, self.card)
                top -= 83
        self._finish()

    def visual(self, path: Path, section: str, next_path: Path | None = None):
        self._start()
        c = self.canvas
        with Image.open(path) as image:
            iw, ih = image.size
        # The image used to bleed to 4pt from the page edge while the
        # commentary underneath began at 42pt, so a wide visual and its own
        # analysis sat on two different left edges. Sharing the text margin
        # costs a wide dashboard about 76pt of width and buys a page that
        # lines up. A tall visual is still centred — nothing can align an
        # image whose shape does not match the column.
        margin_x = TEXT_MARGIN
        image_region_h = PAGE_H - VISUAL_NOTE_H
        scale = min((PAGE_W - 2 * margin_x) / iw, image_region_h / ih)
        width, height = iw * scale, ih * scale
        x = (PAGE_W - width) / 2
        y = VISUAL_NOTE_H + (image_region_h - height) / 2
        c.drawImage(ImageReader(str(path)), x, y, width=width, height=height, preserveAspectRatio=True, mask="auto")
        # Pure black, matching the visual sitting above it. Filling this band
        # with PANEL put #0A0A0A against the image's #000000 and drew a visible
        # horizontal seam across every visual page; the team rule below is what
        # separates the two areas, not a change of ground.
        c.setFillColor(BG)
        c.rect(0, 0, PAGE_W, VISUAL_NOTE_H, fill=1, stroke=0)
        # The same two-tone team rule that tops every rendered visual, repeated
        # here so the commentary band reads as part of the same document rather
        # than as a caption bolted underneath it.
        c.setFillColor(self.home_color); c.rect(0, VISUAL_NOTE_H - 2.5, PAGE_W / 2, 2.5, fill=1, stroke=0)
        c.setFillColor(self.away_color); c.rect(PAGE_W / 2, VISUAL_NOTE_H - 2.5, PAGE_W / 2, 2.5, fill=1, stroke=0)

        title = visual_commentary_title(path, self.context)
        self._paragraph(escape(title), 42, VISUAL_NOTE_H - 24, PAGE_W - 84, 24, self.commentary_title)

        # Two columns. The page is 14 inches wide, so a single measure ran to
        # roughly 830pt at 9pt type — far past the length an eye can track back
        # from. Splitting at a sentence boundary halves the measure.
        narrative = visual_narrative(path, self.context)
        left_text, right_text = _split_for_columns(narrative)
        gutter = 34
        column_w = (PAGE_W - 84 - gutter) / 2
        top = VISUAL_NOTE_H - 52
        self._paragraph(escape(left_text), 42, top, column_w, 165, self.commentary_body)
        if right_text:
            self._paragraph(escape(right_text), 42 + column_w + gutter, top, column_w, 165, self.commentary_body)
            c.setStrokeColor(GRID); c.setLineWidth(0.6)
            c.line(42 + column_w + gutter / 2, top - 158, 42 + column_w + gutter / 2, top + 4)

        c.setStrokeColor(GRID); c.setLineWidth(0.5); c.line(42, 31, PAGE_W - 42, 31)
        self._paragraph(escape(next_visual_step(next_path)), 42, 25, PAGE_W - 250, 18, self.commentary_next)
        c.setFillColor(NEUTRAL); c.setFont("Helvetica-Bold", TYPE_MICRO); c.drawRightString(PAGE_W - 24, 10, f"{section.upper()}  |  PAGE {self.page:02d}")
        c.showPage()

    def save(self):
        self.canvas.save()


def _ordered_section_paths(paths: Iterable[Path]) -> dict[str, list[Path]]:
    order = [
        "Match Story",
        "Chance Creation",
        "Possession and Progression",
        "Pressing and Rest Defence",
        "Transitions and Efficiency",
        "Player Impact Appendix",
    ]
    groups = {title: [] for title in order}
    for path in paths:
        groups[classify_visual(Path(path))].append(Path(path))
    for title in order:
        groups[title] = sorted(groups[title], key=lambda path: path.name.lower())
    return groups


def build_tactical_pdf(
    paths: list[Path],
    output: Path,
    events: pd.DataFrame,
    xg: pd.DataFrame,
    team_metrics: pd.DataFrame,
    player_metrics: pd.DataFrame,
    match_info: dict,
) -> Path:
    global HOME, AWAY
    # Use the fixture's resolved kit colours. These were hard-coded to the role
    # pair, so the PDF framed images drawn in the teams' real colours with a
    # blue/yellow border belonging to neither side.
    home_hex = str((match_info or {}).get("home_color") or "").strip() or "#2F5BFF"
    away_hex = str((match_info or {}).get("away_color") or "").strip() or "#FFD400"
    HOME = _as_pdf_color(home_hex, _DEFAULT_HOME)
    AWAY = _as_pdf_color(away_hex, _DEFAULT_AWAY)
    output.parent.mkdir(parents=True, exist_ok=True)
    valid_paths = [Path(path).resolve() for path in paths if Path(path).exists()]
    context = build_context(events, xg, team_metrics, player_metrics, match_info)
    context["home_color"] = home_hex
    context["away_color"] = away_hex
    section_copy = _section_copy(context)
    groups = _ordered_section_paths(valid_paths)
    core = [
        "Match Story",
        "Chance Creation",
        "Possession and Progression",
        "Pressing and Rest Defence",
        "Transitions and Efficiency",
    ]
    visual_sequence = [path for title in core for path in groups[title]] + groups["Player Impact Appendix"]
    next_visual = {
        path.resolve(): (visual_sequence[index + 1] if index + 1 < len(visual_sequence) else None)
        for index, path in enumerate(visual_sequence)
    }

    page_cursor = 4
    toc_entries: list[tuple[str, int, str]] = []
    for title in core:
        toc_entries.append((title, page_cursor, section_copy[title]["subtitle"]))
        page_cursor += 1 + len(groups[title])
    verdict_page = page_cursor
    toc_entries.append(("Final Tactical Verdict", verdict_page, "Joined performance and data conclusion with coaching priorities"))
    page_cursor += 1
    methodology_page = page_cursor
    toc_entries.append(("Methodology and Caveats", methodology_page, "Metric definitions, game-state context and single-match limitations"))
    page_cursor += 1
    appendix_page = page_cursor
    toc_entries.append(("Player Impact Appendix", appendix_page, section_copy["Player Impact Appendix"]["subtitle"]))

    report = TacticalPDF(output, context)
    report.cover()
    report.executive_summary(section_copy)
    report.toc(toc_entries)
    for idx, title in enumerate(core, start=1):
        report.section_page(title, section_copy[title], idx)
        for path in groups[title]:
            report.visual(path, title, next_visual[path.resolve()])
    report.verdict()
    report.methodology()
    report.section_page("Player Impact Appendix", section_copy["Player Impact Appendix"], len(core) + 1)
    for path in groups["Player Impact Appendix"]:
        report.visual(path, "Player Impact", next_visual[path.resolve()])
    report.save()
    return output
