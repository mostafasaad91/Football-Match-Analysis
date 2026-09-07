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
from visualization_components import (
    C_AWAY,
    C_HOME,
    IS_LIGHT_THEME,
    USE_REAL_TEAM_KIT_COLORS,
)


PAGE_W = 14 * 72
PAGE_H = 12.0 * 72
BASE_PAGE_H = 9 * 72
VISUAL_NOTE_H = PAGE_H - BASE_PAGE_H


# ── The cover's face ─────────────────────────────────────────────────────────
#
# The report was set entirely in Helvetica, which is what reportlab gives you
# for free and reads as the absence of a decision. The cover is the one surface
# that is looked at rather than read, and a condensed grotesque is the register
# the rest of football's graphics are in.
#
# Condensed is the practical half of the argument as well as the aesthetic one:
# Bahnschrift sets "POSSESSION" 9% narrower than Helvetica-Bold at the same
# size, so the labels can be set larger without the letter-spaced caps growing
# wider than the card.
#
# Every candidate is a font Windows ships, and the chain ends at Helvetica, so
# a machine without any of them still builds the same report in the face it
# always used. Nothing about the layout depends on which one wins.
# Two faces, because the two jobs on this page want opposite things.
#
# The display line — the score, the club names, the figure at each end of a bar
# — is large enough that weight is not what makes it legible, and a condensed
# grotesque is what gives the page its character. Bahnschrift ships only as a
# variable font and reportlab takes its default instance, which is light; at
# 44pt that reads as elegant and at 11pt it reads as faint.
#
# So the small letter-spaced caps get a face with a real bold instead. They are
# the ones that were unreadable, and weight is the whole fix.
_DISPLAY_CANDIDATES = (
    ("Bahnschrift", "bahnschrift.ttf"),
    ("SegoeUI-Semilight", "segoeuisl.ttf"),
)
_TEXT_CANDIDATES = (
    ("SegoeUI-Bold", "segoeuib.ttf"),
    ("Tahoma-Bold", "tahomabd.ttf"),
    ("Verdana-Bold", "verdanab.ttf"),
)

_FONT_DIRS = (
    Path("C:/Windows/Fonts"),
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
    Path("/usr/share/fonts"),
    Path("/Library/Fonts"),
)


def _register_first(candidates, fallback: str) -> str:
    """Register the first candidate that exists, or keep the built-in face.

    Attempted once at import. A font that fails — missing, unreadable, or a
    format reportlab will not take — is skipped rather than raised on: a report
    in the wrong face is a better outcome than no report, and the chain ends at
    a face reportlab always has.
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for name, filename in candidates:
        for directory in _FONT_DIRS:
            path = directory / filename
            if not path.exists():
                continue
            try:
                pdfmetrics.registerFont(TTFont(name, str(path)))
                return name
            except Exception:
                continue
    return fallback


COVER_DISPLAY = _register_first(_DISPLAY_CANDIDATES, "Helvetica-Bold")
COVER_TEXT = _register_first(_TEXT_CANDIDATES, "Helvetica-Bold")

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
    # Cover-only greys. Letter-spaced small caps lose stroke weight, so they
    # are held well above the 4.5:1 floor rather than at it.
    COVER_LABEL = colors.HexColor("#3A3F47")   # 9.9:1 on the light page
    COVER_META = colors.HexColor("#4A5058")    # 7.9:1
    # The trailing side's figure on each row. Quieter than the leader's, which
    # keeps its kit colour, but still a number a reader has to be able to read:
    # it is half the comparison the row exists to make.
    COVER_FIGURE_DIM = colors.HexColor("#666D78")   # 5.0:1
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
    COVER_LABEL = colors.HexColor("#C8C8C8")   # 11.6:1 on the black page
    COVER_META = colors.HexColor("#A8A8A8")    # 8.0:1
    COVER_FIGURE_DIM = colors.HexColor("#909090")   # 5.7:1

# Fixture colours fall back to these when the caller supplies none, so the
# fallback has to follow the page too.
_DEFAULT_HOME, _DEFAULT_AWAY = HOME, AWAY

# Structural marks — section numbers, card rules, the spine label. These name
# parts of the report, not parts of the match, so they must not wear a colour
# that competes with the two teams. The previous amber did: it ran to 1,398
# characters against 648 for both kit colours combined, which made a fixed
# accent, rather than the fixture, the loudest thing in the document.
# Structural accent stays deliberately neutral; team identity is carried by the
# teal/coral brand pair so section chrome never reads as a third team.
FOCUS = colors.HexColor("#5B4A45") if IS_LIGHT_THEME else colors.HexColor("#A7B2B5")

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

# The cover's small type had been borrowed from the body pages, where 6.5pt
# grey sits inside a dense column and reads as a footnote should. On a cover it
# is the only text between the score and the figures, at arm's length, and the
# row labels are the part that says what each bar measures — "POSSESSION" is
# not a footnote, it is the axis.
#
# The greys were the other half. NEUTRAL measures 3.0:1 against the black page,
# under the 4.5:1 floor before the letter-spacing thins the strokes further,
# and the byline and the source line were both drawn in it.
#
# Set against the condensed face, which is why they are larger than they look:
# Bahnschrift at 11.5 occupies about the width Helvetica-Bold did at 10.5, so
# the labels grew without the card having to.
TYPE_COVER_LABEL, TYPE_COVER_META = 11.5, 10.0

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
BRAND = colors.HexColor("#B94F3D") if IS_LIGHT_THEME else colors.HexColor("#23C7B7")

# The publisher's badge. Absent on a fresh clone, so every use is guarded and
# the cover falls back to a typographic wordmark rather than failing.
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.jpg"


def _as_pdf_color(value, fallback):
    """Return a reportlab colour for a hex string, or the fallback if unusable."""
    try:
        return colors.HexColor(str(value).strip())
    except (ValueError, AttributeError, TypeError):
        return fallback
VALUE = colors.HexColor("#A33B2E") if IS_LIGHT_THEME else colors.HexColor("#FF6B5E")


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
    from match_metrics import pitch_control
    context['influence'] = pitch_control(events, home_id, away_id)[1]
    context['date'] = match_info.get('date', '')
    context['competition'] = match_info.get('competition', '')
    context['url'] = match_info.get('url', '')
    context['match_id'] = match_info.get('match_id', '')
    context['chart_contracts'] = events.attrs.get('chart_contracts', {})
    for side in ('home', 'away'):
        for state in ('drawing', 'leading', 'trailing'):
            key = f'game_state_{state}_xG'
            if key in team_metrics.columns:
                context[f'{side}_{key}'] = _metric(team_metrics, side, key)
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
    return f"at {minute:02d}:{second:02d} elapsed"




MATCH_STORY = {"01", "04", "14", "15", "18", "23", "43"}
CHANCE_CREATION = {"02", "03", "11", "12", "13", "16", "17", "26", "27", "34", "35"}
POSSESSION = {"05a", "05b", "06a", "06b", "07", "08", "09", "10", "20", "21", "22", "24", "25", "31a", "31b", "32a", "32b", "33", "38", "39"}
PRESSING = {"28", "29", "30", "36", "37", "40"}
TRANSITIONS = {"41", "42"}




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








def next_visual_step(next_path: Path | None) -> str:
    if next_path is None:
        return "This closes the visual appendix. Return to the Final Tactical Verdict to connect the player roles with the team-level coaching priorities."
    return f"Next visual: {_visual_title(next_path)}. Compare its period, population and denominator before combining the evidence."


class TacticalPDF:
    def __init__(self, output: Path, context: dict):
        # Embed a Unicode-capable family rather than relying on viewer fonts.
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from matplotlib import get_data_path
        font_root = Path(get_data_path()) / 'fonts' / 'ttf'
        for name, filename in [('Helvetica','DejaVuSans.ttf'),('Helvetica-Bold','DejaVuSans-Bold.ttf'),('Helvetica-Oblique','DejaVuSans-Oblique.ttf'),('Helvetica-BoldOblique','DejaVuSans-BoldOblique.ttf')]:
            pdfmetrics.registerFont(TTFont(name, str(font_root / filename)))
        pdfmetrics.registerFontFamily('Helvetica', normal='Helvetica', bold='Helvetica-Bold', italic='Helvetica-Oblique', boldItalic='Helvetica-BoldOblique')
        self.output = output
        self.context = context
        # Filled by build_tactical_pdf from the Word-style article. Each
        # visual can therefore carry the same argued reading as the DOCX.
        self.article_readings: dict[str, str] = {}
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
        # NEUTRAL is 3.0:1 against the black page, under the 4.5:1 floor, and
        # this is the one line that appears on all seventy-four of them.
        self.canvas.setFillColor(COVER_META)
        self.canvas.setFont(COVER_TEXT, TYPE_COVER_META)
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
        # New report identity: a teal/coral split ribbon and a neutral spine
        # make every section recognisable even when the embedded visual is dark.
        c.setFillColor(BRAND); c.roundRect(24, base, 7, panel_h, 3, fill=1, stroke=0)
        c.setFillColor(VALUE); c.rect(PAGE_W - 31, base, 7, panel_h, fill=1, stroke=0)
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
        import copy
        fitted = copy.copy(style or self.body)
        paragraph = Paragraph(text, fitted)
        _, height = paragraph.wrap(width, max_height)
        floor = min(8.0, fitted.fontSize)
        while height > max_height + .1 and fitted.fontSize > floor:
            fitted.fontSize = max(floor, fitted.fontSize-.25)
            fitted.leading = fitted.fontSize*1.3
            paragraph = Paragraph(text, fitted)
            _, height = paragraph.wrap(width, max_height)
        if height > max_height + .1:
            raise ValueError(f'PDF text exceeds its allotted area on page {self.page}: {text[:80]}')
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
        ("FIELD TILT",          "field_tilt",          "{:.0f}%"),
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

        # Carry the report identity onto the cover itself. The old cover only
        # exposed team colours at the bottom, so it looked identical to the
        # previous edition until the reader reached an inner section.
        c.setFillColor(BRAND)
        c.rect(0, 0, 12, PAGE_H, stroke=0, fill=1)
        c.setFillColor(VALUE)
        c.rect(PAGE_W - 12, 0, 12, PAGE_H, stroke=0, fill=1)
        c.setFillColor(BRAND)
        c.roundRect(COVER_MARGIN, PAGE_H - 112, 118, 18, 8, stroke=0, fill=1)
        c.setFillColor(BG)
        c.setFont(COVER_TEXT, 7.5)
        c.drawCentredString(COVER_MARGIN + 59, PAGE_H - 106, "TACTICAL MATCH REPORT")

        top = PAGE_H - 54
        c.setFillColor(COVER_LABEL)
        c.setFont(COVER_TEXT, TYPE_COVER_LABEL)
        competition = self._cover_competition().upper()
        c.drawString(COVER_MARGIN, top, _spaced_out(competition or "MATCH ANALYSIS"))
        if competition:
            c.setFillColor(COVER_META)
            c.setFont(COVER_TEXT, TYPE_COVER_META)
            # The article's own headline, not "MATCH ANALYSIS". The report had
            # no tactical line anywhere on its front — the card says what the
            # totals were and nothing said what the match was — while the
            # article next to it opened on a sentence derived from these same
            # frames. One finding, both documents, or they drift.
            self._paragraph(escape(self._cover_headline()), COVER_MARGIN, top-12,
                            PAGE_W-2*COVER_MARGIN-90, 35,
                            ParagraphStyle('cover_thesis', fontName=COVER_TEXT, fontSize=TYPE_COVER_META, leading=13, textColor=COVER_META))
        else:
            self._paragraph(escape(self._cover_headline()), COVER_MARGIN, top-12,
                            PAGE_W-2*COVER_MARGIN-90, 35,
                            ParagraphStyle('cover_thesis', fontName=COVER_TEXT, fontSize=TYPE_COVER_META, leading=13, textColor=COVER_META))
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

        c.setFillColor(COVER_META)
        c.setFont(COVER_TEXT, TYPE_COVER_META)
        byline = str(self.context.get("byline") or "MOSTAFA SAAD").upper()
        c.drawString(COVER_MARGIN, foot_rule - 23, _spaced_out(byline))
        c.drawRightString(PAGE_W - COVER_MARGIN, foot_rule - 23,
                          _spaced_out("WHOSCORED / OPTA EVENT DATA"))

        # The two-colour rule every visual and poster closes on.
        c.setFillColor(self.home_color)
        c.rect(0, 0, PAGE_W / 2, 5, stroke=0, fill=1)
        c.setFillColor(self.away_color)
        c.rect(PAGE_W / 2, 0, PAGE_W / 2, 5, stroke=0, fill=1)
        self._finish()

    def _cover_headline(self) -> str:
        """The tactical line, read from the article's own candidates.

        Built from the same frames the card is, so the two documents cannot
        disagree about what the match was. Falls back to the old wording rather
        than raising: a report with a generic subtitle is a better outcome than
        no report.
        """
        line = str(self.context.get("headline") or "").strip()
        return line.upper() if line else "MATCH ANALYSIS"

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
                c.setFont(COVER_DISPLAY, TYPE_COVER_MARK)
                c.drawCentredString(x, y - 10, _club_initials(name))
            c.setFillColor(TEXT)
            c.setFont(COVER_DISPLAY, TYPE_COVER_TEAM)
            c.drawCentredString(x, y - crest / 2 - 25, str(name).upper())

        c.setFillColor(TEXT)
        c.setFont(COVER_DISPLAY, TYPE_COVER_SCORE)
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

        # The label is what the row measures, so it is set at the cover's own
        # size rather than the body pages' footnote size.
        c.setFillColor(COVER_LABEL)
        c.setFont(COVER_TEXT, TYPE_COVER_LABEL)
        c.drawCentredString(centre, y + 16, _spaced_out(label))

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
        c.setFillColor(self.home_color if leads else COVER_FIGURE_DIM)
        c.setFont(COVER_DISPLAY, TYPE_COVER_FIGURE)
        c.drawRightString(left - 20, y - 7, home_text)
        c.setFillColor(self.away_color if not leads else COVER_FIGURE_DIM)
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
        self._paragraph(escape(self.context.get('headline', 'Match evidence')),42,PAGE_H-124,PAGE_W-84,48,self.body)
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
            ("Local post-shot estimate", "Placement-weighted pre-shot xG. Uncalibrated; no shot velocity or actual goalkeeper position. Do not treat it as measured goals prevented."),
            ("xT", "Expected threat values ball progression by the change in scoring potential between locations."),
            ("PPDA", "Opponent passes per defensive action in the pressing zone. Lower means more frequent actions relative to passes, not necessarily a better press."),
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
        narrative = self.article_readings.get(path.name) or visual_narrative(path, self.context)
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
    # Fixed roles are the production default and must also override colours
    # stored in packages created before that decision. Kit colours are read
    # only in the explicit opt-in mode.
    if USE_REAL_TEAM_KIT_COLORS:
        home_hex = str((match_info or {}).get("home_color") or "").strip() or C_HOME
        away_hex = str((match_info or {}).get("away_color") or "").strip() or C_AWAY
    else:
        home_hex, away_hex = C_HOME, C_AWAY
    HOME = _as_pdf_color(home_hex, _DEFAULT_HOME)
    AWAY = _as_pdf_color(away_hex, _DEFAULT_AWAY)
    output.parent.mkdir(parents=True, exist_ok=True)
    valid_paths = [Path(path).resolve() for path in paths if Path(path).exists()]
    context = build_context(events, xg, team_metrics, player_metrics, match_info)
    context["home_color"] = home_hex
    context["away_color"] = away_hex
    # The article's headline, computed here because this is where the frames
    # are. The cover reads it off the context; match_article derives it from
    # the same candidates that write the article's own title.
    try:
        from match_article import cover_headline

        context["headline"] = cover_headline(
            events, xg, team_metrics, player_metrics, match_info)
    except Exception:
        context["headline"] = ""

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
    # Reuse the article's paragraphs under the corresponding visuals. This
    # keeps PDF and Word aligned: the chart remains evidence, while the prose
    # explains mechanism, game state and coaching meaning.
    try:
        from match_article import build_article
        article = build_article(events, xg, team_metrics, player_metrics, match_info, output.parent)
        for section in article.sections:
            prose = " ".join(str(p).strip() for p in section.paragraphs if str(p).strip())
            for visual_path in section.visuals:
                report.article_readings[Path(visual_path).name] = prose
    except Exception:
        pass
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


# Legacy templates remain private for archived-report compatibility. Production
# uses one evidence-led contract shared with the article.
from match_editorial import (section_copy as _section_copy,
    visual_section as classify_visual, reading as visual_explanation,
    reading as visual_narrative, commentary_title as visual_commentary_title,
    reading as visual_data_read, reading as visual_implication)
from publication_v2 import pdf_verdict
TacticalPDF.verdict = pdf_verdict
