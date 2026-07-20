from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from match_report import compute_ppda_both


PAGE_W = 14 * 72
PAGE_H = 12.0 * 72
BASE_PAGE_H = 9 * 72
VISUAL_NOTE_H = PAGE_H - BASE_PAGE_H

BG = colors.HexColor("#000000")
PANEL = colors.HexColor("#08090B")
PANEL_2 = colors.HexColor("#0D0F12")
GRID = colors.HexColor("#252A31")
TEXT = colors.HexColor("#F5F7FA")
MUTED = colors.HexColor("#9BA3AE")
NEUTRAL = colors.HexColor("#626A75")
HOME = colors.HexColor("#9A99B4")
AWAY = colors.HexColor("#A83246")
FOCUS = colors.HexColor("#FFD43B")
VALUE = colors.HexColor("#9A7CF2")


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
        rows.append(
            {
                "minute": int(float(goal.get("minute", 0))),
                "team_id": team_id,
                "team": team_names.get(team_id, str(team_id)),
                "player": str(goal.get("player", "Goal")),
                "score": dict(running),
            }
        )
    timeline = " | ".join(f"{row['minute']}' {row['player'].split()[-1]} ({row['team']})" for row in rows)
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
        if series.empty:
            return "No qualifying player"
        return f"{series.index[0]} ({fmt.format(float(series.iloc[0]))})"

    def frame_leader(data: pd.DataFrame, key: str, fmt: str) -> str:
        if data.empty or key not in data:
            return "No qualifying player"
        row = data.iloc[0]
        return f"{row['player']} ({fmt.format(float(row[key]))})"

    return {
        "goals": leader(goal_counts, "{:.0f} goals"),
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
        "score": f"{home_goals} - {away_goals}",
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


def _section_copy(c: dict) -> dict[str, dict]:
    home, away = c["home"], c["away"]
    winner, loser = c["winner"], c["loser"]
    early = c["goal_rows"][0] if c["goal_rows"] else None
    early_text = (
        f"{early['team']} scored through {early['player']} in minute {early['minute']}, forcing the opponent to operate from a chasing game state."
        if early else "The score state did not create a clear early tactical constraint."
    )
    press_team = home if c["home_ppda"] < c["away_ppda"] else away
    press_ppda = min(c["home_ppda"], c["away_ppda"])
    transition_team = home if c["home_transition_shot_rate"] > c["away_transition_shot_rate"] else away
    territory_team = home if c["home_field_tilt"] > c["away_field_tilt"] else away
    territory_value = max(c["home_field_tilt"], c["away_field_tilt"])

    return {
        "Match Story": {
            "subtitle": "Score state, momentum and the moments that changed the tactical problem",
            "performance": [
                ("The first goal shaped the match", early_text),
                ("The game never settled", f"The final {c['score']} scoreline came through repeated swings rather than one sustained period of control."),
                ("Chasing changed risk", f"{loser} had to increase forward numbers and accept more space behind the ball as the match developed."),
            ],
            "data": [
                ("Shot volume was level", f"{home} attempted {int(c['home_shots'])} shots; {away} also attempted {int(c['away_shots'])}."),
                ("Chance quality separated them", f"{home} produced {c['home_xG']:.2f} xG; {away} produced {c['away_xG']:.2f} xG."),
                ("Finishing ran hot", f"The match produced {c['home_goals'] + c['away_goals']} goals from {c['home_xG'] + c['away_xG']:.2f} combined xG, so execution exceeded expectation."),
            ],
            "implication": f"Read every territorial and pressing metric through game state: {winner} protected a lead for long periods, while {loser} accumulated attacking activity under greater urgency.",
        },
        "Chance Creation": {
            "subtitle": "Shot quality, final-third access and the difference between threat and conversion",
            "performance": [
                ("The decisive edge was efficiency", f"{winner} converted fewer attacking situations into a higher-quality scoring return."),
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
                ("Territory and possession told different stories", f"{territory_team} owned {territory_value:.1f}% of field tilt without necessarily owning the same share of total passes."),
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
                ("Game-state context amplified the pattern", f"{winner}'s lead created more opportunities to attack space as {loser} committed additional players."),
            ],
            "data": [
                ("Transition shot rate", f"{home}: {c['home_transition_shot_rate']:.1f}% ({int(c['home_transition_shots'])}/{int(c['home_transitions'])}); {away}: {c['away_transition_shot_rate']:.1f}% ({int(c['away_transition_shots'])}/{int(c['away_transitions'])})."),
                ("Transition xG", f"{home}: {c['home_transition_xG']:.2f}; {away}: {c['away_transition_xG']:.2f}."),
                ("Transition goals", f"{home}: {int(c['home_transition_goals'])}; {away}: {int(c['away_transition_goals'])}."),
                ("Average progress", f"{home}: {c['home_avg_transition_progress']:.1f}; {away}: {c['away_avg_transition_progress']:.1f} pitch units."),
            ],
            "implication": f"{transition_team}'s advantage came from turning fewer seconds of disorder into clearer shots. The defensive response is to secure the ball-side rest defence before attacking numbers advance.",
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


def _visual_team(path: Path, context: dict) -> tuple[str | None, str | None]:
    # Use the visual filename only. The output directory contains both team
    # names (for example, France_vs_England) and must not influence attribution.
    identity = path.name.lower()
    for side in ["home", "away"]:
        team = str(context[side])
        if team.lower() in identity:
            return team, side
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
                f"{player}'s match profile combines {profile['goals']} goals, {profile['shots']} shots ({profile['xG']:.2f} xG), "
                f"{profile['key_passes']} key passes and {profile['pass_xT']:.2f} pass xT. The sequence layer ({profile['sequence_xT']:.2f} xT; "
                f"{profile['xGChain']:.2f} xGChain) shows involvement beyond the final action, but the shape must still be interpreted through role, minutes and score state."
            )
        return tactical_lens(path) + " The page should be used to describe role-specific contribution, not to rank unlike positions."

    if "xg_flow" in stem:
        return (
            f"The cumulative curve shows {away} finishing on {context['away_xG']:.2f} xG against {home}'s {context['home_xG']:.2f}. "
            f"Because the match produced {context['home_goals'] + context['away_goals']} goals from {context['home_xG'] + context['away_xG']:.2f} combined xG, the scoreline contains a large execution component. "
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
        return (
            f"Shot volume was level at {int(context['home_shots'])}-{int(context['away_shots'])}, but average chance quality favoured {away} "
            f"({context['away_xG_per_shot']:.3f} vs {context['home_xG_per_shot']:.3f} xG per shot). The analytical separation is therefore quality and execution, not volume. "
            "Use the location maps to identify which entry routes produced that difference."
        )
    if "match_stats" in stem:
        return (
            f"The overview contains the central contradiction: {home} held {context['home_field_tilt']:.1f}% field tilt and more final-third access, while {away} produced the stronger xG return. "
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
            f"{team} entered the box {int(context[f'{side}_box_entries'])} times but converted {context[f'{side}_box_entry_to_shot_rate']:.1f}% of those entries into shots. "
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
            f"The side completed a {context[f'{side}_pass_share']:.1f}% share of match passes and recorded {int(context[f'{side}_progressive_passes'])} progressive passes. The location of risk is more informative than completion percentage alone."
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
            f"{away} turned {int(context['away_transitions'])} transitions into {int(context['away_transition_shots'])} shots ({context['away_transition_shot_rate']:.1f}%) and {context['away_transition_xG']:.2f} xG. "
            f"{home} produced {int(context['home_transition_shots'])} shots from {int(context['home_transitions'])} transitions ({context['home_transition_shot_rate']:.1f}%). "
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
        return (
            f"{role_read} {player} was involved in attacks worth {p['xGChain']:.2f} xGChain and {p['sequence_xT']:.2f} sequence xT, while the direct output was {p['goals']} goals, "
            f"{p['shots']} shots and {p['key_passes']} key passes. Those figures are evidence for the role, not the story by themselves. "
            f"Read the missing or smaller segments as boundaries of the match role: they may reflect position, minutes, the score state or the team's route of attack. The radar is most useful when traced back to the team pages that show where {p.get('team') or 'the team'} created space for this contribution."
        )

    if "xg_flow" in stem:
        return (
            f"The match developed through separate bursts of danger rather than a smooth exchange of chances. The vertical steps show when an attack reached a genuine finishing situation; the flat stretches show periods in which possession did not materially improve the chance of scoring. "
            f"{away}'s curve finished above {home}'s, but the larger tactical point is the timing of the jumps: once {loser} had to chase, attacks became more direct and the spaces between the pressing line and the defensive cover grew. "
            f"The final {context['score']} score came from {context['home_xG'] + context['away_xG']:.2f} combined xG, so finishing amplified the tactical advantages rather than simply mirroring the volume of chances."
        )
    if "goals_breakdown" in stem:
        first = context["goal_rows"][0] if context["goal_rows"] else None
        first_text = f"The opening goal in minute {first['minute']} gave {first['team']} control over the risk level" if first else "The opening phase did not establish a stable score-state advantage"
        return (
            f"{first_text}. From that point, the trailing side had to push more players beyond the ball, shorten the time spent circulating and accept more direct attacks. That changed both teams at once: the chaser gained territory but weakened the distances protecting turnovers, while the leader could defend central space and wait for open-field moments. "
            "The scorer and assist labels identify the final action, but each goal should be read as the end of a chain involving the regain or progression route, the movement that displaced the last line and the final decision in the box. The order of the goals therefore explains why later full-match averages cannot be treated as neutral."
        )
    if "goalkeeper" in stem:
        return (
            "This page separates goalkeeper influence from the defensive workload in front of the goalkeeper. Save count alone can reward a keeper for facing several routine attempts, whereas post-shot quality asks how difficult the shots became after placement and power were known. "
            f"The two goalkeepers faced a match in which {context['home_goals'] + context['away_goals']} goals greatly exceeded the pre-shot expectation, so the analysis must distinguish defensive access, finishing execution and actual shot-stopping. "
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
        return (
            f"This dashboard joins the two sides of the same tactical plan. {home} controlled more advanced territory and reached the final third more often, but that attacking commitment also left more demanding rest-defence situations. "
            f"{away} created the cleaner average shot and converted transitions more efficiently while allowing fewer dangerous counters. The useful interpretation is therefore not attack versus defence as separate departments: spacing during possession determined both the quality of the next attack and the security of the next defensive action."
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
        return (
            f"The zone map shows where each team established more sustained influence, but territory is a platform rather than an outcome. {home}'s stronger field tilt meant more play was located near the attacking end, yet {away} used its dangerous possessions more efficiently. "
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
                f"{p['shots']} shots, {p['key_passes']} key passes and {p['pass_xT']:.2f} threat added by passing. These are single-match contributions, so role and minutes matter more than the total radar area."
            )
        return "The radar is scaled within this match, so it describes relative involvement on the day rather than long-term player quality."
    if "xg_flow" in stem:
        return (
            f"Both sides attempted {int(context['home_shots'])} and {int(context['away_shots'])} shots, but {away} finished with {context['away_xG']:.2f} xG against {home}'s {context['home_xG']:.2f}. "
            f"The {context['home_goals'] + context['away_goals']} goals exceeded the combined {context['home_xG'] + context['away_xG']:.2f} xG, so finishing increased the score gap beyond the underlying chance gap."
        )
    if "goals_breakdown" in stem:
        first = context["goal_rows"][0] if context["goal_rows"] else None
        first_line = f"{first['team']} scored first in minute {first['minute']}" if first else "The game remained level early"
        return f"{first_line}; the match then produced {context['home_goals'] + context['away_goals']} goals. The sequence and assist fields locate the decisive actions, but the score-state split is required before comparing full-match possession or pressure totals."
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
        return (
            f"{home} led field tilt {context['home_field_tilt']:.1f}%-{context['away_field_tilt']:.1f}% and final-third entries {int(context['home_final_third_entries'])}-{int(context['away_final_third_entries'])}. "
            f"{away} led xG per shot {context['away_xG_per_shot']:.3f}-{context['home_xG_per_shot']:.3f} and transition shot rate {context['away_transition_shot_rate']:.1f}%-{context['home_transition_shot_rate']:.1f}%, while rest-defence vulnerability was {context['away_rest_defence_vulnerability']:.1f}% versus {context['home_rest_defence_vulnerability']:.1f}%."
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
            f"{team} reached the box {int(context[f'{side}_box_entries'])} times, with {context[f'{side}_box_entry_to_shot_rate']:.1f}% becoming shots. "
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
        return (
            f"Field tilt favoured {home} {context['home_field_tilt']:.1f}%-{context['away_field_tilt']:.1f}%, while xG favoured {away} {context['away_xG']:.2f}-{context['home_xG']:.2f}. The opposing signals are evidence that zone ownership did not translate proportionally into chance quality."
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
            f"{home} turned {int(context['home_transitions'])} transitions into {int(context['home_transition_shots'])} shots ({context['home_transition_shot_rate']:.1f}%); {away} turned {int(context['away_transitions'])} into {int(context['away_transition_shots'])} ({context['away_transition_shot_rate']:.1f}%). "
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
        self.body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.2, leading=13.2, textColor=TEXT, alignment=TA_LEFT)
        self.small = ParagraphStyle("small", fontName="Helvetica", fontSize=7.6, leading=10.5, textColor=MUTED, alignment=TA_LEFT)
        self.card = ParagraphStyle("card", fontName="Helvetica", fontSize=8.7, leading=12.3, textColor=TEXT, alignment=TA_LEFT)
        self.analysis = ParagraphStyle("analysis", fontName="Helvetica", fontSize=8.15, leading=11.25, textColor=TEXT, alignment=TA_LEFT)
        self.implication = ParagraphStyle("implication", fontName="Helvetica", fontSize=7.85, leading=10.7, textColor=TEXT, alignment=TA_LEFT)
        self.next_step = ParagraphStyle("next_step", fontName="Helvetica", fontSize=7.55, leading=10.1, textColor=MUTED, alignment=TA_LEFT)
        self.commentary_title = ParagraphStyle("commentary_title", fontName="Times-Bold", fontSize=16.5, leading=18.5, textColor=TEXT, alignment=TA_LEFT)
        self.commentary_body = ParagraphStyle("commentary_body", fontName="Times-Roman", fontSize=9.15, leading=11.35, textColor=TEXT, alignment=TA_LEFT)
        self.commentary_next = ParagraphStyle("commentary_next", fontName="Times-Italic", fontSize=8.4, leading=10.2, textColor=MUTED, alignment=TA_LEFT)

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
        self.canvas.setFont("Helvetica-Bold", 6.8)
        self.canvas.drawRightString(PAGE_W - 24, 14, f"PAGE {self.page:02d}  |  REAL MATCH EVENTS")
        self.canvas.saveState()
        self.canvas.setStrokeColor(GRID)
        self.canvas.setLineWidth(0.5)
        self.canvas.line(24, 25, PAGE_W - 24, 25)
        self.canvas.restoreState()
        self.canvas.showPage()

    def _header(self, title: str, subtitle: str, section: str):
        c = self.canvas
        c.setFillColor(PANEL_2)
        c.roundRect(24, PAGE_H - 98, PAGE_W - 48, 70, 9, fill=1, stroke=0)
        c.setFillColor(HOME); c.rect(28, PAGE_H - 96, (PAGE_W - 56) / 2, 3, fill=1, stroke=0)
        c.setFillColor(AWAY); c.rect(PAGE_W / 2, PAGE_H - 96, (PAGE_W - 56) / 2, 3, fill=1, stroke=0)
        c.setFillColor(FOCUS); c.circle(43, PAGE_H - 49, 3.2, fill=1, stroke=0)
        c.setFillColor(MUTED); c.setFont("Helvetica-Bold", 7); c.drawString(54, PAGE_H - 52, section.upper())
        c.setFillColor(TEXT); c.setFont("Helvetica-Bold", 19); c.drawString(42, PAGE_H - 78, title)
        c.setFillColor(MUTED); c.setFont("Helvetica", 7.5); c.drawString(42, PAGE_H - 91, subtitle[:125])
        c.setFillColor(HOME); c.setFont("Helvetica-Bold", 9); c.drawRightString(PAGE_W - 300, PAGE_H - 50, self.context["home"].upper())
        c.setFillColor(TEXT); c.setFont("Helvetica-Bold", 14); c.drawCentredString(PAGE_W - 245, PAGE_H - 50, self.context["score"])
        c.setFillColor(AWAY); c.setFont("Helvetica-Bold", 9); c.drawString(PAGE_W - 190, PAGE_H - 50, self.context["away"].upper())

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
        c.setFont("Helvetica-Bold", 8.2)
        c.drawString(x + 14, y + h - 22, title.upper())

    def cover(self):
        self._start("cover", "Cover")
        c = self.canvas
        c.setFillColor(PANEL_2)
        c.roundRect(42, 70, PAGE_W - 84, PAGE_H - 140, 18, fill=1, stroke=0)
        c.setFillColor(HOME); c.rect(42, 70, (PAGE_W - 84) / 2, 5, fill=1, stroke=0)
        c.setFillColor(AWAY); c.rect(PAGE_W / 2, 70, (PAGE_W - 84) / 2, 5, fill=1, stroke=0)
        c.setFillColor(FOCUS); c.circle(73, PAGE_H - 108, 5, fill=1, stroke=0)
        c.setFillColor(MUTED); c.setFont("Helvetica-Bold", 9); c.drawString(91, PAGE_H - 112, "MATCH INTELLIGENCE REPORT")
        c.setFillColor(TEXT); c.setFont("Helvetica-Bold", 30); c.drawString(72, PAGE_H - 185, "DETAILED TACTICAL")
        c.drawString(72, PAGE_H - 222, "AND DATA ANALYSIS")
        c.setFillColor(MUTED); c.setFont("Helvetica", 12)
        c.drawString(74, PAGE_H - 250, "A connected performance-analysis narrative supported by real match events")
        c.setFillColor(HOME); c.setFont("Helvetica-Bold", 22); c.drawRightString(PAGE_W / 2 - 72, 287, self.context["home"].upper())
        c.setFillColor(TEXT); c.setFont("Helvetica-Bold", 42); c.drawCentredString(PAGE_W / 2, 277, self.context["score"])
        c.setFillColor(AWAY); c.setFont("Helvetica-Bold", 22); c.drawString(PAGE_W / 2 + 72, 287, self.context["away"].upper())
        c.setFillColor(FOCUS); c.setFont("Helvetica-Bold", 9); c.drawString(74, 174, "REPORT SPINE")
        c.setFillColor(MUTED); c.setFont("Helvetica", 9)
        c.drawString(74, 154, "MATCH STORY  |  CHANCE CREATION  |  POSSESSION  |  PRESSING  |  TRANSITIONS  |  PLAYER IMPACT")
        c.setFillColor(NEUTRAL); c.setFont("Helvetica-Bold", 7); c.drawRightString(PAGE_W - 72, 130, "CREATED BY MOSTAFA SAAD")
        self._finish()

    def executive_summary(self, sections: dict[str, dict]):
        self._start("executive_summary", "Executive Summary")
        self._header("Executive Summary", "The result, the mechanism and the main coaching implications", "REPORT OPEN")
        c = self.canvas
        c.setFillColor(TEXT); c.setFont("Helvetica-Bold", 15)
        headline = f"{self.context['winner']} won the execution battle; the losing side's activity did not translate into equal control of shot quality."
        c.drawString(42, PAGE_H - 132, headline[:118])
        bullets = [
            sections["Match Story"]["data"][1],
            sections["Chance Creation"]["data"][0],
            sections["Possession and Progression"]["data"][1],
            sections["Pressing and Rest Defence"]["data"][3],
            sections["Transitions and Efficiency"]["data"][0],
        ]
        positions = [(42, 385, 444, 135), (522, 385, 444, 135), (42, 227, 444, 135), (522, 227, 444, 135)]
        for idx, ((title, body), (x, y, w, h)) in enumerate(zip(bullets[:4], positions), start=1):
            self._card_box(x, y, w, h, f"0{idx} - {title}", HOME if idx % 2 else AWAY)
            self._paragraph(escape(body), x + 14, y + h - 38, w - 28, h - 50, self.card)
        title, body = bullets[4]
        self._card_box(42, 82, PAGE_W - 84, 112, f"05 - {title}", FOCUS)
        self._paragraph(escape(body), 56, 161, PAGE_W - 112, 70, self.card)
        self._finish()

    def toc(self, entries: list[tuple[str, int, str]]):
        self._start("contents", "Contents")
        self._header("Report Contents", "A performance-analysis reading path followed by the complete player appendix", "NAVIGATION")
        y = PAGE_H - 140
        for idx, (title, page, subtitle) in enumerate(entries, start=1):
            accent = HOME if idx % 2 else AWAY
            self.canvas.setFillColor(accent)
            self.canvas.circle(56, y + 5, 4, fill=1, stroke=0)
            self.canvas.setFillColor(TEXT); self.canvas.setFont("Helvetica-Bold", 11)
            self.canvas.drawString(74, y, f"{idx:02d}  {title}")
            self.canvas.setFillColor(MUTED); self.canvas.setFont("Helvetica", 7.5)
            self.canvas.drawString(74, y - 15, subtitle[:106])
            self.canvas.setStrokeColor(GRID); self.canvas.line(510, y + 2, PAGE_W - 86, y + 2)
            self.canvas.setFillColor(FOCUS); self.canvas.setFont("Helvetica-Bold", 10)
            self.canvas.drawRightString(PAGE_W - 55, y, f"PAGE {page:02d}")
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
                self.canvas.setFillColor(FOCUS); self.canvas.setFont("Helvetica-Bold", 7.5)
                self.canvas.drawString(x + 14, top, f"{idx:02d}  {label.upper()}")
                height = self._paragraph(escape(body), x + 14, top - 10, 416, 55, self.small)
                top -= max(62, height + 29)
        self._card_box(42, 97, PAGE_W - 84, 140, "Tactical Implication", FOCUS)
        self._paragraph(escape(copy["implication"]), 60, 202, PAGE_W - 120, 75, self.body)
        self.canvas.setFillColor(NEUTRAL); self.canvas.setFont("Helvetica", 7)
        self.canvas.drawString(60, 118, "Use the following visuals as evidence for this section. Read the explanation and next analytical step below every chart.")
        self._finish()

    def verdict(self):
        self._start("final_verdict", "Final Tactical Verdict")
        self._header("Final Tactical Verdict", "A joined performance and data conclusion", "SYNTHESIS")
        c = self.canvas
        home, away = self.context["home"], self.context["away"]
        c.setFillColor(TEXT); c.setFont("Helvetica-Bold", 18)
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
                self.canvas.setFillColor(FOCUS); self.canvas.setFont("Helvetica-Bold", 8)
                self.canvas.drawString(x + 16, top, label.upper())
                self._paragraph(escape(body), x + 16, top - 12, 410, 55, self.card)
                top -= 83
        self._finish()

    def visual(self, path: Path, section: str, next_path: Path | None = None):
        self._start()
        c = self.canvas
        with Image.open(path) as image:
            iw, ih = image.size
        margin_x = 4
        image_region_h = PAGE_H - VISUAL_NOTE_H
        scale = min((PAGE_W - 2 * margin_x) / iw, image_region_h / ih)
        width, height = iw * scale, ih * scale
        x = (PAGE_W - width) / 2
        y = VISUAL_NOTE_H + (image_region_h - height) / 2
        c.drawImage(ImageReader(str(path)), x, y, width=width, height=height, preserveAspectRatio=True, mask="auto")
        c.setFillColor(PANEL_2)
        c.rect(0, 0, PAGE_W, VISUAL_NOTE_H, fill=1, stroke=0)
        c.setStrokeColor(GRID); c.setLineWidth(0.7); c.line(0, VISUAL_NOTE_H, PAGE_W, VISUAL_NOTE_H)
        c.setFillColor(FOCUS); c.setFont("Helvetica-Bold", 6.8); c.drawString(42, VISUAL_NOTE_H - 18, "TACTICAL COMMENTARY")
        title = visual_commentary_title(path, self.context)
        self._paragraph(escape(title), 42, VISUAL_NOTE_H - 28, PAGE_W - 84, 24, self.commentary_title)
        c.setStrokeColor(FOCUS); c.setLineWidth(1.2); c.line(42, VISUAL_NOTE_H - 51, PAGE_W - 42, VISUAL_NOTE_H - 51)

        performance = escape(visual_explanation(path, self.context))
        evidence = escape(visual_data_read(path, self.context))
        integrated = escape(visual_implication(path, self.context))
        x, width = 42, PAGE_W - 84
        top = VISUAL_NOTE_H - 62
        h1 = self._paragraph(f'<b><font color="{self.context["home_color"]}">PERFORMANCE ANALYST.</font></b> {performance}', x, top, width, 72, self.commentary_body)
        top -= h1 + 6
        h2 = self._paragraph(f'<b><font color="{self.context["away_color"]}">DATA ANALYST.</font></b> {evidence}', x, top, width, 53, self.commentary_body)
        top -= h2 + 6
        self._paragraph(f'<b><font color="#FFD43B">INTEGRATED READ.</font></b> {integrated}', x, top, width, 42, self.commentary_body)

        c.setStrokeColor(GRID); c.setLineWidth(0.5); c.line(42, 31, PAGE_W - 42, 31)
        self._paragraph(escape(next_visual_step(next_path)), 42, 25, PAGE_W - 250, 18, self.commentary_next)
        c.setFillColor(NEUTRAL); c.setFont("Helvetica-Bold", 6.0); c.drawRightString(PAGE_W - 24, 10, f"{section.upper()}  |  PAGE {self.page:02d}")
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
    home_hex = "#9A99B4"
    away_hex = "#A83246"
    HOME = colors.HexColor(home_hex)
    AWAY = colors.HexColor(away_hex)
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
