"""The written report, one chart at a time, argued from :mod:`match_facts`.

Every paragraph here is a function of the numbers rather than of the fixture it
was first written for. A writer receives the facts and the side the chart
belongs to, and returns the paragraphs that go under it; the chapter machinery
appends a hand-off sentence naming the figure that follows, so a reader is
carried from one exhibit to the next instead of being handed a gallery.

Charts the pipeline may or may not export are handled by name, and anything
unrecognised falls back to the reading stored in the package manifest. A new
chart in a later release therefore appears in the document with its own words
rather than being silently dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from match_facts import MatchFacts, Side, next_best, top_players


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------
def n0(value: float) -> str:
    return f"{value:,.0f}"


def n1(value: float) -> str:
    return f"{value:,.1f}"


def n2(value: float) -> str:
    return f"{value:,.2f}"


def pct(value: float, places: int = 1) -> str:
    return f"{value:.{places}f}%"


def ratio(part: float, whole: float, places: int = 1) -> str:
    return f"{(100.0 * part / whole):.{places}f}%" if whole else "n/a"


def times(bigger: float, smaller: float) -> str:
    """"three times" reads better than "3.0x" in a sentence."""
    if not smaller:
        return "with no comparison possible"
    factor = bigger / smaller
    words = {2: "twice", 3: "three times", 4: "four times", 5: "five times",
             6: "six times", 7: "seven times", 8: "eight times"}
    nearest = round(factor)
    if nearest in words and abs(factor - nearest) < 0.35:
        return words[nearest]
    return f"{factor:.1f} times"


def _spelled(count: int) -> str:
    """Small numbers read better as words, and "the three finishes" was written
    into the sentence whatever the score."""
    words = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    return words.get(count, str(count))


def possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def listed(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def minute(value: int) -> str:
    return f"{value}'"


# ----------------------------------------------------------------------
# Chart identity
# ----------------------------------------------------------------------
@dataclass
class Visual:
    """One exported PNG, resolved to what it is a picture of."""
    path: Path
    number: str
    slug: str            # "shot_map", "pass_network", ...
    side: Side | None
    half: str | None     # "first" / "second"
    title: str

    @property
    def is_team_specific(self) -> bool:
        return self.side is not None


_HALVES = {"1h": "first", "2h": "second"}

TITLES = {
    "xg_flow": "Expected-goals flow",
    "shot_map": "Shot map",
    "pass_network": "Passing network",
    "xt_map": "Expected-threat map",
    "pass_map": "Pass map",
    "goalkeeper_saves": "Goalkeeper saves",
    "zone14": "Zone 14",
    "xt_per_minute": "Expected threat per minute",
    "progressive": "Progressive passes",
    "defensive_activity": "Defensive activity",
    "dominating_zones": "Zones of control",
    "box_entries": "Box entries",
    "high_regains": "High regains",
    "pass_targets": "Pass targets",
    "ppda_pressing": "Pressing intensity (PPDA)",
    "transition_outcomes": "Transition outcomes",
    "game_state_splits": "Game-state splits",
    "player_sequence_leaders": "Sequence leaders",
    "match_momentum": "Match momentum",
    "set_pieces": "Set pieces",
    "defensive_shape": "Defensive shape",
    "playing_through": "Playing through the lines",
    "action_value": "Action value",
    "pass_sonar": "Pass sonar",
    "finishing_quality": "Finishing quality",
    "pitch_control": "Pitch control",
    "sequence_types": "Sequence types",
    "goal_origins": "Goal origins",
    "press_triggers": "Press triggers",
    "player_progression_creation": "Progression and chance creation",
    "player_involvement_value": "Involvement and added threat",
    "player_shot_quality": "Shot volume and quality",
    "possession_speed_value": "Possession value by duration",
    "regain_speed": "Time to progress after a regain",
    "possession_funnel": "Possession funnel",
    "entry_routes": "Entry routes",
    "loss_consequences": "Loss locations and consequences",
    "substitution_windows": "Substitution windows",
    "history_territory_chances": "Territory and chances in context",
    "history_pressing_risk": "Pressing and risk in context",
}


def resolve(path: Path, facts: MatchFacts) -> Visual:
    """Read a chart's filename into the thing it is a picture of."""
    stem = path.stem
    number, _, remainder = stem.partition("_")
    if not re.fullmatch(r"\d+[a-z]?", number, flags=re.IGNORECASE):
        number, remainder = "", stem

    half = None
    for code, label in _HALVES.items():
        if remainder.endswith("_" + code):
            half = label
            remainder = remainder[: -(len(code) + 1)]

    side = None
    for candidate in facts.sides:
        if candidate.slug and remainder.endswith("_" + candidate.slug):
            side = candidate
            remainder = remainder[: -(len(candidate.slug) + 1)]
            break

    slug = remainder
    title = TITLES.get(slug, slug.replace("_", " ").capitalize())
    if side is not None:
        title = f"{title} — {side.name}"
    if half:
        title = f"{title}, {half} half"
    return Visual(path=path, number=number, slug=slug, side=side, half=half, title=title)


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------
WRITERS: dict[str, callable] = {}
HANDOFFS: dict[str, str] = {}


def writes(slug: str, handoff: str = ""):
    def register(function):
        WRITERS[slug] = function
        if handoff:
            HANDOFFS[slug] = handoff
        return function
    return register


def _sides_by(facts: MatchFacts, attribute: str) -> tuple[Side, Side]:
    """The two sides ordered by one attribute, larger first."""
    first, second = facts.sides
    return (first, second) if getattr(first, attribute) >= getattr(second, attribute) else (second, first)


@writes("xg_flow", "Where those chances were taken from is the next question, and {next} answers it.")
def _xg_flow(v: Visual, f: MatchFacts) -> list[str]:
    lead, trail = _sides_by(f, "xg")
    opener = f.goals[0] if f.goals else None
    out = [
        f"Start here and the rest of the report follows. {lead.name} finished on "
        f"{n2(lead.xg)} expected goals from {lead.shots} shots, {trail.name} on "
        f"{n2(trail.xg)} from {trail.shots}. That gap is not a gap in volume: it is "
        f"{n2(lead.xg_per_shot)} expected goals per shot against {n2(trail.xg_per_shot)}, "
        f"which is a difference in the quality of the look rather than the number of them."
    ]
    if opener:
        scorer_side = f.side(opener["side"])
        out.append(
            f"The curve also shows how the match was framed. {opener['player']} scored for "
            f"{scorer_side.name} on {minute(opener['minute'])} from a chance worth "
            f"{n2(opener['xG'])} expected goals"
            + (" — a set piece, so from a position the run of play was not producing"
               if opener["set_piece"] else " — a low-value chance taken early")
            + f". Everything after that was played at a scoreline neither side had earned yet."
        )
    if f.goals:
        line = listed([f"{g['player']} {minute(g['minute'])} ({g['team']})" for g in f.goals])
        # "Final score Nottingham Forest 0:1 Leeds" puts a digit against a club
        # name that ends in s, which reads as a count of one and its plural.
        # Naming each side beside its own number avoids the collision and reads
        # as a sentence rather than as a fixture listing.
        home_goals, _, away_goals = str(f.score).partition(":")
        out.append(
            f"Goals in order: {line}. It finished {f.home.name} "
            f"{home_goals.strip()}, {f.away.name} {away_goals.strip()}.")
    return out


@writes("shot_map", "The other side's attempts are the necessary comparison, and {next} carries them.")
def _shot_map(v: Visual, f: MatchFacts) -> list[str]:
    s = v.side
    other = f.other(s)
    shots = [x for x in f.shots if x["side"] == s.key]
    big = [x for x in shots if x["big_chance"]]
    scored_big = [x for x in big if x in f.goals]
    out = [
        f"{s.name} took {s.shots} shots: {s.on_target} on target, {s.blocked} blocked and "
        f"{s.off_target} off. {'Only ' if s.off_target <= 2 else ''}{s.off_target} wild attempt"
        f"{'' if s.off_target == 1 else 's'} says the side was shooting from positions it had "
        f"worked into rather than from hope, and the map bears that out."
    ]
    if big:
        detail = listed([f"{x['player']} {minute(x['minute'])} ({n2(x['xG'])})" for x in big])
        out.append(
            f"{len(big)} clear chance{'' if len(big) == 1 else 's'}: {detail}. "
            f"{len(scored_big)} of them {'was' if len(scored_big) == 1 else 'were'} taken. "
            f"A side that scores {s.goals} while converting {len(scored_big)} of {len(big)} "
            f"clear openings won in spite of its best moments, not because of them."
        )
    else:
        out.append(
            f"No chance on the map was flagged as clear-cut. {n2(s.xg_per_shot)} expected goals "
            f"per attempt is the average of a side shooting from the edge of the box or from "
            f"angles, which is what happens when the last pass never arrives."
        )
    out.append(
        f"For scale, {other.name} managed {other.shots} shots worth {n2(other.xg)} across the "
        f"same ninety minutes."
    )
    return out


@writes("pass_network", "Whether the shape held after the interval is the next thing to check, in {next}.")
def _pass_network(v: Visual, f: MatchFacts) -> list[str]:
    s, half = v.side, v.half or "first"
    data = s.half.get(half, {})
    other_half = "second" if half == "first" else "first"
    previous = s.half.get(other_half, {})
    out = [
        f"{s.name} played {n0(data.get('passes', 0))} passes in the {half} half, "
        f"{n0(data.get('completed', 0))} of them completed, and the network is drawn from where "
        f"those passes started and ended. Read the centre of gravity first: how high it sits and "
        f"which flank it leans towards is the shape {s.manager or 'the side'} actually used, as "
        f"opposed to the one on the team sheet ({s.formation})."
    ]
    produced = (f"{n0(data.get('shots', 0))} shots worth {n2(data.get('xG', 0.0))} expected goals "
                f"and {n0(data.get('box_entries', 0))} entries into the box")
    if half == "first":
        out.append(f"That structure produced {produced}.")
    else:
        change = data.get("xG", 0.0) - previous.get("xG", 0.0)
        direction = "more" if change > 0.05 else ("less" if change < -0.05 else "much the same")
        out.append(
            f"That structure produced {produced} — {direction} than the "
            f"{n2(previous.get('xG', 0.0))} the same side generated before the interval. "
            f"Volume of passing is not the explanation: it went from "
            f"{n0(previous.get('passes', 0))} to {n0(data.get('passes', 0))}."
        )
    return out


@writes("xt_map", "The passing itself is the mechanism behind that map, and {next} shows it.")
def _xt_map(v: Visual, f: MatchFacts) -> list[str]:
    s, other = v.side, f.other(v.side)
    lanes = s.lane_left + s.lane_centre + s.lane_right
    dominant = max((("left", s.lane_left), ("centre", s.lane_centre), ("right", s.lane_right)),
                   key=lambda pair: pair[1])
    out = [
        f"{possessive(s.name)} sequence threat totalled {n2(s.sequence_xt)} against "
        f"{n2(other.sequence_xt)}, or {n2(s.xt_per_possession)} per possession against "
        f"{n2(other.xt_per_possession)}. The total matters less than where it was generated."
    ]
    if lanes:
        out.append(
            f"Of the {lanes} passes that carried {s.name} into the final third, "
            f"{dominant[1]} went down the {dominant[0]} — {ratio(dominant[1], lanes)} of the "
            f"side's entries. Left {s.lane_left}, centre {s.lane_centre}, right {s.lane_right}. "
            f"A concentration like that is a decision about where to attack, not an accident of "
            f"where the ball happened to go."
        )
    out.append(
        f"{s.name} reached the final third in {n1(s.avg_seconds_to_third)} seconds on average "
        f"and touched the ball {s.final_third_touches} times there."
    )
    return out


@writes("pass_map", "How the opposition moved the ball is the necessary contrast, in {next}.")
def _pass_map(v: Visual, f: MatchFacts) -> list[str]:
    s, other = v.side, f.other(v.side)
    forward_share = ratio(s.forward_passes, s.passes)
    out = [
        f"{n0(s.passes)} passes at an average of {n1(s.avg_pass_length)} metres, "
        f"{s.forward_passes} of them forward ({forward_share}) and {s.long_passes} of "
        f"30 metres or more. Short and forward is a different proposition from long and forward: "
        f"the first keeps a supporting player inside the next pass, the second does not."
    ]
    out.append(
        f"The consequence shows in the clock. {s.name} needed "
        f"{n1(s.avg_seconds_to_third)} seconds to reach the final third; {other.name} needed "
        f"{n1(other.avg_seconds_to_third)}. Arriving before an opponent has reset is a different "
        f"attack from arriving after."
    )
    return out


@writes("goalkeeper_saves", "The area where those saves were forced is next, in {next}.")
def _goalkeeper_saves(v: Visual, f: MatchFacts) -> list[str]:
    out = []
    for keeper_side in f.sides:
        attacker = f.other(keeper_side)
        prevented = attacker.xgot - attacker.goals
        verdict = ("above what the shots were worth" if prevented > 0.35 else
                   "below it" if prevented < -0.35 else "at roughly what the shots were worth")
        out.append(
            f"{possessive(keeper_side.name)} goalkeeper made {keeper_side.gk_saves} saves. "
            f"{attacker.name} put {n2(attacker.xgot)} of on-target value behind them and scored "
            f"{attacker.goals}, which puts the goalkeeping {verdict} "
            f"({prevented:+.2f} goals against the post-shot model)."
        )
    blocker = max(f.sides, key=lambda s: s.outfield_blocks)
    if blocker.outfield_blocks:
        out.append(
            f"Saves are a volume statistic before they are a quality one. The other half of the "
            f"resistance was in front of the goalkeeper: {blocker.name} outfield players blocked "
            f"{blocker.outfield_blocks} attempts, and {f.other(blocker).blocked} "
            f"{possessive(f.other(blocker).name)} shots were blocked in total."
        )
    return out


@writes("zone14", "The same box for the other side tells the more revealing half of the story: {next}.")
def _zone14(v: Visual, f: MatchFacts) -> list[str]:
    s, other = v.side, f.other(v.side)
    out = [
        f"{s.name} had {s.zone14_touches} touches and {s.zone14_passes} passes in the strip "
        f"directly in front of the box. On its own that is a measure of arrival, not of damage."
    ]
    out.append(
        f"What it turned into: {s.box_entries} box entries, {pct(s.box_entry_to_shot_rate)} of "
        f"which ended in a shot. {other.name} had {other.zone14_touches} touches in the same "
        f"strip and converted them into {other.box_entries} entries at "
        f"{pct(other.box_entry_to_shot_rate)}. Occupying the zone and breaking out of it are "
        f"different skills, and the pair of numbers separates them."
    )
    return out


@writes("xt_per_minute", "The individual carriers of that threat come next, beginning with {next}.")
def _xt_per_minute(v: Visual, f: MatchFacts) -> list[str]:
    out = []
    for s in f.sides:
        first, second = s.half.get("first", {}), s.half.get("second", {})
        out.append(
            f"{s.name}: {n2(first.get('xT', 0.0))} of threat before the interval and "
            f"{n2(second.get('xT', 0.0))} after, from {n0(first.get('shots', 0))} and "
            f"{n0(second.get('shots', 0))} shots worth {n2(first.get('xG', 0.0))} and "
            f"{n2(second.get('xG', 0.0))} expected goals."
        )
    states = []
    for s in f.sides:
        for label in ("leading", "drawing", "trailing"):
            bucket = s.states.get(label) or {}
            if bucket.get("minutes"):
                states.append(
                    f"{s.name} {label} for {n0(bucket['minutes'])} minutes produced "
                    f"{n2(bucket.get('spell_xG', 0.0))} expected goals")
    if states:
        out.append("Read against the scoreline: " + "; ".join(states) + ".")
    return out


@writes("progressive", "Whoever had to stop that progression is the other half of the exchange, in {next}.")
def _progressive(v: Visual, f: MatchFacts) -> list[str]:
    s, other = v.side, f.other(v.side)
    ranked = sorted(s.players, key=lambda p: p["progressive_passes"], reverse=True)[:3]
    lead = ranked[0] if ranked else None
    out = [
        f"{s.name} played {s.progressive_passes} progressive passes against "
        f"{other.progressive_passes}, and completed {s.deep_completions} of them deep enough to "
        f"count as a deep completion — {other.name} managed {other.deep_completions}."
    ]
    if lead:
        rest = listed([f"{p['name']} {p['progressive_passes']}" for p in ranked[1:]])
        gap = lead["progressive_passes"] - (ranked[1]["progressive_passes"] if len(ranked) > 1 else 0)
        out.append(
            f"{lead['name']} played {lead['progressive_passes']} of them"
            + (f", {gap} more than anyone else in the side ({rest})." if gap > 0 else f" ({rest}).")
        )
    carriers = [p for p in ranked if p["role"] in ("DC", "DR", "DL", "GK")]
    if len(carriers) >= 2:
        out.append(
            f"Note where it came from: {listed([p['name'] for p in carriers])} are defenders. "
            f"Progression that starts and finishes behind the midfield tends to arrive as a "
            f"cross rather than as a break, because nobody ahead of the ball has moved."
        )
    return out


@writes("defensive_activity", "What that defending cost, and where, is the point of {next}.")
def _defensive_activity(v: Visual, f: MatchFacts) -> list[str]:
    s, other = v.side, f.other(v.side)
    stoppers = sorted(s.players,
                      key=lambda p: p["clearances"] + p["tackles_won"] + p["interceptions"],
                      reverse=True)[:2]
    out = [
        f"{s.name} made {s.def_actions} defensive actions at an average height of "
        f"{n1(s.def_action_avg_x)} up the pitch: {s.def_actions_high} in the attacking third, "
        f"{s.def_actions_mid} in the middle and {s.def_actions_low} in its own. That distribution "
        f"describes a "
        + ("high press" if s.def_action_avg_x >= 45 else
           "mid block" if s.def_action_avg_x >= 35 else "low-to-mid block")
        + f", not a side chasing the ball everywhere."
    ]
    if stoppers:
        detail = listed([
            f"{p['name']} ({p['clearances']} clearances, {p['tackles_won']} tackles won, "
            f"{p['interceptions']} interceptions)" for p in stoppers])
        out.append(f"The load was carried by {detail}.")
    out.append(
        f"Defending well is only half of it. {s.name} lost the ball {s.losses} times in open "
        f"play; {s.losses_to_third} of those losses had the opponent in the final third within "
        f"twelve seconds, {s.losses_to_box} in the box, and {s.losses_to_shot} produced a shot "
        f"worth {n2(s.loss_xg_conceded)} in total. {other.name} lost it {other.losses} times for "
        f"{n2(other.loss_xg_conceded)}."
    )
    return out


@writes("dominating_zones", "The next chapter takes that territory apart entry by entry, starting with {next}.")
def _dominating_zones(v: Visual, f: MatchFacts) -> list[str]:
    tilt_lead, tilt_trail = _sides_by(f, "field_tilt")
    poss_lead, _ = _sides_by(f, "possession_share")
    out = [
        f"Field tilt — the share of final-third touches — was {pct(tilt_lead.field_tilt, 0)} to "
        f"{tilt_lead.name} against {pct(tilt_trail.field_tilt, 0)}. Possession was "
        f"{pct(f.home.possession_share)} to {f.home.name} and "
        f"{pct(f.away.possession_share)} to {f.away.name}."
    ]
    if abs(f.home.possession_share - f.away.possession_share) < 8:
        out.append(
            f"Those two lines disagree, and the disagreement is the match. The ball was shared "
            f"almost evenly; the ground was not. {poss_lead.name} did not win by keeping the ball "
            f"more, and {tilt_trail.name} did not lose by having less of it."
        )
    out.append(
        f"Touch distribution: {f.home.name} {pct(f.home.touch_def_pct, 0)} defensive third / "
        f"{pct(f.home.touch_mid_pct, 0)} middle / {pct(f.home.touch_att_pct, 0)} attacking; "
        f"{f.away.name} {pct(f.away.touch_def_pct, 0)} / {pct(f.away.touch_mid_pct, 0)} / "
        f"{pct(f.away.touch_att_pct, 0)}. Deep completions ran "
        f"{f.home.deep_completions} to {f.away.deep_completions}."
    )
    return out


@writes("box_entries", "Entries have to start from a regain, and {next} shows where those regains happened.")
def _box_entries(v: Visual, f: MatchFacts) -> list[str]:
    s, other = v.side, f.other(v.side)
    first = s.half.get("first", {}).get("box_entries", 0)
    second = s.half.get("second", {}).get("box_entries", 0)
    creators = sorted(s.players, key=lambda p: p["box_entries"], reverse=True)[:3]
    out = [
        f"{s.name} entered the box {s.box_entries} times and {pct(s.box_entry_to_shot_rate)} of "
        f"those entries produced a shot. Split by half: {n0(first)} before the interval and "
        f"{n0(second)} after."
    ]
    if creators and creators[0]["box_entries"]:
        out.append("Carried by " + listed([f"{p['name']} {p['box_entries']}" for p in creators
                                           if p["box_entries"]]) + ".")
    out.append(
        f"The comparison is stark: {other.name} entered {other.box_entries} times at "
        f"{pct(other.box_entry_to_shot_rate)}. That is "
        f"{times(max(s.box_entries, other.box_entries), max(min(s.box_entries, other.box_entries), 1))} "
        f"the access, before the conversion rate is applied on top of it."
    )
    return out


@writes("high_regains", "What the side then did with the ball is the question {next} takes up.")
def _high_regains(v: Visual, f: MatchFacts) -> list[str]:
    s, other = v.side, f.other(v.side)
    out = [
        f"{s.name} won the ball back {s.regains} times in all, {s.high_regains} of them high up "
        f"the pitch, and {s.counterpress_regains} of {s.counterpress_attempts} counterpressing "
        f"attempts came off ({pct(s.counterpress_success_rate)}). Those regains were worth "
        f"{n2(s.regain_xg)} expected goals and {n2(s.regain_xt)} of threat."
    ]
    same = abs(s.counterpress_success_rate - other.counterpress_success_rate) < 1.5
    out.append(
        f"{other.name} pressed at {pct(other.counterpress_success_rate)} for {n2(other.regain_xg)}. "
        + ("Identical success rates with different returns means the difference is not how well "
           "either side pressed but where: a regain with players ahead of the ball is a chance, "
           "and the same regain against a set block is a restart."
           if same else
           "The gap in return is larger than the gap in success rate, which points at the "
           "position of the regain rather than the act of winning it.")
    )
    return out


@writes("pass_targets", "Where the ball actually went from there is the subject of {next}.")
def _pass_targets(v: Visual, f: MatchFacts) -> list[str]:
    s = v.side
    receivers = sorted(s.players, key=lambda p: p["touches"], reverse=True)[:3]
    out = [
        "Most-used targets: " + listed([f"{p['name']} ({p['touches']} touches)" for p in receivers])
        + f". {s.name} attempted {s.crosses} crosses and completed {s.completed_crosses} "
          f"({ratio(s.completed_crosses, s.crosses)})."
    ]
    if receivers and receivers[0]["role"] in ("DC", "DR", "DL", "GK", "DMC", "MC"):
        out.append(
            f"The most-found player being {receivers[0]['name']} — a "
            f"{receivers[0]['role']} — puts the centre of the passing map behind the line where "
            f"chances are made. A cross from that distance is a hopeful ball, and the completion "
            f"rate reflects it."
        )
    else:
        out.append(
            f"The reception map sits high and wide, which is what allows a cross to be a delivery "
            f"rather than a clearance in the other direction."
        )
    return out


@writes("ppda_pressing", "What happened in the seconds after a turnover is the point of {next}.")
def _ppda(v: Visual, f: MatchFacts) -> list[str]:
    presser = min(f.sides, key=lambda s: s.ppda if s.ppda else 999)
    other = f.other(presser)
    return [
        f"PPDA — opposition passes allowed per defensive action — ran {n2(f.home.ppda)} for "
        f"{f.home.name} and {n2(f.away.ppda)} for {f.away.name}. Lower is more aggressive, so "
        f"{presser.name} was the side squeezing.",
        f"The consequence is in the clock rather than in the tackle count. {presser.name} needed "
        f"{n1(presser.avg_seconds_to_third)} seconds to reach the final third; {other.name} "
        f"needed {n1(other.avg_seconds_to_third)}. Forcing an extra touch out of a side building "
        f"from the back buys the time a defence needs to be back in position.",
        f"Build-up success under that pressure: {presser.name} {presser.build_up_successes} of "
        f"{presser.build_up_attempts} ({pct(presser.build_up_success_rate)}), {other.name} "
        f"{other.build_up_successes} of {other.build_up_attempts} "
        f"({pct(other.build_up_success_rate)}).",
    ]


@writes("transition_outcomes", "The scoreline changed what both sides were trying to do, which {next} separates.")
def _transitions(v: Visual, f: MatchFacts) -> list[str]:
    out = []
    for s in f.sides:
        out.append(
            f"{s.name}: {s.transitions} transitions producing {s.transition_shots} shots, "
            f"{s.transition_box_entries} box entries and {n2(s.transition_xg)} expected goals, at "
            f"an average of {n1(s.avg_transition_duration)} seconds."
        )
    exposed = max(f.sides, key=lambda s: s.rest_defence_vulnerability)
    safer = f.other(exposed)
    out.append(
        f"The other side of a transition is the one you concede. {safer.name} was exposed "
        f"{safer.rest_defence_exposures} times and only {safer.rest_defence_dangerous} of those "
        f"turned into a dangerous counter ({pct(safer.rest_defence_vulnerability)}); "
        f"{exposed.name} was exposed {exposed.rest_defence_exposures} times for "
        f"{exposed.rest_defence_dangerous} ({pct(exposed.rest_defence_vulnerability)}). Being "
        f"exposed more often and hurt less is what an organised rest defence looks like."
    )
    return out


@writes("game_state_splits", "The individuals behind those numbers are ranked in {next}.")
def _game_state(v: Visual, f: MatchFacts) -> list[str]:
    lines = []
    for s in f.sides:
        for label in ("trailing", "drawing", "leading"):
            bucket = s.states.get(label) or {}
            if not bucket.get("minutes"):
                continue
            lines.append(
                f"{s.name} {label} ({n0(bucket['minutes'])} min): "
                f"{n0(bucket.get('spell_shots', 0))} shots, "
                f"{n2(bucket.get('spell_xG', 0.0))} xG, "
                f"{n0(bucket.get('spell_box_entries', 0))} box entries")
    out = ["Output held against the scoreline at the time:"] + [f"• {line}" for line in lines]
    best = None
    for s in f.sides:
        bucket = s.states.get("trailing") or {}
        if bucket.get("minutes"):
            rate = bucket.get("spell_xG", 0.0) / max(bucket["minutes"], 1) * 90
            if best is None or rate > best[1]:
                best = (s, rate, bucket)
    if best:
        s, rate, bucket = best
        out.append(
            f"The sharpest reading in the report: {s.name} behind produced "
            f"{n2(bucket.get('spell_xG', 0.0))} expected goals in {n0(bucket['minutes'])} minutes, "
            f"a rate of {n2(rate)} per ninety. Having a plan for the moment you have to score is "
            f"not the same as reacting to going behind."
        )
    return out


@writes("player_sequence_leaders", "When each of them was doing it is what {next} plots.")
def _sequence_leaders(v: Visual, f: MatchFacts) -> list[str]:
    out = []
    for s in f.sides:
        ranked = sorted(s.players, key=lambda p: p["xGChain"], reverse=True)[:5]
        out.append(f"{s.name}: " + listed([f"{p['name']} {n2(p['xGChain'])}" for p in ranked]) + ".")
    high, low = _sides_by(f, "xg")
    best_low = max(low.players, key=lambda p: p["xGChain"], default=None)
    ranked_high = sorted(high.players, key=lambda p: p["xGChain"], reverse=True)
    if best_low and len(ranked_high) >= 5:
        beaten = [p for p in ranked_high if p["xGChain"] > best_low["xGChain"]]
        if len(beaten) >= 3:
            out.append(
                f"{possessive(low.name)} leading figure ({best_low['name']}, "
                f"{n2(best_low['xGChain'])}) sits below {len(beaten)} "
                f"{high.name} players. That is not a comment on him; it is a description of how "
                f"far the two attacking units were apart on the day."
            )
    return out


@writes("match_momentum", "The one route that was open to the trailing side is examined in {next}.")
def _momentum(v: Visual, f: MatchFacts) -> list[str]:
    out = []
    if f.goals:
        for goal in f.goals:
            out.append(
                f"{minute(goal['minute'])} — {goal['player']} ({goal['team']}), "
                f"{n2(goal['xG'])} xG"
                + (", set piece" if goal["set_piece"] else "")
                + (", clear chance" if goal["big_chance"] else "")
                + ".")
    post = [x for x in f.shots if x["type"] == "ShotOnPost"]
    if post:
        hit = post[0]
        out.append(
            f"The nearest miss belonged to {hit['team']}: {hit['player']} hit the frame on "
            f"{minute(hit['minute'])} from a chance worth {n2(hit['xG'])}.")
    if f.errors:
        out.append("Recorded errors: " + listed(
            [f"{e['player']} ({e['team']}) {minute(e['minute'])}" for e in f.errors]) + ".")
    return out or ["The momentum trace follows the shot timeline above."]


@writes("set_pieces", "How each side was set up without the ball is the subject of {next}.")
def _set_pieces(v: Visual, f: MatchFacts) -> list[str]:
    out = [
        f"Corners were {f.home.corners} to {f.home.name} and {f.away.corners} to {f.away.name}. "
        f"Shots from a dead ball: {f.home.name} {f.home.set_piece_shots} worth "
        f"{n2(f.home.set_piece_xg)}, {f.away.name} {f.away.set_piece_shots} worth "
        f"{n2(f.away.set_piece_xg)}."
    ]
    set_piece_goals = [g for g in f.goals if g["set_piece"]]
    if set_piece_goals:
        goal = set_piece_goals[0]
        loser = f.other(f.side(goal["side"]))
        out.append(
            f"{goal['player']} scored from one on {minute(goal['minute'])} from a chance worth "
            f"{n2(goal['xG'])}. That is the value of a dead ball in a match nobody could open: "
            f"it manufactures an outcome from a position the run of play would never have created."
        )
        if loser.set_piece_xg > goal["xG"]:
            out.append(
                f"{loser.name} generated {n2(loser.set_piece_xg)} of set-piece value across "
                f"{loser.set_piece_shots} attempts and scored none of them — more dead-ball value "
                f"than the side that scored from one, and nothing to show for it."
            )
    return out


@writes("defensive_shape", "How each side tried to get through that shape is the point of {next}.")
def _defensive_shape(v: Visual, f: MatchFacts) -> list[str]:
    return [
        f"Average height of defensive actions: {f.home.name} {n1(f.home.def_action_avg_x)}, "
        f"{f.away.name} {n1(f.away.def_action_avg_x)}. Distribution by third — "
        f"{f.home.name} {f.home.def_actions_high} attacking / {f.home.def_actions_mid} middle / "
        f"{f.home.def_actions_low} defensive; {f.away.name} {f.away.def_actions_high} / "
        f"{f.away.def_actions_mid} / {f.away.def_actions_low}.",
        f"Aerial duels finished {f.home.aerials_won} to {f.away.aerials_won}, and fouls "
        f"{f.home.fouls} to {f.away.fouls} with {f.home.cards} and {f.away.cards} cards. "
        f"A block that has to foul is a block that is being moved; one that clears is a block "
        f"that is holding.",
    ]


@writes("playing_through", "The value of each of those actions is measured in {next}.")
def _playing_through(v: Visual, f: MatchFacts) -> list[str]:
    s, other = v.side, f.other(v.side)
    total = s.lane_left + s.lane_centre + s.lane_right
    ranked = sorted((("left", s.lane_left), ("centre", s.lane_centre), ("right", s.lane_right)),
                    key=lambda pair: pair[1], reverse=True)
    out = [
        f"{s.name} entered the final third {total} times through the lanes: "
        f"{s.lane_left} left, {s.lane_centre} centre, {s.lane_right} right. The heaviest lane was "
        f"the {ranked[0][0]}, carrying {ratio(ranked[0][1], total)} of the side's entries."
    ]
    concentration = ratio(ranked[0][1], total)
    if total and ranked[0][1] / total >= 0.45:
        out.append(
            f"Loading one side that heavily is a choice to create an overload rather than to "
            f"attack everywhere at once. It also pulls the opposing block across, which is what "
            f"makes the opposite flank worth using later."
        )
    elif total:
        out.append(
            f"An even spread ({concentration} in the busiest lane) means no overload was built "
            f"anywhere. Against a back five that is the harder way to play: the extra defender "
            f"cancels a one-against-one, so an attack has to manufacture a two-against-one to "
            f"get behind."
        )
    out.append(
        f"For context, {other.name} spread its {other.lane_left + other.lane_centre + other.lane_right} "
        f"entries {other.lane_left} / {other.lane_centre} / {other.lane_right}, and turned them "
        f"into {other.box_entries} box entries against {possessive(s.name)} {s.box_entries}."
    )
    return out


@writes("action_value", "The shape of the passing behind that value is drawn in {next}.")
def _action_value(v: Visual, f: MatchFacts) -> list[str]:
    everyone = [(p, s) for s in f.sides for p in s.players]
    ranked = sorted(everyone, key=lambda pair: pair[0]["positive_xT"], reverse=True)[:6]
    out = ["Highest positive expected threat on the pitch: " + listed(
        [f"{p['name']} {n2(p['positive_xT'])} ({s.name})" for p, s in ranked]) + "."]
    leading_side = ranked[0][1] if ranked else None
    if leading_side:
        own = [pair for pair in ranked if pair[1] is leading_side]
        if len(own) >= 3:
            out.append(
                f"{len(own)} of the top {len(ranked)} belong to {leading_side.name}. When the "
                f"value of individual actions clusters in one team this heavily, the difference "
                f"between the sides sits in what players did with the ball rather than in how "
                f"often they had it."
            )
    return out


@writes("pass_sonar", "Whether those passes became good shots is measured in {next}.")
def _pass_sonar(v: Visual, f: MatchFacts) -> list[str]:
    return [
        f"{f.home.name}: {n0(f.home.passes)} passes averaging {n1(f.home.avg_pass_length)} metres, "
        f"{ratio(f.home.forward_passes, f.home.passes)} forward, {f.home.long_passes} of 30 metres "
        f"or more. {f.away.name}: {n0(f.away.passes)} at {n1(f.away.avg_pass_length)} metres, "
        f"{ratio(f.away.forward_passes, f.away.passes)} forward, {f.away.long_passes} long.",
        f"Similar forward shares with different lengths is the distinction worth holding on to. "
        f"A shorter forward pass leaves a supporting player within range of the next one; a "
        f"longer one skips a line and asks the receiver to hold the ball alone. Directness ran "
        f"{n1(f.home.directness)} against {n1(f.away.directness)}.",
    ]


@writes("finishing_quality", "Where on the pitch that value was won is the subject of {next}.")
def _finishing_quality(v: Visual, f: MatchFacts) -> list[str]:
    out = []
    for s in f.sides:
        lost = s.xg - s.xgot
        out.append(
            f"{s.name}: {n2(s.xg)} expected goals became {n2(s.xgot)} on target and "
            f"{s.goals} actual goals. {ratio(lost, s.xg)} of the pre-shot value never reached "
            f"the frame."
        )
    over = [s for s in f.sides if s.goals > s.xgot + 0.3]
    if over:
        out.append(
            listed([s.name for s in over])
            + f" finished above the post-shot model, which is the part of a result that does not "
              f"repeat. It is worth saying plainly rather than folding into a verdict about "
              f"quality."
        )
    return out


@writes("pitch_control", "What kind of possession produced that control is separated in {next}.")
def _pitch_control(v: Visual, f: MatchFacts) -> list[str]:
    return [
        f"Touches: {f.home.name} {n0(f.home.touches)}, {f.away.name} {n0(f.away.touches)}. "
        f"Possessions: {f.home.possessions} and {f.away.possessions}, lasting "
        f"{n1(f.home.avg_possession_seconds)} and {n1(f.away.avg_possession_seconds)} seconds and "
        f"carrying {n2(f.home.avg_possession_passes)} and {n2(f.away.avg_possession_passes)} "
        f"passes each.",
        f"Average furthest point reached per possession was {n1(f.home.avg_max_x)} for "
        f"{f.home.name} and {n1(f.away.avg_max_x)} for {f.away.name}. Two sides can share the "
        f"ball evenly and still finish their possessions twenty metres apart, and that is the "
        f"number a coach can act on.",
    ]


@writes("sequence_types", "Which of those sequences produced the goals is what {next} traces.")
def _sequence_types(v: Visual, f: MatchFacts) -> list[str]:
    out = []
    for s in f.sides:
        out.append(
            f"{s.name}: {s.possessions} possessions — {s.long_sequences} of six passes or more, "
            f"{s.short_sequences} of two or fewer, {s.fastbreaks} recorded fast breaks."
        )
    breaker = max(f.sides, key=lambda s: s.fastbreaks)
    other = f.other(breaker)
    if breaker.fastbreaks > other.fastbreaks:
        out.append(
            f"{breaker.name} produced {breaker.fastbreaks} fast breaks to {other.fastbreaks}. "
            f"Directness of {n1(f.home.directness)} against {n1(f.away.directness)} says which "
            f"side was travelling in straighter lines — and being more direct while creating less "
            f"is usually a symptom rather than a plan."
        )
    return out


@writes("goal_origins", "What set those moves off is the last mechanism to check, in {next}.")
def _goal_origins(v: Visual, f: MatchFacts) -> list[str]:
    if not f.goals:
        return ["No goals were scored, so the chart carries the shot origins instead."]
    out = []
    for goal in f.goals:
        descriptor = []
        if goal["set_piece"]:
            descriptor.append("from a set piece")
        if goal["header"]:
            descriptor.append("with the head")
        if goal["penalty"]:
            descriptor.append("from the spot")
        if goal["big_chance"]:
            descriptor.append("from a clear chance")
        # The feed does not always record which part of the body the ball was
        # struck with, and "foot" as the fallback was a guess printed as a fact.
        body = goal["body"].lower()
        if body and body not in {"head"} and not goal["header"]:
            descriptor.append(f"struck with the {body.replace('foot', ' foot').strip()}")
        tail = (", " + listed(descriptor)) if descriptor else ""
        out.append(
            f"{minute(goal['minute'])} {goal['player']} ({goal['team']}) — {n2(goal['xG'])} "
            f"expected goals{tail}."
        )
    values = [g["xG"] for g in f.goals]
    out.append(
        f"{_spelled(len(values)).capitalize()} {'finish' if len(values) == 1 else 'finishes'} "
        f"worth {listed([n2(x) for x in values])}. Goals are rarely "
        f"scored from a side's best chances; they are scored from the chances that arrive at the "
        f"right moment, which is why the chance count and the scoreline disagree so often."
    )
    return out


@writes("press_triggers", "From the team the report now turns to the individuals, starting with {next}.")
def _press_triggers(v: Visual, f: MatchFacts) -> list[str]:
    out = []
    for s in f.sides:
        out.append(
            f"{s.name}: {s.high_regains} high regains, {s.counterpress_regains} counterpress "
            f"regains from {s.counterpress_attempts} attempts "
            f"({pct(s.counterpress_success_rate)}), returning {n2(s.regain_xg)} expected goals."
        )
    if f.errors:
        out.append("Errors under pressure: " + listed(
            [f"{e['player']} ({e['team']}) {minute(e['minute'])}" for e in f.errors]) + ".")
    return out


def _scatter(v: Visual, f: MatchFacts, fallback: str) -> list[str]:
    """The three player scatters ship their own reading in the manifest."""
    stored = (f.chart_readings.get(v.path.name) or {})
    reading = str(stored.get("reading") or "").strip()
    method = str(stored.get("method") or "").strip()
    out = [reading] if reading else [fallback]
    if method:
        out.append(f"Method: {method}")
    return out


@writes("player_progression_creation", "The same players measured on involvement rather than progression appear in {next}.")
def _progression_creation(v: Visual, f: MatchFacts) -> list[str]:
    return _scatter(v, f, "Progressive actions on one axis, the value they created on the other.")


@writes("player_involvement_value", "Volume and quality of shooting is the third of these comparisons, in {next}.")
def _involvement_value(v: Visual, f: MatchFacts) -> list[str]:
    return _scatter(v, f, "Touches on one axis, added threat on the other.")


@writes("player_shot_quality", "From the players the report returns to the possessions themselves, in {next}.")
def _shot_quality(v: Visual, f: MatchFacts) -> list[str]:
    return _scatter(v, f, "Shot volume on one axis, average chance quality on the other.")


@writes("possession_speed_value", "How quickly each side moved after winning the ball is measured in {next}.")
def _possession_speed(v: Visual, f: MatchFacts) -> list[str]:
    return [
        f"Short possessions of two passes or fewer numbered {f.home.short_sequences} for "
        f"{f.home.name} and {f.away.short_sequences} for {f.away.name}; sequences of six or more "
        f"ran {f.home.long_sequences} and {f.away.long_sequences}.",
        f"The practical reading is not that longer sequences are better. It is that a sequence is "
        f"worth what it finishes near: {f.home.name} ended its possessions at an average of "
        f"{n1(f.home.avg_max_x)} up the pitch and {f.away.name} at {n1(f.away.avg_max_x)}.",
    ]


@writes("regain_speed", "Where those moves stopped is the question the final chapter answers, beginning with {next}.")
def _regain_speed(v: Visual, f: MatchFacts) -> list[str]:
    fast = min(f.sides, key=lambda s: s.avg_seconds_to_third if s.avg_seconds_to_third else 999)
    slow = f.other(fast)
    return [
        f"Average time to reach the final third: {n1(fast.avg_seconds_to_third)} seconds for "
        f"{fast.name} against {n1(slow.avg_seconds_to_third)} for {slow.name}. "
        f"{fast.reached_third} of {fast.possessions} possessions got there "
        f"({ratio(fast.reached_third, fast.possessions)}), against {slow.reached_third} of "
        f"{slow.possessions} ({ratio(slow.reached_third, slow.possessions)}).",
        f"The clock matters more than the count. Arriving in {n1(fast.avg_seconds_to_third)} "
        f"seconds means arriving while the opposition is still recovering shape; arriving in "
        f"{n1(slow.avg_seconds_to_third)} means arriving to a block that is already set. That "
        f"single difference is most of the gap between {fast.box_entries} box entries and "
        f"{slow.box_entries}.",
    ]


@writes("possession_funnel", "Which route those entries came down is the subject of {next}.")
def _funnel(v: Visual, f: MatchFacts) -> list[str]:
    out = []
    for s in f.sides:
        if s.funnel:
            out.append(f"{s.name}: " + " → ".join(n0(x) for x in s.funnel)
                       + " (possessions, final third, box, shot, on target).")
    drops = []
    for s in f.sides:
        if len(s.funnel) >= 3 and s.funnel[1]:
            drops.append((s, 1.0 - s.funnel[2] / s.funnel[1]))
    if drops:
        out.append(
            "The largest fall for both sides is the same one — final third to box ("
            + listed([f"{s.name} {pct(100 * d, 0)}" for s, d in drops])
            + "). A shared bottleneck means the difference between them is not where attacks "
              "stop but how many arrived at the bottleneck in the first place."
        )
    return out


@writes("entry_routes", "What each side paid for losing the ball is measured in {next}.")
def _entry_routes(v: Visual, f: MatchFacts) -> list[str]:
    # route_entries is nought for both sides whenever the classifier finds no
    # route it recognises, and the paragraph then read "Arsenal: 0 recorded
    # entries, 0 followed by a shot in the same possession (n/a)" twice --
    # a board describing itself as having no content. Say that once, plainly.
    if not any(s.route_entries for s in f.sides):
        return [
            "No entry could be assigned a route on this fixture, so the board has "
            "nothing to separate. The entries themselves are counted on the box-entry "
            "figures; what is missing here is the classification of how each one "
            "arrived, not the arrivals."
        ]
    out = []
    for s in f.sides:
        out.append(
            f"{s.name}: {s.route_entries} recorded entries, {s.route_entries_with_shot} followed "
            f"by a shot in the same possession ({ratio(s.route_entries_with_shot, s.route_entries)})."
        )
    low = min(f.sides, key=lambda s: s.route_entries)
    out.append(
        f"{possessive(low.name)} problem is visible in the first number rather than the second. "
        f"The side was not entering and failing; it was not entering. When it did, the entry "
        f"arrived in a shape that could not be converted — {low.completed_crosses} of "
        f"{low.crosses} crosses found a teammate."
    )
    return out


@writes("loss_consequences", "The last thing to check is whether the changes moved any of it: {next}.")
def _loss_consequences(v: Visual, f: MatchFacts) -> list[str]:
    out = []
    for s in f.sides:
        out.append(
            f"{s.name}: {s.losses} open-play losses, {s.losses_to_third} followed by the "
            f"opponent in the final third, {s.losses_to_box} in the box and {s.losses_to_shot} by "
            f"a shot within twelve seconds, worth {n2(s.loss_xg_conceded)} in total."
        )
    cheap = min(f.sides, key=lambda s: s.loss_xg_conceded)
    dear = f.other(cheap)
    if cheap.losses >= dear.losses:
        out.append(
            f"{cheap.name} gave the ball away more often than {dear.name} and paid roughly "
            f"{times(dear.loss_xg_conceded, max(cheap.loss_xg_conceded, 0.01))} less for it. The "
            f"instruction that follows is not to lose the ball less: it is to lose it where the "
            f"team is organised to cover."
        )
    return out


@writes("substitution_windows", "That closes the chart set; the players who decided it come next.")
def _substitution_windows(v: Visual, f: MatchFacts) -> list[str]:
    frame = f.sub_windows
    out = []
    if frame is not None and len(frame):
        for _, row in frame.iterrows():
            side = f.by_slug("") or None
            team = next((s for s in f.sides if s.team_id == int(float(row["team_id"]))), None)
            out.append(
                f"{team.name if team else 'Team'} — {row['phase'].lower()} the change at "
                f"{float(row['minute']):.0f}': {float(row['xG']):.2f} expected goals across a "
                f"{float(row['window_minutes']):.1f}-minute window."
            )
    else:
        out.append("No substitution cluster had a comparable window either side of it.")
    changes = [s for s in f.substitutions if s["direction"] == "on"]
    if changes:
        out.append("Changes made: " + listed(
            [f"{c['player']} {minute(c['minute'])} ({c['team']})" for c in changes]) + ".")
    out.append(
        "A change in output either side of a substitution is not a measured substitution effect. "
        "Score state moves at the same time, and in a match settled early the changes are usually "
        "made to hold a result rather than to alter one."
    )
    return out


# A count of one, printed as a plural
# ----------------------------------------------------------------------
#
# Thirty f-strings in this module write a number and a noun -- "{s.shots}
# shots", "{p['interceptions']} interceptions" -- and a match where the number
# came out as one printed "1 shots". Inflecting at each site means thirty
# chances to miss one and thirty more every time a writer is added.
#
# The paragraph is normalised once instead, after it is assembled. The nouns
# this module uses are all regular, and the three shapes below cover them:
# "entries" and "recoveries" lose "ies" for "y", "crosses" and "losses" lose
# "es", everything else loses its "s". A qualifier after the noun -- "1 tackles
# won" -- is left alone because only the noun is wrong.
_ONE_THEN_PLURAL = re.compile(
    r"(?<![\d.])\b1\s+([a-z]+(?:ies|sses|ches|shes|xes|s))\b")

# Words that end in "s" and are not plurals of anything.
_NOT_PLURAL = {"across", "less", "press", "possess", "loss", "success", "this",
               "its", "was", "has", "is", "as", "gas", "plus", "minus", "versus"}


def _singular(noun: str) -> str:
    if noun in _NOT_PLURAL:
        return noun
    if noun.endswith("ies") and len(noun) > 4:
        return noun[:-3] + "y"
    for ending in ("sses", "ches", "shes", "xes"):
        if noun.endswith(ending):
            return noun[:-2]
    if noun.endswith("s") and not noun.endswith("ss"):
        return noun[:-1]
    return noun


def one_reads_singular(text: str) -> str:
    """"1 shots" -> "1 shot", leaving every other count alone."""
    return _ONE_THEN_PLURAL.sub(lambda m: "1 " + _singular(m.group(1)), text)


DEFAULT_HANDOFF = "That leads into {next}."


def paragraphs_for(visual: Visual, facts: MatchFacts, next_visual: Visual | None) -> list[str]:
    """The prose under one chart, ending on the sentence that opens the next."""
    writer = WRITERS.get(visual.slug)
    if writer is not None:
        body = writer(visual, facts)
    else:
        stored = (facts.chart_readings.get(visual.path.name) or {})
        reading = str(stored.get("reading") or "").strip()
        method = str(stored.get("method") or "").strip()
        body = [reading] if reading else [
            f"{visual.title} is exported with the package; the figure carries its own title, "
            f"subtitle and footer, which state the denominator and the limitation."]
        if method:
            body.append(f"Method: {method}")
    body = [one_reads_singular(text) for text in body if text]
    if next_visual is not None:
        template = HANDOFFS.get(visual.slug, DEFAULT_HANDOFF)
        label = f"figure {next_visual.number}, {next_visual.title.lower()}" if next_visual.number \
            else next_visual.title.lower()
        body.append(template.format(next=label))
    return body
