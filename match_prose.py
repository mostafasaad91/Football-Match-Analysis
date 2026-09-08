"""Tactical prose for the article and the report, argued rather than recited.

The section text used to be one templated sentence of numbers followed by two
paragraphs that were the same words in every match:

    "Brighton recorded more box entries: Chelsea had 11 and Brighton 16. The
     distinction is between arriving near the attack and accessing the penalty
     area..."

The first half says what happened without saying what it means. The second
half is a methodology note, identical whether the match was 0-0 or 5-4, and a
reader who has met it once learns nothing from meeting it again.

What is written here instead:

* a **reading** of this match's numbers, chosen by what the numbers do. Sixteen
  box entries against eleven is not a finding; sixteen against eleven while the
  side with eleven scored four is. The reading names the mechanism the pair
  implies and says which of the two competing explanations the rest of the data
  supports.

* a **handoff**. Each section closes on the question its own evidence leaves
  open, and the next section opens by answering it, so the piece reads as one
  argument rather than six independent notes.

Nothing here invents a fact. Every claim is a comparison between figures the
package already computed, and where a comparison is too close to carry a claim
the sentence says so rather than picking a side.
"""
from __future__ import annotations


def _plural(count, one, many=None):
    """Singular or plural to match the number printed in front of it."""
    return one if int(round(float(count))) == 1 else (many or one + "s")


def _figures(text):
    """The two-decimal numbers a paragraph prints, in the order it prints them."""
    import re
    return re.findall(r"\d+\.\d{2}", text)


def _pair(context, key):
    """(home value, away value) for a metric, as floats."""
    def read(side):
        try:
            return float(context.get(f"{side}_{key}", 0) or 0)
        except (TypeError, ValueError):
            return 0.0
    return read("home"), read("away")


def _leader(context, key, tolerance=0.0):
    """(name of the side ahead, its value, the other value) or (None, ...).

    ``tolerance`` is the margin below which the pair is called level, because a
    gap of one box entry is not a tactical difference and a sentence built on
    it would be noise wearing the shape of a finding.
    """
    home, away = _pair(context, key)
    if abs(home - away) <= tolerance:
        return None, home, away
    if home > away:
        return context["home"], home, away
    return context["away"], away, home


def _share(part, whole):
    return 100.0 * part / whole if whole else 0.0


# ---------------------------------------------------------------------------
# the readings
# ---------------------------------------------------------------------------

def territory_reading(context):
    """Did holding the ground produce anything, and if not, where did it stop?

    The interesting case is the mismatch. A side with the territory and the
    chances was simply better; a side with the territory and none of the
    chances met a block, and that is a different match to review.
    """
    home, away = context["home"], context["away"]
    tilt_home, tilt_away = _pair(context, "field_tilt")
    xg_home, xg_away = _pair(context, "xG")
    box_home, box_away = _pair(context, "box_entries")
    third_home, third_away = _pair(context, "final_third_entries")

    ground = home if tilt_home > tilt_away else away
    chances = home if xg_home > xg_away else away
    ground_tilt = max(tilt_home, tilt_away)
    other = away if ground == home else home

    if ground == chances:
        return (
            f"{ground} held {ground_tilt:.0f}% of the territory and turned it into the "
            f"better chances, so the two measures agree and the match needs no further "
            f"explanation than that one side was closer to the opposition goal more often."
        )

    # The mismatch. Say where the territory stopped converting.
    ground_third = third_home if ground == home else third_away
    ground_box = box_home if ground == home else box_away
    other_third = third_away if ground == home else third_home
    other_box = box_away if ground == home else box_home
    ground_efficiency = _share(ground_box, ground_third)
    other_efficiency = _share(other_box, other_third)

    line = (
        f"{ground} had {ground_tilt:.0f}% of the territory and {other} had the better "
        f"chances, which is the shape of a match decided somewhere between the two. "
    )
    if ground_efficiency < other_efficiency:
        line += (
            f"The answer is in the conversion of ground into access: {ground} turned "
            f"{ground_third:.0f} final-third entries into {ground_box:.0f} box entries "
            f"({ground_efficiency:.0f}%), while {other} needed only {other_third:.0f} to "
            f"reach the area {other_box:.0f} times ({other_efficiency:.0f}%). Territory was "
            f"being held in front of the block rather than through it."
        )
    else:
        line += (
            f"It is not access that separated them — {ground} reached the box "
            f"{ground_box:.0f} times to {other_box:.0f} — so the difference sits in what "
            f"each side did on arrival rather than in how often they arrived."
        )
    return line


def directness_reading(context):
    """How each side chose to cover the pitch, and what it cost or bought."""
    home, away = context["home"], context["away"]
    direct_home, direct_away = _pair(context, "directness")
    prog_home, prog_away = _pair(context, "progressive_passes")
    touch_home, touch_away = _pair(context, "touches")

    if abs(direct_home - direct_away) < 12:
        return (
            f"Neither side committed to a way of covering the pitch: directness was "
            f"{direct_home:.0f}% and {direct_away:.0f}%, close enough that the ball reached "
            f"the final third by similar means for both."
        )
    direct = home if direct_home > direct_away else away
    patient = away if direct == home else home
    direct_value = max(direct_home, direct_away)
    patient_value = min(direct_home, direct_away)
    direct_touches = touch_home if direct == home else touch_away
    patient_touches = touch_away if direct == home else touch_home
    patient_prog = prog_away if direct == home else prog_home

    return (
        f"{direct} moved the ball forward {direct_value:.0f}% of the distance it travelled "
        f"against {patient_value:.0f}%, and the touch counts say why that was a choice and "
        f"not a symptom: {direct_touches:.0f} touches to {patient_touches:.0f}. "
        f"{direct} was not trying to hold the ball. {patient} built through "
        f"{patient_prog:.0f} progressive passes, which is the slower route to the same "
        f"eighteen yards and a different set of risks on losing it."
    )


def pressing_reading(context):
    """Who pressed, whether it worked, and what it left behind."""
    home, away = context["home"], context["away"]
    ppda_home, ppda_away = _pair(context, "ppda")
    regain_home, regain_away = _pair(context, "high_regains")
    rate_home, rate_away = _pair(context, "regain_to_shot_rate")
    exposed_home, exposed_away = _pair(context, "rest_defence_exposures")

    if not ppda_home or not ppda_away:
        return (
            "Neither side recorded enough opponent passing in the pressing zone for a "
            "PPDA that would describe anything."
        )
    presser = home if ppda_home < ppda_away else away
    sitter = away if presser == home else home
    hard, soft = min(ppda_home, ppda_away), max(ppda_home, ppda_away)
    presser_regains = regain_home if presser == home else regain_away
    presser_rate = rate_home if presser == home else rate_away
    sitter_rate = rate_away if presser == home else rate_home
    presser_exposed = exposed_home if presser == home else exposed_away

    line = (
        f"{presser} pressed at {hard:.1f} passes per defensive action against "
        f"{soft:.1f}, and won the ball high {presser_regains:.0f} times. "
    )
    if presser_rate < sitter_rate:
        line += (
            f"The recoveries did not become chances: {presser_rate:.1f}% of them reached a "
            f"shot, against {sitter_rate:.1f}% for {sitter}. A press that wins the ball and "
            f"then has to build again is paying the running cost without collecting the "
            f"thing it is run for."
        )
    else:
        line += (
            f"They also paid: {presser_rate:.1f}% of regains reached a shot against "
            f"{sitter_rate:.1f}%, so the pressure was converted rather than merely applied."
        )
    line += (
        f" The bill is in the rest defence — {presser_exposed:.0f} advanced losses left "
        f"{sitter} running at a defence that had committed men forward."
    )
    return line


def transition_reading(context):
    """What each side did in the seconds after the ball changed hands."""
    home, away = context["home"], context["away"]
    count_home, count_away = _pair(context, "transitions")
    xg_home, xg_away = _pair(context, "transition_xG")
    goals_home, goals_away = _pair(context, "transition_goals")
    total_xg_home, total_xg_away = _pair(context, "xG")

    if abs(count_home - count_away) <= 3 and abs(xg_home - xg_away) < 0.3:
        return (
            f"The transition game was even: {count_home:.0f} against {count_away:.0f}, "
            f"worth {xg_home:.2f} and {xg_away:.2f} xG. Neither side was living on the "
            f"moments after a turnover."
        )
    better = home if xg_home > xg_away else away
    other = away if better == home else home
    better_xg = max(xg_home, xg_away)
    other_xg = min(xg_home, xg_away)
    better_total = total_xg_home if better == home else total_xg_away
    better_goals = goals_home if better == home else goals_away
    better_count = count_home if better == home else count_away
    other_count = count_away if better == home else count_home
    share = _share(better_xg, better_total)

    line = (
        f"{better} took {better_xg:.2f} xG out of transition against {other_xg:.2f}, from "
        f"{better_count:.0f} of these moments to {other_count:.0f}. "
    )
    if share >= 40:
        line += (
            f"That is {share:.0f}% of everything {better} created, so the transition was not "
            f"a supplement to the attack — it was the attack, and the settled possessions "
            f"either side of it produced comparatively little."
        )
    else:
        line += (
            f"It accounts for {share:.0f}% of what {better} created, so the counter was a "
            f"real weapon without being the whole plan."
        )
    if better_goals:
        line += (
            f" {better_goals:.0f} of the goals came from it, which is the part of the plan "
            f"that survived contact with the scoreline."
        )
    return line


def finishing_reading(context):
    """Whether the score reflected the chances, and which way the gap ran."""
    home, away = context["home"], context["away"]
    goals_home, goals_away = _pair(context, "goals")
    xg_home, xg_away = _pair(context, "xG")
    per_shot_home, per_shot_away = _pair(context, "xG_per_shot")
    shots_home, shots_away = _pair(context, "shots")

    over_home = goals_home - xg_home
    over_away = goals_away - xg_away
    quality = home if per_shot_home > per_shot_away else away
    volume = home if shots_home > shots_away else away

    line = (
        f"{home} scored {goals_home:.0f} from {xg_home:.2f} xG and {away} "
        f"{goals_away:.0f} from {xg_away:.2f}: "
        f"{over_home:+.2f} and {over_away:+.2f} against the chances they made. "
    )
    if max(abs(over_home), abs(over_away)) >= 1.0:
        line += (
            "A gap of a goal or more over ninety minutes is finishing or goalkeeping, and "
            "it is the least repeatable thing in the match — the same chances next week "
            "return a different score. "
        )
    if quality == volume:
        line += (
            f"{quality} had both the volume and the average chance quality "
            f"({per_shot_home if quality == home else per_shot_away:.3f} xG a shot), so the "
            f"shooting profile agreed with the shot count."
        )
    else:
        best_quality = max(per_shot_home, per_shot_away)
        worst_quality = min(per_shot_home, per_shot_away)
        line += (
            f"{volume} took more shots but {quality} took better ones "
            f"({best_quality:.3f} xG a shot against {worst_quality:.3f}); volume and quality "
            f"pointed at different sides, which is usually a side shooting from range "
            f"against a side waiting for the area."
        )
    return line


def state_reading(context):
    """What the scoreline did to the way each side played."""
    home, away = context["home"], context["away"]
    lead_home, lead_away = _pair(context, "game_state_leading_xG")
    trail_home, trail_away = _pair(context, "game_state_trailing_xG")
    level_home, level_away = _pair(context, "game_state_drawing_xG")

    leader = home if lead_home > lead_away else away
    lead_xg = max(lead_home, lead_away)
    chaser = away if leader == home else home
    chase_xg = trail_away if leader == home else trail_home
    level = level_home + level_away

    if lead_xg <= 0 and chase_xg <= 0:
        return (
            "Neither side spent long enough ahead or behind for the score state to have "
            "shaped the chances."
        )
    line = (
        f"{leader} created {lead_xg:.2f} xG while ahead and {chaser} {chase_xg:.2f} while "
        f"behind. "
    )
    if level < 0.2:
        line += (
            "Almost nothing was created while the score was level, so the match was played "
            "under a scoreline from very early and every territorial figure above should be "
            "read as a side protecting something or chasing it, not as a neutral contest."
        )
    else:
        line += (
            f"The level periods produced {level:.2f} xG between them, so there was a genuine "
            f"even contest to compare the rest against."
        )
    return line


# ---------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------

# Each section closes on the question its evidence leaves open, and the next
# one opens by taking that question up. The piece then reads as one argument
# instead of six notes that happen to share a fixture.
HANDOFF = {
    "The result and the chances":
        "That gap between what was created and what was scored is the first thing to "
        "explain, and the explanation starts with where on the pitch the chances came "
        "from.",
    "Chance Creation":
        "Knowing how each side reached the penalty area raises the next question: how "
        "they moved the ball to get there at all.",
    "Possession and Progression":
        "A route into the final third only exists if the opponent allows it, which turns "
        "the question to what each side did without the ball.",
    "Pressing and Rest Defence":
        "Pressure creates turnovers, and a turnover is the start of the fastest attack "
        "either side gets.",
    "Transitions and Efficiency":
        "Those moments arrived under a scoreline that was itself changing, and the "
        "scoreline changes what a side is trying to do.",
    "Match Story":
        "Every figure so far is a team total. The players who produced them are next.",
}

OPENERS = {
    "Chance Creation":
        "Taking up where the chances came from: ",
    "Possession and Progression":
        "Behind that access is the question of how the ball travelled: ",
    "Pressing and Rest Defence":
        "Movement is permitted or contested, so: ",
    "Transitions and Efficiency":
        "The contested ball is where the quickest attacks began: ",
    "Match Story":
        "All of it happened under a scoreline: ",
    "Player involvement":
        "Breaking the team totals into the players who made them: ",
}

def squad_reading(context):
    """Whether the team totals came from a few players or from everyone."""
    home, away = context["home"], context["away"]
    touch_home, touch_away = _pair(context, "touches")
    prog_home, prog_away = _pair(context, "progressive_passes")
    volume = home if touch_home > touch_away else away
    quiet = away if volume == home else home
    high, low = max(touch_home, touch_away), min(touch_home, touch_away)
    return (
        f"{volume} took {high:.0f} touches to {quiet}'s {low:.0f}, so the two squads were "
        f"not being asked the same question: one had to find a way through with the ball "
        f"and the other had to make something of the few times it had it. The five below "
        f"are ranked inside their own line and with defensive work adjusted for how much "
        f"of the ball the opponent had, because on those raw totals a side without "
        f"possession looks like a side defending well."
    )


READINGS = {
    "Player Impact Appendix": squad_reading,
    "Chance Creation": territory_reading,
    "Possession and Progression": directness_reading,
    "Pressing and Rest Defence": pressing_reading,
    "Transitions and Efficiency": transition_reading,
    "Match Story": state_reading,
}


def section_reading(context, group):
    """The tactical paragraph for one section of the article."""
    write = READINGS.get(group)
    if write is None:
        return ""
    opener = OPENERS.get(group, "")
    body = write(context)
    if opener and body:
        # Lowercasing the join blindly turned "Brighton had 70%" into
        # "brighton had 70%". A club name keeps its capital; an ordinary word
        # does not need one mid-sentence.
        first = body.split(" ", 1)[0].rstrip(",.:;")
        names = {str(context.get("home", "")), str(context.get("away", ""))}
        starts_a_name = any(name and name.split(" ")[0] == first for name in names)
        body = opener + (body if starts_a_name else body[0].lower() + body[1:])
    return body


def handoff(group):
    """The sentence that hands this section's open question to the next."""
    return HANDOFF.get(group, "")


# ---------------------------------------------------------------------------
# the players
# ---------------------------------------------------------------------------

def _percentile_frame(observations, minimum_minutes):
    """Everyone eligible, with each metric turned into a within-match rank.

    A defender makes ten recoveries and a forward half an expected goal, so a
    weighted sum of the raw numbers is a sum of different units and the biggest
    unit wins every time. Ranking each metric across the twenty-two players
    first puts them on one scale: 1.0 is the best figure in the match for that
    action, 0.0 the worst, and a centre-back's ten recoveries can then weigh
    against a striker's goal instead of disappearing beside it.
    """
    import pandas as pd

    if observations is None or observations.empty:
        return None
    eligible = observations[observations["minutes"] >= minimum_minutes]
    if len(eligible) < 8:
        eligible = observations
    frame = eligible.copy()
    # Rank inside the line, not across the pitch. A centre-back measured on
    # expected goals against a centre-forward is being scored on a job he was
    # not doing; measured against the other defenders, the question becomes
    # whether he did his own well. Lines too thin to rank inside fall back to
    # the whole pitch rather than producing a percentile out of two players.
    # Not "_line": DataFrame.itertuples renames any column starting with an
    # underscore, which is the same trap that made every percentile read zero.
    from player_advanced import line_of
    frame["line"] = [line_of(role) for role in frame.get("role", "")]
    # The prefix must not start with an underscore: DataFrame.itertuples
    # renames any column whose name is not a valid identifier, so "_p_goals"
    # arrived as "_2" and every getattr for it returned the default. Each
    # player then scored zero, the sort was stable, and the "top five" was the
    # first five names alphabetically -- a substitute with six touches among
    # them.
    for column in _WEIGHTS:
        if column not in frame:
            frame[column] = 0.0
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        # rank rather than min-max: one huge outlier should not flatten
        # everybody else into the bottom of the scale.
        if values.nunique() <= 1:
            frame["pct_" + column] = 0.5
            continue
        within = values.groupby(frame["line"]).rank(pct=True)
        counts = frame.groupby("line")["line"].transform("size")
        # "Unknown" is not a line. The feed gives no position for a substitute,
        # and pooling those eight together made a bench of understudies a
        # comparison group in which every one of them ranked near the top: a
        # 30-metre cameo came out as Chelsea's best performance. Anyone whose
        # line cannot be established is ranked against the whole pitch, where
        # a small contribution reads as a small contribution.
        usable = (counts >= 4) & frame["line"].ne("Unknown")
        frame["pct_" + column] = within.where(usable, values.rank(pct=True))
    return frame


# What a complete match looks like, as weights over ranked contributions.
# Attacking output leads because matches are decided by it, but not by so much
# that only forwards can place: defending, progression and retention together
# outweigh it, so a centre-half who wins everything and passes out cleanly can
# finish above a striker who did nothing but score.
_WEIGHTS = {
    # decisive
    "goals": 1.6,
    "xG": 1.1,
    "xA": 1.1,
    # involvement in what led to chances
    "xGChain": 1.0,
    "positive_xT": 0.9,
    "box_entries": 0.5,
    "final_third_receptions": 0.5,
    # moving the ball, in metres rather than in counts
    "progression_metres": 0.9,
    "line_breaking_passes": 0.6,
    "completed_passes": 0.6,
    "pass_pct": 0.5,
    # winning it back, adjusted for how much of the ball the opponent had
    "padj_defensive_actions": 1.0,
    "aerials_won": 0.6,
    "tackles_won": 0.4,
    "interceptions": 0.4,
    # carrying past someone, and the cost of trying
    "takeons_won": 0.5,
    "dispossessed": -0.4,
    # keeping
    "saves": 1.2,
}


def _contribution(row):
    """One number ranking a whole match performance, all phases counted.

    Read from the pre-ranked columns ``_percentile_frame`` adds, so every term
    is a position on the same 0-1 scale and the weights below mean what they
    say. An earlier version summed raw values with attacking weights on top,
    which ranked the same five forwards in every match ever played.
    """
    total = 0.0
    for column, weight in _WEIGHTS.items():
        try:
            total += weight * float(getattr(row, "pct_" + column, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def top_players(observations, team_id, count=5, minimum_minutes=20):
    """The five best total performances for one side, defenders included."""
    frame = _percentile_frame(observations, minimum_minutes)
    if frame is None:
        return []
    own = frame[frame["team_id"].eq(team_id)].copy()
    if own.empty:
        return []
    own["_score"] = [
        _contribution(row) for row in own.itertuples()
    ]
    return list(own.sort_values("_score", ascending=False).head(count).itertuples())


def player_line(row, team_name, printed=None):
    """What this player's match actually was, led by whatever he did most of.

    The opening clause used to be the goal, then the assist, then everything
    else, so a defender who won the match at the back was introduced by the
    number of touches he took. The lead is now whichever phase of his game
    carried the performance.
    """
    def value(key):
        try:
            return float(getattr(row, key, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    def rank(key):
        try:
            return float(getattr(row, "pct_" + key, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    goals, xg, xa = value("goals"), value("xG"), value("xA")
    chain, xt = value("xGChain"), value("positive_xT")
    progressions, touches = value("progressions"), value("touches")
    shots, minutes = value("shots"), value("minutes")
    saves = value("saves")
    duels = value("tackles_won") + value("interceptions")
    recoveries, clearances = value("recoveries"), value("clearances")
    passes, accuracy = value("completed_passes"), value("pass_pct")
    role = str(getattr(row, "role", "") or "").upper()
    name = str(getattr(row, "player", ""))
    # Every supporting clause opened on a pronoun, so two team-mates whose
    # figures rounded the same way produced the identical sentence twice on one
    # page -- Coventry v Hull printed "He turns up in possessions worth 0.79
    # xG, well beyond the 0.12..." for two different players. Naming the man
    # makes the sentence his.
    from frame_values import surname as _surname
    last = _surname(name) or name

    parts = []
    if role in {"GK", "GOALKEEPER"}:
        opening = f"{name} kept goal for {minutes:.0f} minutes"
        if saves:
            opening += f" and made {saves:.0f} recorded saves"
        parts.append(opening + ".")
        if xt >= 0.8:
            parts.append(
                f"His distribution added {xt:.2f} of positive movement threat, which for a "
                f"goalkeeper measures how much of the build-up ran through him rather than "
                f"around him."
            )
        return " ".join(parts)

    # Lead on the phase this player was strongest in, relative to everyone else
    # on the pitch, not on a fixed order of importance.
    defending = max(rank("tackles_won"), rank("interceptions"), rank("recoveries"))
    attacking = max(rank("goals"), rank("xG"), rank("xA"))
    carrying = max(rank("progressions"), rank("positive_xT"))
    keeping = max(rank("completed_passes"), rank("pass_pct"))

    if goals:
        attempt = "attempt" if shots == 1 else "attempts"
        parts.append(
            f"{name} scored {goals:.0f} from {shots:.0f} {attempt} worth {xg:.2f} xG"
            + (f" and set up {xa:.2f} more" if xa >= 0.1 else "") + "."
        )
    elif attacking >= max(defending, carrying, keeping) and (xa >= 0.15 or xg >= 0.15):
        if xa > xg:
            parts.append(
                f"{name} was the supply line rather than the finish: {xa:.2f} inferred xA "
                f"against {xg:.2f} of his own."
            )
        else:
            position = "shooting position" if shots == 1 else "shooting positions"
            parts.append(
                f"{name} got into {shots:.0f} {position} worth {xg:.2f} xG without scoring."
            )
    elif defending >= max(carrying, keeping) and (duels + recoveries) >= 3:
        detail = []
        if duels:
            # A defender with one tackle and no interception was reading
            # "1 tackles and interceptions".
            detail.append(f"{duels:.0f} " + _plural(duels, "tackle or interception",
                                                    "tackles and interceptions"))
        if recoveries:
            detail.append(f"{recoveries:.0f} " + _plural(recoveries, "recovery", "recoveries"))
        if clearances >= 3:
            detail.append(f"{clearances:.0f} " + _plural(clearances, "clearance"))
        parts.append(
            f"{name}'s match was made without the ball: " + ", ".join(detail) + "."
        )
    elif carrying >= keeping and progressions >= 4:
        parts.append(
            f"{name} carried the ball rather than finished it: {progressions:.0f} "
            f"progressive actions from {touches:.0f} touches."
        )
    elif passes >= 25:
        parts.append(
            f"{name} was the side's connection: {passes:.0f} completed passes at "
            f"{accuracy:.0f}%."
        )
    else:
        parts.append(f"{name} took {touches:.0f} touches across {minutes:.0f} minutes.")

    # One supporting clause, whichever says most beyond the opening.
    own_output = xg + xa
    chain_margin = chain - own_output
    candidates = []
    if chain >= 0.4 and chain_margin >= 0.5:
        candidates.append(
            f"{last} turns up in possessions worth {chain:.2f} xG, well beyond the "
            f"{own_output:.2f} his own shooting and passing account for: the work was two "
            f"or three actions before the chance.")
    if xt >= 1.5:
        candidates.append(
            f"{last}'s {xt:.2f} of positive movement threat is the ball finishing nearer "
            f"goal for his having had it.")
    if duels + recoveries >= 6 and not parts[0].endswith("without the ball."):
        candidates.append(f"{last} also won it back {duels + recoveries:.0f} times.")
    if chain >= 0.4:
        candidates.append(f"The possessions {last} touched carried {chain:.2f} xG.")

    # Two team-mates can arrive at the same figures. Naming them stops the
    # sentence being identical, but a reader still meets "0.79 ... 0.12" twice
    # in one section and reads the second paragraph as the first restated.
    # Take the best clause that does not repeat a set of figures already
    # printed here; if every one of them would, the opening stands alone.
    for sentence in candidates:
        signature = tuple(sorted(_figures(" ".join(parts + [sentence]))))
        if printed is not None and len(signature) >= 2 and signature in printed:
            continue
        if printed is not None:
            printed.add(signature)
        parts.append(sentence)
        break
    return " ".join(parts)


def player_section(observations, context, count=5):
    """A paragraph per side naming the five who decided it, and why."""
    blocks = []
    for side in ("home", "away"):
        team_id = context.get(f"{side}_id")
        team_name = context.get(side, "")
        rows = top_players(observations, team_id, count=count)
        if rows:
            blocks.append((team_name, rows))
    if not blocks:
        return []

    # How the ranking is built belongs to the ranking, not to either side of
    # it. Written under both team headings it was the same five lines printed
    # twice on the same page.
    # One section, so the figures are tracked across both teams' paragraphs.
    printed = set()
    written = [
        "Ranked across every phase of the game. Each action is ranked against the "
        "same action by the other players in that line, and defensive work is "
        "counted per hundred opponent touches, so a centre-back is measured on the "
        "job he was doing rather than on chances he was never going to get."
    ]
    for team_name, rows in blocks:
        written.append(f"**{team_name}** — the {len(rows)} best performances.")
        for index, row in enumerate(rows, start=1):
            written.append(f"{index}. {player_line(row, team_name, printed)}")
    return written
