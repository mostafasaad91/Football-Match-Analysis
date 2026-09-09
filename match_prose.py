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
        # No renderer in this package reads markdown, so asterisks around the
        # club name arrived in the Word file and the PDF as asterisks.
        written.append(f"{team_name} — the {len(rows)} best performances.")
        for index, row in enumerate(rows, start=1):
            written.append(f"{index}. {player_line(row, team_name, printed)}")
    return written


# ---------------------------------------------------------------------------
# what one figure says about this match
# ---------------------------------------------------------------------------
#
# The note under every board was a list of the figures already printed on it,
# followed by a caveat that did not change from match to match:
#
#     "Box entries: Nottingham Forest: 3; Tottenham: 10. Box entry to shot
#      rate: Nottingham Forest: 100.0%; Tottenham: 70.0%. Compare entry routes
#      with the shots that followed in the same possession. A raw entry total
#      and a possession conversion rate use different denominators."
#
# Both halves are true and neither is analysis. The first restates the picture
# in words, and a reader looking at the picture already has it; the second is a
# note on method that was printed whether the match was 0-0 or 5-4. What a
# reader cannot get from the chart is which of the competing explanations the
# rest of this match's figures support.
#
# Each writer below argues one board from the surrounding context and stops.
# Where a board has no writer the caller keeps its old recital, which is at
# least accurate.

def _side_named(stem, context):
    """(side, team) when a filename names one of the two clubs, else (None, "")."""
    import re
    for side in ("home", "away"):
        team = str(context.get(side, "") or "")
        slug = re.sub(r"[^a-z0-9]+", "_", team.lower()).strip("_")
        if slug and slug in stem:
            return side, team
    return None, ""


def _other(side):
    return "away" if side == "home" else "home"


def _one(context, side, key):
    try:
        return float(context.get(f"{side}_{key}", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _xg_flow(context, side, team):
    home, away = context["home"], context["away"]
    hx, ax = _pair(context, "xG")
    hs, as_ = _pair(context, "shots")
    hg, ag = _one(context, "home", "goals"), _one(context, "away", "goals")
    lead = home if hx > ax else away
    lead_xg, other_xg = max(hx, ax), min(hx, ax)
    lead_shots = hs if lead == home else as_
    other_shots = as_ if lead == home else hs
    per = lead_xg / lead_shots if lead_shots else 0.0
    per_other = other_xg / other_shots if other_shots else 0.0
    line = (f"The steps are chances and the flat stretches are everything else. "
            f"{lead} climbed to {lead_xg:.2f} against {other_xg:.2f}")
    if lead_shots and other_shots and per < per_other:
        line += (f", and needed {lead_shots:.0f} attempts against {other_shots:.0f} to "
                 f"do it. Each of its shots was worth {per:.2f} to the other side's "
                 f"{per_other:.2f}, so the higher total is volume rather than better "
                 f"chances. A curve that climbs in small steps is a side shooting "
                 f"from wherever it arrives.")
    else:
        line += (f" from {lead_shots:.0f} shots to {other_shots:.0f}, at {per:.2f} per "
                 f"attempt against {per_other:.2f}. The lead is in the quality of the "
                 f"chances and not only in how many there were.")
    if hg + ag == 0:
        line += (" Neither curve became a goal, which makes where the chances came "
                 "from the question rather than how many there were.")
    return line


def _shot_map(context, side, team):
    if not team:
        return ""
    xg = _one(context, side, "xG")
    shots = _one(context, side, "shots")
    per = xg / shots if shots else 0.0
    box = _one(context, side, "box_entries")
    line = f"{team} took {shots:.0f} shots worth {xg:.2f}, an average of {per:.2f} each. "
    if per >= 0.13:
        line += ("That is a side getting into the chances it wants: the average "
                 "attempt is close enough and central enough to be worth backing, so "
                 "the finishing is what decided the total rather than the positions.")
    elif per >= 0.08:
        line += ("Ordinary chance quality - the average attempt is the sort a "
                 "goalkeeper expects to save. Volume, not position, built the total.")
    else:
        line += ("The average attempt is worth less than a tenth of a goal, which is a "
                 "side shooting because the ball arrived rather than because the "
                 "position was worth it.")
    if box:
        line += (f" With {box:.0f} box entries behind them, the map records what "
                 f"{team} did on arrival rather than how often it arrived.")
    return line


def _xt_map(context, side, team):
    if not team:
        return ""
    tilt = _one(context, side, "field_tilt")
    prog = _one(context, side, "progressive_passes")
    third = _one(context, side, "final_third_entries")
    box = _one(context, side, "box_entries")
    line = (f"The hot zones are where {team}'s successful passes and carries moved the "
            f"ball closer to goal, not where it had the most of it. ")
    if third and box / max(third, 1) < 0.25:
        line += (f"It reached the final third {third:.0f} times and the box {box:.0f}, "
                 f"so the threat this map accumulates was added in front of the area "
                 f"rather than into it. Warm cells wide and high are the signature of "
                 f"possession that arrives and stops.")
    else:
        line += (f"With {prog:.0f} progressive passes behind {box:.0f} box entries, the "
                 f"value shown here converted into access rather than stopping at the "
                 f"edge of it.")
    line += (f" {team} held {tilt:.0f}% of the territory, which is the denominator this "
             f"map should be read against.")
    return line


def _pass_network(context, side, team):
    if not team:
        return ""
    prog = _one(context, side, "progressive_passes")
    touches = _one(context, side, "touches")
    passes = _one(context, side, "completed_passes") or touches
    share = 100.0 * prog / passes if passes else 0.0
    line = (f"Thick links are the connections {team} actually used. Of its passing, "
            f"{share:.0f}% was progressive")
    if share < 12:
        line += (", which is a network built to hold the ball rather than to move it: "
                 "the busiest links will be the sideways ones between the players "
                 "furthest from the opposition goal.")
    else:
        line += (", high enough that the network carried the ball forward and not only "
                 "around. The links worth checking are the ones reaching the last "
                 "third, because those are the ones an opponent has to break.")
    line += (" Node positions are averages over minutes played, so a substitute and the "
             "man he replaced occupy one point between them.")
    return line


def _average_positions(context, side, team):
    if not team:
        return ""
    tilt = _one(context, side, "field_tilt")
    direct = _one(context, side, "directness")
    line = (f"Each point is the mean location of a player's touches, so this is where "
            f"{team} had the ball rather than where it stood. ")
    if tilt >= 55:
        line += (f"At {tilt:.0f}% territory the whole picture sits high, and the "
                 f"distances between the points matter more than their height: a "
                 f"compressed shape this far up is a side that kept the ball and never "
                 f"had to run back.")
    else:
        line += (f"At {tilt:.0f}% territory the shape is pulled toward its own goal, and "
                 f"the gap between the deepest and the highest point is the ground the "
                 f"side had to cover every time it won the ball.")
    line += f" Directness of {direct:.0f}% says how much of that ground was covered forward."
    return line


def _pitch_control(context, side, team):
    shares = context.get("influence", {}) or {}
    home, away = context["home"], context["away"]
    h = float(shares.get("home", 0) or 0)
    a = float(shares.get("away", 0) or 0)
    contested = float(shares.get("contested", 0) or 0)
    hx, ax = _pair(context, "xG")
    lead = home if h > a else away
    chance_lead = home if hx > ax else away
    line = (f"The surface is modelled from average touch positions in short windows: "
            f"{home} {h:.0f}%, {away} {a:.0f}%, {contested:.0f}% contested. ")
    if contested >= 30:
        line += ("Nearly a third of the pitch belonged to neither side, which is a match "
                 "played in the space between the lines rather than in either half. ")
    if lead != chance_lead:
        line += (f"{lead} controlled more ground and {chance_lead} made the better "
                 f"chances, so occupying space and using it came apart here. Space is a "
                 f"precondition, not an outcome.")
    else:
        line += (f"{lead} held both the space and the chances, which is the ordinary "
                 f"case and needs no explanation beyond one side being on top.")
    return line


def _defensive_shape(context, side, team):
    if not team:
        home, away = context["home"], context["away"]
        hp, ap = _pair(context, "ppda")
        hh, ah = _pair(context, "defensive_action_height")
        higher = home if hh > ah else away
        lower = away if higher == home else home
        line = (f"Where each side chose to defend, which is a choice and not a "
                f"consequence: {home} allowed {hp:.1f} opponent passes per defensive "
                f"action and {away} {ap:.1f}. ")
        if hh or ah:
            line += (f"{higher} won the ball an average of {max(hh, ah):.0f} metres "
                     f"upfield against {min(hh, ah):.0f} for {lower}, and that gap is "
                     f"the ground {lower} had to cover on every attack it started.")
        else:
            line += ("The side engaging earlier accepts less cover behind the ball; the "
                     "one waiting accepts the territory. Neither is free.")
        return line
    ppda = _one(context, side, "ppda")
    regains = _one(context, side, "high_regains")
    line = f"The spread of {team}'s defensive actions is where it chose to defend. "
    if ppda and ppda < 9:
        line += (f"At {ppda:.1f} opponent passes allowed per action it engaged early, so "
                 f"the actions should cluster high and the {regains:.0f} high regains "
                 f"are the return on that running.")
    else:
        line += (f"At {ppda:.1f} opponent passes allowed per action it let the ball come "
                 f"to it, so the cluster sits deep and the defending was about the last "
                 f"thirty metres rather than the first.")
    line += (" Where a side wins the ball decides how far it then has to travel, which "
             "is the cost this chart does not show.")
    return line


def _playing_through(context, side, team):
    if not team:
        return ""
    deep = _one(context, side, "deep_completions")
    prog = _one(context, side, "progressive_passes")
    box = _one(context, side, "box_entries")
    line = (f"{team} completed {prog:.0f} progressive passes and {deep:.0f} of them "
            f"landed near the opponent's last line. ")
    if deep and box / max(deep, 1) >= 0.6:
        line += ("Most of what got behind the line turned into box access, so these were "
                 "the passes that mattered rather than the ones that merely arrived.")
    else:
        line += ("Getting behind the line was not the same as getting into the area: the "
                 "receptions happened and the attack still had to be rebuilt from there.")
    line += (" The line is inferred from where the opponent defended, so each highlighted "
             "pass is a candidate to check rather than a confirmed break.")
    return line


def _unlocking(context, side, team):
    if not team:
        return ""
    deep = _one(context, side, "deep_completions")
    box = _one(context, side, "box_entries")
    line = (f"These are receptions in the space the opponent was trying to protect: "
            f"{deep:.0f} of them for {team}, against {box:.0f} box entries. ")
    if deep >= 6:
        line += ("A number that high means the space was available repeatedly rather "
                 "than once, which moves the question from finding it to using it.")
    else:
        line += ("Few enough that the space was closed rather than found, which moves the "
                 "question to whether the ball ever arrived in a position to use it.")
    line += (" The event record cannot say which way the receiver was facing, and that is "
             "usually what separates a reception from a chance.")
    return line


def _press_triggers(context, side, team):
    if not team:
        home, away = context["home"], context["away"]
        hr, ar = _pair(context, "high_regains")
        hs, as_ = _pair(context, "regain_to_shot_rate")
        lead = home if hr > ar else away
        other = away if lead == home else home
        lead_rate = hs if lead == home else as_
        other_rate = as_ if lead == home else hs
        return (f"What the opponent was doing in the moment before the ball was lost "
                f"high: {lead} forced {max(hr, ar):.0f} of these against "
                f"{min(hr, ar):.0f}. The count is only half the question - "
                f"{lead_rate:.0f}% of {lead}'s reached a shot and {other_rate:.0f}% of "
                f"{other}'s did, and a recovery the opponent has time to defend is a "
                f"turnover rather than an attack. The preceding action is an "
                f"association and not proof of a coached cue.")
    regains = _one(context, side, "high_regains")
    rate = _one(context, side, "regain_to_shot_rate")
    line = (f"What the opponent did immediately before {team} won the ball high, "
            f"{regains:.0f} times. ")
    if rate >= 12:
        line += (f"{rate:.0f}% of those became shots, so the moments shown here were "
                 f"worth the running: the ball was won where the opponent could not "
                 f"reorganise in time.")
    else:
        line += (f"Only {rate:.0f}% became shots, so the ball was won high and the attack "
                 f"still had to be built from scratch. Winning it early is half of a "
                 f"press; the other half is being ready to attack when it arrives.")
    line += " The preceding action is an association and not proof of a coached cue."
    return line


def _ppda(context, side, team):
    home, away = context["home"], context["away"]
    hp, ap = _pair(context, "ppda")
    hr, ar = _pair(context, "high_regains")
    hs, as_ = _pair(context, "regain_to_shot_rate")
    if not hp or not ap:
        return ("Neither side allowed enough opponent passing in the pressing zone for "
                "this measure to describe anything about the match.")
    presser = home if hp < ap else away
    sitter = away if presser == home else home
    hard, soft = min(hp, ap), max(hp, ap)
    press_regains = hr if presser == home else ar
    press_rate = hs if presser == home else as_
    sit_rate = as_ if presser == home else hs
    line = (f"{presser} allowed {hard:.1f} opponent passes per defensive action and "
            f"{sitter} allowed {soft:.1f}, so one side went to the ball and the other "
            f"waited for it. {presser} won it high {press_regains:.0f} times. ")
    if press_rate < sit_rate:
        line += (f"The return does not follow the effort: {press_rate:.0f}% of those "
                 f"recoveries reached a shot against {sit_rate:.0f}% for {sitter}. "
                 f"Winning the ball early pays only when the attack is already there to "
                 f"start.")
    else:
        line += (f"And it was paid for: {press_rate:.0f}% of them reached a shot against "
                 f"{sit_rate:.0f}%, which is a press producing chances rather than only "
                 f"turnovers.")
    return line


def _high_regains(context, side, team):
    home, away = context["home"], context["away"]
    hr, ar = _pair(context, "high_regains")
    hs, as_ = _pair(context, "regain_to_shot_rate")
    if team:
        regains = _one(context, side, "high_regains")
        rate = _one(context, side, "regain_to_shot_rate")
        opponent = context[_other(side)]
        tail = ("That is a side turning pressure into chances inside the few seconds an "
                "opponent spends out of position."
                if rate >= 12 else
                f"The other {100 - rate:.0f}% were recoveries {opponent} had time to "
                f"defend, which is the difference between winning the ball high and "
                f"attacking from it.")
        return (f"{team} won the ball high {regains:.0f} times and {rate:.0f}% of those "
                f"possessions reached a shot. " + tail)
    return (f"{home if hr > ar else away} won the ball high more often, "
            f"{max(hr, ar):.0f} to {min(hr, ar):.0f}, at conversion rates of {hs:.0f}% "
            f"and {as_:.0f}%. The count says who pressed and the rate says whether it "
            f"was worth it; the two do not have to agree.")


def _box_entries(context, side, team):
    if team:
        box = _one(context, side, "box_entries")
        third = _one(context, side, "final_third_entries")
        rate = _one(context, side, "box_entry_to_shot_rate")
        share = 100.0 * box / third if third else 0.0
        opponent = context[_other(side)]
        line = (f"{team} turned {third:.0f} final-third entries into {box:.0f} box "
                f"entries, {share:.0f}% of them, and {rate:.0f}% of those reached a "
                f"shot. ")
        if share < 20:
            line += (f"Four in five arrivals in the last third never became access, so "
                     f"{opponent}'s block was doing its work at the edge of the area "
                     f"rather than inside it. The entries that did get through are the "
                     f"ones worth studying, because they are the pattern that worked.")
        elif rate >= 70:
            line += ("Getting in nearly always meant a shot, so the constraint was "
                     "arriving rather than deciding once there.")
        else:
            line += ("The ball got in and often came back out, which points at the "
                     "decision on arrival rather than at the route into the area.")
        return line
    home, away = context["home"], context["away"]
    hb, ab = _pair(context, "box_entries")
    ht, at = _pair(context, "final_third_entries")
    hr, ar = _pair(context, "box_entry_to_shot_rate")
    lead = home if hb > ab else away
    other = away if lead == home else home
    lead_box, other_box = max(hb, ab), min(hb, ab)
    lead_third = ht if lead == home else at
    other_third = at if lead == home else ht
    lead_share = 100.0 * lead_box / lead_third if lead_third else 0.0
    other_share = 100.0 * other_box / other_third if other_third else 0.0
    lead_rate = hr if lead == home else ar
    other_rate = ar if lead == home else hr
    line = (f"{lead} reached the penalty area {lead_box:.0f} times to {other_box:.0f}, "
            f"and the entries are worth reading against the arrivals behind them: "
            f"{lead_share:.0f}% of {lead}'s final-third entries became box entries "
            f"against {other_share:.0f}% for {other}. ")
    if other_rate > lead_rate and other_box:
        line += (f"The smaller number was the sharper one - {other} put {other_rate:.0f}% "
                 f"of its entries to a shot against {lead_rate:.0f}% - so one side got "
                 f"in often and the other got in when it meant something.")
    else:
        line += (f"{lead} also finished the job more often, {lead_rate:.0f}% of entries "
                 f"reaching a shot against {other_rate:.0f}%, so the access and the use "
                 f"of it point the same way.")
    return line


def _progressive(context, side, team):
    if team:
        prog = _one(context, side, "progressive_passes")
        deep = _one(context, side, "deep_completions")
        box = _one(context, side, "box_entries")
        opponent = context[_other(side)]
        line = (f"{team} played {prog:.0f} progressive passes and {deep:.0f} of them "
                f"finished near {opponent}'s last line. ")
        if prog and deep / prog < 0.12:
            line += (f"Most of the forward passing stopped short of the place it had to "
                     f"reach: ground was gained and the last line was not, which is why "
                     f"{box:.0f} box entries followed all that progression.")
        else:
            line += (f"That is forward passing arriving where it is hard to defend "
                     f"rather than only gaining ground, and {box:.0f} box entries came "
                     f"out of it.")
        return line
    home, away = context["home"], context["away"]
    hp, ap = _pair(context, "progressive_passes")
    hd, ad = _pair(context, "deep_completions")
    lead = home if hp > ap else away
    lead_prog, other_prog = max(hp, ap), min(hp, ap)
    lead_deep = hd if lead == home else ad
    other_deep = ad if lead == home else hd
    line = f"{lead} played {lead_prog:.0f} progressive passes to {other_prog:.0f}. "
    if other_deep > lead_deep:
        line += (f"The deep completions run the other way, {other_deep:.0f} to "
                 f"{lead_deep:.0f}: progression that stops short of the last line is "
                 f"movement without arrival, and this chart separates the two.")
    else:
        line += (f"{lead_deep:.0f} of them finished near the opponent's last line against "
                 f"{other_deep:.0f}, so the forward passing reached the places that are "
                 f"hard to defend rather than only gaining ground.")
    return line


def _ball_losses(context, side, team):
    if not team:
        home, away = context["home"], context["away"]
        he, ae = _pair(context, "rest_defence_exposures")
        hd, ad = _pair(context, "rest_defence_dangerous_counters")
        hshare = 100.0 * hd / he if he else 0.0
        ashare = 100.0 * ad / ae if ae else 0.0
        worse = home if hshare > ashare else away
        better = away if worse == home else home
        return (f"Not where the ball was lost but what it cost: {home} was exposed "
                f"{he:.0f} times and punished {hd:.0f}, {away} {ae:.0f} and {ad:.0f}. "
                f"{worse} paid on {max(hshare, ashare):.0f}% of its advanced losses "
                f"against {min(hshare, ashare):.0f}% for {better}, which is a question "
                f"about who was left behind the ball rather than about carelessness in "
                f"front of it.")
    exposures = _one(context, side, "rest_defence_exposures")
    dangerous = _one(context, side, "rest_defence_dangerous_counters")
    opponent = context[_other(side)]
    share = 100.0 * dangerous / exposures if exposures else 0.0
    line = (f"{team} lost the ball {exposures:.0f} times in positions its own defence had "
            f"not recovered from, and {dangerous:.0f} of those became a dangerous counter "
            f"for {opponent}")
    if share >= 25:
        line += (f" - {share:.0f}%, high enough that the losses shown here are a "
                 f"structural cost of how the side attacked rather than accidents.")
    else:
        line += (f" - {share:.0f}%, so the losses happened and the rest defence mostly "
                 f"held. Read the map for where they happened rather than for what they "
                 f"cost.")
    return line


def _transition(context, side, team):
    home, away = context["home"], context["away"]
    ht, at = _pair(context, "transitions")
    hx, ax = _pair(context, "transition_xG")
    total_h, total_a = _pair(context, "xG")
    lead = home if hx > ax else away
    lead_xg, other_xg = max(hx, ax), min(hx, ax)
    lead_count = ht if lead == home else at
    other_count = at if lead == home else ht
    total = total_h if lead == home else total_a
    share = 100.0 * lead_xg / total if total else 0.0
    line = (f"{lead} took {lead_xg:.2f} expected goals out of transition against "
            f"{other_xg:.2f}, from {lead_count:.0f} of these moments to "
            f"{other_count:.0f}. ")
    if share >= 40:
        line += (f"That is {share:.0f}% of everything it created, so the transition was "
                 f"not a supplement to the attack - it was the attack, and the settled "
                 f"possession is the part that produced nothing.")
    elif (lead_count > other_count
          and lead_xg / max(lead_count, 1) < other_xg / max(other_count, 1)):
        line += ("More of them and less from each: the moments were available and the "
                 "decisions inside them were not sharp enough to use them.")
    else:
        line += (f"At {share:.0f}% of its total it was one route among several, which "
                 f"means the side did not have to break to threaten.")
    return line


def _zone14(context, side, team):
    if not team:
        return ""
    deep = _one(context, side, "deep_completions")
    box = _one(context, side, "box_entries")
    line = (f"Central receptions in front of the area, where a forward-facing player can "
            f"pick a pass: {deep:.0f} of them for {team}, against {box:.0f} box entries. ")
    if deep and box / max(deep, 1) >= 0.8:
        line += ("The central route was the route: most of what arrived here continued "
                 "into the area rather than being recycled.")
    else:
        line += ("Arriving centrally was not enough on its own; the ball reached the zone "
                 "and the attack was rebuilt from there more often than it went through.")
    return line


def _pass_map(context, side, team):
    if not team:
        return directness_reading(context)
    share = _one(context, side, "pass_share")
    prog = _one(context, side, "progressive_passes")
    direct = _one(context, side, "directness")
    third = _one(context, side, "final_third_entries")
    line = (f"{team} had {share:.0f}% of the passing and {prog:.0f} of it was "
            f"progressive, at {direct:.0f}% directness. ")
    if share >= 55 and direct < 40:
        line += ("A side with the ball and no hurry: the highlighted movements will be "
                 "the ones that had to break a set defence, because nothing else was "
                 "going to.")
    elif share < 45 and direct >= 45:
        line += ("Less of the ball and more urgency with it - the map should show long "
                 "movements from deep rather than circulation, which is what a side "
                 "does when it cannot expect a second possession.")
    else:
        line += (f"Balanced enough that neither the volume nor the speed explains the "
                 f"{third:.0f} final-third entries on its own; the route does.")
    return line


def _dominating(context, side, team):
    return territory_reading(context)


def _win_probability(context, side, team):
    home, away = context["home"], context["away"]
    hg, ag = _one(context, "home", "goals"), _one(context, "away", "goals")
    hx, ax = _pair(context, "xG")
    lead = home if hx > ax else away
    line = "The curve is a heuristic on score and time, not a forecast. "
    if hg == ag:
        line += (f"It stays near even because the score did, which is the point worth "
                 f"taking from it: neither side was ever playing a scoreline, so the "
                 f"{hx + ax:.2f} combined expected goals were created against a settled "
                 f"game rather than a chasing one.")
    else:
        winner = home if hg > ag else away
        line += (f"Each step is a goal, and the shape between them says how long "
                 f"{winner} spent protecting the result. A side ahead defends "
                 f"differently, so the chances created after the last step belong to a "
                 f"different match from the ones before it.")
        if lead != winner:
            line += (f" {lead} made the better chances and did not win, which the curve "
                     f"cannot show and the expected goals can.")
    return line


def _goalkeeper(context, side, team):
    if not team:
        home, away = context["home"], context["away"]
        hx, ax = _pair(context, "xG")
        hg, ag = _one(context, "home", "goals"), _one(context, "away", "goals")
        # Each keeper is judged against what the *other* side created.
        home_gap, away_gap = ax - ag, hx - hg
        if abs(home_gap - away_gap) < 0.5:
            return (f"Both goals were beaten about as often as the chances said they "
                    f"should be - {home} conceded {ag:.0f} from {ax:.2f} and {away} "
                    f"{hg:.0f} from {hx:.2f} - so neither goalkeeper separates himself "
                    f"here. The saves worth watching are the individual ones.")
        better = home if home_gap > away_gap else away
        gap = max(home_gap, away_gap)
        return (f"{better}'s goal gave up {gap:.2f} fewer than the chances against it "
                f"were worth. Over one match that is shot-stopping or it is the "
                f"opposition finishing badly, and only the placement of the individual "
                f"attempts tells you which. The model behind these figures is "
                f"uncalibrated and knows neither shot speed nor the keeper's position.")
    opponent = context[_other(side)]
    faced = _one(context, _other(side), "shots")
    faced_xg = _one(context, _other(side), "xG")
    conceded = _one(context, _other(side), "goals")
    line = (f"{team}'s goal faced {faced:.0f} attempts worth {faced_xg:.2f} and conceded "
            f"{conceded:.0f}. ")
    gap = faced_xg - conceded
    if gap >= 0.7:
        line += (f"That is {gap:.2f} fewer than the chances were worth, which over one "
                 f"match is shot-stopping or it is {opponent} finishing badly, and only "
                 f"the placement of the individual attempts separates the two.")
    elif gap <= -0.7:
        line += (f"That is {abs(gap):.2f} more than the chances were worth. One match "
                 f"cannot tell a goalkeeping error from a well-struck shot; the "
                 f"individual attempts can.")
    else:
        line += ("Chances and goals matched closely enough that this figure says nothing "
                 "about the goalkeeping either way.")
    line += (" The post-shot estimate here is uncalibrated and knows neither shot speed "
             "nor the keeper's position.")
    return line


def _set_piece(context, side, team):
    hx, ax = _pair(context, "xG")
    who = f"{team}'s" if team else "Both sides'"
    return (f"{who} dead-ball attempts, kept apart from open play because they are a "
            f"different problem: the defence is set, the delivery is chosen, and nothing "
            f"about the {hx + ax:.2f} expected goals in this match tells you whether a "
            f"side is good at them. Judge the routine and the first contact, not the "
            f"total.")


def _possession_funnel(context, side, team):
    home, away = context["home"], context["away"]
    hb, ab = _pair(context, "box_entries")
    ht, at = _pair(context, "final_third_entries")
    hshare = 100.0 * hb / ht if ht else 0.0
    ashare = 100.0 * ab / at if at else 0.0
    worse = home if hshare < ashare else away
    better = away if worse == home else home
    return (f"Each step drops the possessions that did not reach the next stage, and the "
            f"biggest drop is the one to fix. {worse} lost the most between the final "
            f"third and the area - {min(hshare, ashare):.0f}% survived that step against "
            f"{max(hshare, ashare):.0f}% for {better} - so its problem was access rather "
            f"than possession: the ball arrived in the last third and stopped there.")


def _entry_routes(context, side, team):
    home, away = context["home"], context["away"]
    hb, ab = _pair(context, "box_entries")
    hr, ar = _pair(context, "box_entry_to_shot_rate")
    lead = home if hb > ab else away
    other = away if lead == home else home
    lead_rate = hr if lead == home else ar
    other_rate = ar if lead == home else hr
    tail = (f"{other} was the more decisive on arrival despite the smaller total."
            if other_rate > lead_rate else
            f"{lead} was both the more frequent and the more decisive, which is the "
            f"straightforward case.")
    return (f"How each side got in, and whether the way in produced a shot in the same "
            f"possession: {lead} at {lead_rate:.0f}% and {other} at {other_rate:.0f}%. A "
            f"route that arrives and recycles is not the same attack as one that arrives "
            f"and shoots, and a count of entries hides the difference. " + tail)


def _game_state(context, side, team):
    return state_reading(context) + (
        " Exposure time matters as much as the totals: a side that spent ten minutes "
        "behind cannot be judged on what it created while chasing.")


def _pass_targets(context, side, team):
    if not team:
        return directness_reading(context)
    share = _one(context, side, "pass_share")
    third = _one(context, side, "final_third_entries")
    box = _one(context, side, "box_entries")
    opponent = context[_other(side)]
    line = (f"Where {team}'s passes were aimed, over {share:.0f}% of the match's "
            f"passing. ")
    if third and box / max(third, 1) < 0.25:
        line += (f"The destinations to look at are the ones inside the last third: "
                 f"{third:.0f} arrivals produced {box:.0f} entries, so the ball was "
                 f"being sent to the front of {opponent}'s block rather than past it.")
    else:
        line += (f"{third:.0f} arrivals in the last third became {box:.0f} entries, so "
                 f"the destinations were carrying the attack through rather than "
                 f"stopping it at the line.")
    line += (" A destination is where the ball went, not where it was meant to go; a "
             "failed pass records the interception, not the intention.")
    return line


def _defensive_activity(context, side, team):
    if not team:
        return pressing_reading(context)
    ppda = _one(context, side, "ppda")
    regains = _one(context, side, "high_regains")
    height = _one(context, side, "defensive_action_height")
    opponent = context[_other(side)]
    line = (f"Every recorded defensive action by {team}: {regains:.0f} of them won the "
            f"ball high, at {ppda:.1f} {opponent} passes allowed per action. ")
    if ppda and ppda < 9:
        line += ("The density should sit in the opponent's half. A side defending that "
                 "far from its own goal is choosing to be short of cover behind the "
                 "ball, and the loss maps are where that bill arrives.")
    else:
        line += ("The density should sit in its own half. Defending deep concedes the "
                 "ball and the ground with it, and buys a compact block in exchange - "
                 "the trade shows up in what the opponent did on arrival, not here.")
    if height:
        line += f" Mean action height was {height:.0f} metres upfield."
    return line


def _xt_per_minute(context, side, team):
    home, away = context["home"], context["away"]
    hx, ax = _pair(context, "sequence_xT")
    hg, ag = _one(context, "home", "goals"), _one(context, "away", "goals")
    lead = home if hx > ax else away
    other = away if lead == home else home
    return (f"Threat added minute by minute, which is a different question from when the "
            f"shots came: {lead} accumulated {max(hx, ax):.2f} of it against "
            f"{min(hx, ax):.2f}. The peaks are spells in which the ball was repeatedly "
            f"moved into more dangerous ground, whether or not anything was struck at "
            f"the end of them. A flat stretch for {other} is not a quiet spell; it is a "
            f"spell in which possession did not travel."
            + (" With the match goalless, these peaks are the whole of what either side "
               "built." if hg + ag == 0 else ""))


def _player_sequence_leaders(context, side, team):
    home, away = context["home"], context["away"]
    hx, ax = _pair(context, "xG")
    return (f"Possession credit, not chance creation: a player appears here for having "
            f"been in the moves that ended in a shot, wherever in the move he was. It "
            f"names the players a side's attack ran through, which is why a deep "
            f"midfielder can outrank a forward on a day the forward touched nothing. "
            f"The credit is shared across everyone in the possession and does not add "
            f"up to the {hx:.2f} and {ax:.2f} expected goals the two sides made - read "
            f"it as involvement, and check the ranking against who actually finished.")


def _pass_sonar(context, side, team):
    home, away = context["home"], context["away"]
    hd, ad = _pair(context, "directness")
    lead = home if hd > ad else away
    other = away if lead == home else home
    return (f"Each spoke is a direction and its length is how much passing went that "
            f"way. {lead} played the more direct match, {max(hd, ad):.0f}% against "
            f"{min(hd, ad):.0f}%, so its sonar should lean forward while {other}'s "
            f"opens sideways. A fan that is symmetrical about the halfway line is a "
            f"side circulating; one that is lopsided is a side with a preferred route, "
            f"and a preferred route is something an opponent can close.")


def _finishing_quality(context, side, team):
    home, away = context["home"], context["away"]
    hx, ax = _pair(context, "xG")
    hg, ag = _one(context, "home", "goals"), _one(context, "away", "goals")
    hs, as_ = _pair(context, "shots")
    hgap, agap = hg - hx, ag - ax
    if abs(hgap - agap) < 0.4:
        return (f"Chance quality against what was scored: {home} {hg:.0f} from "
                f"{hx:.2f}, {away} {ag:.0f} from {ax:.2f}. Both sides finished about as "
                f"the chances said they would, so nothing here separates them and the "
                f"explanation for the result sits further back, in who made the chances "
                f"rather than in who took them.")
    better = home if hgap > agap else away
    worse = away if better == home else home
    return (f"Chance quality against what was scored: {home} {hg:.0f} from {hx:.2f}, "
            f"{away} {ag:.0f} from {ax:.2f}. {better} came out ahead of its chances and "
            f"{worse} behind them. Over ninety minutes that is the least repeatable "
            f"thing on this page - {int(hs + as_)} shots is a sample that tells you "
            f"about this afternoon and nothing about either side's finishing.")


def _sequence_types(context, side, team):
    home, away = context["home"], context["away"]
    ht, at = _pair(context, "transitions")
    hx, ax = _pair(context, "transition_xG")
    thx, tax = _pair(context, "xG")
    hshare = 100.0 * hx / thx if thx else 0.0
    ashare = 100.0 * ax / tax if tax else 0.0
    reliant = home if hshare > ashare else away
    other = away if reliant == home else home
    return (f"How each attack started, which is the question of whether a side had more "
            f"than one way to threaten. {reliant} took {max(hshare, ashare):.0f}% of its "
            f"chances out of transition against {min(hshare, ashare):.0f}% for {other}, "
            f"from {ht:.0f} and {at:.0f} of those moments. A side heavily weighted to "
            f"one starting condition is a side that needs the opponent to give it "
            f"something; the build-up bars are what it can do when nothing is given.")


def _match_momentum(context, side, team):
    home, away = context["home"], context["away"]
    hx, ax = _pair(context, "xG")
    hg, ag = _one(context, "home", "goals"), _one(context, "away", "goals")
    ht, at = _pair(context, "transition_xG")
    lead = home if hx > ax else away
    other = away if lead == home else home
    swing = 100.0 * max(ht, at) / max(hx if lead == home else ax, 0.01)
    line = (f"Five-minute blocks of shot value, which answers when rather than how "
            f"much: {lead} finished ahead {max(hx, ax):.2f} to {min(hx, ax):.2f}. ")
    if swing >= 40:
        line += (f"The spikes should sit around the turnovers - {max(ht, at):.2f} of "
                 f"that total came out of transition - so this is a match of bursts "
                 f"rather than of pressure, and a flat stretch is not a quiet period so "
                 f"much as a period nobody lost the ball badly in.")
    else:
        line += (f"Chances arrived across the match rather than in one spell, which "
                 f"means {other} was under a steady problem rather than a moment of "
                 f"one, and there is no single passage to point at.")
    if hg + ag == 0:
        line += " Nothing here became a goal, so every spike is a spell that led nowhere."
    return line


def _action_value(context, side, team):
    home, away = context["home"], context["away"]
    hx, ax = _pair(context, "sequence_xT")
    hxg, axg = _pair(context, "xG")
    lead = home if hx > ax else away
    other = away if lead == home else home
    chance_lead = home if hxg > axg else away
    line = (f"Value added by moving the ball, scored on where it went rather than on "
            f"what happened at the end: {lead} accumulated {max(hx, ax):.2f} against "
            f"{min(hx, ax):.2f}. ")
    if lead != chance_lead:
        line += (f"{chance_lead} made the better chances on less of this, which is the "
                 f"gap between carrying the ball into dangerous ground and arriving "
                 f"there with someone able to shoot. Movement is not chance creation, "
                 f"and this board measures the first.")
    else:
        line += (f"{lead} led the movement and the chances, so the ground it gained was "
                 f"the ground it used. {other} did neither.")
    line += (" The zone values are an explicit model and not a fitted rating; treat the "
             "ordering as informative and the level as arbitrary.")
    return line


def _goal_origins(context, side, team):
    rows = context.get("goal_rows", []) or []
    home, away = context["home"], context["away"]
    hx, ax = _pair(context, "xG")
    if not rows:
        return (f"No goals to trace. {home} and {away} made {hx:.2f} and {ax:.2f} "
                f"expected goals and finished none of them, so what this board would "
                f"normally show - where the winning possessions began - has to be read "
                f"off the chances instead. The possessions worth following back are the "
                f"ones that reached a shot, not the ones that lasted longest.")
    from match_clock import event_label
    named = "; ".join(
        f"{row['team']} through {row.get('player') or 'an unnamed scorer'} at "
        f"{event_label(row)}" for row in rows)
    return (f"Each goal traced to the moment its possession started: {named}. What to "
            f"read here is the distance between the origin and the finish. A possession "
            f"that began in the attacking half is a turnover converted; one that began "
            f"at the goalkeeper is a side that played through the opponent. The "
            f"starting location alone cannot tell you the opponent was disorganised - "
            f"the actions in between can.")


def _substitution_windows(context, side, team):
    home, away = context["home"], context["away"]
    hx, ax = _pair(context, "xG")
    hg, ag = _one(context, "home", "goals"), _one(context, "away", "goals")
    line = ("Output in the minutes either side of each change, which is the closest "
            "this data comes to asking whether a substitution did anything. ")
    if hg == ag:
        line += (f"With the score level throughout, the changes were not made to protect "
                 f"or chase anything, so a shift in output after one is a shift in "
                 f"personnel rather than in intent.")
    else:
        line += ("The score moved during the match, and a side ahead or behind changes "
                 "what it is trying to do independently of who is on the pitch. Read "
                 "the windows against the scoreline before crediting the change.")
    line += (f" Combined chance output was {hx + ax:.2f}, which over a handful of "
             f"three-minute windows is far too small a sample for any of this to be "
             f"more than a place to start looking.")
    return line


def _regain_speed(context, side, team):
    home, away = context["home"], context["away"]
    hr, ar = _pair(context, "high_regains")
    hs, as_ = _pair(context, "regain_to_shot_rate")
    ht, at = _pair(context, "transition_xG")
    fast = home if hs > as_ else away
    slow = away if fast == home else home
    line = (f"How long each side took to reach the final third after winning the ball. "
            f"{fast} turned {max(hs, as_):.0f}% of its high recoveries into shots "
            f"against {min(hs, as_):.0f}% for {slow}, off {hr:.0f} and {ar:.0f} "
            f"recoveries. ")
    if max(ht, at) >= 0.4:
        line += (f"The {max(ht, at):.2f} expected goals that came out of transition are "
                 f"the return on the quick end of this distribution; the slow end is "
                 f"possession that was won and then rebuilt, which is a different attack "
                 f"against a different defence.")
    else:
        line += ("Neither side got much out of the quick end, so the recoveries were "
                 "being won and then played out slowly - the opponent had time to get "
                 "back, and the chart's long tail is where that happened.")
    line += " Recoveries that never reached the final third are counted, not dropped."
    return line


_VISUAL_WRITERS = (
    ("possession_funnel", _possession_funnel),
    ("entry_routes", _entry_routes),
    ("game_state", _game_state),
    ("win_probability", _win_probability),
    ("xg_flow", _xg_flow),
    ("shot_map", _shot_map),
    ("goalkeeper", _goalkeeper),
    ("set_piece", _set_piece),
    ("xt_map", _xt_map),
    ("pass_network", _pass_network),
    ("pass_map", _pass_map),
    ("pass_targets", _pass_targets),
    ("defensive_activity", _defensive_activity),
    ("average_positions", _average_positions),
    ("pitch_control", _pitch_control),
    ("defensive_shape", _defensive_shape),
    ("playing_through", _playing_through),
    ("unlocking", _unlocking),
    ("press_triggers", _press_triggers),
    ("ppda", _ppda),
    ("high_regains", _high_regains),
    ("box_entries", _box_entries),
    ("zone14", _zone14),
    ("progressive", _progressive),
    ("ball_losses", _ball_losses),
    ("turnovers", _ball_losses),
    ("loss_consequences", _ball_losses),
    ("transition", _transition),
    ("dominating", _dominating),
    ("xt_per_minute", _xt_per_minute),
    ("player_sequence_leaders", _player_sequence_leaders),
    ("pass_sonar", _pass_sonar),
    ("finishing_quality", _finishing_quality),
    ("sequence_types", _sequence_types),
    ("match_momentum", _match_momentum),
    ("momentum", _match_momentum),
    ("action_value", _action_value),
    ("goal_origins", _goal_origins),
    ("goals_breakdown", _goal_origins),
    ("substitution_windows", _substitution_windows),
    ("regain_speed", _regain_speed),
)


def visual_reading(path, context):
    """What one board says about this match, or "" when nothing here fits.

    Returns the argument, not the numbers. The caller keeps its own figure
    recital for the boards this does not cover.
    """
    from pathlib import Path

    stem = Path(path).stem.lower()
    side, team = _side_named(stem, context)
    for token, writer in _VISUAL_WRITERS:
        if token in stem:
            try:
                text = (writer(context, side, team) or "").strip()
            except Exception:
                return ""
            if not text:
                return ""
            half = ("first half" if stem.endswith("1h")
                    else "second half" if stem.endswith("2h") else "")
            if half:
                # Every figure in the context is a full-match total. Printed
                # under a half board with no qualifier it reads as that half's
                # own number, which it is not.
                text = (f"This board is the {half} alone; the figures quoted are "
                        f"full-match totals, so read the picture for what changed "
                        f"and the numbers for what the match came to. ") + text
            return text
    return ""
