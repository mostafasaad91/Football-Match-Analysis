"""The match article as a tactical argument, one team phase at a time.

The previous article was assembled around the figures: a section per board,
headed by the board's own statistic ("Liverpool created 1.41 expected goals to
0.90, Liverpool led the box entries 23 to 8"), a paragraph that read the board,
and a hand-off to the next one. It described charts. A reader wanted the match:
how each side tried to play with the ball, how it defended without it, where
the game turned and why the result went the way it did.

This module writes that article from the same facts. The sections follow the
phases of play for each side, the figures sit inside them as evidence, and the
sentences are chosen by what the numbers say rather than printed beside them.
Numbers are used where they settle a point and left out where they only repeat
one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_analysis.metrics.frame_values import surname

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------


def _n(value, digits=0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "0"
    return f"{value:.{digits}f}"


def _pct(value):
    return f"{float(value or 0):.0f}%"


def _share(part, whole):
    return 100.0 * float(part or 0) / float(whole) if whole else 0.0


def _words(n):
    small = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve"]
    n = int(n)
    return small[n] if 0 <= n < len(small) else str(n)


def _plural(n, word, plural=None):
    return word if int(n) == 1 else (plural or word + "s")


def _listed(items):
    items = [i for i in items if i]
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _minute(m):
    return f"{int(m)}'"


def _flag(frame, column):
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].astype(str).str.strip().str.lower().isin(("true", "1"))


# ----------------------------------------------------------------------------
# What the article needs that the facts do not already hold
# ----------------------------------------------------------------------------


class _Extras:
    """Per-match readings taken straight from the event stream."""

    def __init__(self, facts, events: pd.DataFrame):
        self.events = events
        self.thirds = {}
        for side in facts.sides:
            own = events[(events["team_id"] == side.team_id) & _flag(events, "is_pass")]
            own = own.dropna(subset=["x", "end_x"])
            done = own["outcome"].astype(str).str.lower().eq("successful")
            rows = {}
            for name, lo, hi in (("own", 0, 33.34), ("middle", 33.34, 66.67), ("final", 66.67, 101)):
                part = own[(own["x"] >= lo) & (own["x"] < hi)]
                ok = done[part.index]
                rows[name] = {
                    "passes": len(part),
                    "completion": _share(ok.sum(), len(part)),
                    "forward": _share((part["end_x"] > part["x"] + 5).sum(), len(part)),
                    "onward": int((part.loc[ok, "end_x"] >= min(hi, 100)).sum()) if name != "final" else 0,
                    "long": int((pd.to_numeric(part.get("pass_length"), errors="coerce") >= 30).sum()),
                }
            self.thirds[side.key] = rows
        self.goals = self._goals(facts)
        self.missed_penalties = self._missed_penalties(facts)
        self.woodwork = self._woodwork(facts)

    def _goals(self, facts):
        from football_analysis.reports.match_report import classify_goal_type

        try:
            from football_analysis.metrics.match_metrics import goal_origin_chains

            chains = goal_origin_chains(self.events, facts.home.team_id, facts.away.team_id)
        except Exception:
            chains = pd.DataFrame()
        events = self.events
        goal_rows = events[_flag(events, "is_goal")].sort_values(["minute", "second"], kind="stable")
        errors = events[events["type"].astype(str).eq("Error")]
        q = errors.get("qualifier_names", pd.Series("", index=errors.index)).astype(str)
        out = []
        for _, row in goal_rows.iterrows():
            own_goal = str(row.get("is_own_goal")).lower() == "true"
            credited = row.get("scoring_team", row.get("team_id"))
            side = facts.home if int(float(credited)) == facts.home.team_id else facts.away
            category, subtype = classify_goal_type(row, events)
            near = errors[(abs(errors["minute"] - row["minute"]) <= 1) & q.str.contains("LeadingToGoal")
                          & (errors["team_id"] != side.team_id)]
            chain = None
            if not chains.empty:
                hit = chains[(chains["minute"] == row["minute"]) & (chains["scorer"] == row["player"])]
                chain = hit.iloc[0].to_dict() if len(hit) else None
            assist = str(row.get("assist_player") or "")
            out.append({
                "minute": int(row["minute"]),
                "scorer": str(row.get("player")),
                "side": side,
                "own_goal": own_goal,
                "category": category,
                "subtype": subtype,
                "xg": float(pd.to_numeric(row.get("xG"), errors="coerce") or 0),
                "assist": "" if assist.lower() in ("", "nan", "none") else assist,
                "header": str(row.get("is_header")).lower() == "true",
                "penalty": str(row.get("is_penalty")).lower() == "true",
                "errors": [str(p) for p in near["player"].astype(str).unique()],
                "chain": chain,
            })
        return out

    def _missed_penalties(self, facts):
        e = self.events
        pens = e[_flag(e, "is_penalty") & _flag(e, "is_shot") & ~_flag(e, "is_goal")]
        return [{"minute": int(r["minute"]), "player": str(r["player"]),
                 "side": facts.home if r["team_id"] == facts.home.team_id else facts.away,
                 "post": str(r["type"]) == "ShotOnPost"} for _, r in pens.iterrows()]

    def _woodwork(self, facts):
        e = self.events
        hit = e[e["type"].astype(str).eq("ShotOnPost") & ~_flag(e, "is_penalty")]
        return [{"minute": int(r["minute"]), "player": str(r["player"]),
                 "side": facts.home if r["team_id"] == facts.home.team_id else facts.away}
                for _, r in hit.iterrows()]


# ----------------------------------------------------------------------------
# Headline and standfirst
# ----------------------------------------------------------------------------


def headline(facts, x: _Extras) -> tuple[str, str]:
    """A headline about the whole match, both sides in it, and a standfirst."""
    home, away = facts.home, facts.away
    win = facts.winner()
    reds = [c for c in facts.cards if c.get("colour") == "Red"]
    if win is None:
        better = max(facts.sides, key=lambda s: s.xg)
        other = facts.other(better)
        if better.xg - other.xg >= 0.6:
            title = f"{better.name} Do Enough to Win but {other.name} Hold On for a Draw"
        else:
            title = f"{home.name} and {away.name} Cancel Each Other Out in a Tight Contest"
    else:
        lose = facts.other(win)
        dominated = lose.field_tilt >= 58 and lose.xg >= win.xg
        if dominated and lose.on_target == 0:
            title = f"{win.name} Hold Firm as {lose.name} Dominate the Ball but Never Test the Keeper"
        elif dominated and win.gk_prevented >= 0.8:
            keeper = surname(win.gk_name) or "the goalkeeper"
            title = f"{keeper} Defies {lose.name} as {win.name} Win Against the Run of Play"
        elif dominated:
            title = f"{win.name} Make Their Moments Count While {lose.name} Have the Ball and Little Else"
        elif win.xg >= lose.xg * 1.8 and win.field_tilt >= 55:
            title = f"{win.name} Control {lose.name} From Front to Back"
        else:
            title = f"{win.name} Edge {lose.name} in a Game Settled by the Details"
        if reds:
            red = reds[0]
            sent = surname(red["player"])
            if red.get("side") == lose.key:
                title = f"{win.name} Beat {lose.name} After {sent}'s Red Card Swings the Game"
            else:
                title = f"{win.name} Beat {lose.name} Despite {sent}'s Red Card"
    info_title = str(facts.info.get("headline") or "").strip()
    title = info_title or title

    # Standfirst: the result, how it was won, and the shape of the game.
    first = _decisive_goal(facts, x)
    sentences = []
    if win is None:
        sentences.append(f"{home.name} and {away.name} drew {facts.score.replace(':', '–')}.")
    else:
        sentences.append(f"{win.name} beat {facts.other(win).name} {max(home.goals, away.goals)}–"
                         f"{min(home.goals, away.goals)}.")
    if first is not None:
        how = _goal_mechanism(first)
        label = "The equaliser came on" if win is None else "The decisive moment came on"
        sentences.append(f"{label} {_minute(first['minute'])}: {how}.")
    top = max(facts.sides, key=lambda s: s.field_tilt)
    if win is top:
        tail = "and turned that territory into the better chances."
    elif win is None:
        tail = "but could not turn that territory into a win."
    else:
        tail = "but could not turn that territory into goals."
    sentences.append(
        f"{top.name} had {_pct(top.possession_share)} of the ball and {_pct(top.field_tilt)} of the "
        f"final-third passing, {tail}"
    )
    override = str(facts.info.get("standfirst") or "").strip()
    return title, override or " ".join(sentences)


def _decisive_goal(facts, x: _Extras):
    """The goal that settled it: the winner's last go-ahead goal, or a draw's last equaliser."""
    if not x.goals:
        return None
    score = {facts.home.key: 0, facts.away.key: 0}
    win = facts.winner()
    decisive = None
    for goal in x.goals:
        key = goal["side"].key
        before = score[key] - score["away" if key == "home" else "home"]
        score[key] += 1
        if win is not None and goal["side"] is win and before == 0:
            decisive = goal
        if win is None and before == -1:
            decisive = goal
    return decisive or x.goals[-1]


def _goal_mechanism(goal) -> str:
    """One clause describing how a goal was made."""
    scorer = surname(goal["scorer"])
    side = goal["side"].name
    chain = goal.get("chain") or {}
    route = str(chain.get("sequence_type") or "")
    if goal["own_goal"]:
        return f"{scorer} turned the ball into his own net to put {side} ahead"
    if goal["penalty"]:
        return f"{scorer} scored from the spot"
    if goal["category"] == "Set Piece":
        delivery = goal["subtype"].lower()
        if delivery == "direct free kick":
            return f"{scorer} scored directly from a free kick"
        return f"{scorer} finished from a {delivery}" + (f" delivered by {surname(goal['assist'])}" if goal["assist"] else "")
    if goal["errors"]:
        culprits = _listed([surname(p) for p in goal["errors"]])
        pressed = str(chain.get("started_from") or "") == "recovery" and float(chain.get("start_x") or 0) >= 60
        if pressed:
            return (f"{side} pressed {culprits} into a mistake, won the ball high and {scorer} finished "
                    f"the move" + (f" from {surname(goal['assist'])}'s pass" if goal["assist"] else ""))
        return (f"a mistake by {culprits} let {side} in, and {scorer} finished"
                + (f" from {surname(goal['assist'])}'s pass" if goal["assist"] else ""))
    if route == "counter":
        return f"{side} broke quickly after winning the ball and {scorer} finished" + (
            f" from {surname(goal['assist'])}'s pass" if goal["assist"] else "")
    if route in ("build_up", "sustained") and float(chain.get("duration") or 0) >= 15:
        passes = int(chain.get("passes") or 0)
        article = "an" if str(passes)[:1] == "8" or passes in (11, 18) else "a"
        return f"{scorer} finished {article} {passes}-pass move that {side} built patiently"
    if goal["xg"] < 0.08:
        return f"{scorer} scored from distance, a chance the model valued at only {goal['xg']:.2f}"
    return f"{scorer} scored for {side}" + (f" from {surname(goal['assist'])}'s pass" if goal["assist"] else "")


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------


def _a(name: str) -> str:
    """'an Arsenal shot', 'a Liverpool shot'."""
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name


def _keeper(side) -> str:
    return side.gk_name or f"{side.name}'s goalkeeper"


def _lead_context(facts, x: _Extras, side) -> str:
    """Whether this side spent the match ahead, behind or level."""
    states = side.states or {}
    lead = float((states.get("leading") or {}).get("minutes") or 0)
    trail = float((states.get("trailing") or {}).get("minutes") or 0)
    if lead >= 45:
        return "leading"
    if trail >= 45:
        return "trailing"
    return "level"


def _story(facts, x: _Extras) -> list[str]:
    home, away = facts.home, facts.away
    out = []
    if not x.goals:
        out.append(
            f"There were no goals, so the match has to be read through what each side built. "
            f"{home.name} finished with {_n(home.xg, 2)} expected goals from {home.shots} shots, "
            f"{away.name} with {_n(away.xg, 2)} from {away.shots}."
        )
    elif len(x.goals) == 1:
        g = x.goals[0]
        out.append(f"The only goal came on {_minute(g['minute'])}: {_goal_mechanism(g)}.")
    else:
        parts = [f"on {_minute(g['minute'])} {_goal_mechanism(g)}" for g in x.goals]
        out.append("The goals came in this order: " + "; ".join(parts) + ".")
    win = facts.winner()
    if win is None:
        turning = []
        reds = [c for c in facts.cards if c.get("colour") == "Red"]
        for c in reds:
            how = "a second yellow card" if c.get("second_yellow") else "a straight red card"
            turning.append((c["minute"], f"{c['player']} was sent off for {how} on {_minute(c['minute'])}"))
        for p in x.missed_penalties:
            turning.append((p["minute"], f"{p['player']} {'hit the post from' if p['post'] else 'missed'} a penalty on {_minute(p['minute'])}"))
        for w in x.woodwork:
            turning.append((w["minute"], f"{surname(w['player'])} hit the woodwork on {_minute(w['minute'])}"))
        if turning:
            out.append("Around the goals: " + _listed([t for _, t in sorted(turning)]) + ".")
        if reds:
            red = reds[0]
            minute = int(red["minute"])
            end = max([90] + [int(sh.get("minute") or 0) for sh in facts.shots])
            if end - minute >= 10:
                rates = []
                for side in facts.sides:
                    own = [sh for sh in facts.shots if sh["side"] == side.key]
                    before = 30 * sum(sh["xG"] or 0 for sh in own if int(sh["minute"]) < minute) / max(minute, 1)
                    after = 30 * sum(sh["xG"] or 0 for sh in own if int(sh["minute"]) >= minute) / max(end - minute, 1)
                    rates.append((side, before, after))
                down = next(s_ for s_ in facts.sides if s_.key == red.get("side"))
                full = facts.other(down)
                r_full = next(r for r in rates if r[0] is full)
                r_down = next(r for r in rates if r[0] is down)
                out.append(
                    f"The red card changed the game. Before it, {full.name} were producing {r_full[1]:.2f} expected "
                    f"goals per half-hour and {down.name} {r_down[1]:.2f}; afterwards it was {r_full[2]:.2f} against "
                    f"{r_down[2]:.2f}. With ten men, {down.name} could no longer build pressure, and {full.name} had "
                    f"the space a missing player leaves."
                )
        if len(x.goals) >= 2:
            leads = []
            score = {facts.home.key: 0, facts.away.key: 0}
            for goal in x.goals:
                score[goal["side"].key] += 1
                if score[facts.home.key] != score[facts.away.key]:
                    leads.append(goal["side"].name)
            if leads:
                out.append(
                    f"{_listed(sorted(set(leads)))} led at different points, and each time the lead was taken "
                    f"away again. A draw that has to be read as two halves of an argument rather than a stalemate."
                    if len(set(leads)) > 1 else
                    f"{leads[0]} led {len(leads)} {_plural(len(leads), 'time')} and could not hold on."
                )
        better = max(facts.sides, key=lambda s_: s_.xg)
        other = facts.other(better)
        if better.xg - other.xg >= 0.6:
            out.append(f"On chances {better.name} deserved more: {_n(better.xg, 2)} expected goals to {_n(other.xg, 2)}.")
        return out
    lose = facts.other(win)
    first_minute = x.goals[0]["minute"] if x.goals else 90
    if first_minute <= 30:
        out.append(
            f"Because it came so early, the match became {lose.name} with the ball and {win.name} without "
            f"it for well over an hour. That frame matters for everything below: {lose.name}'s territory "
            f"was partly the territory a side is given when its opponent has something to protect."
        )
    elif first_minute >= 70:
        out.append("The game was level for most of its length, so the numbers describe a genuine contest "
                   "rather than one side sitting on a lead.")
    else:
        out.append(f"From then on {win.name} had something to protect, and the balance of the game shifted with it.")
    turning = []
    reds = [c for c in facts.cards if c.get("colour") == "Red"]
    for c in reds:
        how = "a second yellow card" if c.get("second_yellow") else "a straight red card"
        turning.append((c["minute"], f"{c['player']} was sent off for {how} on {_minute(c['minute'])}"))
    for p in x.missed_penalties:
        turning.append((p["minute"], f"{p['player']} {'hit the post from' if p['post'] else 'missed'} a penalty on {_minute(p['minute'])}"))
    for w in x.woodwork:
        turning.append((w["minute"], f"{surname(w['player'])} hit the woodwork on {_minute(w['minute'])}"))
    if turning:
        out.append("Around the goals: " + _listed([t for _, t in sorted(turning)]) + ".")
    if reds:
        red = reds[0]
        minute = int(red["minute"])
        end = max([90] + [int(sh.get("minute") or 0) for sh in facts.shots])
        if end - minute >= 10:
            rates = []
            for side in facts.sides:
                own = [sh for sh in facts.shots if sh["side"] == side.key]
                before = 30 * sum(sh["xG"] or 0 for sh in own if int(sh["minute"]) < minute) / max(minute, 1)
                after = 30 * sum(sh["xG"] or 0 for sh in own if int(sh["minute"]) >= minute) / max(end - minute, 1)
                rates.append((side, before, after))
            down = next(s_ for s_ in facts.sides if s_.key == red.get("side"))
            full = facts.other(down)
            r_full = next(r for r in rates if r[0] is full)
            r_down = next(r for r in rates if r[0] is down)
            numbers = (f"Before it, {full.name} were producing {r_full[1]:.2f} expected goals per half-hour and "
                       f"{down.name} {r_down[1]:.2f}; afterwards it was {r_full[2]:.2f} against {r_down[2]:.2f}.")
            if r_down[2] <= 0.6 * r_down[1]:
                verdict = (f" With ten men, {down.name} could no longer build pressure, and {full.name} had the "
                           f"space a missing player leaves.")
                lead = "The red card changed the game."
            elif r_full[2] >= 1.5 * max(r_full[1], 0.05):
                verdict = f" {full.name} used the extra man to attack the space it opened."
                lead = "The red card opened the game up."
            else:
                verdict = (f" {down.name} were chasing by then and kept coming, so the extra man changed less "
                           f"than it might have.")
                lead = "The red card changed less than expected."
            out.append(f"{lead} {numbers}{verdict}")

    if lose.xg > win.xg:
        if lose.on_target == 0:
            out.append(
                f"{lose.name} ended with more expected goals, {_n(lose.xg, 2)} to {_n(win.xg, 2)}, and not a "
                f"single one of their {lose.shots} shots on target. That is not bad luck; it is a defence "
                f"getting bodies between the ball and the goal every time."
            )
        else:
            out.append(
                f"{lose.name} created more, {_n(lose.xg, 2)} expected goals to {_n(win.xg, 2)}, but {win.name} "
                f"took their moment and {lose.name} did not take theirs."
            )
    else:
        out.append(f"The result matched the chances: {win.name} created {_n(win.xg, 2)} expected goals to "
                   f"{_n(lose.xg, 2)}.")
    return out


def _vary(key: str, options: list[str]) -> str:
    """One of several phrasings, fixed per side and sentence.

    The two team sections are written by the same code, so without this the
    second side's paragraphs read as a copy of the first's with the names
    swapped: "a clear plan to overload one flank" twice on one page.
    """
    import zlib

    return options[zlib.crc32(key.encode("utf-8")) % len(options)]


def _with_ball(facts, x: _Extras, side) -> list[str]:
    other = facts.other(side)
    t = x.thirds[side.key]
    context = _lead_context(facts, x, side)
    lanes = {"left": side.lane_left, "centre": side.lane_centre, "right": side.lane_right}
    total_lanes = sum(lanes.values())
    k = side.name
    out = []

    # 1. Shape and intent
    shape = f" in a {side.formation}" if side.formation else ""
    if side.possession_share >= 58:
        intent = _vary(k + "poss", [
            f"{side.name} lined up{shape} and had the ball for long spells, {_pct(side.possession_share)} of it in all.",
            f"Set up{shape}, {side.name} controlled the ball: {_pct(side.possession_share)} of possession.",
            f"{side.name} played{shape} and kept the ball for most of the afternoon, {_pct(side.possession_share)} of possession.",
        ])
        if context == "trailing":
            intent += " Some of that was their own plan and some of it was the scoreline handing it to them."
    elif side.possession_share <= 42:
        intent = _vary(k + "poss", [
            f"{side.name} lined up{shape} and were comfortable without the ball, with {_pct(side.possession_share)} of possession.",
            f"Set up{shape}, {side.name} were happy to let the other side have it: {_pct(side.possession_share)} of possession.",
        ])
        if context == "leading":
            intent += " With a lead to protect, there was no reason to take risks on the ball."
    else:
        intent = _vary(k + "poss", [
            f"{side.name} lined up{shape} and shared the ball roughly evenly, with {_pct(side.possession_share)} of possession.",
            f"Set up{shape}, {side.name} had {_pct(side.possession_share)} of the ball, neither dominating it nor giving it up.",
        ])
    if side.directness >= 55:
        intent += _vary(k + "dir", [
            " When they had it, they went forward quickly: more than half the distance their passes covered was towards goal.",
            " Their passing was vertical and quick, looking for the space behind before the opponent could settle.",
        ])
    elif side.directness <= 42:
        intent += _vary(k + "dir", [
            " They moved it patiently, more across the pitch than forward, waiting for an opening rather than forcing one.",
            " The circulation was slow and lateral, the ball worked side to side in search of a gap.",
            " Most of their passing went across the pitch rather than up it; they were trying to move the block before breaking it.",
        ])
    out.append(intent)

    # 2. Build-up under pressure
    if side.build_up_attempts:
        rate = side.build_up_success_rate
        own = t["own"]
        if rate <= 35:
            text = _vary(k + "build", [
                f"Playing out from the back was a problem. Only {_pct(rate)} of their attempts got the ball cleanly into midfield",
                f"The first phase was where they lost control: just {_pct(rate)} of their attempts to play out reached midfield cleanly",
            ])
            text += (f", and {other.name}'s pressure forced "
                     + ("long balls" if own["forward"] >= 55 else "them back and sideways")
                     + " more often than they would have liked.")
        elif rate >= 50:
            text = (f"Playing out from the back was not an issue: {_pct(rate)} of their attempts reached midfield "
                    f"cleanly, so {other.name}'s first line rarely forced a mistake.")
        else:
            text = _vary(k + "build", [
                f"Their build-up was solid without being smooth: {_pct(rate)} of their attempts to play out reached midfield cleanly.",
                f"They got out from the back often enough, {_pct(rate)} of the time cleanly into midfield, without ever making it look easy.",
            ])
        if own["onward"]:
            text += f" In all, {own['onward']} passes from their own third were completed straight into midfield."
        out.append(text)

    # 3. Progression and where they attacked
    mid = t["middle"]
    text = _vary(k + "mid", [
        f"In midfield they completed {_pct(mid['completion'])} of their passes and reached the final third {side.final_third_entries} times",
        f"Through the middle third their passing held up at {_pct(mid['completion'])}, and they got into the final third {side.final_third_entries} times",
    ])
    if total_lanes:
        top_lane = max(lanes, key=lanes.get)
        share = _share(lanes[top_lane], total_lanes)
        if share >= 45 and top_lane != "centre":
            text += _vary(k + "lane", [
                f", most of it down the {top_lane}: {_pct(share)} of those entries came on that side, a clear plan to overload one flank.",
                f". The {top_lane} side was the route: {_pct(share)} of those entries went that way.",
                f", leaning heavily on the {top_lane}, which took {_pct(share)} of the entries.",
            ])
        elif share >= 45:
            text += ", most of it straight through the middle."
        else:
            text += ", spread fairly evenly across the left, the centre and the right."
    else:
        text += "."
    out.append(text)

    # 4. The final third
    if side.box_entries:
        text = _vary(k + "box", [
            f"They got into the box {side.box_entries} times, and {_pct(side.box_entry_to_shot_rate)} of those entries ended in a shot",
            f"They reached the penalty area {side.box_entries} times, and {_pct(side.box_entry_to_shot_rate)} of those visits produced a shot",
        ])
        text += (" — a side that made its few visits count." if side.box_entries <= 10 and side.box_entry_to_shot_rate >= 70
                 else ".")
    else:
        text = "They never got into the box with the ball, which says everything about how the attack worked."
    if side.crosses >= 15:
        rate = _share(side.completed_crosses, side.crosses)
        if rate < 25:
            text += (f" Too much of the attack ended with a cross: {side.crosses} of them, only "
                     f"{side.completed_crosses} reaching a teammate. That is usually a sign the way through the "
                     f"middle has been closed and the ball keeps arriving at the touchline with nowhere else to go.")
        elif rate >= 35:
            # A completion rate is not a weapon on its own: Hull found a
            # teammate with 6 of 15 and finished with 0.35 xG in all. Judge
            # the crossing by what the attack produced.
            if side.xg >= 1.0:
                text += (f" Their crossing was a real weapon: {side.completed_crosses} of "
                         f"{side.crosses} found a teammate.")
            else:
                text += (f" Their crosses found a teammate often enough, {side.completed_crosses} "
                         f"of {side.crosses}, but the whole attack added up to "
                         f"{side.xg:.2f} expected goals.")
    out.append(text)

    # 5. Chance quality and finishing
    if side.shots:
        if side.xg_per_shot < 0.09 and side.blocked >= side.shots * 0.35:
            text = (f"The chances were the problem. {side.shots} shots at an average of {_n(side.xg_per_shot, 2)} "
                    f"expected goals each, {side.blocked} of them blocked: shooting into a crowded box rather than "
                    f"creating clear openings.")
        elif side.xg_per_shot >= 0.14:
            text = (f"When they shot it was from good positions: {_n(side.xg_per_shot, 2)} expected goals per shot "
                    f"across {side.shots} attempts, {_n(side.xg, 2)} in total.")
        else:
            text = _vary(k + "shots", [
                f"They finished with {side.shots} shots worth {_n(side.xg, 2)} expected goals.",
                f"It added up to {side.shots} shots and {_n(side.xg, 2)} expected goals.",
            ])
        if side.big_chances == 1:
            text += " Only one was a clear chance."
        elif side.big_chances > 1:
            text += f" {_words(side.big_chances).capitalize()} were clear chances."
        if side.on_target == 0:
            text += f" Not one of them was on target, so {_keeper(other)} did not have to make a save."
        elif side.goals == 0 and other.gk_prevented >= 0.6:
            text += (f" When they did hit the target, {_keeper(other)} was equal to everything: {side.on_target} on "
                     f"target and nothing to show for it.")
        elif side.goals >= side.xg + 0.8:
            text += f" The finishing made the difference: {side.goals} goals from chances worth {_n(side.xg, 2)}."
        out.append(text)
    else:
        out.append("They did not manage a single shot.")

    # 6. Half-by-half
    first, second = side.half.get("first") or {}, side.half.get("second") or {}
    if first and second:
        f_xg, s_xg = float(first.get("xG") or 0), float(second.get("xG") or 0)
        if s_xg >= f_xg + 0.4:
            out.append(f"They were more dangerous after the break, {_n(s_xg, 2)} expected goals against "
                       f"{_n(f_xg, 2)} in the first half.")
        elif f_xg >= s_xg + 0.4:
            out.append(f"Their threat faded after the break: {_n(f_xg, 2)} expected goals in the first half, "
                       f"{_n(s_xg, 2)} in the second.")
    return out


def _without_ball(facts, x: _Extras, side) -> list[str]:
    other = facts.other(side)
    context = _lead_context(facts, x, side)
    k = side.name
    out = []
    height = side.def_action_avg_x
    ppda = side.ppda or 0
    if ppda and ppda <= 8 and height >= 38:
        text = _vary(k + "press", [
            f"{side.name} pressed high and with intent: {other.name} managed only {_n(ppda, 1)} passes in their own half for every defensive action, and the average defensive action came {height:.0f} metres up the pitch.",
            f"The press was the identity of {side.name}'s game without the ball. {other.name} got {_n(ppda, 1)} passes in their own half for every defensive action, and {side.name}'s defending happened on average {height:.0f} metres up the pitch.",
        ])
    elif ppda and ppda <= 8:
        block = "a mid-block" if height >= 33 else "a deep block"
        text = _vary(k + "press", [
            f"{side.name} were aggressive whenever {other.name} played in their own half — {_n(ppda, 1)} passes per defensive action — but the rest of the time they settled into {block}, with their defensive actions averaging {height:.0f} metres up the pitch.",
            f"Without the ball {side.name} mixed two plans: short, sharp pressing when {other.name} played out ({_n(ppda, 1)} passes per defensive action), then a drop into {block}, their defending averaging {height:.0f} metres from their own goal line.",
        ])
    elif ppda >= 13:
        text = (f"{side.name} did not press. They let {other.name} have the ball in their own half "
                f"({_n(ppda, 1)} passes per defensive action) and defended from {height:.0f} metres up the pitch "
                f"on average.")
    else:
        text = (f"{side.name} pressed selectively — {_n(ppda, 1)} passes per defensive action — and defended "
                f"{'in a mid-block' if height >= 35 else 'deep'}, {height:.0f} metres up the pitch on average.")
    if context == "leading" and height < 35:
        text += " With the lead, that was the sensible way to play the game."
    out.append(text)

    if side.high_regains:
        text = _vary(k + "regain", [
            f"They won the ball back {side.high_regains} times in {other.name}'s half",
            f"They took the ball off {other.name} {side.high_regains} times in the opposition half",
        ])
        if side.transition_shots:
            text += (f", and their attacks after winning it produced {side.transition_shots} "
                     f"{_plural(side.transition_shots, 'shot')} worth {_n(side.transition_xg, 2)} expected goals.")
            if side.transition_xg >= 0.5:
                text += " Winning the ball and going straight at goal was one of their main routes to a chance."
        else:
            text += ", but none of those regains turned into a shot."
        out.append(text)

    text = _vary(k + "boxdef", [
        f"Closer to their own goal, {other.name} got into the box {other.box_entries} times and shot {other.shots} times.",
        f"In their own third, they faced {other.box_entries} box entries and {other.shots} shots.",
    ])
    if other.shots and other.xg_per_shot < 0.09 and other.blocked >= 0.3 * other.shots:
        text += (f" Most of those attempts came from poor positions and {other.blocked} were blocked: the box was "
                 f"defended with numbers, and {other.name} were made to shoot from where {side.name} wanted them to.")
    elif other.shots and other.xg_per_shot < 0.09:
        text += f" {other.name} rarely got a clean look: their shots averaged {_n(other.xg_per_shot, 2)} expected goals."

    elif other.big_chances >= 2:
        text += f" They still gave away {_words(other.big_chances)} clear chances, which is where the defending broke down."
    out.append(text)

    if side.losses_to_shot:
        out.append(_vary(k + "loss", [
            f"The risk of losing the ball was real: {side.losses_to_shot} of their {side.losses} open-play losses led to {_a(other.name)} shot within twelve seconds, worth {_n(side.loss_xg_conceded, 2)} expected goals.",
            f"Losing the ball cost them {side.losses_to_shot} times: that many of their {side.losses} open-play losses became {_a(other.name)} shot inside twelve seconds, {_n(side.loss_xg_conceded, 2)} expected goals in all.",
        ]))
    for goal in x.goals:
        if goal["errors"] and goal["side"] is other:
            out.append(
                f"The goal they conceded came from their own mistake: {_listed([surname(p) for p in goal['errors']])} "
                f"lost the ball on {_minute(goal['minute'])} and {other.name} needed only seconds to punish it."
            )
            break
    return out


def _changes(facts) -> list[str]:
    out = []
    bench = facts.bench
    if bench is not None and len(bench):
        for side in facts.sides:
            mine = bench[bench["team_id"].astype(float).astype(int) == side.team_id].sort_values("minute")
            if mine.empty:
                continue
            first = mine.iloc[0]
            decisive = mine[(mine["goals"] > 0) | (mine["assists"] > 0)]
            names = _listed([surname(r["player"]) for _, r in mine.head(3).iterrows()])
            text = (f"{side.name} made {len(mine)} {_plural(len(mine), 'change')}, the first on "
                    f"{_minute(float(first['minute']))}, bringing on {names}"
                    + (" among others" if len(mine) > 3 else "") + ".")
            if len(decisive):
                text += (f" The bench changed the game: {_listed([surname(r['player']) for _, r in decisive.iterrows()])} "
                         f"had a direct hand in {int(decisive['goals'].sum() + decisive['assists'].sum())} "
                         f"goal {_plural(int(decisive['goals'].sum() + decisive['assists'].sum()), 'involvement')}.")
            elif float(mine["xG"].sum() + mine["xA"].sum()) >= 0.3:
                text += (f" The substitutes added threat without a goal: "
                         f"{_n(mine['xG'].sum() + mine['xA'].sum(), 2)} expected goals and assists between them.")
            else:
                text += " The changes did not alter the pattern of the game."
            out.append(text)
    for side in facts.sides:
        trailing = side.states.get("trailing", {})
        minutes = float(trailing.get("minutes") or 0)
        if minutes >= 20:
            rate = 30 * float(trailing.get("xG") or 0) / minutes
            out.append(
                f"Chasing the game for {minutes:.0f} minutes, {side.name} produced {rate:.2f} expected goals every "
                f"half-hour" + (", a sustained push that deserved more." if rate >= 0.45 else
                                ", which was never enough to force an equaliser.")
            )
    return out


_ROLE_WORDS = {
    "GK": "goalkeeper", "DC": "centre-back", "DL": "left-back", "DR": "right-back",
    "DMC": "holding midfielder", "MC": "central midfielder", "ML": "left midfielder",
    "MR": "right midfielder", "AMC": "attacking midfielder", "AML": "left winger",
    "AMR": "right winger", "FW": "centre-forward", "FWL": "forward", "FWR": "forward",
    "DMR": "midfielder", "DML": "midfielder", "Sub": "substitute",
}


def _player_paragraph(side, p, goals_for: list) -> str:
    """Two or three sentences on one player: his job, and how he did it."""
    name = p["name"]
    short = surname(name)
    role = _ROLE_WORDS.get(str(p.get("role") or "").upper(), "")
    minutes = float(p.get("minutes") or 0)
    at = f" as a {role}" if role and role != "substitute" else ""
    scored = [g for g in goals_for if g["scorer"] == name]
    assisted = [g for g in goals_for if g["assist"] == name]
    sentences = []
    if scored:
        g = scored[0]
        sentences.append(f"{name} scored on {_minute(g['minute'])}" + (" and again later" if len(scored) > 1 else "") + ".")
    elif assisted:
        sentences.append(f"{name} made the goal for {surname(assisted[0]['scorer'])} on {_minute(assisted[0]['minute'])}.")
    elif minutes >= 85:
        sentences.append(f"{name} played the full match{at}.")
    else:
        sentences.append(f"{name} played {minutes:.0f} minutes{at}.")
    detail = []
    if p.get("progressive_passes", 0) >= 6:
        detail.append(f"moved the ball forward with {p['progressive_passes']} progressive passes")
    if p.get("progressive_carries", 0) >= 3:
        detail.append(f"carried it {p['progressive_carries']} times into more dangerous areas")
    if p.get("xA", 0) >= 0.25:
        detail.append(f"set up chances worth {p['xA']:.2f} expected goals")
    if p.get("shots", 0) >= 3 and not scored:
        detail.append(f"took {p['shots']} shots worth {p.get('xG', 0):.2f} expected goals without scoring")
    if p.get("takeons_won", 0) >= 3:
        detail.append(f"beat his man {p['takeons_won']} times")
    defending = p.get("recoveries", 0) + p.get("interceptions", 0) + p.get("tackles_won", 0)
    if defending >= 8:
        detail.append(f"won the ball back {defending} times")
    if p.get("clearances", 0) >= 5:
        detail.append(f"made {p['clearances']} clearances")
    if not detail:
        if p.get("key_passes", 0) >= 2:
            detail.append(f"created {p['key_passes']} shooting chances for teammates")
        if p.get("touches", 0) >= 40:
            detail.append(f"was on the ball {p['touches']} times and completed {p.get('pass_pct', 0):.0f}% of his passes")
        if p.get("box_entries", 0) >= 2:
            detail.append(f"got the ball into the box {p['box_entries']} times")
    if detail:
        line = f"{short} " + _listed(detail[:3]) + "."
        sentences.append(line[0].upper() + line[1:])
    chain = float(p.get("xGChain") or 0)
    own = float(p.get("xG") or 0) + float(p.get("xA") or 0)
    if chain >= 0.6 and chain >= 2 * max(own, 0.05):
        sentences.append(
            f"His real value was earlier in the move: the possessions he touched carried {chain:.2f} expected "
            f"goals, far more than his own shooting and passing account for."
        )
    elif p.get("pass_pct", 0) >= 88 and p.get("passes", 0) >= 50:
        sentences.append(f"He was the side's metronome, completing {p['pass_pct']:.0f}% of {p['passes']} passes.")
    return " ".join(sentences)


def _players(facts, x: _Extras) -> list[str]:
    out = []
    for side in facts.sides:
        goals_for = [g for g in x.goals if g["side"] is side and not g["own_goal"]]
        ranked = sorted(side.players, key=lambda p: p.get("score", 0), reverse=True)
        must = [p for p in side.players if any(g["scorer"] == p["name"] or g["assist"] == p["name"] for g in goals_for)]
        chosen = []
        for p in must + ranked:
            if p not in chosen and str(p.get("role") or "").upper() != "GK":
                chosen.append(p)
            if len(chosen) == 3:
                break
        for i, p in enumerate(chosen):
            text = _player_paragraph(side, p, goals_for)
            out.append(f"For {side.name}, {text}" if i == 0 else text)
    for side in facts.sides:
        if side.gk_prevented >= 0.6 and side.gk_name:
            out.append(
                f"{side.gk_name} deserves his own line: he faced shots worth {side.gk_psxg_faced:.2f} post-shot "
                f"expected goals and conceded {side.gk_conceded}, about {side.gk_prevented:.1f} goals prevented."
            )
    return out


def _verdict(facts, x: _Extras) -> list[str]:
    win = facts.winner()
    if win is None:
        return ["Neither side found the decisive idea, and the draw reflected how evenly the main battles were split."]
    lose = facts.other(win)
    out = []
    if lose.field_tilt > win.field_tilt and lose.xg >= win.xg:
        out.append(
            f"{win.name} won without the ball. They defended their box with numbers, took the opening their "
            f"press created, and left {lose.name} trying to beat a compact block from wide areas."
        )
        out.append(
            f"For {lose.name} the problem was not getting the ball forward; it was what happened once it arrived. "
            f"The territory and the entries were there, the clear chances were not, and the shots that came "
            + ("never tested the goalkeeper." if lose.on_target == 0 else "were dealt with.")
        )
    else:
        out.append(
            f"{win.name} were the better side and the result reflects it: more of the ball where it mattered, "
            f"better chances and the composure to take them."
        )
        out.append(
            f"{lose.name} lost the game in "
            + ("their build-up, where they could not get out under pressure."
               if lose.build_up_success_rate < 40 else "the final third, where they could not create clear chances.")
        )
    return out


# ----------------------------------------------------------------------------
# Measures added in the November 2026 review
# ----------------------------------------------------------------------------


def _extras(facts, events) -> dict:
    from football_analysis.metrics.match_extras import match_extras

    info = {"home_id": facts.home.team_id, "away_id": facts.away.team_id}
    try:
        return match_extras(events, info)
    except Exception:
        return {}


def _deserved(facts, extras: dict) -> list[str]:
    """Expected points, control, and where the chances came from."""
    if not extras:
        return []
    home, away = facts.home, facts.away
    xp, control = extras["expected_points"], extras["game_control"]
    out = [
        f"Replay every shot at its expected-goals value and {home.name} win "
        f"{xp['home_win']:.0%} of the time, {away.name} {xp['away_win']:.0%}, with "
        f"{xp['draw']:.0%} draws: {xp['home_xpts']:.1f} expected points to {xp['away_xpts']:.1f}. "
        f"{home.name} were the likelier winner for {control['home']:.0f}% of the match, "
        f"{away.name} for {control['away']:.0f}%."
    ]
    lines = []
    for side, key in ((home, "home"), (away, "away")):
        sit = extras[key]["xg_by_situation"]
        total = sum(v["xG"] for v in sit.values()) or 0.0
        dead = sum(sit[s]["xG"] for s in ("Corner", "Free kick", "Throw-in"))
        if total >= 0.3 and dead / total >= 0.4:
            lines.append(f"{dead:.2f} of {side.name}'s {total:.2f} expected goals came from "
                         f"dead balls, so the set pieces were a real part of their threat")
        counter = sit["Counter"]
        if counter["shots"] >= 2:
            lines.append(f"{side.name} had {counter['shots']} shots on the counter worth {counter['xG']:.2f}")
    if lines:
        out.append("; ".join(lines) + ".")
    sp = [(side, extras[key]["set_pieces"]) for side, key in ((home, "home"), (away, "away"))]
    out.append(" ".join(
        f"{side.name} took {s['corners']} corners and {s['corners_to_shot']} of them brought a shot within ten seconds."
        for side, s in sp))
    return out


def _pressing_over_time(facts, extras: dict) -> list[str]:
    if not extras:
        return []
    out = []
    for side, key in ((facts.home, "home"), (facts.away, "away")):
        rows = [r for r in extras[key]["ppda_timeline"] if r["ppda"] is not None]
        if len(rows) < 3:
            continue
        first = sum(r["passes"] for r in rows if r["start"] < 45) / max(1, sum(r["actions"] for r in rows if r["start"] < 45))
        second = sum(r["passes"] for r in rows if r["start"] >= 45) / max(1, sum(r["actions"] for r in rows if r["start"] >= 45))
        hardest = min(rows, key=lambda r: r["ppda"])
        trend = ("pressed harder after the break" if second < first * 0.85 else
                 "stopped pressing after the break" if second > first * 1.15 else
                 "pressed at much the same intensity in both halves")
        out.append(f"{side.name} {trend} (PPDA {first:.1f} before, {second:.1f} after), "
                   f"and were at their most aggressive between {hardest['start']}' and {hardest['end']}' "
                   f"({hardest['ppda']:.1f}).")
    return out


def _vaep_lines(events, facts) -> list[str]:
    from football_analysis.metrics.vaep import load_model, player_vaep

    if load_model() is None:
        return []
    table = player_vaep(events)
    if table.empty:
        return []
    top = table.head(3)
    names = [f"{_surname_of(r.player)} ({r.vaep:+.2f})" for r in top.itertuples()]
    defender = table.sort_values("defensive", ascending=False).iloc[0]
    out = [f"On VAEP, which values every action by how much it changed the chance of scoring and of "
           f"conceding, the biggest contributions came from {', '.join(names[:-1])} and {names[-1]}."]
    if defender.player not in set(top.player) and defender.defensive >= 0.15:
        out.append(f"Without the ball, {_surname_of(defender.player)} did the most to lower the opponent's "
                   f"chances ({defender.defensive:+.2f}).")
    return out


def _surname_of(name: str) -> str:
    return surname(str(name))


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------


def _visual(out: Path, token: str, slug: str | None = None) -> list[Path]:
    for path in sorted(out.glob("[0-9]*.png")):
        stem = path.stem
        if token in stem and (slug is None or stem.endswith(slug)):
            return [path]
    return []


def build_tactical_article(out_dir, match_info, context=None):
    """The article, or None when the package lacks what it needs."""
    from football_analysis.metrics.match_facts import load_match
    from football_analysis.prose.match_article import Article, Section, _cover_image
    from football_analysis.prose.prose_hygiene import clean

    out = Path(out_dir)
    facts = load_match(out)
    events = pd.read_csv(out / "events.csv", encoding="utf-8-sig")
    x = _Extras(facts, events)
    title, standfirst = headline(facts, x)
    home, away = facts.home, facts.away

    extras = _extras(facts, events)
    sections = [
        Section("How the match was decided", _story(facts, x),
                _visual(out, "xg_flow") + _visual(out, "goal_origins")),
    ]
    deserved = _deserved(facts, extras)
    if deserved:
        sections.append(Section("What the chances were worth", deserved,
                                _visual(out, "control_and_points") + _visual(out, "set_pieces")))
    for side in facts.sides:
        sections.append(Section(f"{side.name} with the ball", _with_ball(facts, x, side),
                                _visual(out, "pass_map", side.slug) + _visual(out, "box_entries", side.slug)))
        sections.append(Section(f"{side.name} without the ball", _without_ball(facts, x, side),
                                _visual(out, "defensive_activity", side.slug) + _visual(out, "high_regains", side.slug)))
    pressing = _pressing_over_time(facts, extras)
    if pressing:
        sections.append(Section("Pressing through the match", pressing, _visual(out, "pressing_timeline")))
    changes = _changes(facts)
    if changes:
        sections.append(Section("Changes, game state and the last half-hour", changes,
                                _visual(out, "match_momentum") + _visual(out, "game_state")))
    players_text = _players(facts, x) + _vaep_lines(events, facts)
    sections.append(Section("The players who shaped it", players_text, _visual(out, "action_value")))
    sections.append(Section("What decided it", _verdict(facts, x), []))

    for section in sections:
        section.paragraphs = [clean(p) for p in section.paragraphs if p]
        section.notes = False

    used = {p.resolve() for s in sections for p in s.visuals}
    appendix = [p for p in sorted(out.glob("[0-9]*.png")) if p.resolve() not in used]
    appendix += sorted((out / "player_profiles").glob("*/*.png"))
    sections.append(Section(
        "Complete visual guide",
        ["Every other board in the package, each with its own reading. The argument above uses the "
         "figures that carry it; these are the reference behind it."],
        appendix,
        gallery=True,
    ))
    strap = " · ".join(str(v) for v in [match_info.get("competition", ""), match_info.get("date", ""), "MATCH STUDY"] if v)
    # The cover approved in the November 2026 review: floodlights in both
    # clubs' colours, the gradient score and every shot standing on a
    # broadcast pitch. The report's comparison card is the fallback.
    from football_analysis.visuals.cover_art import build_cover_art

    try:
        xg_frame = pd.read_csv(out / "xg.csv", encoding="utf-8-sig")
    except Exception:
        xg_frame = pd.DataFrame()
    cover = build_cover_art(events, xg_frame, {**match_info, "home_id": home.team_id, "away_id": away.team_id,
                                               "home_name": home.name, "away_name": away.name},
                            out, home_color=home.colour, away_color=away.colour,
                            headline=clean(title), standfirst=clean(standfirst))
    if cover is not None:
        import shutil

        shutil.copyfile(cover, out / "article_cover.png")
        cover = out / "article_cover.png"
    else:
        cover = _cover_image(out)
    return Article(clean(title), clean(standfirst), strap, sections, cover, home.name, away.name, context)


