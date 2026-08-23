"""Did a side play well, or did it spend an hour chasing?

Expected goals answer a narrow question — how good were the chances — and the
report was reading them as a broad one. A losing side whose xG finished higher
was reported as the better team, in the headline and in the opening paragraph,
and that is often the opposite of what happened.

PSG 2-1 Aston Villa is the shape. Villa finished on 2.15 expected goals against
1.11 and lost, so the article opened "PSG won the match Aston Villa played
better in". Two paragraphs later it printed the number that contradicts it:
**1.99 of those 2.15 came while Villa were behind.** Ninety-three per cent of
the performance arrived after the game had gone against them, against a side
that had stopped attacking. Sixteen hundredths of an expected goal is what they
managed while the match was level.

That is not a better performance. It is a worse one with a longer tail, and the
evidence was already in the frames — the article simply never weighed it before
forming its verdict.

Nothing here reaches outside the data. Every judgement below is built from the
game-state splits, the chance quality and the box access the pipeline already
computes, which is what keeps a published claim checkable.
"""
from __future__ import annotations

from dataclasses import dataclass

from frame_values import number as _number, ratio as _ratio


# Above this share of a side's xG arriving while behind, the total is a chasing
# total. Two thirds is deliberately short of "almost all": a side that creates
# a third of its danger in a level game was in the match.
CHASING_SHARE = 0.66

# What a side managed before the score sent it chasing. Below this there is no
# evidence it could hurt the opponent in a fair contest.
LEVEL_XG_FLOOR = 0.55

# A chance worth less than this is volume rather than threat.
THIN_CHANCE = 0.10


@dataclass(frozen=True)
class SideVerdict:
    """One team's attacking output, judged by when and how it was created."""

    team: str
    xg: float
    # Called level_xg until an article printed "level, the two sides created
    # 1.63 and 0.11" for a Hull side that made 0.56 of that 1.63 while ahead —
    # and printed 1.07 as their level-state figure two sections earlier. The
    # quantity is everything created before falling behind, which is the right
    # complement to chasing_xg; the name now says so, and seven sentences that
    # called it "level" were reworded off the back of it.
    not_chasing_xg: float    # created while level or ahead
    chasing_xg: float        # created while behind
    chasing_share: float
    xg_per_shot: float
    big_chances: int
    box_entries: int
    final_third_entries: int
    level_only_xg: float = 0.0   # created with the score level, nothing else

    @property
    def was_chasing(self) -> bool:
        return self.chasing_share >= CHASING_SHARE

    @property
    def flattered(self) -> bool:
        """Does the xG total overstate how well this side actually played?

        Both conditions have to hold. A high chasing share alone is ordinary —
        every trailing side creates more — but a high share *and* nothing to
        show from the level phase means the total is a record of the deficit,
        not of a performance.
        """
        return self.was_chasing and self.not_chasing_xg < LEVEL_XG_FLOOR

    @property
    def chances_were_thin(self) -> bool:
        return self.xg_per_shot > 0 and self.xg_per_shot < THIN_CHANCE


def _side_verdict(team: str, metrics, xg_row) -> SideVerdict:
    leading = _number(metrics.get("game_state_leading_xG"))
    drawing = _number(metrics.get("game_state_drawing_xG"))
    trailing = _number(metrics.get("game_state_trailing_xG"))
    total = _number(xg_row.get("xG"))
    return SideVerdict(
        team=str(team),
        xg=total,
        not_chasing_xg=leading + drawing,
        level_only_xg=drawing,
        chasing_xg=trailing,
        chasing_share=_ratio(trailing, total),
        xg_per_shot=_number(xg_row.get("xG_per_shot")),
        big_chances=int(_number(xg_row.get("big_chances"))),
        box_entries=int(_number(metrics.get("box_entries"))),
        final_third_entries=int(_number(metrics.get("final_third_entries"))),
    )


@dataclass(frozen=True)
class Verdict:
    """What the two sides' numbers actually support saying about the match."""

    home: SideVerdict
    away: SideVerdict
    winner: str | None
    loser: str | None

    def of(self, team: str) -> SideVerdict:
        return self.home if team == self.home.team else self.away

    @property
    def xg_leader(self) -> SideVerdict:
        return self.home if self.home.xg >= self.away.xg else self.away

    @property
    def loser_out_created_winner(self) -> bool:
        """The shape the old prose read as "played better"."""
        if not self.loser:
            return False
        return self.of(self.loser).xg > self.of(self.winner).xg + 0.15

    @property
    def loser_was_only_chasing(self) -> bool:
        """Out-created the winner, but only after falling behind."""
        return self.loser_out_created_winner and self.of(self.loser).flattered

    def summary(self) -> str:
        """One sentence naming what the numbers support, for the headline."""
        if not self.loser:
            return "level"
        if self.loser_was_only_chasing:
            return "chasing"
        if self.loser_out_created_winner:
            return "deserved more"
        return "matched the result"


def read_match(team_metrics, xg, info) -> Verdict:
    """Judge both sides from the frames the pipeline already produced."""
    def side(name: str):
        rows = team_metrics[team_metrics["side"].astype(str).eq(name)]
        return rows.iloc[0] if not rows.empty else {}

    def xg_row(team: str):
        rows = xg[xg["team"].astype(str).str.lower().eq(str(team).lower())]
        return rows.iloc[0] if not rows.empty else {}

    home_name = str(info["home_name"])
    away_name = str(info["away_name"])
    home = _side_verdict(home_name, side("home"), xg_row(home_name))
    away = _side_verdict(away_name, side("away"), xg_row(away_name))

    home_goals = int(_number(xg_row(home_name).get("goals")))
    away_goals = int(_number(xg_row(away_name).get("goals")))
    if home_goals > away_goals:
        winner, loser = home_name, away_name
    elif away_goals > home_goals:
        winner, loser = away_name, home_name
    else:
        winner = loser = None

    return Verdict(home=home, away=away, winner=winner, loser=loser)
