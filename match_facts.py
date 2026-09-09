"""Every number the written report argues from, read once out of one package.

The Word report used to carry its figures inline, which meant a second fixture
needed a second report written by hand. Everything a paragraph can claim is
computed here instead, from the same files the charts were drawn from, so the
prose is a function of the match rather than of the match it was first written
for.

Nothing in here is rounded for display. Formatting belongs to the writer; a
fact that arrives pre-rounded cannot be compared against another one.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Pitch thirds and the box, in the 0-100 coordinates the package stores.
THIRD = 100.0 / 3.0
FINAL_THIRD = 2 * THIRD
BOX_X = 83.0
BOX_Y = (21.1, 78.9)
ZONE14 = ((FINAL_THIRD, 83.3), (THIRD, 2 * THIRD))

DEFENSIVE_ACTIONS = ("Tackle", "Interception", "BallRecovery", "Clearance",
                     "Challenge", "BlockedPass")
PRESSING_ACTIONS = ("Tackle", "Interception", "Challenge", "Foul")


def slugify(name: str) -> str:
    """The filename form of a club name: ``Aston Villa`` -> ``aston_villa``."""
    folded = unicodedata.normalize("NFKD", str(name))
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "_", folded.lower()).strip("_")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def _int(value: Any, default: int = 0) -> int:
    return int(round(_num(value, default)))


def _truthy(series: pd.Series) -> pd.Series:
    """CSV booleans arrive as the strings ``True``/``False``."""
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


@dataclass
class Side:
    """One team's match, in the order the report argues it."""
    key: str                      # "home" or "away"
    team_id: int
    name: str
    slug: str
    colour: str
    formation: str = ""
    manager: str = ""

    # Chances
    xg: float = 0.0
    xgot: float = 0.0
    xg_per_shot: float = 0.0
    shots: int = 0
    on_target: int = 0
    off_target: int = 0
    blocked: int = 0
    goals: int = 0
    big_chances: int = 0

    # Territory and progression
    possession_share: float = 0.0
    pass_share: float = 0.0
    field_tilt: float = 0.0
    touches: int = 0
    touch_def_pct: float = 0.0
    touch_mid_pct: float = 0.0
    touch_att_pct: float = 0.0
    progressive_passes: int = 0
    deep_completions: int = 0
    final_third_entries: int = 0
    box_entries: int = 0
    box_entry_to_shot_rate: float = 0.0
    final_third_entry_efficiency: float = 0.0
    crosses: int = 0
    completed_crosses: int = 0
    sequence_xt: float = 0.0
    xt_per_possession: float = 0.0
    directness: float = 0.0
    build_up_attempts: int = 0
    build_up_successes: int = 0
    build_up_success_rate: float = 0.0

    # Pressing, transitions, risk
    ppda: float = 0.0
    regains: int = 0
    high_regains: int = 0
    counterpress_regains: int = 0
    counterpress_attempts: int = 0
    counterpress_success_rate: float = 0.0
    regain_xg: float = 0.0
    regain_xt: float = 0.0
    transitions: int = 0
    transition_shots: int = 0
    transition_xg: float = 0.0
    transition_box_entries: int = 0
    transition_shot_rate: float = 0.0
    avg_transition_duration: float = 0.0
    rest_defence_exposures: int = 0
    rest_defence_dangerous: int = 0
    rest_defence_vulnerability: float = 0.0

    # Possession shape
    possessions: int = 0
    avg_possession_seconds: float = 0.0
    avg_possession_passes: float = 0.0
    long_sequences: int = 0
    short_sequences: int = 0
    fastbreaks: int = 0
    avg_seconds_to_third: float = 0.0
    reached_third: int = 0
    avg_max_x: float = 0.0
    funnel: tuple[int, ...] = ()

    # Event-derived
    passes: int = 0
    completed_passes: int = 0
    avg_pass_length: float = 0.0
    forward_passes: int = 0
    long_passes: int = 0
    corners: int = 0
    fouls: int = 0
    cards: int = 0
    aerials_won: int = 0
    aerials: int = 0
    zone14_touches: int = 0
    zone14_passes: int = 0
    final_third_touches: int = 0
    def_actions: int = 0
    def_actions_high: int = 0
    def_actions_mid: int = 0
    def_actions_low: int = 0
    def_action_avg_x: float = 0.0
    lane_left: int = 0
    lane_centre: int = 0
    lane_right: int = 0
    set_piece_shots: int = 0
    set_piece_xg: float = 0.0
    gk_saves: int = 0
    outfield_blocks: int = 0

    # Per half
    half: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per score state, from spells.csv
    states: dict[str, dict[str, float]] = field(default_factory=dict)

    # Losses
    losses: int = 0
    losses_to_third: int = 0
    losses_to_box: int = 0
    losses_to_shot: int = 0
    loss_xg_conceded: float = 0.0

    # Entry routes
    route_entries: int = 0
    route_entries_with_shot: int = 0

    players: list[dict[str, Any]] = field(default_factory=list)

    @property
    def short(self) -> str:
        """A name short enough to repeat in every sentence."""
        return self.name

    def shot_conversion(self) -> float:
        return self.goals / self.shots if self.shots else 0.0


@dataclass
class MatchFacts:
    directory: Path
    info: dict[str, Any]
    home: Side
    away: Side
    score: str
    date: str
    venue: str
    competition: str
    season: str
    round_name: str
    goals: list[dict[str, Any]] = field(default_factory=list)
    cards: list[dict[str, Any]] = field(default_factory=list)
    substitutions: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    shots: list[dict[str, Any]] = field(default_factory=list)
    chart_readings: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped_charts: list[dict[str, str]] = field(default_factory=list)
    sub_windows: pd.DataFrame | None = None

    def side(self, key: str) -> Side:
        return self.home if key == "home" else self.away

    def by_slug(self, slug: str) -> Side | None:
        for side in (self.home, self.away):
            if side.slug == slug:
                return side
        return None

    def other(self, side: Side) -> Side:
        return self.away if side is self.home else self.home

    @property
    def sides(self) -> tuple[Side, Side]:
        return self.home, self.away

    @property
    def title(self) -> str:
        return f"{self.home.name} vs {self.away.name}"

    def leader(self, attribute: str) -> Side:
        """Whichever side holds the larger value of one attribute."""
        return max(self.sides, key=lambda s: _num(getattr(s, attribute)))

    def winner(self) -> Side | None:
        if self.home.goals > self.away.goals:
            return self.home
        if self.away.goals > self.home.goals:
            return self.away
        return None


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return frame


def _team_row(frame: pd.DataFrame, team_id: int, column: str = "team_id") -> dict:
    if frame.empty or column not in frame.columns:
        return {}
    match = frame[frame[column].apply(lambda v: _int(v) == team_id)]
    return match.iloc[0].to_dict() if len(match) else {}


def _formation(info: dict, side_key: str) -> str:
    """The starting shape, hyphenated the way a reader writes it."""
    for entry in info.get("formations") or []:
        if entry.get("side") == side_key:
            digits = str(entry.get("formation") or "")
            return "-".join(digits) if digits.isdigit() else digits
    raw = str(info.get("home_form" if side_key == "home" else "away_form") or "")
    return "-".join(raw) if raw.isdigit() else raw


def _clean_text(value) -> str:
    """A field's text, or "" when the feed left it empty.

    pandas fills a missing string cell with float NaN, which is truthy, so the
    usual `value or ""` guard passes it straight through to str().
    """
    if value is None:
        return ""
    try:
        if value != value:            # NaN is the only value unequal to itself
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat", "<na>"} else text


def load_match(directory: str | Path) -> MatchFacts:
    """Read one rendered match package into the facts the report argues from."""
    out = Path(directory)
    info = json.loads((out / "match_info.json").read_text(encoding="utf-8"))
    events = _read_csv(out / "events.csv")
    xg = _read_csv(out / "xg.csv")
    team_metrics = _read_csv(out / "team_advanced_metrics.csv")
    sequences = _read_csv(out / "player_sequence_metrics.csv")
    tables = out / "analysis_tables"
    players = _read_csv(tables / "players.csv")
    possessions = _read_csv(tables / "possessions.csv")
    losses = _read_csv(tables / "losses.csv")
    routes = _read_csv(tables / "routes.csv")
    spells = _read_csv(tables / "spells.csv")
    sub_windows = _read_csv(tables / "substitution_windows.csv")

    sides = {}
    for key in ("home", "away"):
        name = str(info.get(f"{key}_name") or key.title())
        sides[key] = Side(
            key=key,
            team_id=_int(info.get(f"{key}_id")),
            name=name,
            slug=slugify(name),
            colour=str(info.get(f"{key}_color") or ("#C8102E" if key == "home" else "#1D4ED8")),
            formation=_formation(info, key),
            manager=str((info.get("managers") or {}).get(key) or ""),
        )

    for side in sides.values():
        _fill_from_xg(side, xg)
        _fill_from_team_metrics(side, team_metrics)
        _fill_from_events(side, events, sides)
        _fill_from_possessions(side, possessions)
        _fill_from_losses(side, losses)
        _fill_from_routes(side, routes)
        _fill_from_spells(side, spells)
        _fill_players(side, players, sequences, events)

    facts = MatchFacts(
        directory=out,
        info=info,
        home=sides["home"],
        away=sides["away"],
        score=str(info.get("score") or "").replace(" ", ""),
        date=str(info.get("date") or ""),
        venue=str(info.get("venue") or ""),
        competition=str(info.get("competition") or ""),
        season=str(info.get("season") or ""),
        round_name=str(info.get("round_name") or ""),
        sub_windows=sub_windows if not sub_windows.empty else None,
    )
    _fill_timeline(facts, events, sides)
    _fill_manifest(facts, out)
    return facts


def _fill_from_xg(side: Side, xg: pd.DataFrame) -> None:
    if xg.empty:
        return
    rows = xg[xg["team"].astype(str) == side.name] if "team" in xg.columns else pd.DataFrame()
    if not len(rows):
        return
    row = rows.iloc[0].to_dict()
    side.xg = _num(row.get("xG"))
    side.xgot = _num(row.get("xGoT"))
    side.xg_per_shot = _num(row.get("xG_per_shot"))
    side.shots = _int(row.get("shots"))
    side.on_target = _int(row.get("on_target"))
    side.off_target = _int(row.get("off_target"))
    side.blocked = _int(row.get("blocked"))
    side.goals = _int(row.get("goals"))
    side.big_chances = _int(row.get("big_chances"))


_TEAM_METRIC_FIELDS = {
    "possession_share": "possession_share", "pass_share": "pass_share",
    "field_tilt": "field_tilt", "touches": "touches",
    "touch_def_pct": "touch_def_pct", "touch_mid_pct": "touch_mid_pct",
    "touch_att_pct": "touch_att_pct",
    "progressive_passes": "progressive_passes",
    "deep_completions": "deep_completions",
    "final_third_entries": "final_third_entries",
    "final_third_entry_efficiency": "final_third_entry_efficiency",
    "box_entries": "box_entries",
    "box_entry_to_shot_rate": "box_entry_to_shot_rate",
    "crosses": "crosses", "completed_crosses": "completed_crosses",
    "sequence_xt": "sequence_xT", "xt_per_possession": "xt_per_possession",
    "directness": "directness",
    "build_up_attempts": "build_up_attempts",
    "build_up_successes": "build_up_successes",
    "build_up_success_rate": "build_up_success_rate",
    "regains": "possession_regains", "high_regains": "high_regains",
    "counterpress_regains": "counterpress_regains",
    "counterpress_attempts": "counterpress_attempts",
    "counterpress_success_rate": "counterpress_success_rate",
    "regain_xg": "regain_xG", "regain_xt": "regain_xT",
    "transitions": "transitions", "transition_shots": "transition_shots",
    "transition_xg": "transition_xG",
    "transition_box_entries": "transition_box_entries",
    "transition_shot_rate": "transition_shot_rate",
    "avg_transition_duration": "avg_transition_duration",
    "rest_defence_exposures": "rest_defence_exposures",
    "rest_defence_dangerous": "rest_defence_dangerous_counters",
    "rest_defence_vulnerability": "rest_defence_vulnerability",
}
_INTEGER_FIELDS = {
    "touches", "progressive_passes", "deep_completions", "final_third_entries",
    "box_entries", "crosses", "completed_crosses", "build_up_attempts",
    "build_up_successes", "regains", "high_regains", "counterpress_regains",
    "counterpress_attempts", "transitions", "transition_shots",
    "transition_box_entries", "rest_defence_exposures", "rest_defence_dangerous",
}


def _fill_from_team_metrics(side: Side, frame: pd.DataFrame) -> None:
    row = _team_row(frame, side.team_id)
    if not row:
        return
    for attribute, column in _TEAM_METRIC_FIELDS.items():
        if column in row:
            value = _int(row[column]) if attribute in _INTEGER_FIELDS else _num(row[column])
            setattr(side, attribute, value)
    for state in ("leading", "drawing", "trailing"):
        side.states.setdefault(state, {})
        for metric in ("possessions", "completed_passes", "shots", "xG",
                       "sequence_xT", "transitions", "box_entries"):
            column = f"game_state_{state}_{metric}"
            if column in row:
                side.states[state][metric] = _num(row[column])


def _fill_from_events(side: Side, events: pd.DataFrame, sides: dict[str, Side]) -> None:
    if events.empty:
        return
    own = events[events["team_id"].apply(lambda v: _int(v) == side.team_id)]
    opponent_id = next(s.team_id for s in sides.values() if s.team_id != side.team_id)
    opponent = events[events["team_id"].apply(lambda v: _int(v) == opponent_id)]

    passes = own[_truthy(own["is_pass"])] if "is_pass" in own.columns else own.iloc[0:0]
    completed = passes[passes["outcome"].astype(str) == "Successful"]
    side.passes = len(passes)
    side.completed_passes = len(completed)
    if "pass_length" in passes.columns:
        lengths = pd.to_numeric(passes["pass_length"], errors="coerce").dropna()
        side.avg_pass_length = float(lengths.mean()) if len(lengths) else 0.0
        side.long_passes = int((lengths >= 30).sum())
    forward = passes[pd.to_numeric(passes["end_x"], errors="coerce")
                     > pd.to_numeric(passes["x"], errors="coerce")]
    side.forward_passes = len(forward)

    side.corners = int((own["type"].astype(str) == "CornerAwarded").sum())
    side.fouls = int(((own["type"].astype(str) == "Foul")
                      & (own["outcome"].astype(str) == "Unsuccessful")).sum())
    side.cards = int((own["type"].astype(str) == "Card").sum())
    aerials = own[own["type"].astype(str) == "Aerial"]
    side.aerials = len(aerials)
    side.aerials_won = int((aerials["outcome"].astype(str) == "Successful").sum())

    x = pd.to_numeric(own.get("x"), errors="coerce")
    y = pd.to_numeric(own.get("y"), errors="coerce")
    in_zone14 = ((x >= ZONE14[0][0]) & (x <= ZONE14[0][1])
                 & (y >= ZONE14[1][0]) & (y <= ZONE14[1][1]))
    side.zone14_touches = int(in_zone14.sum())
    side.zone14_passes = int((in_zone14 & _truthy(own["is_pass"])).sum())
    side.final_third_touches = int((x >= FINAL_THIRD).sum())

    defensive = own[own["type"].astype(str).isin(DEFENSIVE_ACTIONS)]
    dx = pd.to_numeric(defensive.get("x"), errors="coerce")
    side.def_actions = len(defensive)
    side.def_actions_high = int((dx >= FINAL_THIRD).sum())
    side.def_actions_mid = int(((dx >= THIRD) & (dx < FINAL_THIRD)).sum())
    side.def_actions_low = int((dx < THIRD).sum())
    side.def_action_avg_x = float(dx.mean()) if len(dx.dropna()) else 0.0

    # PPDA: the opponent's passes in their own 60%, per pressing action of ours
    # in the same area of the pitch.
    opponent_passes = opponent[_truthy(opponent["is_pass"])] if "is_pass" in opponent.columns else opponent.iloc[0:0]
    deep = pd.to_numeric(opponent_passes.get("x"), errors="coerce") <= 60
    pressing = own[own["type"].astype(str).isin(PRESSING_ACTIONS)]
    high = pd.to_numeric(pressing.get("x"), errors="coerce") >= 40
    actions = int(high.sum())
    side.ppda = float(int(deep.sum()) / actions) if actions else 0.0

    crosses = own[_truthy(own["is_cross"])] if "is_cross" in own.columns else own.iloc[0:0]
    if len(crosses) and not side.crosses:
        side.crosses = len(crosses)
        side.completed_crosses = int((crosses["outcome"].astype(str) == "Successful").sum())

    entries = completed[(pd.to_numeric(completed["x"], errors="coerce") < FINAL_THIRD)
                        & (pd.to_numeric(completed["end_x"], errors="coerce") >= FINAL_THIRD)]
    end_y = pd.to_numeric(entries.get("end_y"), errors="coerce")
    side.lane_left = int((end_y >= FINAL_THIRD).sum())
    side.lane_centre = int(((end_y >= THIRD) & (end_y < FINAL_THIRD)).sum())
    side.lane_right = int((end_y < THIRD).sum())

    shots = own[_truthy(own["is_shot"])] if "is_shot" in own.columns else own.iloc[0:0]
    qualifiers = shots.get("qualifier_names", pd.Series(dtype=str)).astype(str)
    set_piece = shots[qualifiers.str.contains("SetPiece|Corner|FreeKick|ThrowIn",
                                              case=False, na=False)]
    side.set_piece_shots = len(set_piece)
    side.set_piece_xg = float(pd.to_numeric(set_piece.get("xG"), errors="coerce").sum() or 0.0)

    saves = own[own["type"].astype(str) == "Save"]
    if len(saves):
        counts = saves["player"].astype(str).value_counts()
        keeper = _keeper_name(side, events)
        side.gk_saves = int(counts.get(keeper, 0)) if keeper else int(counts.iloc[0])
        side.outfield_blocks = int(len(saves) - side.gk_saves)

    for label, code in (("first", "1h"), ("second", "2h")):
        half = own[own["period_code"].astype(str) == code] if "period_code" in own.columns else own.iloc[0:0]
        half_passes = half[_truthy(half["is_pass"])] if len(half) else half
        half_shots = half[_truthy(half["is_shot"])] if len(half) else half
        side.half[label] = {
            "passes": float(len(half_passes)),
            "completed": float((half_passes["outcome"].astype(str) == "Successful").sum()) if len(half_passes) else 0.0,
            "shots": float(len(half_shots)),
            "xG": float(pd.to_numeric(half_shots.get("xG"), errors="coerce").sum() or 0.0) if len(half_shots) else 0.0,
            "xT": float(pd.to_numeric(half.get("xT"), errors="coerce").sum() or 0.0) if len(half) else 0.0,
            "box_entries": float(_box_entries(half)),
        }


def _box_entries(frame: pd.DataFrame) -> int:
    """Successful actions that finish inside the box having started outside."""
    if frame.empty or "end_x" not in frame.columns:
        return 0
    end_x = pd.to_numeric(frame["end_x"], errors="coerce")
    end_y = pd.to_numeric(frame["end_y"], errors="coerce")
    x = pd.to_numeric(frame["x"], errors="coerce")
    y = pd.to_numeric(frame["y"], errors="coerce")
    inside = (end_x >= BOX_X) & (end_y >= BOX_Y[0]) & (end_y <= BOX_Y[1])
    outside = (x < BOX_X) | (y < BOX_Y[0]) | (y > BOX_Y[1])
    ok = frame["outcome"].astype(str) == "Successful"
    return int((inside & outside & ok).sum())


def _keeper_name(side: Side, events: pd.DataFrame) -> str:
    meta = events[events["team_id"].apply(lambda v: _int(v) == side.team_id)]
    keepers = meta[meta["type"].astype(str).isin(("KeeperPickup", "Claim", "Smother", "Punch"))]
    if len(keepers):
        return str(keepers["player"].astype(str).mode().iloc[0])
    return ""


def _fill_from_possessions(side: Side, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    own = frame[frame["team_id"].apply(lambda v: _int(v) == side.team_id)]
    if not len(own):
        return
    passes = pd.to_numeric(own["passes"], errors="coerce").fillna(0)
    side.possessions = len(own)
    side.avg_possession_seconds = float(pd.to_numeric(own["duration"], errors="coerce").mean() or 0.0)
    side.avg_possession_passes = float(passes.mean() or 0.0)
    side.long_sequences = int((passes >= 6).sum())
    side.short_sequences = int((passes <= 2).sum())
    side.fastbreaks = int(_truthy(own["provider_fastbreak"]).sum())
    seconds = pd.to_numeric(own["seconds_to_third"], errors="coerce").dropna()
    side.avg_seconds_to_third = float(seconds.mean()) if len(seconds) else 0.0
    side.reached_third = int(_truthy(own["reached_final_third"]).sum())
    side.avg_max_x = float(pd.to_numeric(own["max_x"], errors="coerce").mean() or 0.0)
    reached_box = int(_truthy(own["reached_box_after_third"]).sum())
    # The funnel counts a possession at the stage it reached, so the shot stage
    # is the one taken after the box was entered - not any shot in the
    # possession, which would let a first-minute effort from distance stand in
    # for box access the possession never had.
    with_shot = int(_truthy(own["shot_after_box"]).sum())
    on_target = int(_truthy(own["target_after_box"]).sum())
    side.funnel = (len(own), side.reached_third, reached_box, with_shot, on_target)


def _fill_from_losses(side: Side, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    own = frame[frame["team_id"].apply(lambda v: _int(v) == side.team_id)]
    if not len(own):
        return
    side.losses = len(own)
    side.losses_to_third = int(_truthy(own["third_within_12"]).sum())
    side.losses_to_box = int(_truthy(own["box_within_12"]).sum())
    side.losses_to_shot = int(_truthy(own["shot_within_12"]).sum())
    side.loss_xg_conceded = float(pd.to_numeric(own["xG_conceded"], errors="coerce").sum() or 0.0)


def _fill_from_routes(side: Side, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    own = frame[frame["team_id"].apply(lambda v: _int(v) == side.team_id)]
    side.route_entries = len(own)
    side.route_entries_with_shot = int(_truthy(own["shot_followed"]).sum()) if len(own) else 0


def _fill_from_spells(side: Side, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    own = frame[frame["team_id"].apply(lambda v: _int(v) == side.team_id)]
    for state in ("leading", "level", "trailing"):
        rows = own[own["state"].astype(str) == state]
        if not len(rows):
            continue
        bucket = side.states.setdefault(state if state != "level" else "drawing", {})
        bucket["minutes"] = float(pd.to_numeric(rows["minutes"], errors="coerce").sum() or 0.0)
        bucket.setdefault("shots", float(pd.to_numeric(rows["shots"], errors="coerce").sum() or 0.0))
        bucket["spell_shots"] = float(pd.to_numeric(rows["shots"], errors="coerce").sum() or 0.0)
        bucket["spell_xG"] = float(pd.to_numeric(rows["xG"], errors="coerce").sum() or 0.0)
        bucket["spell_box_entries"] = float(pd.to_numeric(rows["box_entries"], errors="coerce").sum() or 0.0)


def _fill_players(side: Side, players: pd.DataFrame, sequences: pd.DataFrame,
                  events: pd.DataFrame) -> None:
    if players.empty:
        return
    own = players[players["team_id"].apply(lambda v: _int(v) == side.team_id)]
    chains = {}
    if not sequences.empty:
        for _, row in sequences.iterrows():
            if _int(row.get("team_id")) == side.team_id:
                chains[str(row.get("player"))] = {
                    "xGChain": _num(row.get("xGChain")),
                    "xGBuildup": _num(row.get("xGBuildup")),
                    "sequences": _num(row.get("sequences")),
                    "sequence_xT": _num(row.get("sequence_xT")),
                }
    key_passes = {}
    if not events.empty and "is_key_pass" in events.columns:
        marked = events[_truthy(events["is_key_pass"])
                        & events["team_id"].apply(lambda v: _int(v) == side.team_id)]
        key_passes = marked["player"].astype(str).value_counts().to_dict()

    records = []
    for _, row in own.iterrows():
        name = str(row.get("player"))
        chain = chains.get(name, {})
        record = {
            "name": name,
            "role": str(row.get("role") or ""),
            "minutes": _num(row.get("minutes")),
            "touches": _int(row.get("touches")),
            "passes": _int(row.get("passes")),
            "completed_passes": _int(row.get("completed_passes")),
            "pass_pct": _num(row.get("pass_pct")),
            "progressive_passes": _int(row.get("progressive_passes")),
            "progressive_carries": _int(row.get("progressive_carries")),
            "box_entries": _int(row.get("box_entries")),
            "shots": _int(row.get("shots")),
            "xG": _num(row.get("xG")),
            "xA": _num(row.get("xA")),
            "positive_xT": _num(row.get("positive_xT")),
            "takeons": _int(row.get("takeons")),
            "takeons_won": _int(row.get("takeons_won")),
            "recoveries": _int(row.get("recoveries")),
            "interceptions": _int(row.get("interceptions")),
            "tackles_won": _int(row.get("tackles_won")),
            "clearances": _int(row.get("clearances")),
            "goals": _int(row.get("goals")),
            "saves": _int(row.get("saves")),
            "key_passes": _int(key_passes.get(name, 0)),
            "xGChain": chain.get("xGChain", 0.0),
            "xGBuildup": chain.get("xGBuildup", 0.0),
            "sequences": chain.get("sequences", 0.0),
        }
        record["score"] = _impact_score(record)
        records.append(record)
    side.players = sorted(records, key=lambda r: r["score"], reverse=True)


# Weights are deliberately blunt and printed in the report, so a reader can
# disagree with the ranking by disagreeing with a number rather than a taste.
IMPACT_WEIGHTS = {
    "goals": 1.20, "xG": 0.70, "xA": 0.90, "xGChain": 0.80,
    "positive_xT": 0.12, "progressive_passes": 0.035, "box_entries": 0.07,
    "key_passes": 0.10, "takeons_won": 0.03, "saves": 0.06,
}
DEFENSIVE_WEIGHT = 0.03
MIN_MINUTES = 20.0


def _impact_score(player: dict) -> float:
    if player["minutes"] < MIN_MINUTES:
        return 0.0
    total = sum(weight * _num(player.get(field_name))
                for field_name, weight in IMPACT_WEIGHTS.items())
    stops = player["tackles_won"] + player["interceptions"] + player["clearances"]
    return total + DEFENSIVE_WEIGHT * stops


def top_players(side: Side, count: int = 3) -> list[dict]:
    return [p for p in side.players if p["score"] > 0][:count]


def next_best(side: Side, count: int = 3) -> dict | None:
    eligible = [p for p in side.players if p["score"] > 0]
    return eligible[count] if len(eligible) > count else None


# ----------------------------------------------------------------------
# Timeline and manifest
# ----------------------------------------------------------------------
def _fill_timeline(facts: MatchFacts, events: pd.DataFrame, sides: dict[str, Side]) -> None:
    if events.empty:
        return
    by_id = {side.team_id: side for side in sides.values()}

    def side_of(row) -> Side | None:
        return by_id.get(_int(row.get("team_id")))

    shots = events[_truthy(events["is_shot"])] if "is_shot" in events.columns else events.iloc[0:0]
    for _, row in shots.sort_values("minute").iterrows():
        side = side_of(row)
        qualifiers = str(row.get("qualifier_names") or "")
        facts.shots.append({
            "minute": _int(row.get("minute")),
            "player": str(row.get("player")),
            "team": side.name if side else "",
            "side": side.key if side else "",
            "xG": _num(row.get("xG")),
            "type": str(row.get("type")),
            "big_chance": str(row.get("big_chance")).lower() == "true",
            "header": str(row.get("is_header")).lower() == "true",
            "penalty": str(row.get("is_penalty")).lower() == "true",
            "set_piece": bool(re.search(r"SetPiece|Corner|FreeKick", qualifiers, re.I)),
            # NaN is truthy, so `row.get("body_part") or ""` never fired on a
            # missing value and str(nan) reached the page: "0.00 expected
            # goals, nan".
            "body": _clean_text(row.get("body_part")),
        })
        if str(row.get("is_goal")).lower() == "true":
            facts.goals.append(facts.shots[-1])

    cards = events[events["type"].astype(str) == "Card"]
    for _, row in cards.sort_values("minute").iterrows():
        side = side_of(row)
        facts.cards.append({
            "minute": _int(row.get("minute")),
            "player": str(row.get("player")),
            "team": side.name if side else "",
            "colour": "Red" if "Red" in str(row.get("qualifier_names")) else "Yellow",
        })

    subs = events[events["type"].astype(str).str.startswith("Substitution")]
    for _, row in subs.sort_values("minute").iterrows():
        side = side_of(row)
        facts.substitutions.append({
            "minute": _int(row.get("minute")),
            "player": str(row.get("player")),
            "team": side.name if side else "",
            "direction": "on" if str(row.get("type")).endswith("On") else "off",
        })

    for _, row in events[events["type"].astype(str) == "Error"].iterrows():
        side = side_of(row)
        facts.errors.append({
            "minute": _int(row.get("minute")),
            "player": str(row.get("player")),
            "team": side.name if side else "",
        })


def _fill_manifest(facts: MatchFacts, out: Path) -> None:
    manifest = out / "package_manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            facts.chart_readings = payload.get("charts") or {}
        except Exception:
            facts.chart_readings = {}
    skipped = out / "analysis_tables" / "skipped_charts.json"
    if skipped.exists():
        try:
            facts.skipped_charts = json.loads(skipped.read_text(encoding="utf-8"))
        except Exception:
            facts.skipped_charts = []
