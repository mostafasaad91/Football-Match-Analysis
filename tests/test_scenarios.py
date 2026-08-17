"""Every match shape the prose can meet, not only the two that are rendered.

The rest of the suite reads the two fixtures in `output/`. Both are home wins,
both were settled without the lead changing hands, neither finished level on
anything, and in both the home side's name is one word. So a large part of the
generated prose has never executed: the level branches, the draw branches, the
away-win branches, the lead-change branch, the goalless branch.

That is exactly where the defects lived. "Man City shot more often, 9 to 12"
survived because the home side happened to shoot fewer; the team matcher's slug
bug was invisible for a one-word away side. A branch nobody has run is a branch
nobody has checked.

These fixtures are derived from a real match by transformation, so the frames
stay internally consistent, and every prose writer is run over every board for
each of them. The assertions are the ones that hold for any match: no crash, no
placeholder, no filler, no sentence that claims a difference over a tie.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from match_article import build_article
from tactical_pdf_report import (
    _section_copy,
    build_context,
    visual_data_read,
    visual_explanation,
    visual_implication,
)

ROOT = Path(__file__).resolve().parent.parent
BASE = "Arsenal_vs_Man_City_3-0"

WRITERS = (visual_explanation, visual_implication, visual_data_read)

# Strings that mean the generator gave up or leaked its own plumbing.
FILLER = (
    "The deeper reading is the causal chain behind the pattern",
    "Translate the pattern into a coachable behaviour",
    "The numerical layer should confirm the visual pattern",
)
# Whole words only: "nan" hides inside "dominance" and "inf" inside
# "reinforced". "None" is a real English word — "None of that is the striker's
# doing" is correct prose — so it counts as a leak only mid-sentence, where a
# repr would land and the pronoun would not.
LEAK = re.compile(r"\b(nan|NaN|inf)\b|(?<![.!?:]\s)\bNone\b|[{}]|\s\|\s|No qualifying player")


def _leaks(text: str) -> list[str]:
    return [m.group(0) for m in LEAK.finditer(text)]


def _base():
    out = ROOT / "output" / BASE
    if not (out / "match_info.json").exists():
        pytest.skip(f"{BASE} has not been rendered")
    return (
        pd.read_csv(out / "events.csv"),
        pd.read_csv(out / "xg.csv"),
        pd.read_csv(out / "team_advanced_metrics.csv"),
        pd.read_csv(out / "player_sequence_metrics.csv"),
        json.loads((out / "match_info.json").read_text(encoding="utf-8")),
        out,
    )


def _boards(out: Path) -> list[Path]:
    """Board paths only — the writers read the filename, not the pixels."""
    return sorted(out.glob("[0-9]*.png"))


# --------------------------------------------------------------------------
# the shapes
# --------------------------------------------------------------------------

def _rename(events, xg, team_metrics, player_metrics, info, home, away):
    """Give the two sides different names, keeping every id intact."""
    old_home, old_away = str(info["home_name"]), str(info["away_name"])
    mapping = {old_home: home, old_away: away}
    xg, team_metrics, player_metrics = xg.copy(), team_metrics.copy(), player_metrics.copy()
    for frame in (xg, team_metrics, player_metrics):
        if "team" in frame:
            frame["team"] = frame["team"].map(lambda v: mapping.get(str(v), v))
    info = dict(info, home_name=home, away_name=away)
    return events, xg, team_metrics, player_metrics, info


def _goalless(events, xg, team_metrics, player_metrics, info):
    """Nobody scored: the score-state and first-goal branches have nothing."""
    events = events.copy()
    events["is_goal"] = False
    xg = xg.copy()
    xg["goals"] = 0
    return events, xg, team_metrics, player_metrics, info


def _level(events, xg, team_metrics, player_metrics, info):
    """Both sides equal on everything the prose compares."""
    xg = xg.copy()
    for column in xg.select_dtypes("number").columns:
        xg[column] = xg[column].mean()
    team_metrics = team_metrics.copy()
    for column in team_metrics.select_dtypes("number").columns:
        if column.endswith("_id"):
            continue
        team_metrics[column] = team_metrics[column].mean()
    events = events.copy()
    events["is_goal"] = False
    xg["goals"] = 1
    # One goal each, so the match is a draw that was level throughout.
    goals = events.index[:2]
    events.loc[goals, "is_goal"] = True
    ids = [int(info["home_id"]), int(info["away_id"])]
    for position, index in enumerate(goals):
        events.loc[index, "team_id"] = ids[position]
    return events, xg, team_metrics, player_metrics, info


def _away_win(events, xg, team_metrics, player_metrics, info):
    """The away side wins — the mirror of every rendered fixture."""
    xg = xg.copy()
    home_name = str(info["home_name"])
    is_home = xg["team"].astype(str).eq(home_name)
    xg.loc[is_home, "goals"] = 0
    xg.loc[~is_home, "goals"] = 2
    events = events.copy()
    events["is_goal"] = False
    away_id = int(info["away_id"])
    away_rows = events.index[events["team_id"].eq(away_id)][:2]
    events.loc[away_rows, "is_goal"] = True
    return events, xg, team_metrics, player_metrics, info


def _lead_changed(events, xg, team_metrics, player_metrics, info):
    """Away scores first, home turns it round: the lead changes hands."""
    xg = xg.copy()
    home_name = str(info["home_name"])
    is_home = xg["team"].astype(str).eq(home_name)
    xg.loc[is_home, "goals"] = 2
    xg.loc[~is_home, "goals"] = 1
    events = events.copy()
    events["is_goal"] = False
    home_id, away_id = int(info["home_id"]), int(info["away_id"])
    away_first = events.index[events["team_id"].eq(away_id)][0]
    home_later = [i for i in events.index[events["team_id"].eq(home_id)] if i > away_first][:2]
    events.loc[[away_first, *home_later], "is_goal"] = True
    return events, xg, team_metrics, player_metrics, info


def _without_optional_columns(events, xg, team_metrics, player_metrics, info):
    """A thinner export: the optional metrics simply are not there."""
    drop = ["ppda", "rest_defence_dangerous_counters", "counterpress_success_rate",
            "transition_goals", "deep_completions", "build_up_success_rate"]
    team_metrics = team_metrics.drop(columns=[c for c in drop if c in team_metrics], errors="ignore")
    return events, xg, team_metrics, player_metrics, info


SHAPES = {
    "as_rendered": lambda *a: a,
    "two_word_home": lambda e, x, t, p, i: _rename(e, x, t, p, i, "Real Sociedad", "Leeds"),
    "one_word_both": lambda e, x, t, p, i: _rename(e, x, t, p, i, "Porto", "Ajax"),
    "goalless": _goalless,
    "level_everything": _level,
    "away_win": _away_win,
    "lead_changed_hands": _lead_changed,
    "thin_export": _without_optional_columns,
}


def _shaped(name):
    events, xg, team_metrics, player_metrics, info, out = _base()
    events, xg, team_metrics, player_metrics, info = SHAPES[name](
        events, xg, team_metrics, player_metrics, info)
    return events, xg, team_metrics, player_metrics, info, out


# --------------------------------------------------------------------------
# what must hold for any of them
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_report_writes_every_board_without_filler_or_leaks(shape):
    events, xg, team_metrics, player_metrics, info, out = _shaped(shape)
    context = build_context(events, xg, team_metrics, player_metrics, info)
    boards = _boards(out)
    if shape in ("two_word_home", "one_word_both"):
        # The renamed sides own no files, so team-specific boards cannot be
        # attributed. Only the whole-match boards are meaningful here.
        boards = [b for b in boards if b.stem[-1].isdigit() or "_" not in b.stem]

    for board in boards:
        for write in WRITERS:
            text = write(board, context)
            assert text and text.strip(), (shape, board.name, write.__name__)
            for filler in FILLER:
                assert filler not in text, (shape, board.name, write.__name__)
            found = _leaks(text)
            assert not found, (shape, board.name, write.__name__, found, text[:160])


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_section_copy_survives_every_shape(shape):
    events, xg, team_metrics, player_metrics, info, _ = _shaped(shape)
    context = build_context(events, xg, team_metrics, player_metrics, info)
    copy = _section_copy(context)
    assert copy, shape
    for section in copy.values():
        for heading, text in section["performance"] + section["data"]:
            assert heading and text, (shape, heading)
            found = _leaks(text)
            assert not found, (shape, heading, found, text[:160])
        assert section["implication"].strip(), shape


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_article_is_written_for_every_shape(shape):
    events, xg, team_metrics, player_metrics, info, out = _shaped(shape)
    article = build_article(events, xg, team_metrics, player_metrics, info, out)
    assert article.sections, shape
    assert article.title.strip() and article.standfirst.strip(), shape
    prose = " ".join(p for s in article.sections for p in s.paragraphs)
    found = _leaks(prose)
    assert not found, (shape, found)
    assert "in minute 0" not in prose, shape


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_no_goal_is_timed_as_minute_zero_in_any_shape(shape):
    events, xg, team_metrics, player_metrics, info, out = _shaped(shape)
    context = build_context(events, xg, team_metrics, player_metrics, info)
    text = " ".join(write(b, context) for b in _boards(out) for write in WRITERS)
    for section in _section_copy(context).values():
        text += " " + " ".join(t for _, t in section["performance"] + section["data"])
    assert "minute 0" not in text, shape
    assert not re.search(r"\b0'", text), shape


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_no_count_of_one_is_printed_as_a_plural_in_any_shape(shape):
    events, xg, team_metrics, player_metrics, info, out = _shaped(shape)
    context = build_context(events, xg, team_metrics, player_metrics, info)
    texts = [write(b, context) for b in _boards(out) for write in WRITERS]
    for section in _section_copy(context).values():
        texts += [t for _, t in section["performance"] + section["data"]]
    offenders = []
    for text in texts:
        for found in re.finditer(r"(?<![\d—–-]\s)(?<![\d—–-])\b1 (\w+?)(s|es)\b", text):
            if not found.group(0).endswith(("ss", "ess")):
                offenders.append(found.group(0))
    assert not offenders, (shape, sorted(set(offenders)))


def test_a_level_match_is_never_described_as_having_a_leader():
    """Every comparison in a match where both sides are identical."""
    events, xg, team_metrics, player_metrics, info, out = _shaped("level_everything")
    context = build_context(events, xg, team_metrics, player_metrics, info)
    text = " ".join(write(b, context) for b in _boards(out) for write in WRITERS)
    for section in _section_copy(context).values():
        text += " " + " ".join(t for _, t in section["performance"] + section["data"])
    for claim in ("shot more often", "The same asymmetry", "And it was paid for",
                  "stronger field tilt", "'s curve finished above"):
        assert claim not in text, (claim, "claimed in a match where the sides are equal")


def test_a_goalless_match_does_not_describe_an_opening_goal():
    events, xg, team_metrics, player_metrics, info, out = _shaped("goalless")
    context = build_context(events, xg, team_metrics, player_metrics, info)
    text = " ".join(write(b, context) for b in _boards(out) for write in WRITERS)
    for section in _section_copy(context).values():
        text += " " + " ".join(t for _, t in section["performance"] + section["data"])
    assert "The first goal arrived" not in text
    assert "The opening goal" not in text
    assert "scored first" not in text


def test_an_away_win_is_named_as_an_away_win():
    events, xg, team_metrics, player_metrics, info, out = _shaped("away_win")
    context = build_context(events, xg, team_metrics, player_metrics, info)
    assert context["winner"] == str(info["away_name"]), context["winner"]
    text = " ".join(write(b, context) for b in _boards(out) for write in WRITERS)
    # The loser must not be described as protecting a lead.
    assert f"{info['home_name']} protected a lead" not in text
