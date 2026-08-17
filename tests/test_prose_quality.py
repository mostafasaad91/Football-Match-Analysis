"""No board may be explained with filler, and no sentence with a stray number.

Twenty-nine of the fifty-three visuals in each match reached the same two
closing sentences — "The tactical conclusion should be checked against score
state … the causal chain behind the pattern" — and the Word article printed
that under them too, because it asks the report for its readings.

Two separate causes, both of which these tests would have caught:

- ``_visual_team`` compared "man city" against "03_shot_map_man_city.png".
  A two-word side never matched, so all fourteen of its boards lost their
  branch, while a one-word side matched and kept it. The defect was invisible
  in any fixture where both names happened to be one word.
- Fifteen boards had no branch written for them at all.

The rest of the file guards the sentences that carry a number: a machine string
must not appear inside prose, a goal from the kick-off must not be reported as
"minute 0", and a card that names a leader must print that leader's figure.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from tactical_pdf_report import (
    _section_copy,
    _visual_team,
    build_context,
    visual_explanation,
    visual_implication,
)

ROOT = Path(__file__).resolve().parent.parent
MATCHES = ["Arsenal_vs_Man_City_3-0", "PSG_vs_Aston_Villa_2-1"]

GENERIC = "The deeper reading is the causal chain behind the pattern"


def _context(match):
    out = ROOT / "output" / match
    if not (out / "match_info.json").exists():
        pytest.skip(f"{match} has not been rendered")
    info = json.loads((out / "match_info.json").read_text(encoding="utf-8"))
    return build_context(
        pd.read_csv(out / "events.csv"),
        pd.read_csv(out / "xg.csv"),
        pd.read_csv(out / "team_advanced_metrics.csv"),
        pd.read_csv(out / "player_sequence_metrics.csv"),
        info,
    ), out


@pytest.mark.parametrize("match", MATCHES)
def test_every_visual_has_its_own_reading(match):
    context, out = _context(match)
    fell_back = [p.name for p in sorted(out.glob("[0-9]*.png"))
                 if GENERIC in visual_explanation(p, context)]
    assert not fell_back, (
        f"{len(fell_back)} boards carry the generic ending: {fell_back}")


@pytest.mark.parametrize("match", MATCHES)
def test_a_two_word_side_is_recognised_in_its_own_filenames(match):
    """The root cause, asserted directly rather than through its symptom."""
    context, out = _context(match)
    for side in ("home", "away"):
        team = str(context[side])
        slug = re.sub(r"[^a-z0-9]+", "_", team.lower()).strip("_")
        named = [p for p in out.glob(f"*{slug}*.png") if "player_radars" not in p.parts]
        if not named:
            continue
        found, found_side = _visual_team(named[0], context)
        assert found == team and found_side == side, (named[0].name, team, found)


@pytest.mark.parametrize("match", MATCHES)
def test_no_explanation_repeats_another_boards_wording_verbatim(match):
    """Two boards may share a template; they must not share a whole paragraph.

    A shared template fills in different teams and different numbers, so an
    identical string means the fill-in did not happen.
    """
    context, out = _context(match)
    seen: dict[str, str] = {}
    duplicates = []
    for path in sorted(out.glob("[0-9]*.png")):
        text = visual_explanation(path, context)
        if text in seen:
            duplicates.append((seen[text], path.name))
        seen[text] = path.name
    assert not duplicates, duplicates


@pytest.mark.parametrize("match", MATCHES)
def test_every_visual_has_its_own_coaching_note(match):
    """The implication under each board had the same fifteen-way fallback.

    A third of the report ended on "Translate the pattern into a coachable
    behaviour", which tells a coach nothing about the board above it.
    """
    context, out = _context(match)
    notes: dict[str, list[str]] = {}
    for path in sorted(out.glob("[0-9]*.png")):
        notes.setdefault(visual_implication(path, context), []).append(path.name)
    shared = {note: names for note, names in notes.items() if len(names) > 2}
    assert not shared, (
        "one coaching note is repeated across unrelated boards: "
        + "; ".join(f"{names}" for names in shared.values()))


@pytest.mark.parametrize("match", MATCHES)
def test_no_machine_string_leaks_into_prose(match):
    """Pipe-delimited timelines and raw dict text are not sentences."""
    context, out = _context(match)
    for path in sorted(out.glob("[0-9]*.png")):
        text = visual_explanation(path, context)
        assert " | " not in text, (path.name, text[:160])
        assert "'" + "," not in text.replace("'s", ""), (path.name, text[:160])
        assert not re.search(r"\{\w+\}|nan\b|None\b", text), (path.name, text[:160])


@pytest.mark.parametrize("match", MATCHES)
def test_no_goal_is_reported_as_minute_zero(match):
    """Opta's opening minute is 0; nobody says a goal arrived in minute 0."""
    context, out = _context(match)
    prose = " ".join(visual_explanation(p, context)
                     for p in sorted(out.glob("[0-9]*.png")))
    for section in _section_copy(context).values():
        for group in ("performance", "data"):
            prose += " " + " ".join(t for _, t in section.get(group, []))
        prose += " " + str(section.get("implication", ""))
    assert "minute 0" not in prose, prose[prose.find("minute 0") - 120:][:240]
    assert "0'" not in prose


@pytest.mark.parametrize("match", MATCHES)
def test_the_shot_volume_card_agrees_with_the_counts(match):
    """A card naming the side that shot more must print its number first."""
    context, _ = _context(match)
    cards = dict(_section_copy(context)["Match Story"]["data"])
    heading = next(h for h in cards if "hot volume" in h or "shot more often" in h)
    home, away = int(context["home_shots"]), int(context["away_shots"])
    gap = abs(home - away)
    if "shot more often" in heading:
        leader = context["home"] if home > away else context["away"]
        assert heading.startswith(leader), (heading, home, away)
        assert cards[heading].startswith(leader), cards[heading]
        first = int(re.search(r"(\d+)", cards[heading]).group(1))
        assert first == max(home, away), (cards[heading], home, away)
    else:
        assert gap <= 2 or gap / max(home, away) <= 0.15, (heading, home, away)


@pytest.mark.parametrize("match", MATCHES)
def test_the_match_story_does_not_assert_swings_that_did_not_happen(match):
    """"The game never settled" was printed over every result, settled or not."""
    context, _ = _context(match)
    cards = dict(_section_copy(context)["Match Story"]["performance"])
    if "The lead changed hands" not in cards:
        return
    scorers, running, changes, ahead = set(), {}, 0, None
    for row in context["goal_rows"]:
        scorers.add(row["team"])
        running[row["team"]] = running.get(row["team"], 0) + 1
        counts = [running.get(context["home"], 0), running.get(context["away"], 0)]
        now = None if counts[0] == counts[1] else (
            context["home"] if counts[0] > counts[1] else context["away"])
        if now is not None and ahead is not None and now != ahead:
            changes += 1
        ahead = now
    assert changes > 0, "claims the lead changed hands in a match where it did not"


@pytest.mark.parametrize("match", MATCHES)
def test_the_player_cards_read_as_english(match):
    context, _ = _context(match)
    cards = _section_copy(context)["Player Impact Appendix"]
    text = " ".join(t for _, t in cards["performance"] + cards["data"])
    assert "1 goals" not in text, text
    assert "No qualifying player" not in text, text
