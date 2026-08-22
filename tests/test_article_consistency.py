"""A sentence must agree with the numbers standing next to it.

test_article_numbers.py asks where each figure came from. That is a different
question from whether the sentence around it is true, and the article shipped
five sentences that passed the first check and failed the second:

- "Man City shot more often, 9 to 12" — the leader was named correctly and the
  pair was printed in fixed home-then-away order.
- "On volume there was little in it" — printed even when one side had out-shot
  the other by a third, contradicting the clause after the dash.
- "The same asymmetry runs through the regains. Arsenal turned 4.4% … Man City
  4.5%" — the second number is larger, so it is not the same asymmetry.
- "And it was paid for: 4.5% against 4.4%" — a tenth of a point framed as a
  return, directly after a paragraph saying pressing leads often buy nothing.
- "PPDA read 0.00 for Arsenal and 0.00 for Man City" — PPDA is absent from the
  export and a zero default was printed as a measurement. A side allowing no
  passes per defensive action has not played.

Each test reads the finished prose and re-derives the comparison from the frames
rather than trusting the generator's own branch.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from match_article import build_article
from conftest import match_dir

ROOT = Path(__file__).resolve().parent.parent
MATCHES = ["Arsenal_vs_Man_City_3-0", "PSG_vs_Aston_Villa_2-1"]


def _built(match):
    out = match_dir(match)
    if not (out / "match_info.json").exists():
        pytest.skip(f"{match} has not been rendered")
    info = json.loads((out / "match_info.json").read_text(encoding="utf-8"))
    frames = (
        pd.read_csv(out / "events.csv"),
        pd.read_csv(out / "xg.csv"),
        pd.read_csv(out / "team_advanced_metrics.csv"),
        pd.read_csv(out / "player_sequence_metrics.csv"),
    )
    article = build_article(*frames, info, out)
    return article, frames, info


def _prose(article) -> str:
    return " ".join(p for s in article.sections for p in s.paragraphs)


def _sides(team_metrics, info):
    """(home row, away row) from the metrics frame."""
    by_side = team_metrics.set_index("side") if "side" in team_metrics else None
    if by_side is not None and {"home", "away"} <= set(by_side.index):
        return by_side.loc["home"], by_side.loc["away"]
    keyed = team_metrics.set_index("team")
    return keyed.loc[info["home_name"]], keyed.loc[info["away_name"]]


@pytest.mark.parametrize("match", MATCHES)
def test_the_side_named_as_shooting_more_carries_the_larger_count(match):
    article, (_, xg, _, _), info = _built(match)
    text = _prose(article)
    names = [str(info["home_name"]), str(info["away_name"])]
    pattern = "|".join(re.escape(n) for n in names)
    found = re.search(rf"({pattern}) shot more often, (\d+) to (\d+)", text)
    if not found:
        pytest.skip("this fixture's shooting was level")

    named, first, second = found.group(1), int(found.group(2)), int(found.group(3))
    assert first > second, f"'{found.group(0)}' prints the smaller count first"

    shots = {str(r["team"]): float(r["shots"]) for _, r in xg.iterrows()}
    assert shots[named] == first, (named, first, shots)
    assert shots[named] == max(shots.values()), (named, shots)


@pytest.mark.parametrize("match", MATCHES)
def test_little_in_it_is_only_claimed_when_the_counts_are_close(match):
    article, (_, xg, _, _), _ = _built(match)
    if "On volume there was little in it" not in _prose(article):
        return
    counts = sorted(float(r["shots"]) for _, r in xg.iterrows())
    assert counts[-1] - counts[0] <= 2, (
        f"'little in it' claimed over {counts[0]:.0f} and {counts[-1]:.0f} shots")


@pytest.mark.parametrize("match", MATCHES)
def test_the_same_asymmetry_is_only_claimed_when_it_is_the_same(match):
    """The transition leader must also lead the regain rate to claim both."""
    article, (_, _, team_metrics, _), info = _built(match)
    text = _prose(article)
    if "The same asymmetry runs through the regains" not in text:
        return
    home, away = _sides(team_metrics, info)
    found = re.search(r"asymmetry runs through the regains\. (.+?) turned "
                      r"([\d.]+)% of possession regains", text)
    assert found, text
    named, printed = found.group(1), float(found.group(2))
    rates = {str(home["team"]): float(home["regain_to_shot_rate"]),
             str(away["team"]): float(away["regain_to_shot_rate"])}
    assert rates[named] == printed, (named, printed, rates)
    assert printed == max(rates.values()), (
        f"'the same asymmetry' names {named} at {printed}, but "
        f"{max(rates, key=rates.get)} leads on {max(rates.values())}")


@pytest.mark.parametrize("match", MATCHES)
def test_a_press_is_only_paid_for_when_the_rates_actually_differ(match):
    article, (_, _, team_metrics, _), info = _built(match)
    if "And it was paid for" not in _prose(article):
        return
    home, away = _sides(team_metrics, info)
    rates = sorted((float(home["regain_to_shot_rate"]),
                    float(away["regain_to_shot_rate"])))
    assert rates[1] - rates[0] > 0.5, (
        f"'paid for' claimed over {rates[0]:.1f}% and {rates[1]:.1f}%")


@pytest.mark.parametrize("match", MATCHES)
def test_ppda_is_printed_only_when_the_export_carries_it(match):
    """An absent metric must be silent, not zero."""
    article, (_, _, team_metrics, _), _ = _built(match)
    text = _prose(article)
    if "PPDA read" not in text:
        return
    assert "ppda" in {c.lower() for c in team_metrics.columns}, (
        "the article printed a PPDA reading the export does not contain")
    for figure in re.findall(r"PPDA read ([\d.]+)[^.]*?([\d.]+)", text)[0]:
        assert float(figure) > 0, f"PPDA of {figure} is not a possible reading"


@pytest.mark.parametrize("match", MATCHES)
def test_a_goal_inside_the_first_minute_is_described_in_seconds(match):
    """"Minute 0" is not how anyone reports a goal from the kick-off."""
    article, (events, _, _, _), _ = _built(match)
    text = _prose(article)
    assert "in minute 0" not in text, text
    goals = events[events["is_goal"].astype(str).str.lower().eq("true")]
    if goals.empty:
        return
    first = goals.sort_values(["minute", "second"], kind="stable").iloc[0]
    if int(float(first["minute"])) == 0 and "The first goal arrived" in text:
        assert re.search(r"The first goal arrived (after \d+ seconds|from the kick-off)",
                         text), text


@pytest.mark.parametrize("match", MATCHES)
def test_no_paragraph_asserts_a_difference_and_then_prints_a_tie(match):
    """A sentence claiming a gap must not be followed by two equal figures.

    The generic form of every defect above: the branch chose the wording, the
    f-string printed the numbers, and nothing compared the two.
    """
    article, _, _ = _built(match)
    claims = ("the same asymmetry", "and it was paid for", "it was not paid for")
    for section in article.sections:
        for paragraph in section.paragraphs:
            lowered = paragraph.lower()
            if not any(claim in lowered for claim in claims):
                continue
            percentages = [float(v) for v in
                           re.findall(r"([\d.]+)% of (?:possession )?regains", paragraph)]
            if len(percentages) == 2:
                assert abs(percentages[0] - percentages[1]) > 0.5, (
                    f"{section.heading}: claims a difference, prints "
                    f"{percentages[0]} and {percentages[1]}")


# --------------------------------------------------------------------------
# the finishing paragraph
# --------------------------------------------------------------------------

@pytest.mark.parametrize("match", MATCHES)
def test_the_finishing_verdict_does_not_contradict_itself(match):
    """One article said both things about the same 0.04.

    "Three goals came from 2.96 combined expected goals, so finishing ran ahead
    of the chances" and, four lines later, "Conversion tracked the chances
    closely here — 3 from 2.96". The first sentence had no tolerance at all,
    the paragraph under it used 0.8, and a gap of four hundredths satisfied
    both.
    """
    article, _frames, _info = _built(match)
    text = _prose(article)
    ran = any(phrase in text for phrase in
              ("finishing ran ahead of the chances", "finishing fell short of the chances"))
    tracked = "tell the same story" in text or "tracked the chances closely" in text
    assert not (ran and tracked), text[:400]


@pytest.mark.parametrize("match", MATCHES)
def test_the_finishing_verdict_matches_the_gap(match):
    """Whichever sentence appears, the numbers have to support it."""
    article, (_events, xg, _tm, _pm), _info = _built(match)
    text = _prose(article)
    goals = float(pd.to_numeric(xg["goals"], errors="coerce").fillna(0).sum())
    expected = float(pd.to_numeric(xg["xG"], errors="coerce").fillna(0).sum())
    gap = goals - expected

    if "finishing ran ahead of the chances" in text:
        assert gap > 0.8, gap
    if "finishing fell short of the chances" in text:
        assert gap < -0.8, gap
    if "tell the same story" in text:
        assert abs(gap) <= 0.8, gap


@pytest.mark.parametrize("match", MATCHES)
def test_a_match_that_converted_what_it_created_is_not_warned_about(match):
    """A caveat about a small sample, where nothing deviated, reads as a hedge
    against the players rather than a note about the data."""
    article, (_events, xg, _tm, _pm), _info = _built(match)
    goals = float(pd.to_numeric(xg["goals"], errors="coerce").fillna(0).sum())
    expected = float(pd.to_numeric(xg["xG"], errors="coerce").fillna(0).sum())
    if abs(goals - expected) > 0.8:
        return
    text = _prose(article)
    assert "warning about the sample" not in text, text[:400]
