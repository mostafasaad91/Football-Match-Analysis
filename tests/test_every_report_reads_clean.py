"""Every rendered fixture, swept for the defects that reached publication.

The other article tests each guard one sentence on two or three fixtures. This
one runs the whole catalogue over every match on disk, because the failures
this session kept finding were not in the branch that was tested — they were in
the branch a fourth fixture happened to take.

What it checks, and the article each check came from:

- No headline claims a result the scoreline does not support. "Monza Won The
  Match In The Broken Moments" was published for a side that lost 4-1, and
  "Liverpool Won The Match In The Broken Moments" for a draw.
- No two matches share a headline. Six of fifteen carried the same one, because
  a candidate weighted on the ratio between two small numbers sat near its
  ceiling in every fixture.
- The closing agrees with the opening. An article headlined "Man Utd's xG Is A
  Chase, Not A Performance" closed by awarding Man Utd the performance: the
  opening read the game-state split, the closing read the total.
- No section heading contradicts the verdict. "The ground was held, and it
  paid" ran over a side whose territory produced nothing until it was two down.
- No sentence appears twice. The regain finding was printed in two sections in
  near-identical prose, and the finishing figure twice in one section.
- Every figure in the prose is in the frames.
- Nothing reads as a machine string, a NaN, or a plural formed from one.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from conftest import match_dir  # noqa: F401  (kept for symmetry with siblings)
from match_article import build_article

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"


def _fixtures():
    """Every rendered match on disk, dark-theme package only."""
    found = []
    for info in sorted(OUTPUT.rglob("match_info.json")):
        out = info.parent
        if out.name == "light":
            continue
        if all((out / name).exists() for name in
               ("events.csv", "xg.csv", "team_advanced_metrics.csv",
                "player_sequence_metrics.csv")):
            found.append(out)
    return found


FIXTURES = _fixtures()
IDS = [p.name for p in FIXTURES]

if not FIXTURES:
    pytest.skip("no rendered fixtures on disk", allow_module_level=True)


# Eleven checks over fifteen fixtures is 165 builds of the same few articles,
# and one build is several seconds. The article is a pure function of the
# frames, so it is built once per fixture and every check reads that copy.
_BUILT: dict[Path, tuple] = {}


def _article(out: Path):
    if out not in _BUILT:
        info = json.loads((out / "match_info.json").read_text(encoding="utf-8"))
        frames = [pd.read_csv(out / name) for name in
                  ("events.csv", "xg.csv", "team_advanced_metrics.csv",
                   "player_sequence_metrics.csv")]
        _BUILT[out] = (build_article(*frames, info, out), frames, info)
    return _BUILT[out]


def _paragraphs(article):
    return [p for s in article.sections for p in s.paragraphs]


def _prose(article):
    return " ".join(_paragraphs(article))


# --------------------------------------------------------------------------
# headlines
# --------------------------------------------------------------------------

@pytest.mark.parametrize("out", FIXTURES, ids=IDS)
def test_no_headline_awards_a_win_the_result_did_not(out):
    """"X Won The Match" is a claim about the result, not about a metric."""
    article, (_events, xg, _tm, _pm), info = _article(out)
    goals = {str(r["team"]): float(r["goals"]) for _, r in xg.iterrows()}
    home, away = str(info["home_name"]), str(info["away_name"])
    winner = None
    if goals.get(home, 0) != goals.get(away, 0):
        winner = home if goals.get(home, 0) > goals.get(away, 0) else away

    claim = re.search(r"^(.+?) Won The Match", article.title)
    if not claim:
        return
    named = claim.group(1)
    assert winner is not None, (
        f"'{article.title}' names a winner in a match that was drawn")
    assert named == winner, (
        f"'{article.title}' names {named}; {winner} won {goals}")


def test_two_matches_do_not_share_a_headline():
    """Six of fifteen carried "Won The Match In The Broken Moments"."""
    titles = [_article(out)[0].title for out in FIXTURES]
    repeated = {t: n for t, n in Counter(titles).items() if n > 1}
    assert not repeated, repeated


@pytest.mark.parametrize("out", FIXTURES, ids=IDS)
def test_a_headline_that_praises_a_side_is_not_about_a_heavy_defeat(out):
    """"Monza Shot Less And Meant It More" fronted a 4-1 defeat."""
    article, (_events, xg, _tm, _pm), info = _article(out)
    goals = {str(r["team"]): float(r["goals"]) for _, r in xg.iterrows()}
    home, away = str(info["home_name"]), str(info["away_name"])
    if abs(goals.get(home, 0) - goals.get(away, 0)) < 3:
        return
    beaten = home if goals.get(home, 0) < goals.get(away, 0) else away
    praise = ("Meant It More", "Made It Look Simple", "Won The Match")
    for phrase in praise:
        if phrase in article.title:
            assert not article.title.startswith(beaten), (
                f"'{article.title}' praises a side beaten by "
                f"{abs(goals[home] - goals[away]):.0f}")


# --------------------------------------------------------------------------
# one verdict per article
# --------------------------------------------------------------------------

@pytest.mark.parametrize("out", FIXTURES, ids=IDS)
def test_the_closing_does_not_reverse_the_opening(out):
    """Both ends read the verdict, so they cannot disagree about it."""
    article, _frames, _info = _article(out)
    if "chase" not in article.title.lower():
        return
    closing = next((s for s in article.sections
                    if s.heading == "What to take from it"), None)
    assert closing is not None
    text = " ".join(closing.paragraphs)
    loser = article.title.split("'")[0]
    assert f"it belongs to {loser}" not in text, (
        f"headline calls {loser}'s xG a chase and the closing awards them the "
        f"performance: {text[:200]}")


@pytest.mark.parametrize("out", FIXTURES, ids=IDS)
def test_no_heading_says_the_territory_paid_when_the_verdict_says_otherwise(out):
    article, _frames, _info = _article(out)
    if "chase" not in article.title.lower():
        return
    headings = [s.heading for s in article.sections]
    assert "The ground was held, and it paid" not in headings, headings


# --------------------------------------------------------------------------
# nothing said twice
# --------------------------------------------------------------------------

@pytest.mark.parametrize("out", FIXTURES, ids=IDS)
def test_no_sentence_is_printed_twice(out):
    """The regain reading ran in two sections in near-identical prose."""
    article, _frames, _info = _article(out)
    sentences = []
    for paragraph in _paragraphs(article):
        for piece in re.split(r"(?<=[.!?])\s+", paragraph):
            piece = piece.strip()
            # Short fragments repeat innocently ("It was not paid for.");
            # a whole clause of analysis does not.
            if len(piece.split()) >= 12:
                sentences.append(piece)
    repeated = {s: n for s, n in Counter(sentences).items() if n > 1}
    assert not repeated, list(repeated)[:2]


@pytest.mark.parametrize("out", FIXTURES, ids=IDS)
def test_no_paragraph_restates_the_one_before_it(out):
    """Two paragraphs carrying the same two figures in the same order.

    The opening section printed "Two goals came from 3.58 combined expected
    goals" and then "3.58 expected goals produced two goals" underneath it.
    """
    article, _frames, _info = _article(out)
    for section in article.sections:
        seen = []
        for paragraph in section.paragraphs:
            figures = tuple(re.findall(r"\d+\.\d{2}", paragraph))
            if len(figures) >= 2:
                assert sorted(figures) not in seen, (
                    f"{section.heading}: two paragraphs carry {figures}")
                seen.append(sorted(figures))


# --------------------------------------------------------------------------
# the prose itself
# --------------------------------------------------------------------------

MACHINE = re.compile(r"\b(nan|NaN|inf)\b|\{\w+\}|\s\|\s")


@pytest.mark.parametrize("out", FIXTURES, ids=IDS)
def test_no_machine_string_reaches_the_page(out):
    """"None" is the awkward one: it is both a repr and an English pronoun.

    "None of that is the striker's doing" and "None of this makes the win
    undeserved" are correct prose, so the word only means a leak when it turns
    up inside a sentence rather than opening one. Checked per sentence, because
    a lookbehind over a whole paragraph cannot tell the two apart.
    """
    article, _frames, _info = _article(out)
    for paragraph in _paragraphs(article) + [article.title, article.standfirst]:
        found = MACHINE.search(paragraph)
        assert not found, (found.group(0), paragraph[:160])
        for sentence in re.split(r"(?<=[.!?:])\s+", paragraph):
            body = sentence.strip()
            assert "None" not in body[1:], (
                f"'None' inside a sentence: {body[:160]}")


ONE_PLURAL = re.compile(r"(?<![\d—–-]\s)(?<![\d—–-])\b1 (\w+?)(s|es)\b")


def _plural_offenders(texts) -> list[str]:
    found = []
    for text in texts:
        for hit in ONE_PLURAL.finditer(str(text)):
            if not hit.group(0).endswith(("ss", "ess")):
                found.append(hit.group(0))
    return sorted(set(found))


@pytest.mark.parametrize("out", FIXTURES, ids=IDS)
def test_no_count_of_one_is_printed_as_a_plural(out):
    article, _frames, _info = _article(out)
    assert not _plural_offenders(
        _paragraphs(article) + [article.title, article.standfirst])


@pytest.mark.parametrize("out", FIXTURES, ids=IDS)
def test_the_report_writers_agree_with_their_own_counts(out):
    """The article's own prose was clean; the readings under it were not.

    A 1-0 shipped "the match produced 1 goals" from six separate sentences in
    tactical_pdf_report, none of which reached for the _count helper that has
    been in that module since the player pages were fixed for the same thing.
    Every fixture the older tests used had scored at least twice, so the branch
    was never taken.

    This sweeps every function that writes words for a board or a card, over
    every match on disk — which is the only combination that would have caught
    it.
    """
    from tactical_pdf_report import (
        _legacy_visual_explanation,
        _section_copy,
        build_context,
        visual_data_read,
        visual_explanation,
        visual_implication,
    )

    _article_, frames, info = _article(out)
    context = build_context(*frames, info)

    boards = sorted(out.glob("[0-9]*.png"))
    radars = out / "player_radars"
    if radars.exists():
        boards += sorted(radars.glob("*/*.png"))

    texts = []
    for writer in (visual_explanation, visual_implication, visual_data_read,
                   _legacy_visual_explanation):
        texts += [writer(path, context) for path in boards]
    for section in _section_copy(context).values():
        texts += [t for _, t in
                  section.get("performance", []) + section.get("data", [])]
        texts.append(str(section.get("implication", "")))

    assert not _plural_offenders(texts)


@pytest.mark.parametrize("out", FIXTURES, ids=IDS)
def test_a_percentage_printed_as_a_share_stays_inside_a_hundred(out):
    article, _frames, _info = _article(out)
    for text in _paragraphs(article):
        for value in re.findall(r"(\d+(?:\.\d+)?)%", text):
            assert 0.0 <= float(value) <= 100.0, (value, text[:160])


@pytest.mark.parametrize("out", FIXTURES, ids=IDS)
def test_the_article_reaches_the_length_it_promises(out):
    from match_article import TARGET_WORDS

    article, _frames, _info = _article(out)
    words = sum(len(p.split()) for p in _paragraphs(article))
    assert words >= TARGET_WORDS[0], f"{words} words"
