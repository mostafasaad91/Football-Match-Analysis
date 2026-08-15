"""The fixture cluster in every visual's header: crest, name, score, name, crest.

The first version placed the crests at fixed coordinates and capped the names
at twelve characters. A character is not a width: "MAN CITY" measured 0.052 of
the figure at twelve characters and "BLACKBURN RO" 0.083, so the cap let the
long names run into the crest drawn beside them while leaving the short ones
adrift. Everything is measured now, and this checks it stays that way.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import visual_redesign_preview as base


FIGSIZE, DPI = (12, 9), 150

# Real fixtures, chosen for the lengths that broke the old placement.
PAIRS = [
    ("PSG", "Aston Villa"),
    ("Wolverhampton Wanderers", "Blackburn Rovers"),
    ("Manchester City", "Tottenham Hotspur"),
    ("Borussia Dortmund", "Brighton and Hove Albion"),
    ("Real Madrid", "Atletico Madrid"),
    ("A", "B"),
]


@pytest.fixture
def figure():
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.canvas.draw()  # a renderer has to exist before anything is measured
    yield fig
    plt.close(fig)


def _fractions(fig, artist):
    box = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    scale = fig.get_figwidth() * fig.dpi
    return box.x0 / scale, box.x1 / scale


def _draw(fig, home, away, score="2 — 1"):
    base.HOME_NAME, base.AWAY_NAME = home, away
    base.HOME_ID, base.AWAY_ID = 304, 24
    base.MATCH_SCORE = score
    base.fixture_cluster(fig, [])
    names = [_fractions(fig, item) for item in fig.texts]
    crests = [(ax.get_position().x0, ax.get_position().x1) for ax in fig.axes]
    return names, crests


@pytest.mark.parametrize("home,away", PAIRS)
def test_no_name_touches_a_crest(home, away, figure):
    names, crests = _draw(figure, home, away)
    assert crests, "no crest was drawn"
    for x0, x1 in names:
        for cx0, cx1 in crests:
            assert x1 <= cx0 or x0 >= cx1, (
                f"{home} v {away}: text {x0:.4f}-{x1:.4f} overlaps crest "
                f"{cx0:.4f}-{cx1:.4f}"
            )


@pytest.mark.parametrize("home,away", PAIRS)
def test_the_cluster_stays_inside_the_strip(home, away, figure):
    names, crests = _draw(figure, home, away)
    for x0, x1 in names + crests:
        assert x0 >= base.FIXTURE_LEFT - 1e-6, f"{home} v {away}: {x0:.4f} runs left"
        assert x1 <= base.FIXTURE_RIGHT + 1e-6, f"{home} v {away}: {x1:.4f} runs right"


@pytest.mark.parametrize("home,away", PAIRS)
def test_nothing_in_the_cluster_overlaps_anything_else(home, away, figure):
    names, _crests = _draw(figure, home, away)
    ordered = sorted(names)
    for (_a0, a1), (b0, _b1) in zip(ordered, ordered[1:]):
        assert b0 >= a1 - 1e-6, f"{home} v {away}: two labels overlap"


# --------------------------------------------------------------------------
# how a name shortens
# --------------------------------------------------------------------------

def _budget(fig):
    half = base._text_width(fig, "2 — 1", base.FIXTURE_SCORE_SIZE) / 2
    crest = base.FIXTURE_CREST_W + base.FIXTURE_GAP
    return (base.FIXTURE_SCORE_X - half - base.FIXTURE_GAP) - (base.FIXTURE_LEFT + crest)


def test_a_long_name_loses_its_tail_not_its_letters(figure):
    """Cutting to a width gave "TOTTENHAM HOTSP"; clubs do not shorten that way."""
    budget = _budget(figure)
    assert base._fit_name(figure, "Tottenham Hotspur", budget) == "TOTTENHAM"
    assert base._fit_name(figure, "Borussia Dortmund", budget) == "BORUSSIA"


def test_a_name_that_fits_is_left_whole(figure):
    budget = _budget(figure)
    assert base._fit_name(figure, "Aston Villa", budget) == "ASTON VILLA"
    assert base._fit_name(figure, "Manchester City", budget) == "MANCHESTER CITY"


def test_a_dropped_tail_never_leaves_a_dangling_joiner(figure):
    """"Brighton and Hove Albion" cut to "BRIGHTON AND", which reads unfinished."""
    assert base._fit_name(figure, "Brighton and Hove Albion", _budget(figure)) == "BRIGHTON"


def test_one_word_too_wide_is_cut_and_marked(figure):
    """A cut has to look like an abbreviation, not like a rendering fault."""
    label = base._fit_name(figure, "Wolverhampton", 0.04)
    assert label.endswith("…")
    assert base._text_width(figure, label, base.FIXTURE_NAME_SIZE) <= 0.04


def test_no_budget_yields_no_label_rather_than_a_stray_glyph(figure):
    assert base._fit_name(figure, "Aston Villa", 0.0) == ""


def test_the_measurement_survives_a_figure_with_no_renderer():
    """The estimate has to be wide enough to stay safe, never narrower."""
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    try:
        assert base._text_width(fig, "ASTON VILLA", base.FIXTURE_NAME_SIZE) > 0
    finally:
        plt.close(fig)
