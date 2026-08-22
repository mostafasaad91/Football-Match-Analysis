"""A club is drawn in the colour it plays in.

Two defects, found on one cover.

The palette table covers the top five leagues. Every other club fell through to
a deterministic colour picked from a general pool — stable across runs, and
unrelated to the kit. Hull City, who play in amber and black, were drawn in
rose on every visual, every poster and the cover.

And a bright kit was mistaken for a white one. Both near-white tests measured
luminance alone, so Watford's #FBEE23 at 0.818 was replaced with a silver that
had *less* contrast against Southampton's red (4.01) than the yellow it
replaced (4.29). White is colourless; a saturated yellow is not near it.
"""

import colorsys

import matplotlib.colors as mcolors
import pytest

import football_match_analysis as fa
from visualization_components import contrast_ratio

# Clubs the collected fixtures have met that sit outside the top five leagues.
OUTSIDE_THE_TOP_FIVE = [
    "Hull", "Coventry", "Leicester", "Southampton", "Watford", "Blackburn",
    "Bolton", "Middlesbrough", "Preston", "Portsmouth", "QPR", "Notts Co.",
    "Lincoln City", "Malaga", "Casa Pia AC",
]


def _saturation(colour: str) -> float:
    return colorsys.rgb_to_hls(*mcolors.to_rgb(colour))[2]


@pytest.mark.parametrize("team", OUTSIDE_THE_TOP_FIVE)
def test_a_club_outside_the_big_leagues_has_its_own_kit(team):
    palette = fa._team_palette(team, "#888888")
    assert len(palette) > 1, (
        f"{team} falls through to a colour picked from a general pool")


def test_hull_is_amber():
    """The colour that started this: rose on every board for an amber club."""
    assert fa._team_palette("Hull", "#888888")[0].upper() == "#F5A12D"
    home, _away = fa.choose_matchup_colors("Hull", "Man Utd", "home", "auto")
    assert home.upper() == "#F5A12D"


@pytest.mark.parametrize("colour", ["#FBEE23", "#FFF200", "#FDBE11", "#FFC23C"])
def test_a_bright_saturated_kit_survives_the_near_white_test(colour):
    """The docstring promised yellows were kept; the test measured brightness."""
    assert _saturation(colour) >= 0.45, colour
    assert fa._visible_on_dark("Anyone", colour, "#B91C1C").upper() == colour.upper()


@pytest.mark.parametrize("colour", ["#FFFFFF", "#F5F5F5", "#FAFAFA", "#EFEFEF"])
def test_an_actually_white_kit_is_still_replaced(colour):
    """The fix must not let white through: it owns the pitch markings."""
    assert _saturation(colour) < 0.45, colour
    assert fa._visible_on_dark("Anyone", colour, "#B91C1C").upper() != colour.upper()


def test_watford_keeps_the_yellow_that_reads_better():
    home, away = fa.choose_matchup_colors("Watford", "Southampton", "home", "auto")
    assert home.upper() == "#FBEE23"
    # The substitute was worse on the measure the substitution existed for.
    assert contrast_ratio(home, away) > contrast_ratio(fa.WHITE_KIT_SILVER, away)


def test_both_near_white_tests_agree():
    """The same rule was written twice and only one copy would have been fixed."""
    for colour in ("#FBEE23", "#FFFFFF", "#FDBE11", "#F5F5F5"):
        kept = fa._visible_on_dark("Anyone", colour, "#B91C1C").upper() == colour.upper()
        bright_and_plain = (fa._relative_luminance(colour) >= 0.82
                            and _saturation(colour) < 0.45)
        assert kept != bright_and_plain, colour
