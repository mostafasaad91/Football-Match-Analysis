"""The value chips on a player radar.

Each chip is an 8pt digit on a tile coloured from the club's own kit ramp, and
WCAG's luminance formula flatters saturated hues: red carries a coefficient of
0.2126, so a fully saturated red computes as "dark" and near-black on it scores
4.9:1 while reading as a smudge. Arsenal's #fe0107 was exactly that, and the
outline meant as a backstop never fired because 4.92 clears its 4.5 threshold.

The tile is now moved until its own best ink clears a floor well above the
nominal minimum, and the group's colour moves to the tile's border.
"""

import pytest

from player_radar import (
    CHIP_CONTRAST_FLOOR,
    CHIP_SEPARATION,
    GROUPS,
    _chip_fill,
    _chip_text_color,
    chip_fills,
    team_group_colors,
)
from visualization_components import contrast_ratio

# Real kits, chosen for the hues that break a luminance-only judgement.
KITS = {
    "Arsenal": "#EF0107",
    "Liverpool": "#C8102E",
    "Manchester City": "#6CABDD",
    "PSG": "#004170",
    "Juventus": "#DCE3EC",
    "Norwich": "#FFF200",
    "Borussia Dortmund": "#FDE100",
    "near-black kit": "#111111",
}


@pytest.mark.parametrize("team,kit", KITS.items())
def test_every_chip_carries_its_number(team, kit):
    for fill in chip_fills(team_group_colors(kit, len(GROUPS))):
        ratio = contrast_ratio(_chip_text_color(fill), fill)
        assert ratio >= CHIP_CONTRAST_FLOOR - 0.05, (
            f"{team}: {fill} gives its label only {ratio:.2f}"
        )


@pytest.mark.parametrize("team,kit", KITS.items())
def test_the_groups_stay_distinguishable(team, kit):
    """Fixing legibility alone sent three of Arsenal's five groups to #b60105."""
    chips = chip_fills(team_group_colors(kit, len(GROUPS)))
    assert len(set(chips)) == len(chips), f"{team}: duplicate chips {chips}"


def test_the_saturated_red_that_started_this_is_fixed():
    """Near-black on #fe0107 measured 4.92 and was unreadable at 8pt."""
    before = contrast_ratio(_chip_text_color("#fe0107"), "#fe0107")
    assert before < CHIP_CONTRAST_FLOOR
    after_fill = _chip_fill("#fe0107")
    assert contrast_ratio(_chip_text_color(after_fill), after_fill) >= CHIP_CONTRAST_FLOOR


def test_a_fill_that_already_reads_is_left_exactly_alone():
    """A pale tile with dark text is fine; moving it would only lose the ramp."""
    pale = "#ff999c"
    assert contrast_ratio(_chip_text_color(pale), pale) >= CHIP_CONTRAST_FLOOR
    assert _chip_fill(pale) == pale
    assert chip_fills([pale]) == [pale]


def test_a_light_fill_goes_lighter_and_a_dark_one_darker():
    """The ramp's shape survives: pale groups stay pale, deep ones stay deep."""
    from matplotlib import colors as mcolors

    def luminance(colour):
        return sum(mcolors.to_rgb(colour))

    light = "#fe676a"   # already light, just under the floor
    dark = "#cb0106"    # deep, just under the floor
    assert luminance(_chip_fill(light)) >= luminance(light)
    assert luminance(_chip_fill(dark)) <= luminance(dark)


def test_the_nudge_only_ever_raises_contrast():
    fill = "#fe0107"
    plain = _chip_fill(fill)
    nudged = _chip_fill(fill, nudge=CHIP_SEPARATION)
    assert (contrast_ratio(_chip_text_color(nudged), nudged)
            >= contrast_ratio(_chip_text_color(plain), plain))


def test_an_unparseable_colour_is_returned_untouched():
    assert _chip_fill("not a colour") == "not a colour"
    assert chip_fills(["not a colour"]) == ["not a colour"]
