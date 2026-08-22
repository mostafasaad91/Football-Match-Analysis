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

import player_radar as pr
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


# --------------------------------------------------------------------------
# how loudly a value is printed
# --------------------------------------------------------------------------

def test_a_tile_reading_zero_is_quiet():
    """A third of a defensive midfielder's radar is zeroes.

    Every tile carried the same saturated fill and the same bold digit, so a
    value the player never registered shouted as loudly as the best figure on
    the pitch and the eye had nothing to prioritise.
    """
    for value, shown in ((0, "0"), (0.0, "0.0"), (0.045, "0.0"),
                         (0, "0 / 2"), (0, "0 / 0")):
        assert pr._is_zero(value, shown), (value, shown)


def test_a_negative_value_is_not_a_zero():
    """Threat below zero is a finding, not an absence."""
    for value, shown in ((-0.071, "-0.071"), (-0.06, "-0.06")):
        assert not pr._is_zero(value, shown), (value, shown)


def test_a_ratio_with_a_scoring_numerator_is_not_a_zero():
    assert not pr._is_zero(2, "2 / 6")
    assert pr._is_zero(0, "0 / 6"), "none won from six contested is still none"


def test_the_loudest_tier_is_reserved_for_the_highest_percentiles():
    loud = pr._chip_style("#8c2b2b", "#8c2b2b", 92.0, zero=False)
    middle = pr._chip_style("#8c2b2b", "#8c2b2b", 50.0, zero=False)
    quiet = pr._chip_style("#8c2b2b", "#8c2b2b", 10.0, zero=False)

    assert loud["bbox"]["fc"] != "none", "a leading value must carry a filled tile"
    assert middle["bbox"]["fc"] != "none"
    assert quiet["bbox"]["fc"] == "none", "a trailing value must not"
    assert loud["weight"] == "bold" and quiet["weight"] == "normal"


def test_a_zero_carries_no_fill_whatever_its_percentile():
    for percentile in (0.0, 50.0, 99.0):
        style = pr._chip_style("#8c2b2b", "#8c2b2b", percentile, zero=True)
        assert style["bbox"]["fc"] == "none", percentile


def test_the_group_colours_are_calmer_than_the_kit():
    """Thirty wedges at shirt saturation is a page that shouts in one tone."""
    import colorsys

    import matplotlib.colors as mcolors

    kit = "#fe0107"                       # a fully saturated red
    _h, _l, kit_saturation = colorsys.rgb_to_hls(*mcolors.to_rgb(kit))
    for shade in pr.team_group_colors(kit, 5):
        _h2, _l2, shade_saturation = colorsys.rgb_to_hls(*mcolors.to_rgb(shade))
        assert shade_saturation < kit_saturation, (shade, shade_saturation)


def test_neighbouring_groups_do_not_land_on_the_same_colour():
    """DEFENCE and DUELS sit next to each other and looked like one group."""
    shades = pr.team_group_colors("#fe0107", 5)
    assert len(set(shades)) == 5, shades
    for first, second in zip(shades, shades[1:]):
        assert first != second


def test_every_group_colour_carries_its_weight_on_the_page():
    """Equal HLS lightness is not equal weight.

    Built at one lightness for every hue, the green measured 2.88 against the
    light page while the violet measured 6.95 — five colours meant as siblings
    arrived as two faint ones and three solid. Each hue is solved for contrast
    instead, which is what the eye is reading.
    """
    for shade in pr.group_palette(len(GROUPS)):
        assert contrast_ratio(shade, pr.BG_DARK) >= pr.GROUP_PAGE_CONTRAST - 0.05, (
            shade, contrast_ratio(shade, pr.BG_DARK))


def test_the_group_colours_share_one_saturation():
    """What makes five hues read as one family rather than five decisions."""
    import colorsys

    import matplotlib.colors as mcolors

    levels = {round(colorsys.rgb_to_hls(*mcolors.to_rgb(shade))[2], 2)
              for shade in pr.group_palette(len(GROUPS))}
    assert len(levels) == 1, levels


def test_the_palette_does_not_depend_on_the_kit():
    """Identity is the crest and the score; the wedges say attack or defence.

    Five steps of one team colour put ATTACK beside THREAT and DEFENCE beside
    DUELS in shades a reader had to compare side by side to separate.
    """
    assert pr.group_palette(5) == pr.group_palette(5)
    hues = set(pr.GROUP_HUES)
    assert len(hues) == 5, "two groups share a hue"
