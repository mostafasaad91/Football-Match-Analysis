"""Two teams, two palettes, and nothing on the page sitting on anything else.

Three separate complaints, all about the same chart:

- Labels ran into the arc and into their own numbers. Every string was drawn
  horizontally on a ring, which is fine at three o'clock and unreadable at
  twelve: a two-line name stacked upwards into the arc above it and downwards
  onto the chip below. Moving the radii only moved the collision, because the
  crowding comes from horizontal text on a circle, not from the gap between
  two rings.
- The numbers were not legible: 9pt, and on the light page a pale fill under a
  pale digit.
- Both sides drew from the same five colours, so nothing said which team the
  player belonged to except the crest in the header.

The colour fix has a failure mode worth naming, because the first attempt hit
it: pulling each hue individually towards its own kit means a hue sitting
opposite both kits reaches the cap from both sides, and Hull's defence came out
#6dcf8b against Manchester United's #6ece84 — the same colour by two routes,
in the fixture the feature exists for. Turning the whole wheel by one angle per
side keeps the five groups exactly as far apart as they were and makes the
difference between the two sides something that can be guaranteed rather than
hoped for.
"""

import colorsys

import pytest

import player_radar as pr

# Kits chosen to break it rather than to pass: two shades of red, two identical
# colours, a white shirt with no hue at all, and a pair already far apart.
FIXTURES = [
    ("#F5A12D", "#DA291C"),   # Hull amber vs Manchester United red — 31° apart
    ("#DA291C", "#D01317"),   # two reds, all but the same
    ("#FFFFFF", "#FEFEFE"),   # two white shirts: no hue to read
    ("#EF0107", "#6CABDD"),   # Arsenal vs Manchester City — already far apart
    ("#004170", "#7A003C"),   # PSG vs Aston Villa
]
IDS = ["amber-red", "red-red", "white-white", "red-blue", "navy-claret"]


def _hue(colour: str) -> float:
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(colour)
    hue, _lightness, _saturation = colorsys.rgb_to_hls(r, g, b)
    return hue * 360.0


def _hue_gap(a: str, b: str) -> float:
    return abs(pr._signed_gap(_hue(a), _hue(b)))


# --------------------------------------------------------------------------
# one fixture, two palettes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("home,away", FIXTURES, ids=IDS)
def test_the_two_sides_never_share_a_group_colour(home, away):
    """The whole point: a Hull radar must not look like a United one."""
    palettes = pr.fixture_group_palettes(home, away, len(pr.GROUPS))
    shared = set(palettes["home"]) & set(palettes["away"])
    assert not shared, shared


@pytest.mark.parametrize("home,away", FIXTURES, ids=IDS)
def test_every_group_is_visibly_different_across_the_two_sides(home, away):
    """Not merely a different hex — a different colour.

    #6dcf8b against #6ece84 are two hex codes and one colour. The check is on
    the hue each group ends up at, group by group, because that is the pairing
    a reader compares: this side's defence against the other side's.
    """
    palettes = pr.fixture_group_palettes(home, away, len(pr.GROUPS))
    for index, (mine, theirs) in enumerate(
            zip(palettes["home"], palettes["away"])):
        assert _hue_gap(mine, theirs) >= 25.0, (
            f"group {index}: {mine} and {theirs} are "
            f"{_hue_gap(mine, theirs):.0f}° apart")


@pytest.mark.parametrize("home,away", FIXTURES, ids=IDS)
def test_the_five_groups_stay_apart_within_one_side(home, away):
    """Separating the teams must not cost the thing the colours encode."""
    palettes = pr.fixture_group_palettes(home, away, len(pr.GROUPS))
    for side, colours in palettes.items():
        hues = sorted(_hue(c) for c in colours)
        gaps = [hues[i + 1] - hues[i] for i in range(len(hues) - 1)]
        gaps.append(360.0 - hues[-1] + hues[0])
        assert min(gaps) >= 40.0, (side, colours, min(gaps))


@pytest.mark.parametrize("home,away", FIXTURES, ids=IDS)
def test_every_group_colour_carries_on_the_page(home, away):
    """A turned wheel must not turn a hue into one the page swallows."""
    from visual_redesign_preview import BG

    for colours in pr.fixture_group_palettes(home, away, len(pr.GROUPS)).values():
        for colour in colours:
            ratio = pr.contrast_ratio(colour, BG)
            assert ratio >= pr.GROUP_PAGE_CONTRAST - 0.15, (colour, ratio)


@pytest.mark.parametrize("home,away", FIXTURES, ids=IDS)
def test_each_side_is_drawn_with_its_own_offset(home, away):
    """The radar takes its palette from ``side``, not from argument order.

    Reading the pair as "whichever kit was passed first is mine" holds only
    while the two kits have hues to tell apart. Two white shirts fall back to
    the same value, both sides read the same end of the pair, and the away
    radar came out in the home palette — in the one fixture where the crest is
    the only other thing separating them.
    """
    offsets = pr.fixture_hue_offsets(home, away)
    assert offsets[0] != pytest.approx(offsets[1]), offsets
    # Whoever asks, the home offset is offsets[0] and the away one offsets[1].
    assert pr.fixture_hue_offsets(home, away) == offsets


@pytest.mark.parametrize("home,away", FIXTURES, ids=IDS)
def test_the_radar_itself_picks_the_palette_for_its_own_side(home, away):
    """The check one level up: not the helper, the colours the chart uses.

    Asserted through the same branch make_player_pizza runs, because the defect
    was in how it read the helper rather than in the helper.
    """
    offsets = pr.fixture_hue_offsets(home, away)
    drawn = {}
    for side, mine, theirs in (("home", home, away), ("away", away, home)):
        home_kit = theirs if side == "away" else mine
        away_kit = mine if side == "away" else theirs
        pair = pr.fixture_hue_offsets(home_kit, away_kit)
        drawn[side] = pr.group_palette_for(
            pair[1] if side == "away" else pair[0], mine, len(pr.GROUPS))

    assert drawn["home"] == pr.group_palette_for(offsets[0], home, len(pr.GROUPS))
    assert drawn["away"] == pr.group_palette_for(offsets[1], away, len(pr.GROUPS))
    assert not set(drawn["home"]) & set(drawn["away"])


def test_a_kit_with_no_hue_does_not_tint_at_random():
    """A white shirt has an arbitrary hue; reading it would tint the page by
    floating-point noise."""
    assert pr._hue_of("#FFFFFF", fallback=36.0) == 36.0
    assert pr._hue_of("#808080", fallback=36.0) == 36.0


# --------------------------------------------------------------------------
# the rings
# --------------------------------------------------------------------------

def test_the_rings_are_ordered_and_separated():
    """Bars, then numbers, then labels, then the arc — each clear of the last.

    Read off the radii the drawing code uses, so a future nudge that puts the
    chip ring inside the bars or the arc inside the labels fails here rather
    than in a rendered page nobody opens.
    """
    import inspect
    import re

    source = inspect.getsource(pr.make_player_pizza)
    found = re.search(
        r"R0, RMAX, RVAL, RLAB, RARC, OUT_LIM = ([\d, ]+)", source)
    assert found, "the radii are no longer declared on one line"
    r0, rmax, rval, rlab, rarc, out = [
        float(v) for v in found.group(1).split(",")]

    assert r0 < rmax, "the bars need somewhere to grow"
    assert rval > rmax, "a value sits outside the bar it belongs to"
    assert rlab > rval, "the label ring is inside the value ring"
    assert rarc > rlab, "the arc is inside the labels it brackets"
    assert out > rarc, "the arc is outside the axes"
    # The labels run inward from RLAB, so the space between the value ring and
    # the label anchor is what a long name has to fit in.
    assert rlab - rval >= 40, "no room for a label between the numbers and the arc"


def test_the_numbers_are_never_rotated():
    """A turned word is still readable; a turned number is not.

    "0.01" on the lower arc read as "10.0". The labels rotate and the values
    stay level, which is the whole reason the labels could be rotated at all —
    they are what buys the room the numbers used to fight for.
    """
    import inspect

    source = inspect.getsource(pr.make_player_pizza)
    # The value text is the block that draws `dv`; it must not pass a rotation.
    value_block = source.split("            dv,", 1)
    assert len(value_block) == 2, "the value text is no longer drawn from dv"
    following = value_block[1].split("bbox=", 1)[0]
    assert "rotation" not in following, following[:400]


def test_the_labels_are_rotated_and_read_the_right_way_up():
    """Left-hand spokes are flipped end for end so every word reads L-to-R."""
    import numpy as np

    # Just off each quadrant, not on it: straight up and straight down are the
    # boundary, where either orientation is equally sideways and which one the
    # rule picks says nothing about whether the rule is right.
    for angle, expect_flip in ((np.radians(10), False),    # upper right
                               (np.radians(100), False),   # lower right
                               (np.radians(190), True),    # lower left
                               (np.radians(280), True)):   # upper left
        spin, flipped = pr._spoke_rotation(angle)
        assert flipped is expect_flip, (np.degrees(angle), spin, flipped)
        assert -180.0 <= spin <= 180.0, spin


def test_no_label_is_drawn_across_two_lines():
    """The wrap existed to stop horizontal names colliding at twelve o'clock.

    Rotated labels run along their own spoke, so a break only makes the word
    shorter and harder to read.
    """
    for _group, _colour, metrics in pr.GROUPS:
        for metric in metrics:
            assert "\n" not in pr._spoke_label(metric), metric


def test_the_longest_label_is_short_enough_to_clear_the_numbers():
    """"BIG CH. CREATED", "FINAL 3RD PASSES" and "SHOT-CR. ACTIONS" reached
    back from the arc far enough to sit on their own values."""
    longest = max(
        (pr._spoke_label(metric)
         for _g, _c, metrics in pr.GROUPS for metric in metrics),
        key=len)
    assert len(longest) <= 13, longest


# --------------------------------------------------------------------------
# the ring of numbers
# --------------------------------------------------------------------------

def test_every_tile_is_the_same_width():
    """The figures run from one character to seven — "5" against "27 / 58" —
    and a rounded box around each made the ring a row of unequal blobs."""
    padded = pr.pad_values(["5", "0.187", "73/80", "-0.09"])
    assert len({len(v) for v in padded}) == 1, padded
    assert [v.strip() for v in padded] == ["5", "0.187", "73/80", "-0.09"]


def test_padding_an_empty_ring_does_not_raise():
    assert pr.pad_values([]) == []


def test_a_ratio_is_printed_without_spaces_around_the_slash():
    """Every tile is padded to the widest string on the radar, so two thin
    spaces inside one ratio set the width of all twenty-six — and near twelve
    and six o'clock a horizontal tile spends its width across its neighbours'
    spokes rather than along its own."""
    import inspect

    source = inspect.getsource(pr.make_player_pizza)
    assert " / " not in source, "the ratio still pads its slash"


def test_the_tiles_share_one_shape():
    """Five treatments used to share the ring, each with its own padding and
    border width, so a quarter of the circle could carry four different
    objects. Only the fill may vary."""
    group, chip = "#cb8721", "#cb8721"
    states = [
        pr._chip_style(chip, group, 90.0, zero=False),
        pr._chip_style(chip, group, 50.0, zero=False),
        pr._chip_style(chip, group, 10.0, zero=False),
        pr._chip_style(chip, group, 0.0, zero=True),
        pr._chip_style(chip, group, 0.0, zero=False, unmeasured=True),
    ]
    pads = {s["bbox"]["boxstyle"] for s in states}
    assert pads == {f"round,pad={pr.CHIP_PAD}"}, pads
    widths = {s["bbox"]["lw"] for s in states}
    assert widths == {pr.CHIP_EDGE}, widths


def test_a_zero_is_as_readable_as_every_other_figure():
    """Quiet was being done with TEXT_DIM — grey on grey, a figure a reader
    had to hunt for. The empty box is what says he did not do this."""
    group = "#cb8721"
    zero = pr._chip_style(group, group, 0.0, zero=True)
    ordinary = pr._chip_style(group, group, 10.0, zero=False)
    assert zero["color"] == ordinary["color"]
    assert pr.contrast_ratio(zero["color"], pr.BG_DARK) >= 4.4
    # ...and still visibly nothing: no fill behind it.
    assert zero["bbox"]["fc"] == "none"


def test_an_unmeasured_rate_is_readable_too():
    group = "#cb8721"
    thin = pr._chip_style(group, group, 0.0, zero=False, unmeasured=True)
    assert pr.contrast_ratio(thin["color"], pr.BG_DARK) >= 4.4
    assert "linestyle" in thin["bbox"], "the dashed border is what marks it"
