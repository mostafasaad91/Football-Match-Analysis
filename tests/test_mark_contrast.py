"""Marks drawn on the pitch have to survive the pure-black ground.

Real kit colours are frequently dark. Drawn unmodified they measured under
2:1 — PSG's #004170 at 1.99, Aston Villa's #7A003C at 1.89 — which is why
arrows, network links and thin strokes read as muted rather than as the team.
"""

import colorsys

import pytest
from matplotlib import colors as mcolors

from visual_redesign_full import (
    MARK_CONTRAST_FLOOR,
    _contrast_on_bg,
    lift_to_floor,
)

# Real kits that measure below the floor on black.
DARK_KITS = {
    "PSG navy": "#004170",
    "Aston Villa claret": "#7A003C",
    "Boca navy": "#1E4B9B",
    "Burnley claret": "#6C1D45",
    "near-black": "#111111",
}

# Real kits that already clear it and must be left alone.
BRIGHT_KITS = {
    "Juventus silver": "#DCE3EC",
    "Man City sky": "#6CABDD",
    "Bournemouth red": "#DA291C",
    "Bayern red": "#DC052D",
}


def test_the_floor_is_above_the_graphical_minimum():
    """WCAG puts non-text graphics at 3:1; a thin arrow needs more."""
    assert MARK_CONTRAST_FLOOR >= 3.0


@pytest.mark.parametrize(("name", "colour"), DARK_KITS.items())
def test_a_dark_kit_is_lifted_to_the_floor(name, colour):
    before = _contrast_on_bg(mcolors.to_rgb(colour))
    assert before < MARK_CONTRAST_FLOOR, f"{name} was not a dark kit to begin with"
    after = _contrast_on_bg(mcolors.to_rgb(lift_to_floor(colour)))
    assert after >= MARK_CONTRAST_FLOOR - 0.05, f"{name}: {after:.2f} still under the floor"


@pytest.mark.parametrize(("name", "colour"), BRIGHT_KITS.items())
def test_a_kit_that_already_reads_is_left_alone(name, colour):
    assert lift_to_floor(colour).lower() == colour.lower(), f"{name} was altered needlessly"


@pytest.mark.parametrize(("name", "colour"), DARK_KITS.items())
def test_lifting_preserves_the_hue(name, colour):
    """A lifted claret must still be claret. Changing the hue would swap the
    team's identity for legibility instead of reconciling them."""
    if colour == "#111111":
        pytest.skip("an achromatic kit has no hue to preserve")
    before = colorsys.rgb_to_hls(*mcolors.to_rgb(colour))
    after = colorsys.rgb_to_hls(*mcolors.to_rgb(lift_to_floor(colour)))
    gap = abs(before[0] - after[0])
    assert min(gap, 1 - gap) < 0.02, f"{name}: hue moved from {before[0]:.3f} to {after[0]:.3f}"


@pytest.mark.parametrize(("name", "colour"), DARK_KITS.items())
def test_lifting_does_not_wash_the_colour_out(name, colour):
    if colour == "#111111":
        pytest.skip("an achromatic kit has no saturation to keep")
    before = colorsys.rgb_to_hls(*mcolors.to_rgb(colour))
    after = colorsys.rgb_to_hls(*mcolors.to_rgb(lift_to_floor(colour)))
    assert after[2] >= before[2] * 0.75, f"{name}: saturation fell from {before[2]:.2f} to {after[2]:.2f}"


def test_lifting_is_idempotent():
    once = lift_to_floor("#004170")
    assert lift_to_floor(once) == once


def test_an_unparseable_colour_is_returned_unchanged():
    assert lift_to_floor("not-a-colour") == "not-a-colour"


def test_the_two_sides_of_this_fixture_both_clear_the_floor():
    """PSG v Aston Villa is the fixture that exposed this: navy against claret,
    both dark, on black."""
    for colour in ("#004170", "#7A003C"):
        assert _contrast_on_bg(mcolors.to_rgb(lift_to_floor(colour))) >= MARK_CONTRAST_FLOOR - 0.05
