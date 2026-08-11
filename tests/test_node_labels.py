"""Placement rules for the name written beside a pass-network node.

The failure these guard against is quiet: the label renders, the page looks
finished, and one player's name simply sits across another player's circle.
Nothing errors, so only a geometric check catches it.
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from visual_redesign_full import (
    _LABEL_HALF_HEIGHT,
    _LABEL_HALF_WIDTH_PER_CHAR,
    _network_node_radius,
    _node_neighbours,
    compact_player_label,
    draw_node_label,
)


@pytest.fixture()
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def placed_label(ax, name, x, y, radius, neighbours):
    """Draw one label and return where it actually landed."""
    before = len(ax.texts)
    draw_node_label(ax, x, y, name, touches=10, max_touch=10,
                    node_color="#8899AA", shirt=None,
                    node_radius=radius, neighbours=neighbours)
    assert len(ax.texts) > before, "no label was drawn"
    return ax.texts[-1]


def label_centre(text):
    """The centre of the drawn box, accounting for its alignment."""
    x, y = text.get_position()
    half_width = _LABEL_HALF_WIDTH_PER_CHAR * len(text.get_text())
    if text.get_ha() == "left":
        x += half_width
    elif text.get_ha() == "right":
        x -= half_width
    if text.get_va() == "bottom":
        y += _LABEL_HALF_HEIGHT
    elif text.get_va() == "top":
        y -= _LABEL_HALF_HEIGHT
    return x, y


def test_a_lone_node_keeps_its_name_above_it(ax):
    """No crowding means no change: the uncrowded network must look as before."""
    text = placed_label(ax, "Locatelli", 0.0, 0.0, radius=3.0, neighbours=())
    assert text.get_position() == (0.0, 3.0)
    assert (text.get_ha(), text.get_va()) == ("center", "bottom")


def test_the_label_moves_off_a_node_sitting_directly_above(ax):
    """The McKennie/Koopmeiners case: two central midfielders stacked in y."""
    neighbour = (0.0, 54.2, 3.0)
    text = placed_label(ax, "McKennie", 0.0, 50.0, radius=3.0, neighbours=(neighbour,))
    assert text.get_va() != "bottom", "label still written straight into the node above"

    cx, cy = label_centre(text)
    gap = math.hypot(cx - neighbour[0], cy - neighbour[1]) - neighbour[2]
    assert gap > 0, f"label centre still inside the neighbouring marker (gap {gap:.2f})"


def test_a_node_boxed_in_above_and_below_goes_sideways(ax):
    text = placed_label(ax, "Kelly", 0.0, 50.0, radius=3.0,
                        neighbours=((0.0, 54.5, 3.0), (0.0, 45.5, 3.0)))
    assert text.get_va() == "center"
    assert text.get_ha() in {"left", "right"}


def test_every_label_in_a_congested_network_clears_every_other_node(ax):
    """The real shape of the bug: a whole midfield packed into one area.

    Coordinates are in pitch space — x across the 58m width, y along the 105m
    length — because placement also weighs how close a label sits to an edge.
    """
    nodes = {
        "Kostic": (-9.0, 52.0, 40.0),
        "Koopmeiners": (-3.0, 54.0, 34.0),
        "McKennie": (-3.2, 49.0, 31.0),
        "Locatelli": (1.0, 47.0, 44.0),
        "González": (2.0, 55.0, 22.0),
    }
    max_touch = max(t for _x, _y, t in nodes.values())
    radii = {name: _network_node_radius(t, max_touch)
             for name, (_x, _y, t) in nodes.items()}

    for name, (x, y, touches) in nodes.items():
        neighbours = _node_neighbours(nodes, radii, name)
        text = placed_label(ax, name, x, y, radii[name], neighbours)
        cx, cy = label_centre(text)
        for other_x, other_y, other_r in neighbours:
            gap = math.hypot(cx - other_x, cy - other_y) - other_r
            assert gap > 0, f"{name}'s label lands inside another marker (gap {gap:.2f})"


def test_a_wide_player_is_not_labelled_off_the_edge_of_the_pitch(ax):
    """A left-back's name pushed further left gets clipped by the axis, which
    is how "Kostic" rendered as "ostic"."""
    from visual_redesign_full import PITCH_WIDTH

    x = -PITCH_WIDTH / 2 + 2.4  # where the separator clamps a touchline player
    text = placed_label(ax, "Kostic", x, 40.0, radius=3.4,
                        neighbours=((x + 3.0, 44.0, 3.0),))
    assert text.get_ha() != "right", "label pushed further into the touchline"

    cx, _cy = label_centre(text)
    half_width = _LABEL_HALF_WIDTH_PER_CHAR * len("Kostic")
    assert cx - half_width >= -PITCH_WIDTH / 2, "label runs off the left touchline"


def test_a_label_never_leaves_the_pitch_on_either_flank(ax):
    from visual_redesign_full import PITCH_WIDTH

    for x in (-PITCH_WIDTH / 2 + 2.4, PITCH_WIDTH / 2 - 2.4):
        text = placed_label(ax, "Koopmeiners", x, 50.0, radius=3.0,
                            neighbours=((x, 54.0, 3.0), (x, 46.0, 3.0)))
        cx, _cy = label_centre(text)
        half_width = _LABEL_HALF_WIDTH_PER_CHAR * len(text.get_text())
        assert cx - half_width >= -PITCH_WIDTH / 2 - 1e-6
        assert cx + half_width <= PITCH_WIDTH / 2 + 1e-6


def test_neighbours_never_include_the_node_being_labelled():
    nodes = {"a": (0.0, 0.0, 5.0), "b": (4.0, 0.0, 5.0)}
    radii = {"a": 2.0, "b": 2.0}
    assert _node_neighbours(nodes, radii, "a") == ((4.0, 0.0, 2.0),)


def substitution_row_positions(count):
    """Mirror of the panel's row layout, so the spacing rule can be checked
    without rendering a full page."""
    top, floor = 0.278, 0.150
    gap = 0.042 if count < 2 else min(0.042, (top - floor) / (count - 1))
    return [top - idx * gap for idx in range(count)]


FOOTER_Y = 0.105
LEGEND_Y = 0.055


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_substitution_rows_never_reach_the_footer(count):
    """Five substitutions in one half put the last row exactly on top of the
    "Completed pass links" line, which the fixed 0.042 step could not see."""
    rows = substitution_row_positions(count)
    assert rows[-1] > FOOTER_Y, (
        f"{count} substitutions put the last row at {rows[-1]:.3f}, "
        f"on the footer at {FOOTER_Y}"
    )
    assert rows[-1] - FOOTER_Y >= 0.04, "last row sits too close to the footer"


def test_the_footer_still_clears_the_legend():
    assert FOOTER_Y - LEGEND_Y >= 0.04


def test_four_substitutions_keep_their_original_spacing():
    """The fix must not reflow the common case."""
    assert substitution_row_positions(4) == pytest.approx(
        [0.278, 0.236, 0.194, 0.152]
    )


def test_rows_stay_in_order_and_never_collide_with_each_other():
    for count in range(1, 6):
        rows = substitution_row_positions(count)
        assert rows == sorted(rows, reverse=True)
        gaps = [a - b for a, b in zip(rows, rows[1:])]
        # 0.03 of the panel is about 25px against 10px of text — the tightest
        # the five-row case can be while still clearing the footer.
        assert all(gap >= 0.03 for gap in gaps), gaps


def test_the_side_panel_keeps_names_the_pitch_has_to_shorten():
    """Truncating in a column with room to spare threw away a legible name."""
    assert compact_player_label("Manuel Locatelli") == "Locate…"
    assert compact_player_label("Manuel Locatelli", 16) == "Locatelli"
    assert compact_player_label("Teun Koopmeiners", 16) == "Koopmeiners"


def test_a_genuinely_long_name_is_still_cut_to_the_limit():
    label = compact_player_label("Alexander-Arnold", 11)
    assert len(label) == 11 and label.endswith("…")
