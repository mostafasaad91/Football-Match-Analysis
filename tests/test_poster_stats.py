"""The posters' aggregated numbers, and the panels that had none.

Four defects in the sixteen-indicator table, one of them arithmetic:

- A tie coloured both figures. `home_w >= away_w` and `away_w >= home_w` are
  both true at equality, so "BIG CHANCES 1 — 1" printed each side's number in
  its own club colour, as though both had led the row.
- The bar was drawn through the label. The name sat at y + 0.21 and the bar
  spanned y − 0.16 to y − 0.01, which are the same band, so all sixteen rows
  had a coloured bar running across their own name. Sixteen rows leave 0.06 of
  the axes each and a 6.6pt name occupies very nearly that on its own, so a
  name above a bar was always going to be a name across a bar.
- The trailing figure was NEUTRAL, 3.0:1 against the black page. Half the
  table was the hard half to read — the same defect the report's cover had.
- The bar was each side's share of the sum, which cannot separate 15 against 3
  from 46 against 40: both land near the middle of their own half.

And on the new poster, a counting error worth more than the layout ones. Opta
writes a foul, a corner and an aerial twice — once for the player who won it
and once for the player who did not — so counting rows by team gave both sides
the same figure in every row: fouls 23-23, corners 10-10, aerials 28-28.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import match_posters as mp


def _table_rows():
    return [
        ("Expected goals", "1.67", "0.21", 1.67, 0.21),
        ("Big chances", "1", "1", 1.0, 1.0),
        ("Box entries", "15", "3", 15.0, 3.0),
        ("Pitch control", "46%", "40%", 46.0, 40.0),
    ]


def _drawn(rows):
    """Render one table and hand back the text and patch artists."""
    fig = plt.figure(figsize=(4, 6))
    ax = fig.add_axes([0, 0, 1, 1])
    mp.panel_stat_table(ax, rows, "#EF0107", "#78D2F2")
    texts = [(t.get_text(), t.get_position(), t.get_color()) for t in ax.texts]
    patches = list(ax.patches)
    plt.close(fig)
    return texts, patches


# --------------------------------------------------------------------------
# the sixteen indicators
# --------------------------------------------------------------------------

def test_a_tie_is_not_a_win_for_either_side():
    texts, _patches = _drawn(_table_rows())
    ones = [colour for text, _pos, colour in texts if text == "1"]
    assert len(ones) == 2, ones
    assert set(ones) == {mp.TABLE_DIM}, ones


def test_the_leader_keeps_its_club_colour():
    texts, _patches = _drawn(_table_rows())
    by_text = {text: colour for text, _pos, colour in texts}
    assert by_text["1.67"] == "#EF0107"
    assert by_text["0.21"] == mp.TABLE_DIM


def test_the_trailing_figure_clears_the_contrast_floor():
    """NEUTRAL is 3.0:1 on this page and was carrying one number per row."""
    def channel(value):
        value /= 255.0
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (int(mp.TABLE_DIM[i:i + 2], 16) for i in (1, 3, 5))
    luminance = 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)
    assert (luminance + 0.05) / 0.05 >= 4.5


def test_nothing_is_drawn_across_its_own_label():
    """The name and the bar may not occupy the same band."""
    texts, patches = _drawn(_table_rows())
    labels = [(t, pos) for t, pos, _c in texts if t.isupper() and len(t) > 3]
    assert labels, [t for t, _p, _c in texts]
    # Every bar rectangle spans the full width of the track; the names sit to
    # the left of where any of them begins.
    left_edges = [p.get_x() for p in patches if p.get_width() > 0.2]
    assert left_edges
    for text, (x, _y) in labels:
        assert x < min(left_edges), f"{text} starts at {x}, bars start at {min(left_edges)}"


def test_no_table_label_is_wider_than_the_column_it_sits_in():
    """The name column ends where the home figure begins.

    Every row is one lane, so a name that runs past the column runs into its
    own number: "FINISHING VS EXPECTED0.36", "SEQUENCE XT PER POSSESSION0.552".
    Three tables feed the same renderer and each was written without knowing
    what the others' longest name was, so the cap is asserted here rather than
    remembered.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 6))
    ax = fig.add_axes([0, 0, 1, 1])
    mp.panel_stat_table(ax, _table_rows(), "#EF0107", "#78D2F2")
    figures = [t for t in ax.texts if t.get_text().strip() == "1.67"]
    assert figures, "the home figure is no longer drawn"
    column_ends = figures[0].get_position()[0]
    plt.close(fig)

    tables = {
        "shooting": mp.build_shooting_rows,
        "transition": mp.build_transition_rows,
    }
    longest = max(
        (str(label) for source in _table_labels(tables) for label in source),
        key=len)
    # 5.9pt monospace-ish caps run about 0.0105 of the panel per character.
    assert len(longest) * 0.0105 < column_ends - 0.012, (
        f"{longest!r} is {len(longest)} characters; the column holds about "
        f"{int((column_ends - 0.012) / 0.0105)}")


def _table_labels(tables):
    """The label column of every stat table, without building the frames."""
    import inspect
    import re

    for build in tables.values():
        source = inspect.getsource(build)
        yield re.findall(r'\(\s*"([^"]+)"\s*,', source)


def test_the_bar_is_scaled_against_the_larger_side_not_the_sum():
    """A share of the sum cannot separate 15-3 from 46-40."""
    _texts, patches = _drawn(_table_rows())
    bars = [p for p in patches if 0 < p.get_width() < 0.2]
    widths = sorted(p.get_width() for p in bars)
    # 3 against 15 is a fifth; 40 against 46 is seven eighths. Under a share of
    # the sum those become 0.17 and 0.47 of a half — much closer together.
    assert max(widths) / max(min(widths), 1e-9) > 4.0, widths


# --------------------------------------------------------------------------
# the tactical book
# --------------------------------------------------------------------------

def test_a_shape_is_read_from_the_back():
    assert mp._formation_shape("4231") == [4, 2, 3, 1]
    assert mp._formation_shape("352") == [3, 5, 2]
    assert mp._formation_shape("41212") == [4, 1, 2, 1, 2]


def test_a_shape_that_does_not_add_up_draws_nothing():
    """Ten outfielders or it is not a formation.

    The digits are what is read, so "442x" is 4-4-2 with a stray character and
    is accepted; "4444" adds up to sixteen and is not a shape anybody played.
    """
    for bad in ("", "4", "4444", "99", "1234567"):
        assert mp._formation_shape(bad) == [], bad
    assert mp._formation_shape("442x") == [4, 4, 2]


def test_the_keeper_is_at_the_bottom_of_the_drawn_shape():
    """"4231" is read from the back, so the bands run keeper, 4, 2, 3, 1
    upwards. Drawing them downwards put the lone striker on his own goal line
    and the back four in front of the opposition box."""
    spells = [{"side": "home", "formation": "4231",
               "start_minute": 0, "end_minute": 94}]
    fig = plt.figure(figsize=(3, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    mp.panel_formation(ax, spells, "home", "#EF0107", "Arsenal")
    circles = sorted(ax.patches, key=lambda c: c.center[1])
    plt.close(fig)

    assert len(circles) == 11, len(circles)
    lowest = circles[0].center[1]
    on_the_line = [c for c in circles if abs(c.center[1] - lowest) < 1e-6]
    assert len(on_the_line) == 1, "the keeper shares his line with somebody"
    highest = circles[-1].center[1]
    at_the_top = [c for c in circles if abs(c.center[1] - highest) < 1e-6]
    assert len(at_the_top) == 1, "the lone striker is not alone at the top"


def _arsenal_xi():
    """Positions as the squad export records them, in listing order."""
    rows = [
        ("GK", "David Raya"), ("DR", "Ben White"), ("DC", "Gabriel Magalhães"),
        ("DC", "Cristhian Mosquera"), ("DL", "Riccardo Calafiori"),
        ("DMC", "Declan Rice"), ("DMC", "Myles Lewis-Skelly"),
        ("AMR", "Bukayo Saka"), ("AMC", "Martin Ødegaard"),
        ("AML", "Christos Tzolis"), ("FW", "Kai Havertz"),
    ]
    return pd.DataFrame([
        {"team_id": 13, "position": p, "name": n, "is_first_xi": True}
        for p, n in rows])


def test_the_shape_comes_from_the_positions_not_the_provider_slots():
    """formationSlots are the provider's own layout ids, not a back-to-front
    reading order, and treating them as one put Declan Rice at left-back and
    Bukayo Saka in the double pivot."""
    rows = mp.lineup_rows(_arsenal_xi(), 13)
    assert [len(r) for r in rows] == [1, 4, 2, 3, 1], rows
    assert rows[0] == ["Raya"]
    assert set(rows[2]) == {"Rice", "Lewis-Skelly"}
    assert rows[4] == ["Havertz"]


def test_a_right_back_is_drawn_on_the_right():
    """The side attacks up the panel, so the team's right is the reader's
    right. Sorting the other way mirrored every wide player."""
    rows = mp.lineup_rows(_arsenal_xi(), 13)
    back_four = rows[1]
    assert back_four[0] == "Calafiori", back_four   # left-back, leftmost
    assert back_four[-1] == "White", back_four      # right-back, rightmost
    front_three = rows[3]
    assert front_three[0] == "Tzolis", front_three  # left winger
    assert front_three[-1] == "Saka", front_three   # right winger


def test_a_squad_without_positions_draws_nothing_rather_than_guessing():
    assert mp.lineup_rows(None, 13) == []
    assert mp.lineup_rows(pd.DataFrame({"name": ["a"]}), 13) == []


def test_a_fixture_without_formations_draws_a_note_rather_than_failing():
    fig = plt.figure(figsize=(3, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    mp.panel_formation(ax, [], "home", "#EF0107", "Arsenal")
    assert not ax.patches
    assert ax.texts
    plt.close(fig)


def _duel_events():
    """Two fouls and two aerials, each written for both participants."""
    rows = []
    for minute, winner, loser in ((10, 1, 2), (20, 2, 1)):
        rows += [
            {"minute": minute, "team_id": winner, "type": "Foul",
             "outcome": "Successful", "x": 50, "y": 50},
            {"minute": minute, "team_id": loser, "type": "Foul",
             "outcome": "Unsuccessful", "x": 50, "y": 50},
            {"minute": minute, "team_id": winner, "type": "Aerial",
             "outcome": "Successful", "x": 50, "y": 50},
            {"minute": minute, "team_id": loser, "type": "Aerial",
             "outcome": "Unsuccessful", "x": 50, "y": 50},
        ]
    return pd.DataFrame(rows)


def test_a_foul_is_counted_once_against_the_side_that_committed_it():
    """Opta writes it twice — for the player fouled and for the fouler — so
    counting rows by team gave both sides 23 in the same match."""
    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    mp.panel_dead_ball(ax, _duel_events(), 1, 2, "#EF0107", "#78D2F2",
                       "Home", "Away")
    figures = [t.get_text() for t in ax.texts if t.get_text().isdigit()]
    plt.close(fig)
    # One foul conceded and one aerial won each, not two of everything.
    assert figures.count("2") == 0, figures
    assert figures.count("1") >= 4, figures


# --------------------------------------------------------------------------
# saving when the machine cannot allocate
# --------------------------------------------------------------------------

def test_a_save_that_cannot_allocate_is_retried_smaller():
    """A four-hour run died on its first PNG with "bad allocation".

    The data was ordinary and the machine was the problem: 22.5 GB of a 23.7 GB
    commit limit was already spoken for. That is a fair reason for one image to
    fail and a bad reason to lose the whole package.
    """
    from visualization_components import SAVE_DPI_STEPS, save_figure

    class Stubborn:
        def __init__(self, fails):
            self.fails, self.tried = fails, []

        def savefig(self, path, *, dpi, **kwargs):
            self.tried.append(dpi)
            if len(self.tried) <= self.fails:
                raise MemoryError("bad allocation")

    fig = Stubborn(fails=2)
    got = save_figure(fig, "x.png", dpi=200)
    assert len(fig.tried) == 3, fig.tried
    assert fig.tried == sorted(fig.tried, reverse=True), fig.tried
    assert got == fig.tried[-1]


def test_a_machine_that_cannot_draw_at_all_still_raises():
    """A silent half-sized image would be worse than the failure."""
    from visualization_components import save_figure

    class Hopeless:
        def savefig(self, path, *, dpi, **kwargs):
            raise MemoryError("bad allocation")

    with pytest.raises(MemoryError):
        save_figure(Hopeless(), "x.png", dpi=155)


def test_the_step_down_actually_reduces_the_buffer():
    from visualization_components import SAVE_DPI_STEPS

    assert SAVE_DPI_STEPS[0] == 1.0
    assert list(SAVE_DPI_STEPS) == sorted(SAVE_DPI_STEPS, reverse=True)
    # Each step has to be worth taking: a buffer scales with dpi squared, so
    # the smallest step must at least halve the memory the first one wanted.
    assert SAVE_DPI_STEPS[-1] ** 2 <= 0.5
