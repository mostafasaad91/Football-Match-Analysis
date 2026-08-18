"""A value printed at the end of a bar has to fit inside the axes.

Matplotlib does not clip text drawn past the axis limit — it draws it anyway,
over whatever is next to the panel. On a two-panel board that means one side's
numbers land on the other side's category names, and on a single panel they run
off the page.

It only shows when the longest bar reaches the axis edge, which happens exactly
when the counts are small: `49_press_triggers` printed "2 (22%)" across Man
City's labels because Arsenal's press produced two regains at most, and the
axis therefore ended at two. Two fixtures had never produced a press that
small, so nothing caught it.

So the test is about geometry rather than about any one chart: for every
horizontal bar chart, every label must sit inside the axes that drew it.
"""

import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
MATCH = "Arsenal_vs_Man_City_3-0"


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    out = ROOT / "output" / MATCH
    if not (out / "match_info.json").exists():
        pytest.skip(f"{MATCH} has not been rendered")

    import visual_redesign_full as v

    info = json.loads((out / "match_info.json").read_text(encoding="utf-8"))
    target = tmp_path_factory.mktemp("bars")
    v.configure_match(info, target)
    Path(v.OUT).mkdir(parents=True, exist_ok=True)
    v.base.theme()
    return v, out


def _labels_inside_every_axes(figure) -> list[str]:
    """Which text objects sit outside the data range of their own axes."""
    escaped = []
    for ax in figure.axes:
        low, high = ax.get_xlim()
        for text in ax.texts:
            x = text.get_position()[0]
            if x < min(low, high) or x > max(low, high):
                escaped.append(f"{text.get_text()!r} at x={x:.3f}, axis {low:.2f}..{high:.2f}")
    return escaped


def test_press_trigger_labels_stay_inside_their_panel(rendered):
    """The board that printed one side's numbers over the other's labels."""
    v, out = rendered
    events = pd.read_csv(out / "events.csv")
    import matplotlib.pyplot as plt

    v.press_and_rest(events)
    for figure in map(plt.figure, plt.get_fignums()):
        assert not _labels_inside_every_axes(figure), _labels_inside_every_axes(figure)
    plt.close("all")


def test_press_triggers_are_counted_in_whole_numbers(rendered):
    """A tick at 0.25 of a regain does not exist."""
    v, out = rendered
    import matplotlib.pyplot as plt

    v.press_and_rest(pd.read_csv(out / "events.csv"))
    for figure in map(plt.figure, plt.get_fignums()):
        for ax in figure.axes:
            if ax.get_xlabel() != "High regains":
                continue
            ticks = [t for t in ax.get_xticks()
                     if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
            assert all(abs(t - round(t)) < 1e-9 for t in ticks), ticks
    plt.close("all")


@pytest.mark.parametrize("chart", ["player_sequence", "action_value_leaders"])
def test_the_other_bar_charts_keep_their_labels_inside(rendered, chart):
    v, out = rendered
    import matplotlib.pyplot as plt

    events = pd.read_csv(out / "events.csv")
    players = pd.read_csv(out / "player_sequence_metrics.csv")
    getattr(v, chart)(players if chart == "player_sequence" else events)
    for figure in map(plt.figure, plt.get_fignums()):
        escaped = _labels_inside_every_axes(figure)
        assert not escaped, escaped
    plt.close("all")


def test_a_label_past_the_axis_would_be_caught():
    """A guard on the guard: the detector must actually detect."""
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    ax.barh([0], [2])
    ax.set_xlim(0, 2)
    ax.text(2.4, 0, "outside")
    assert _labels_inside_every_axes(figure)
    plt.close(figure)
