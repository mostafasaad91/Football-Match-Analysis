"""The xG-flow page's sentence about the first goal says when it came, correctly.

It used to call every open-play opener "a low-value chance taken early" and say
the rest was played "at a scoreline neither side had earned yet" -- printed on a
79th-minute winner worth 0.05 xG.
"""

from __future__ import annotations

import pytest

from football_analysis.prose.report_narrative import _chance_kind, _what_the_opener_framed


@pytest.mark.parametrize(
    "at,word", [(79, "late"), (90, "late"), (12, "early"), (30, "early"), (50, "ahead")]
)
def test_the_timing_follows_the_minute(at, word):
    assert word in _what_the_opener_framed(at)


def test_no_goal_is_said_to_come_at_an_unearned_scoreline():
    for at in (1, 30, 45, 60, 79, 95):
        assert "neither side had earned" not in _what_the_opener_framed(at)


@pytest.mark.parametrize(
    "xg,kind", [(0.05, "low-value"), (0.2, "half-chance"), (0.45, "clear chance")]
)
def test_the_chance_is_named_by_its_value(xg, kind):
    assert kind in _chance_kind(xg)
