"""Comparison pools: each role code lands in the line the player was playing."""

import pytest

from football_analysis.metrics.player_advanced import line_of


@pytest.mark.parametrize(
    "role, line",
    [
        ("GK", "Goalkeeper"),
        ("DC", "Defence"),
        ("DL", "Defence"),
        ("DR", "Defence"),
        ("DMC", "Midfield"),
        ("DML", "Midfield"),
        ("DMR", "Midfield"),
        ("MC", "Midfield"),
        ("AMC", "Attack"),
        ("FW", "Attack"),
        ("Sub", "Unknown"),
        (None, "Unknown"),
    ],
)
def test_each_role_code_is_pooled_with_its_own_line(role, line):
    assert line_of(role) == line
