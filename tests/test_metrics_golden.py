"""Every published metric, compared against a frozen reference run.

The unit tests in ``test_match_metrics.py`` check each function against hand
built events: they prove a definition is implemented as written. This file
proves something different and complementary — that the *whole pipeline*, run
end to end on a real match, still produces the numbers it produced before.

The input is ``sample_data/France_vs_England_4-6/events.csv``, a parsed match
committed to the repository, so a failure here is always the code and never a
changed download. The reference in ``tests/golden/`` was frozen from this
pipeline's own output, which makes the comparison exact rather than
approximate: a metric may only move when someone deliberately re-freezes it.

Re-freeze after an intended change:

    python scripts/freeze_golden.py --fixture france_vs_england
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from match_metrics import advanced_metrics_frames

REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS = REPO_ROOT / "sample_data" / "France_vs_England_4-6" / "events.csv"
GOLDEN = REPO_ROOT / "tests" / "golden" / "france_vs_england"

# The fixture the reference was frozen from. Hard-coded rather than parsed out
# of a filename so a renamed directory fails loudly instead of silently
# comparing one match against another's numbers.
MATCH_INFO = {
    "home_id": 341,
    "away_id": 345,
    "home_name": "France",
    "away_name": "England",
    "score": "4-6",
}

# Columns that name the row rather than measure it.
IDENTITY_COLUMNS = {"side", "team", "team_id", "player"}

# Half a rounding step at the one-decimal precision the reference is published
# at. Deliberately not loosened: a tolerance wide enough to absorb a changed
# definition would defeat the point of having a reference.
TOLERANCE = 0.051
PLAYER_TOLERANCE = 0.0051


def _require(path: Path, what: str) -> None:
    if not path.exists():
        pytest.skip(f"{what} not available at {path}")


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    _require(EVENTS, "sample events")
    return pd.read_csv(EVENTS, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def computed(events):
    team_frame, player_frame = advanced_metrics_frames(events, MATCH_INFO)
    return team_frame, player_frame


@pytest.fixture(scope="module")
def golden_team() -> pd.DataFrame:
    path = GOLDEN / "team_advanced_metrics.csv"
    _require(path, "frozen team reference")
    return pd.read_csv(path, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def golden_player() -> pd.DataFrame:
    path = GOLDEN / "player_sequence_metrics.csv"
    _require(path, "frozen player reference")
    return pd.read_csv(path, encoding="utf-8-sig")


def _by_team(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Key on the team id as text, so 341 and "341" compare equal."""
    return {str(row["team_id"]): row for _, row in frame.iterrows()}


def test_the_sample_match_still_parses(events):
    """Guards the fixture itself: a truncated CSV would make every comparison
    below pass vacuously on two empty frames."""
    assert len(events) == 1560
    assert set(events["team_id"].astype(str)) == {"341", "345"}


def test_no_published_column_disappeared(computed, golden_team):
    """Removing a metric is a breaking change for anything reading the export."""
    team_frame, _ = computed
    published = set(golden_team.columns) - IDENTITY_COLUMNS
    assert published <= set(team_frame.columns), (
        f"dropped since the reference was frozen: "
        f"{sorted(published - set(team_frame.columns))}"
    )


def test_every_team_metric_matches_the_reference(computed, golden_team):
    """The whole contract in one assertion, reporting every column that moved
    rather than stopping at the first."""
    team_frame, _ = computed
    current = _by_team(team_frame)
    drifted = []
    for _, expected in golden_team.iterrows():
        team_id = str(expected["team_id"])
        assert team_id in current, f"team {team_id} missing from the output"
        row = current[team_id]
        for column in golden_team.columns:
            if column in IDENTITY_COLUMNS:
                continue
            was, now = float(expected[column]), float(row[column])
            if abs(was - now) >= TOLERANCE:
                drifted.append(f"{team_id}.{column}: {was} -> {now}")
    assert not drifted, "metrics moved without a re-freeze:\n  " + "\n  ".join(drifted)


def test_the_two_sides_are_labelled_home_and_away(computed):
    team_frame, _ = computed
    assert sorted(team_frame["side"]) == ["away", "home"]


@pytest.mark.parametrize("column", ["field_tilt", "possession_share", "pass_share"])
def test_shares_still_split_a_hundred_percent(computed, column):
    """A share metric that stops summing to 100 is wrong even if both halves
    happen to match a stale reference."""
    team_frame, _ = computed
    assert float(team_frame[column].sum()) == pytest.approx(100.0, abs=0.2)


def test_game_states_still_partition_every_possession(computed):
    """Each possession is played at exactly one scoreline."""
    team_frame, _ = computed
    for _, row in team_frame.iterrows():
        across = sum(
            float(row[f"game_state_{state}_possessions"])
            for state in ("leading", "drawing", "trailing")
        )
        assert across == float(row["possession_count"]), row["team"]


def test_every_player_metric_matches_the_reference(computed, golden_player):
    _, player_frame = computed
    merged = golden_player.merge(
        player_frame, on=["player", "team_id"], suffixes=("_was", "_now")
    )
    assert len(merged) == len(golden_player), (
        "players in the reference are missing from the output: "
        f"{sorted(set(golden_player['player']) - set(player_frame['player']))}"
    )
    drifted = []
    for column in ("xGChain", "xGBuildup", "sequence_xT", "sequences"):
        gap = (merged[f"{column}_was"] - merged[f"{column}_now"]).abs()
        for index in gap[gap >= PLAYER_TOLERANCE].index:
            drifted.append(
                f"{merged.loc[index, 'player']}.{column}: "
                f"{merged.loc[index, f'{column}_was']} -> "
                f"{merged.loc[index, f'{column}_now']}"
            )
    assert not drifted, "player metrics moved without a re-freeze:\n  " + "\n  ".join(
        drifted
    )
