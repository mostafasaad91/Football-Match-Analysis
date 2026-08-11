"""Rules for the QA contact sheets.

The defect these guard against is silent: the sheet list was written when the
project rendered thirty-four visuals and was never revisited when it grew to
forty-nine, so fifteen visuals were produced on every run and reached no sheet
at all. Nothing errored — they simply were not there.
"""

import re
from pathlib import Path

import pytest

from build_qa_contact_sheets import (
    CELL,
    DASHBOARDS,
    _SHEET_HEIGHT,
    _layout,
    _split_entry,
)

REPO = Path(__file__).resolve().parent.parent


def referenced_slots() -> set[str]:
    """Every filename the sheets ask for, with the team slug normalised."""
    slots = set()
    for _title, _subtitle, entries in DASHBOARDS:
        for entry in entries:
            name, _note = _split_entry(entry)
            slots.add(re.sub(r"_(france|england)", "_team", name.lower()))
    return slots


def rendered_slots() -> set[str]:
    """Numbered visuals from a rendered fixture, if one is on disk."""
    output = REPO / "output"
    if not output.is_dir():
        return set()
    best, slots = 0, set()
    for match_dir in output.iterdir():
        if not match_dir.is_dir():
            continue
        names = [p.name for p in match_dir.glob("[0-9]*.png")]
        if len(names) <= best:
            continue
        # Directory names read "Home_vs_Away_<score>"; the score has to come
        # off first or the away slug keeps it and never matches a filename.
        stem = re.sub(r"_\d+-\d+$", "", match_dir.name).lower()
        if "_vs_" not in stem:
            continue
        # Longest first, so a slug that contains another is replaced whole.
        slugs = sorted(stem.split("_vs_"), key=len, reverse=True)
        pattern = "|".join(re.escape(s) for s in slugs)
        best = len(names)
        slots = {re.sub(rf"_({pattern})", "_team", name.lower()) for name in names}
    return slots


def test_every_rendered_visual_reaches_a_sheet():
    produced = rendered_slots()
    if not produced:
        pytest.skip("no rendered fixture on disk to check coverage against")
    wanted = referenced_slots()
    orphans = sorted(produced - wanted)
    assert not orphans, f"visuals rendered but on no contact sheet: {orphans}"


def test_the_story_sheet_comes_first_and_carries_twelve():
    title, _subtitle, entries = DASHBOARDS[0]
    assert title == "The Match Story"
    assert len(entries) == 12


def test_every_story_entry_has_a_narrative_note():
    """The note is what makes an ordered set of visuals a reading."""
    _title, _subtitle, entries = DASHBOARDS[0]
    for entry in entries:
        name, note = _split_entry(entry)
        assert note, f"{name} has no note"
        assert not note.endswith("."), f"{name}: note is a label, not a sentence"


def test_no_sheet_asks_for_more_than_its_grid_holds():
    for title, _subtitle, entries in DASHBOARDS:
        rows, cols = _layout(len(entries))
        assert len(entries) <= rows * cols, f"{title} overflows its {rows}x{cols} grid"


def test_every_layout_has_a_sheet_height():
    for count in (1, 4, 6, 8, 12):
        rows, _cols = _layout(count)
        assert rows in _SHEET_HEIGHT, f"{count} visuals -> {rows} rows with no height"


def test_a_third_row_makes_the_sheet_taller_not_the_thumbnails_smaller():
    """Twelve squeezed into the two-row canvas would cost a third of every
    thumbnail's height."""
    assert _SHEET_HEIGHT[3] > _SHEET_HEIGHT[2]
    per_row_two = _SHEET_HEIGHT[2] / 2
    per_row_three = _SHEET_HEIGHT[3] / 3
    assert per_row_three >= per_row_two * 0.9


def test_the_thumbnail_frame_is_wider_than_it_is_tall():
    """The visuals are landscape; a portrait cell would letterbox every one."""
    assert CELL[0] > CELL[1]


def test_no_sheet_is_empty_and_every_title_is_distinct():
    titles = [title for title, _s, _e in DASHBOARDS]
    assert len(titles) == len(set(titles))
    for title, _subtitle, entries in DASHBOARDS:
        assert entries, f"{title} has no visuals"
