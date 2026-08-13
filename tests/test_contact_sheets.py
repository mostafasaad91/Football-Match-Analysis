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
    THREAD_OMISSIONS,
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


def omitted_slots() -> set[str]:
    return {re.sub(r"_(france|england)", "_team", name.lower())
            for name in THREAD_OMISSIONS}


def test_every_rendered_visual_is_either_in_the_thread_or_deliberately_out():
    """Fifteen visuals once reached no sheet at all. A visual may now be left
    out of the thread, but only by being named with a reason."""
    produced = rendered_slots()
    if not produced:
        pytest.skip("no rendered fixture on disk to check coverage against")
    accounted = referenced_slots() | omitted_slots()
    orphans = sorted(produced - accounted)
    assert not orphans, f"visuals neither in the thread nor listed as omitted: {orphans}"


def test_no_omission_is_left_unexplained():
    for name, reason in THREAD_OMISSIONS.items():
        assert reason and len(reason) > 15, f"{name}: omission has no real reason"


def test_nothing_is_both_omitted_and_used():
    overlap = omitted_slots() & referenced_slots()
    assert not overlap, f"listed as omitted but still on a sheet: {sorted(overlap)}"


def test_the_thread_is_twelve_posts_of_four():
    """One sheet per post, four visuals per post."""
    assert len(DASHBOARDS) == 12
    for title, _subtitle, entries in DASHBOARDS:
        assert len(entries) == 4, f"{title} has {len(entries)} visuals, not 4"


def test_every_entry_carries_a_narrative_note():
    """Without the notes the thread is a set of images that happens to be in
    an order; the note is what states the order is doing something."""
    for title, _subtitle, entries in DASHBOARDS:
        for entry in entries:
            name, note = _split_entry(entry)
            assert note, f"{title}: {name} has no note"
            assert not note.endswith("."), f"{title}: {name} note reads as a sentence fragment"


def test_the_thread_opens_on_the_result_and_closes_on_the_players():
    assert DASHBOARDS[0][0] == "The Result"
    assert DASHBOARDS[-1][0] == "The Difference"


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


def sheet_contents() -> list[tuple[str, list[str]]]:
    return [
        (title, [_split_entry(entry)[0] for entry in entries])
        for title, _subtitle, entries in DASHBOARDS
    ]


def test_no_sheet_is_made_entirely_of_visuals_shown_elsewhere():
    """Adding the story sheet emptied "Transitions and Final Verdict" of its
    own content: all three of its visuals moved into the story, and it stayed
    in the set showing the same three images a second time."""
    sheets = sheet_contents()
    counts = {}
    for _title, names in sheets:
        for name in names:
            counts[name] = counts.get(name, 0) + 1

    hollow = [
        title for title, names in sheets
        if names and all(counts[name] > 1 for name in names)
    ]
    assert not hollow, f"sheets with nothing of their own: {hollow}"


def test_a_sheet_never_repeats_a_visual_within_itself():
    for title, names in sheet_contents():
        assert len(names) == len(set(names)), f"{title} repeats a visual"


def test_duplication_across_sheets_stays_the_exception():
    """A visual may earn a place on two sheets; most should not."""
    sheets = sheet_contents()
    counts = {}
    for _title, names in sheets:
        for name in names:
            counts[name] = counts.get(name, 0) + 1
    repeated = [name for name, n in counts.items() if n > 1]
    assert len(repeated) <= len(counts) * 0.1, f"too much repetition: {sorted(repeated)}"


def test_no_sheet_is_empty_and_every_title_is_distinct():
    titles = [title for title, _s, _e in DASHBOARDS]
    assert len(titles) == len(set(titles))
    for title, _subtitle, entries in DASHBOARDS:
        assert entries, f"{title} has no visuals"
