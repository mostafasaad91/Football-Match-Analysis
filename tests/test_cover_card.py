"""The cover states figures, not a claim.

It used to be the pitch-control artwork with one sentence under it. That
sentence was a single assertion, which made it the part of the report most
likely to be wrong — and it was: Manchester United, beaten 2-0 at Hull after
creating 94% of their xG from behind, were introduced as the side that "created
the better chances and lost".

A comparison card cannot do that. Every row is two figures out of the frames
and a bar drawn from them, so the cover carries the shape of the match without
asserting anything about it.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import tactical_pdf_report as tp
from conftest import match_dir

ROOT = Path(__file__).resolve().parent.parent
MATCHES = [
    "England_Premier_League/2026-2027/Week_of_2026-08-17/Hull_vs_Man_Utd_2-0",
    "Europe_UEFA_Super_Cup/2025-2026/Week_of_2026-08-10/PSG_vs_Aston_Villa_2-1",
]


def _context(match):
    out = match_dir(match)
    if not (out / "match_info.json").exists():
        pytest.skip(f"{match} has not been rendered")
    info = json.loads((out / "match_info.json").read_text(encoding="utf-8"))
    return tp.build_context(
        pd.read_csv(out / "events.csv"),
        pd.read_csv(out / "xg.csv"),
        pd.read_csv(out / "team_advanced_metrics.csv"),
        pd.read_csv(out / "player_sequence_metrics.csv"),
        info,
    ), out


def _card(context):
    """The cover's rows, without drawing anything."""
    report = tp.TacticalPDF.__new__(tp.TacticalPDF)
    report.context = context
    return report._cover_rows()


@pytest.mark.parametrize("match", MATCHES)
def test_every_row_is_filled_from_the_frames(match):
    context, _ = _context(match)
    rows = _card(context)
    assert len(rows) == len(tp.TacticalPDF.COVER_ROWS)
    for label, home_text, away_text, share in rows:
        assert label and home_text and away_text, label
        assert 0.0 <= share <= 1.0, (label, share)
        assert "nan" not in f"{home_text}{away_text}".lower(), (label, home_text)


@pytest.mark.parametrize("match", MATCHES)
def test_the_longer_bar_belongs_to_the_larger_figure(match):
    """Four lines of arithmetic overwrote each other and every one of them
    pinned the leading side at exactly half, so 2.15 against 1.11 drew the same
    bar as 1.11 against 2.15."""
    context, _ = _context(match)
    for label, home_text, away_text, share in _card(context):
        home = _leading_number(home_text)
        away = _leading_number(away_text)
        if home is None or away is None or home == away:
            continue
        assert (share > 0.5) == (home > away), (label, home_text, away_text, share)


def _leading_number(text: str):
    import re

    found = re.match(r"[-+]?\d+(?:\.\d+)?", str(text).strip())
    return float(found.group(0)) if found else None


@pytest.mark.parametrize("match", MATCHES)
def test_the_competition_line_is_read_not_typed(match):
    """The header printed its own subtitle twice when the line came back empty."""
    context, _ = _context(match)
    report = tp.TacticalPDF.__new__(tp.TacticalPDF)
    report.context = context
    line = report._cover_competition()
    assert line, "no competition line for a fixture whose URL is on record"
    assert line.upper() != "MATCH ANALYSIS"


@pytest.mark.parametrize("match", MATCHES)
def test_the_cover_page_carries_the_figures_and_no_verdict(match):
    """Read back out of the finished PDF, which is what a reader sees."""
    import fitz

    _context_unused, out = _context(match)
    pdf = out / "full_visual_redesign_real_data.pdf"
    if not pdf.exists():
        pytest.skip("not rendered")
    document = fitz.open(pdf)
    try:
        text = document[0].get_text()
    finally:
        document.close()

    squashed = text.replace(" ", "").upper()
    for label, _key, _shape in tp.TacticalPDF.COVER_ROWS:
        assert label.replace(" ", "").upper() in squashed, label

    for claim in ("created the better chances and lost",
                  "won the execution battle",
                  "played better in"):
        assert claim.lower() not in text.lower(), claim


def test_a_side_with_no_crest_still_gets_a_mark():
    """The disc and initials that stand in for a crest that never downloaded."""
    # One letter per word: three words give three letters. AVFC is on the
    # crest, not derivable from "Aston Villa FC".
    assert tp._club_initials("Aston Villa FC") == "AVF"
    assert tp._club_initials("Aston Villa") == "AV"
    assert tp._club_initials("PSG") == "PSG"
    assert tp._club_initials("Hull") == "HUL"
    assert tp._club_initials("") == "?"
    assert tp._club_initials("Borussia Monchengladbach") == "BM"


def test_letter_spacing_does_not_lose_characters():
    assert tp._spaced_out("ABC").replace(" ", "") == "ABC"
