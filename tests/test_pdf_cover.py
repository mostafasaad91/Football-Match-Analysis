"""The report's cover page.

It used to lead with a divided bar only when field tilt, possession or pass
share differed by 25 points or more, and to fall back to a single thin strip
otherwise. That fallback was the normal case -- a 59/41 possession match did
not qualify -- and the cover then carried content on 29% of its rows, with two
dead bands of 24% and 26%.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

import tactical_pdf_report as report
from tactical_pdf_report import TacticalPDF, build_context
from conftest import match_dir

ROOT = Path(__file__).resolve().parent.parent


def _context():
    out = match_dir("PSG_vs_Aston_Villa_2-1")
    if not (out / "match_info.json").exists():
        pytest.skip("no rendered fixture available")
    info = json.loads((out / "match_info.json").read_text(encoding="utf-8"))
    context = build_context(
        pd.read_csv(out / "events.csv"),
        pd.read_csv(out / "xg.csv"),
        pd.read_csv(out / "team_advanced_metrics.csv"),
        pd.read_csv(out / "player_sequence_metrics.csv"),
        info,
    )
    context["home_color"] = info.get("home_color", "#004170")
    context["away_color"] = info.get("away_color", "#7A003C")
    return context


def _cover(tmp_path, context=None):
    pdf = TacticalPDF(tmp_path / "cover.pdf", context or _context())
    pdf.cover()
    pdf.canvas.save()
    return pdf


# --------------------------------------------------------------------------
# the lead statistic
# --------------------------------------------------------------------------

def test_every_match_gets_a_lead_statistic():
    """The old pool of three share metrics left most matches with none."""
    assert _cover.__doc__ is None or True  # keeps the helper referenced
    pdf = TacticalPDF.__new__(TacticalPDF)
    pdf.context = _context()
    assert pdf._cover_lead() is not None


def test_the_lead_ranks_on_the_relative_gap():
    """9.1 against 3.7 is a bigger difference than 54 against 46."""
    pdf = TacticalPDF.__new__(TacticalPDF)
    pdf.context = _context()
    kind, name, _note, home, away = pdf._cover_lead()
    assert kind == "rate"
    assert name == "Regain to shot"
    assert (home, away) == (pytest.approx(3.7), pytest.approx(9.1))


def test_a_rate_is_never_drawn_as_a_split():
    """Two independent percentages do not add up, so one divided bar would lie."""
    for kind, key, _name, _note in TacticalPDF.COVER_LEADS:
        assert kind in {"split", "rate"}
        if kind == "split":
            assert key in {"field_tilt", "possession_share", "pass_share"}


def test_split_candidates_are_rejected_when_they_do_not_sum_to_a_whole():
    pdf = TacticalPDF.__new__(TacticalPDF)
    pdf.context = {"home_field_tilt": 90.0, "away_field_tilt": 90.0}
    assert pdf._cover_lead() is None


def test_a_goalless_metric_is_skipped_rather_than_dividing_by_zero():
    pdf = TacticalPDF.__new__(TacticalPDF)
    pdf.context = {"home_regain_to_shot_rate": 0.0, "away_regain_to_shot_rate": 0.0}
    assert pdf._cover_lead() is None


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

def _coverage(tmp_path, context):
    """Share of the cover's rows carrying anything, its deepest gap, and reach.

    ``reach`` is the span from the first inked row to the last, as a share of
    the page. The comparison card leaves deliberate air between its rows, so a
    fill ratio alone cannot tell a well-spaced card from a page whose bottom
    half failed to draw — the span can.
    """
    fitz = pytest.importorskip("fitz")
    import numpy as np

    _cover(tmp_path, context)
    doc = fitz.open(tmp_path / "cover.pdf")
    pix = doc[0].get_pixmap(dpi=60)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[..., :3].astype(int)
    page = image[4, 4]
    rows = (np.abs(image - page).sum(axis=2) > 12).any(axis=1)
    doc.close()

    longest = current = 0
    for hit in rows:
        current = 0 if hit else current + 1
        longest = max(longest, current)
    inked = np.flatnonzero(rows)
    reach = (inked[-1] - inked[0] + 1) / len(rows) if inked.size else 0.0
    return rows.sum() / len(rows), longest / len(rows), reach


# The cover used to be a full-bleed pass-network render, so "fills the page"
# was measured as a fill ratio and held above 80%. It is now the comparison
# card the reference asked for: a header, two crests either side of the score
# and eight labelled bars, with air between the rows because that is what
# makes a row of figures readable. Forty-four per cent of the rows carry ink
# by design, and holding the old number would mean either a failing suite or
# a cover crushed back together to satisfy it.
#
# What the tests were really guarding is that the cover does not go blank, and
# that no band of it silently fails to draw. Both still hold, measured against
# the design that ships.
COVER_MIN_FILL = 0.30
COVER_MAX_GAP = 0.20
COVER_MIN_REACH = 0.90


def test_the_cover_draws_every_band_of_the_page(tmp_path):
    """Measured, because 'looks empty' is exactly what went unnoticed before."""
    covered, gap, reach = _coverage(tmp_path, _context())
    assert covered > COVER_MIN_FILL, (
        f"only {covered:.0%} of the cover's rows carry anything")
    assert gap < COVER_MAX_GAP, f"a dead band {gap:.0%} of the page deep"
    assert reach > COVER_MIN_REACH, (
        f"the cover's ink spans only {reach:.0%} of the page")


def test_the_card_sits_between_its_rules_with_equal_air(tmp_path):
    """The card is one block, so its two margins have to match.

    The header rule was fixed to the sheet and the footer floated under the
    last row, which put 129pt of air above the crests and 40 under the final
    figure — and made the footer's position a function of how many rows
    COVER_ROWS happened to hold.

    Measured off the render rather than off the arithmetic: reproducing the
    layout maths here would agree with itself no matter what the page looked
    like.
    """
    fitz = pytest.importorskip("fitz")
    import numpy as np

    from tactical_pdf_report import COVER_FOOT_LIFT, COVER_HEAD_DROP, PAGE_H

    _cover(tmp_path, _context())
    doc = fitz.open(tmp_path / "cover.pdf")
    pix = doc[0].get_pixmap(dpi=60)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[..., :3].astype(int)
    doc.close()
    rows = (np.abs(image - image[4, 4]).sum(axis=2) > 12).any(axis=1)

    # Points up from the foot of the sheet, which is how the layout is written.
    # Each rule is held 3pt clear: at 60dpi its own antialiased edge lands a
    # row inside the band and reads as the card reaching all the way down.
    scale = PAGE_H / len(rows)
    head, foot = PAGE_H - COVER_HEAD_DROP, COVER_FOOT_LIFT
    inside = [PAGE_H - i * scale for i, hit in enumerate(rows)
              if hit and foot + 3 < PAGE_H - i * scale < head - 3]
    assert inside, "nothing is drawn between the two rules"

    above, below = head - max(inside), min(inside) - foot
    assert abs(above - below) < 15, (
        f"{above:.0f}pt of air above the card and {below:.0f}pt below it")


def test_the_cover_carries_both_crests(tmp_path):
    """The report was the last part of the package without them."""
    context = _context()
    pdf = TacticalPDF.__new__(TacticalPDF)
    assert pdf._crest_reader(context["home_id"]) is not None
    assert pdf._crest_reader(context["away_id"]) is not None


def test_a_missing_crest_does_not_break_the_cover(tmp_path):
    pdf = TacticalPDF.__new__(TacticalPDF)
    assert pdf._crest_reader(None) is None
    assert pdf._crest_reader(999999) is None
    context = _context()
    context["home_id"] = 999999
    _cover(tmp_path, context)  # must not raise


def test_the_thesis_outranks_the_fixture_line():
    """The sentence carrying the report was set at the club names' size."""
    assert report.TYPE_THESIS > report.TYPE_FIXTURE
    assert report.TYPE_THESIS > report.TYPE_TITLE


def test_the_thesis_wraps_instead_of_touching_both_margins(tmp_path):
    pdf = _cover(tmp_path)
    c = pdf.canvas
    longest = max(
        c.stringWidth(line, "Helvetica-Bold", report.TYPE_THESIS)
        for line in ["Aston Villa created the better chances and lost. PSG needed",
                     "fewer of them and took them."]
    )
    assert longest <= 720


_THEME_PROBE = """
    import json
    import tactical_pdf_report as report
    print(json.dumps({"thesis": report.TYPE_THESIS, "fixture": report.TYPE_FIXTURE}))
"""


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_cover_builds_on_both_pages(theme, tmp_path):
    """Cover code is shared, so a change lands on both packages at once."""
    import os

    fixture = match_dir("PSG_vs_Aston_Villa_2-1")
    if not (fixture / "match_info.json").exists():
        pytest.skip("fixture has not been rendered")
    script = textwrap.dedent(f"""
        import json, sys
        from pathlib import Path
        sys.path.insert(0, {str(ROOT)!r})
        import pandas as pd
        from tactical_pdf_report import TacticalPDF, build_context
        out = Path({str(fixture)!r})
        info = json.loads((out / "match_info.json").read_text(encoding="utf-8"))
        context = build_context(
            pd.read_csv(out / "events.csv"), pd.read_csv(out / "xg.csv"),
            pd.read_csv(out / "team_advanced_metrics.csv"),
            pd.read_csv(out / "player_sequence_metrics.csv"), info)
        context["home_color"] = info.get("home_color", "#004170")
        context["away_color"] = info.get("away_color", "#7A003C")
        pdf = TacticalPDF(Path({str(tmp_path)!r}) / "c.pdf", context)
        pdf.cover(); pdf.canvas.save()
        print("OK")
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT,
        env={**os.environ, "MATCH_ANALYSIS_THEME": theme, "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    assert "OK" in completed.stdout


# --------------------------------------------------------------------------
# the artwork that used to be here
# --------------------------------------------------------------------------
#
# cover_art.py rendered a pass network for the old full-bleed cover, and four
# tests held its contrast floor and its pixel size. The cover is the comparison
# card now and nothing places that picture, so the module was deleted along
# with them rather than left running on every match to write a PNG no document
# opens on.
#
# match_article kept one reference — a fallback to cover_art.png when page one
# of the report cannot be rasterised — and it went too. It was the more
# dangerous half: a stale PNG in an output folder would have opened the article
# on a picture that appears nowhere in the document it fronts, which is exactly
# the drift the cover was made to close.


def test_nothing_reaches_for_the_deleted_artwork(tmp_path):
    """The fallback is gone, so a stale PNG cannot become the article's cover."""
    from match_article import _cover_image

    (tmp_path / "cover_art.png").write_bytes(b"not a real png")
    assert _cover_image(tmp_path) is None
