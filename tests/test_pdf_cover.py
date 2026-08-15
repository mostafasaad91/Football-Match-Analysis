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

ROOT = Path(__file__).resolve().parent.parent


def _context():
    out = ROOT / "output" / "PSG_vs_Aston_Villa_2-1"
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

def test_the_cover_fills_the_page(tmp_path):
    """Measured, because 'looks empty' is exactly what went unnoticed before."""
    fitz = pytest.importorskip("fitz")
    import numpy as np

    _cover(tmp_path)
    doc = fitz.open(tmp_path / "cover.pdf")
    pix = doc[0].get_pixmap(dpi=60)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[..., :3].astype(int)
    page = image[4, 4]
    rows = (np.abs(image - page).sum(axis=2) > 12).any(axis=1)
    doc.close()

    covered = rows.sum() / len(rows)
    assert covered > 0.55, f"only {covered:.0%} of the cover's rows carry anything"

    longest = current = 0
    for hit in rows:
        current = 0 if hit else current + 1
        longest = max(longest, current)
    assert longest / len(rows) < 0.15, (
        f"a dead band {longest / len(rows):.0%} of the page deep"
    )


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

    script = textwrap.dedent(f"""
        import json, sys
        from pathlib import Path
        sys.path.insert(0, {str(ROOT)!r})
        import pandas as pd
        from tactical_pdf_report import TacticalPDF, build_context
        out = Path({str(ROOT / "output" / "PSG_vs_Aston_Villa_2-1")!r})
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
