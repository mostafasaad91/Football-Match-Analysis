"""The small type on the cover has to be readable at arm's length.

Every grey on the cover was borrowed from the body pages, where 6.5pt inside a
dense column reads as the footnote it is. A cover is looked at from further
away and has almost nothing on it: the row labels are the only thing saying
what each bar measures, and the byline and the source line are the only text
under the card.

NEUTRAL measures 3.0:1 against the black page — under the 4.5:1 floor before
the letter-spacing thins the strokes any further — and it was carrying the
byline, the source line, the competition line and the running page footer on
all seventy-four pages.

Contrast is computed here rather than asserted as a hex, so changing a grey
fails on what it does to a reader rather than on what it is called.
"""

import re
from pathlib import Path

import pytest

import tactical_pdf_report as pdf

ROOT = Path(__file__).resolve().parent.parent

# WCAG AA for text. The cover greys are held above it rather than at it,
# because letter-spaced small caps lose stroke weight the ratio cannot see.
BODY_FLOOR = 4.5
COVER_FLOOR = 7.0


def _channel(value: float) -> float:
    value /= 255.0
    return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4


def _luminance(colour) -> float:
    r, g, b = colour.red * 255, colour.green * 255, colour.blue * 255
    return (0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b))


def contrast(a, b) -> float:
    first, second = _luminance(a), _luminance(b)
    high, low = max(first, second), min(first, second)
    return (high + 0.05) / (low + 0.05)


def test_the_cover_greys_clear_the_floor_they_were_added_for():
    for name in ("COVER_LABEL", "COVER_META"):
        ratio = contrast(getattr(pdf, name), pdf.BG)
        assert ratio >= COVER_FLOOR, f"{name} is {ratio:.1f}:1 on this page"


def test_the_page_text_colours_are_readable():
    """TEXT and MUTED carry the report; both have to clear AA."""
    for name in ("TEXT", "MUTED"):
        ratio = contrast(getattr(pdf, name), pdf.BG)
        assert ratio >= BODY_FLOOR, f"{name} is {ratio:.1f}:1 on this page"


def test_the_cover_labels_are_larger_than_the_body_footnote():
    """They name the metric each bar measures; they are not a footnote."""
    assert pdf.TYPE_COVER_LABEL > pdf.TYPE_MICRO
    assert pdf.TYPE_COVER_META > pdf.TYPE_MICRO
    assert pdf.TYPE_COVER_LABEL >= pdf.TYPE_COVER_META


def test_nothing_on_the_cover_is_drawn_in_the_faintest_grey():
    """NEUTRAL is below the floor on the dark page, so the cover may not use it.

    Read off the source rather than the render: a colour that is never set
    cannot be measured in a PDF, and the point is to keep it out of the method
    rather than to notice it once it has shipped.
    """
    import inspect

    source = "".join(inspect.getsource(fn) for fn in (
        pdf.TacticalPDF.cover, pdf.TacticalPDF._cover_row,
        pdf.TacticalPDF._finish))
    # Comments stripped: the note explaining why the colour was replaced names
    # it, and a test that reads that as a use would forbid saying why.
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    assert "NEUTRAL" not in code, (
        "the cover sets NEUTRAL, which is 3.0:1 against the black page")


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_both_themes_clear_the_floor(theme):
    """The greys are declared per theme, so both branches need checking."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        sys.path.insert(0, {str(ROOT / "tests")!r})
        import tactical_pdf_report as pdf
        from test_cover_legibility import contrast, COVER_FLOOR, BODY_FLOOR
        for name in ("COVER_LABEL", "COVER_META"):
            ratio = contrast(getattr(pdf, name), pdf.BG)
            assert ratio >= COVER_FLOOR, (name, round(ratio, 2))
        for name in ("TEXT", "MUTED", "COVER_FIGURE_DIM"):
            ratio = contrast(getattr(pdf, name), pdf.BG)
            assert ratio >= BODY_FLOOR, (name, round(ratio, 2))
        print("OK")
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT,
        env={**os.environ, "MATCH_ANALYSIS_THEME": theme,
             "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    assert "OK" in completed.stdout


def test_the_row_labels_name_every_metric_on_the_card():
    """A label that is clearer but wrong is not an improvement.

    "xT" keeps its own case: it is the name of the metric, not a word that has
    been left uncapitalised.
    """
    for label, key, _shape in pdf.TacticalPDF.COVER_ROWS:
        assert key
        rest = label.replace("xT", "")
        assert rest == rest.upper(), label
        assert not re.search(r"[a-z]", rest), label


def test_the_cover_is_set_in_the_two_faces_it_declares():
    """Two faces, because the two jobs want opposite things.

    The display line is large enough that weight is not what makes it legible,
    and a condensed grotesque is what gives the page its character. The small
    letter-spaced caps are the opposite case: they were the unreadable part and
    weight is the whole fix, so they take a face with a real bold rather than
    the variable font's light default instance.

    Read off the source so a setFont that slips back to the built-in face fails
    here rather than in a rendered page.
    """
    import inspect

    source = "".join(inspect.getsource(fn) for fn in (
        pdf.TacticalPDF.cover, pdf.TacticalPDF._cover_row,
        pdf.TacticalPDF._cover_badges, pdf.TacticalPDF._finish))
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    assert '"Helvetica' not in code, (
        "the cover names Helvetica directly instead of COVER_DISPLAY/COVER_TEXT")

    for name in ("COVER_DISPLAY", "COVER_TEXT"):
        assert getattr(pdf, name), name


def test_the_registration_chain_ends_somewhere_that_always_exists():
    """A machine with none of the candidates still has to build the report."""
    assert pdf._register_first((), "Helvetica-Bold") == "Helvetica-Bold"
    assert pdf._register_first(
        (("NotAFace", "definitely-not-a-font.ttf"),), "Helvetica") == "Helvetica"


def test_the_trailing_figure_is_quieter_than_the_leader_and_still_readable():
    """Half of every row's comparison was drawn at 3.0:1.

    Greying the side that did not lead a row is the right emphasis and the
    wrong colour: NEUTRAL sits under the floor on both pages, so eight of the
    sixteen figures on the card were the hard ones to read.
    """
    dim = contrast(pdf.COVER_FIGURE_DIM, pdf.BG)
    assert dim >= BODY_FLOOR, f"the trailing figure is {dim:.1f}:1"
    # ...and still visibly quieter than the label ring above it.
    assert dim < contrast(pdf.COVER_LABEL, pdf.BG)
