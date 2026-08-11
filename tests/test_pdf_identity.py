"""Identity rules for the report: one typeface, one accent system, no
mid-word breaks. Each of these was a defect found by reading the built PDF.
"""

import re

import pytest

import tactical_pdf_report as pdf
from player_radar import GROUPS, display_label


# ── typography ────────────────────────────────────────────────────────────────

def test_no_style_is_set_in_times():
    """The commentary was Times while every embedded visual is sans, so a page
    carried two unrelated families."""
    source = pdf.__file__
    with open(source, encoding="utf-8") as handle:
        body = handle.read()
    offenders = re.findall(r'fontName\s*=\s*"(Times[^"]*)"', body)
    assert not offenders, f"Times still in use: {sorted(set(offenders))}"


def test_every_named_font_is_one_family():
    with open(pdf.__file__, encoding="utf-8") as handle:
        body = handle.read()
    families = {
        name.split("-")[0]
        for name in re.findall(r'fontName\s*=\s*"([^"]+)"', body)
        + re.findall(r'setFont\(\s*"([^"]+)"', body)
    }
    assert families == {"Helvetica"}, families


# ── colour ────────────────────────────────────────────────────────────────────

def test_the_amber_accent_is_gone():
    """It ran to 1,398 characters against 648 for both kit colours combined."""
    with open(pdf.__file__, encoding="utf-8") as handle:
        body = handle.read()
    assert "#FFC23C" not in body.upper()


def test_the_brand_colour_is_distinct_from_both_default_team_colours():
    def rgb(colour):
        return (colour.red, colour.green, colour.blue)

    brand = rgb(pdf.BRAND)
    for other in (pdf.HOME, pdf.AWAY):
        distance = sum((a - b) ** 2 for a, b in zip(brand, rgb(other))) ** 0.5
        assert distance > 0.25, "brand colour is too close to a team colour"


def test_structural_accent_is_neutral():
    """FOCUS marks sections, not sides, so it must not read as a team colour."""
    r, g, b = pdf.FOCUS.red, pdf.FOCUS.green, pdf.FOCUS.blue
    assert max(r, g, b) - min(r, g, b) < 0.12, "structural accent is saturated"


def test_the_commentary_band_shares_the_visual_s_ground():
    """PANEL under the commentary put #0A0A0A against the image's #000000 and
    ruled a seam across every visual page."""
    with open(pdf.__file__, encoding="utf-8") as handle:
        body = handle.read()
    marker = "c.rect(0, 0, PAGE_W, VISUAL_NOTE_H, fill=1, stroke=0)"
    assert marker in body
    before = body[: body.index(marker)]
    fill = before.rsplit("c.setFillColor(", 1)[1].split(")")[0]
    assert fill == "BG", f"commentary band drawn on {fill}, not the page ground"


def test_the_page_ground_is_pure_black():
    assert (pdf.BG.red, pdf.BG.green, pdf.BG.blue) == (0, 0, 0)


# ── type scale ────────────────────────────────────────────────────────────────

SCALE_NAMES = (
    "TYPE_DISPLAY", "TYPE_TITLE", "TYPE_SECTION",
    "TYPE_BODY", "TYPE_CAPTION", "TYPE_MICRO",
)
# The cover's lead statistic is a named, deliberate exception; `minor` and
# `major` are the locals that carry those two constants to the draw call.
ALLOWED_SIZE_NAMES = set(SCALE_NAMES) | {
    "TYPE_LEAD_MINOR", "TYPE_LEAD_MAJOR", "minor", "major",
}


def test_the_scale_has_six_steps_each_clearly_apart():
    sizes = [getattr(pdf, name) for name in SCALE_NAMES]
    assert sizes == sorted(sizes, reverse=True), sizes
    for larger, smaller in zip(sizes, sizes[1:]):
        assert larger / smaller >= 1.15, f"{larger} and {smaller} are too close to tell apart"


def test_no_font_size_is_a_bare_number():
    """23 distinct literals is what a document looks like when every element
    was sized on its own."""
    with open(pdf.__file__, encoding="utf-8") as handle:
        body = handle.read()
    literals = re.findall(r'setFont\(\s*"[A-Za-z-]+"\s*,\s*([\d.]+)\s*\)', body)
    literals += re.findall(r"fontSize=([\d.]+)", body)
    assert not literals, f"font sizes not on the scale: {sorted(set(literals))}"


def test_every_size_reference_is_a_scale_name():
    with open(pdf.__file__, encoding="utf-8") as handle:
        body = handle.read()
    used = set(re.findall(r'setFont\(\s*"[A-Za-z-]+"\s*,\s*([A-Za-z_]+)\s*\)', body))
    used |= set(re.findall(r"fontSize=([A-Za-z_]+)", body))
    stray = used - ALLOWED_SIZE_NAMES
    assert not stray, f"sizes off the scale: {sorted(stray)}"


# ── shared measure ────────────────────────────────────────────────────────────

def test_the_embedded_visual_uses_the_text_margin():
    """Images bled to 4pt while the commentary began at 42pt, so a wide visual
    and its own analysis sat on two different left edges."""
    with open(pdf.__file__, encoding="utf-8") as handle:
        body = handle.read()
    assert "margin_x = TEXT_MARGIN" in body
    assert pdf.TEXT_MARGIN == 42


# ── score glyph ───────────────────────────────────────────────────────────────

def test_the_score_uses_the_same_dash_as_the_visuals():
    with open(pdf.__file__, encoding="utf-8") as handle:
        body = handle.read()
    assert '"score": f"{home_goals} — {away_goals}"' in body


# ── radar labels ──────────────────────────────────────────────────────────────

# Whole words the labels are built from. A break is mid-word exactly when
# gluing the two lines back together with no space reproduces one of these —
# "CLEAR" + "ANCES" does, "TACKLES" + "WON" does not.
METRIC_WORDS = {
    "clearances", "recoveries", "interceptions", "intercepts", "tackles",
    "blocks", "passes", "dribbles", "duels", "aerials", "assists", "shots",
    "goals", "created", "actions", "buildup", "chain", "completions", "balls",
    "progressive", "contribution", "won",
}


def test_no_radar_label_is_drawn_broken_inside_a_word():
    """CLEAR/ANCES, RECOV/ERIES and INTERCEP/TIONS all rendered split."""
    broken = []
    for _group, _colour, metrics in GROUPS:
        for key in metrics:
            drawn = display_label(key)
            if "\n" not in drawn:
                continue
            head, _, tail = drawn.partition("\n")
            glued = (head + tail).replace(" ", "").lower()
            if glued in METRIC_WORDS:
                broken.append((key, drawn))
    assert not broken, f"labels break inside a word: {broken!r}"


def test_the_detector_would_have_caught_the_original_defect():
    """Guards the guard: with the overrides removed these must be flagged."""
    for head, tail in (("CLEAR", "ANCES"), ("RECOV", "ERIES"), ("INTERCEP", "TIONS")):
        assert (head + tail).lower() in METRIC_WORDS


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("Clear\nances", "CLEARANCES"),
        ("Recov\neries", "RECOVERIES"),
        ("Intercep\ntions", "INTERCEPTS"),
    ],
)
def test_the_three_broken_labels_render_whole(key, expected):
    assert display_label(key) == expected


def test_labels_with_a_real_space_still_wrap_there():
    assert display_label("Tackles\nwon") == "TACKLES\nWON"
    assert display_label("Aerials\nwon") == "AERIALS\nWON"


def test_every_override_key_still_exists_in_the_layout():
    """An override for a renamed metric would silently stop applying."""
    live = {m for _g, _c, ms in GROUPS for m in ms}
    missing = set(pdf_overrides()) - live
    assert not missing, f"override keys no longer in the layout: {missing}"


def pdf_overrides():
    from player_radar import _LABEL_OVERRIDES

    return _LABEL_OVERRIDES
