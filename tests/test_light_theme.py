"""The light page is a second publishing target, not a recolour of the first.

Everything here exists because a value that is right on #000000 is frequently
wrong on #F5F5F5, and the first light pass shipped several of them. The worst
was the contrast lift: written for the black page, it only ever searched
lightness upward, so on paper Manchester City's #6CABDD and Juventus' #DCE3EC
were both driven to pure white -- a contrast of 1.09 against the page they were
drawn on.

The theme is fixed when ``visualization_components`` is first imported, so the
light values cannot be read from this process. They are checked in a child.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import render_light

ROOT = Path(__file__).resolve().parent.parent


def _in_theme(theme: str, body: str):
    """Run a snippet in a child process pinned to one theme, return its JSON."""
    environment = {
        **os.environ,
        "MATCH_ANALYSIS_THEME": theme,
        "PYTHONIOENCODING": "utf-8",
    }
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        cwd=ROOT, env=environment, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


# Kits that clear the floor on one page and fail on the other.
BRIGHT_KITS = {"Man City sky": "#6CABDD", "Juventus silver": "#DCE3EC",
               "Norwich yellow": "#FFF200"}
DARK_KITS = {"PSG navy": "#004170", "Aston Villa claret": "#7A003C",
             "near-black": "#111111"}


_LIFT_PROBE = """
    import json
    from matplotlib import colors as mcolors
    import visual_redesign_full as v

    kits = %r
    out = {}
    for name, colour in kits.items():
        lifted = v.lift_to_floor(colour)
        out[name] = {
            "lifted": lifted,
            "contrast": v._contrast_on_bg(mcolors.to_rgb(lifted)),
            "again": v.lift_to_floor(lifted),
            "floor": v.MARK_CONTRAST_FLOOR,
        }
    print(json.dumps(out))
"""


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_every_kit_clears_the_floor_on_both_pages(theme):
    kits = {**BRIGHT_KITS, **DARK_KITS}
    result = _in_theme(theme, _LIFT_PROBE % kits)
    for name, row in result.items():
        assert row["contrast"] >= row["floor"], (
            f"{name} measures {row['contrast']:.2f} on the {theme} page"
        )


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_lift_is_idempotent_on_both_pages(theme):
    """A second call must be a no-op, or repeated configuration drifts."""
    result = _in_theme(theme, _LIFT_PROBE % {**BRIGHT_KITS, **DARK_KITS})
    for name, row in result.items():
        assert row["again"] == row["lifted"], f"{name} moved on a second call"


def test_a_bright_kit_is_darkened_on_paper_not_whitened():
    """The direction bug: upward-only search drove these to #ffffff at 1.09."""
    result = _in_theme("light", _LIFT_PROBE % BRIGHT_KITS)
    for name, row in result.items():
        assert row["lifted"].lower() != "#ffffff", f"{name} was whitened out"


def test_a_dark_kit_is_left_alone_on_paper():
    """Navy and claret already read on #F5F5F5; touching them loses the kit."""
    result = _in_theme("light", _LIFT_PROBE % DARK_KITS)
    assert result["PSG navy"]["lifted"] == "#004170"
    assert result["Aston Villa claret"]["lifted"] == "#7A003C"


def test_the_dark_page_still_lifts_upward():
    """The fix must not change what the black package already publishes."""
    result = _in_theme("dark", _LIFT_PROBE % DARK_KITS)
    assert result["PSG navy"]["lifted"] == "#0069b4"
    assert result["Aston Villa claret"]["lifted"] == "#c70062"


_POSTER_PROBE = """
    import json
    import match_posters as mp
    from visualization_components import contrast_ratio
    print(json.dumps({
        "bg": mp.BG,
        "ink": mp.INK,
        "ink_contrast": contrast_ratio(mp.INK, mp.BG),
        "text_contrast": contrast_ratio(mp.TEXT, mp.BG),
        "muted_contrast": contrast_ratio(mp.MUTED, mp.BG),
        "neutral_contrast": contrast_ratio(mp.NEUTRAL, mp.BG),
    }))
"""


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_poster_chrome_reads_on_its_own_page(theme):
    result = _in_theme(theme, _POSTER_PROBE)
    # Body text at the WCAG normal-text minimum.
    assert result["text_contrast"] >= 4.5, result
    # Captions and micro-labels are small but still have to be legible.
    assert result["muted_contrast"] >= 3.0, result
    assert result["neutral_contrast"] >= 2.5, result
    # INK marks the goal star, the average-height rule and scatter edges; it
    # exists to separate from the page, so it is held to the graphics floor.
    assert result["ink_contrast"] >= 3.0, result


def test_the_two_pages_do_not_share_a_background():
    dark = _in_theme("dark", _POSTER_PROBE)
    light = _in_theme("light", _POSTER_PROBE)
    assert dark["bg"] != light["bg"]
    assert dark["ink"] != light["ink"]


# --------------------------------------------------------------------------
# crest plate
# --------------------------------------------------------------------------

def test_the_plate_opposes_the_page_not_the_crest():
    """A silver crest on paper needs a dark plate, not another light one."""
    import numpy as np

    import crests

    silver = np.full((8, 8, 4), 230, dtype=np.uint8)
    silver[..., 3] = 255
    navy = np.zeros((8, 8, 4), dtype=np.uint8)
    navy[..., 2] = 90
    navy[..., 3] = 255

    assert crests.plate_colour(navy, "#000000") == crests.PLATE_ON_DARK_PAGE
    assert crests.plate_colour(silver, "#F5F5F5") == crests.PLATE_ON_LIGHT_PAGE


def test_a_crest_that_reads_on_its_page_gets_no_plate():
    import numpy as np

    import crests

    silver = np.full((8, 8, 4), 230, dtype=np.uint8)
    silver[..., 3] = 255
    assert not crests.needs_plate(silver, "#000000")
    assert crests.needs_plate(silver, "#F5F5F5")


def test_the_plate_is_decided_per_pixel_not_on_the_crest_mean():
    """A crest can average light and still separate: Villa's claret border."""
    import numpy as np

    import crests

    # Half near-white, half deep claret. The mean is light enough to look
    # unreadable on paper; half the crest reads perfectly.
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    image[..., 3] = 255
    image[:4] = [245, 245, 245, 255]
    image[4:] = [122, 0, 60, 255]
    assert crests.readable_fraction(image, "#F5F5F5") >= 0.45
    assert not crests.needs_plate(image, "#F5F5F5")


# --------------------------------------------------------------------------
# the light pass itself
# --------------------------------------------------------------------------

def test_light_output_is_a_subfolder_so_neither_run_clobbers_the_other():
    assert render_light.light_dir("output/x").name == "light"
    assert render_light.light_dir("output/x").parent.name == "x"


def test_missing_inputs_are_named_rather_than_raised(tmp_path):
    absent = render_light.missing_inputs(tmp_path)
    assert "events.csv" in absent and "match_info.json" in absent


def test_a_light_run_cannot_spawn_another(tmp_path, monkeypatch):
    """Without the guard the child would render a light copy of the light copy."""
    seen = {}

    class _Completed:
        returncode = 1
        stdout = stderr = ""

    def _fake_run(_cmd, env=None, **_kw):
        seen.update(env or {})
        return _Completed()

    for name in render_light._REQUIRED:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(render_light.subprocess, "run", _fake_run)
    render_light.render_light_package(tmp_path)
    assert seen["MATCH_ANALYSIS_THEME"] == "light"
    assert seen["MATCH_ANALYSIS_LIGHT_COPY"] == "0"
    # Piped stdout is cp1252 on Windows and the child prints accented names.
    assert seen["PYTHONIOENCODING"] == "utf-8"


def test_a_failed_light_run_returns_none_rather_than_raising(tmp_path, monkeypatch):
    """The dark package is already finished; it must survive this failing."""
    for name in render_light._REQUIRED:
        (tmp_path / name).write_text("{}", encoding="utf-8")

    def _boom(*_a, **_kw):
        raise OSError("no interpreter")

    monkeypatch.setattr(render_light.subprocess, "run", _boom)
    assert render_light.render_light_package(tmp_path) is None


def test_the_child_refuses_to_render_under_the_dark_theme():
    """Rendering without the env var would write the black set into light/."""
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import render_light, pathlib, sys
            try:
                render_light._render_here(pathlib.Path("."))
            except RuntimeError as error:
                print("REFUSED", error)
                sys.exit(0)
            sys.exit(1)
        """)],
        cwd=ROOT,
        env={**os.environ, "MATCH_ANALYSIS_THEME": "dark", "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    assert "REFUSED" in completed.stdout
