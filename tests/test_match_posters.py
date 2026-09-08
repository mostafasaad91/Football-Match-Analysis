"""The four post-match posters, and the crest resolution behind them.

This file used to test ``match_posters``: its panel grid, its coordinate
helpers, its indicator tables. None of that shipped. ``match_posters`` ended
with ``from poster_dashboard import build_match_posters``, which replaced the
name every caller imported, so the module's own eleven hundred lines of
drawing code had been unreachable since the redesign while the tests around
them stayed green. The module is gone; what is left here is what still runs.

Rendering tests stay in the default run: a poster is the artefact most likely
to be posted unreviewed, so a broken one has to fail here rather than on a
timeline.
"""

from pathlib import Path

import numpy as np
import pytest

import crests
import poster_dashboard as pd_
from conftest import match_dir


def _frames():
    import pandas as pd

    out = match_dir("Chelsea_vs_Brighton_4-3")
    if not (out / "events.csv").exists():
        pytest.skip("no rendered fixture available")
    return (
        pd.read_csv(out / "events.csv", low_memory=False),
        pd.read_csv(out / "players.csv"),
        pd.read_csv(out / "xg.csv"),
        pd.read_csv(out / "team_advanced_metrics.csv"),
        pd.read_csv(out / "player_sequence_metrics.csv"),
        out,
    )


# --------------------------------------------------------------------------
# crests
# --------------------------------------------------------------------------

def test_crest_url_is_addressed_by_the_provider_team_id():
    """No name matching: the id on the event row is the id on the CDN."""
    assert crests.CREST_URL.format(team_id=304).endswith("/304.png")


def test_a_dark_crest_gets_a_plate_and_a_light_one_does_not():
    dark = np.zeros((16, 16, 4), dtype=np.uint8)
    dark[..., 3] = 255
    light = np.full((16, 16, 4), 255, dtype=np.uint8)
    assert crests.needs_plate(dark, "#000000")
    assert not crests.needs_plate(light, "#000000")


def test_transparent_pixels_do_not_count_toward_brightness():
    """A crest is mostly transparent corner; including it calls everything light."""
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    image[6:10, 6:10, 3] = 255  # a small opaque black mark, rest transparent
    assert crests.needs_plate(image, "#000000")


def test_missing_crest_falls_back_without_raising(tmp_path, monkeypatch):
    """A poster must build for a club whose crest cannot be fetched."""
    monkeypatch.setattr(crests, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(crests, "_MEMO", {})
    monkeypatch.setattr(crests, "download_crest", lambda *a, **k: None)
    assert crests.crest_image(999999) is None


def test_a_failed_download_never_poisons_the_cache(tmp_path, monkeypatch):
    """An HTML error body must not land in the cache under a .png name."""
    import urllib.request

    class _Response:
        def read(self):
            return b"<html>not an image</html>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(crests, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response())
    assert crests.download_crest(4242) is None
    assert not (tmp_path / "4242.png").exists()


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _render(builder, tmp_path):
    import visual_redesign_full as visual

    events, players, xg, team_metrics, player_metrics, out = _frames()
    import json

    info = json.loads((out / "match_info.json").read_text(encoding="utf-8-sig"))
    visual.configure_match(info, tmp_path)
    return builder(
        events, xg, team_metrics, player_metrics, players,
        out_dir=tmp_path,
        home_id=info["home_id"], away_id=info["away_id"],
        home_name=info["home_name"], away_name=info["away_name"],
        home_color=visual.HOME, away_color=visual.AWAY,
        score=info["score"],
        competition="MATCH ANALYSIS",
        allow_download=False,  # offline: the monogram fallback must carry it
    )


def test_every_poster_renders_at_four_by_five(tmp_path):
    """One set, at the ratio X shows uncropped, and none of them near-empty.

    There were briefly two sets. The second was built to be read on a phone,
    then widened back to the reference six-panel grid so four boards could
    cover the match -- at which point it was the same boards twice, into the
    same folder, under two names.
    """
    from PIL import Image

    paths = _render(pd_.build_match_posters, tmp_path)
    assert {path.name for path in paths} == set(pd_.POSTERS)
    for path in paths:
        with Image.open(path) as image:
            assert image.height / image.width == pytest.approx(1.25, abs=0.01), path.name
        assert path.stat().st_size > 50_000, f"{path.name} rendered nearly empty"


def test_there_is_only_one_poster_set():
    """A second builder writing near-identical boards is a duplicate, not a variant."""
    assert not hasattr(pd_, "build_thread_posters")
    assert not hasattr(pd_, "THREAD_POSTERS")


def test_the_contact_sheets_are_gone():
    """They were replaced, not supplemented; two sources would drift apart."""
    root = Path(__file__).resolve().parent.parent
    assert not (root / "build_qa_contact_sheets.py").exists()
    assert not list(root.glob("output/*/qa_contact_sheet_*.png"))


def test_the_replaced_poster_module_is_gone():
    """Its last line re-exported the live builder, so nothing in it could run.

    Left on disk it kept three test files pointed at code that never drew a
    pixel, and a reader looking for the poster layout found eleven hundred
    lines of the wrong one first.
    """
    root = Path(__file__).resolve().parent.parent
    assert not (root / "match_posters.py").exists()
