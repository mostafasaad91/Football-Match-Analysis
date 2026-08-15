"""The two post-match posters, and the crest resolution behind them.

The contact sheets these replace failed in a way no test caught: panels were
placed against a grid that only described the axes, so every caption printed
into the heading of the row below it. The layout arithmetic is checked here
directly, because the failure is silent -- the sheet renders, it is simply
unreadable.

Rendering tests are marked slow-ish but stay in the default run: a poster is
the artefact most likely to be posted unreviewed, so a broken one has to fail
here rather than on a timeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import crests
import match_posters as mp


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

def test_rows_do_not_overlap_and_stay_inside_the_chrome():
    """Each row's axes must clear the next row's title, and the page's edges."""
    bounds = [mp._row_bounds(row) for row in range(3)]
    for (y0, y1) in bounds:
        assert y1 > y0
    # Header rule sits at 0.9215 and the footer rule at 0.0495.
    assert bounds[0][1] < 0.9215
    assert bounds[-1][0] > 0.0495
    for row in range(2):
        title_below = mp.ROW_TOP - (row + 1) * mp.ROW_PITCH
        # The caption hangs below the axes; leave it room above the next title.
        assert bounds[row][0] - title_below > 0.02, (
            f"row {row} caption would print into row {row + 1}'s title"
        )


def test_columns_do_not_overlap():
    ordered = sorted(mp.COL.values())
    for (_, left_end), (right_start, _) in zip(ordered, ordered[1:]):
        assert right_start > left_end


def test_poster_is_four_by_five():
    """Twitter crops anything taller in the timeline."""
    assert mp.H_PX / mp.W_PX == pytest.approx(1.25, abs=0.01)


# --------------------------------------------------------------------------
# coordinates
# --------------------------------------------------------------------------

def test_flip_puts_each_side_at_its_own_end():
    """A shot at x=95 is near the top attacking up, near the bottom flipped."""
    _, up = mp._xy([95.0], [50.0])
    _, down = mp._xy([95.0], [50.0], flip=True)
    assert up[0] > mp.PITCH_LENGTH * 0.9
    assert down[0] < mp.PITCH_LENGTH * 0.1


def test_coordinates_are_clamped_to_the_pitch():
    """Providers emit the odd out-of-range coordinate; an arrow is not clipped."""
    x, y = mp._xy([-8.0, 130.0], [-5.0, 140.0])
    half = mp.PITCH_WIDTH / 2
    assert np.all(y >= 0) and np.all(y <= mp.PITCH_LENGTH)
    assert np.all(x >= -half) and np.all(x <= half)


def test_lateral_axis_is_mirrored_consistently():
    """_xy, the control surface and the zone grid must all face the same way."""
    left, _ = mp._xy([50.0], [0.0])
    right, _ = mp._xy([50.0], [100.0])
    assert left[0] > right[0], "provider y increases toward negative display x"


# --------------------------------------------------------------------------
# indicators
# --------------------------------------------------------------------------

def _frames():
    out = Path(__file__).resolve().parent.parent / "output" / "PSG_vs_Aston_Villa_2-1"
    if not (out / "events.csv").exists():
        pytest.skip("no rendered fixture available")
    return (
        pd.read_csv(out / "events.csv"),
        pd.read_csv(out / "players.csv"),
        pd.read_csv(out / "xg.csv"),
        pd.read_csv(out / "team_advanced_metrics.csv"),
        pd.read_csv(out / "player_sequence_metrics.csv"),
        out,
    )


def test_indicator_rows_are_complete_and_weighted():
    events, _players, xg, team_metrics, _pm, _out = _frames()
    rows = mp.build_indicator_rows(
        events, xg, team_metrics, 304, 24, "PSG", "Aston Villa",
        ppda=(5.14, 8.69), control_shares=(21.0, 30.0, 49.0),
    )
    assert len(rows) == 16
    labels = [row[0] for row in rows]
    assert len(set(labels)) == 16, "an indicator is listed twice"
    for label, home_text, away_text, home_w, away_w in rows:
        assert home_text and away_text, f"{label} has no value"
        assert float(home_w) + float(away_w) > 0, f"{label} has an empty bar"


def test_ppda_bar_favours_the_side_that_pressed_harder():
    """PPDA is better when lower, so its bar has to be inverted."""
    events, _players, xg, team_metrics, _pm, _out = _frames()
    rows = mp.build_indicator_rows(
        events, xg, team_metrics, 304, 24, "PSG", "Aston Villa",
        ppda=(5.14, 8.69), control_shares=(21.0, 30.0, 49.0),
    )
    ppda = next(row for row in rows if row[0].startswith("PPDA"))
    _label, home_text, away_text, home_w, away_w = ppda
    assert home_text == "5.14" and away_text == "8.69"
    assert home_w > away_w, "the harder-pressing side must hold the wider bar"


def test_indicator_values_match_the_pipeline_frames():
    """The poster must never restate a number the report computed differently."""
    events, _players, xg, team_metrics, _pm, _out = _frames()
    rows = mp.build_indicator_rows(
        events, xg, team_metrics, 304, 24, "PSG", "Aston Villa",
        ppda=(5.14, 8.69), control_shares=(21.0, 30.0, 49.0),
    )
    by_label = {row[0]: row for row in rows}
    home_xg = float(xg[xg["team"].eq("PSG")].iloc[0]["xG"])
    assert by_label["Expected goals"][1] == f"{home_xg:.2f}"
    home_box = int(team_metrics[team_metrics["side"].eq("home")].iloc[0]["box_entries"])
    assert by_label["Box entries"][1] == str(home_box)


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

def test_all_four_posters_render_at_the_declared_size(tmp_path):
    from PIL import Image

    events, players, xg, team_metrics, player_metrics, _out = _frames()
    paths = mp.build_match_posters(
        events, xg, team_metrics, player_metrics, players,
        out_dir=tmp_path,
        home_id=304, away_id=24,
        home_name="PSG", away_name="Aston Villa",
        home_color="#2F7FD0", away_color="#D6216F",
        score="2 — 1",
        competition="UEFA SUPER CUP",
        allow_download=False,  # offline: the monogram fallback must carry it
    )
    assert len(paths) == 4
    names = {path.name for path in paths}
    assert names == {
        "match_poster_1_report.png",
        "match_poster_2_tactics.png",
        "match_poster_3_transitions.png",
        "match_poster_4_final_ball.png",
    }
    for path in paths:
        with Image.open(path) as image:
            assert image.size == (mp.W_PX, mp.H_PX)
        assert path.stat().st_size > 50_000, f"{path.name} rendered nearly empty"


def test_the_contact_sheets_are_gone():
    """They were replaced, not supplemented; two sources would drift apart."""
    root = Path(__file__).resolve().parent.parent
    assert not (root / "build_qa_contact_sheets.py").exists()
    assert not list(root.glob("output/*/qa_contact_sheet_*.png"))


def test_the_extra_tables_are_complete_and_weighted():
    """Posters 3 and 4 add twenty-four more indicators; none may be empty."""
    events, _players, xg, team_metrics, _pm, _out = _frames()
    tables = {
        "transition": mp.build_transition_rows(team_metrics),
        "shooting": mp.build_shooting_rows(xg, team_metrics, "PSG", "Aston Villa"),
    }
    for name, rows in tables.items():
        assert len(rows) == 12, name
        assert len(set(row[0] for row in rows)) == 12, f"{name} lists an indicator twice"
        for label, home_text, away_text, home_w, away_w in rows:
            assert home_text and away_text, f"{name}/{label} has no value"
            assert float(home_w) + float(away_w) > 0, f"{name}/{label} has an empty bar"


def test_no_indicator_is_repeated_across_the_four_posters():
    """Twenty-eight cells of one table are wasted if they restate another."""
    events, _players, xg, team_metrics, _pm, _out = _frames()
    labels = [
        row[0].lower()
        for rows in (
            mp.build_indicator_rows(events, xg, team_metrics, 304, 24, "PSG",
                                    "Aston Villa", (5.14, 8.69), (21.0, 30.0, 49.0)),
            mp.build_transition_rows(team_metrics),
            mp.build_shooting_rows(xg, team_metrics, "PSG", "Aston Villa"),
        )
        for row in rows
    ]
    # Expected goals appears on poster 1 as the headline and on poster 4 beside
    # the finishing it is measured against; nothing else may repeat.
    duplicated = {label for label in labels if labels.count(label) > 1}
    assert duplicated <= {"expected goals"}, duplicated


def test_inverted_bars_favour_the_better_side():
    """Three indicators are better when lower; their bars have to be flipped."""
    events, _players, xg, team_metrics, _pm, _out = _frames()
    rows = {row[0]: row for row in mp.build_transition_rows(team_metrics)}
    # PSG were exposed 14.6% of the time against Villa's 9.5%: Villa's bar wins.
    _label, home_text, away_text, home_w, away_w = rows["Rest-defence vulnerability"]
    assert home_text == "14.6%" and away_text == "9.5%"
    assert away_w > home_w

    shooting = {row[0]: row for row in
                mp.build_shooting_rows(xg, team_metrics, "PSG", "Aston Villa")}
    _label, home_text, away_text, home_w, away_w = shooting["Off target"]
    assert home_text == "6" and away_text == "8"
    assert home_w > away_w, "missing more often must not win the bar"


def test_the_restart_bars_share_one_scale():
    """Two panels normalised to their own peak cannot be read against each other."""
    events, _players, _xg, _tm, _pm, _out = _frames()
    peak = mp.set_piece_peak(events, 304, 24)
    both = [
        float(row.get("xG", 0.0) or 0.0)
        for team_id in (304, 24)
        for row in mp.set_piece_breakdown(events, team_id).values()
    ]
    assert peak == max(both)
