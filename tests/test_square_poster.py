"""The 1:1 summary page."""

import json

import matplotlib
import pandas as pd
import pytest
from conftest import match_dir
from PIL import Image

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def package():
    out = match_dir("Arsenal_vs_Coventry_3-0")
    if not (out / "match_info.json").exists():
        pytest.skip("Arsenal_vs_Coventry_3-0 has not been rendered")
    return out


def test_the_summary_is_square_and_carries_the_new_figures(package, tmp_path):
    from football_analysis.visuals import visual_redesign_full as v
    from football_analysis.visuals.poster_square import FILENAME, build_square_poster

    info = json.loads((package / "match_info.json").read_text(encoding="utf-8-sig"))
    events = pd.read_csv(package / "events.csv", low_memory=False)
    xg = pd.read_csv(package / "xg.csv")
    teams = pd.read_csv(package / "team_advanced_metrics.csv")
    v.configure_match(info, tmp_path)
    v.base.theme()
    path = build_square_poster(events, xg, teams, info, tmp_path)
    assert path.name == FILENAME
    with Image.open(path) as image:
        assert image.width == image.height == 2400


def test_the_figures_come_from_the_same_places_as_the_other_boards(package):
    from football_analysis.visuals.poster_square import _team_figures

    info = json.loads((package / "match_info.json").read_text(encoding="utf-8-sig"))
    events = pd.read_csv(package / "events.csv", low_memory=False)
    xg = pd.read_csv(package / "xg.csv")
    teams = pd.read_csv(package / "team_advanced_metrics.csv")
    side = _team_figures(events, xg, teams, info["home_id"], info["home_name"])
    row = teams[teams["team_id"].eq(info["home_id"])].iloc[0]
    assert side["xGoT"] == pytest.approx(float(row["xGoT"]))
    assert side["line_breaking"] == int(row["line_breaking_completed"])
    assert side["on_target"] <= side["shots"]
