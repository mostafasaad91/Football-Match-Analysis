"""Every number the article prints must exist in the frames it came from.

The earlier tests assert a handful of named claims — the title, the pull
quotes, the upset framing. That leaves most of the prose unverified: the
generator has forty-five paragraph slots and sixty branches, and a typo in an
f-string produces a confident sentence with a number nothing supports.

So this reads the finished article back, pulls out every figure in it, and
requires each one to be findable in the fixture's own data. It is the same
question a sub-editor asks — where did this come from — asked mechanically.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from match_article import build_article
from conftest import match_dir

ROOT = Path(__file__).resolve().parent.parent
MATCHES = ["Arsenal_vs_Man_City_3-0", "PSG_vs_Aston_Villa_2-1"]

# Figures that belong to the language rather than to the data: percentages of
# a whole, ordinals, and the handful of counts the prose spells out.
PROSE_NUMBERS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "14",
                 "20", "90", "100"}


def _frames(match):
    out = match_dir(match)
    if not (out / "match_info.json").exists():
        pytest.skip(f"{match} has not been rendered")
    return (
        pd.read_csv(out / "events.csv"),
        pd.read_csv(out / "xg.csv"),
        pd.read_csv(out / "team_advanced_metrics.csv"),
        pd.read_csv(out / "player_sequence_metrics.csv"),
        json.loads((out / "match_info.json").read_text(encoding="utf-8")),
        out,
    )


def _known_values(xg, team_metrics, player_metrics, info) -> set[str]:
    """Every number the frames can justify, in the forms the prose prints them.

    A figure is accepted if it appears at any of the precisions the article
    uses, and derived values — ratios, shares, sums — are computed here the same
    way the generator computes them, so a wrong derivation still fails.
    """
    values: set[float] = set()

    def add(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if pd.isna(value):
            return
        values.add(round(value, 3))

    for frame in (xg, team_metrics, player_metrics):
        for column in frame.columns:
            for cell in pd.to_numeric(frame[column], errors="coerce").dropna():
                add(cell)

    numeric_xg = xg.select_dtypes("number")
    numeric_tm = team_metrics.select_dtypes("number")
    for column in numeric_xg.columns:
        add(numeric_xg[column].sum())
        if len(numeric_xg[column]) == 2:
            add(numeric_xg[column].iloc[0] - numeric_xg[column].iloc[1])
            add(numeric_xg[column].iloc[1] - numeric_xg[column].iloc[0])
    for column in numeric_tm.columns:
        add(numeric_tm[column].sum())

    # Ratios and shares the prose builds from two columns.
    for frame in (numeric_xg, numeric_tm):
        for left in frame.columns:
            for right in frame.columns:
                for i in range(len(frame)):
                    a, b = frame[left].iloc[i], frame[right].iloc[i]
                    if pd.notna(a) and pd.notna(b) and b:
                        add(a / b)
                        add(100.0 * a / b)

    rendered = set()
    for value in values:
        for text in (f"{value:.0f}", f"{value:.1f}", f"{value:.2f}", f"{value:.3f}"):
            rendered.add(text.lstrip("+"))
            rendered.add(text.replace("-", "−"))
    return rendered


def _figures(text: str) -> list[str]:
    """Every number printed in the prose, as it appears."""
    return re.findall(r"(?<![\w/])[−-]?\d+(?:\.\d+)?(?=[^\w.]|$)", text)


@pytest.mark.parametrize("match", MATCHES)
def test_every_figure_in_the_prose_comes_from_the_frames(match):
    events, xg, team_metrics, player_metrics, info, out = _frames(match)
    article = build_article(events, xg, team_metrics, player_metrics, info, out)
    known = _known_values(xg, team_metrics, player_metrics, info)

    unsupported = []
    for section in article.sections:
        for paragraph in section.paragraphs:
            for figure in _figures(paragraph):
                bare = figure.lstrip("−-")
                if bare in PROSE_NUMBERS or figure in known or bare in known:
                    continue
                unsupported.append((section.heading, figure, paragraph[:120]))

    assert not unsupported, "\n".join(
        f"{heading}: {figure!r} — {snippet}" for heading, figure, snippet in unsupported
    )


@pytest.mark.parametrize("match", MATCHES)
def test_every_figure_in_a_pull_quote_comes_from_the_frames(match):
    events, xg, team_metrics, player_metrics, info, out = _frames(match)
    article = build_article(events, xg, team_metrics, player_metrics, info, out)
    known = _known_values(xg, team_metrics, player_metrics, info)

    unsupported = []
    for section in article.sections:
        for figure in _figures(section.pull_quote or ""):
            bare = figure.lstrip("−-")
            if bare in PROSE_NUMBERS or figure in known or bare in known:
                continue
            unsupported.append((section.heading, figure, section.pull_quote))
    assert not unsupported, unsupported


@pytest.mark.parametrize("match", MATCHES)
def test_the_scoreline_in_the_standfirst_is_the_scoreline(match):
    events, xg, team_metrics, player_metrics, info, out = _frames(match)
    article = build_article(events, xg, team_metrics, player_metrics, info, out)

    def goals(team):
        row = xg[xg["team"].astype(str).str.lower().eq(str(team).lower())]
        return int(float(row.iloc[0]["goals"])) if not row.empty else None

    home, away = goals(info["home_name"]), goals(info["away_name"])
    assert f"{home}–{away}" in article.standfirst + article.strap, (
        article.standfirst, article.strap, home, away)


def test_the_checker_would_catch_an_invented_number():
    """A guard on the guard: the matcher must not accept anything at all."""
    events, xg, team_metrics, player_metrics, info, out = _frames(MATCHES[0])
    known = _known_values(xg, team_metrics, player_metrics, info)
    assert "8317.42" not in known
    assert _figures("Arsenal produced 1.88 xG from 9 shots") == ["1.88", "9"]
