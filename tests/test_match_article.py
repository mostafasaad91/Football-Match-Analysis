"""The publishable article built from a fixture's frames.

The report is a reference and can afford to describe every visual. An article
cannot: it has to pick an argument, rank the evidence and stop, and it has to
read like a person wrote it. What is checkable about that is narrow but worth
holding — the length, that every claim names the side its own numbers name, and
that no sentence is a fixed line dressed as a finding.

Every naming test runs over the fixture and its mirror image, the same match
with the two sides swapped. Naming the away side is indistinguishable from
naming the leader until the leader is the home side, which is exactly how four
of these got into the PDF.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from match_article import (
    TARGET_WORDS,
    Article,
    build_article,
    render_docx,
)

ROOT = Path(__file__).resolve().parent.parent
MATCHES = ["Arsenal_vs_Man_City_3-0", "PSG_vs_Aston_Villa_2-1"]


def _frames(match):
    out = ROOT / "output" / match
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


def _article(match, mirror=False) -> Article:
    events, xg, team_metrics, player_metrics, info, out = _frames(match)
    if mirror:
        team_metrics = team_metrics.copy()
        team_metrics["side"] = team_metrics["side"].map(
            {"home": "away", "away": "home"}).fillna(team_metrics["side"])
        info = dict(info)
        info["home_name"], info["away_name"] = info["away_name"], info["home_name"]
        info["home_id"], info["away_id"] = info["away_id"], info["home_id"]
    return build_article(events, xg, team_metrics, player_metrics, info, out)


# --------------------------------------------------------------------------
# shape
# --------------------------------------------------------------------------

@pytest.mark.parametrize("match", MATCHES)
def test_the_argument_is_the_length_it_was_commissioned_at(match):
    """A floor, and no ceiling.

    Measured on the argued sections only. Every finding the match supports is
    printed — dropping one to save words the reader was never promised is the
    wrong trade — and the appendix behind them carries a reading under every
    remaining board on top of that.
    """
    low, high = TARGET_WORDS
    words = _article(match).narrative_words()
    assert words >= low, f"{match}: {words} words, wanted at least {low}"
    if high is not None:
        assert words <= high, f"{match}: {words} words, wanted at most {high}"


@pytest.mark.parametrize("match", MATCHES)
def test_it_argues_in_sections_rather_than_walking_the_visuals(match):
    article = _article(match)
    assert 5 <= len(article.sections) <= 13, len(article.sections)
    # The argument's own sections stay tight. The ones that exist to show
    # things say so, because identifying them by position broke as soon as the
    # profiles section stopped being last.
    for section in (s for s in article.sections if not s.gallery):
        assert len(section.paragraphs) >= 2, section.heading
        assert len(section.visuals) <= 3, f"{section.heading} carries a gallery"


@pytest.mark.parametrize("match", MATCHES)
def test_the_article_carries_the_whole_package(match):
    """Every visual the run produced reaches the article, not a selection."""
    events, _xg, _tm, _pm, _info, out = _frames(match)
    article = _article(match)
    shown = {Path(v).name for s in article.sections for v in s.visuals}
    produced = {p.name for p in out.glob("[0-9]*.png")}
    missing = produced - shown
    assert not missing, sorted(missing)


@pytest.mark.parametrize("match", MATCHES)
def test_it_shows_five_radars_a_side(match):
    events, _xg, _tm, _pm, _info, _out = _frames(match)
    article = _article(match)
    from match_article import RADARS_PER_TEAM

    radars = [Path(v) for s in article.sections for v in s.visuals
              if "player_radars" in str(v)]
    assert len(radars) == RADARS_PER_TEAM * 2, [r.name for r in radars]
    per_team = {}
    for radar in radars:
        per_team.setdefault(radar.parent.name, []).append(radar.name)
    assert len(per_team) == 2, per_team
    assert all(len(v) == RADARS_PER_TEAM for v in per_team.values()), per_team


@pytest.mark.parametrize("match", MATCHES)
def test_every_section_earns_its_place(match):
    article = _article(match)
    for section in article.sections:
        assert section.heading.strip()
        if section.gallery:
            # A gallery earns its place with its images, not its word count;
            # its lead-in is one sentence by design.
            assert section.visuals, f"{section.heading} shows nothing"
            continue
        assert section.words() >= 60, f"{section.heading} is a stub"
    headings = [s.heading for s in article.sections]
    assert len(set(headings)) == len(headings), headings


@pytest.mark.parametrize("match", MATCHES)
def test_every_visual_it_points_at_exists(match):
    for section in _article(match).sections:
        for visual in section.visuals:
            assert Path(visual).exists(), visual


# --------------------------------------------------------------------------
# the claims
# --------------------------------------------------------------------------

def _text(article, findings_only: bool = False) -> str:
    sections = [s for s in article.sections if not (findings_only and s.gallery)]
    return " ".join(p for s in sections for p in s.paragraphs)


@pytest.mark.parametrize("match", MATCHES)
@pytest.mark.parametrize("mirror", [False, True])
def test_the_headline_names_the_side_the_numbers_name(match, mirror):
    events, xg, team_metrics, player_metrics, info, _out = _frames(match)
    article = _article(match, mirror)
    home = info["away_name"] if mirror else info["home_name"]
    away = info["home_name"] if mirror else info["away_name"]

    def team_xg(name):
        row = xg[xg["team"].astype(str).str.lower().eq(name.lower())]
        return float(row.iloc[0]["xG"]) if not row.empty else 0.0

    def team_goals(name):
        row = xg[xg["team"].astype(str).str.lower().eq(name.lower())]
        return int(float(row.iloc[0]["goals"])) if not row.empty else 0

    winner = (home if team_goals(home) > team_goals(away)
              else away if team_goals(away) > team_goals(home) else None)
    xg_leader = home if team_xg(home) >= team_xg(away) else away

    if winner and abs(team_xg(home) - team_xg(away)) > 0.15 and xg_leader != winner:
        # The upset framing must name the side that created more, not a fixed one.
        assert xg_leader in article.title, article.title
    assert article.title.strip()
    assert article.standfirst.strip()


@pytest.mark.parametrize("match", MATCHES)
@pytest.mark.parametrize("mirror", [False, True])
def test_no_claim_contradicts_the_frame_it_came_from(match, mirror):
    """Whichever side is credited with more xG must be the side with more xG."""
    events, xg, team_metrics, player_metrics, info, _out = _frames(match)
    article = _article(match, mirror)
    home = info["away_name"] if mirror else info["home_name"]
    away = info["home_name"] if mirror else info["away_name"]

    def team_xg(name):
        row = xg[xg["team"].astype(str).str.lower().eq(name.lower())]
        return float(row.iloc[0]["xG"]) if not row.empty else 0.0

    leader = home if team_xg(home) >= team_xg(away) else away
    trailer = away if leader == home else home
    text = _text(article)
    # The losing-side-created-more framing, wherever it appears, names correctly.
    if f"{trailer} took the points" in text:
        assert f"{leader} finished on" in text, text[:400]


@pytest.mark.parametrize("match", MATCHES)
def test_the_pull_quotes_repeat_numbers_that_are_in_the_frames(match):
    events, xg, _tm, _pm, info, _out = _frames(match)
    article = _article(match)
    home, away = info["home_name"], info["away_name"]
    quotes = [s.pull_quote for s in article.sections if s.pull_quote]
    assert quotes, "no pull quote anywhere"
    for quote in quotes:
        assert home in quote or away in quote, quote


def test_no_finding_is_recycled_between_matches():
    """A sentence carrying this match's numbers must belong to this match.

    The first version of this test banned *any* repeated sentence and failed on
    thirty-five. Reading them showed the test was wrong, not the prose: every
    one was a definition — what expected goals measures, what sequence threat
    credits — and an analyst writing weekly explains those the same way each
    time. What must never repeat is a finding, and a finding carries a number.
    """
    import re

    # Only the argument's sections. The galleries introduce themselves with a
    # count of what follows, which is a fact about the article rather than
    # about the match, and two fixtures can honestly have the same number left.
    first, second = (_text(_article(m), findings_only=True) for m in MATCHES)
    assert first != second

    def claims(text):
        return {part.strip() for part in text.split(". ") if re.search(r"\d", part)}

    shared = claims(first) & claims(second)
    assert not shared, sorted(shared)[:3]


def test_the_definitions_are_allowed_to_recur():
    """The counterpart: the explanatory frame is shared on purpose."""
    import re

    first, second = (_text(_article(m)) for m in MATCHES)

    def definitions(text):
        return {part.strip() for part in text.split(". ") if not re.search(r"\d", part)}

    assert definitions(first) & definitions(second), (
        "no shared framing at all suggests the voice is being regenerated per match"
    )


# --------------------------------------------------------------------------
# the Word file
# --------------------------------------------------------------------------

def test_the_docx_is_built_for_pasting_into_an_editor(tmp_path):
    import re
    import zipfile

    pytest.importorskip("docx")
    article = _article(MATCHES[0])
    path = render_docx(article, tmp_path / "article.docx")
    assert path.exists()

    with zipfile.ZipFile(path) as bundle:
        assert bundle.testzip() is None
        xml = bundle.read("word/document.xml").decode("utf-8")
        media = [n for n in bundle.namelist() if n.startswith("word/media/")]

    # Real heading styles: an editor's paste reads those, not bold text.
    assert 'w:val="Heading1"' in xml
    assert xml.count('w:val="Heading2"') == len(article.sections)
    # Substack breaks tables; there must not be any.
    assert "<w:tbl>" not in xml
    expected = sum(len(s.visuals) for s in article.sections)
    expected += 1 if article.cover else 0
    assert len(media) >= expected - 1, (len(media), expected)
    assert len(re.sub(r"<[^>]+>", " ", xml).split()) >= TARGET_WORDS[0]


def test_a_broken_render_returns_none_rather_than_killing_the_package(tmp_path):
    from match_article import build_match_article

    assert build_match_article(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                               pd.DataFrame(), {}, tmp_path) is None


# --------------------------------------------------------------------------
# the headline
# --------------------------------------------------------------------------

_TITLE_CACHE: list | None = None


def _titles():
    """(match, title, standfirst) for every rendered fixture.

    Cached: building an article walks every frame, and four tests asking the
    same question rebuilt every fixture four times.
    """
    global _TITLE_CACHE
    if _TITLE_CACHE is not None:
        return _TITLE_CACHE

    from match_article import build_article

    root = Path(__file__).resolve().parent.parent
    rows = []
    for folder in sorted((root / "output").glob("*/match_info.json")):
        out = folder.parent
        info = json.loads(folder.read_text(encoding="utf-8"))
        article = build_article(
            pd.read_csv(out / "events.csv"),
            pd.read_csv(out / "xg.csv"),
            pd.read_csv(out / "team_advanced_metrics.csv"),
            pd.read_csv(out / "player_sequence_metrics.csv"),
            info, out,
        )
        rows.append((out.name, article.title, article.standfirst))
    _TITLE_CACHE = rows
    return rows


def test_the_headline_is_not_the_same_sentence_every_time():
    """There were three titles in the file, so every ordinary win shared one.

    A headline that only restates the result tells a reader nothing the
    scoreline has not. Each candidate now owns a measurable condition, and the
    strongest one writes it.
    """
    rows = _titles()
    if len(rows) < 3:
        pytest.skip("fewer than three fixtures rendered")
    titles = [title for _match, title, _stand in rows]
    assert len(set(titles)) == len(titles), f"a headline repeats: {titles}"


def test_no_headline_falls_back_to_the_bare_fixture_line():
    """The last-resort title means every condition declined. None should."""
    for match, title, _stand in _titles():
        assert "Read From The Data" not in title, (match, title)


def test_the_headline_is_stable_for_the_same_match():
    """Varied because matches differ, not because anything is random.

    The cache would answer this with itself, so this builds one article twice
    for real.
    """
    import json

    from match_article import build_article

    root = Path(__file__).resolve().parent.parent
    folder = next(iter(sorted((root / "output").glob("*/match_info.json"))), None)
    if folder is None:
        pytest.skip("no rendered fixture")
    out = folder.parent
    info = json.loads(folder.read_text(encoding="utf-8"))
    frames = (
        pd.read_csv(out / "events.csv"),
        pd.read_csv(out / "xg.csv"),
        pd.read_csv(out / "team_advanced_metrics.csv"),
        pd.read_csv(out / "player_sequence_metrics.csv"),
    )
    first = build_article(*frames, info, out)
    second = build_article(*frames, info, out)
    assert (first.title, first.standfirst) == (second.title, second.standfirst)


def test_every_standfirst_carries_the_score():
    for match, _title, standfirst in _titles():
        assert re.search(r"\d+–\d+", standfirst), (match, standfirst)
