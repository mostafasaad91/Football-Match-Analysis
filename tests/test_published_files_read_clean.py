"""The shipped .docx and .pdf, read as files rather than as generator output.

test_every_report_reads_clean.py builds each article in memory and checks the
paragraphs it returns. That misses everything ``render_docx`` adds on the way
to the page — the board readings the report writes, which the article carries
under its visuals — and a defect there reaches the reader exactly as surely.

It reached them: a 1-0 shipped with "The two goalkeepers faced a match in which
1 goals came from 1.36 xG". ``_count`` had been in tactical_pdf_report since
the player pages were fixed for the same thing, and this branch never called
it. No test failed, because no test opened the file.

Everything here reads what is on disk. A fixture that has not been rendered is
skipped rather than failed, so the suite still runs on a clean checkout.
"""

import re
import zipfile
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"

ARTICLES = [p for p in sorted(OUTPUT.rglob("match_article.docx"))
            if p.parent.name != "light"]
REPORTS = [p for p in sorted(OUTPUT.rglob("full_visual_redesign_real_data.pdf"))
           if p.parent.name != "light"]
ARTICLE_IDS = [p.parent.name for p in ARTICLES]
REPORT_IDS = [p.parent.name for p in REPORTS]

if not ARTICLES:
    pytest.skip("no rendered articles on disk", allow_module_level=True)


def _docx_text(path: Path) -> str:
    """Every run of text in the document body, paragraph by paragraph."""
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    xml = xml.replace("</w:p>", "\n")
    return re.sub(r"<[^>]+>", "", xml)


def _pdf_text(path: Path) -> str:
    fitz = pytest.importorskip("fitz")
    document = fitz.open(path)
    try:
        return "\n".join(page.get_text() for page in document)
    finally:
        document.close()


_CACHE: dict[Path, str] = {}


def _text(path: Path) -> str:
    if path not in _CACHE:
        _CACHE[path] = (_docx_text(path) if path.suffix == ".docx"
                        else _pdf_text(path))
    return _CACHE[path]


# A count of one followed by a plural. The "1" has to be the whole number, not
# the tail of something longer: the lookbehind keeps a scoreline out — "the
# final 2 — 1 makes it look like" is not a count — and a decimal point with it,
# because "carried 2.1 times the value" is correct English and was being
# reported as "1 times". The suffix guard keeps "1 less" and "1 series" out.
ONE_PLURAL = re.compile(r"(?<![\d—–-]\s)(?<![\d.—–-])\b1 (\w+?)(s|es)\b")
MACHINE = re.compile(r"\b(nan|NaN|inf)\b|\{\w+\}|\s\|\s")

# The report's running footer carries a pipe, so one is page furniture there
# and a leaked machine string anywhere else. Dropping the rule for PDFs would
# give up the check that put it here — a pipe-delimited timeline printed as a
# sentence — so the chrome is removed and the rest is still scanned.
#
# The footer takes two forms and the first version of this only knew one:
# section pages read "MATCH STORY  |  PAGE 05" and the plates between them read
# "PAGE 23  |  REAL MATCH EVENTS". Both are upper case, both are one line, and
# nothing in the prose is either.
CHROME = re.compile(
    r"^\s*(?:PAGE \d+\s*\|\s*[A-Z ]+|[A-Z][A-Z ]+\|\s*PAGE \d+)\s*$",
    re.MULTILINE)


def _prose_only(text: str) -> str:
    return CHROME.sub("", text)


def _plural_offenders(text: str) -> list[str]:
    """Every "1 <plural noun>", and nothing that only looks like one.

    The pattern reads the word after the digit non-greedily, so an adjective
    ending in s trips it: "conceding 1 dangerous counter" is correct English and
    was reported as "1 dangerous". No English plural ends in "us", so skipping
    that ending cannot hide a real one.
    """
    return [found.group(0) for found in ONE_PLURAL.finditer(text)
            if not found.group(0).endswith(("ss", "ess", "us"))]


@pytest.mark.parametrize("path", ARTICLES, ids=ARTICLE_IDS)
def test_no_shipped_article_prints_a_count_of_one_as_a_plural(path):
    assert not _plural_offenders(_text(path))


@pytest.mark.parametrize("path", REPORTS, ids=REPORT_IDS)
def test_no_shipped_report_prints_a_count_of_one_as_a_plural(path):
    assert not _plural_offenders(_text(path))


@pytest.mark.parametrize("path", ARTICLES, ids=ARTICLE_IDS)
def test_no_shipped_article_carries_a_machine_string(path):
    found = MACHINE.search(_prose_only(_text(path)))
    assert not found, found.group(0)


@pytest.mark.parametrize("path", REPORTS, ids=REPORT_IDS)
def test_no_shipped_report_carries_a_machine_string(path):
    text = _prose_only(_text(path))
    found = MACHINE.search(text)
    assert not found, (found.group(0),
                       text[max(0, found.start() - 90):found.start() + 60])


# Wordings that were published and then withdrawn. Each one is here because it
# shipped, not because it might: a broken clause, a stutter, a headline naming
# nobody, and a claim about the result that the result did not support.
WITHDRAWN = (
    "there was a great deal:",
    "One side shot more —",
    "The funnel is where",
    "The Last Twenty Metres Decided It",
    "both sides finished below what the chances",
    "Read that as a warning about the sample, not a verdict on the players: "
    "conversion is the noisiest thing in the match and the least likely part "
    "of it to repeat. 3.58",
)


@pytest.mark.parametrize("path", ARTICLES, ids=ARTICLE_IDS)
def test_no_withdrawn_wording_is_still_on_disk(path):
    text = _text(path)
    assert not [phrase for phrase in WITHDRAWN if phrase in text]


@pytest.mark.parametrize("path", ARTICLES, ids=ARTICLE_IDS)
def test_no_shipped_article_says_a_beaten_side_won_the_match(path):
    """"Monza Won The Match In The Broken Moments", after a 4-1 defeat."""
    import json

    text = _text(path)
    claim = re.search(r"(.{3,40}?) Won The Match In The Broken Moments", text)
    if not claim:
        return
    info = json.loads((path.parent / "match_info.json").read_text(encoding="utf-8"))
    score = str(info.get("score") or "")
    numbers = [int(n) for n in re.findall(r"\d+", score)][:2]
    assert len(numbers) == 2 and numbers[0] != numbers[1], (
        f"{path.parent.name}: claims a win in a {score}")
    winner = (str(info["home_name"]) if numbers[0] > numbers[1]
              else str(info["away_name"]))
    assert claim.group(1).strip().endswith(winner), (
        f"{path.parent.name}: names {claim.group(1)!r}, {winner} won {score}")


def test_two_shipped_articles_do_not_open_on_the_same_headline():
    """Six of fifteen carried one headline; the reader sees the file, not the
    weighting that produced it."""
    headlines = {}
    for path in ARTICLES:
        lines = [line.strip() for line in _text(path).splitlines() if line.strip()]
        # The strap is first (competition and score), the headline follows.
        headlines[path.parent.name] = lines[1] if len(lines) > 1 else ""
    repeated = {h: n for h, n in Counter(headlines.values()).items()
                if n > 1 and h}
    assert not repeated, repeated
