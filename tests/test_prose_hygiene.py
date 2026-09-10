"""One gate, and a test that stands on it.

Five separate faults of the same kind shipped to readers this week -- "1
shots", "1 touches", "0.00 expected goals, nan", a dataclass repr on the
cover, a scoreline read as a count -- and each was found by opening a
published document and reading it. Each was then fixed at the writer that
produced it, which is thirty writers' worth of places to reproduce it.

prose_hygiene.clean is applied where text becomes a document, so no writer can
ship the repairable ones again. This tests the gate itself, and then tests
every shipped file through it: if a paragraph reaches a reader carrying a
machine string, this fails rather than the reader finding out.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from prose_hygiene import clean, offences, one_reads_singular

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("written,published", [
    ("he had 1 touches", "he had 1 touch"),
    ("played 1 minutes", "played 1 minute"),
    ("1 entries into the box", "1 entry into the box"),
    ("1 recoveries", "1 recovery"),
    ("1 crosses", "1 cross"),
    ("1 losses", "1 loss"),
    ("1 passes", "1 pass"),
    ("1 shots on target", "1 shot on target"),
])
def test_a_count_of_one_reads_as_one(written, published):
    assert clean(written) == published


@pytest.mark.parametrize("text", [
    "21.1 metres",              # a decimal is not a count of one
    "11 shots",                 # nor is a number ending in one
    "1 dangerous counter",      # an adjective ending in s is not a plural
    "1 press",                  # nor is a noun that ends in ss
    "1 analysis",               # nor one that ends in is
    "0:1 Leeds",                # nor a club name after a scoreline
    "2 shots",
])
def test_what_only_looks_like_a_plural_is_left_alone(text):
    assert clean(text) == text


def test_tidying_does_not_rewrite():
    """The gate repairs; it does not edit."""
    sentence = ("Arsenal entered the box 22 times and 64.7% of those entries "
                "produced a shot.")
    assert clean(sentence) == sentence


@pytest.mark.parametrize("text,expected", [
    ("two  spaces", "two spaces"),
    ("a space before the stop .", "a space before the stop."),
    ("a value ( ) that never arrived", "a value that never arrived"),
])
def test_the_gate_tidies_what_has_one_correct_form(text, expected):
    assert clean(text) == expected


@pytest.mark.parametrize("text", [
    "0.00 expected goals, nan.",
    "Verdict(home=SideVerdict(team='Arsenal'))",
    "the value is None",
    "a {placeholder} left unfilled",
    "home | away | contested",
])
def test_a_machine_string_is_reported_rather_than_patched(text):
    """These carry a number that was never computed.

    Substituting anything for one would invent the figure, so the gate reports
    it and lets a test fail instead.
    """
    assert offences(text), text


def test_a_clean_sentence_has_no_offences():
    assert not offences(
        "Arsenal reached the penalty area 22 times to 7, and 64.7% of those "
        "entries produced a shot.")


@pytest.mark.parametrize("written,published", [
    # The two that shipped: a label naming a list, the list empty, and a
    # finished sentence beside it that does carry something.
    ("Man City:. Coventry:. When each of them was doing it is what figure 31 plots.",
     "When each of them was doing it is what figure 31 plots."),
    ("Highest positive expected threat on the pitch:. The shape is in figure 37.",
     "The shape is in figure 37."),
    # A run of them, because the writers build one label per side and sub()
    # resumes after the text it consumed.
    ("A:. B:. C:. Something real at the end.", "Something real at the end."),
    ("Most-used targets:. Arsenal went long twice.", "Arsenal went long twice."),
    # Nothing but the label leaves nothing.
    ("Arsenal:.", ""),
])
def test_a_label_whose_list_came_back_empty_is_dropped(written, published):
    assert clean(written) == published


@pytest.mark.parametrize("text", [
    "Method: the corridor is read where the entry landed.",
    "Top scorer: Saka.",
    "He pressed high; the block sat deep.",
    "Denominator · Arsenal 43 final-third entries, Coventry 32.",
])
def test_a_label_that_did_get_its_list_is_left_alone(text):
    assert clean(text) == text


def test_a_colon_the_repair_cannot_reach_is_still_reported():
    """offences() reads the repaired text, so it reports only what survives.

    A label past the length bound is not one the rule will drop, and the value
    it named is still missing: that has to fail a test rather than ship.
    """
    too_long = ("A label so long that it goes past the eighty character bound "
                "this rule sets for what may be a label at all:.")
    assert offences(too_long) == ["a value that never arrived"]
    assert not offences("Man City:. Coventry:. Figure 31 plots it.")


# --------------------------------------------------------------------------
# every file a reader is handed
# --------------------------------------------------------------------------

def _published(path: Path) -> bool:
    parts = path.relative_to(OUTPUT).parts
    return not any(part.startswith(".") for part in parts) and len(parts) >= 4


ARTICLES = [p for p in sorted(OUTPUT.rglob("match_article.docx")) if _published(p)]
IDS = [f"{p.parent.parent.name}/{p.parent.name}" if p.parent.name == "light"
       else p.parent.name for p in ARTICLES]

if not ARTICLES:
    pytest.skip("no rendered articles on disk", allow_module_level=True)


@pytest.mark.parametrize("path", ARTICLES, ids=IDS)
def test_no_shipped_paragraph_carries_a_fault_the_gate_exists_to_stop(path):
    import docx

    document = docx.Document(str(path))
    found = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        # offences() only. clean() is compared against a single run, and
        # paragraph.text joins several: a figure caption is "FIGURE 01" and its
        # title set as two runs with a deliberate gap between them, which reads
        # as a double space once joined and is not a fault in the prose.
        found.extend(offences(text))
    assert not found, sorted(set(found))[:4]
