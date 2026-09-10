"""The last thing every sentence passes through before it is published.

Prose reaches the report and the article from a dozen writers: the board
narrator, the section readings, the player-card caption, the headline, the
stored chart contracts. Each one had its own way of going wrong in the same
few ways -- a count of one printed as a plural, a NaN reaching the page as the
word "nan", a machine repr where a paragraph belonged -- and each was found and
fixed separately, one shipped document at a time.

Fixing them at the writers means every new writer is a new chance to reproduce
them. This is the other end: one function, applied where text becomes a
document, so the class of fault cannot ship regardless of who wrote the
sentence. A writer that gets it right is unaffected; a writer that does not is
corrected on the way out.

``clean`` repairs what can be repaired. ``offences`` reports what cannot, so a
test can fail on a machine string rather than a reader finding it.
"""
from __future__ import annotations

import re

# "1 shots", "1 tackles", "1 entries". The noun is read non-greedily so an
# adjective that happens to end in s -- "1 dangerous counter" -- is not caught,
# and a decimal is excluded by the lookbehind: "21.1 metres" is not a count.
_ONE_THEN_PLURAL = re.compile(r"(?<![\d.,])\b1\s+([a-z]+(?:ies|sses|ches|shes|xes|s))\b")

# Words ending in s that are not plurals of anything.
_NOT_PLURAL = frozenset({
    "across", "less", "press", "loss", "success", "this", "its", "was", "has",
    "is", "as", "gas", "plus", "minus", "versus", "status", "focus", "bonus",
    "always", "perhaps", "towards", "afterwards", "unless", "whereas",
})

# A repr, a placeholder, or a missing value that reached the text. These cannot
# be repaired into a sentence -- the number they should have carried is gone --
# so they are reported rather than patched over.
_MACHINE = re.compile(
    r"\bnan\b|\bNaN\b|\bNaT\b|\bNone\b|<NA>|\{[a-z_]+\}"
    r"|\b\w+\((?:[a-z_]+=|')"          # Verdict(home=..., SideVerdict('...
    r"|\[\s*'"                          # a list of strings printed raw
    r"|\s\|\s",                         # a pipe-delimited machine line
    # Deliberately case-sensitive on None: "scored none of them" is a
    # sentence, and None is a value that never arrived.
)

# Two spaces, a space before punctuation, an empty parenthesis left by a value
# that never arrived.
_SPACE_BEFORE_STOP = re.compile(r"\s+([.,;:!?%])")
_DOUBLE_SPACE = re.compile(r"[ \t]{2,}")
_EMPTY_BRACKET = re.compile(r"\s*\(\s*\)")
_ORPHAN_COMMA = re.compile(r",\s*([.;])")

# A label whose list came back empty: "Man City:. Coventry:. When each of them
# was doing it is what figure 31 plots."
#
# report_narrative drops a paragraph that is nothing but its own label, but its
# rule is anchored to the whole paragraph, and the shipped fault is a label
# stranded mid-paragraph among sentences that do carry something. Dropping the
# empty clause and keeping the rest is the repair: the label named a list, the
# list was empty, and what follows it is a finished sentence.
#
# The label may not contain sentence punctuation or a second colon, so the
# match cannot run past the sentence it belongs to. The prefix group holds the
# start of the string or the end of the sentence before, because a variable
# lookbehind is not available here.
_EMPTY_LABEL = re.compile(r"(\A|[.!?]\s*)[^.!?:;]{1,80}:\s*[.;,]\s*")


def _singular(noun: str) -> str:
    # No English plural ends in "us", so an adjective caught by the pattern --
    # "1 dangerous counter" -- keeps its ending. Skipping it cannot hide a real
    # plural and it covers the whole -ous family along with focus and status.
    if noun in _NOT_PLURAL or noun.endswith(('us', 'ss', 'is')):
        return noun
    if noun.endswith("ies") and len(noun) > 4:
        return noun[:-3] + "y"
    for ending in ("sses", "ches", "shes", "xes"):
        if noun.endswith(ending):
            return noun[:-2]
    if noun.endswith("s") and not noun.endswith("ss"):
        return noun[:-1]
    return noun


def one_reads_singular(text: str) -> str:
    """"1 shots" -> "1 shot", leaving every other count alone."""
    return _ONE_THEN_PLURAL.sub(lambda m: "1 " + _singular(m.group(1)), text)


def clean(text) -> str:
    """The published form of one paragraph.

    Repairs the faults that have a correct form -- a count of one, a doubled
    space, a space before a full stop. Leaves everything else exactly as the
    writer set it: this tidies, it does not rewrite.
    """
    if text is None:
        return ""
    out = str(text)
    if not out.strip():
        return ""
    out = one_reads_singular(out)
    # Before the bracket and spacing rules: dropping the clause can leave a
    # doubled space behind it, which _DOUBLE_SPACE then closes up.
    #
    # To a fixed point, because the faults arrive in runs -- "Man City:.
    # Coventry:." is one label per side -- and sub() resumes after the text it
    # consumed. The second label's only prefix was the full stop the first
    # match took with it, so a single pass leaves every label but the first.
    while True:
        shorter = _EMPTY_LABEL.sub(lambda m: m.group(1), out)
        if shorter == out:
            break
        out = shorter
    out = _EMPTY_BRACKET.sub("", out)
    out = _SPACE_BEFORE_STOP.sub(r"\1", out)
    out = _ORPHAN_COMMA.sub(r"\1", out)
    out = _DOUBLE_SPACE.sub(" ", out)
    return out.strip()


def offences(text) -> list[str]:
    """What is wrong with a paragraph that cleaning cannot fix.

    A NaN or a repr means a number the writer meant to print was never
    computed. Substituting anything for it would be inventing the figure, so it
    is reported and the sentence is left to fail a test rather than quietly
    reaching a reader.
    """
    text = str(text or "")
    found = [m.group(0).strip() for m in _MACHINE.finditer(text)]
    # Against the repaired text, not the raw: an empty label and an empty
    # bracket both have a correct published form, and clean() writes it. What
    # is left here is a colon or a bracket that survived the repair -- a label
    # too long to be one, or a pair the rules do not recognise -- and that is
    # still a value the writer meant to print and never had.
    repaired = clean(text)
    if re.search(r':\s*[.,;]', repaired) or re.search(r'\(\s*\)', repaired):
        found.append('a value that never arrived')
    for hit in _ONE_THEN_PLURAL.finditer(str(text or "")):
        if _singular(hit.group(1)) != hit.group(1):
            found.append(hit.group(0))
    # clean() repairs these, so offences() only reports one that survived it.
    found = [f for f in found if not f.startswith("1 ") or f in clean(text)]
    return sorted(set(found))
