"""A chart that cannot name a player must still draw the player.

``str(name).split()[-1]`` raises IndexError on an empty string, and a name is
empty more often than it looks: a team-level event, a goal the provider never
attributed, a row the parse could not resolve. The exception does not spoil one
label — it takes the whole run down, which is what happened on
`action_value_leaders` mid-render.

The same line had already been fixed once in the xG-flow chart. Fixing one copy
at a time is how it survived, so this asserts the property across every module
that draws a label rather than against the one that crashed.
"""

import ast
import re
from pathlib import Path

import pandas as pd
import pytest

from frame_values import surname

ROOT = Path(__file__).resolve().parent.parent

# The modules that put a player's name on a chart. tactical_visualizations and
# match_report are excluded: nothing in the published package imports them, and
# their calls carry their own `if name else "—"` guards.
RENDERERS = [
    "visual_redesign_full.py",
    "visual_redesign_preview.py",
    "player_radar.py",
    "match_posters.py",
    "tactical_pdf_report.py",
    "match_article.py",
]


def _unguarded(path: Path) -> list[tuple[int, str]]:
    """Lines that index a split with nothing to fall back on."""
    found = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if ".split()[-1]" not in line or line.lstrip().startswith("#"):
            continue
        guarded = re.search(r"if\s+\S+.*\selse\s", line) or re.search(r"or\s+[\"']", line)
        if not guarded:
            found.append((number, line.strip()))
    return found


@pytest.mark.parametrize("module", RENDERERS)
def test_no_renderer_takes_the_last_word_without_a_guard(module):
    path = ROOT / module
    if not path.exists():
        pytest.skip(f"{module} is not present")
    offenders = _unguarded(path)
    assert not offenders, (
        f"{module} indexes a split with no fallback — use frame_values.surname:\n"
        + "\n".join(f"  line {n}: {t}" for n, t in offenders))


def test_surname_survives_every_shape_of_missing_name():
    for value in ("", "   ", None, float("nan"), pd.NA):
        assert surname(value) == ""
        assert surname(value, "—") == "—"


def test_surname_returns_the_last_word():
    assert surname("Martin Ødegaard") == "Ødegaard"
    assert surname("  Rúben  Dias  ") == "Dias"
    assert surname("Vitinha") == "Vitinha"


def test_the_chart_that_crashed_labels_a_nameless_row():
    """action_value_leaders, with the row that took the run down."""
    import visual_redesign_full as v

    names = pd.Series(["Ana Silva", "", None, "Beto"])
    assert [v._surname(n) for n in names] == ["Silva", "", "", "Beto"]


@pytest.mark.parametrize("module", RENDERERS)
def test_every_renderer_that_uses_the_helper_imports_it(module):
    """A NameError at draw time is the same crash wearing a different hat."""
    path = ROOT / module
    if not path.exists():
        pytest.skip(f"{module} is not present")
    source = path.read_text(encoding="utf-8")
    if "_surname(" not in source:
        return
    tree = ast.parse(source)
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "frame_values"
        and any(alias.asname == "_surname" or alias.name == "surname"
                for alias in node.names)
        for node in ast.walk(tree)
    )
    defined = any(isinstance(node, ast.FunctionDef) and node.name == "_surname"
                  for node in ast.walk(tree))
    assert imported or defined, f"{module} calls _surname without importing it"
