"""Keep one test's fixture out of the next one's numbers.

``visual_redesign_full.configure_match`` writes eleven module-level constants —
the two team ids, their names, their colours, the output directory, the match
key and the score — and ``visual_redesign_preview`` keeps a parallel set of
twelve. Nothing puts either back, so a test that renders a package leaves both
renderers pointing at that match.

The damage is not obvious, because the module that reads them reads *half* its
state from somewhere else. ``visual_redesign_preview.load_data`` opens the
sample match from a DATA_DIR fixed at import time, and then builds its ``info``
from whatever HOME_ID and AWAY_ID happen to be now. Point those at Arsenal and
it computes France vs England's events under Arsenal's ids: no event matches
either side, every frame comes back empty, and the failure surfaces three
functions later as ``KeyError: 'transition_shots'``.

A full run reported three failures that each passed on their own — Arsenal's
#EF0107 turning up where #2F5BFF was expected, and two half-by-half tests
raising that KeyError. Same code, same data, a different answer depending on
collection order, which makes a green suite mean less than it should.

The baseline is taken once, at session start, before any test has run. Saving
per-test would capture whatever the previous test left behind and restore the
pollution rather than the truth.
"""

import importlib

import pytest

_STATEFUL_MODULES = ("visual_redesign_full", "visual_redesign_preview")


def _module_constants(module) -> dict:
    return {
        name: getattr(module, name)
        for name in dir(module)
        if name.isupper() and not name.startswith("_")
    }


@pytest.fixture(scope="session")
def _pristine_renderer_state():
    """Every module constant as it was before the first test ran.

    Captured wholesale rather than from a list of the ones configure_match is
    known to touch. Two attempts at that list missed something — the first left
    the preview module's parallel copy behind, the second still missed
    COMPARE_DIR — and each miss looked exactly like no fix at all.
    """
    baseline = []
    for name in _STATEFUL_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        baseline.append((module, _module_constants(module)))
    return baseline


@pytest.fixture(autouse=True)
def _restore_renderer_globals(_pristine_renderer_state):
    """Hand every test the renderers in the state the session started in."""
    def restore():
        for module, values in _pristine_renderer_state:
            for name, value in values.items():
                setattr(module, name, value)

    restore()
    try:
        yield
    finally:
        restore()
