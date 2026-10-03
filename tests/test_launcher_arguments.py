"""The root launcher takes the match from the command line."""

import importlib.util
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parent.parent / "football_match_analysis.py"


def _apply(argv):
    spec = importlib.util.spec_from_file_location("launcher_under_test", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # __name__ is not "__main__", so nothing runs
    environ = {}
    return module.apply_arguments(argv, environ), environ


def test_a_url_and_a_round_become_the_environment():
    rest, environ = _apply(["https://www.whoscored.com/matches/1/live/x-y", "Matchweek 3"])
    assert environ == {
        "MATCH_ANALYSIS_URL": "https://www.whoscored.com/matches/1/live/x-y",
        "MATCH_ANALYSIS_ROUND": "Matchweek 3",
    }
    assert rest == []


def test_a_url_alone_leaves_the_round_to_the_default():
    rest, environ = _apply(["https://www.whoscored.com/matches/1/live/x-y"])
    assert "MATCH_ANALYSIS_ROUND" not in environ and rest == []


def test_no_arguments_change_nothing():
    rest, environ = _apply([])
    assert rest == [] and environ == {}


def test_other_arguments_pass_through_untouched():
    rest, environ = _apply(["--something"])
    assert rest == ["--something"] and environ == {}
