"""Analyse one WhoScored match.

    python football_match_analysis.py                       # the default in the pipeline file
    python football_match_analysis.py <match url>           # another fixture
    python football_match_analysis.py <match url> <round>   # ... and its matchweek

The pipeline lives in football_analysis/pipeline/football_match_analysis.py,
which is also where the default URL and round are set. Passing them here sets
MATCH_ANALYSIS_URL and MATCH_ANALYSIS_ROUND for this run, so changing match does
not mean editing a file, and an edit to the wrong file cannot leave an old match
running.
"""

import os
import runpy
import sys


def apply_arguments(argv, environ):
    """Put a URL and optional round from ``argv`` into ``environ``; return what is left."""
    rest = list(argv)
    if rest and rest[0].lower().startswith(("http://", "https://")):
        environ["MATCH_ANALYSIS_URL"] = rest.pop(0)
        if rest and not rest[0].startswith("-"):
            environ["MATCH_ANALYSIS_ROUND"] = rest.pop(0)
    return rest


if __name__ == "__main__":
    sys.argv[1:] = apply_arguments(sys.argv[1:], os.environ)
    runpy.run_module("football_analysis.pipeline.football_match_analysis", run_name="__main__")
