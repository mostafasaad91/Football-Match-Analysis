"""Analyse one WhoScored match: python football_match_analysis.py

The pipeline lives in football_analysis/pipeline/football_match_analysis.py,
which is also where the match URL and round defaults are set. This file only
runs it, as ``__main__``, exactly as when it sat here itself.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module("football_analysis.pipeline.football_match_analysis", run_name="__main__")
