"""Rebuild a package's article, cover and article PDF from its saved frames.

The boards are left as they are; only the writing is regenerated, so a fix to
the prose reaches every built match in minutes rather than the hours a full
rebuild takes. The team and player frames are recomputed from the saved
events, exactly as generate_match_package does, so the article reads the same
numbers the boards were drawn from.

    python scripts/rebuild_article.py <package dir> [<package dir> ...]

Set MATCH_ANALYSIS_THEME=light and pass the package's light/ folder to rebuild
the light copy.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


def rebuild(folder: Path) -> None:
    from football_analysis.metrics.expected_threat import rescore
    from football_analysis.prose.match_article import build_match_article
    from football_analysis.visuals import visual_redesign_full as full

    info = json.loads((folder / "match_info.json").read_text(encoding="utf-8"))
    events = rescore(pd.read_csv(folder / "events.csv", low_memory=False))
    players = pd.read_csv(folder / "players.csv")
    xg = full.shot_counts_from_events(events, pd.read_csv(folder / "xg.csv"), info)
    team_metrics, player_metrics = full.advanced_metrics_frames(events, info)
    full.configure_match(info, folder)
    full.base.theme()
    xg.to_csv(folder / "xg.csv", index=False, encoding="utf-8-sig")
    team_metrics.to_csv(folder / "team_advanced_metrics.csv", index=False, encoding="utf-8-sig")
    player_metrics.to_csv(folder / "player_sequence_metrics.csv", index=False, encoding="utf-8-sig")
    # build_match_article never raises (a package must survive a failed
    # article), so judge it by whether the files were actually rewritten.
    started = time.time()
    build_match_article(events, xg, team_metrics, player_metrics, info, folder, players)
    for name in ("match_article.md", "match_article.docx"):
        path = folder / name
        if not path.exists() or path.stat().st_mtime < started - 1:
            raise RuntimeError(f"{name} was not rewritten (open in Word, or the article was refused)")


def main(argv: list[str]) -> int:
    failed = 0
    for arg in argv:
        folder = Path(arg.strip())
        try:
            rebuild(folder)
            print(f"ok      {folder}", flush=True)
        except Exception:
            failed += 1
            print(f"FAILED  {folder}", flush=True)
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
