"""Move already-parsed matches into their competition folders.

New matches are filed by competition, season and round as they are written.
Everything collected before that is still sitting flat in ``output/``, and this
moves it — reading each fixture's URL from the match history so the shelf is
worked out the same way the pipeline works it out.

    python scripts/shelve_matches.py            # show what would move
    python scripts/shelve_matches.py --apply    # move it

Nothing is deleted and nothing is overwritten: a folder whose destination
already exists is left where it is and reported. A match whose URL is not in
the history stays put too, because guessing its competition from the team names
would be a guess.
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from match_fixture import shelf  # noqa: E402

OUTPUT = ROOT / "output"
# Not fixtures: the history database, the raw feed archive, and anything the
# pipeline writes beside the matches.
SKIP = {"raw_snapshots", "comparisons"}


def _known_urls() -> dict[str, tuple[str, str]]:
    """{match folder name: (url, played_on)} from the stored history."""
    database = OUTPUT / "match_history.db"
    if not database.exists():
        return {}
    found: dict[str, tuple[str, str]] = {}
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        for row in connection.execute(
                "SELECT home_team, away_team, score, url, played_on FROM matches"):
            score = str(row["score"] or "").replace(" ", "").replace(":", "-")
            name = f"{row['home_team']}_vs_{row['away_team']}"
            if score:
                name = f"{name}_{score}"
            found[name.replace(" ", "_")] = (row["url"] or "", row["played_on"] or "")
    finally:
        connection.close()
    return found


def _plan() -> list[tuple[Path, Path]]:
    urls = _known_urls()
    moves = []
    for folder in sorted(OUTPUT.iterdir()):
        if not folder.is_dir() or folder.name in SKIP:
            continue
        if not (folder / "match_info.json").exists():
            continue                      # already a competition directory
        url, played_on = urls.get(folder.name, ("", ""))
        if not url:
            print(f"  ? {folder.name}: no URL in the history, left where it is")
            continue
        parts = shelf(url, "", played_on)
        if not parts:
            continue
        moves.append((folder, OUTPUT.joinpath(*parts, folder.name)))
    return moves


def main(argv: list[str]) -> int:
    if not OUTPUT.exists():
        print("No output/ directory.")
        return 1

    moves = _plan()
    if not moves:
        print("Nothing to move.")
        return 0

    apply = "--apply" in argv
    for source, destination in moves:
        shown = destination.relative_to(OUTPUT)
        if destination.exists():
            print(f"  ! {source.name}: {shown} already exists, skipped")
            continue
        if not apply:
            print(f"  → {source.name}  ->  {shown}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        print(f"  ✓ {shown}")

    if not apply:
        print(f"\n{len(moves)} match(es) would move. Pass --apply to do it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
