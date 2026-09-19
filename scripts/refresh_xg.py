"""Re-price every shot in the packages on disk, without redrawing anything.

The xG model runs once, in the parse, and the value it produced is written into
``events.csv`` and into the three frames derived from it. Nothing downstream
recomputes it: ``regenerate_packages`` hands the stored frames straight back to
the renderer, so a change to the model reaches a rendered package only if the
frames are rewritten first. A rebuild on its own republishes the same numbers.

This rewrites them. It reads each package's own events, applies the current
model, and regenerates the frames that carry an xG-derived figure:

    events.csv                      the per-shot value
    xg.csv                          team totals, xG on target, per shot
    team_advanced_metrics.csv       regain, transition and game-state xG
    player_sequence_metrics.csv     xGChain and xGBuildup

    python scripts/refresh_xg.py                 # every package
    python scripts/refresh_xg.py Napoli Arsenal  # ones whose folder matches
    python scripts/refresh_xg.py --dry-run       # report the change, write nothing

Seconds a package rather than minutes, because no figure is drawn. Redraw
afterwards -- the visuals, the article and the PDFs all read these frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "output"


def packages(patterns: list[str]) -> list[Path]:
    found = []
    for info in sorted(OUTPUT.rglob("match_info.json")):
        folder = info.parent
        if folder.name == "light" or any(p.startswith(".") for p in folder.parts):
            continue
        if not (folder / "events.csv").exists():
            continue
        if patterns and not any(p.lower() in folder.name.lower() for p in patterns):
            continue
        found.append(folder)
    return found


def refresh(folder: Path, write: bool) -> tuple[float, float]:
    """Re-price one package. Returns the team xG total before and after."""
    import football_match_analysis as F

    info = json.loads((folder / "match_info.json").read_text(encoding="utf-8-sig"))
    events = pd.read_csv(folder / "events.csv", low_memory=False)

    shots = events["is_shot"].fillna(False).astype(bool) if "is_shot" in events else None
    before = float(pd.to_numeric(events.loc[shots, "xG"], errors="coerce").fillna(0).sum()) \
        if shots is not None else 0.0

    # The stored column is what the model said last time; drop it so the model
    # is asked again rather than handed its own previous answer.
    events = events.drop(columns=["xG", "xg_source"], errors="ignore")
    events = F.apply_best_open_source_xg(events, info)
    # The same Opta values a fresh render would take, from the stored shot map
    # when there is one, so a re-price does not undo them.
    from reference_xg import apply_reference_xg

    events, _ = apply_reference_xg(events, info, package=str(folder.relative_to(OUTPUT)))
    after = float(pd.to_numeric(events.loc[shots, "xG"], errors="coerce").fillna(0).sum()) \
        if shots is not None else 0.0

    if not write:
        return before, after

    events.to_csv(folder / "events.csv", index=False, encoding="utf-8-sig")
    xg_data = F.xg_stats(events, info)
    (pd.DataFrame(xg_data).T.reset_index().rename(columns={"index": "team"})
       .to_csv(folder / "xg.csv", index=False, encoding="utf-8-sig"))
    team_frame, sequence_frame = F.advanced_metrics_frames(events, info)
    team_frame.to_csv(folder / "team_advanced_metrics.csv", index=False,
                      encoding="utf-8-sig")
    sequence_frame.to_csv(folder / "player_sequence_metrics.csv", index=False,
                          encoding="utf-8-sig")
    return before, after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patterns", nargs="*", help="folder-name fragments")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change, write nothing")
    parser.add_argument("--list", help="read folder names from a file, one per line")
    args = parser.parse_args()

    patterns = list(args.patterns)
    if args.list:
        patterns = [line.strip() for line
                    in Path(args.list).read_text(encoding="utf-8").splitlines()
                    if line.strip()]

    targets = packages(patterns)
    if not targets:
        print("No packages matched.")
        return 1
    print(f"Re-pricing {len(targets)} package(s)"
          f"{' (dry run)' if args.dry_run else ''}\n")

    started = time.time()
    moved = total_before = total_after = 0.0
    failed = []
    for index, folder in enumerate(targets, 1):
        try:
            before, after = refresh(folder, write=not args.dry_run)
        except Exception as error:                       # noqa: BLE001
            failed.append((folder.name, f"{type(error).__name__}: {error}"))
            print(f"[{index}/{len(targets)}] {folder.name}: FAILED {error}")
            continue
        total_before += before
        total_after += after
        moved += abs(after - before)
        print(f"[{index}/{len(targets)}] {folder.name:<44} "
              f"{before:6.2f} -> {after:6.2f}")

    print(f"\n{len(targets) - len(failed)}/{len(targets)} in "
          f"{(time.time() - started) / 60:.1f} min")
    print(f"total xG {total_before:.1f} -> {total_after:.1f}")
    for name, reason in failed:
        print(f"  {name}: {reason}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
