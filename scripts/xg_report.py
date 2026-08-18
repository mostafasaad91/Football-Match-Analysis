"""Score the xG model against every match collected, and calibrate if earned.

    python scripts/xg_report.py            # measure only
    python scripts/xg_report.py --fit      # measure, and write a correction
                                           # if the data supports one

Reads the raw snapshots the collector keeps, recomputes each shot with the
shipped model, and compares the predictions with what actually happened.

Nothing is written unless the correction clears every gate in xg_calibration:
enough shots, enough goals, and an improvement on folds it was not fitted to.
Today it does not, and that is the honest result — the model's total sits well
inside the sampling noise of the goals scored.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import xg_calibration as xc  # noqa: E402


def shots_from_snapshots() -> pd.DataFrame:
    """Every non-penalty shot the collector has stored, with its outcome."""
    rows = []
    for path in sorted((ROOT / "output" / "raw_snapshots").glob("*.json.gz")):
        try:
            data = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
        except Exception:
            continue
        for event in data.get("events", []):
            if not event.get("isShot"):
                continue
            names = [str((q.get("type") or {}).get("displayName", ""))
                     for q in event.get("qualifiers", [])]
            rows.append({
                "match": path.stem,
                "x": event.get("x"), "y": event.get("y"),
                "is_goal": bool(event.get("isGoal")),
                "qualifier_names": ",".join(names),
                "is_header": "Head" in names,
                "is_penalty": "Penalty" in names,
                "is_own_goal": "OwnGoal" in names,
                "is_shot": True,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Penalties are a fixed value and would flatter any measure of the model
    # that has to judge open play. Own goals are not shots at all: the pipeline
    # already keeps them out of xG, and leaving them in here priced one from
    # the scorer's own six-yard box as a 0.95 chance that was duly converted,
    # which made a broken model look accurate.
    return frame[~frame["is_penalty"] & ~frame["is_own_goal"]]


def main(argv: list[str]) -> int:
    import football_match_analysis as fa

    shots = shots_from_snapshots()
    if shots.empty:
        print("No stored snapshots under output/raw_snapshots.")
        return 1

    shots["xG"] = shots.apply(fa._opta_like_local_xg_from_row, axis=1)
    report = xc.evaluate(shots["xG"], shots["is_goal"].astype(float))

    print(f"matches   {shots['match'].nunique()}")
    print(f"shots     {report['shots']}   goals {report['goals']}")
    print(f"predicted {report['predicted']:.1f} xG   ratio {report['ratio']:.2f}")
    print(f"          {report['sigma']:+.1f} standard deviations from the goals scored")
    print(f"log-loss  {report['log_loss']:.4f}   (predicting the base rate: "
          f"{report['baseline_log_loss']:.4f})")
    print(f"Brier     {report['brier']:.4f}")

    print("\ncalibration by predicted band")
    bands = [(0, .05), (.05, .10), (.10, .20), (.20, .40), (.40, 1.01)]
    for low, high in bands:
        inside = shots[(shots["xG"] >= low) & (shots["xG"] < high)]
        if inside.empty:
            continue
        print(f"  {low:.2f}-{high:.2f}  n={len(inside):4d}  "
              f"predicted {inside['xG'].mean():.3f}  actual {inside['is_goal'].mean():.3f}")

    if "--fit" not in argv:
        print(f"\nMeasure only. Pass --fit to write a correction when one is earned.")
        return 0

    found = xc.fit(shots["xG"], shots["is_goal"].astype(float))
    if found is None:
        print(f"\nNo correction written. Needs {xc.MIN_SHOTS} shots and "
              f"{xc.MIN_GOALS} goals (have {report['shots']} and {report['goals']}), "
              f"and it has to beat the current model on held-out folds.")
        return 0

    path = xc.save(found)
    print(f"\nWrote {path.name}: slope {found.slope:.3f}, intercept {found.intercept:+.3f}")
    print(f"held-out log-loss {found.log_loss_before:.4f} → {found.log_loss_after:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
