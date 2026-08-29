"""Score the xG model against every match collected, and calibrate if earned.

    python scripts/xg_report.py            # measure only
    python scripts/xg_report.py --fit      # measure, and write a correction
                                           # if the data supports one

Reads the raw snapshots the collector keeps, recomputes each shot with the
shipped model, and compares the predictions with what actually happened.

Nothing is written unless the correction clears every gate in xg_calibration:
enough shots, goals and matches, plus an improvement on complete matches it was
not fitted to. The report always scores the raw submodels first, so an existing
correction can never be fitted to its own already-corrected output.
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
                # Pipe, not comma. _qnames splits this field on "|", so a
                # comma-joined string came back as one nonsense qualifier and
                # every name in it was lost — BigChance, Cross and Head with
                # them. The model then priced every shot as an ordinary foot
                # shot from open play, 22% below what the pipeline gives the
                # same thirty shots, and the calibration fitted here was
                # measuring a model the package does not ship.
                "qualifier_names": "|".join(names),
                # The geometry reads these as columns rather than through the
                # qualifier set, so supplying the set alone is not enough.
                "big_chance": "BigChance" in names,
                "is_cross": "Cross" in names,
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

    # Always measure the raw submodels. Otherwise an existing calibration is
    # applied here and then fitted a second time on its own output.
    fa._XG_CALIBRATION = None
    shots["distance"] = shots.apply(
        lambda row: fa._shot_geometry_features(row)["distance"], axis=1)
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
    for low, high in xc.BANDS:
        inside = shots[(shots["xG"] >= low) & (shots["xG"] < high)]
        if inside.empty:
            continue
        print(f"  {low:.2f}-{high:.2f}  n={len(inside):4d}  "
              f"predicted {inside['xG'].mean():.3f}  actual {inside['is_goal'].mean():.3f}")
    print(f"  {xc.banded_error(shots['xG'], shots['is_goal'].astype(float)):.1f} "
          f"goals mispriced once the bands are counted apart")

    # The total hides this and the bands only hint at it: the error runs with
    # distance, not with the model's own probability, which is why no Platt
    # correction can reach it.
    print("\ncalibration by distance")
    for low, high in xc.DISTANCE_BANDS:
        inside = shots[(shots["distance"] >= low) & (shots["distance"] < high)]
        if inside.empty:
            continue
        predicted, goals = inside["xG"].sum(), inside["is_goal"].sum()
        spread = float((inside["xG"] * (1 - inside["xG"])).sum()) ** 0.5
        name = f"{low:g}-{high:g} m" if high < 999 else f"{low:g}+ m"
        print(f"  {name:14s} n={len(inside):4d}  predicted {predicted:6.1f}  "
              f"goals {int(goals):4d}  {(goals - predicted) / max(spread, 1e-9):+.1f} "
              f"standard deviations")

    if "--fit" not in argv:
        print(f"\nMeasure only. Pass --fit to write a correction when one is earned.")
        return 0

    found = xc.fit(shots["xG"], shots["is_goal"].astype(float),
                   shots["distance"], shots["match"])
    if found is None:
        print(f"\nNo correction written. Needs {xc.MIN_SHOTS} shots and "
              f"{xc.MIN_GOALS} goals (have {report['shots']} and {report['goals']}), "
              f"and it has to beat the current model on held-out folds.")
        return 0

    path = xc.save(found)
    corrected = pd.Series(
        [found.apply(p, d) for p, d in zip(shots["xG"], shots["distance"])],
        index=shots.index)
    print(f"\nWrote {path.name}: excess above {xc.HIGH_KNEE:.2f} keeps "
          f"{found.high_gain:.0%}; shooting-range term {found.range_logit:+.3f}")
    print(f"held-out log-loss {found.log_loss_before:.4f} → {found.log_loss_after:.4f}")
    print(f"distance-banded mispricing "
          f"{xc.distance_banded_error(shots['xG'], shots['is_goal'], shots['distance']):.1f} "
          f"→ {xc.distance_banded_error(corrected, shots['is_goal'], shots['distance']):.1f} goals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
