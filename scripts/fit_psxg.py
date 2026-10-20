"""Fit our post-shot xG to Opta's value for every on-target shot on disk.

``football_analysis.xg.psxg_model`` explains why the model exists; this builds
it. Every stored FotMob shot map is paired with its package's events by
``reference_xg``, and every on-target shot Opta prices becomes one row: our
pre-shot xG, where the ball crossed the line, whether it was headed, and Opta's
xGOT as the target.

    python scripts/fit_psxg.py              # fit and write
    python scripts/fit_psxg.py --dry-run    # report, write nothing

Scores are out of fold with whole matches as folds. Nothing is written unless
the fit is closer to Opta than the placement heuristic, its total stays within
five percent of Opta's, and a corner is still harder to save than a shot at
the keeper.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fit_xg_alignment import _sigmoid  # noqa: E402
from fit_xg_reference import _fit  # noqa: E402

from football_analysis.metrics import match_metrics as MM  # noqa: E402
from football_analysis.paths import MODELS_DIR  # noqa: E402
from football_analysis.xg import psxg_model as PM  # noqa: E402
from football_analysis.xg import reference_xg as RX  # noqa: E402

FOLDS = 5
MAX_TOTAL_DRIFT = 0.05


def _flag(row, column) -> bool:
    return str(row.get(column)).lower() in {"true", "1", "1.0"}


def build_table() -> pd.DataFrame:
    from football_analysis.pipeline import football_match_analysis as F

    # The heuristic is the baseline to beat, so neither Opta's stored value nor
    # a previous fit may stand in for it.
    PM._LOADED = None
    rows = []
    for path in sorted(RX.STORE.glob("*.json.gz")):
        payload = RX.cached(path.name.split(".")[0])
        package = (payload or {}).get("package")
        if not package:
            continue
        folder = ROOT / "output" / package
        if not (folder / "events.csv").is_file() or not (folder / "match_info.json").is_file():
            continue
        info = json.loads((folder / "match_info.json").read_text(encoding="utf-8-sig"))
        events = pd.read_csv(folder / "events.csv", low_memory=False)
        heuristic = MM.post_shot_xg(events.drop(columns=["xgot_reference"], errors="ignore"))
        ours, theirs = RX.our_shots(events, info), RX.their_shots(payload)
        for i, j in RX.pair(ours, theirs):
            if theirs[j]["xgot"] is None:
                continue
            index = ours[i]["index"]
            shot = events.loc[index]
            if str(shot.get("shot_whoscored_type")) not in MM.ON_TARGET_SHOT_TYPES:
                continue
            if _flag(shot, "is_penalty") or _flag(shot, "is_own_goal"):
                continue
            point = MM.shot_placement(shot)
            if point is None:
                continue
            # The pre-shot xG a render in the default mode publishes: ours,
            # not the Opta value a publish-mode render may have stored.
            try:
                xg = float(F._opta_like_local_xg_from_row(shot.to_dict()))
            except Exception:
                continue
            header = str(shot.get("body_part")) == "Head"
            rows.append(
                {
                    "pkg": package,
                    "goal": float(str(shot.get("shot_whoscored_type")) == "Goal"),
                    "heuristic": float(heuristic.at[index]),
                    "opta": float(theirs[j]["xgot"]),
                    **PM.features(xg, point[0], point[1], header),
                }
            )
    return pd.DataFrame(rows)


def _out_of_fold(frame, target, folds):
    design = frame[list(PM.TERMS)].to_numpy(dtype=float)
    predicted = np.zeros(len(frame))
    for fold in range(FOLDS):
        train, test = folds != fold, folds == fold
        weights = _fit(design[train], target[train])
        predicted[test] = _sigmoid(design[test] @ weights)
    return predicted


def _report(name, frame, predicted):
    target = frame["opta"].to_numpy()
    by_match = pd.DataFrame({"pkg": frame["pkg"], "p": predicted, "opta": target})
    by_match = by_match.groupby("pkg").sum(numeric_only=True)
    print(
        f"   {name:<20} rmse vs Opta {np.sqrt(np.mean((predicted - target) ** 2)):.4f}"
        f" | match total off by {np.mean(np.abs(by_match.p - by_match.opta)):.3f}"
        f" | {predicted.sum():.1f} for {frame['goal'].sum():.0f} goals"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    frame = build_table()
    if frame.empty:
        print("No on-target shot could be paired with Opta's map; nothing written.")
        return 1
    print(f"{len(frame)} on-target shots over {frame['pkg'].nunique()} matches")
    target = frame["opta"].clip(1e-4, 1 - 1e-4).to_numpy(dtype=float)
    folds = np.array([sum(map(ord, name)) % FOLDS for name in frame["pkg"]])
    fitted = _out_of_fold(frame, target, folds)

    print("held-out matches:")
    _report("placement heuristic", frame, frame["heuristic"].to_numpy())
    _report("fitted (new)", frame, fitted)
    _report("Opta itself", frame, frame["opta"].to_numpy())

    rmse = lambda p: float(np.sqrt(np.mean((p - frame["opta"].to_numpy()) ** 2)))  # noqa: E731
    if rmse(fitted) >= rmse(frame["heuristic"].to_numpy()):
        print("The fit is no closer to Opta than the heuristic; nothing written.")
        return 1
    drift = abs(fitted.sum() / frame["opta"].sum() - 1)
    if drift > MAX_TOTAL_DRIFT:
        print(f"The fit's total is {drift:.1%} from Opta's; nothing written.")
        return 1

    weights = dict(
        zip(PM.TERMS, (float(w) for w in _fit(frame[list(PM.TERMS)].to_numpy(float), target)))
    )
    model = {"method": PM.METHOD, "weights": weights}
    at_keeper = PM.predict(0.10, 0.0, 0.5, False, model=model)
    in_corner = PM.predict(0.10, 0.9, 0.9, False, model=model)
    if not in_corner > at_keeper:
        print("A corner prices below a shot at the keeper; nothing written.")
        return 1

    payload = {
        **model,
        "reference": "Opta xGOT via FotMob shot maps",
        "fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shots": int(len(frame)),
        "matches": int(frame["pkg"].nunique()),
        "rmse_vs_reference": round(rmse(fitted), 4),
        "terms": list(PM.TERMS),
    }
    if args.dry_run:
        print("--dry-run: not written")
        return 0
    path = MODELS_DIR / PM.MODEL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
