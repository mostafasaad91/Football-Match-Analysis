"""Fit the xG correction against Opta's value for every shot, not against goals.

``fit_xg_alignment.py`` fits the layer on top of the local engine to what went
in. A few hundred fixtures of goals cannot price the rare shots: on today's
archive it learned "big chance" at +0.78 in log odds and "from a corner" at
-0.64, flat, whatever the chance looked like. Brighton 3-0 Arsenal shows the
cost -- a big chance struck from the byline came out at 0.19 where Opta has
0.05, Havertz one-on-one at 0.28 against 0.63, and the match read the other way
round from every published source.

Opta's number for a shot is a far richer target than a goal: a probability
from a model trained on millions of shots, where a goal is one coin toss. So
this fits the same layer, same reader (``xg_alignment.align``), to that number.
The references come from ``collect_reference_xg.py`` and are only needed here;
nothing at render time reaches FotMob.

    python scripts/fit_xg_reference.py              # fit and write
    python scripts/fit_xg_reference.py --dry-run    # report, write nothing

Every score is out of fold, folds are whole matches, and the written fit has to
beat both the engine and the goal-fitted layer on held-out matches. Two guards
come with fitting to another model: the result is also scored against goals, so
agreeing with Opta cannot come at the price of predicting worse; and xG has to
keep falling with distance, which a free geometric term could otherwise bend.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import reference_xg as RX  # noqa: E402
import xg_alignment as XA  # noqa: E402
from fit_xg_alignment import _log_loss, _sigmoid  # noqa: E402

FOLDS = 5
GAIN = 0.0002          # soft targets carry far more signal than goals do
RIDGE = 2.0


def _cross_entropy(predicted, target) -> float:
    p = np.clip(np.asarray(predicted, dtype=float), 1e-9, 1 - 1e-9)
    t = np.asarray(target, dtype=float)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def build_table() -> pd.DataFrame:
    import football_match_analysis as F

    # The engine's own number is what this layer sits on, so the stored layer
    # is switched off while it is read.
    F._XG_ALIGNMENT = None
    shipped = XA.load()
    rows, unmatched_ours, unmatched_theirs = [], 0, 0
    # Every stored shot map names the package it was fetched for, whether the
    # collector fetched it or a render did, so the archive this fits on grows
    # with every fixture rendered and nothing has to be listed by hand.
    for path in sorted(RX.STORE.glob("*.json.gz")):
        payload = RX.cached(path.name.split(".")[0])
        package = (payload or {}).get("package")
        if not package:
            continue
        folder = ROOT / "output" / package
        if not (folder / "events.csv").is_file() or not (folder / "match_info.json").is_file():
            continue
        info = json.loads((folder / "match_info.json").read_text(encoding="utf-8-sig"))
        side_of = {info.get("home_id"): "home", info.get("away_id"): "away"}
        events = pd.read_csv(folder / "events.csv", low_memory=False)
        shots = events[events["is_shot"].fillna(False).astype(bool)].to_dict("records")
        ours = []
        for shot in shots:
            if shot.get("is_own_goal") or shot.get("is_penalty") or shot.get("is_penalty_shootout"):
                continue
            try:
                raw = float(F._opta_like_local_xg_from_row(shot))
                geometry = F._shot_geometry_features(shot)
                context = F._shot_context_features(shot, geometry)
                design = XA.features(raw, geometry, context, shot)
            except Exception:
                continue
            # Whatever layer ships today: the goal fit before the first run of
            # this script, this script's own previous fit after it.
            goal_layer = (XA.align(raw, geometry, context, shot, stored=shipped)
                          if shipped else raw)
            ours.append({"event_id": shot.get("event_id"),
                         "side": side_of.get(shot.get("team_id")),
                         "minute": int(shot.get("minute") or 0),
                         "period": str(shot.get("period") or ""),
                         "sur": RX.surname(shot.get("player")),
                         "x": float(shot.get("x") or 0) * 1.05,
                         "y": float(shot.get("y") or 0) * 0.68,
                         "goal": float(bool(shot.get("is_goal"))),
                         "engine": raw, "goal_layer": goal_layer,
                         "distance": float(geometry["distance"]),
                         "design": design})
        theirs = RX.their_shots(payload)
        pairs = RX.pair(ours, theirs)
        unmatched_ours += len(ours) - len(pairs)
        unmatched_theirs += len(theirs) - len(pairs)
        for i, j in pairs:
            a = ours[i]
            rows.append({"pkg": package, "event_id": a["event_id"],
                         "side": a["side"], "minute": a["minute"],
                         "sur": a["sur"], "goal": a["goal"], "engine": a["engine"],
                         "goal_layer": a["goal_layer"], "distance": a["distance"],
                         "opta": theirs[j]["xg"], **a["design"]})
    table = pd.DataFrame(rows)
    table.attrs["unmatched"] = (unmatched_ours, unmatched_theirs)
    return table


def _fit(design, target, ridge=RIDGE):
    """Ridge logistic regression on a soft target; const and logit unpenalised."""
    weights = np.zeros(design.shape[1])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = penalty[1, 1] = 0.0
    weights[1] = 1.0
    for _ in range(100):
        probability = _sigmoid(design @ weights)
        gradient = design.T @ (probability - target) + penalty @ weights
        hessian = (design * (probability * (1 - probability))[:, None]).T @ design
        step = np.linalg.solve(hessian + penalty + 1e-6 * np.eye(design.shape[1]), gradient)
        weights = weights - step
        if np.max(np.abs(step)) < 1e-9:
            break
    return weights


def _out_of_fold(frame, terms, target, folds):
    design = frame[list(terms)].to_numpy(dtype=float)
    predicted = np.zeros(len(frame))
    for fold in range(FOLDS):
        train, test = folds != fold, folds == fold
        weights = _fit(design[train], target[train])
        predicted[test] = _sigmoid(design[test] @ weights)
    return predicted


def choose_terms(frame, target, folds):
    chosen = ["const", "logit"]
    best = _cross_entropy(_out_of_fold(frame, chosen, target, folds), target)
    remaining = [t for t in XA.TERMS if t not in chosen]
    while remaining:
        loss, term = min((_cross_entropy(_out_of_fold(frame, chosen + [t], target, folds),
                                         target), t) for t in remaining)
        if best - loss < GAIN:
            break
        chosen.append(term); remaining.remove(term); best = loss
        print(f"   + {term:<20} cross-entropy {loss:.5f}")
    return chosen, best


def _falls_with_distance(frame, predicted) -> bool:
    """Ordinary footed shots: the mean must not rise from one 5 m band to the next."""
    plain = frame[(frame["header"] == 0) & (frame["big_chance"] == 0)]
    bands = pd.cut(plain["distance"], [5, 10, 15, 20, 25, 30])
    means = pd.Series(predicted[plain.index]).groupby(bands.values, observed=True).mean()
    print("   by distance:", "  ".join(f"{b.left:.0f}-{b.right:.0f}m {v:.3f}" for b, v in means.items()))
    return bool(np.all(np.diff(means.to_numpy()) <= 0.002))


def _report(name, frame, predicted):
    target, goals = frame["opta"].to_numpy(), frame["goal"].to_numpy()
    by_team = pd.DataFrame({"pkg": frame["pkg"], "side": frame["side"], "p": predicted,
                            "opta": target}).groupby(["pkg", "side"]).sum(numeric_only=True)
    print(f"   {name:<22} rmse vs Opta {np.sqrt(np.mean((predicted - target) ** 2)):.4f}"
          f" | team-match total off by {np.mean(np.abs(by_team.p - by_team.opta)):.3f}"
          f" | goal log loss {_log_loss(predicted, goals):.5f}"
          f" | {predicted.sum():.1f} xG for {goals.sum():.0f} goals")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    frame = build_table()
    ours_left, theirs_left = frame.attrs["unmatched"]
    print(f"{len(frame)} shots paired over {frame['pkg'].nunique()} matches "
          f"({ours_left} of ours and {theirs_left} of Opta's left unpaired)")
    target = frame["opta"].to_numpy(dtype=float)
    folds = np.array([sum(map(ord, name)) % FOLDS for name in frame["pkg"]])

    print("choosing terms:")
    terms, _ = choose_terms(frame, target, folds)
    fitted = _out_of_fold(frame, terms, target, folds)
    print(f"chosen: {', '.join(terms)}\n\nheld-out matches:")
    _report("engine alone", frame, frame["engine"].to_numpy())
    _report("what ships today", frame, frame["goal_layer"].to_numpy())
    _report("Opta-fitted (new)", frame, fitted)
    _report("Opta itself", frame, target)

    print("\nguard:")
    if not _falls_with_distance(frame, fitted):
        print("xG rises with distance somewhere; nothing written.")
        return 1
    rmse = lambda p: np.sqrt(np.mean((p - target) ** 2))  # noqa: E731
    if rmse(fitted) >= rmse(frame["engine"].to_numpy()):
        print("The fit is no closer to Opta than the engine alone; nothing written.")
        return 1
    # What ships was fitted on most of these shots, so its score here is in
    # sample and flatters it; the new fit's is out of fold. Holding the refit to
    # beating it outright would freeze the first fit for the season. A new fit
    # well behind it, though, means something broke, and that is not written.
    if rmse(fitted) > rmse(frame["goal_layer"].to_numpy()) + 0.003:
        print("The fit is well behind what ships; nothing written.")
        return 1

    weights = dict(zip(terms, (float(w) for w in _fit(frame[terms].to_numpy(dtype=float), target))))
    payload = {"method": XA.REFERENCE_METHOD, "reference": "Opta via FotMob shot maps",
               "fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "shots": int(len(frame)), "matches": int(frame["pkg"].nunique()),
               "terms": terms, "rmse_vs_reference": float(rmse(fitted)),
               "weights": weights}
    if args.dry_run:
        print("\n--dry-run: not written")
        return 0
    path = ROOT / XA.ALIGNMENT_FILE
    if path.is_file():
        keep = ROOT / "logs" / f"xg_alignment.before_reference.{datetime.now():%Y%m%d-%H%M%S}.json"
        keep.parent.mkdir(exist_ok=True)
        shutil.copy2(path, keep)
        print(f"\nprevious fit kept at {keep.relative_to(ROOT)}")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
