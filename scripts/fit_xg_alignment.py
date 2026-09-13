"""Fit the correction that sits on top of the local xG engine, against goals.

The engine prices a chance from geometry and a dozen context flags, with a
distance calibration earned on held-out matches. Read against what actually
went in, it still runs hot and it mis-weighs one thing badly: 2050 non-penalty
shots on disk carry 224 expected goals for 208 scored, and inside that the
chances the provider flags "big chance" convert at 33% while the engine calls
them 27%, with everything else called 7.4% against an outcome of 5.0%.

This fits that correction. The target is the goal, not another model's
published number: a model trained to agree with a competitor inherits the
competitor's errors, and on this archive that competitor prices 206 for 196.

    python scripts/fit_xg_alignment.py              # fit and write
    python scripts/fit_xg_alignment.py --dry-run    # report, write nothing
    python scripts/fit_xg_alignment.py --report     # also score against the
                                                    # cached provider pull

How many terms it uses is not fixed. Two hundred goals support three
coefficients and not twenty, so the script adds one term at a time and keeps
the addition only while the out-of-fold loss falls by more than a threshold.
On today's archive that stops at three; as the archive grows the same run will
keep more, which is what "it improves with more matches" has to mean if it is
to mean anything.

Every score is out of fold and the folds are whole matches. Two shots from one
game are not independent, so splitting them across the divide would report a
fit better than the one that ships.

Post-shot facts are excluded by construction. Whether the ball was blocked, and
how far it travelled first, describe the chance -- but a blocked shot never
scores, so a model given that column would learn the outcome instead of the
chance and score beautifully while saying nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import xg_alignment as XA  # noqa: E402

REFERENCE_CACHE = ROOT / "logs" / "xg_reference_shots.csv"
FOLDS = 5
MIN_SHOTS, MIN_GOALS, MIN_MATCHES = 600, 80, 20
# A term has to earn its place by this much out-of-fold log loss, or the next
# league's data will simply move it around.
GAIN = 0.0015


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _log_loss(probability, outcome) -> float:
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    return float(-np.mean(outcome * np.log(p) + (1 - outcome) * np.log(1 - p)))


def surname(name: str) -> str:
    plain = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    parts = re.sub(r"[^a-z ]", " ", plain.lower()).split()
    return parts[-1] if parts else ""


def packages() -> list[Path]:
    return [p.parent for p in sorted((ROOT / "output").rglob("match_info.json"))
            if p.parent.name != "light"
            and not any(x.startswith(".") for x in p.parent.parts)
            and (p.parent / "events.csv").exists()]


def shots_on_disk(folders: list[Path]) -> pd.DataFrame:
    """Every non-penalty shot with its engine value, design row and outcome."""
    import football_match_analysis as F

    rows = []
    for folder in folders:
        events = pd.read_csv(folder / "events.csv", low_memory=False)
        if "is_shot" not in events:
            continue
        for shot in events[events["is_shot"].fillna(False).astype(bool)].to_dict("records"):
            # Own goals carry the scorer's own end as the location and no value;
            # penalties are priced identically by every model and teach nothing.
            if shot.get("is_own_goal") or shot.get("is_penalty"):
                continue
            value = shot.get("xG")
            if value is None or pd.isna(value):
                continue
            try:
                geometry = F._shot_geometry_features(shot)
                context = F._shot_context_features(shot, geometry)
            except Exception:
                continue
            rows.append({"pkg": folder.name, "minute": shot.get("minute"),
                         "sur": surname(shot.get("player")),
                         "goal": float(bool(shot.get("is_goal"))),
                         **XA.features(float(value), geometry, context, shot)})
    return pd.DataFrame(rows)


def _fit(design, outcome, offset, ridge=1.0):
    """Ridge-penalised logistic regression around a fixed offset.

    The engine's own log odds enter as the offset rather than as a term to be
    weighed. Left free, its coefficient fell to 0.35 on this archive: two
    hundred goals cannot argue with shrinkage, and the loss falls fastest by
    squashing every shot toward the base rate. That reads well as a number and
    is nonsense as a model -- a tap-in came out at 0.15 and a forty-metre shot
    at 0.026, outside every published range. Fixing the slope at one keeps the
    engine's ordering and its spread, and asks only where the level is wrong.
    """
    weights = np.zeros(design.shape[1])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(200):
        probability = _sigmoid(offset + design @ weights)
        gradient = design.T @ (probability - outcome) + penalty @ weights
        hessian = (design * (probability * (1 - probability))[:, None]).T @ design
        hessian = hessian + penalty + 1e-6 * np.eye(design.shape[1])
        step = np.linalg.solve(hessian, gradient)
        weights = weights - step
        if np.max(np.abs(step)) < 1e-9:
            break
    return weights


def _out_of_fold(frame, terms, outcome, folds):
    offset = frame["logit"].to_numpy(dtype=float)
    design = frame[list(terms)].to_numpy(dtype=float)
    predicted = np.zeros(len(frame))
    for fold in range(FOLDS):
        train, test = folds != fold, folds == fold
        if not train.any() or not test.any():
            continue
        weights = _fit(design[train], outcome[train], offset[train])
        predicted[test] = _sigmoid(offset[test] + design[test] @ weights)
    return predicted


def choose_terms(frame, outcome, folds) -> tuple[list[str], float]:
    """Add terms while each one still pays for itself out of fold."""
    chosen = ["const"]
    best = _log_loss(_out_of_fold(frame, chosen, outcome, folds), outcome)
    # "logit" is the offset, not a term, and geometry belongs to the engine.
    remaining = [t for t in XA.LEVEL_TERMS if t not in chosen]
    while remaining:
        scored = []
        for term in remaining:
            loss = _log_loss(_out_of_fold(frame, chosen + [term], outcome, folds), outcome)
            scored.append((loss, term))
        loss, term = min(scored)
        if best - loss < GAIN:
            break
        chosen.append(term)
        remaining.remove(term)
        best = loss
        print(f"   + {term:<20} log loss {loss:.5f}")
    return chosen, best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    parser.add_argument("--report", action="store_true",
                        help="also score against the cached provider pull")
    args = parser.parse_args()

    folders = packages()
    if not folders:
        print("No rendered packages to fit on.")
        return 1
    frame = shots_on_disk(folders)
    goals = int(frame["goal"].sum()) if len(frame) else 0
    matches = frame["pkg"].nunique() if len(frame) else 0
    print(f"{len(frame)} shots, {goals} goals, {matches} matches")
    if len(frame) < MIN_SHOTS or goals < MIN_GOALS or matches < MIN_MATCHES:
        print(f"Too little to fit: need {MIN_SHOTS} shots, {MIN_GOALS} goals, "
              f"{MIN_MATCHES} matches.")
        return 1

    outcome = frame["goal"].to_numpy(dtype=float)
    folds = np.array([hash(name) % FOLDS for name in frame["pkg"]])
    engine = _sigmoid(frame["logit"].to_numpy())
    before = _log_loss(engine, outcome)
    print(f"engine as it stands   log loss {before:.5f}   {engine.sum():.1f} xG for {goals} goals")

    print("choosing terms:")
    terms, after = choose_terms(frame, outcome, folds)
    fitted = _out_of_fold(frame, terms, outcome, folds)
    print(f"chosen: {', '.join(terms)}")
    print(f"fitted                log loss {after:.5f}   {fitted.sum():.1f} xG for {goals} goals")

    if after >= before:
        print("The fit predicts no better than the engine; nothing written.")
        return 1

    weights = dict(zip(terms, (float(w) for w in _fit(
        frame[terms].to_numpy(dtype=float), outcome,
        frame["logit"].to_numpy(dtype=float)))))
    # The reader multiplies every stored name by its feature; the offset is a
    # coefficient of exactly one on the engine's own log odds.
    weights["logit"] = 1.0

    if args.report and REFERENCE_CACHE.is_file():
        reference = pd.read_csv(REFERENCE_CACHE)
        pairs = frame.merge(reference, on=["pkg", "minute", "sur"], how="inner")
        if len(pairs) > 200:
            mine = _sigmoid(pairs["logit"].to_numpy())
            print(f"\nagainst the cached provider pull ({len(pairs)} shots):")
            print(f"   engine rmse {np.sqrt(np.mean((mine - pairs.reference) ** 2)):.4f}")

    payload = {"method": XA.METHOD, "shots": int(len(frame)), "goals": goals,
               "matches": int(matches), "terms": terms,
               "log_loss_before": before, "log_loss_after": after,
               "xg_before": float(engine.sum()), "xg_after": float(fitted.sum()),
               "weights": weights}
    if args.dry_run:
        print("\n--dry-run: not written")
        return 0
    path = ROOT / XA.ALIGNMENT_FILE
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
