"""VAEP: Valuing Actions by Estimating Probabilities (Decroos et al., 2019).

Every on-ball action changes two probabilities: that the acting team scores
within the next ten actions, and that it concedes within them. An action's
value is how much it raised the first and lowered the second:

    VAEP(a_i) = [P_score(a_i) - P_score(a_{i-1})] - [P_concede(a_i) - P_concede(a_{i-1})]

with the previous action's probabilities flipped to the acting team's point of
view when the opponent made it, and reset after a goal. Unlike xG and xA it
credits defenders and midfielders: a tackle that ends a dangerous attack, or a
pass that moves the ball into a better zone without being a "key pass".

Two gradient-boosted classifiers are fitted on every stored match
(``scripts/fit_vaep.py``) and saved to ``data/models/vaep_model.joblib``. The
features follow the paper's SPADL set, built from WhoScored events: action
type, result, body part, start and end location, distance and angle to goal,
movement, and the same for the two previous actions, plus time gaps, whether
those actions were by the same team, and the score difference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from football_analysis.paths import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "vaep_model.joblib"
LOOKAHEAD = 10  # actions
SHOT_TYPES = {"Goal", "SavedShot", "MissedShots", "ShotOnPost"}
ACTION_TYPES = [
    "Pass", "TakeOn", "Shot", "Tackle", "Interception", "Clearance", "BallRecovery",
    "Aerial", "BlockedPass", "Dispossessed", "BallTouch", "Challenge", "Foul",
    "KeeperPickup", "Save", "Claim", "Punch", "KeeperSweeper", "OffsidePass", "Error",
    "CornerAwarded", "Other",
]
_ON_BALL = set(ACTION_TYPES) | SHOT_TYPES
_SET_PIECE = ("CornerTaken", "FreekickTaken", "ThrowIn", "GoalKick", "IndirectFreekickTaken")


def _truthy(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.fillna(False).to_numpy()
    return series.astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy()


def actions(events: pd.DataFrame) -> pd.DataFrame:
    """On-ball actions in match order, with the columns the features need."""
    if events is None or events.empty:
        return pd.DataFrame()
    work = events.copy()
    work["_row"] = np.arange(len(work))
    work = work[work["type"].isin(_ON_BALL)].copy()
    if "is_penalty_shootout" in work.columns:
        work = work[~_truthy(work["is_penalty_shootout"])]
    period = work.get("period_code", pd.Series("", index=work.index)).astype(str).str.lower()
    work["_period"] = period.map({"1h": 1, "2h": 2, "et1": 3, "et2": 4}).fillna(1).astype(int)
    work["_t"] = (pd.to_numeric(work["minute"], errors="coerce").fillna(0) * 60
                  + pd.to_numeric(work.get("second", 0), errors="coerce").fillna(0))
    work = work.sort_values(["_period", "_t", "_row"], kind="stable").reset_index()
    work.rename(columns={"index": "event_index"}, inplace=True)
    for col in ("x", "y", "end_x", "end_y"):
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    work["end_x"] = work["end_x"].fillna(work["x"])
    work["end_y"] = work["end_y"].fillna(work["y"])
    work[["x", "y", "end_x", "end_y"]] = work[["x", "y", "end_x", "end_y"]].fillna(50.0)
    work["_type"] = np.where(work["type"].isin(SHOT_TYPES), "Shot",
                             np.where(work["type"].isin(ACTION_TYPES), work["type"], "Other"))
    work["_ok"] = (work.get("outcome", pd.Series("", index=work.index)).astype(str).str.lower()
                   .eq("successful") | work["type"].eq("Goal")).astype(int)
    work["_team"] = pd.to_numeric(work["team_id"], errors="coerce").fillna(-1).astype(int)
    q = work.get("qualifier_names", pd.Series("", index=work.index)).astype(str)
    work["_head"] = (q.str.contains("Head", regex=False)
                     | _truthy(work.get("is_header", pd.Series(False, index=work.index)))).astype(int)
    work["_set_piece"] = np.zeros(len(work), dtype=int)
    for name in _SET_PIECE:
        work["_set_piece"] |= q.str.contains(name, regex=False).astype(int)
    goal = work["type"].eq("Goal").to_numpy()
    own = _truthy(work.get("is_own_goal", pd.Series(False, index=work.index)))
    scorer = work["_team"].to_numpy().copy()
    # An own goal is logged on the defending side; it counts for the other.
    teams = [t for t in work["_team"].unique() if t >= 0]
    if len(teams) == 2:
        other = {teams[0]: teams[1], teams[1]: teams[0]}
        scorer = np.where(own & goal, [other.get(t, t) for t in scorer], scorer)
    work["_goal_for"] = np.where(goal, scorer, -1)
    return work


def _goal_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = (100.0 - x) * 1.05
    dy = (50.0 - y) * 0.68
    return np.hypot(dx, dy), np.abs(np.arctan2(dy, np.maximum(dx, 1e-6)))


def features(acts: pd.DataFrame) -> pd.DataFrame:
    """The SPADL-style feature table, one row per action."""
    if acts.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=acts.index)
    team = acts["_team"].to_numpy()
    score_for = np.zeros(len(acts))
    score_against = np.zeros(len(acts))
    goals: dict[int, int] = {}
    for i, (t, g) in enumerate(zip(team, acts["_goal_for"].to_numpy())):
        score_for[i] = goals.get(t, 0)
        score_against[i] = sum(v for k, v in goals.items() if k != t)
        if g >= 0:
            goals[g] = goals.get(g, 0) + 1
    out["score_diff"] = score_for - score_against
    out["period"] = acts["_period"].to_numpy()
    out["time"] = acts["_t"].to_numpy() / 60.0
    for lag in (0, 1, 2):
        s = acts.shift(lag) if lag else acts
        tag = f"a{lag}"
        x, y = s["x"].to_numpy(dtype=float), s["y"].to_numpy(dtype=float)
        ex, ey = s["end_x"].to_numpy(dtype=float), s["end_y"].to_numpy(dtype=float)
        same = (s["_team"].to_numpy() == team) if lag else np.ones(len(acts), dtype=bool)
        # Previous actions by the opponent are seen from the acting team's side.
        x = np.where(same, x, 100 - x)
        y = np.where(same, y, 100 - y)
        ex = np.where(same, ex, 100 - ex)
        ey = np.where(same, ey, 100 - ey)
        dist, angle = _goal_xy(x, y)
        edist, _ = _goal_xy(ex, ey)
        out[f"{tag}_x"], out[f"{tag}_y"] = x, y
        out[f"{tag}_end_x"], out[f"{tag}_end_y"] = ex, ey
        out[f"{tag}_dist"], out[f"{tag}_angle"], out[f"{tag}_end_dist"] = dist, angle, edist
        out[f"{tag}_dx"] = ex - x
        out[f"{tag}_ok"] = s["_ok"].to_numpy(dtype=float)
        out[f"{tag}_head"] = s["_head"].to_numpy(dtype=float)
        out[f"{tag}_set_piece"] = s["_set_piece"].to_numpy(dtype=float)
        out[f"{tag}_type"] = pd.Categorical(s["_type"], categories=ACTION_TYPES).codes
        if lag:
            out[f"{tag}_same_team"] = same.astype(float)
            out[f"{tag}_dt"] = acts["_t"].to_numpy() - s["_t"].to_numpy()
    return out.fillna(0.0)


def labels(acts: pd.DataFrame, lookahead: int = LOOKAHEAD) -> tuple[np.ndarray, np.ndarray]:
    """Does the acting team score / concede within the next ``lookahead`` actions?"""
    n = len(acts)
    team = acts["_team"].to_numpy()
    goal_for = acts["_goal_for"].to_numpy()
    period = acts["_period"].to_numpy()
    scores = np.zeros(n, dtype=int)
    concedes = np.zeros(n, dtype=int)
    for k in range(lookahead):
        g = np.roll(goal_for, -k)
        p = np.roll(period, -k)
        valid = (np.arange(n) + k < n) & (p == period)
        scores |= (valid & (g == team)).astype(int)
        concedes |= (valid & (g >= 0) & (g != team)).astype(int)
    return scores, concedes


_MODEL_MEMO: list = []


def load_model(path: Path = MODEL_PATH):
    if not _MODEL_MEMO:
        try:
            import joblib

            _MODEL_MEMO.append(joblib.load(path))
        except Exception:
            _MODEL_MEMO.append(None)
    return _MODEL_MEMO[0]


def action_values(events: pd.DataFrame, model=None) -> pd.DataFrame:
    """One row per on-ball action: offensive, defensive and total VAEP."""
    model = load_model() if model is None else model
    acts = actions(events)
    columns = ["event_index", "player", "team_id", "type", "minute", "x", "y",
               "offensive", "defensive", "vaep"]
    if model is None or acts.empty:
        return pd.DataFrame(columns=columns)
    X = features(acts)[model["columns"]]
    p_score = model["scores"].predict_proba(X)[:, 1]
    p_concede = model["concedes"].predict_proba(X)[:, 1]
    team = acts["_team"].to_numpy()
    prev_team = np.roll(team, 1)
    prev_score = np.roll(p_score, 1)
    prev_concede = np.roll(p_concede, 1)
    same = prev_team == team
    # Seen from the acting team: an opponent's chance to score is our chance to concede.
    base_score = np.where(same, prev_score, prev_concede)
    base_concede = np.where(same, prev_concede, prev_score)
    reset = np.zeros(len(acts), dtype=bool)
    reset[0] = True
    reset[1:] |= acts["_goal_for"].to_numpy()[:-1] >= 0  # kick-off after a goal
    reset[1:] |= acts["_period"].to_numpy()[1:] != acts["_period"].to_numpy()[:-1]
    base_score = np.where(reset, 0.0, base_score)
    base_concede = np.where(reset, 0.0, base_concede)
    offensive = p_score - base_score
    defensive = -(p_concede - base_concede)
    frame = acts[["event_index", "player", "team_id", "type", "minute", "x", "y"]].copy()
    frame["offensive"] = offensive.round(4)
    frame["defensive"] = defensive.round(4)
    frame["vaep"] = (offensive + defensive).round(4)
    return frame[columns]


def player_vaep(events: pd.DataFrame, players: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-player VAEP totals, with a per-90 rate when minutes are known."""
    values = action_values(events)
    columns = ["player", "team_id", "actions", "offensive", "defensive", "vaep", "minutes", "vaep_per90"]
    if values.empty:
        return pd.DataFrame(columns=columns)
    table = (values.dropna(subset=["player"])
             .groupby(["player", "team_id"], as_index=False)
             .agg(actions=("vaep", "size"), offensive=("offensive", "sum"),
                  defensive=("defensive", "sum"), vaep=("vaep", "sum")))
    minutes = {}
    if players is not None and not players.empty and "minutes" in players.columns:
        name_col = "name" if "name" in players.columns else "player"
        minutes = dict(zip(players[name_col].astype(str), pd.to_numeric(players["minutes"], errors="coerce")))
    table["minutes"] = table["player"].astype(str).map(minutes)
    table["vaep_per90"] = np.where(table["minutes"].fillna(0) >= 30,
                                   table["vaep"] / table["minutes"].clip(lower=1) * 90, np.nan)
    for col in ("offensive", "defensive", "vaep", "vaep_per90"):
        table[col] = table[col].round(3)
    return table.sort_values("vaep", ascending=False)[columns].reset_index(drop=True)


def fit(frames, model_path: Path = MODEL_PATH) -> dict:
    """Fit both classifiers on many matches' events and save them."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import brier_score_loss, roc_auc_score
    import joblib

    xs, ys, yc, groups = [], [], [], []
    for gid, events in enumerate(frames):
        acts = actions(events)
        if acts.empty:
            continue
        xs.append(features(acts))
        s, c = labels(acts)
        ys.append(s)
        yc.append(c)
        groups.append(np.full(len(acts), gid))
    X = pd.concat(xs, ignore_index=True)
    y_s, y_c, g = np.concatenate(ys), np.concatenate(yc), np.concatenate(groups)
    test = g % 10 == 0  # every tenth match held out
    categorical = [c.endswith("_type") for c in X.columns]
    report = {}
    models = {}
    for name, y in (("scores", y_s), ("concedes", y_c)):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=48,
                                             l2_regularization=1.0, categorical_features=categorical,
                                             random_state=7)
        clf.fit(X[~test], y[~test])
        p = clf.predict_proba(X[test])[:, 1]
        report[name] = {"auc": round(float(roc_auc_score(y[test], p)), 4),
                        "brier": round(float(brier_score_loss(y[test], p)), 5),
                        "base_rate": round(float(y.mean()), 4)}
        clf.fit(X, y)  # final model on everything
        models[name] = clf
    bundle = {"scores": models["scores"], "concedes": models["concedes"],
              "columns": list(X.columns), "lookahead": LOOKAHEAD,
              "matches": int(g.max() + 1) if len(g) else 0, "actions": int(len(X)),
              "validation": report}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    _MODEL_MEMO.clear()
    return bundle
