"""
Player radar / pizza charts.

One vertical-bar "pizza" per player, with the 20 metrics grouped and
colour-coded by category (attacking / passing / defence). Bar length is the
player's percentile among all match players; the chip shows the raw match value.

Public API
----------
compute_metrics_pool(events)            -> (allm, elig)
player_metrics(events, player)          -> dict
make_player_pizza(events, player, team_name, role, allm, elig, kit=None) -> Figure
export_player_radars(events, info, out_dir, dpi=115) -> dict
    saves every participating player's PNG under
    <out_dir>/player_radars/<TeamName>/<Player>.png and returns a ranking dict
    {"home": [(player, rating), ...], "away": [...]} sorted best-first.
top_players_per_team(events, info, n=5) -> {"home":[...], "away":[...]}
"""

from __future__ import annotations

import colorsys
import os
import re

import numpy as np
import pandas as pd

from frame_values import surname as _surname, text as _text
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from match_metrics import (
    box_entry_mask,
    player_sequence_metrics,
    progressive_pass_mask,
    touch_mask,
)
from visualization_components import (
    C_AWAY,
    C_HOME,
    C_GOLD,
    IS_LIGHT_THEME,
    contrast_ratio,
    label_outline,
    text_on_fill,
)

try:
    import visual_redesign_preview as _identity
    BG_DARK, TEXT_BRIGHT, TEXT_DIM = _identity.BG, _identity.TEXT, _identity.MUTED
except Exception:  # pragma: no cover - fallback colours
    _identity = None
    BG_DARK, TEXT_BRIGHT, TEXT_DIM = "#000000", "#F7F7F5", "#A3A3A3"

RADAR_GRID = _identity.GRID if _identity is not None else "#242424"


def team_group_colors(team_color: str, n_groups: int) -> list[str]:
    """Return one shade per metric group, all drawn from the team's own colour.

    A radar belongs to a player, and a player belongs to a team, so the whole
    chart should read in that team's colour rather than a fixed category
    palette. Hue and saturation are held constant and only lightness varies, so
    the groups stay separable while the radar still reads as one team.

    Kits with almost no saturation (white/silver sides) fall back to a grey
    ramp of the same shape, which is the honest rendering of a white shirt.
    """
    import colorsys

    try:
        r, g, b = mcolors.to_rgb(team_color)
    except (ValueError, TypeError):
        r, g, b = mcolors.to_rgb(C_HOME)
    hue, _lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    # Lightness range chosen so the darkest step still separates from the black
    # page and the lightest stays below pure white.
    lows, highs = 0.40, 0.80
    if n_groups <= 1:
        levels = [(lows + highs) / 2]
    else:
        levels = [
            lows + (highs - lows) * i / (n_groups - 1) for i in range(n_groups)
        ]
    # Alternate dark/light so neighbouring groups never sit on adjacent steps.
    ordered = []
    front, back = 0, len(levels) - 1
    while front <= back:
        ordered.append(levels[back])
        if front != back:
            ordered.append(levels[front])
        front += 1
        back -= 1
    return [
        mcolors.to_hex(colorsys.hls_to_rgb(hue, level, max(saturation, 0.06)))
        for level in ordered[:n_groups]
    ]

# ── Metric layout: (group name, colour, [metric labels]) ──────────────────────
# A full tactical + numerical match profile, grouped by role of the action.
# (Minutes are shown under the player name in the header, not as a slice.)
GROUPS = [
    (
        "ATTACK",
        C_GOLD,
        ["Goals", "Assists", "Big ch.\ncreated", "Shots", "xT\ncontrib", "Dribbles"],
    ),
    (
        "PASSING",
        C_HOME,
        [
            "Passes",
            "Pass %",
            "Prog\npasses",
            "Final 3rd\npasses",
            "Long\nballs",
            "Key\npasses",
        ],
    ),
    (
        "THREAT",
        C_HOME,
        [
            "xG",
            "npxG",
            "xA",
            "xGOT",
            "xG/\nShot",
            "G\N{MINUS SIGN}xG",
            "Shot-cr.\nactions",
            "Deep\ncompl.",
            "xG\nChain",
            "xG\nBuildup",
        ],
    ),
    (
        "DEFENCE",
        C_AWAY,
        ["Tackles\nwon", "Intercep\ntions", "Recov\neries", "Blocks", "Clear\nances"],
    ),
    ("DUELS", C_AWAY, ["Grd duels\nwon", "Aerials\nwon", "Duels\nwon"]),
]


# Three metric keys were written with the line break inside the word so they
# would fit the ring — they rendered as CLEAR/ANCES, RECOV/ERIES and
# INTERCEP/TIONS, which reads as a typesetting fault rather than a label. The
# keys are load-bearing (they index the metric dictionaries and the percentile
# lookups), so the repair happens at draw time: same key, readable label.
_LABEL_OVERRIDES = {
    "Clear\nances": "CLEARANCES",
    "Recov\neries": "RECOVERIES",
    "Intercep\ntions": "INTERCEPTS",
}


def display_label(key: str) -> str:
    """Return the drawn form of a metric key.

    Any break that survives is at a space the label already had, never inside
    a word.
    """
    override = _LABEL_OVERRIDES.get(key)
    if override is not None:
        return override
    return str(key).upper()


def _chip_text_color(color: str) -> str:
    """Keep metric values readable on the group-coloured chip fill.

    Contrast is measured against the *fill*, not the page, so the returned
    colours are absolute rather than the theme's text/background — otherwise a
    dark fill on the light theme would take dark text and vanish.

    This used to split on a fixed luminance of 0.36, which broke as soon as the
    chips took their colour from the team. Every ramp has mid steps that sit
    just under that line and got white text at 2.6–4.1 contrast (Man City
    #59a0d9, Liverpool #f47187, Juventus #7b95b7). Picking whichever tier
    actually measures highest against the fill fixes all of them at once.
    """
    return text_on_fill(color)


# The chip is an 8pt digit on a coloured tile, and WCAG's luminance formula
# flatters saturated hues: red carries a coefficient of 0.2126, so a fully
# saturated red computes as "dark" and near-black on it scores 4.9:1 while
# reading as a smudge. Arsenal's #fe0107 was exactly that. The tile is pushed
# until its own best ink clears a floor well above the nominal minimum.
CHIP_CONTRAST_FLOOR = 7.0


# How far apart two adjusted chips are kept, in lightness. Without it every
# group that needed moving stopped at the same boundary: Arsenal's passing,
# defence and duels chips all became #b60105 and three of the five groups
# stopped being told apart.
CHIP_SEPARATION = 0.055


def _chip_fill(color: str, floor: float = CHIP_CONTRAST_FLOOR,
               nudge: float = 0.0) -> str:
    """Move a group colour's lightness until a label on it is properly legible.

    The direction follows whichever tier the fill already prefers, so the pale
    groups stay pale and the deep ones stay deep. Only the saturated middle —
    where neither tier is comfortable — is moved far, and it goes darker,
    because the light end of the ramp is already occupied.

    ``nudge`` pushes further in the same direction, which only ever raises the
    contrast. ``chip_fills`` uses it to keep adjusted chips apart.
    """
    try:
        rgb = mcolors.to_rgb(color)
    except (ValueError, TypeError):
        return color
    if not nudge and contrast_ratio(text_on_fill(color), color) >= floor:
        return color

    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    # A fill that is already light goes lighter; anything else goes darker.
    target = 1.0 if lightness > 0.62 else 0.0
    low, high = lightness, target
    for _ in range(20):
        mid = (low + high) / 2
        candidate = mcolors.to_hex(colorsys.hls_to_rgb(hue, mid, saturation))
        if contrast_ratio(text_on_fill(candidate), candidate) >= floor:
            high = mid
        else:
            low = mid
    if nudge:
        high = float(np.clip(high + (nudge if target > lightness else -nudge),
                             0.06, 0.96))
    return mcolors.to_hex(colorsys.hls_to_rgb(hue, high, saturation))


def chip_fills(colors, floor: float = CHIP_CONTRAST_FLOOR) -> list[str]:
    """Legible chip fills for one radar's groups, still distinguishable.

    Taken as a set rather than one at a time, because the fix for legibility is
    to move a fill to the edge of the safe zone and several groups of the same
    kit arrive at the same edge. Each one after the first is pushed a little
    further, which keeps them apart and can only improve their contrast.
    """
    adjusted, moved = [], 0
    for colour in colors:
        try:
            safe = contrast_ratio(text_on_fill(colour), colour) >= floor
        except Exception:
            adjusted.append(colour)
            continue
        if safe:
            adjusted.append(colour)
            continue
        adjusted.append(_chip_fill(colour, floor, nudge=moved * CHIP_SEPARATION))
        moved += 1
    return adjusted


def _readable_on_page(color: str, min_ratio: float = 4.0) -> str:
    """Lift a group shade until it reads as text on the page background.

    The darkest step of a ramp is fine as a fill or a legend dot but too dim as
    a label on black. Blend it toward the page's text colour until it clears.
    """
    try:
        if contrast_ratio(color, BG_DARK) >= min_ratio:
            return color
        rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
        target = np.asarray(mcolors.to_rgb(TEXT_BRIGHT), dtype=float)
        for amount in np.linspace(0.1, 0.85, 16):
            lifted = mcolors.to_hex(rgb * (1 - amount) + target * amount)
            if contrast_ratio(lifted, BG_DARK) >= min_ratio:
                return lifted
        return TEXT_BRIGHT
    except Exception:
        return TEXT_BRIGHT
# metrics whose chip shows "numerator / denominator" instead of a single number.
# Value is (numerator_key, denominator_key); the bar still uses the label's own
# value. Passes/Shots/Long balls -> completed·on-target / total; duels -> won / contested.
_RATIO_DISPLAY = {
    "Passes": ("Passes_comp", "Passes"),
    "Shots": ("Shots_ot", "Shots"),
    "Long\nballs": ("Longballs_comp", "Long\nballs"),
    "Grd duels\nwon": ("Grd duels\nwon", "Grd_duels_att"),
    "Aerials\nwon": ("Aerials\nwon", "Aer_att"),
    "Duels\nwon": ("Duels\nwon", "Duels_att"),
}
MIN_POOL_TOUCHES = 12  # players below this don't seed the percentile pool


def _safe(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(name)).strip("_") or "player"


def _valid_name(x) -> bool:
    """True for a real player name (not NaN / blank / team-level placeholder)."""
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    s = str(x).strip()
    return len(s) >= 2 and s.lower() != "nan"


def _creation_credits(events: pd.DataFrame) -> dict:
    """Link each shot back to the key pass that created it and credit the passer.

    WhoScored/Opta feeds here leave ``assist_player`` empty, so xA / assists /
    big-chances-created / shot-creating-actions must be reconstructed from the
    event order: the shot's expected-goals value is attributed to the most
    recent key pass by a team-mate immediately before it.

    Returns {player: {"xA": float, "assists": int, "bcc": int, "sca": int}}.
    """
    from collections import defaultdict

    credits = defaultdict(lambda: {"xA": 0.0, "assists": 0, "bcc": 0, "sca": 0})
    if "is_shot" not in events.columns:
        return credits

    ev = events.reset_index(drop=True)
    is_shot = ev["is_shot"].fillna(False) == True
    is_kp = (
        ev.get("is_key_pass", pd.Series(False, index=ev.index)).fillna(False) == True
    )
    typ = ev["type"].astype(str) if "type" in ev else pd.Series("", index=ev.index)
    team = ev["team_id"] if "team_id" in ev else pd.Series(0, index=ev.index)
    minute = ev["minute"] if "minute" in ev else pd.Series(0, index=ev.index)
    xg = ev["xG"].fillna(0) if "xG" in ev else pd.Series(0.0, index=ev.index)
    goal = (
        ev["is_goal"].fillna(False) == True
        if "is_goal" in ev
        else pd.Series(False, index=ev.index)
    )
    own = (
        ev["is_own_goal"].fillna(False) == True
        if "is_own_goal" in ev
        else pd.Series(False, index=ev.index)
    )
    big = (
        ev["big_chance"].fillna(False) == True
        if "big_chance" in ev
        else pd.Series(False, index=ev.index)
    )

    shot_idx = list(ev.index[is_shot])
    for i in shot_idx:
        t = team.iloc[i]
        shooter = ev["player"].iloc[i]
        creator = None
        j = i - 1
        while j >= 0 and (i - j) <= 6 and abs(minute.iloc[i] - minute.iloc[j]) <= 1:
            if (
                is_kp.iloc[j]
                and typ.iloc[j] == "Pass"
                and team.iloc[j] == t
                and _valid_name(ev["player"].iloc[j])
                and ev["player"].iloc[j] != shooter
            ):
                creator = ev["player"].iloc[j]
                break
            j -= 1
        if creator is None or not _valid_name(creator):
            continue
        c = credits[str(creator)]
        c["xA"] += float(xg.iloc[i])
        c["sca"] += 1
        if bool(goal.iloc[i]) and not bool(own.iloc[i]):
            c["assists"] += 1
        if bool(big.iloc[i]):
            c["bcc"] += 1
    return credits


def _get_credits(events: pd.DataFrame) -> dict:
    """Memoise creation credits on the DataFrame so it is computed once per run."""
    try:
        cached = events.attrs.get("_radar_credits")
    except Exception:
        cached = None
    if cached is None:
        cached = _creation_credits(events)
        try:
            events.attrs["_radar_credits"] = cached
        except Exception:
            pass
    return cached


# ─────────────────────────────────────────────────────────────────────────────
# Participation & minutes played (WhoScored event-stream reconstruction)
# ─────────────────────────────────────────────────────────────────────────────
# WhoScored does not expose isFirstEleven / subbedInExpandedMinute as event
# columns here, so participation is reconstructed from the event stream:
#   • SubstitutionOn  event  -> player entered at that (expanded) minute
#   • SubstitutionOff event  -> player left at that (expanded) minute
#   • a player with on-ball/defensive events but no SubstitutionOn started
#   • each period ends at its recorded End event, including added time
# Every player is classified starter / sub / unused, and real minutes are the
# span they were actually on the pitch — the basis for any per-90 normalisation.
_MARKER_TYPES = {
    "SubstitutionOn",
    "SubstitutionOff",
    "Card",
    "FormationSet",
    "FormationChange",
    "Start",
    "End",
    "OffsideProvoked",
}


_PLAYING_PERIODS = (
    ("1h", 0, 45),
    ("2h", 45, 45),
    ("et1", 90, 15),
    ("et2", 105, 15),
)


def _normalise_period(value) -> str:
    """Return a stable code for a regulation or extra-time period."""
    raw = str(value or "").strip().lower().replace(" ", "")
    aliases = {
        "firsthalf": "1h",
        "firstperiod": "1h",
        "secondhalf": "2h",
        "secondperiod": "2h",
        "firstperiodofextratime": "et1",
        "extratimefirsthalf": "et1",
        "secondperiodofextratime": "et2",
        "extratimesecondhalf": "et2",
    }
    return aliases.get(raw, raw)


def _period_timeline(events: pd.DataFrame) -> dict:
    """Build an elapsed-time timeline that includes added time in every period.

    WhoScored timestamps use the nominal match clock: the second half begins at
    45:00 even when the first half lasted beyond 45 minutes. Extra time follows
    the same pattern at 90:00 and 105:00. Summing each period separately avoids
    losing added time from earlier periods.
    """
    if events is None or events.empty:
        return {}

    period_series = events.get("period_code", pd.Series("", index=events.index))
    period_codes = period_series.map(_normalise_period)
    minutes = pd.to_numeric(events.get("minute", 0), errors="coerce").fillna(0.0)
    seconds = pd.to_numeric(events.get("second", 0), errors="coerce").fillna(0.0)
    event_seconds = minutes * 60.0 + seconds.clip(lower=0.0, upper=59.999)

    timeline = {}
    elapsed_start = 0.0
    for code, clock_start_minute, nominal_minutes in _PLAYING_PERIODS:
        mask = period_codes == code
        if not mask.any():
            continue

        clock_start = clock_start_minute * 60.0
        period_events = event_seconds[mask]
        event_types = events.loc[mask, "type"].astype(str)
        end_events = period_events[event_types.eq("End")]
        clock_end = (
            float(end_events.max())
            if not end_events.empty
            else float(period_events.max())
        )
        duration = max(clock_end - clock_start, nominal_minutes * 60.0)
        timeline[code] = {
            "clock_start": clock_start,
            "elapsed_start": elapsed_start,
            "duration": duration,
            "elapsed_end": elapsed_start + duration,
        }
        elapsed_start += duration
    return timeline


def _elapsed_event_seconds(row, timeline: dict, default: float) -> float:
    """Translate a WhoScored period clock value to true elapsed match seconds."""
    code = _normalise_period(row.get("period_code", row.get("period", "")))
    period = timeline.get(code)
    if period is None:
        return default
    minute = pd.to_numeric(pd.Series([row.get("minute")]), errors="coerce").iloc[0]
    second = pd.to_numeric(pd.Series([row.get("second")]), errors="coerce").iloc[0]
    minute = 0.0 if pd.isna(minute) else float(minute)
    second = 0.0 if pd.isna(second) else min(max(float(second), 0.0), 59.999)
    offset = max(minute * 60.0 + second - period["clock_start"], 0.0)
    return min(period["elapsed_start"] + offset, period["elapsed_end"])


def _format_played_time(total_seconds: float) -> str:
    """Format an exact playing duration as minutes and seconds."""
    rounded_seconds = max(int(round(total_seconds)), 0)
    minutes, seconds = divmod(rounded_seconds, 60)
    return f"{minutes}′ {seconds:02d}″"


def player_participation(events: pd.DataFrame) -> dict:
    """Return {player: {"status": starter|sub|unused, "minutes": int,
    "start_seconds": float, "end_seconds": float}} reconstructed from events.

    Durations include added time in every period and extra time when played.
    Penalty shootouts are excluded."""
    timeline = _period_timeline(events)
    match_end = max(
        (period["elapsed_end"] for period in timeline.values()), default=0.0
    )
    ty = events["type"].astype(str)
    subs_on, subs_off, sent_off = {}, {}, {}
    for _, r in events[ty == "SubstitutionOn"].iterrows():
        if _valid_name(r.get("player")):
            subs_on[str(r["player"])] = _elapsed_event_seconds(r, timeline, 0.0)
    for _, r in events[ty == "SubstitutionOff"].iterrows():
        if _valid_name(r.get("player")):
            subs_off[str(r["player"])] = _elapsed_event_seconds(r, timeline, match_end)
    # a red card (straight or second yellow) ends the player's match too, even
    # though WhoScored logs no SubstitutionOff for a dismissal
    for _, r in events[ty == "Card"].iterrows():
        q = str(r.get("qualifier_names", "")).lower()
        if _valid_name(r.get("player")) and ("red" in q or "secondyellow" in q):
            p = str(r["player"])
            dismissal = _elapsed_event_seconds(r, timeline, match_end)
            sent_off[p] = min(sent_off.get(p, float("inf")), dismissal)

    real = events[~ty.isin(_MARKER_TYPES)]
    real_counts = real[real["player"].map(_valid_name)].groupby("player").size()

    out = {}
    players = set(str(p) for p in events["player"].tolist() if _valid_name(p))
    for p in players:
        has_actions = int(real_counts.get(p, 0)) > 0
        if p in subs_on:
            start = subs_on[p]
            end = subs_off.get(p, match_end)
            status = "sub"
        else:
            start = 0.0
            end = subs_off.get(p, match_end)
            status = "starter"
        # a dismissal ends the match earlier than any sub-off / full time
        if p in sent_off:
            end = min(end, sent_off[p])
            status = "sent_off"
        start = min(max(start, 0.0), match_end)
        end = min(max(end, start), match_end)
        played_seconds = max(end - start, 0.0)
        if played_seconds <= 0 or not has_actions:
            status = "unused"
            played_seconds = 0.0
        rounded_minutes = int(round(played_seconds / 60.0))
        out[p] = {
            "status": status,
            "minutes": rounded_minutes,
            "played_seconds": played_seconds,
            "played_time": _format_played_time(played_seconds),
            "start_seconds": start,
            "end_seconds": end,
        }
    return out


def _get_participation(events) -> dict:
    try:
        cached = events.attrs.get("_radar_participation")
    except Exception:
        cached = None
    if cached is None:
        cached = player_participation(events)
        try:
            events.attrs["_radar_participation"] = cached
        except Exception:
            pass
    return cached


# ─────────────────────────────────────────────────────────────────────────────
# Expected Threat (xT) — grid value model, Karun-Singh style
# ─────────────────────────────────────────────────────────────────────────────
# A transparent, from-scratch implementation of the open grid xT model. NOTE:
# this is an approximation from the same family as — but NOT a reproduction of —
# proprietary Opta possession-value or StatsBomb OBV outputs, and is trained
# only on the events available (a single match here, so the surface is coarse;
# feed a larger event corpus for a smoother model).
#
# Pitch is split into an nx (length, x∈[0,100]) × ny (width, y∈[0,100]) grid.
# For every cell z we estimate from the events:
#     shot_prob   s(z) = shots(z) / (shots(z) + moves(z))
#     move_prob   m(z) = 1 − s(z)
#     goal_prob   g(z) = goals(z) / shots(z)              (finish | shot in z)
#     transition  T(z→z') = moves z→z' / all moves from z
# The value of holding possession in z solves, by iteration to convergence:
#     V(z) = s(z)·g(z) + m(z)·Σ_z' T(z→z')·V(z')
# The xT added by a move is V(end cell) − V(start cell); summed per player this
# is their ball-progression threat contribution.
def compute_xt_grid(events, nx=16, ny=12, n_iter=500, eps=1e-7):
    ncell = nx * ny

    def cell(x, y):
        xi = min(max(int(x / 100.0 * nx), 0), nx - 1)
        yi = min(max(int(y / 100.0 * ny), 0), ny - 1)
        return yi * nx + xi

    move_from = np.zeros(ncell)
    shot_from = np.zeros(ncell)
    goal_from = np.zeros(ncell)
    Tcount = np.zeros((ncell, ncell))

    isp = (
        events.get("is_pass", pd.Series(False, index=events.index)).fillna(False)
        == True
    )
    succ = (
        events.get("outcome", pd.Series("", index=events.index)).astype(str)
        == "Successful"
    )
    have_xy = (
        events["x"].notna()
        & events["y"].notna()
        & events["end_x"].notna()
        & events["end_y"].notna()
    )
    mv = events[isp & succ & have_xy]
    for x, y, ex, ey in zip(mv["x"], mv["y"], mv["end_x"], mv["end_y"]):
        z, z2 = cell(x, y), cell(ex, ey)
        move_from[z] += 1
        Tcount[z, z2] += 1

    issh = (
        events.get("is_shot", pd.Series(False, index=events.index)).fillna(False)
        == True
    )
    sh = events[issh & events["x"].notna() & events["y"].notna()]
    isg = sh.get("is_goal", pd.Series(False, index=sh.index)).fillna(False) == True
    isog = sh.get("is_own_goal", pd.Series(False, index=sh.index)).fillna(False) == True
    for x, y, g in zip(sh["x"], sh["y"], (isg & ~isog)):
        z = cell(x, y)
        shot_from[z] += 1
        if bool(g):
            goal_from[z] += 1

    total = move_from + shot_from
    s = np.divide(shot_from, np.maximum(total, 1.0))
    m = 1.0 - s
    # Bayesian shrinkage of goal-per-shot toward the global rate so that a single
    # scored shot in a cell does not push g(z) to 1.0 on this small sample.
    g_global = (goal_from.sum() / shot_from.sum()) if shot_from.sum() else 0.10
    alpha = 4.0
    g = (goal_from + alpha * g_global) / (shot_from + alpha)
    Trow = Tcount.sum(axis=1, keepdims=True)
    T = np.divide(Tcount, np.maximum(Trow, 1.0))

    V = np.zeros(ncell)
    for _ in range(n_iter):
        V_new = s * g + m * (T @ V)
        if np.max(np.abs(V_new - V)) < eps:
            V = V_new
            break
        V = V_new
    return {"V": V, "nx": nx, "ny": ny}


def _get_xt_grid(events):
    try:
        cached = events.attrs.get("_radar_xtgrid")
    except Exception:
        cached = None
    if cached is None:
        cached = compute_xt_grid(events)
        try:
            events.attrs["_radar_xtgrid"] = cached
        except Exception:
            pass
    return cached


def _player_xt(events, player, grid=None):
    """Sum of V(end)-V(start) over the player's successful ball-moving actions."""
    grid = grid or _get_xt_grid(events)
    V, nx, ny = grid["V"], grid["nx"], grid["ny"]

    def cell(x, y):
        xi = min(max(int(x / 100.0 * nx), 0), nx - 1)
        yi = min(max(int(y / 100.0 * ny), 0), ny - 1)
        return yi * nx + xi

    d = events[events["player"].astype(str) == str(player)]
    isp = d.get("is_pass", pd.Series(False, index=d.index)).fillna(False) == True
    succ = d.get("outcome", pd.Series("", index=d.index)).astype(str) == "Successful"
    have = d["x"].notna() & d["y"].notna() & d["end_x"].notna() & d["end_y"].notna()
    mv = d[isp & succ & have]
    tot = 0.0
    for x, y, ex, ey in zip(mv["x"], mv["y"], mv["end_x"], mv["end_y"]):
        tot += float(V[cell(ex, ey)] - V[cell(x, y)])
    return tot


# ─────────────────────────────────────────────────────────────────────────────
# Carries (approximate) — WhoScored does not log carries directly.
# ─────────────────────────────────────────────────────────────────────────────
# APPROXIMATION: a carry is inferred when the same player has two consecutive
# on-ball events in the same possession and the ball location moved between the
# end of one and the start of the next. A "progressive" carry advances the ball
# at least PROG_CARRY (in 0–100 pitch units) towards the opponent goal. This is
# a coarse proxy, not an Opta/StatsBomb carry model.
_ONBALL_TYPES = {
    "Pass",
    "TakeOn",
    "BallTouch",
    "BallRecovery",
    "Clearance",
    "Interception",
    "Tackle",
}
PROG_CARRY = 5.0  # pitch units of forward progress for a progressive carry
MIN_CARRY = 3.0  # minimum displacement to count as a carry


def _compute_carries(events) -> dict:
    from collections import defaultdict

    out = defaultdict(lambda: {"carries": 0, "prog": 0})
    ev = events.reset_index(drop=True)
    ty = ev["type"].astype(str)
    prev = None
    for i in range(len(ev)):
        t = ty.iloc[i]
        if t not in _ONBALL_TYPES:
            prev = None
            continue
        x, y = ev["x"].iloc[i], ev["y"].iloc[i]
        pl, tm = ev["player"].iloc[i], ev["team_id"].iloc[i]
        if (
            prev is not None
            and _valid_name(pl)
            and prev["player"] == pl
            and prev["team"] == tm
            and pd.notna(x)
            and pd.notna(y)
            and pd.notna(prev["ex"])
            and pd.notna(prev["ey"])
        ):
            dist = ((x - prev["ex"]) ** 2 + (y - prev["ey"]) ** 2) ** 0.5
            if dist >= MIN_CARRY:
                c = out[str(pl)]
                c["carries"] += 1
                if (x - prev["ex"]) >= PROG_CARRY:
                    c["prog"] += 1
        ex = ev["end_x"].iloc[i] if pd.notna(ev["end_x"].iloc[i]) else x
        ey = ev["end_y"].iloc[i] if pd.notna(ev["end_y"].iloc[i]) else y
        prev = {"player": pl, "team": tm, "ex": ex, "ey": ey}
    return dict(out)


def _get_carries(events) -> dict:
    try:
        cached = events.attrs.get("_radar_carries")
    except Exception:
        cached = None
    if cached is None:
        cached = _compute_carries(events)
        try:
            events.attrs["_radar_carries"] = cached
        except Exception:
            pass
    return cached


def player_metrics(events: pd.DataFrame, player: str) -> dict:
    """Raw per-match stats for one player from the event stream."""
    ev = events
    d = ev[ev["player"].astype(str) == str(player)]
    ty = d["type"].astype(str) if "type" in d else pd.Series([], dtype=str)
    o = (
        d["outcome"].astype(str)
        if "outcome" in d
        else pd.Series(index=d.index, dtype=str)
    )
    isp = d.get("is_pass", False)
    isp = (
        (isp.fillna(False) == True)
        if hasattr(isp, "fillna")
        else pd.Series(False, index=d.index)
    )
    pc = d[isp & (o == "Successful")]

    prog = int(progressive_pass_mask(d).sum())
    ptobox = int(box_entry_mask(d).sum())
    tib = 0
    if {"x", "y"}.issubset(d.columns):
        tib = int((touch_mask(d) & (d["x"] >= 83) & d["y"].between(21, 79)).sum())

    xt_pos = 0.0
    if "xT" in d.columns:
        xt = d["xT"].fillna(0)
        xt_pos = float(xt[isp & (o == "Successful") & (xt > 0)].sum())

    cr = _get_credits(ev).get(
        str(player), {"xA": 0.0, "assists": 0, "bcc": 0, "sca": 0}
    )
    sequence = player_sequence_metrics(ev).get(
        str(player), {"xGChain": 0.0, "xGBuildup": 0.0, "sequence_xT": 0.0}
    )
    asst = int(cr["assists"])
    xa = round(float(cr["xA"]), 2)
    bcc = int(cr["bcc"])

    def cnt(t):
        return int((ty == t).sum())

    def cnt_ok(t):
        return int(((ty == t) & (o == "Successful")).sum())

    is_cross = d.get("is_cross", pd.Series(False, index=d.index)).fillna(False) == True
    cross_tot = int(is_cross.sum())
    cross_comp = int((is_cross & (o == "Successful")).sum())
    pass_tot = int(isp.sum())
    pass_comp = int(len(pc))
    shots_tot = int(
        (d.get("is_shot", pd.Series(False, index=d.index)).fillna(False) == True).sum()
    )
    shots_ot = int(
        d.get("shot_whoscored_type", pd.Series(index=d.index, dtype=object))
        .isin(["Goal", "SavedShot"])
        .sum()
    )

    goals = int((d["is_goal"].fillna(False) == True).sum()) if "is_goal" in d else 0
    xg_tot = float(d["xG"].fillna(0).sum()) if "xG" in d else 0.0
    pen = d.get("is_penalty", pd.Series(False, index=d.index)).fillna(False) == True
    pso = (
        d.get("is_penalty_shootout", pd.Series(False, index=d.index)).fillna(False)
        == True
    )
    npxg = float(d.loc[~(pen | pso), "xG"].fillna(0).sum()) if "xG" in d else 0.0
    deep = 0
    if "end_x" in d.columns:
        dp = d[isp & (o == "Successful") & ~is_cross]
        deep = int((dp["end_x"] >= 81).sum())
    sca = int(cr["sca"])

    # ── advanced defence: quality + aggression ──
    tkl_won = cnt_ok("Tackle")
    tkl_att = cnt("Tackle") + cnt("Challenge")  # Challenge = dribbled past
    tackle_pct = round(100 * tkl_won / tkl_att) if tkl_att else 0
    aer_won = cnt_ok("Aerial")
    duel_won = aer_won + tkl_won
    duel_att = cnt("Aerial") + tkl_att
    duel_pct = round(100 * duel_won / duel_att) if duel_att else 0
    high_reg = 0
    if "x" in d.columns:
        high_reg = int(
            (ty.isin(["Tackle", "Interception", "BallRecovery"]) & (d["x"] > 50)).sum()
        )

    # ── advanced passing ──
    prog_pct = round(100 * prog / pass_tot) if pass_tot else 0
    f3 = 0
    if {"x", "end_x"}.issubset(d.columns):
        f3 = int(
            (isp & (o == "Successful") & (d["x"] < 66.67) & (d["end_x"] >= 66.67)).sum()
        )
    qn = d.get("qualifier_names", pd.Series("", index=d.index)).astype(str)
    lb_mask = isp & qn.str.contains("Longball")
    lb_tot = int(lb_mask.sum())
    lb_comp = int((lb_mask & (o == "Successful")).sum())

    # ── xGOT: post-shot expected goals ──
    # This used to bucket the placement qualifier into four hand-picked
    # multipliers, so every shot in any corner was priced identically and a
    # shot the provider gave exact coordinates for was rounded into a zone.
    # match_metrics.post_shot_xg reads goal_mouth_y/goal_mouth_z — where the
    # ball actually crossed the line — and was written, tested and never
    # called. The radar and the team card now share it.
    try:
        from match_metrics import post_shot_xg

        xgot = round(float(post_shot_xg(d).sum()), 2)
    except Exception:
        xgot = 0.0

    key_passes = int(
        (
            d.get("is_key_pass", pd.Series(False, index=d.index)).fillna(False) == True
        ).sum()
    )

    # ── grid-model xT contribution (see compute_xt_grid) ──
    xt_contrib = round(_player_xt(ev, player), 3)

    # ── duels as absolute counts + win% (never percentage-only) ──
    # ground = tackle contests (as tackler) + take-on contests (as dribbler)
    g_won = tkl_won + cnt_ok("TakeOn")
    g_att = tkl_att + cnt("TakeOn")  # tackles+challenges + take-ons
    g_lost = max(g_att - g_won, 0)
    g_pct = round(100 * g_won / g_att) if g_att else 0
    a_att = cnt("Aerial")
    a_won = aer_won
    a_lost = max(a_att - a_won, 0)
    a_pct = round(100 * a_won / a_att) if a_att else 0
    t_att = g_att + a_att
    t_won = g_won + a_won
    t_lost = max(t_att - t_won, 0)
    t_pct = round(100 * t_won / t_att) if t_att else 0

    # ── defence: blocks / clearances ──
    blocks = cnt("BlockedPass")
    clearances = cnt("Clearance")
    tackles_won = tkl_won

    # ── discipline / reliability ──
    fouls_comm = int(((ty == "Foul") & (o == "Unsuccessful")).sum())  # committer
    cards = cnt("Card")
    dispossessed = cnt("Dispossessed")
    miscontrol = int(((ty == "BallTouch") & (o == "Unsuccessful")).sum())

    # ── carrying / dribbling ──
    carr = _get_carries(ev).get(str(player), {"carries": 0, "prog": 0})
    carries = int(carr["carries"])
    prog_carries = int(carr["prog"])
    dribbles = cnt_ok("TakeOn")

    # ── positional / tactical ──
    part = _get_participation(ev).get(str(player), {"minutes": 0})
    minutes = int(part.get("minutes", 0))
    onball = d[d["x"].notna()]
    avg_x = round(float(onball["x"].mean()), 1) if len(onball) else 0.0
    n_t = max(len(onball), 1)
    tch_def = int((onball["x"] < 33.33).sum())
    tch_mid = int(((onball["x"] >= 33.33) & (onball["x"] < 66.67)).sum())
    tch_att = int((onball["x"] >= 66.67).sum())
    tch_att_pct = round(100 * tch_att / n_t)

    return {
        # ── threat / attack ──
        "Goals": goals,
        "Shots": shots_tot,
        "Shots_ot": shots_ot,  # on-target, "ot/total" chip
        "xG": round(xg_tot, 2),
        "xGOT": xgot,
        "xT\ncontrib": xt_contrib,
        # ── creation ──
        "Key\npasses": key_passes,
        "Assists": asst,
        "xA": xa,
        "Big ch.\ncreated": bcc,
        # ── passing / progression ──
        "Touches": int(len(d)),
        "Passes": pass_tot,
        "Passes_comp": pass_comp,  # "comp/total" chip
        "Pass %": round(100 * pass_comp / max(pass_tot, 1)),
        "Prog\npasses": prog,
        "Final 3rd\npasses": f3,
        "Passes\nto box": ptobox,
        "Long\nballs": lb_tot,
        "Longballs_comp": lb_comp,  # "comp/total" chip
        # ── carrying / dribbling ──
        "Carries": carries,
        "Prog\ncarries": prog_carries,
        "Dribbles": dribbles,
        # ── defence ──
        "Tackles\nwon": tackles_won,
        "Intercep\ntions": cnt("Interception"),
        "Recov\neries": cnt("BallRecovery"),
        "Blocks": blocks,
        "Clear\nances": clearances,
        # ── duels (absolute counts, shown won/total) ──
        "Grd duels\nwon": g_won,
        "Grd_duels_att": g_att,
        "Grd_duels_lost": g_lost,
        "Grd_duels_pct": g_pct,
        "Aerials\nwon": a_won,
        "Aer_att": a_att,
        "Aer_lost": a_lost,
        "Aer_pct": a_pct,
        "Duels\nwon": t_won,
        "Duels_att": t_att,
        "Duels_lost": t_lost,
        "Duels_pct": t_pct,
        # ── discipline / reliability ──
        "Fouls": fouls_comm,
        "Cards": cards,
        "Dispos\nsessed": dispossessed,
        "Mis\ncontrol": miscontrol,
        # ── positional / tactical ──
        "Minutes": minutes,
        "Avg\nheight": avg_x,
        "Att 3rd\ntouch %": tch_att_pct,
        "Touch_def": tch_def,
        "Touch_mid": tch_mid,
        "Touch_att": tch_att,
        # ── retained extras (not plotted, used by commentary/rating) ──
        "npxG": round(npxg, 2),
        "xG/\nShot": round(xg_tot / shots_tot, 2) if shots_tot else 0.0,
        "G\N{MINUS SIGN}xG": round(goals - xg_tot, 2),
        "Shot-cr.\nactions": sca,
        "Deep\ncompl.": deep,
        "xG\nChain": round(sequence["xGChain"], 2),
        "xG\nBuildup": round(sequence["xGBuildup"], 2),
        "Sequence\nxT": round(sequence["sequence_xT"], 2),
        "Box\ntouches": tib,
    }


def compute_metrics_pool(events: pd.DataFrame):
    """Return (allm: {player: metrics}, elig: [players in percentile pool])."""
    part = _get_participation(events)
    tc = events.groupby("player").size()
    elig = [
        p
        for p in tc.index
        if tc[p] >= MIN_POOL_TOUCHES
        and _valid_name(p)
        and part.get(str(p), {}).get("status") != "unused"
    ]
    allm = {p: player_metrics(events, p) for p in elig}
    return allm, elig


def _percentile(allm, elig, metric, val):
    """Where this value sits among the players on the pitch, as a percentage.

    Ties take the mid-rank, which is the right answer between two real values
    and the wrong one at zero. Twenty-three of thirty players finished a match
    on 0.00 xGoT, so a zero scored 38% and drew a wedge two-fifths of the way
    out for a player who never had a shot on target; a zero on goal difference
    against expectation scored 62%, past the halfway ring. The chip beside the
    bar read 0.0 while the bar said otherwise.

    Not doing a thing is not doing it averagely, so a zero draws nothing. That
    holds for the signed metrics too: a goal difference against expectation of
    exactly zero is no deviation to show, and it drew the longest zero bar on
    the board for a full-back who never took a shot.
    """
    vs = [allm[p][metric] for p in elig]
    n = len(vs) or 1
    if val == 0:
        return 0.0
    below = sum(1 for v in vs if v < val)
    equal = sum(1 for v in vs if v == val)
    return (below + 0.5 * equal) / n * 100


def _player_role(events: pd.DataFrame, player: str, default="Player") -> str:
    if "player_role" not in events.columns:
        return default
    s = events[events["player"].astype(str) == str(player)]["player_role"].dropna()
    return str(s.iloc[0]) if len(s) else default


def make_player_pizza(
    events, player, team_name, role, allm, elig, subtitle_extra="", opponent_name="",
    team_color=None,
):
    """Build and return the pizza Figure for one player.

    ``team_color`` paints the whole radar in the player's team colour, one
    lightness step per metric group. Omit it to keep the legacy fixed palette.
    """
    me_m = player_metrics(events, player)

    group_colors = (
        team_group_colors(team_color, len(GROUPS))
        if team_color
        else [gc for _gn, gc, _ms in GROUPS]
    )

    labels, colors, vals, disps, pcts, gidx = [], [], [], [], [], []
    for gi, (_gn, _gc, ms) in enumerate(GROUPS):
        gc = group_colors[gi]
        for m in ms:
            v = me_m.get(m, 0)
            labels.append(m)
            colors.append(gc)
            vals.append(v)
            if m in _RATIO_DISPLAY:  # "numerator / denominator"
                num_k, den_k = _RATIO_DISPLAY[m]
                disps.append(f"{me_m.get(num_k, 0)}/{me_m.get(den_k, 0)}")
            else:
                disps.append(f"{v}")
            pcts.append(_percentile(allm, elig, m, v) if elig else 0)
            gidx.append(gi)

    N = len(labels)
    n_groups = len(GROUPS)
    GAP_DEG = 6.0
    # one equal gap per group boundary (internal boundaries + the wrap gap)
    per = (360 - GAP_DEG * n_groups) / N
    angs, cur, prev = [], 0.0, None
    for i in range(N):
        if prev is not None and gidx[i] != prev:
            cur += GAP_DEG
        angs.append(np.radians(cur + per / 2))
        cur += per
        prev = gidx[i]
    angs = np.array(angs)
    width = np.radians(per) * 0.90
    gap_mid = np.radians((cur + 360) / 2)

    R0, RMAX, RVAL, RLAB, RARC, OUT_LIM = 14, 100, 113, 130, 140, 154

    fig = plt.figure(figsize=(12, 12.6), facecolor=BG_DARK)
    ax = fig.add_axes([0.06, 0.035, 0.88, 0.735], projection="polar")
    ax.set_facecolor(BG_DARK)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, OUT_LIM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    def rmap(p):
        return R0 + (p / 100) * (RMAX - R0)

    for pv in (25, 50, 75, 100):
        ax.plot(
            np.linspace(0, 2 * np.pi, 200),
            [rmap(pv)] * 200,
            color=RADAR_GRID,
            lw=0.8,
            zorder=1,
        )

    # faint per-category background zones
    for gi, (_gn, _gc, _ms) in enumerate(GROUPS):
        gc = group_colors[gi]
        ids = [i for i in range(N) if gidx[i] == gi]
        a0 = angs[ids[0]] - width / 2 - np.radians(1.2)
        a1 = angs[ids[-1]] + width / 2 + np.radians(1.2)
        ax.bar(
            [(a0 + a1) / 2],
            [RMAX - R0],
            width=(a1 - a0),
            bottom=R0,
            color=gc,
            alpha=0.07,
            edgecolor="none",
            zorder=0.5,
        )

    ax.bar(
        angs,
        [rmap(p) - R0 for p in pcts],
        width=width,
        bottom=R0,
        color=colors,
        edgecolor=BG_DARK,
        lw=2.5,
        alpha=0.95,
        zorder=3,
    )

    # category arcs
    for gi, (_gn, _gc, _ms) in enumerate(GROUPS):
        gc = group_colors[gi]
        ids = [i for i in range(N) if gidx[i] == gi]
        a0 = angs[ids[0]] - width / 2 - np.radians(1.5)
        a1 = angs[ids[-1]] + width / 2 + np.radians(1.5)
        ax.plot(
            np.linspace(a0, a1, 60),
            [RARC] * 60,
            color=gc,
            lw=6,
            solid_capstyle="round",
            zorder=4,
            clip_on=False,
        )

    # metric labels + value chips (all horizontal). The bar keeps the group's
    # own colour; the chip behind the number is adjusted until the number on it
    # is legible — a tile carrying an 8pt digit and a wedge carrying a
    # percentile do not have the same job.
    chip_by_group = dict(zip(group_colors, chip_fills(group_colors)))
    for a, lab, dv, p, c in zip(angs, labels, disps, pcts, colors):
        chip = chip_by_group.get(c, _chip_fill(c))
        ax.text(
            a,
            RLAB,
            display_label(lab),
            color=TEXT_BRIGHT,
            fontsize=8.6,
            fontweight="bold",
            family="monospace",
            ha="center",
            va="center",
            zorder=5,
            linespacing=0.9,
            clip_on=False,
        )
        ax.text(
            a,
            RVAL,
            dv,
            color=_chip_text_color(chip),
            fontsize=9,
            fontweight="bold",
            family="monospace",
            ha="center",
            va="center",
            zorder=7,
            clip_on=False,
            # Saturated mid-lightness fills (a red team's middle ramp step)
            # sit where neither tier quite clears 4.5:1 on its own. The stroke
            # is a no-op everywhere else.
            path_effects=label_outline(chip, linewidth=1.4),
            # The fill carries the number's legibility, the border carries the
            # group: a chip pushed dark enough for white text sits almost on
            # the black page and stopped reading as a tile at all.
            bbox=dict(boxstyle="round,pad=0.24", fc=chip, ec=c, lw=1.3),
        )
    for sp in ax.spines.values():
        sp.set_visible(False)

    # clean centre ring
    ax.plot(
        np.linspace(0, 2 * np.pi, 120), [R0] * 120, color="#2e2e2e", lw=1.2, zorder=6
    )

    participation = _get_participation(events).get(str(player), {})
    played_time = participation.get("played_time", "0′ 00″")
    sub = f"{str(team_name).upper()}  ·  {role}  ·  {played_time} played"
    if subtitle_extra:
        sub += f"  ·  {subtitle_extra}"
    if _identity is not None:
        _identity.amoled_header(
            fig, str(player).upper(), sub, section="PLAYER PIZZA",
            active_team=team_name,
        )

    ng = len(GROUPS)
    step = min(0.135, 0.90 / max(ng - 1, 1))
    lfs = 11 if ng <= 5 else 9
    lx = 0.5 - (ng - 1) * step / 2 - 0.02  # centre the legend row
    for gi, (gn, _gc, _ms) in enumerate(GROUPS):
        gc = group_colors[gi]
        fig.add_artist(
            mpatches.Circle(
                (lx, 0.822), 0.006, transform=fig.transFigure, facecolor=gc, ec="none"
            )
        )
        fig.text(
            lx + 0.012,
            0.822,
            gn,
            # The dot carries the exact group shade; the label is lifted until
            # it reads on the page, because the darkest step of a ramp is too
            # dim as text even though it is fine as a fill.
            color=_readable_on_page(gc),
            fontsize=lfs,
            fontweight="bold",
            family="monospace",
            va="center",
        )
        lx += step
    fig.text(
        0.5,
        0.022,
        "bar length = percentile vs all match players   ·   chip = match value "
        "(passes/long balls = completed/total · shots = on-target/total · duels = won/contested)",
        ha="center",
        color="#555",
        fontsize=9,
        style="italic",
    )
    return fig


def player_commentary(events, player, team_name, opp_name, allm, elig, role="Player"):
    """Professional, data-driven tactical read of one player's match, built from
    raw output and where each number ranks among all players on the pitch.
    Returns two paragraphs separated by a blank line."""
    m = allm.get(player) or player_metrics(events, player)

    def pc(label):
        return _percentile(allm, elig, label, m.get(label, 0)) if elig else 0.0

    def lvl(p):
        return (
            "elite"
            if p >= 85
            else (
                "strong"
                if p >= 70
                else "solid" if p >= 50 else "modest" if p >= 30 else "limited"
            )
        )

    goals = m.get("Goals", 0)
    shots = m.get("Shots", 0)
    ot = m.get("Shots_ot", 0)
    xg = m.get("xG", 0.0)
    gmx = m.get("G\N{MINUS SIGN}xG", 0.0)
    xa = m.get("xA", 0.0)
    sca = m.get("Shot-cr.\nactions", 0)
    assists = m.get("Assists", 0)
    kp = m.get("Key\npasses", 0)
    bcc = m.get("Big ch.\ncreated", 0)
    passes = m.get("Passes", 0)
    pcomp = m.get("Passes_comp", 0)
    ppct = m.get("Pass %", 0)
    prog = m.get("Prog\npasses", 0)
    ptb = m.get("Passes\nto box", 0)
    touches = m.get("Touches", 0)
    box_t = m.get("Box\ntouches", 0)
    drib = m.get("Dribbles", 0)
    carries = m.get("Carries", 0)
    prog_car = m.get("Prog\ncarries", 0)
    tkl = m.get("Tackles\nwon", 0)
    intc = m.get("Intercep\ntions", 0)
    rec = m.get("Recov\neries", 0)
    clr = m.get("Clear\nances", 0)
    blocks = m.get("Blocks", 0)
    duels = m.get("Duels\nwon", 0)
    duels_att = m.get("Duels_att", 0)
    minutes = m.get("Minutes", 0)
    xt = m.get("xT\ncontrib", 0.0)

    # ── Paragraph 1: headline role + attacking / creation ──
    s1 = []
    # standout strengths: top percentile metrics (skip zero raw & non-achievement)
    _skip = {"Minutes", "Fouls", "Dispos\nsessed", "Pass %"}
    ranked = sorted(
        (
            (lab, pc(lab))
            for _gn, _gc, ms in GROUPS
            for lab in ms
            if m.get(lab, 0) and lab not in _skip
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    tops = [lab.replace("\n", " ").lower() for lab, p in ranked[:3] if p >= 60]
    mins_txt = f" across {minutes} minutes" if minutes else ""
    lead = (
        f"Operating as {('a ' + role.lower()) if role and role.lower() not in ('player','') else 'an outfield option'} "
        f"for {team_name} against {opp_name}{mins_txt}, "
    )
    if tops:
        lead += (
            f"{_surname(player)} ranked among the match's best for {', '.join(tops)}."
        )
    else:
        lead += f"{_surname(player)} operated in a supporting role by the underlying numbers."
    s1.append(lead)

    if shots:
        fin = (
            "clinical, beating his expected return"
            if gmx > 0.15
            else (
                "wasteful relative to the chances"
                if gmx < -0.25
                else "in line with the chance quality"
            )
        )
        s1.append(
            f"He took {shots} shot{'s' if shots != 1 else ''} ({ot} on target) worth {xg:.2f} xG"
            + (f" and scored {goals}" if goals else " without scoring")
            + f" — finishing that reads as {fin} (G−xG {gmx:+.2f})."
        )
    if xa > 0 or sca or assists or bcc:
        parts = []
        if sca:
            parts.append(f"{sca} shot-creating action{'s' if sca != 1 else ''}")
        if xa > 0:
            parts.append(f"{xa:.2f} xA")
        if bcc:
            parts.append(f"{bcc} big chance{'s' if bcc != 1 else ''} created")
        line = "As a creator he generated " + ", ".join(parts) + "."
        if assists:
            line += (
                f" That converted into {assists} assist{'s' if assists != 1 else ''}."
            )
        s1.append(line)

    # ── Paragraph 2: possession involvement + defence + verdict ──
    s2 = []
    pp = pc("Passes")
    if passes:
        s2.append(
            f"On the ball he had {touches} touches and completed {pcomp}/{passes} passes ({ppct}%), "
            f"a {lvl(pp)} volume for the game, moving it forward with {prog} progressive pass{'es' if prog != 1 else ''} "
            f"and {ptb} into the box."
        )
    if box_t or drib or carries:
        s2.append(
            f"He arrived in the penalty area {box_t} time{'s' if box_t != 1 else ''}, "
            f"carried the ball {carries} time{'s' if carries != 1 else ''} ({prog_car} progressively)"
            + (
                f" and completed {drib} dribble{'s' if drib != 1 else ''}"
                if drib
                else ""
            )
            + "."
        )
    defw = tkl + intc + rec + clr + blocks
    dp = (pc("Tackles\nwon") + pc("Intercep\ntions") + pc("Recov\neries")) / 3
    if defw:
        s2.append(
            f"Without the ball he won {tkl} tackle{'s' if tkl != 1 else ''}, {intc} interception"
            f"{'s' if intc != 1 else ''}, {rec} recover{'ies' if rec != 1 else 'y'}, {blocks} block"
            f"{'s' if blocks != 1 else ''} and {clr} clearance{'s' if clr != 1 else ''} "
            f"({duels}/{duels_att} duels won) — a {lvl(dp)} defensive shift."
        )
    # closing verdict tied to threat
    threat_p = (pc("xA") + pc("npxG") + pc("xT\ncontrib")) / 3
    if threat_p >= 70:
        s2.append(
            f"Overall, one of {team_name}'s primary threat carriers on the day (xT {xt:.2f})."
        )
    elif threat_p >= 40:
        s2.append(
            f"A useful contributor to {team_name}'s attacking phases without being the focal point."
        )
    else:
        s2.append(
            f"His influence was felt more in structure and workload than in direct threat generation."
        )

    return " ".join(s1) + "\n\n" + " ".join(s2)


def _team_split(events, info):
    """Return {'home': (name, [players]), 'away': (name, [players])}."""
    hid = info.get("home_id")
    aid = info.get("away_id")
    hn = _text(info.get("home_name"), "Home")
    an = _text(info.get("away_name"), "Away")
    # dominant team_id per player (real player names, participants only)
    part = _get_participation(events)
    ev = events[events["player"].map(_valid_name)]
    pt = ev.groupby("player")["team_id"].agg(lambda s: s.value_counts().index[0])

    def _played(p):
        return part.get(str(p), {}).get("status") != "unused"

    home_players = [p for p, t in pt.items() if t == hid and _played(p)]
    away_players = [p for p, t in pt.items() if t == aid and _played(p)]
    return {"home": (hn, home_players), "away": (an, away_players)}


# positional / negative metrics that must not reward a player in the ranking
_RATING_SKIP = {"Minutes", "Fouls", "Dispos\nsessed"}


def _rating(allm, elig, events, player):
    """Mean percentile across performance metrics — used to rank players."""
    m = allm.get(player) or player_metrics(events, player)
    if not elig:
        return 0.0
    ps = [
        _percentile(allm, elig, lab, m.get(lab, 0))
        for _gn, _gc, ms in GROUPS
        for lab in ms
        if lab not in _RATING_SKIP
    ]
    return float(np.mean(ps)) if ps else 0.0


def _side_team_color(info, side: str) -> str:
    """Resolve the fixture colour for one side, so a player's radar carries
    their own team's colour rather than a fixed category palette."""
    key = "home_color" if side == "home" else "away_color"
    supplied = str((info or {}).get(key) or "").strip()
    if supplied:
        try:
            mcolors.to_rgb(supplied)
            return supplied
        except ValueError:
            pass
    return C_HOME if side == "home" else C_AWAY


def export_player_radars(events, info, out_dir, dpi=115):
    """Save one pizza PNG per participating player into per-team folders.

    Returns {"home": [(player, rating), ...], "away": [...]} best-first.
    """
    allm, elig = compute_metrics_pool(events)
    split = _team_split(events, info)
    base = os.path.join(out_dir, "player_radars")
    ranking = {"home": [], "away": []}

    for side in ("home", "away"):
        team_name, players = split[side]
        team_dir = os.path.join(base, _safe(team_name))
        os.makedirs(team_dir, exist_ok=True)
        scored = []
        for p in players:
            role = _player_role(events, p)
            try:
                opponent = info.get("away_name") if side == "home" else info.get("home_name")
                fig = make_player_pizza(
                    events, p, team_name, role, allm, elig,
                    opponent_name=str(opponent or ""),
                    team_color=_side_team_color(info, side),
                )
                fig.savefig(
                    os.path.join(team_dir, f"{_safe(p)}.png"),
                    dpi=dpi,
                    facecolor=BG_DARK,
                )
                plt.close(fig)
            except Exception:
                plt.close("all")
            scored.append((p, _rating(allm, elig, events, p)))
        ranking[side] = sorted(scored, key=lambda x: x[1], reverse=True)
    return ranking


def build_report_radars(events, info, out_dir, top_n=5, dpi=115):
    """Save every player's PNG into per-team folders AND return open figures
    for the top-N players of each team, for embedding in the PDF report.

    Returns {"home": {"name": str, "figs": [(player, fig, role, commentary), ...]},
             "away": {...}}  with figs best-first.
    Caller owns the returned figures and must close them.
    """
    allm, elig = compute_metrics_pool(events)
    split = _team_split(events, info)
    opp = {
        "home": _text(info.get("away_name"), "the opponent"),
        "away": _text(info.get("home_name"), "the opponent"),
    }
    base = os.path.join(out_dir, "player_radars")
    result = {}

    for side in ("home", "away"):
        team_name, players = split[side]
        team_dir = os.path.join(base, _safe(team_name))
        os.makedirs(team_dir, exist_ok=True)
        scored = []
        for p in players:
            role = _player_role(events, p)
            try:
                opponent = info.get("away_name") if side == "home" else info.get("home_name")
                fig = make_player_pizza(
                    events, p, team_name, role, allm, elig,
                    opponent_name=str(opponent or ""),
                    team_color=_side_team_color(info, side),
                )
                fig.savefig(
                    os.path.join(team_dir, f"{_safe(p)}.png"),
                    dpi=dpi,
                    facecolor=BG_DARK,
                )
                plt.close(fig)
            except Exception:
                plt.close("all")
            scored.append((p, _rating(allm, elig, events, p)))
        scored.sort(key=lambda x: x[1], reverse=True)
        figs = []
        for rank, (p, rt) in enumerate(scored[:top_n], start=1):
            role = _player_role(events, p)
            try:
                f = make_player_pizza(
                    events,
                    p,
                    team_name,
                    role,
                    allm,
                    elig,
                    subtitle_extra=f"Team rank #{rank}",
                    opponent_name=str(opp[side]),
                    team_color=_side_team_color(info, side),
                )
                try:
                    note = player_commentary(
                        events, p, team_name, opp[side], allm, elig, role
                    )
                except Exception:
                    note = ""
                figs.append((p, f, role, note))
            except Exception:
                plt.close("all")
        result[side] = {"name": team_name, "figs": figs}
    return result


def top_players_per_team(events, info, n=5):
    allm, elig = compute_metrics_pool(events)
    split = _team_split(events, info)
    out = {}
    for side in ("home", "away"):
        _name, players = split[side]
        scored = sorted(
            ((p, _rating(allm, elig, events, p)) for p in players),
            key=lambda x: x[1],
            reverse=True,
        )
        out[side] = [p for p, _r in scored[:n]]
    return out
