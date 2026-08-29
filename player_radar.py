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
from pathlib import Path

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
    USE_REAL_TEAM_KIT_COLORS,
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


# One hue per metric group, in the order GROUPS declares them, chosen so the
# colour says something about the actions underneath it: warm for the shot,
# blue for the pass, rose for the danger it creates, green for the work without
# the ball, violet for the contest.
#
# The team's own colour used to drive all five as a single ramp. That kept the
# radar in the club's identity and cost it everything else: five steps of one
# hue put ATTACK next to THREAT and DEFENCE next to DUELS in shades a reader
# had to compare side by side to separate, and a saturated kit made the whole
# page shout in one tone. Identity is carried by the crest, the score and the
# rule under the header — the wedges are better spent saying which group they
# belong to.
GROUP_HUES = (
    36.0,    # ATTACK   — amber
    212.0,   # PASSING  — blue
    338.0,   # THREAT   — rose
    162.0,   # DEFENCE  — green
    276.0,   # DUELS    — violet
)

# Saturation is held constant across the five hues — that is what makes them
# read as one family — but it used to be held low. At S=0.46 and L=0.62 the
# wedges were pastel: correct, harmonised and washed out on a black page that
# can carry far more colour than that.
GROUP_LIGHTNESS = 0.42 if IS_LIGHT_THEME else 0.62
GROUP_SATURATION = 0.68 if IS_LIGHT_THEME else 0.72


# The contrast every group shade is solved *to*, not merely held above.
#
# Equal HLS lightness is not equal weight: at a fixed L=0.62 and S=0.46 the
# green measured 10.9:1 against the black page and the violet 5.9:1, so five
# colours built to be siblings arrived with one nearly twice the weight of
# another. The old solve was a floor at 4.6 and every hue already cleared it,
# so it never moved anything — the harmonisation was in the comment rather
# than in the palette.
#
# Solving to a target instead equalises what the eye actually reads, and it is
# what lets the saturation go up: a hue that would shout is darkened until it
# weighs the same as its siblings rather than being kept quiet from the start.
GROUP_PAGE_CONTRAST = 4.6 if IS_LIGHT_THEME else 7.0


def group_palette(n_groups: int) -> list[str]:
    """One colour per metric group, harmonised by construction.

    Saturation is held constant and the hues are fixed; lightness is solved per
    hue so all five weigh the same against the page. Two colours built that way
    cannot clash, and none of them can dominate the others.
    """
    import colorsys

    hues = list(GROUP_HUES)
    while len(hues) < n_groups:            # more groups than hues: keep spacing
        hues.append((hues[-1] + 360.0 / max(n_groups, 1)) % 360.0)

    return [_hue_at_page_contrast(hue) for hue in hues[:n_groups]]


def _hue_at_page_contrast(hue: float, saturation: float | None = None) -> str:
    """One hue, set to the lightness that lands it on the target contrast.

    Bisected rather than stepped, and to a target rather than a floor. The
    stepped version stopped at the first level that cleared the minimum, which
    for every hue in this palette was the level it started at — so a hue that
    happened to read heavy stayed heavy and the five were never equalised.

    Contrast is monotonic in lightness on either side of the page's own value,
    so the search runs from the page's end of the range outwards and takes the
    first crossing.
    """
    import colorsys

    if saturation is None:
        saturation = GROUP_SATURATION

    def at(level: float) -> str:
        return mcolors.to_hex(colorsys.hls_to_rgb(hue / 360.0, level, saturation))

    # The band spans both sides of the nominal lightness. Starting it at
    # GROUP_LIGHTNESS meant a hue could only ever be moved away from the page,
    # so the two that already read heavy — amber at 10.3:1 and green at 13.2 —
    # had nowhere to go and stayed twice the weight of the violet. Equalising
    # needs the freedom to darken as well as to lighten.
    low, high = (0.14, 0.62) if IS_LIGHT_THEME else (0.34, 0.97)
    try:
        # Nothing in this range reaches the target: take the end that is
        # furthest from the page rather than failing.
        best_end = low if IS_LIGHT_THEME else high
        if contrast_ratio(at(best_end), BG_DARK) < GROUP_PAGE_CONTRAST:
            return at(best_end)
        # Contrast rises as the ink moves away from the page, so the wanted
        # level is the *nearest* one that reaches the target: any further and
        # the colour is paler than it needs to be, which is the washed-out
        # palette this replaces.
        for _ in range(30):
            mid = (low + high) / 2
            reached = contrast_ratio(at(mid), BG_DARK) >= GROUP_PAGE_CONTRAST
            if IS_LIGHT_THEME:
                # Darker is higher contrast; keep the lightest that reaches it.
                low, high = (mid, high) if reached else (low, mid)
            else:
                # Lighter is higher contrast; keep the deepest that reaches it.
                low, high = (low, mid) if reached else (mid, high)
        return at(low if IS_LIGHT_THEME else high)
    except Exception:
        return at(GROUP_LIGHTNESS)


# ── One fixture, two palettes ────────────────────────────────────────────────
#
# Both sides drew from the same five colours, so nothing on a radar said which
# team the player belonged to except the crest in the header. Two pages side by
# side were indistinguishable.
#
# Each side's five hues are now rotated a little way towards its own kit, which
# gives one page a warm cast and the other a cool one while leaving the five
# groups in the same order, the same spacing and the same meaning. The reader
# learns the key once.
#
# The whole wheel turns by one angle per side, rather than each hue being
# pulled towards the kit on its own. Pulling individually collapses the wheel —
# and it also fails at the thing it is for: a hue sitting opposite both kits
# reaches the cap from both, so Hull's defence came out #6dcf8b and Manchester
# United's #6ece8b, the same colour by two routes. Turning the wheel keeps the
# five groups exactly as far apart as they were and moves both sides' palettes
# by a difference that can be guaranteed.
#
# Mixing in RGB was the other candidate and is worse still: 35% of amber into a
# blue gives an olive grey.
TEAM_TINT_SHARE = 0.5      # fraction of the way from the anchor hue to the kit
TEAM_TINT_MAX = 34.0       # ...but never further than this, in degrees
TEAM_TINT_SATURATION = 0.10  # how much of the kit's saturation is carried over

# How far apart the two sides' wheels are held. Below this the two pages read
# as the same palette with a slightly different mood, which is not a difference
# a reader glancing at two radars would notice.
TEAM_OFFSET_MIN_GAP = 40.0

# Two clubs can wear the same colour. Hull's amber sits 31° from Manchester
# United's red, so tinting each towards its own kit would have produced two
# palettes a reader could not tell apart — the exact problem this is for. When
# the kits are closer than this the two casts are pushed apart symmetrically,
# so the fixture always yields two distinguishable pages even in a derby.
TEAM_CAST_MIN_GAP = 70.0


def _hue_of(color: str, fallback: float = 36.0) -> float:
    import colorsys

    try:
        r, g, b = mcolors.to_rgb(color)
    except (ValueError, TypeError):
        return fallback
    hue, _lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    # A near-grey kit has no hue to speak of; rotating towards its arbitrary
    # value would tint the page at random, so it stays where it is.
    return fallback if saturation < 0.08 else hue * 360.0


def _saturation_of(color: str, fallback: float = 0.5) -> float:
    import colorsys

    try:
        r, g, b = mcolors.to_rgb(color)
    except (ValueError, TypeError):
        return fallback
    _hue, _lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    return saturation


def _signed_gap(a: float, b: float) -> float:
    """Shortest signed rotation from a to b, in degrees."""
    return (b - a + 180.0) % 360.0 - 180.0


def cast_hues(home_color: str, away_color: str) -> tuple[float, float]:
    """The hue each side's palette leans towards, held apart from each other."""
    home = _hue_of(home_color, 36.0)
    away = _hue_of(away_color, 212.0)
    gap = _signed_gap(home, away)
    if abs(gap) >= TEAM_CAST_MIN_GAP:
        return home, away
    # Push both, not one, so neither side is the one that gets moved off its
    # own kit while the other keeps it.
    push = (TEAM_CAST_MIN_GAP - abs(gap)) / 2.0
    direction = 1.0 if gap >= 0 else -1.0
    return (home - direction * push) % 360.0, (away + direction * push) % 360.0


def fixture_hue_offsets(home_color: str, away_color: str) -> tuple[float, float]:
    """How far each side's wheel turns, guaranteed to differ.

    The turn is read off the first group's hue against the kit, so a warm kit
    turns the wheel warm and a cool one turns it cool. Two kits close enough to
    produce nearly the same turn are pushed apart, because two palettes a
    reader cannot separate is the failure this exists to prevent.
    """
    home_cast, away_cast = cast_hues(home_color, away_color)
    anchor = GROUP_HUES[0]

    def turn(cast: float) -> float:
        return float(np.clip(_signed_gap(anchor, cast) * TEAM_TINT_SHARE,
                             -TEAM_TINT_MAX, TEAM_TINT_MAX))

    home, away = turn(home_cast), turn(away_cast)
    gap = away - home
    if abs(gap) < TEAM_OFFSET_MIN_GAP:
        push = (TEAM_OFFSET_MIN_GAP - abs(gap)) / 2.0
        direction = 1.0 if gap >= 0 else -1.0
        home, away = home - direction * push, away + direction * push
    return home, away


def group_palette_for(offset: float, kit_color: str,
                      n_groups: int) -> list[str]:
    """The five group colours with one side's turn applied to all of them."""
    hues = list(GROUP_HUES)
    while len(hues) < n_groups:
        hues.append((hues[-1] + 360.0 / max(n_groups, 1)) % 360.0)

    # A saturated kit lifts the whole page a little, which is as much of the
    # club's own colour as the wedges should carry: they are spent saying which
    # group an action belongs to.
    kit_saturation = _saturation_of(kit_color, GROUP_SATURATION)
    saturation = float(np.clip(
        GROUP_SATURATION * (1 - TEAM_TINT_SATURATION)
        + kit_saturation * TEAM_TINT_SATURATION, 0.30, 0.70))

    return [_hue_at_page_contrast((hue + offset) % 360.0, saturation)
            for hue in hues[:n_groups]]


def fixture_group_palettes(home_color: str, away_color: str,
                           n_groups: int) -> dict[str, list[str]]:
    """Both sides' palettes, built together so they cannot collide."""
    home_offset, away_offset = fixture_hue_offsets(home_color, away_color)
    return {
        "home": group_palette_for(home_offset, home_color, n_groups),
        "away": group_palette_for(away_offset, away_color, n_groups),
    }


def team_group_colors(team_color: str, n_groups: int) -> list[str]:
    """Return one shade per metric group, all drawn from the team's own colour.

    Kept for callers that want the single-hue ramp. The radar itself now uses
    ``group_palette``: a chart whose five groups are five steps of one colour
    asks the reader to compare shades to tell a tackle from a through ball.

    Kits with almost no saturation (white/silver sides) fall back to a grey
    ramp of the same shape, which is the honest rendering of a white shirt.
    """
    import colorsys

    try:
        r, g, b = mcolors.to_rgb(team_color)
    except (ValueError, TypeError):
        r, g, b = mcolors.to_rgb(C_HOME)
    hue, _lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    # A kit colour is chosen to be seen from the back of a stand. Thirty wedges
    # of it, at full saturation, is a page that shouts in one tone and gives the
    # eye nowhere to rest, so the ramp is drawn at a calmer saturation than the
    # shirt. The hue still identifies the team; the volume comes down.
    saturation = min(max(saturation, 0.06) * 0.72, 0.62)

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

    # Lightness alone left DEFENCE and DUELS looking like one group: on a five
    # step ramp the last two steps are close, and they sit next to each other on
    # the ring. Nudging the hue a few degrees per step separates the neighbours
    # without letting the radar stop reading as one team.
    spread = [(-1.4, 0.0, 1.4, -0.7, 0.7)[i % 5] for i in range(len(ordered))]
    return [
        mcolors.to_hex(colorsys.hls_to_rgb(
            (hue + shift / 360.0) % 1.0, level, saturation))
        for level, shift in zip(ordered[:n_groups], spread)
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
        # Ten slices, six of them the same shot sample asked six ways. A player
        # who took two shots filled xG, npxG, xGOT, xG/Shot and G−xG with one
        # afternoon's worth of information, and the ring read as a busy profile
        # rather than as two shots.
        #
        # Four are kept and they answer different questions: what the chances
        # were worth (xG), what he did to them (G−xG), what he created for
        # someone else (xA), and what the possessions he touched were worth
        # (xGChain). npxG duplicates xG on any player who took no penalty;
        # xGOT duplicates it for anyone who hit the target; xG/Shot is xG over
        # a count already on the ring; xGBuildup is xGChain minus the last two
        # touches, and the pair moved together on every radar.
        "THREAT",
        C_HOME,
        [
            "xG",
            "xA",
            "G\N{MINUS SIGN}xG",
            "xG\nChain",
            "Shot-cr.\nactions",
            "Deep\ncompl.",
        ],
    ),
    (
        "DEFENCE",
        C_AWAY,
        ["Tackles\nwon", "Intercep\ntions", "Recov\neries", "Blocks", "Clear\nances"],
    ),
    ("DUELS", C_AWAY, ["Grd duels\nwon", "Aerials\nwon", "Duels\nwon"]),
]


# ── The goalkeeper's own radar ───────────────────────────────────────────────
#
# A keeper was drawn on the outfield layout: goals, dribbles, expected goals,
# aerial duels. Twenty-two of his thirty slices were structurally zero and not
# one of them described his match.
#
# Sixteen slices, all of them things a goalkeeper does. Post-shot expected
# goals is absent on purpose — see goalkeeper_metrics for why.
GK_GROUPS = [
    ("SHOT STOPPING", C_GOLD,
     ["Saves", "Save %", "Shots\nfaced", "Goals\nconceded", "Penalties\nfaced"]),
    ("BOX COMMAND", C_HOME,
     ["Claims", "Punches", "Pickups", "Smothers"]),
    ("OFF THE LINE", C_AWAY,
     ["Sweeps", "Recov\neries", "Clear\nances", "Errors"]),
    ("DISTRIBUTION", C_HOME,
     ["Passes", "Pass %", "Long\nballs", "Long ball %"]),
]


# What counts as a full bar, per action, measured from every goalkeeper match
# this project has rendered — thirty-eight of them — at the ninetieth
# percentile. There are two keepers in a match, so a percentile against the
# other one is either 0% or 100%; a reference drawn from real keeper matches
# is the honest alternative to a pool of two.
GK_FULL_BAR = {
    "Saves": 5, "Save %": 100, "Shots\nfaced": 8,
    "Claims": 2, "Punches": 2, "Pickups": 9, "Smothers": 1,
    "Sweeps": 2, "Recov\neries": 3, "Clear\nances": 2,
    "Passes": 51, "Pass %": 88, "Long\nballs": 12, "Long ball %": 64,
}

# More of these is worse, so they carry the figure and no bar. A long wedge
# beside GOALS CONCEDED reads as an achievement whatever the label says.
GK_NO_BAR = ("Goals\nconceded", "Errors", "Penalties\nfaced")


def gk_bar(metric: str, value) -> float:
    """How far a goalkeeper's wedge reaches, as a percentage of a full one.

    Not a percentile. There are two keepers in a match, so ranking one against
    the other returns 0 or 100 and says nothing; the scale is a strong keeper
    match for that action, measured from every fixture this project has
    rendered rather than chosen.

    The metrics where more is worse draw nothing, because a long wedge beside
    GOALS CONCEDED reads as an achievement whatever the label says.
    """
    if metric in GK_NO_BAR:
        return 0.0
    full = GK_FULL_BAR.get(metric)
    if not full:
        return 0.0
    try:
        share = float(value) / float(full)
    except (TypeError, ValueError):
        return 0.0
    return float(min(max(share, 0.0), 1.0) * 100)


# Three metric keys were written with the line break inside the word so they
# would fit the ring — they rendered as CLEAR/ANCES, RECOV/ERIES and
# INTERCEP/TIONS, which reads as a typesetting fault rather than a label. The
# keys are load-bearing (they index the metric dictionaries and the percentile
# lookups), so the repair happens at draw time: same key, readable label.
#
# Three more were added when the labels stopped wrapping. On one line, the
# longest names reach back from the arc far enough to sit on the value ring:
# "BIG CH. CREATED", "FINAL 3RD PASSES" and "SHOT-CR. ACTIONS" all ran into
# their own numbers. Each has a shorter form that is a real phrase rather than
# a truncation, which is a better label anyway — "BIG CH. CREATED" was already
# an abbreviation nobody says out loud.
_LABEL_OVERRIDES = {
    "Clear\nances": "CLEARANCES",
    "Recov\neries": "RECOVERIES",
    "Intercep\ntions": "INTERCEPTS",
    "Big ch.\ncreated": "BIG CHANCES",
    "Final 3rd\npasses": "FINAL THIRD",
    "Shot-cr.\nactions": "SHOT-CREATING",
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


# How far a chip tile stands off a light page.
#
# On black the tile takes care of itself: a group colour deepened until its
# figure is legible is still plainly a tile, and the quiet tier lands at 2.87
# against the page. On white the same rule collapsed. The light palette is built
# darker than the dark one — L=0.42 against 0.62 — so no group cleared the text
# floor as it stood and every one of them drove to the near-black end: five
# black blobs punched into a white page, and the quiet tier, which is the loud
# fill mixed 42% back toward the page, came out grey. A low figure read as
# disabled rather than as quiet.
#
# Driving to the pale end instead fixed the ink and lost the tile: the search
# stops at the first level whose figure is legible, and paling a colour raises
# that fast, so DEFENCE settled at 1.59 against the page and DUELS at 1.17 —
# a number floating on nothing.
#
# So the tile is solved to a target the way the ring's own hues are, and the
# target is the most presence the legibility floor will allow. At 2.4 the
# darkest ink on every group clears CHIP_CONTRAST_FLOOR with 7.5 to spare; at
# 2.8 it does not clear it at all.
CHIP_PAGE_CONTRAST = 2.4


def _chip_at_page_contrast(hue: float, saturation: float, floor: float,
                           nudge: float) -> str:
    """The lightest tile of this hue that still reads as a tile on the page.

    Lightest, because contrast against a white page rises as the tile darkens
    and the figure's own contrast falls with it: the lightest level that clears
    CHIP_PAGE_CONTRAST is the one that leaves the most room for the ink.
    """
    def at(level: float) -> str:
        return mcolors.to_hex(colorsys.hls_to_rgb(hue, level, saturation))

    def legible(level: float) -> bool:
        tile = at(level)
        return contrast_ratio(text_on_fill(tile), tile) >= floor

    low, high = 0.0, 1.0
    for _ in range(30):
        mid = (low + high) / 2
        if contrast_ratio(at(mid), BG_DARK) >= CHIP_PAGE_CONTRAST:
            low = mid
        else:
            high = mid

    if not legible(low):
        # This hue cannot carry a figure at the level the page wants. Deepen
        # until it can and take the loss in tile weight, which is the old
        # behaviour and the right one for the few hues that need it.
        level = low
        for _ in range(40):
            level = max(level - 0.02, 0.0)
            if legible(level):
                break
        return at(level)

    # A nudge deepens, which lifts the tile further off the page and keeps two
    # groups that landed together apart. It stops at the point the figure would
    # stop reading — unclamped it walked straight through the band where
    # neither black nor white ink clears the floor and came out at the dark end,
    # which is the black-blob tile this function exists to avoid.
    level = low
    for _ in range(40):
        step = max(low - nudge, 0.06)
        step = level - (level - step) / 2 if level > step else level
        if step >= level - 1e-4 or not legible(step):
            break
        level = step
    return at(level)


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
    if IS_LIGHT_THEME:
        return _chip_at_page_contrast(hue, saturation, floor, nudge)
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


# Above this percentile a value gets a filled tile; below the second figure it
# gets an outline only. The middle band keeps a fill at reduced strength.
CHIP_LOUD = 75.0
CHIP_QUIET = 35.0


def _is_zero(value, displayed: str) -> bool:
    """Did the player not do this at all?

    A ratio chip reads "won/contested", so 0/6 is a zero even though the string
    carries a six. Six duels entered and none won is still nothing on the board
    the wedge is measuring.
    """
    numerator = str(displayed).split("/")[0].strip()
    try:
        # The printed figure, not the stored one: 0.004 xA is rounded to "0.0"
        # on the tile, and a tile reading zero should be as quiet as the value
        # it shows. A negative number is not a zero and keeps its outline.
        if float(numerator) != 0:
            return False
    except ValueError:
        return True
    try:
        return float(value) >= 0
    except (TypeError, ValueError):
        return True


# One shape for every tile on the ring. Five treatments used to share the
# space — solid fill, soft fill, plain outline, dashed outline, and a fifth
# padding value — each with its own padding, its own border width and its own
# text weight, so a quarter of the circle could carry four different objects.
# Around a ring that reads as scatter rather than as a scale.
#
# The shape, the padding and the border are now fixed. Only the fill changes,
# which is the one thing that has to: a tile the player led on should not look
# like a tile he did nothing on.
CHIP_PAD = 0.30
CHIP_EDGE = 1.0


def _chip_style(chip: str, group: str, percentile: float, zero: bool,
                unmeasured: bool = False) -> dict:
    """How loudly one value should be printed.

    Four states, one shape. ``unmeasured`` is a rate whose denominator is too
    small to rank — 100% from three passes is a true statement about the match
    and a false one about the player — so it keeps the figure, loses the wedge,
    and is the only state that changes the border rather than the fill.
    """
    def tile(fill, text, edge, weight="bold", dashed=False, effects=None):
        box = dict(boxstyle=f"round,pad={CHIP_PAD}", fc=fill, ec=edge,
                   lw=CHIP_EDGE)
        if dashed:
            box["linestyle"] = (0, (2.5, 1.6))
        return {"color": text, "weight": weight,
                "effects": effects or [], "bbox": box}

    # A zero and an unmeasurable rate are both quiet, and quiet was being done
    # with TEXT_DIM — grey on grey, a figure a reader had to hunt for. The
    # digit is now the same ink as every other figure in its group; the empty
    # box around it is what says the player did not do this, which is a job for
    # the shape rather than for the contrast.
    quiet_ink = _readable_on_page(group, 4.5)
    if unmeasured:
        return tile("none", quiet_ink, _mix(group, BG_DARK, 0.30),
                    weight="normal", dashed=True)
    if zero:
        # Present for anyone who looks, silent for anyone who does not.
        return tile("none", quiet_ink, _mix(group, BG_DARK, 0.55),
                    weight="normal")
    if percentile >= CHIP_LOUD:
        return tile(chip, _chip_text_color(chip), _mix(chip, BG_DARK, 0.35),
                    effects=label_outline(chip, linewidth=1.4))
    if percentile >= CHIP_QUIET:
        soft = _mix(chip, BG_DARK, 0.42)
        return tile(soft, _chip_text_color(soft), _mix(soft, BG_DARK, 0.35),
                    effects=label_outline(soft, linewidth=1.2))
    return tile("none", _readable_on_page(group, 4.5),
                _mix(group, BG_DARK, 0.45), weight="normal")


def pad_values(values: list[str]) -> list[str]:
    """Every tile the same width, so the ring reads as a scale.

    The figures run from one character to seven — "5" against "27 / 58" — and
    a rounded box drawn around each one made the ring a row of unequal blobs.
    The font is monospace, so padding to a common count makes every box the
    same size without moving a single digit off its own spoke.
    """
    widest = max((len(str(v)) for v in values), default=0)
    return [str(v).center(widest) for v in values]


def _spoke_rotation(angle: float) -> tuple[float, bool]:
    """Degrees to rotate text so it runs along its own spoke, and whether the
    spoke points left.

    The axis is drawn with theta zero at twelve o'clock running clockwise, so a
    stored angle of 0 is straight up. Text on the left half is flipped end for
    end and anchored on its right, which keeps every word reading left to right
    instead of upside down.
    """
    screen = (np.pi / 2) - angle          # matplotlib's own frame
    degrees = np.degrees(screen) % 360.0
    # bool(), not the numpy scalar np.degrees hands back: callers and
    # tests compare it with `is True` / `is False`.
    flipped = bool(90.0 < degrees < 270.0)
    spin = degrees - 180.0 if flipped else degrees
    # Normalised to (-180, 180]. matplotlib takes 350 and -10 alike, but one of
    # them is readable in a traceback and the other is not.
    return float((spin + 180.0) % 360.0 - 180.0), flipped


def _spoke_label(label: str) -> str:
    """One line, because a rotated spoke has the room the ring did not.

    The two-line wrapping existed to stop horizontal names colliding at twelve
    o'clock. Rotated labels run outward, so the wrap only made them shorter and
    harder to read: "BIG CH.\\nCREATED" over two lines is a worse word than
    "BIG CH. CREATED" along one.
    """
    return display_label(label).replace("\n", " ")


def _mix(color: str, towards: str, amount: float) -> str:
    """Move one colour a fraction of the way towards another."""
    try:
        a = mcolors.to_rgb(color)
        b = mcolors.to_rgb(towards)
    except (ValueError, TypeError):
        return color
    amount = min(max(float(amount), 0.0), 1.0)
    return mcolors.to_hex(tuple(x + (y - x) * amount for x, y in zip(a, b)))


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


# ── Rates need a denominator before they mean anything ───────────────────────
#
# Lucas Herrington came on for thirty-three minutes, played three passes and
# completed all three. That is 100%, which ranked him in the 95th percentile
# for passing accuracy — above Bruno Fernandes, who played ninety-two at 80%
# and scored 53. The radar said the substitute was the better passer, from a
# sample of three.
#
# A rate is a ratio, and a ratio built on almost nothing carries almost no
# information. Below the floor the value is still printed — it happened — but
# it is not ranked against anyone, so it draws no bar and claims nothing.
#
# The floors are per metric because the denominators are not comparable: a
# player can reasonably contest four aerials in a match and would have to be
# uninvolved to play only fifteen passes.
RATE_FLOORS = {
    "Pass %": ("Passes", 15),
    "Grd duels\nwon": ("Grd_duels_att", 4),
    "Aerials\nwon": ("Aer_att", 4),
    "Duels\nwon": ("Duels_att", 5),
    "Long\nballs": ("Long\nballs", 4),
    "Shots": ("Shots", 2),
}


def rate_is_measured(metrics: dict, metric: str) -> bool:
    """Does this player's rate rest on enough attempts to be ranked?

    True for every metric that is not a rate, so callers can ask about any
    slice without knowing which is which.
    """
    floor = RATE_FLOORS.get(metric)
    if floor is None:
        return True
    key, minimum = floor
    try:
        return float(metrics.get(key, 0) or 0) >= minimum
    except (TypeError, ValueError):
        return False


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


def goalkeeper_metrics(events: pd.DataFrame, player: str) -> dict:
    """Per-match goalkeeping from the event stream.

    Nothing here existed. A keeper was drawn on the outfield radar — goals,
    dribbles, expected goals, aerial duels — so twenty-two of his thirty
    slices were structurally zero and not one of them described his match. The
    provider had been sending Save, Claim, Punch, KeeperSweeper, Smother,
    KeeperPickup and PenaltyFaced the whole time.

    Shot-stopping is measured against post-shot expected goals rather than
    against saves alone: a keeper who faces eight tame shots and saves all
    eight has had an easier afternoon than one who faces three that were going
    in, and the save count cannot tell them apart.
    """
    from match_metrics import post_shot_xg

    ev = events
    mine = ev[ev["player"].astype(str) == str(player)]
    if not len(mine):
        return {}
    kind = mine["type"].astype(str) if "type" in mine else pd.Series([], dtype=str)
    outcome = (mine["outcome"].astype(str) if "outcome" in mine
               else pd.Series(index=mine.index, dtype=str))
    ok = outcome.eq("Successful")

    def count(*types, only_successful=False):
        hit = kind.isin(types)
        if only_successful:
            hit &= ok
        return int(hit.sum())

    # Everything the opposition put on target while he was the keeper.
    team_id = mine["team_id"].dropna()
    team_id = int(team_id.iloc[0]) if len(team_id) else None
    faced = ev[ev["team_id"].ne(team_id)] if team_id is not None else ev.iloc[0:0]
    on_target = faced[faced.get("is_shot", False).fillna(False).astype(bool)]
    psxg = float(post_shot_xg(on_target).sum()) if len(on_target) else 0.0

    conceded = 0
    if "is_goal" in faced and team_id is not None:
        goals = faced[faced["is_goal"].astype(str).str.lower().eq("true")]
        own = goals.get("is_own_goal")
        if own is not None:
            goals = goals[~own.astype(str).str.lower().eq("true")]
        conceded = int(len(goals))

    saves = count("Save")
    shots_faced = saves + conceded

    passes = int((kind.eq("Pass")).sum())
    passes_done = int((kind.eq("Pass") & ok).sum())
    long_att = long_done = 0
    if "pass_length" in mine:
        lengths = pd.to_numeric(mine["pass_length"], errors="coerce")
        is_long = kind.eq("Pass") & lengths.ge(32)
        long_att = int(is_long.sum())
        long_done = int((is_long & ok).sum())

    return {
        # shot stopping
        "Saves": saves,
        "Save %": int(round(100 * saves / shots_faced)) if shots_faced else 0,
        "Goals\nconceded": conceded,
        "Shots\nfaced": shots_faced,
        "Penalties\nfaced": count("PenaltyFaced"),
        # Kept in the dictionary, kept off the radar. Post-shot expected goals
        # is the measure a keeper should be judged on, and this project's
        # implementation is a heuristic rather than a fitted model: across
        # every rendered fixture it totals 30.0 against 60 goals actually
        # scored, a ratio of 0.50. "Goals prevented" off that baseline would
        # put every keeper in every report eight tenths of a goal below
        # expectation — a statement about the model that reads as one about
        # the man. It returns to the radar when the model is calibrated.
        "PSxG\nfaced": round(psxg, 2),
        # command of the box
        "Claims": count("Claim", only_successful=True),
        "Punches": count("Punch"),
        "Pickups": count("KeeperPickup"),
        "Smothers": count("Smother"),
        # off the line
        "Sweeps": count("KeeperSweeper"),
        "Recov\neries": count("BallRecovery"),
        "Clear\nances": count("Clearance"),
        "Errors": count("Error"),
        # distribution
        "Passes": passes,
        "Passes_comp": passes_done,
        # int, not round(): "47.0" is a percentage printed with a decimal it
        # does not have. And the long-ball slice holds the attempt count,
        # because _RATIO_DISPLAY reads the metric itself as the denominator —
        # holding the completed count in both places printed "9 / 9" beside a
        # long-ball accuracy of 22%.
        "Pass %": int(round(100 * passes_done / passes)) if passes else 0,
        "Long\nballs": long_att,
        "Longballs_comp": long_done,
        "Long ball %": int(round(100 * long_done / long_att)) if long_att else 0,
    }


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


# ── Who a player is measured against ─────────────────────────────────────────
#
# Every bar was a percentile against all twenty-nine players on the pitch, so a
# centre-back was ranked on expected goals against forwards and a forward on
# clearances against centre-backs. Half of every radar came out empty — the
# median player had fifteen of thirty slices at zero — and the shape described
# the position rather than the performance. Two centre-backs looked alike
# because they were centre-backs, not because they played alike.
#
# Comparing within a line makes a long bar mean something a reader can act on:
# this defender did that more than the other defenders did.
#
# Three lines rather than eleven positions, because a match has about ten
# players a side and splitting further leaves a pool of two. A pool that small
# says more about who else was picked than about the player, so a line that
# thin falls back to the whole pitch.
POSITION_LINES = {
    "GK": "keeper",
    "DC": "defence", "DL": "defence", "DR": "defence",
    "DMC": "midfield", "MC": "midfield", "ML": "midfield", "MR": "midfield",
    "AMC": "attack", "AML": "attack", "AMR": "attack", "FW": "attack",
}
MIN_LINE_POOL = 5


def position_line(code: str) -> str:
    """Which of the three lines a position belongs to, or "" if unknown."""
    return POSITION_LINES.get(str(code).strip(), "")


def line_pool(players: pd.DataFrame | None, elig, line: str) -> list:
    """The eligible players in one line, or the whole pool if too few.

    A keeper is never pooled with outfielders — there is only one a side, and
    ranking him against ten players who are not keepers is what produced a
    radar of thirty empty slices. He falls through to the full pool, where the
    passing metrics at least mean something, until the goalkeeper's own
    measures exist.
    """
    if not line or line == "keeper" or players is None:
        return list(elig)
    if "name" not in getattr(players, "columns", []):
        return list(elig)
    same = {
        str(row["name"])
        for _, row in players.iterrows()
        if position_line(row.get("position", "")) == line
    }
    pool = [p for p in elig if str(p) in same]
    return pool if len(pool) >= MIN_LINE_POOL else list(elig)


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


# Opta's position codes, spelled out. The subtitle used to print the value of
# ``player_role``, which is a participation status and not a position: the
# goalkeeper was labelled "Player" and Semi Ajayi, who scored, was labelled
# "sub_out". players.csv has carried the real position all along.
POSITION_NAMES = {
    "GK": "Goalkeeper",
    "DC": "Centre-back",
    "DL": "Left-back",
    "DR": "Right-back",
    "DMC": "Defensive midfielder",
    "MC": "Central midfielder",
    "ML": "Left midfielder",
    "MR": "Right midfielder",
    "AMC": "Attacking midfielder",
    "AML": "Left winger",
    "AMR": "Right winger",
    "FW": "Forward",
    "Sub": "Substitute",
}


def _player_role(events: pd.DataFrame, player: str, default="Player") -> str:
    if "player_role" not in events.columns:
        return default
    s = events[events["player"].astype(str) == str(player)]["player_role"].dropna()
    return str(s.iloc[0]) if len(s) else default


def player_position(players: pd.DataFrame | None, player: str) -> str:
    """Opta's code for where this player lined up, or "" if it is not known."""
    if players is None or "position" not in getattr(players, "columns", []):
        return ""
    if "name" not in players.columns:
        return ""
    found = players[players["name"].astype(str) == str(player)]["position"].dropna()
    return str(found.iloc[0]) if len(found) else ""


def describe_position(code: str, fallback: str = "Player") -> str:
    """The position as a reader would say it."""
    return POSITION_NAMES.get(str(code).strip(), fallback if not code else str(code))


def make_player_pizza(
    events, player, team_name, role, allm, elig, subtitle_extra="", opponent_name="",
    team_color=None, opponent_color=None, side="home", players=None,
):
    """Build and return the pizza Figure for one player.

    ``team_color`` and ``opponent_color`` decide which way this side's five
    group colours lean. Both are needed, not just the player's: the two casts
    are held apart from each other, so a fixture between two red teams still
    produces two pages a reader can tell apart. Omit them and the untinted
    palette is used.

    ``players`` is the squad export. It carries the position, which decides two
    things the radar was getting wrong without it: what the subtitle calls the
    player, and who the bars measure him against.
    """
    # Ranked within his own line rather than against everyone on the pitch.
    position = player_position(players, player)
    keeper = position_line(position) == "keeper"
    groups = GK_GROUPS if keeper else GROUPS
    me_m = goalkeeper_metrics(events, player) if keeper else player_metrics(events, player)
    if keeper and not me_m:
        # No keeper events at all: fall back rather than draw an empty ring.
        keeper, groups, me_m = False, GROUPS, player_metrics(events, player)
    pool = line_pool(players, elig, position_line(position))

    # One colour per group, not five steps of the team's: five shades of one
    # kit asks the reader to compare lightness to tell a tackle from a through
    # ball. The kit decides which way all five lean instead, which names the
    # side without spending the encoding.
    if team_color and opponent_color:
        # ``side`` rather than argument order. Reading the pair as "mine first"
        # works only while the two kits differ: two white shirts have no hue to
        # read, both fall back to the same value, and the away radar was drawn
        # in the home palette — the one case where telling the sides apart
        # matters most and nothing else on the page does it.
        home_kit = opponent_color if side == "away" else team_color
        away_kit = team_color if side == "away" else opponent_color
        offsets = fixture_hue_offsets(home_kit, away_kit)
        offset = offsets[1] if side == "away" else offsets[0]
        group_colors = group_palette_for(offset, team_color, len(groups))
    else:
        group_colors = group_palette(len(groups))

    labels, colors, vals, disps, pcts, gidx, thin = [], [], [], [], [], [], []
    for gi, (_gn, _gc, ms) in enumerate(groups):
        gc = group_colors[gi]
        for m in ms:
            v = me_m.get(m, 0)
            labels.append(m)
            colors.append(gc)
            vals.append(v)
            if m in _RATIO_DISPLAY:  # "numerator / denominator"
                num_k, den_k = _RATIO_DISPLAY[m]
                # No space around the slash. Every tile is padded to the
                # widest string on the radar, so the two thin spaces that
                # separated "won" from "contested" set the width of all
                # twenty-six — and near twelve and six o'clock a horizontal
                # tile spends its width across its neighbours' spokes rather
                # than along its own, so the widest string is what decides
                # whether the ring collides with itself.
                disps.append(f"{me_m.get(num_k, 0)}/{me_m.get(den_k, 0)}")
            else:
                disps.append(f"{v}")
            # A rate resting on three passes is printed and not ranked: the
            # wedge is a comparison and there is nothing here to compare with.
            measured = rate_is_measured(me_m, m)
            thin.append(not measured and not keeper)
            if keeper:
                pcts.append(gk_bar(m, v))
            else:
                pcts.append(
                    _percentile(allm, pool, m, v) if (pool and measured) else 0)
            gidx.append(gi)

    # Every tile the same width: monospace plus a common character count.
    disps = pad_values(disps)

    N = len(labels)
    n_groups = len(groups)
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

    # Every label and every chip used to be drawn horizontally on a ring. That
    # is fine at three o'clock and unreadable at twelve: two-line names stack
    # towards the arc above them and the chip below, so "GRD DUELS WON" sat on
    # the violet arc and "5 / 12" sat on "DUELS WON". Nudging the radii only
    # moved the collision somewhere else, because the crowding is a property of
    # horizontal text on a circle, not of the gap between two rings.
    #
    # Both now rotate with their own spoke. The spacing between neighbours is
    # then constant in angle at every clock position, and no two can overlap
    # however many metrics the radar carries.
    # Labels grow inwards from RLAB rather than outwards from it. Outward, a
    # long name such as "FINAL 3RD PASSES" ran through the arc, and pushing the
    # arc out far enough to clear it left the plot itself small in a wide ring
    # of empty page. Inward, every label ends flush against the arc whatever
    # its length, the long ones reach back across space that was empty anyway,
    # and the plot keeps the room.
    R0, RMAX, RVAL, RLAB, RARC, OUT_LIM = 14, 100, 112, 168, 176, 186

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
    for gi, (_gn, _gc, _ms) in enumerate(groups):
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
    for gi, (_gn, _gc, _ms) in enumerate(groups):
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
    for a, lab, dv, p, c, v, unmeasured in zip(
            angs, labels, disps, pcts, colors, vals, thin):
        chip = chip_by_group.get(c, _chip_fill(c))
        spin, flipped = _spoke_rotation(a)
        ax.text(
            a,
            RLAB,
            _spoke_label(lab),
            color=TEXT_BRIGHT,
            fontsize=9.4,
            fontweight="bold",
            family="monospace",
            rotation=spin,
            rotation_mode="anchor",
            # Inverted from the usual outward anchoring: the text ends at RLAB
            # and runs back towards the centre, so every label is flush with
            # the arc no matter how long it is.
            ha="left" if flipped else "right",
            va="center",
            zorder=5,
            clip_on=False,
        )

        # Thirty tiles of equal weight give the eye nothing to prioritise: a
        # zero shouted as loudly as the best figure on the pitch, and a third
        # of a defensive midfielder's radar is zeroes. Weight now follows the
        # percentile the wedge already draws, so the two or three things the
        # player actually led are the ones that carry filled tiles.
        style = _chip_style(chip, c, float(p), _is_zero(v, dv), unmeasured)
        ax.text(
            a,
            RVAL,
            dv,
            color=style["color"],
            fontsize=10.4,
            fontweight=style["weight"],
            family="monospace",
            # Deliberately not rotated. A turned label is still a word and the
            # eye rights it; a turned number is not — "0.01" on the lower arc
            # read as "10.0" upside down. Numbers stay level at every clock
            # position, and the rotated labels outside them are what buys the
            # room that used to force them to collide.
            ha="center",
            va="center",
            zorder=7,
            clip_on=False,
            # Saturated mid-lightness fills (a red team's middle ramp step)
            # sit where neither tier quite clears 4.5:1 on its own. The stroke
            # is a no-op everywhere else, and none at all on an unfilled tile.
            path_effects=style["effects"],
            bbox=style["bbox"],
        )
    for sp in ax.spines.values():
        sp.set_visible(False)

    # clean centre ring
    ax.plot(
        np.linspace(0, 2 * np.pi, 120), [R0] * 120, color="#2e2e2e", lw=1.2, zorder=6
    )

    participation = _get_participation(events).get(str(player), {})
    played_time = participation.get("played_time", "0′ 00″")
    # ``role`` is a participation status — "Player", "sub_in", "sub_out" — and
    # was printed where a reader expects a position, so the goalkeeper read
    # "Player" and a centre-back who scored read "sub_out". The position comes
    # from the squad export; whether he started is said in the words for it.
    described = describe_position(position, fallback=str(role))
    entrance = {"sub_in": "on as a substitute",
                "sub_out": "substituted"}.get(str(role), "")
    # A player listed as "Sub" has no position recorded, so "Substitute" is all
    # the squad knows about him and the entrance note would only repeat it.
    if described == "Substitute":
        entrance = ""
    sub = f"{str(team_name).upper()}  ·  {described}  ·  {played_time} played"
    if entrance:
        sub += f" ({entrance})"
    if subtitle_extra:
        sub += f"  ·  {subtitle_extra}"
    if _identity is not None:
        _identity.amoled_header(
            fig, str(player).upper(), sub, section="PLAYER PIZZA",
            active_team=team_name,
        )

    ng = len(groups)
    step = min(0.135, 0.90 / max(ng - 1, 1))
    lfs = 11 if ng <= 5 else 9
    lx = 0.5 - (ng - 1) * step / 2 - 0.02  # centre the legend row
    for gi, (gn, _gc, _ms) in enumerate(groups):
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
    # The caption has to name the pool the bars were actually drawn against.
    # It said "vs all match players" while the comparison was being made
    # against every player on the pitch, which was the defect; saying it still,
    # now that the comparison is within the line, would be the same sentence
    # telling a different lie.
    against = {"defence": "the defenders", "midfield": "the midfielders",
               "attack": "the attackers"}.get(position_line(position),
                                              "all match players")
    if keeper:
        measured_against = "share of a strong goalkeeping match for that action"
    elif len(pool) < len(elig):
        measured_against = f"percentile among {against} on the pitch"
    else:
        measured_against = "percentile vs all match players"
    # Two lines. Naming the pool made the single line wide enough to run off
    # both edges of the page, and the half a reader needs first — what the bar
    # length means — was the half that got cut.
    fig.text(
        0.5, 0.030,
        f"bar length = {measured_against}   ·   "
        "dashed chip = too few attempts to rank",
        ha="center", color="#5f5f5f", fontsize=9.5, style="italic",
    )
    fig.text(
        0.5, 0.010,
        "chip = match value   ·   passes and long balls = completed/total   ·   "
        "shots = on-target/total   ·   duels = won/contested",
        ha="center", color="#4a4a4a", fontsize=8.5, style="italic",
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
    # Stored packages may contain kit colours from an older run. The active
    # production identity must win when those packages are regenerated, just
    # as it does for a newly collected fixture.
    if not USE_REAL_TEAM_KIT_COLORS:
        return C_HOME if side == "home" else C_AWAY
    key = "home_color" if side == "home" else "away_color"
    supplied = str((info or {}).get(key) or "").strip()
    if supplied:
        try:
            mcolors.to_rgb(supplied)
            return supplied
        except ValueError:
            pass
    return C_HOME if side == "home" else C_AWAY


def _squad_frame(out_dir, squad=None):
    """The squad export, which carries each player's position.

    Read from the package the radars are being written into rather than passed
    down through four call sites, and treated as optional throughout: a fixture
    without it still renders, in the shape the radar had before positions were
    read.
    """
    if squad is not None:
        return squad
    path = Path(str(out_dir)) / "players.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def export_player_radars(events, info, out_dir, dpi=115, squad=None):
    """Save one pizza PNG per participating player into per-team folders.

    Returns {"home": [(player, rating), ...], "away": [...]} best-first.
    """
    allm, elig = compute_metrics_pool(events)
    squad = _squad_frame(out_dir, squad)
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
                    opponent_color=_side_team_color(
                        info, "away" if side == "home" else "home"),
                    side=side,
                    players=squad,
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
    squad = _squad_frame(out_dir)
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
                    opponent_color=_side_team_color(
                        info, "away" if side == "home" else "home"),
                    side=side,
                    players=squad,
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
                    opponent_color=_side_team_color(
                        info, "away" if side == "home" else "home"),
                    side=side,
                    players=squad,
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
