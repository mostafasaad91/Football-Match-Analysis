from __future__ import annotations

import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "output" / "France_vs_England_4-6"

# Sheet chrome follows the active theme so light-theme thumbnails sit on a
# matching light page instead of floating on the AMOLED black.
from visualization_components import (  # noqa: E402  (after matplotlib backend)
    C_AWAY as _ROLE_AWAY,
    C_HOME as _ROLE_HOME,
    EVENT_HIGHLIGHT as _ROLE_FOCUS,
    IS_LIGHT_THEME,
)

if IS_LIGHT_THEME:
    BG = "#F5F5F5"
    PANEL = "#FFFFFF"
    GRID = "#CCCCCC"
    TEXT = "#333333"
    MUTED = "#666666"
    NEUTRAL = "#888888"
else:
    BG = "#000000"
    PANEL = "#08090B"
    GRID = "#252A31"
    TEXT = "#F5F7FA"
    MUTED = "#9BA3AE"
    NEUTRAL = "#626A75"
HOME = _ROLE_HOME
AWAY = _ROLE_AWAY
# The sheet label used to be amber, the same orphan accent the report carried
# until it was measured against the kit colours and lost. Neutral here too, so
# the contact sheets and the PDF agree on what a structural mark looks like.
FOCUS = _ROLE_FOCUS if IS_LIGHT_THEME else "#C8CDD4"


# Each entry is a filename, or a (filename, note) pair. The note is the line
# printed under the thumbnail's title: on the story sheet it carries the
# narrative, which is what makes an ordered set of visuals a reading rather
# than a grid.
DASHBOARDS = [
    (
        "The Match Story",
        "Result, rhythm, goals, territory and the players who moved them",
        [
            ("14_post_match_advanced_dashboard.png", "The result in 32 indicators"),
            ("01_xg_flow.png", "How danger accumulated"),
            ("35_match_momentum.png", "Who was on top, and when"),
            ("40_win_probability.png", "When the result became likely"),
            ("04_goals_breakdown.png", "What the goals were"),
            ("46_goal_origins.png", "Where they came from"),
            ("33_game_state_splits.png", "How the scoreline changed behaviour"),
            ("44_pitch_control.png", "Who held the space"),
            ("45_sequence_types.png", "How the danger was built"),
            ("32_transition_outcomes.png", "What the open field produced"),
            ("43_action_value.png", "Who moved the needle"),
            ("34_player_sequence_leaders.png", "Who connected the valuable attacks"),
        ],
    ),
    (
        "Finishing and Shot Quality",
        "Volume, location, post-shot placement, set plays and goalkeeper workload",
        ["02_shot_map_france.png", "03_shot_map_england.png", "11_goalkeeper_saves.png", "15_xt_per_minute.png", "36_set_pieces.png"],
    ),
    (
        "Chance Creation",
        "Final-third access, central occupation, playing through a block and penalty-area conversion",
        ["12_zone14_france.png", "13_zone14_england.png", "25_box_entries_france.png", "26_box_entries_england.png",
         "41_playing_through_france.png", "42_playing_through_england.png",
         "47_unlocking_france.png", "48_unlocking_england.png"],
    ),
    (
        "Possession Structure by Half",
        "Passing relationships and occupation before and after the interval",
        ["05a_pass_network_france_1h.png", "05b_pass_network_france_2h.png", "06a_pass_network_england_1h.png", "06b_pass_network_england_2h.png", "22a_average_positions_france_1h.png", "22b_average_positions_france_2h.png", "23a_average_positions_england_1h.png", "23b_average_positions_england_2h.png"],
    ),
    (
        "Progression and Territory",
        "Where possession started, landed, added threat and was surrendered",
        ["07_xt_map_france.png", "08_xt_map_england.png", "09_pass_map_france.png", "10_pass_map_england.png",
         "24_dominating_zones.png", "37_ball_losses_france.png", "38_ball_losses_england.png"],
    ),
    (
        "Final-Third Delivery",
        "Progressive passing, wide delivery and preferred receiving zones",
        ["16_progressive_france.png", "17_progressive_england.png", "18_crosses_france.png", "19_crosses_england.png", "29_pass_targets_france.png", "30_pass_targets_england.png"],
    ),
    (
        "Pressing and Defensive Control",
        "Engagement height, what triggered the press, high regains and the shape behind it",
        ["20_defensive_activity_france.png", "21_defensive_activity_england.png", "27_high_regains_france.png", "28_high_regains_england.png",
         "31_ppda_pressing.png", "49_press_triggers.png", "39_defensive_shape.png"],
    ),
    (
        "Transitions and Final Verdict",
        "Open-field efficiency and sequence leaders",
        ["32_transition_outcomes.png", "34_player_sequence_leaders.png", "14_post_match_advanced_dashboard.png"],
    ),
]


def _friendly_title(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        stem = stem.split("_", 1)[1]
    title = stem.replace("_", " ").replace(" 1h", " · 1H").replace(" 2h", " · 2H").title()
    return title.replace("Xg", "xG").replace("Xt", "xT").replace("Ppda", "PPDA")


def _layout(count: int) -> tuple[int, int]:
    if count <= 4:
        return 2, 2
    if count <= 6:
        return 2, 3
    if count <= 8:
        return 2, 4
    return 3, 4


# The thumbnail box, in pixels. Every visual is letterboxed onto it before it
# is drawn: the source visuals have different aspect ratios, so at their
# natural shape each one sat at a different height inside its cell and the
# titles stepped up and down across a row.
CELL = (1180, 720)
# Height of the sheet in inches, by row count. A third row makes the page
# taller rather than the thumbnails smaller — squeezing twelve into the
# two-row canvas would have cost a third of every thumbnail's height.
#
# The two-row sheets were 11.25in, sized before the thumbnails were letterboxed
# to a common frame. Measured on a rendered sheet that left 39% of the page
# empty, including a 222pt band between the two rows — taller than a thumbnail.
_SHEET_HEIGHT = {2: 9.0, 3: 12.8}


def _framed(path: Path) -> np.ndarray:
    """Return the visual centred on a constant-size panel."""
    with Image.open(path) as source:
        source = source.convert("RGB")
        source.thumbnail(CELL, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", CELL, PANEL)
        canvas.paste(source, ((CELL[0] - source.width) // 2,
                              (CELL[1] - source.height) // 2))
        return np.asarray(canvas, dtype=np.uint8)


def _split_entry(entry) -> tuple[str, str]:
    """Accept either a bare filename or a (filename, note) pair."""
    if isinstance(entry, str):
        return entry, ""
    name, note = entry
    return name, note


def build_qa_contact_sheets(
    out_dir: Path | str = DEFAULT_OUT,
    *,
    home_name: str = "France",
    away_name: str = "England",
    score: str = "4 - 6",
    home_color: str = HOME,
    away_color: str = AWAY,
    home_slug: str = "france",
    away_slug: str = "england",
) -> list[Path]:
    """Build the story-led QA dashboards from the strongest visuals.

    The set opens with a twelve-up narrative sheet and continues with the
    thematic reviews. Every rendered visual appears on at least one sheet —
    fifteen of them used to reach none, because the list was written when the
    project produced thirty-four and was never revisited when it grew to
    forty-nine.
    """
    # The sheet chrome must match the thumbnails it frames, so the caller's
    # fixture colours are used as given. They default to the role pair, which
    # is also what kit mode falls back to when two kits are too close.
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("qa_contact_sheet_*.png"):
        old.unlink(missing_ok=True)

    dashboards = []
    for title, subtitle, entries in DASHBOARDS:
        resolved = []
        for entry in entries:
            name, note = _split_entry(entry)
            resolved.append(
                (name.replace("france", home_slug).replace("england", away_slug), note)
            )
        dashboards.append((title, subtitle, resolved))
    total = len(dashboards)

    generated: list[Path] = []
    for sheet_index, (title, subtitle, entries) in enumerate(dashboards, start=1):
        present = [(out / name, note) for name, note in entries if (out / name).exists()]
        missing = [name for name, _ in entries if not (out / name).exists()]
        if not present:
            raise FileNotFoundError(f"Dashboard {sheet_index} has no source visuals. Missing: {missing}")

        rows, cols = _layout(len(present))
        fig, axes = plt.subplots(rows, cols, figsize=(20, _SHEET_HEIGHT[rows]), facecolor=BG)
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.set_facecolor(PANEL)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # A taller sheet needs its grid pushed further down, or the team rule
        # closing the header lands on the first row of titles.
        top = 1.0 - 1.62 / _SHEET_HEIGHT[rows]
        title_pad = 16 if rows == 3 else 6

        for item_index, ((path, note), ax) in enumerate(zip(present, axes), start=1):
            ax.imshow(_framed(path))
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=GRID, linewidth=0.9))
            ax.set_title(f"{item_index:02d}  {_friendly_title(path)}", fontsize=8.8, color=TEXT, pad=title_pad, loc="left", fontweight="bold")
            if note:
                ax.text(0, 1.012, note.upper(), transform=ax.transAxes, color=MUTED,
                        fontsize=6.6, fontweight="bold", va="bottom")
            ax.axis("off")

        for ax in axes[len(present):]:
            ax.axis("off")

        # Header positions are given in inches from the top and converted, so
        # the block keeps its spacing on both sheet heights. As figure
        # fractions the subtitle closed on the team rule as soon as the
        # two-row sheet got shorter.
        height = _SHEET_HEIGHT[rows]

        def from_top(inches: float) -> float:
            return 1.0 - inches / height

        fig.subplots_adjust(left=0.018, right=0.982, top=top, bottom=0.046, wspace=0.028, hspace=0.10)
        fig.text(0.025, from_top(0.30), "POST-MATCH VISUAL REVIEW", color=FOCUS, fontsize=8.5, fontweight="bold", va="top")
        fig.text(0.025, from_top(0.62), title, color=TEXT, fontsize=22, fontweight="bold", va="top")
        fig.text(0.025, from_top(1.00), subtitle, color=MUTED, fontsize=10, va="top")
        fig.text(0.735, from_top(0.60), home_name.upper(), color=home_color, fontsize=11, fontweight="bold", ha="right")
        fig.text(0.79, from_top(0.64), score, color=TEXT, fontsize=18, fontweight="bold", ha="center")
        fig.text(0.845, from_top(0.60), away_name.upper(), color=away_color, fontsize=11, fontweight="bold", ha="left")
        fig.text(0.975, from_top(0.30), f"DASHBOARD {sheet_index:02d} / {total:02d}", color=NEUTRAL, fontsize=7.5, fontweight="bold", ha="right", va="top")
        rule_y = from_top(1.28)
        fig.add_artist(Line2D([0.025, 0.50], [rule_y, rule_y], transform=fig.transFigure, color=home_color, lw=2.2))
        fig.add_artist(Line2D([0.50, 0.975], [rule_y, rule_y], transform=fig.transFigure, color=away_color, lw=2.2))
        fig.text(0.025, 0.018, "CURATED FROM REAL MATCH EVENTS · QA + STORY SELECTION", color=NEUTRAL, fontsize=7)
        if missing:
            fig.text(0.975, 0.018, f"Missing {len(missing)} optional source visual(s)", color=away_color, fontsize=7, ha="right")

        path = out / f"qa_contact_sheet_{sheet_index:02d}.png"
        fig.savefig(path, dpi=135, facecolor=BG)
        plt.close(fig)
        generated.append(path)
        del fig, axes
        gc.collect()

    return generated


if __name__ == "__main__":
    outputs = build_qa_contact_sheets()
    print(f"Generated {len(outputs)} curated QA dashboards")
