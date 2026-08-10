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
FOCUS = _ROLE_FOCUS if IS_LIGHT_THEME else "#FFD43B"


DASHBOARDS = [
    (
        "Match Story",
        "Result, chance rhythm and the effect of game state",
        ["14_post_match_advanced_dashboard.png", "01_xg_flow.png", "04_goals_breakdown.png", "33_game_state_splits.png"],
    ),
    (
        "Finishing and Shot Quality",
        "Volume, location, post-shot placement and goalkeeper workload",
        ["02_shot_map_france.png", "03_shot_map_england.png", "11_goalkeeper_saves.png", "15_xt_per_minute.png"],
    ),
    (
        "Chance Creation",
        "Final-third access, central occupation and penalty-area conversion",
        ["12_zone14_france.png", "13_zone14_england.png", "25_box_entries_france.png", "26_box_entries_england.png"],
    ),
    (
        "Possession Structure by Half",
        "Passing relationships and occupation before and after the interval",
        ["05a_pass_network_france_1h.png", "05b_pass_network_france_2h.png", "06a_pass_network_england_1h.png", "06b_pass_network_england_2h.png", "22a_average_positions_france_1h.png", "22b_average_positions_france_2h.png", "23a_average_positions_england_1h.png", "23b_average_positions_england_2h.png"],
    ),
    (
        "Progression and Territory",
        "Where possession started, landed and added threat",
        ["07_xt_map_france.png", "08_xt_map_england.png", "09_pass_map_france.png", "10_pass_map_england.png", "24_dominating_zones.png", "29_pass_targets_france.png"],
    ),
    (
        "Final-Third Delivery",
        "Progressive passing, wide delivery and preferred receiving zones",
        ["16_progressive_france.png", "17_progressive_england.png", "18_crosses_france.png", "19_crosses_england.png", "29_pass_targets_france.png", "30_pass_targets_england.png"],
    ),
    (
        "Pressing and Defensive Control",
        "Engagement height, high regains and protection behind the press",
        ["20_defensive_activity_france.png", "21_defensive_activity_england.png", "27_high_regains_france.png", "28_high_regains_england.png", "31_ppda_pressing.png"],
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
    return 2, 4


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
    """Build exactly eight story-led QA dashboards from the strongest visuals."""
    # The sheet chrome must match the thumbnails it frames, so the caller's
    # fixture colours are used as given. They default to the role pair, which
    # is also what kit mode falls back to when two kits are too close.
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("qa_contact_sheet_*.png"):
        old.unlink(missing_ok=True)

    dashboards = [
        (
            title,
            subtitle,
            [
                name.replace("france", home_slug).replace("england", away_slug)
                for name in filenames
            ],
        )
        for title, subtitle, filenames in DASHBOARDS
    ]

    generated: list[Path] = []
    for sheet_index, (title, subtitle, filenames) in enumerate(dashboards, start=1):
        paths = [out / name for name in filenames if (out / name).exists()]
        missing = [name for name in filenames if not (out / name).exists()]
        if not paths:
            raise FileNotFoundError(f"Dashboard {sheet_index} has no source visuals. Missing: {missing}")

        rows, cols = _layout(len(paths))
        fig, axes = plt.subplots(rows, cols, figsize=(20, 11.25), facecolor=BG)
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.set_facecolor(PANEL)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        for item_index, (ax, path) in enumerate(zip(axes, paths), start=1):
            with Image.open(path) as source:
                source = source.convert("RGB")
                source.thumbnail((1180, 720), Image.Resampling.LANCZOS)
                image = np.asarray(source, dtype=np.uint8)
            ax.imshow(image)
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=GRID, linewidth=0.9))
            ax.set_title(f"{item_index:02d}  {_friendly_title(path)}", fontsize=8.8, color=TEXT, pad=6, loc="left", fontweight="bold")
            ax.axis("off")

        for ax in axes[len(paths):]:
            ax.axis("off")

        fig.subplots_adjust(left=0.018, right=0.982, top=0.845, bottom=0.055, wspace=0.028, hspace=0.095)
        fig.text(0.025, 0.958, "POST-MATCH VISUAL REVIEW", color=FOCUS, fontsize=8.5, fontweight="bold", va="top")
        fig.text(0.025, 0.923, title, color=TEXT, fontsize=22, fontweight="bold", va="top")
        fig.text(0.025, 0.885, subtitle, color=MUTED, fontsize=10, va="top")
        fig.text(0.735, 0.928, home_name.upper(), color=home_color, fontsize=11, fontweight="bold", ha="right")
        fig.text(0.79, 0.925, score, color=TEXT, fontsize=18, fontweight="bold", ha="center")
        fig.text(0.845, 0.928, away_name.upper(), color=away_color, fontsize=11, fontweight="bold", ha="left")
        fig.text(0.975, 0.958, f"DASHBOARD {sheet_index:02d} / 08", color=NEUTRAL, fontsize=7.5, fontweight="bold", ha="right", va="top")
        fig.add_artist(Line2D([0.025, 0.50], [0.862, 0.862], transform=fig.transFigure, color=home_color, lw=2.2))
        fig.add_artist(Line2D([0.50, 0.975], [0.862, 0.862], transform=fig.transFigure, color=away_color, lw=2.2))
        fig.text(0.025, 0.021, "CURATED FROM REAL MATCH EVENTS · QA + STORY SELECTION", color=NEUTRAL, fontsize=7)
        if missing:
            fig.text(0.975, 0.021, f"Missing {len(missing)} optional source visual(s)", color=away_color, fontsize=7, ha="right")

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
