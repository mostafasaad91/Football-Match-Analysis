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

BG = "#000000"
PANEL = "#08090B"
GRID = "#252A31"
TEXT = "#F5F7FA"
MUTED = "#9BA3AE"
NEUTRAL = "#626A75"
HOME = "#2563EB"
AWAY = "#FF734D"
FOCUS = "#FFD43B"


DASHBOARDS = [
    (
        "Match Story",
        "Result, chance rhythm and the effect of game state",
        ["19_post_match_advanced_dashboard.png", "01_xg_flow.png", "04_goals_breakdown.png", "43_game_state_splits.png"],
    ),
    (
        "Finishing and Shot Quality",
        "Volume, location, post-shot execution and goalkeeper workload",
        ["11_shot_profile.png", "02_shot_map_france.png", "03_shot_map_england.png", "14_goalkeeper_saves.png", "15_xg_summary.png"],
    ),
    (
        "Chance Creation",
        "Final-third access, central occupation and penalty-area conversion",
        ["12_danger_creation_france.png", "13_danger_creation_england.png", "16_zone14_france.png", "17_zone14_england.png", "34_box_entries_france.png", "35_box_entries_england.png"],
    ),
    (
        "Possession Structure by Half",
        "Passing relationships and occupation before and after the interval",
        ["05a_pass_network_france_1h.png", "05b_pass_network_france_2h.png", "06a_pass_network_england_1h.png", "06b_pass_network_england_2h.png", "31a_average_positions_france_1h.png", "31b_average_positions_france_2h.png", "32a_average_positions_england_1h.png", "32b_average_positions_england_2h.png"],
    ),
    (
        "Progression and Territory",
        "Where possession started, landed and added threat",
        ["07_xt_map_france.png", "08_xt_map_england.png", "09_pass_map_france.png", "10_pass_map_england.png", "21_pass_thirds_france.png", "22_pass_thirds_england.png", "33_dominating_zones.png", "20_ball_touches.png"],
    ),
    (
        "Final-Third Delivery",
        "Progressive passing, wide delivery and preferred receiving zones",
        ["24_progressive_france.png", "25_progressive_england.png", "26_crosses_france.png", "27_crosses_england.png", "38_pass_targets_france.png", "39_pass_targets_england.png"],
    ),
    (
        "Pressing and Defensive Control",
        "Engagement height, high regains and protection behind the press",
        ["28_defensive_activity_france.png", "29_defensive_activity_england.png", "30_defensive_summary.png", "36_high_regains_france.png", "37_high_regains_england.png", "40_ppda_pressing.png"],
    ),
    (
        "Transitions and Final Verdict",
        "Open-field efficiency, advanced team metrics and sequence leaders",
        ["41_transition_outcomes.png", "42_advanced_metrics.png", "18_match_stats.png", "44_player_sequence_leaders.png"],
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


def build_qa_contact_sheets(out_dir: Path | str = DEFAULT_OUT) -> list[Path]:
    """Build exactly eight story-led QA dashboards from the strongest visuals."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("qa_contact_sheet_*.png"):
        old.unlink(missing_ok=True)

    generated: list[Path] = []
    for sheet_index, (title, subtitle, filenames) in enumerate(DASHBOARDS, start=1):
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
        fig.text(0.735, 0.928, "FRANCE", color=HOME, fontsize=11, fontweight="bold", ha="right")
        fig.text(0.79, 0.925, "4 - 6", color=TEXT, fontsize=18, fontweight="bold", ha="center")
        fig.text(0.845, 0.928, "ENGLAND", color=AWAY, fontsize=11, fontweight="bold", ha="left")
        fig.text(0.975, 0.958, f"DASHBOARD {sheet_index:02d} / 08", color=NEUTRAL, fontsize=7.5, fontweight="bold", ha="right", va="top")
        fig.add_artist(Line2D([0.025, 0.50], [0.862, 0.862], transform=fig.transFigure, color=HOME, lw=2.2))
        fig.add_artist(Line2D([0.50, 0.975], [0.862, 0.862], transform=fig.transFigure, color=AWAY, lw=2.2))
        fig.text(0.025, 0.021, "CURATED FROM REAL MATCH EVENTS · QA + STORY SELECTION", color=NEUTRAL, fontsize=7)
        if missing:
            fig.text(0.975, 0.021, f"Missing {len(missing)} optional source visual(s)", color=AWAY, fontsize=7, ha="right")

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
