"""The graphics for an article about one club, in the order the article reads.

The league cards put all twenty clubs on a page and colour the best three. An article about
one side needs the same pages with that side picked out and the rest greyed, plus two pages
the league view has no use for: the run of matches, and the squad. ``build_story`` draws the
set into one folder, numbered in reading order, so the files line up with the text.
"""

from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from football_analysis.visuals import league_boards as lb
from football_analysis.visuals import social_cards as sc
from football_analysis.visuals import visual_redesign_preview as base
from football_analysis.visuals.typography import display

WIN, DRAW, LOSS = "#3DDC84", "#8E99A4", "#FF6B5B"


def by_match(league: lb.League, team: str) -> pd.DataFrame:
    """The club's matches in order, from the per-match table the aggregation writes."""
    table = pd.read_csv(league.folder / "teams_by_match.csv", encoding="utf-8-sig")
    return table[table["team"].eq(team)].sort_values(["round", "date"]).reset_index(drop=True)


def match_log(league: lb.League, team: str) -> Path:
    """Each match as a pair of bars, chances made against chances given up, with the result."""
    games = by_match(league, team)
    fig = sc.frame(
        league,
        "Overview",
        f"{team}, match by match",
        "Expected goals created and conceded in each game, with the result above",
        "xG is the model's value of every shot. PPDA is opponent passes per defensive action: lower is a harder press.",
        key=[("■", "xG created", league.colour(team)), ("■", "xG conceded", "#5A646E")],
    )
    ax = fig.add_axes([0.37, 0.24, 0.60, 0.60])
    sc.clean(ax, grid="y")
    xs = np.arange(len(games))
    ax.bar(xs - 0.2, games["xG"], width=0.38, color=league.colour(team))
    ax.bar(xs + 0.2, games["xG_against"], width=0.38, color="#5A646E")
    top = float(max(games["xG"].max(), games["xG_against"].max()))
    ax.set_ylim(0, top * 1.38)
    ax.set_xlim(-0.6, len(games) - 0.4)
    ax.set_xticks([])
    for x, game in zip(xs, games.itertuples()):
        ax.text(
            x - 0.2,
            game.xG + top * 0.02,
            f"{game.xG:.2f}",
            ha="center",
            va="bottom",
            color=base.TEXT,
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            x + 0.2,
            game.xG_against + top * 0.02,
            f"{game.xG_against:.2f}",
            ha="center",
            va="bottom",
            color=base.MUTED,
            fontsize=9,
            fontweight="bold",
        )
        colour = (
            WIN
            if game.goals_for > game.goals_against
            else (LOSS if game.goals_for < game.goals_against else DRAW)
        )
        result = "W" if colour == WIN else ("L" if colour == LOSS else "D")
        ax.text(
            x,
            top * 1.27,
            f"{int(game.goals_for)}–{int(game.goals_against)}",
            ha="center",
            va="center",
            color=colour,
            **display(30),
        )
        ax.text(
            x,
            top * 1.14,
            result,
            ha="center",
            va="center",
            color=colour,
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            x,
            -top * 0.06,
            f"{game.opponent} ({game.venue})",
            ha="center",
            va="top",
            color=base.TEXT,
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            x,
            -top * 0.14,
            f"Round {int(game.round)}",
            ha="center",
            va="top",
            color=base.MUTED,
            fontsize=8,
        )
        ax.text(
            x,
            -top * 0.215,
            f"shots {int(game.shots)}–{int(game.shots_against)}",
            ha="center",
            va="top",
            color=base.MUTED,
            fontsize=8,
        )
        press = game.ppda_passes_allowed / max(game.ppda_defensive_actions, 1)
        ax.text(
            x,
            -top * 0.285,
            f"PPDA {press:.1f}",
            ha="center",
            va="top",
            color=base.MUTED,
            fontsize=8,
        )
    return sc.save(fig, league, "teams", f"match_log_{team.replace(' ', '_')}")


# (column, header, format, lower_is_better)
SQUAD_COLUMNS = [
    ("minutes", "MIN", "{:.0f}", False),
    ("goals", "G", "{:.0f}", False),
    ("assists", "A", "{:.0f}", False),
    ("xG", "xG", "{:.2f}", False),
    ("xA", "xA", "{:.2f}", False),
    ("key_passes", "KEY", "{:.0f}", False),
    ("passes_into_box", "BOX", "{:.0f}", False),
    ("progressive_passes", "PROG", "{:.0f}", False),
    ("line_breaking_passes", "LBP", "{:.0f}", False),
    ("pass_pct", "PASS%", "{:.0f}", False),
    ("tackles_won", "TKL", "{:.0f}", False),
    ("interceptions", "INT", "{:.0f}", False),
    ("recoveries", "REC", "{:.0f}", False),
]


def squad_table(league: lb.League, team: str, minimum_minutes: float = 90.0) -> Path:
    """Who did what: the squad's totals, each column shaded from the lowest to the highest in it."""
    squad = league.players[
        (league.players["team"].eq(team)) & (league.players["minutes"] >= minimum_minutes)
    ]
    squad = squad.sort_values("minutes", ascending=False).reset_index(drop=True)
    fig = sc.frame(
        league,
        "Players",
        f"{team}: who did what",
        f"Totals over the block for players with {minimum_minutes:.0f}+ minutes · each column shaded from the lowest to the highest in it",
        "KEY key passes · BOX passes into the box · PROG progressive passes · LBP line-breaking passes · TKL tackles won · REC recoveries.",
    )
    ax = sc.table_axes(fig)
    ramp = sc.percentile_ramp()
    left, right, top, bottom = sc.TABLE_LEFT, 0.975, 0.905, 0.07
    name_w = 0.15
    col_w = (right - left - name_w) / len(SQUAD_COLUMNS)
    row_h = min(0.055, (top - bottom) / max(len(squad), 1))
    for j, (_key, head, _fmt, _low) in enumerate(SQUAD_COLUMNS):
        fig.text(
            left + name_w + col_w * (j + 0.5),
            top + 0.008,
            head,
            color=base.MUTED,
            fontsize=7.6,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
    ranks = {c[0]: lb.percentile_rank(squad[c[0]], c[3]) for c in SQUAD_COLUMNS}
    for i, player in squad.iterrows():
        y = top - (i + 1) * row_h
        ax.add_patch(
            Rectangle(
                (left, y),
                right - left,
                row_h - 0.003,
                facecolor=base.PANEL if i % 2 == 0 else base.BG,
                edgecolor="none",
            )
        )
        fig.text(
            left + 0.006,
            y + row_h / 2,
            lb.surname(player["player"]),
            color=base.TEXT,
            fontsize=9.2,
            fontweight="bold",
            va="center",
        )
        role = str(player["role"])
        fig.text(
            left + name_w - 0.008,
            y + row_h / 2,
            "sub" if role == "Unknown" else role[:3].upper(),
            color=base.NEUTRAL,
            fontsize=6.8,
            va="center",
            ha="right",
        )
        for j, (key, _head, fmt, _low) in enumerate(SQUAD_COLUMNS):
            cx = left + name_w + col_w * j
            rank = float(ranks[key].iloc[i]) if pd.notna(ranks[key].iloc[i]) else 0.0
            ax.add_patch(
                Rectangle(
                    (cx + 0.002, y + 0.004),
                    col_w - 0.004,
                    row_h - 0.011,
                    facecolor=ramp(rank),
                    edgecolor="none",
                )
            )
            value = player[key]
            fig.text(
                cx + col_w / 2,
                y + row_h / 2,
                "—" if pd.isna(value) else fmt.format(value),
                color=base.BG if rank > 0.62 else base.TEXT,
                fontsize=8.6,
                fontweight="bold",
                va="center",
                ha="center",
            )
    return sc.save(fig, league, "teams", f"squad_{team.replace(' ', '_')}")


def ascii_slug(text: str) -> str:
    """A file-name-safe, lower-case ASCII form of a name (``Ødegaard`` -> ``odegaard``)."""
    table = str.maketrans({"ø": "o", "Ø": "O", "ß": "ss", "æ": "ae", "đ": "d"})
    plain = unicodedata.normalize("NFKD", text.translate(table)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", plain.lower()).strip("_")


def _scatter(name: str):
    return next(card.draw for card in sc._team_scatter_cards() if card.name == name)


def _ranking(key: str):
    return sc.team_ranking_card(next(spec for spec in sc.TEAM_RANKINGS if spec[0] == key)).draw


def story(team: str, players: list[str]):
    """The pages of the article, in reading order: (slug, how to draw it)."""
    pages = [
        ("profile", lambda lg: lb.team_profile(lg, team)),
        ("match_by_match", lambda lg: match_log(lg, team)),
        ("chances_for_and_against", sc._xg_quadrant),
        ("xg_conceded", sc.t_xg_against),
        ("what_the_defence_is_made_of", sc.t_defence_makeup),
        ("pressing_and_conceding", _scatter("defence_00_pressing_and_conceding")),
        ("ppda", sc.t_ppda),
        ("pressing_that_pays", _scatter("pressing_00_press_wins")),
        ("safe_or_direct", _scatter("passing_00_safe_or_direct")),
        ("progressive_and_line_breaking", sc.t_progress),
        ("box_entries", _ranking("box_entries")),
        ("entries_against_xg", _scatter("attack_00_entries_against_xg")),
        ("volume_against_quality", _scatter("attack_00_volume_against_quality")),
        ("chance_sources", sc.t_set_pieces),
        ("goals_against_xg", sc.t_finishing),
        ("squad", lambda lg: squad_table(lg, team)),
    ]
    pages += [
        (
            f"player_{ascii_slug(lb.surname(name))}",
            lambda lg, name=name: lb.player_profile(lg, name, team),
        )
        for name in players
    ]
    return pages


def build_story(league: lb.League, team: str, players: list[str], out_dir: Path) -> list[Path]:
    """Draw the article's pages into ``out_dir`` as ``NN_slug.png``, the club picked out on every one."""
    base.theme()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_drawing"
    paths = []
    sc.FOCUS, sc.OUT_DIR = team, scratch
    scratch.mkdir(exist_ok=True)
    try:
        for index, (slug, draw) in enumerate(story(team, players), start=1):
            drawn = draw(league)
            if drawn is None:
                continue
            target = out_dir / f"{index:02d}_{slug}.png"
            shutil.copyfile(drawn, target)
            paths.append(target)
    finally:
        sc.FOCUS, sc.OUT_DIR = None, None
        shutil.rmtree(scratch, ignore_errors=True)
    return paths


def build_everything(league: lb.League, team: str, players: list[str], out_dir: Path) -> list[Path]:
    """Every club graphic of the block with ``team`` picked out, plus its own pages.

    The league cards keep the names they have in the visuals folder (``attack_02_xg.png``);
    the club's own pages are ``club_*`` and its players ``player_*``.
    """
    base.theme()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sequence: dict[tuple[str, str], int] = {}
    paths = []
    sc.FOCUS, sc.OUT_DIR = team, out_dir
    try:
        for card in sc.cards(league):
            if card.scope != "teams":
                continue
            drawn = card.draw(league)
            if drawn is not None:
                paths.append(sc.number_in_group(drawn, card, sequence))
        own = [
            ("club_profile", lambda lg: lb.team_profile(lg, team)),
            ("club_match_by_match", lambda lg: match_log(lg, team)),
        ]
        if players:
            own.append(("club_squad", lambda lg: squad_table(lg, team)))
        own += [
            (
                f"player_{ascii_slug(lb.surname(name))}",
                lambda lg, name=name: lb.player_profile(lg, name, team),
            )
            for name in players
        ]
        for slug, draw in own:
            drawn = draw(league)
            if drawn is not None:
                paths.append(Path(drawn).replace(out_dir / f"{slug}.png"))
    finally:
        sc.FOCUS, sc.OUT_DIR = None, None
    return paths
