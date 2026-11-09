"""The match as a five-slide story carousel, 4:5 for X, LinkedIn and Instagram.

The November 2026 review found the four dashboard posters dense to the point
of being unreadable on a phone: one template, six small panels and four KPI
bars each, the same numbers repeated across boards and no sentence saying what
any of it meant. These slides carry one idea each, drawn large, under a
headline that states the finding:

    1  the cover      score, headline, the xG race, three numbers
    2  the attack     where each side got into the box
    3  the press      when each side pressed and where it won the ball back
    4  the players    who added the most value (VAEP), and the goalkeepers
    5  the takeaways  three sentences, the score and the crests again

2160 x 2700 px. The smallest type is about 28 px at 1080 wide.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from football_analysis.metrics.frame_values import surname
from football_analysis.visuals import crests
from football_analysis.visuals.typography import display
from football_analysis.visuals.visualization_components import (
    BG_DARK, BG_PANEL, GRID_COL, PITCH_LINE, PITCH_LINE_ALPHA, TEXT_DIM, TEXT_MAIN,
    themed_pitch,
)

W, H, DPI = 10.8, 13.5, 200
MARGIN = 0.06
SLIDES = 5


def _truthy(s: pd.Series) -> pd.Series:
    return s.fillna(False) if s.dtype == bool else s.astype(str).str.lower().isin({"true", "1", "yes"})


class Story:
    def __init__(self, events, xg, info, out_dir, home_color, away_color, score, competition, date):
        self.e = events
        self.xg = xg
        self.info = info
        self.out = Path(out_dir)
        self.hid, self.aid = info.get("home_id"), info.get("away_id")
        self.hname, self.aname = str(info.get("home_name")), str(info.get("away_name"))
        self.hc, self.ac = home_color, away_color
        self.score = score
        self.competition = competition
        self.date = date

    # ── chrome ─────────────────────────────────────────────────────────
    def page(self, number: int, kicker: str, headline: str):
        fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG_DARK)
        fig.text(MARGIN, 0.955, f"{kicker.upper()}", color=TEXT_DIM, fontsize=15, fontweight="bold")
        fig.text(1 - MARGIN, 0.955, f"{number} / {SLIDES}", color=TEXT_DIM, fontsize=15,
                 fontweight="bold", ha="right")
        if chr(10) not in headline:
            headline = _wrap(headline, 38)
        fig.text(MARGIN, 0.905, headline, color=TEXT_MAIN, va="top", **display(40))
        # Score strip at the foot of every slide, so a single slide shared
        # alone still says which match it is.
        fig.add_artist(plt.Line2D([MARGIN, 1 - MARGIN], [0.075, 0.075], color=GRID_COL, lw=1.2))
        fig.text(MARGIN, 0.04, f"{self.hname}  {self.score}  {self.aname}", color=TEXT_MAIN,
                 fontsize=17, fontweight="bold", va="center")
        fig.text(1 - MARGIN, 0.04, f"{self.competition} · {self.date}", color=TEXT_DIM,
                 fontsize=14, ha="right", va="center")
        fig.add_artist(plt.Rectangle((0, 0), 0.5, 0.008, color=self.hc, transform=fig.transFigure))
        fig.add_artist(plt.Rectangle((0.5, 0), 0.5, 0.008, color=self.ac, transform=fig.transFigure))
        return fig

    def save(self, fig, name: str) -> Path:
        path = self.out / name
        fig.savefig(path, dpi=DPI, facecolor=BG_DARK)
        plt.close(fig)
        return path

    def stat(self, fig, x, y, label, home, away, fmt="{:.2f}"):
        fig.text(x, y, label.upper(), color=TEXT_DIM, fontsize=14, fontweight="bold")
        fig.text(x, y - 0.045, fmt.format(home), color=self.hc, **display(44))
        fig.text(x + 0.13, y - 0.045, fmt.format(away), color=self.ac, **display(44))

    # ── data ───────────────────────────────────────────────────────────
    def shots(self):
        s = self.e[_truthy(self.e.get("is_shot", pd.Series(False, index=self.e.index)))]
        if "is_penalty_shootout" in s.columns:
            s = s[~_truthy(s["is_penalty_shootout"])]
        return s

    # ── slides ─────────────────────────────────────────────────────────
    def cover(self, headline: str, extras: dict) -> Path:
        fig = self.page(1, "Match study", headline)
        for x, tid, name, colour, ha in ((MARGIN + 0.07, self.hid, self.hname, self.hc, "left"),
                                         (1 - MARGIN - 0.07, self.aid, self.aname, self.ac, "right")):
            crests.place_crest(fig, x, 0.70, int(tid), monogram=name[:3].upper(), colour=colour,
                               width=0.12, background=BG_DARK, zorder=10)
        fig.text(0.5, 0.70, self.score, color=TEXT_MAIN, ha="center", va="center", **display(96))
        fig.text(MARGIN + 0.07, 0.625, self.hname, color=TEXT_MAIN, ha="center", fontsize=18, fontweight="bold")
        fig.text(1 - MARGIN - 0.07, 0.625, self.aname, color=TEXT_MAIN, ha="center", fontsize=18, fontweight="bold")
        # The xG race, large.
        ax = fig.add_axes([MARGIN + 0.02, 0.27, 1 - 2 * MARGIN - 0.04, 0.30])
        ax.set_facecolor((0, 0, 0, 0))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_COL)
        ax.tick_params(colors=TEXT_DIM, labelsize=13)
        shots = self.shots()
        end = max(90, int(pd.to_numeric(self.e["minute"], errors="coerce").max()))
        for tid, colour in ((self.hid, self.hc), (self.aid, self.ac)):
            own = shots[shots["team_id"] == tid].sort_values("minute")
            minutes = np.concatenate([[0], pd.to_numeric(own["minute"]).to_numpy(), [end]])
            cum = np.concatenate([[0], pd.to_numeric(own["xG"], errors="coerce").fillna(0).cumsum().to_numpy()])
            cum = np.append(cum, cum[-1])
            ax.step(minutes, cum, where="post", color=colour, lw=4)
            goals = own[_truthy(own.get("is_goal", pd.Series(False, index=own.index)))]
            for _, g in goals.iterrows():
                y = float(pd.to_numeric(own.loc[:g.name, "xG"], errors="coerce").fillna(0).sum())
                ax.scatter([g["minute"]], [y], s=260, color=colour, edgecolor="white", lw=2, zorder=5)
                ax.annotate(f"{surname(str(g['player']))} {int(g['minute'])}'", (g["minute"], y),
                            xytext=(0, 14), textcoords="offset points", ha="center", color=TEXT_MAIN,
                            fontsize=13, fontweight="bold")
        ax.set_xlim(0, end)
        ax.set_xlabel("Minute", color=TEXT_DIM, fontsize=13)
        ax.set_ylabel("Cumulative xG", color=TEXT_DIM, fontsize=13)
        ax.grid(axis="y", color=GRID_COL, lw=0.8)
        xg = {str(r["team"]): float(r["xG"]) for _, r in self.xg.iterrows()} if "team" in self.xg else {}
        xp = extras.get("expected_points", {})
        ctl = extras.get("game_control", {})
        self.stat(fig, MARGIN, 0.19, "Expected goals", xg.get(self.hname, 0), xg.get(self.aname, 0))
        self.stat(fig, MARGIN + 0.31, 0.19, "Expected points", xp.get("home_xpts", 0), xp.get("away_xpts", 0), "{:.1f}")
        self.stat(fig, MARGIN + 0.62, 0.19, "Time as likelier winner", ctl.get("home", 0), ctl.get("away", 0), "{:.0f}%")
        return self.save(fig, "story_01_cover.png")

    def attack(self) -> Path:
        from football_analysis.metrics.match_metrics import box_entry_mask

        mask = box_entry_mask(self.e)
        counts, lanes = {}, {}
        for tid in (self.hid, self.aid):
            own = self.e[mask & (self.e["team_id"] == tid)]
            counts[tid] = len(own)
            y = pd.to_numeric(own["y"], errors="coerce")
            lane = np.where(y < 33.3, "right", np.where(y > 66.7, "left", "centre"))
            lanes[tid] = pd.Series(lane).value_counts()
        leader = self.hid if counts[self.hid] >= counts[self.aid] else self.aid
        lname = self.hname if leader == self.hid else self.aname
        top_lane = lanes[leader].idxmax() if len(lanes[leader]) else "centre"
        share = lanes[leader].max() / max(counts[leader], 1)
        headline = (f"{lname} got into the box {counts[leader]} times, "
                    f"{share:.0%} of them down the {top_lane}")
        fig = self.page(2, "How they attacked", headline)
        for i, (tid, name, colour) in enumerate(((self.hid, self.hname, self.hc), (self.aid, self.aname, self.ac))):
            ax = fig.add_axes([MARGIN + i * 0.46, 0.18, 0.42, 0.58])
            themed_pitch(ax)
            ax.set_xlim(50, 102)  # attacking half only: the action is all there
            ax.set_ylim(-2, 102)
            own = self.e[mask & (self.e["team_id"] == tid)]
            for _, r in own.iterrows():
                ax.annotate("", xy=(r["end_x"], r["end_y"]), xytext=(r["x"], r["y"]),
                            arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.2, alpha=0.9))
            fig.text(MARGIN + i * 0.46, 0.79, name.upper(), color=colour, fontsize=18, fontweight="bold")
            lane = lanes[tid]
            fig.text(MARGIN + i * 0.46, 0.15,
                     f"{counts[tid]} box entries  ·  L {int(lane.get('left', 0))} · "
                     f"C {int(lane.get('centre', 0))} · R {int(lane.get('right', 0))}",
                     color=TEXT_MAIN, fontsize=15)
        fig.text(MARGIN, 0.11, "Arrows: every completed pass or carry into the penalty area. Attacking left to right.",
                 color=TEXT_DIM, fontsize=12)
        return self.save(fig, "story_02_attack.png")

    def press(self, extras: dict) -> Path:
        from football_analysis.metrics.match_metrics import high_regain_events

        def overall(rows):
            p, a = sum(r["passes"] for r in rows), sum(r["actions"] for r in rows)
            return p / a if a else float("nan")

        h, a = extras["home"]["ppda_timeline"], extras["away"]["ppda_timeline"]
        harder = self.hname if overall(h) <= overall(a) else self.aname
        headline = (f"{harder} pressed harder: {min(overall(h), overall(a)):.1f} opponent passes "
                    f"per defensive action against {max(overall(h), overall(a)):.1f}")
        fig = self.page(3, "Without the ball", headline)
        ax = fig.add_axes([MARGIN + 0.04, 0.55, 1 - 2 * MARGIN - 0.06, 0.25])
        ax.set_facecolor((0, 0, 0, 0))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID_COL)
        ax.tick_params(colors=TEXT_DIM, labelsize=13)
        xs = np.arange(len(h))
        for rows, colour in ((h, self.hc), (a, self.ac)):
            ax.plot(xs, [min(r["ppda"], 30) if r["ppda"] is not None else np.nan for r in rows],
                    color=colour, lw=4, marker="o", ms=10)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r['start']}–{r['end']}'" for r in h])
        ax.invert_yaxis()
        ax.set_ylabel("PPDA  (higher on the chart = more pressing)", color=TEXT_DIM, fontsize=12)
        ax.grid(axis="y", color=GRID_COL, lw=0.8)
        for i, (tid, name, colour) in enumerate(((self.hid, self.hname, self.hc), (self.aid, self.aname, self.ac))):
            pitch = fig.add_axes([MARGIN + i * 0.46, 0.12, 0.42, 0.36])
            themed_pitch(pitch)
            pitch.set_xlim(48, 102)
            regains = high_regain_events(self.e, tid)
            pitch.scatter(pd.to_numeric(regains.get("x"), errors="coerce"),
                          pd.to_numeric(regains.get("y"), errors="coerce"),
                          s=180, color=colour, edgecolor="white", lw=1.2, zorder=4)
            fig.text(MARGIN + i * 0.46, 0.49, f"{name.upper()}  ·  {len(regains)} HIGH REGAINS",
                     color=colour, fontsize=15, fontweight="bold")
        return self.save(fig, "story_03_press.png")

    def players(self, players: pd.DataFrame | None) -> Path:
        from football_analysis.metrics.vaep import load_model, player_vaep

        table = player_vaep(self.e, players) if load_model() is not None else pd.DataFrame()
        top = table.iloc[0] if len(table) else None
        headline = (f"{surname(str(top['player']))} added the most value: {top['vaep']:+.2f} goals on VAEP"
                    if top is not None else "The players who shaped it")
        fig = self.page(4, "The players", headline)
        from football_analysis.metrics.vaep import action_values
        from football_analysis.visuals.vaep_visuals import draw_split_bars, draw_value_map

        values = action_values(self.e)
        spots = []
        # Two value maps stacked: each side's players where they did their
        # work, discs sized by the value they added.
        ax = fig.add_axes([MARGIN - 0.02, 0.33, 1 - 2 * MARGIN + 0.04, 0.42])
        placed: list = []
        for i, (tid, name, colour) in enumerate(((self.hid, self.hname, self.hc), (self.aid, self.aname, self.ac))):
            spots.append((draw_value_map(ax, values, tid, colour, label_top=5, size_scale=2400,
                                         fontsize=12, flip=bool(i), draw_pitch=(i == 0),
                                         placed=placed), colour))
            team_total = values[values["team_id"] == tid]["vaep"].sum() if len(values) else 0.0
            label = (f"{name.upper()}  {team_total:+.2f}  →" if i == 0
                     else f"←  {name.upper()}  {team_total:+.2f}")
            fig.text(MARGIN if i == 0 else 1 - MARGIN, 0.775, label, color=colour, fontsize=19,
                     fontweight="bold", ha="left" if i == 0 else "right")
        bars = fig.add_axes([MARGIN + 0.12, 0.155, 0.62, 0.12])
        draw_split_bars(bars, spots, n=5, fontsize=12)
        fig.text(MARGIN, 0.128, _wrap("Disc: a player at the average spot of their actions, sized by the value "
                                      "they added (VAEP: how much each action raised the chance of scoring and "
                                      "lowered the chance of conceding). Bars: attack solid, defence faded.", 96),
                 color=TEXT_DIM, fontsize=11, va="top")
        return self.save(fig, "story_04_players.png")

    def takeaways(self, lines: list[str]) -> Path:
        fig = self.page(5, "Takeaways", "What the match told us")
        for i, line in enumerate(lines[:3]):
            y = 0.76 - i * 0.18
            fig.text(MARGIN, y, f"{i + 1}", color=self.hc if i % 2 == 0 else self.ac, va="top", **display(64))
            fig.text(MARGIN + 0.09, y - 0.01, line, color=TEXT_MAIN, va="top", fontsize=21, wrap=True)
        for x, tid, name, colour in ((0.40, self.hid, self.hname, self.hc), (0.60, self.aid, self.aname, self.ac)):
            crests.place_crest(fig, x, 0.19, int(tid), monogram=name[:3].upper(), colour=colour,
                               width=0.10, background=BG_DARK, zorder=10)
        fig.text(0.5, 0.19, self.score, color=TEXT_MAIN, ha="center", va="center", **display(54))
        return self.save(fig, "story_05_takeaways.png")


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width))


def _takeaway_lines(story: Story, extras: dict, table: pd.DataFrame) -> list[str]:
    xp = extras.get("expected_points", {})
    lines = []
    hg, ag = (int(v) for v in str(story.score).replace("—", "-").replace("–", "-").split("-")[:2])
    favoured = story.hname if xp.get("home_xpts", 0) >= xp.get("away_xpts", 0) else story.aname
    winner = story.hname if hg > ag else story.aname if ag > hg else None
    if winner and winner == favoured:
        lines.append(f"{winner} deserved it: the chances were worth {max(xp['home_xpts'], xp['away_xpts']):.1f} "
                     f"expected points to them.")
    elif winner:
        lines.append(f"{winner} won it against the run of the chances: {favoured} had the better of them, "
                     f"{max(xp['home_xpts'], xp['away_xpts']):.1f} expected points to "
                     f"{min(xp['home_xpts'], xp['away_xpts']):.1f}.")
    else:
        lines.append(f"A draw, with {favoured} creating the better chances "
                     f"({max(xp['home_xpts'], xp['away_xpts']):.1f} expected points to "
                     f"{min(xp['home_xpts'], xp['away_xpts']):.1f}).")
    for key, name in (("home", story.hname), ("away", story.aname)):
        sp = extras[key]["set_pieces"]
        if sp["set_piece_goals"]:
            lines.append(f"{name} scored {sp['set_piece_goals']} from dead balls; set pieces were worth "
                         f"{sp['set_piece_xg']:.2f} xG to them.")
            break
    if len(table):
        r = table.iloc[0]
        lines.append(f"{surname(str(r['player']))} was the most valuable player on the pitch "
                     f"({r['vaep']:+.2f} goals added on VAEP).")
    ctl = extras.get("game_control", {})
    if len(lines) < 3 and ctl:
        side = story.hname if ctl.get("home", 0) >= ctl.get("away", 0) else story.aname
        lines.append(f"{side} were the likelier winner for {max(ctl['home'], ctl['away']):.0f}% of the match.")
    return [_wrap(t, 46) for t in lines]


def build_story_posters(events, xg, players, info, out_dir, *, home_color, away_color, score,
                        competition, date, headline) -> list[Path]:
    from football_analysis.metrics.match_extras import match_extras
    from football_analysis.metrics.vaep import load_model, player_vaep

    extras = match_extras(events, info)
    story = Story(events, xg, info, out_dir, home_color, away_color, score, competition, date)
    table = player_vaep(events, players) if load_model() is not None else pd.DataFrame()
    paths = [
        story.cover(_wrap(headline, 34), extras),
        story.attack(),
        story.press(extras),
        story.players(players),
        story.takeaways(_takeaway_lines(story, extras, table)),
    ]
    return paths
