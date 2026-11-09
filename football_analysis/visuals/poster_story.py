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
from matplotlib.patches import FancyBboxPatch
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
                early = float(g["minute"]) < 12  # centred, it sat on the y-axis labels
                ax.annotate(f"{surname(str(g['player']))} {int(g['minute'])}'", (g["minute"], y),
                            xytext=(12 if early else 0, 14), textcoords="offset points",
                            ha="left" if early else "center", color=TEXT_MAIN, fontsize=13, fontweight="bold")
        from football_analysis.visuals.visualization_components import resolve_label_overlaps

        resolve_label_overlaps(ax)
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

    # -- shared drawing helpers ------------------------------------------
    def clean(self, ax):
        ax.set_facecolor((0, 0, 0, 0))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_COL)
        ax.tick_params(colors=TEXT_DIM, labelsize=12)

    def section(self, fig, x, y, text):
        fig.text(x, y, text.upper(), color=TEXT_DIM, fontsize=13, fontweight="bold")

    def mirror_bars(self, fig, rect, rows, fmt="{:g}"):
        """Comparison rows: label in the middle, home bar growing left, away bar growing right."""
        x0, y0, w, h = rect
        step = h / max(len(rows), 1)
        for i, (label, hv, av) in enumerate(rows):
            y = y0 + h - (i + 0.5) * step
            top = max(hv, av, 1e-9)
            half = w / 2 - 0.12
            fig.text(x0 + w / 2, y + step * 0.24, label.upper(), ha="center", va="center", color=TEXT_DIM,
                     fontsize=11, fontweight="bold")
            for value, colour, sign in ((hv, self.hc, -1), (av, self.ac, 1)):
                length = half * value / top
                start = x0 + w / 2 + sign * 0.012
                left = start - length if sign < 0 else start
                fig.add_artist(plt.Rectangle((left, y - step * 0.20), length, step * 0.28,
                                             transform=fig.transFigure, color=colour, alpha=0.9))
                fig.text(start + sign * (length + 0.012), y - step * 0.06, fmt.format(value), color=colour,
                         ha="right" if sign < 0 else "left", va="center", **display(22))

    def half_pitch(self, fig, rect, x_from=60):
        ax = fig.add_axes(rect)
        themed_pitch(ax)
        ax.set_xlim(x_from, 102)
        ax.set_ylim(-2, 102)
        ax.set_aspect(68 / 105)
        return ax

    # -- slide 2: the attack ---------------------------------------------
    def attack(self, extras: dict) -> Path:
        from football_analysis.metrics.match_metrics import box_entry_mask, build_possessions, team_advanced_metrics

        mask = box_entry_mask(self.e)
        annotated, possessions = build_possessions(self.e)
        shot_ids = set(possessions.loc[possessions["shots"].fillna(0) > 0, "possession_id"])
        counts, lanes = {}, {}
        for tid in (self.hid, self.aid):
            own = self.e[mask & (self.e["team_id"] == tid)]
            counts[tid] = len(own)
            y = pd.to_numeric(own["y"], errors="coerce")
            lanes[tid] = pd.Series(np.where(y < 33.3, "right", np.where(y > 66.7, "left", "centre"))).value_counts()
        leader = self.hid if counts[self.hid] >= counts[self.aid] else self.aid
        lname = self.hname if leader == self.hid else self.aname
        top_lane = lanes[leader].idxmax() if len(lanes[leader]) else "centre"
        share = lanes[leader].max() / max(counts[leader], 1)
        fig = self.page(2, "How they attacked",
                        f"{lname} got into the box {counts[leader]} times, {share:.0%} of them down the {top_lane}")
        # Row 1: box entries on the attacking third; the ones that led to a shot lit up.
        for i, (tid, name, colour) in enumerate(((self.hid, self.hname, self.hc), (self.aid, self.aname, self.ac))):
            x0 = MARGIN + i * 0.46
            ax = self.half_pitch(fig, [x0, 0.54, 0.42, 0.25])
            own = self.e[mask & (self.e["team_id"] == tid)]
            led = (annotated.loc[own.index, "possession_id"].isin(shot_ids) if len(own)
                   else pd.Series(dtype=bool))
            for (_, r), hit in zip(own.iterrows(), led):
                ax.annotate("", xy=(r["end_x"], r["end_y"]), xytext=(r["x"], r["y"]),
                            arrowprops=dict(arrowstyle="-|>", color="white" if hit else colour,
                                            lw=2.4 if hit else 1.6, alpha=0.95 if hit else 0.65))
            fig.text(x0, 0.80, f"{name.upper()}  ·  {counts[tid]} ENTRIES  ·  {int(led.sum())} → SHOT",
                     color=colour, fontsize=12, fontweight="bold")
        fig.text(MARGIN, 0.515, "Box entries · white arrows: a shot followed in the same possession",
                 color=TEXT_DIM, fontsize=11)
        # Row 2 left: the attacking funnel.
        self.section(fig, MARGIN, 0.47, "From possession to shot")
        rows = []
        tests = (("Possessions", lambda q: q["team_id"] == q["team_id"]),
                 ("Reached final third", lambda q: q["final_third_entries"].fillna(0) > 0),
                 ("Got into the box", lambda q: q["box_entries"].fillna(0) > 0),
                 ("Box visit ended in a shot", lambda q: (q["box_entries"].fillna(0) > 0) & (q["shots"].fillna(0) > 0)))
        for stage, test in tests:
            hv = int(test(possessions[possessions["team_id"] == self.hid]).sum())
            av = int(test(possessions[possessions["team_id"] == self.aid]).sum())
            rows.append((stage, hv, av))
        self.mirror_bars(fig, (MARGIN, 0.265, 0.43, 0.195), rows)
        # Row 2 right: where the chances came from.
        self.section(fig, 0.53, 0.47, "Where the xG came from")
        ax = fig.add_axes([0.62, 0.30, 0.32, 0.15])
        self.clean(ax)
        shades = {"Open play": 1.0, "Counter": 0.78, "Corner": 0.58, "Free kick": 0.45, "Throw-in": 0.34,
                  "Penalty": 0.25}
        for j, (key, colour) in enumerate((("home", self.hc), ("away", self.ac))):
            left = 0.0
            sit = extras[key]["xg_by_situation"]
            for label, alpha in shades.items():
                v = sit[label]["xG"]
                if v <= 0:
                    continue
                ax.barh(1 - j, v, left=left, color=colour, alpha=alpha, height=0.6, edgecolor=BG_DARK, lw=1.5)
                if v >= 0.4:  # narrow segments stay unlabelled rather than crowd each other
                    ax.text(left + v / 2, 1 - j, label.split()[0], ha="center", va="center", fontsize=9.5,
                            color="#111111" if _light(colour) else "white", fontweight="bold")
                left += v
            ax.text(left + 0.04, 1 - j, f"{left:.2f}", va="center", color=colour, **display(20))
        ax.set_yticks([1, 0])
        ax.set_yticklabels([self.hname, self.aname], color=TEXT_MAIN, fontsize=12)
        ax.set_xlabel("expected goals by situation", color=TEXT_DIM, fontsize=10.5)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.18)
        # Row 3: the numbers behind it.
        tm = team_advanced_metrics(self.e, {"home_id": self.hid, "away_id": self.aid})
        self.section(fig, MARGIN, 0.225, "The numbers behind it")
        self.mirror_bars(fig, (MARGIN, 0.09, 1 - 2 * MARGIN, 0.13), [
            ("Progressive passes", tm["home"]["progressive_passes"], tm["away"]["progressive_passes"]),
            ("Deep completions", tm["home"]["deep_completions"], tm["away"]["deep_completions"]),
        ])
        return self.save(fig, "story_02_attack.png")

    # -- slide 3: the press ----------------------------------------------
    def press(self, extras: dict) -> Path:
        from football_analysis.metrics.match_metrics import build_possessions, high_regain_events, team_advanced_metrics

        def overall(rows):
            p, a = sum(r["passes"] for r in rows), sum(r["actions"] for r in rows)
            return p / a if a else float("nan")

        h, a = extras["home"]["ppda_timeline"], extras["away"]["ppda_timeline"]
        harder = self.hname if overall(h) <= overall(a) else self.aname
        fig = self.page(3, "Without the ball",
                        f"{harder} pressed harder: PPDA {min(overall(h), overall(a)):.1f} "
                        f"against {max(overall(h), overall(a)):.1f}")
        # Row 1: pressing through the match.
        self.section(fig, MARGIN, 0.80, "Pressing through the match  ·  higher = more pressing")
        ax = fig.add_axes([MARGIN + 0.05, 0.605, 1 - 2 * MARGIN - 0.07, 0.175])
        self.clean(ax)
        xs = np.arange(len(h))
        for rows, colour in ((h, self.hc), (a, self.ac)):
            vals = [min(r["ppda"], 30) if r["ppda"] is not None else np.nan for r in rows]
            ax.plot(xs, vals, color=colour, lw=4, marker="o", ms=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r['start']}–{r['end']}'" for r in h])
        ax.set_ylim(32, 0)
        ax.set_ylabel("PPDA", color=TEXT_DIM, fontsize=11)
        ax.grid(axis="y", color=GRID_COL, lw=0.8)
        # Row 2: where each side won the ball high, by what followed.
        annotated, possessions = build_possessions(self.e)
        by_id = possessions.set_index("possession_id")
        for i, (tid, name, colour) in enumerate(((self.hid, self.hname, self.hc), (self.aid, self.aname, self.ac))):
            x0 = MARGIN + i * 0.46
            pitch = self.half_pitch(fig, [x0, 0.30, 0.42, 0.235], x_from=55)
            regains = high_regain_events(self.e, tid).dropna(subset=["x", "y"])
            if len(regains):
                pid = annotated.loc[regains.index, "possession_id"]
                shots = pid.map(by_id["shots"]).fillna(0).to_numpy()
                box = pid.map(by_id["box_entries"]).fillna(0).to_numpy()
            else:
                shots = box = np.array([])
            xs_, ys_ = pd.to_numeric(regains["x"]).to_numpy(), pd.to_numeric(regains["y"]).to_numpy()
            groups = ((shots > 0, "*", 520, "white", colour, 1.0),
                      ((shots == 0) & (box > 0), "s", 170, colour, "white", 0.95),
                      ((shots == 0) & (box == 0), "o", 120, colour, "white", 0.55))
            for sel, marker, size, face, edge, alpha in groups:
                if len(sel) and sel.any():
                    pitch.scatter(xs_[sel], ys_[sel], s=size, marker=marker, color=face, edgecolor=edge,
                                  lw=1.4, alpha=alpha, zorder=4)
            fig.text(x0, 0.548, f"{name.upper()}  ·  {len(regains)} HIGH REGAINS  ·  {int((shots > 0).sum())} → SHOT",
                     color=colour, fontsize=12, fontweight="bold")
        fig.text(MARGIN, 0.278, "Star: a shot followed  ·  square: reached the box  ·  dot: nothing came of it",
                 color=TEXT_DIM, fontsize=11)
        # Row 3: winning it back, and paying for losing it.
        tm = team_advanced_metrics(self.e, {"home_id": self.hid, "away_id": self.aid})
        self.section(fig, MARGIN, 0.24, "Winning it back, and paying for losing it")
        self.mirror_bars(fig, (MARGIN, 0.085, 1 - 2 * MARGIN, 0.15), [
            ("Counter-press regains", tm["home"]["counterpress_regains"], tm["away"]["counterpress_regains"]),
            ("Dangerous counters conceded", tm["home"]["rest_defence_dangerous_counters"],
             tm["away"]["rest_defence_dangerous_counters"]),
            ("Tactical fouls", extras["home"]["tactical_fouls"], extras["away"]["tactical_fouls"]),
        ])
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

    def takeaways(self, lines: list[str], extras: dict, table: pd.DataFrame) -> Path:
        fig = self.page(5, "Takeaways", "What the match told us")
        xp, ctl = extras.get("expected_points", {}), extras.get("game_control", {})
        grey = "#7A7488"
        minis = [
            ("Chances replayed", [(f"{self.hname} win", xp.get("home_win", 0), self.hc),
                                  ("Draw", xp.get("draw", 0), grey),
                                  (f"{self.aname} win", xp.get("away_win", 0), self.ac)]),
            ("Likelier winner, share of the match", [(self.hname, ctl.get("home", 0) / 100, self.hc),
                                                     ("Level", ctl.get("level", 0) / 100, grey),
                                                     (self.aname, ctl.get("away", 0) / 100, self.ac)]),
        ]
        sp = (extras["home"]["set_pieces"], extras["away"]["set_pieces"])
        for i, line in enumerate(lines[:3]):
            if "dead balls" in line and i < len(minis):
                # The chart follows the point it sits under.
                total = max(sp[0]["set_piece_xg"] + sp[1]["set_piece_xg"], 0.01)
                minis[i] = ("Set-piece xG", [(self.hname, sp[0]["set_piece_xg"] / total, self.hc),
                                             (self.aname, sp[1]["set_piece_xg"] / total, self.ac)])
            y = 0.80 - i * 0.205
            fig.add_artist(FancyBboxPatch((MARGIN, y - 0.18), 1 - 2 * MARGIN, 0.19,
                                          boxstyle="round,pad=0,rounding_size=0.015", transform=fig.transFigure,
                                          facecolor=(1, 1, 1, 0.045), edgecolor=(1, 1, 1, 0.12), lw=1))
            fig.text(MARGIN + 0.025, y - 0.012, f"{i + 1}", color=self.hc if i % 2 == 0 else self.ac, va="top",
                     **display(58))
            fig.text(MARGIN + 0.10, y - 0.016, line, color=TEXT_MAIN, va="top", fontsize=16.5)
            mx, mw, my = MARGIN + 0.10, 1 - 2 * MARGIN - 0.13, y - 0.148
            if i < len(minis):
                title, parts = minis[i]
                fig.text(mx, my + 0.036, title.upper(), color=TEXT_DIM, fontsize=10, fontweight="bold")
                left = mx
                for label, share, colour in parts:
                    width = mw * max(share, 0)
                    fig.add_artist(plt.Rectangle((left, my), width, 0.026, transform=fig.transFigure,
                                                 color=colour, alpha=0.95))
                    if share >= 0.14:
                        fig.text(left + width / 2, my + 0.013, f"{label} {share:.0%}", ha="center", va="center",
                                 fontsize=10, fontweight="bold", color="#111111" if _light(colour) else "white")
                    left += width
            elif len(table):
                top = table.head(3)
                fig.text(mx, my + 0.068, "MOST VALUE ADDED (VAEP)", color=TEXT_DIM, fontsize=10, fontweight="bold")
                peak = max(float(top["vaep"].max()), 0.1)
                slot = mw / 3
                for j, r in enumerate(top.itertuples()):
                    colour = self.hc if r.team_id == self.hid else self.ac
                    bx = mx + j * slot
                    width = (slot - 0.15) * max(r.vaep, 0) / peak
                    # Name above its own bar, so long surnames never run into the bar.
                    fig.text(bx, my + 0.04, surname(str(r.player)), color=TEXT_MAIN, va="center", fontsize=11,
                             fontweight="bold")
                    fig.add_artist(plt.Rectangle((bx, my), width, 0.026, transform=fig.transFigure,
                                                 color=colour, alpha=0.95))
                    fig.text(bx + width + 0.008, my + 0.013, f"{r.vaep:+.2f}", color=_readable_on_dark(colour),
                             va="center", fontsize=11, fontweight="bold")
        for x, tid, name, colour in ((0.36, self.hid, self.hname, self.hc), (0.64, self.aid, self.aname, self.ac)):
            crests.place_crest(fig, x, 0.13, int(tid), monogram=name[:3].upper(), colour=colour,
                               width=0.085, background=BG_DARK, zorder=10)
        fig.text(0.5, 0.13, self.score, color=TEXT_MAIN, ha="center", va="center", **display(46))
        return self.save(fig, "story_05_takeaways.png")


def _readable_on_dark(colour) -> str:
    from football_analysis.visuals.cover_art import _readable

    return _readable(colour)


def _light(colour) -> bool:
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(colour)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55


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
    return [_wrap(t, 58) for t in lines]


def build_story_posters(events, xg, players, info, out_dir, *, home_color, away_color, score,
                        competition, date, headline) -> list[Path]:
    from football_analysis.metrics.match_extras import match_extras
    from football_analysis.metrics.vaep import load_model, player_vaep

    extras = match_extras(events, info)
    story = Story(events, xg, info, out_dir, home_color, away_color, score, competition, date)
    table = player_vaep(events, players) if load_model() is not None else pd.DataFrame()
    paths = [
        story.cover(_wrap(headline, 34), extras),
        story.attack(extras),
        story.press(extras),
        story.players(players),
        story.takeaways(_takeaway_lines(story, extras, table), extras, table),
    ]
    return paths
