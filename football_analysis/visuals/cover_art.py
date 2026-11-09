"""The article and PDF cover: one page that makes someone open the document.

Approved in the November 2026 review, merging three concepts:

    atmosphere   floodlights in each club's colour from the top corners, and
                 both crests blown up behind them as soft watermarks
    score        the result in huge condensed figures filled with a gradient
                 from one club's colour to the other's, crests either side
    hero         the match seen from the main camera: a pitch in perspective
                 with every shot standing on it as a column of light, taller
                 for a better chance, goals labelled

then the headline, the standfirst, three stat cards and the publisher's mark.
A4 portrait at 200 dpi. Everything is read from the fixture: colours, crests,
shots, score, headline. Dark pages use the Midnight palette; the light theme
gets the same composition on paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch, Polygon
from matplotlib.textpath import TextPath

from football_analysis.visuals import crests
from football_analysis.visuals.typography import display, display_family
from football_analysis.visuals.visualization_components import (
    BG_DARK, IS_LIGHT_THEME, TEXT_DIM, TEXT_MAIN,
)

W, H, DPI = 8.27, 11.69, 180
COVER_NAME = "cover_art.png"
INK = "#1B1B1F" if IS_LIGHT_THEME else "#FFFFFF"


def _truthy(s: pd.Series) -> pd.Series:
    return s.fillna(False) if s.dtype == bool else s.astype(str).str.lower().isin({"true", "1", "yes"})


class _Light:
    """All the soft lights of the page composited once, then drawn as one image.

    Each light used to be its own full-page imshow, and at 200 dpi every one
    of them was resampled to the whole canvas in float64: on a machine with
    little memory free the cover died with MemoryError.
    """

    def __init__(self, n=420, aspect=1.414):
        self.n, self.aspect = n, aspect
        self.rgb = np.zeros((int(n * aspect), n, 3))
        self.a = np.zeros((int(n * aspect), n))

    def add(self, cx, cy, colour, radius, alpha):
        yy, xx = np.mgrid[0:1:complex(0, self.a.shape[0]), 0:1:complex(0, self.n)]
        d = np.sqrt((xx - cx) ** 2 + ((yy - cy) / self.aspect) ** 2) / radius
        a = np.clip(1 - d, 0, 1) ** 2.2 * alpha
        # "over" compositing, colour premultiplied
        self.rgb = self.rgb * (1 - a[..., None]) + np.array(to_rgb(colour)) * a[..., None]
        self.a = self.a + a * (1 - self.a)

    def draw(self, ax, z=1):
        img = np.zeros(self.a.shape + (4,), dtype=np.float32)
        safe = np.maximum(self.a, 1e-6)[..., None]
        img[..., :3] = np.clip(self.rgb / safe, 0, 1)
        img[..., 3] = self.a
        ax.imshow(img, extent=(0, 1, 0, 1), origin="lower", aspect="auto", zorder=z, interpolation="bilinear")


def _watermark(fig, team_id, x, y, width, alpha):
    img = crests.crest_image(int(team_id), allow_download=False)
    if img is None:
        return
    from PIL import Image, ImageFilter

    big = Image.fromarray(img).resize((img.shape[1] * 10, img.shape[0] * 10), Image.LANCZOS)
    big = np.asarray(big.filter(ImageFilter.GaussianBlur(3))).copy()
    big[..., 3] = (big[..., 3] * alpha).astype(np.uint8)
    h = width * fig.get_figwidth() / fig.get_figheight()
    ax = fig.add_axes([x - width / 2, y - h / 2, width, h], zorder=1)
    ax.imshow(big, interpolation="antialiased", aspect="auto")
    ax.set_axis_off()


class Cover:
    def __init__(self, events, xg, info, home_color, away_color, headline, standfirst):
        self.info = info
        self.hid, self.aid = info.get("home_id"), info.get("away_id")
        self.hname, self.aname = str(info.get("home_name")), str(info.get("away_name"))
        self.hc, self.ac = home_color, away_color
        score = str(info.get("score") or "0 : 0").replace("–", ":").replace("-", ":")
        self.hs, self.as_ = [s.strip() for s in score.split(":")[:2]]
        self.headline, self.standfirst = headline, standfirst
        shots = events[_truthy(events.get("is_shot", pd.Series(False, index=events.index)))]
        if "is_penalty_shootout" in shots.columns:
            shots = shots[~_truthy(shots["is_penalty_shootout"])]
        if "is_own_goal" in shots.columns:
            shots = shots[~_truthy(shots["is_own_goal"])]
        self.shots = shots
        self.xg = {str(r["team"]): float(r["xG"]) for _, r in xg.iterrows()} if "team" in xg.columns else {}

    # ── the score ──────────────────────────────────────────────────────
    def score(self, fig, rect):
        fp = FontProperties(family=display_family(), weight="bold")
        tax = fig.add_axes(rect, zorder=8)
        tax.set_axis_off()
        path = TextPath((0, 0), f"{self.hs}–{self.as_}", size=100, prop=fp)
        e = path.get_extents()
        tax.set_xlim(e.x0 - 2, e.x1 + 2)
        tax.set_ylim(e.y0 - 2, e.y1 + 2)
        tax.set_aspect("equal")
        mid = "#FFFFFF" if not IS_LIGHT_THEME else "#F2E6DA"
        cmap = LinearSegmentedColormap.from_list("score", [self.hc, mid, self.ac])
        grad = np.linspace(0, 1, 512)[None, :]
        im = tax.imshow(np.vstack([grad] * 8), cmap=cmap, extent=(e.x0, e.x1, e.y0, e.y1), aspect="auto", zorder=3)
        clip = PathPatch(path, transform=tax.transData, facecolor="none", edgecolor="none")
        tax.add_patch(clip)
        im.set_clip_path(clip)
        halo = PathPatch(path, facecolor="none", edgecolor=INK, lw=0.8, alpha=0.5, zorder=4)
        halo.set_path_effects([pe.withStroke(linewidth=10, foreground=INK, alpha=0.06)])
        tax.add_patch(halo)

    # ── the pitch ──────────────────────────────────────────────────────
    def pitch(self, fig, ax):
        top_y, bot_y, top_w, bot_w, cx = 0.625, 0.33, 0.50, 0.94, 0.5

        def P(px, py):
            tt = (py / 100.0) ** 0.78  # the far half foreshortened
            return cx + (px / 100.0 - 0.5) * (bot_w + (top_w - bot_w) * tt), bot_y + (top_y - bot_y) * tt

        def depth(py):
            return 1.0 - 0.45 * (py / 100.0) ** 0.78

        n = 60
        for i in range(n):
            y0, y1 = 100 * i / n, 100 * (i + 1) / n
            f = i / n
            turf = ((0.80 - 0.05 * f, 0.88 - 0.05 * f, 0.82 - 0.05 * f) if IS_LIGHT_THEME else
                    (0.035 - 0.01 * f, 0.105 - 0.04 * f, 0.085 - 0.02 * f))
            ax.add_patch(Polygon([P(0, y0), P(100, y0), P(100, y1), P(0, y1)], color=turf, lw=0, zorder=3, alpha=0.95))
        for k in range(0, 12, 2):
            ax.add_patch(Polygon([P(k * 100 / 12, 0), P((k + 1) * 100 / 12, 0), P((k + 1) * 100 / 12, 100),
                                  P(k * 100 / 12, 100)], color=INK, alpha=0.022, lw=0, zorder=3))
        line_colour = "#FFFFFF" if not IS_LIGHT_THEME else "#5A6B60"

        def line(pts, lw=1.1, a=0.55):
            xy = np.array([P(*p) for p in pts])
            ax.plot(xy[:, 0], xy[:, 1], color=line_colour, alpha=a, lw=lw, zorder=4, solid_capstyle="round")

        line([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)], lw=1.4)
        line([(50, 0), (50, 100)])
        th = np.linspace(0, 2 * np.pi, 120)
        line(list(zip(50 + 9.15 * np.cos(th) * 100 / 105, 50 + 9.15 * np.sin(th) * 100 / 68)))
        for x0, s in ((0, 1), (100, -1)):
            line([(x0, 21.1), (x0 + s * 16.5, 21.1), (x0 + s * 16.5, 78.9), (x0, 78.9)])
            line([(x0, 36.8), (x0 + s * 5.5, 36.8), (x0 + s * 5.5, 63.2), (x0, 63.2)])
            arc = np.linspace(-0.93, 0.93, 40)
            line(list(zip(x0 + s * (11 + 9.15 * np.cos(arc) * 100 / 105), 50 + 9.15 * np.sin(arc) * 100 / 68)), a=0.4)
            gx1, gy1 = P(x0, 44.6)
            gx2, gy2 = P(x0, 55.4)
            hgt = 0.022 * depth(50)
            ax.plot([gx1, gx1, gx2, gx2], [gy1, gy1 + hgt, gy2 + hgt, gy2], color=line_colour, lw=1.8, zorder=6)
        for g, colour in ((P(100, 50), self.hc), (P(0, 50), self.ac)):
            ax.add_patch(Ellipse(g, 0.16, 0.035, color=colour, alpha=0.16, lw=0, zorder=4))
        rows = []
        for tid, col, flip in ((self.hid, self.hc, False), (self.aid, self.ac, True)):
            for _, r in self.shots[self.shots["team_id"] == tid].iterrows():
                x, y = float(r["x"]), float(r["y"])
                if flip:
                    x, y = 100 - x, 100 - y
                rows.append((y, x, col, float(pd.to_numeric(r.get("xG"), errors="coerce") or 0.0),
                             str(r.get("is_goal")).lower() == "true", r))
        labels = []
        for y, x, col, v, goal, r in sorted(rows, key=lambda t: -t[0]):  # far to near
            bx, by = P(x, y)
            d = depth(y)
            h = (0.02 + 0.30 * np.sqrt(max(v, 0.0))) * d
            ax.add_patch(Ellipse((bx, by), 0.026 * d, 0.008 * d, color=col, alpha=0.6, lw=0, zorder=5))
            steps = 14
            core = INK if goal else col
            for k in range(steps):  # the column fades as it rises
                ya, yb = by + h * k / steps, by + h * (k + 1) / steps
                a = (1 - k / steps) ** 1.4
                ax.plot([bx, bx], [ya, yb], color=col, lw=13 * d, alpha=0.10 * a, zorder=6, solid_capstyle="butt")
                ax.plot([bx, bx], [ya, yb], color=core, lw=(3.6 if goal else 2.8) * d, alpha=0.95 * a + 0.05,
                        zorder=7, solid_capstyle="butt")
            if goal:
                ax.scatter([bx], [by + h], s=180 * d, color="white", edgecolor=col, lw=2, zorder=8)
                name = str(r.get("player") or "").split()[-1].upper() if str(r.get("player") or "").strip() else ""
                labels.append((bx, by + h, f"{name}  {int(r['minute'])}'", col))
        placed: list[tuple[float, float]] = []
        for bx, top, text, col in labels:
            ty = top + 0.02
            while any(abs(bx - px) < 0.22 and abs(ty - py) < 0.025 for px, py in placed):
                ty += 0.03
            placed.append((bx, ty))
            fig.text(min(max(bx, 0.16), 0.84), ty, f"  GOAL  ·  {text}  ", color="white", ha="center", va="center",
                     fontsize=10, fontweight="bold", zorder=9,
                     bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.9", fc=col, ec="white", lw=1.0, alpha=0.95))
        fig.text(0.5, 0.29, "EVERY SHOT AS A COLUMN OF LIGHT  ·  TALLER = BETTER CHANCE  ·  "
                 f"{self.hname.upper()} ATTACK →   ← {self.aname.upper()} ATTACK", color=TEXT_DIM, fontsize=6.8,
                 ha="center", fontweight="bold")

    # ── the cards ──────────────────────────────────────────────────────
    def cards(self, fig, ax, y):
        sh, sa = self.shots[self.shots["team_id"] == self.hid], self.shots[self.shots["team_id"] == self.aid]
        big = lambda s: int(_truthy(s["big_chance"]).sum()) if "big_chance" in s.columns else 0  # noqa: E731
        items = [("EXPECTED GOALS", f"{self.xg.get(self.hname, 0):.2f}", f"{self.xg.get(self.aname, 0):.2f}"),
                 ("SHOTS", f"{len(sh)}", f"{len(sa)}"),
                 ("BIG CHANCES", f"{big(sh)}", f"{big(sa)}")]
        w, gap, hgt = 0.27, 0.025, 0.078
        glass = (0, 0, 0, 0.04) if IS_LIGHT_THEME else (1, 1, 1, 0.05)
        edge = (0, 0, 0, 0.14) if IS_LIGHT_THEME else (1, 1, 1, 0.14)
        for i, (lab, hv, av) in enumerate(items):
            x = 0.07 + i * (w + gap)
            mid = x + w / 2
            ax.add_patch(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0,rounding_size=0.014",
                                        facecolor=glass, edgecolor=edge, lw=0.8, zorder=5))
            ax.plot([x + 0.02, mid], [y + hgt, y + hgt], color=self.hc, lw=2.2, zorder=6, solid_capstyle="round")
            ax.plot([mid, x + w - 0.02], [y + hgt, y + hgt], color=self.ac, lw=2.2, zorder=6, solid_capstyle="round")
            fig.text(mid, y + hgt - 0.02, lab, color=TEXT_DIM, fontsize=7, fontweight="bold", ha="center", zorder=7)
            ax.plot([mid, mid], [y + 0.012, y + 0.042], color=edge, lw=0.8, zorder=6)
            fig.text(mid - 0.065, y + 0.013, hv, color=self.hc, ha="center", zorder=7, **display(24))
            fig.text(mid + 0.065, y + 0.013, av, color=self.ac, ha="center", zorder=7, **display(24))

    # ── the page ───────────────────────────────────────────────────────
    def render(self, path: Path) -> Path:
        import textwrap

        fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG_DARK)
        ax = fig.add_axes([0, 0, 1, 1], zorder=0)
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        strength = 0.6 if IS_LIGHT_THEME else 1.0
        light = _Light()
        light.add(-0.05, 0.86, self.hc, 0.85, 0.50 * strength)
        light.add(1.05, 0.86, self.ac, 0.85, 0.42 * strength)
        if not IS_LIGHT_THEME:
            light.add(0.5, 0.48, "#2A3B8F", 0.75, 0.30)
        light.add(0.5, 0.80, INK, 0.24, 0.10 if not IS_LIGHT_THEME else 0.05)
        light.draw(ax)
        _watermark(fig, self.hid, 0.03, 0.84, 0.70, 0.11)
        _watermark(fig, self.aid, 0.97, 0.84, 0.70, 0.09)
        beam = "#FFFFFF" if not IS_LIGHT_THEME else "#000000"
        for x0 in (0.0, 1.0):
            ax.add_patch(Polygon([(x0, 1.0), (0.30, 0.33), (0.70, 0.33)], color=beam,
                                 alpha=0.035 if not IS_LIGHT_THEME else 0.02, lw=0, zorder=2))
        pax = fig.add_axes([0, 0, 1, 1], zorder=3)
        pax.set_axis_off()
        pax.set_xlim(0, 1)
        pax.set_ylim(0, 1)
        self.pitch(fig, pax)
        comp = str(self.info.get("competition") or "").upper()
        rnd = str(self.info.get("round_name") or "")
        rnd = f"MATCHWEEK {rnd}" if rnd.isdigit() else rnd.upper()
        strap = "  ·  ".join(v for v in (comp, rnd, str(self.info.get("date") or "")) if v)
        fig.text(0.5, 0.962, strap, color=INK, alpha=0.85, fontsize=8.5, fontweight="bold", ha="center")
        crests.place_logo(fig, 0.5, 0.925, width=0.05, background=BG_DARK)
        for x, tid, name, colour in ((0.15, self.hid, self.hname, self.hc), (0.85, self.aid, self.aname, self.ac)):
            crests.place_crest(fig, x, 0.80, int(tid), monogram=name[:3].upper(), colour=colour, width=0.17,
                               background=BG_DARK, zorder=10)
            fig.text(x, 0.715, name.upper(), color=INK, ha="center", zorder=10, **display(18))
        self.score(fig, [0.29, 0.725, 0.42, 0.15])
        rule = np.linspace(0, 1, 256)[None, :]
        rax = fig.add_axes([0.30, 0.268, 0.40, 0.0025], zorder=10)
        rax.set_axis_off()
        rax.imshow(np.vstack([rule] * 2), aspect="auto",
                   cmap=LinearSegmentedColormap.from_list("rule", [self.hc, INK, self.ac]))
        fig.text(0.5, 0.247, "M A T C H   S T U D Y", color=INK, alpha=0.75, fontsize=8.5, fontweight="bold",
                 ha="center", zorder=10)
        title = "\n".join(textwrap.wrap(self.headline, 44))
        fig.text(0.5, 0.232, title, color=INK, va="top", ha="center", zorder=10, linespacing=1.0,
                 **display(31 if "\n" not in title else 26))
        if self.standfirst:
            lines = textwrap.wrap(self.standfirst, 92)[:2]
            top = (0.172 if "\n" not in title else 0.150) + (0.006 if len(lines) > 1 else 0)
            fig.text(0.5, top, "\n".join(lines), color=TEXT_DIM, fontsize=10 if len(lines) == 1 else 9.5,
                     style="italic", va="top", ha="center", zorder=10, linespacing=1.25)
        self.cards(fig, ax, 0.058)
        crests.place_logo(fig, 0.095, 0.035, width=0.045, background=BG_DARK)
        fig.text(0.13, 0.035, "MOSTAFA SAAD  ·  TACTICAL", color=TEXT_MAIN, fontsize=8, fontweight="bold", va="center")
        fig.text(0.93, 0.035, "MATCH STUDY  ·  REAL EVENT DATA", color=TEXT_DIM, fontsize=7, ha="right", va="center")
        fig.savefig(path, dpi=DPI, facecolor=BG_DARK)
        plt.close(fig)
        return path


def build_cover_art(events, xg, info, out_dir, *, home_color, away_color, headline, standfirst="") -> Path | None:
    """Render the cover into ``out_dir``; None if anything it needs is missing."""
    try:
        cover = Cover(events, xg, info, home_color, away_color, headline, standfirst)
        return cover.render(Path(out_dir) / COVER_NAME)
    except Exception as error:  # a cover is never worth a failed package
        print(f"  ! cover art not drawn — {type(error).__name__}: {error}")
        return None
