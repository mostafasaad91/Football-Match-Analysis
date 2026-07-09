# Football Match Analysis

A Python post-match analysis toolkit for WhoScored/Opta event data. From a single match link it builds a full **dark-theme tactical package**: high-resolution visuals, per-player radar profiles, curated dashboard boards, and a multi-page **PDF report with written, data-driven tactical analysis under every visual** — plus each report page exported as a separate image, ready to post.

Created by **Mostafa Saad**.

---

## What it produces

For every match the pipeline outputs, under `output/<Home>_vs_<Away>_<score>/`:

- **`match_report_<timestamp>.pdf`** — the full tactical report (cover → executive summary → glance → contents → phase sections → **player radars** → glossary → verdict → closing).
- **`report_pages/page_01.png … `** — every PDF page as a standalone image for social posting.
- **`player_radars/<Team>/<Player>.png`** — a radar pizza for every player who took part, sorted into team folders.
- **8 curated dashboard boards** (`board_01…08`) — 2-up summary panels for The Story, Chance Creation, Danger Zones, Build-up, Progression & Threat, Defence & Pressing, Territory & Control, Pressing & Regains.
- **~40 individual tactical visuals** (shot maps, xG flow, pass networks, xT maps, defensive heatmaps, high-turnovers, …).
- **CSV outputs** — events, players, xG, goals log.

---

## Highlights

### Written tactical analysis (not just charts)
Every visual in the PDF carries a **connected, data-driven commentary in an analyst voice** — computed from that match's own numbers, not a generic template:
- Names the key players (top distributor, creator, shooter, defender) with their figures.
- Reads a **qualitative team shape** from average positions and the pass network (how high, how compact, how wide, which channel the play leaned through).
- Cross-references the metrics into one argument (e.g. possession vs. penetration, chance quality vs. volume, press as suppression vs. creation).

### Player radar profiles
A 28-metric **pizza chart per player**, grouped and colour-coded:
- **Attack** · **Passing** · **Threat (expected)** · **Defence** · **Duels**
- Bar length = percentile vs. every player in the match; chip = raw value (passes/long-balls as completed/total, shots as on-target/total, duels as won/contested).
- Minutes shown under the player's name; a **tactical read** is written beneath each radar in the report.

### From-scratch analytics
- **Expected Threat (xT):** a transparent, from-scratch grid value model (Karun-Singh style) with documented probabilities, transition matrix and value iteration. An open-family approximation — **not** a reproduction of proprietary Opta/StatsBomb possession-value.
- **Participation & minutes:** starter / substitute / unused reconstructed from substitution and red-card events; minutes on the official clock (regulation 90 / 120, stoppage excluded).
- **Creation model:** xA, assists, big-chances-created and shot-creating actions reconstructed by linking each shot back to the key pass that made it.
- **Duels** reported as absolute won/total for ground, aerial and overall; **xGOT** as a placement-based post-shot proxy.

### Robust rendering
- Pass networks keep participating **substitutes** on the map.
- Dashboard boards retry saving at lower DPI so all eight always persist under memory pressure.

---

## Requirements

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

Key libraries: `numpy`, `pandas`, `matplotlib`, `scipy`, `cloudscraper` / `requests` / `beautifulsoup4` (scraping), `undetected-chromedriver` / `selenium` (browser fallback), `pypdf` (merge), `pymupdf` (per-page image export).

A Chrome/Chromium install is used for the Selenium fallback when the HTTP scrape can't reach the match data.

---

## Usage

Run the main script and follow the prompt for the WhoScored match URL:

```bash
python Match_Analysis_Dark.py
```

The tool scrapes the match page, extracts the embedded event data, computes the internal metrics, and writes the full package to `output/<match folder>/`.

---

## Project layout

| File | Role |
|------|------|
| `Match_Analysis_Dark.py` | Main entry point — scraping, metrics, ~40 visuals, dashboard boards, orchestration |
| `match_extensions.py` | PDF report assembly, analyst commentary, team-shape read, verdict/executive pages, player-radar report section, per-page export |
| `player_radar.py` | Per-player radar pizzas, grid-xT model, participation/minutes, duels and creation models |
| `viz_v2_charts.py` | Individual v2 tactical charts (shot maps, pass networks, xT maps, heatmaps, …) |
| `viz_v2.py`, `viz_design_system.py` | Shared v2 chart chrome and design system |

---

## Method & limitations

All figures are computed directly from a **single match's event stream** and are best read as description, not verdict. The xT surface and the xGOT proxy are open, transparent approximations from the same family as public models — they do **not** reproduce proprietary Opta possession-value or StatsBomb OBV outputs. Team-shape and any positional inference come from **average touch positions** and are approximate by nature.

---

## Attribution

Personal analysis project by Mostafa Saad. WhoScored/Opta are the underlying data source; this toolkit only reads and visualises publicly rendered match data.
