# ⚽ WhoScored Post-Match Analyzer

> **Automated football match analysis** — scrapes WhoScored data, calibrates an Opta-style xG model with the official totals, and renders 37 dark-mode tactical figures plus a fully-formatted multi-page PDF tactical report.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Matplotlib](https://img.shields.io/badge/Matplotlib-darkmode-black)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Output Visualizations](#-output-visualizations-37-figures)
- [Category Summary Boards](#-category-summary-boards)
- [Tactical PDF Report](#-tactical-pdf-report)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Anti-Blocking Strategy](#-anti-blocking-strategy)
- [Official Stats & xG Calibration](#-official-stats--xg-calibration)
- [xG Model](#-xg-model)
- [Project Structure](#-project-structure)
- [Credits](#-credits)

---

## 🔍 Overview

**WhoScored Post-Match Analyzer** is a single-file Python toolkit (`Analyzer.py`) that automatically fetches match event data from [WhoScored.com](https://www.whoscored.com), recovers the official Opta totals embedded in the page, calibrates a built-in xG model against them, and produces a comprehensive suite of tactical charts — all in a dark-mode aesthetic — together with a compiled multi-page PDF tactical report.

Designed for **football analysts, data journalists, and tactical content creators** who want professional-grade visuals without manual data entry.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 Smart Scraping | 3-tier anti-blocking system (cloudscraper → requests → undetected Chrome stealth) |
| 📊 37 Visualizations | Shot maps, heatmaps, pass networks, xT, defensive actions, dominating zones, box entries, high turnovers, pass target zones, and more |
| 🧩 4 Summary Boards | Grouped collages: Match Overview · Attacking · Build-up · Defensive/Territory |
| 📄 PDF Report | Multi-page tactical PDF (cover + executive summary + per-visual commentary) generated automatically |
| 🎨 Dynamic Team Colours | Auto-selects each team's official kit colour for every chart, with contrast-aware fallbacks for clashing palettes |
| ⚡ xG Model | Built-in StatsBomb/Soccermatics-style open-event xG (distance, angle, body part, situation) |
| 🎯 Official Calibration | Pulls real Opta totals (xG, shots, on target, possession, etc.) from `matchCentreData` and rescales the local model to match |
| 🥅 Own Goals | Automatically attributed and coloured for the benefiting team |
| ⏱️ Period Awareness | Handles Regular Time, Extra Time, and Penalty Shootouts |
| 🌍 Top-5 League Palettes | Built-in 2025/26 kit palettes for Premier League, LaLiga, Serie A, Bundesliga, and Ligue 1 |
| 💾 CSV Exports | Events, players, and per-team xG totals saved alongside every run |

---

## 📊 Output Visualizations (37 Figures)

Filenames are timestamped and prefixed with the figure number — figure numbers 9 and 10 are intentionally skipped (legacy slots), so the run produces 37 figures across the 1–39 range.

### Individual Analytics (Figs 1–8)

| # | Figure |
|---|---|
| 1 | xG Flow (cumulative xG timeline + match events) |
| 2 | Shot Map — Home |
| 3 | Shot Map — Away |
| 4 | Breakdown & Goals (shot bars + goals/assists table) |
| 5 | Pass Network — Home |
| 6 | Pass Network — Away |
| 7 | xT Map — Home |
| 8 | xT Map — Away |

### Standalone Visuals (Figs 11–32)

| # | Figure | # | Figure |
|---|---|---|---|
| 11 | Shot Comparison (both teams) | 12 | Danger Creation — Home |
| 13 | Danger Creation — Away | 14 | Goalkeeper Saves |
| 15 | xG / xGoT / On Target tiles | 16 | Zone 14 & Half-Spaces — Home |
| 17 | Zone 14 & Half-Spaces — Away | 18 | Match Statistics |
| 19 | Territorial Control | 20 | Ball Touches / Possession |
| 21 | Pass Map by Third — Home | 22 | Pass Map by Third — Away |
| 23 | xT per Minute | 24 | Progressive Passes — Home |
| 25 | Progressive Passes — Away | 26 | Crosses — Home |
| 27 | Crosses — Away | 28 | Defensive Heatmap — Home |
| 29 | Defensive Heatmap — Away | 30 | Defensive Summary |
| 31 | Average Positions — Home | 32 | Average Positions — Away |

### Advanced Analytics (Figs 33–39)

| # | Figure |
|---|---|
| 33 | Dominating Zone (touch-share map) |
| 34 | Box Entries — Home |
| 35 | Box Entries — Away |
| 36 | High Turnovers — Home |
| 37 | High Turnovers — Away |
| 38 | Pass Target Zones — Home |
| 39 | Pass Target Zones — Away |

> ℹ️ The legacy *Shot Summary Tiles* visual was removed because it could rebuild buckets that diverged from the official Opta totals. Shot summaries are now driven directly from the calibrated official numbers.

---

## 🧩 Category Summary Boards

After the 37 figures are saved, the analyzer composes **4 grouped collage boards** for quick social-media or presentation use:

1. **Match Overview** — score story, xG flow, key tiles
2. **Attacking** — shots, danger zones, crosses, progressive passes
3. **Build-up** — pass networks, pass thirds, pass target zones, xT
4. **Defensive / Territory** — defensive heatmaps, dominating zone, territorial control, high turnovers

---

## 📄 Tactical PDF Report

The script automatically compiles a polished, print-ready PDF containing:

- A **cover page** with score, venue, competition, and credits
- An **executive summary** with the calibrated xG, shots, possession, and key differentials
- One **commentary page per visual**, pairing the rendered figure with an automatically generated tactical note (uses thresholds on xT, box entries, progressive carries, dominant lanes, etc.)
- **Page header / footer** with the matchup and "Analysis by Mostafa Saad"

Output filename:

```
output/match_analysis_report_<Home>_vs_<Away>_<timestamp>.pdf
```

---

## 🛠 Installation

### Prerequisites
- Python 3.9+
- Google Chrome (only required if cloudscraper + requests both fail and the Chrome fallback is enabled)
- Optional: a ChromeDriver matching your Chrome version (otherwise `undetected-chromedriver` downloads one automatically)

### Install Dependencies

```bash
pip install cloudscraper undetected-chromedriver selenium scipy \
            beautifulsoup4 numpy pandas matplotlib rich
```

Optional (extra stealth for the Chrome fallback):

```bash
pip install selenium-stealth
```

---

## ⚙️ Configuration

Open `Analyzer.py` and edit the **SETTINGS** block near the top of the file:

```python
# ── Match URL ────────────────────────────────────────────────────
MATCH_URL = "https://www.whoscored.com/matches/XXXXXXX/live/..."

# ── Output directory ─────────────────────────────────────────────
SAVE_DIR = "output"

# ── ChromeDriver path (only needed for the Chrome fallback) ──────
CHROMEDRIVER_PATH = ""    # leave empty to let undetected-chromedriver auto-resolve

# ── Your real Chrome profile (optional, used only as a la