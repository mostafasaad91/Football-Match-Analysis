# ⚽ WhoScored Post-Match Analyzer — Internal xG Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/xG_Model-V7_Internal-orange.svg" alt="xG Model V7"/>
  <img src="https://img.shields.io/badge/Data_Source-WhoScored/Opta-green.svg" alt="Data Source"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
</p>

A comprehensive post-match football analytics tool that scrapes match data from **WhoScored**, computes **Expected Goals (xG)** using an internal logistic regression model, and produces **high-resolution dark-mode visualizations** and **PDF tactical reports** covering every aspect of the game.

> **Created by Mostafa Saad**

---

## 🌟 Features

### Data Acquisition
- **Triple fallback scraping**: Automatically tries `cloudscraper` → `requests + rotating headers` → `undetected-chromedriver + Selenium` to bypass WhoScored's anti-bot protections.
- **matchCentreData extraction**: Parses the embedded JSON from WhoScored's match centre using robust brace-counting for complete extraction.
- **Official stat integration**: Extracts official Opta/WhoScored team statistics from DOM or HTTP when available.

### Internal xG Engine (V7)
- **Shot-level xG model**: Logistic regression based on shot distance, angle, body part, zone, and contextual qualifiers.
- **Context-aware bonuses**: Big Chance, layoff/cutback, through-ball, cross, rebound, direct free kick, and set-piece adjustments.
- **Ensemble blending**: Three model variants (Opta-like, SPADL-like, academic) with weighted averaging.
- **Team-stat calibration**: Optionally calibrates team xG totals using match statistics (shots on target, possession, etc.).
- **Bounded rescaling**: Ensures individual shot xG values are capped and team totals remain realistic.

### Visualizations (11+ Figures)
| # | Visualization | Description |
|---|--------------|-------------|
| 1 | **xG Flow** | Timeline of cumulative xG by minute with goal markers |
| 2 | **Shot Map** | Pitch view of all shots with xG values and outcomes |
| 3 | **Goals Breakdown** | Detailed view of all goals with xG context |
| 4 | **Pass Map** | All passes on a pitch view, color-coded by zone |
| 5 | **Pass Network** | Player passing network with node positioning |
| 6 | **xT Map** | Expected Threat visualization for progressive actions |
| 7 | **Match Report** | Full multi-panel match analysis dashboard |
| 8 | **Grouped Boards** | Category-based visual boards for social sharing |
| 9 | **Tactical PDF** | Multi-page PDF with tactical commentary and visuals |
| 10+ | **Mini Panels** | Individual stat panels (GK saves, crosses, zones, etc.) |

### Team Color System
- **100+ teams** across Top-5 European leagues with official kit colors.
- **Kit-based palette system**: Home, accent, and alternate (away) colors per team.
- **Automatic contrast selection**: `choose_matchup_colors()` ensures readable color pairs for any matchup.
- **Dark-mode optimization**: Low-luminance colors are automatically lifted for visibility on dark backgrounds.

### PDF Tactical Report
- **Cover page** with match info, team colors, and scoreline.
- **Executive summary** with key metrics and tactical commentary.
- **Visual pages** with AI-generated tactical notes for each chart.
- **Match statistics** comparison table.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Google Chrome (for Selenium fallback)
- ChromeDriver (auto-downloaded by `undetected-chromedriver`)

### Installation

```bash
# Clone the repository
git clone https://github.com/mostafasaad91/Football-Match-Analysis.git
cd Football-Match-Analysis

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run with default match URL (edit MATCH_URL in SETTINGS section)
python Match_Analysis.py
```

Or customize the match URL directly in the `SETTINGS` section at the top of the file:

```python
MATCH_URL = "https://www.whoscored.com/matches/XXXXXXX/live/..."
```

All output is saved to the `output/` directory:
- High-resolution PNG images (420 DPI)
- Match events CSV
- Players CSV
- xG summary CSV
- Full tactical PDF report

---

## ⚙️ Configuration

All settings are at the top of the script in the `SETTINGS` section:

| Setting | Default | Description |
|---------|---------|-------------|
| `MATCH_URL` | *(match URL)* | WhoScored match page URL |
| `SAVE_DIR` | `"output"` | Output directory |
| `CHROMEDRIVER_PATH` | `""` | Empty = auto-download |
| `BROWSER_HEADLESS` | `True` | Run Chrome in headless mode |
| `BROWSER_USE_REAL_PROFILE` | `False` | Use real Chrome profile |
| `SHOW_WINDOWS` | `False` | Open interactive matplotlib windows |
| `OUTPUT_IMAGE_DPI` | `420` | Image resolution |
| `PDF_EXPORT_DPI` | `400` | PDF resolution |
| `XG_SINGLE_SHOT_CAP` | `0.78` | Maximum xG for a single shot |
| `XG_PENALTY_VALUE` | `0.76` | xG value for penalties |
| `STRICT_OFFICIAL_PAGE_XG` | `False` | Fail if official xG unavailable |

---

## 🏗️ Architecture

```
Match_Analysis.py
├── SETTINGS & CONSTANTS
│   ├── Match URL & Browser config
│   ├── xG model parameters
│   └── Output settings
├── TEAM COLORS & PALETTES
│   ├── TEAM_COLORS (100+ teams)
│   ├── TOP5_2025_26_TEAM_PALETTES
│   ├── TEAM_ALIASES
│   └── choose_matchup_colors()
├── SCRAPING LAYER
│   ├── _try_cloudscraper()
│   ├── _try_requests()
│   ├── _try_chrome()
│   └── scrape_match()
├── DATA PARSING
│   ├── _extract_match_data()
│   ├── parse_all()
│   └── Official stats extraction
├── xG ENGINE (V7)
│   ├── Shot geometry features
│   ├── Context features & bonuses
│   ├── Logistic regression model
│   ├── Ensemble blending
│   └── Bounded rescaling
├── VISUALIZATIONS
│   ├── draw_xg_flow()
│   ├── draw_shot_map_full()
│   ├── draw_pass_map_full()
│   ├── draw_pass_network_full()
│   ├── draw_xt_map_full()
│   ├── draw_match_report()
│   └── 40+ panel functions
├── PDF REPORT
│   ├── build_tactical_pdf()
│   ├── Tactical commentary generation
│   └── Multi-page layout
└── main()
    ├── Scrape → Parse → xG → Visualize → PDF
    └── Full pipeline orchestration
```

---

## 🧮 xG Model Details

The internal xG engine uses a **logistic regression model** with the following feature pipeline:

1. **Shot Geometry**: Distance to goal center, angle to goal posts, zone classification (box, central box, six-yard box).
2. **Context Qualifiers**: Big chance flag, body part (foot/head), assist type (through ball, cross, cutback, layoff), set piece type.
3. **Logistic Model**: Three variants with different coefficient sets (Opta-like, SPADL-like, academic).
4. **Context Bonuses**: Additional logit adjustments for specific event contexts.
5. **Value Capping**: Individual shot xG capped at `XG_SINGLE_SHOT_CAP` (0.78).
6. **Ensemble**: Weighted average of three model outputs (0.48 / 0.26 / 0.26).
7. **Rescaling**: Bounded rescaling to team-level targets derived from match statistics.

---

## 🎨 Supported Leagues

The team color system covers **100+ clubs** from:

| League | Country |
|--------|---------|
| Premier League | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England |
| La Liga EA Sports | 🇪🇸 Spain |
| Serie A | 🇮🇹 Italy |
| Bundesliga | 🇩🇪 Germany |
| Ligue 1 | 🇫🇷 France |

Each team has **3 color entries**: home kit dominant, accent/stripe, and alternate (away) color for contrast on charts.

---

## 📁 Output Structure

```
output/
├── events_20260430_143052.csv          # All match events
├── players_20260430_143052.csv         # Player statistics
├── xg_20260430_143052.csv             # xG summary
├── xg_flow_20260430_143052.png        # xG timeline
├── shot_map_20260430_143052.png       # Shot map
├── goals_breakdown_20260430_143052.png # Goals analysis
├── pass_map_20260430_143052.png       # Pass map
├── pass_network_20260430_143052.png   # Pass network
├── xt_map_20260430_143052.png         # xT map
├── match_report_20260430_143052.png   # Full report
├── board_*.png                        # Category boards
└── tactical_report_20260430_143052.pdf # PDF report
```

---

## 🛡️ Anti-Blocking Strategy

WhoScored employs aggressive anti-bot protections. This tool uses a **triple fallback** approach:

1. **cloudscraper** — Fastest, no browser needed. Handles basic Cloudflare challenges.
2. **requests + rotating headers** — Tries different User-Agent strings and retry strategies.
3. **undetected-chromedriver + Selenium Stealth** — Full browser automation with real Chrome profile support for the most stubborn protections.

> **Tip**: If all three fail, try opening the match page manually in Chrome first, then re-run the script with `BROWSER_USE_REAL_PROFILE = True`.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This tool is for **educational and analytical purposes only**. It accesses publicly available match data from WhoScored. Please respect WhoScored's Terms of Service and rate limits. The author is not responsible for any misuse of this tool.

---

## 📧 Contact

**Mostafa Saad** — [GitHub](https://github.com/mostafasaad91)

If you find this tool useful, please consider giving it a ⭐!
