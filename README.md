# ⚽ WhoScored Post-Match Analyzer

> **Automated football match analysis** — scrapes WhoScored data and generates 39 professional tactical visualizations + a full PDF report.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Matplotlib](https://img.shields.io/badge/Matplotlib-darkmode-black)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Output Visualizations](#-output-visualizations-39-figures)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Anti-Blocking Strategy](#-anti-blocking-strategy)
- [xG Model](#-xg-model)
- [Project Structure](#-project-structure)
- [Credits](#-credits)

---

## 🔍 Overview

**WhoScored Post-Match Analyzer** is a Python script that automatically fetches match event data from [WhoScored.com](https://www.whoscored.com) and produces a comprehensive suite of tactical charts — all in a dark-mode aesthetic — plus a compiled PDF tactical report.

Designed for **football analysts, data journalists, and tactical content creators** who want professional-grade visuals without manual data entry.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 Smart Scraping | 3-tier anti-blocking system (cloudscraper → requests → Chrome stealth) |
| 📊 39 Visualizations | Shot maps, heatmaps, pass maps, xT, defensive actions, and more |
| 📄 PDF Report | Full tactical PDF compiled automatically after all figures are generated |
| 🎨 Dark Mode | Professional dark-mode styling across all 39 figures |
| ⚡ xG Model | Built-in Opta-style xG calculator (distance + angle + situational modifiers) |
| 🔄 Own Goals | Automatically colored to the benefiting team |
| ⏱️ Period Awareness | Handles Regular Time, Extra Time, and Penalty Shootouts |

---

## 📊 Output Visualizations (39 Figures)

### Individual Analytics (Figs 1–8)
| # | Figure |
|---|---|
| 1 | Shot Map — Both Teams |
| 2 | xG Timeline |
| 3 | Pass Network — Home |
| 4 | Pass Network — Away |
| 5 | Player Heatmap — Home |
| 6 | Player Heatmap — Away |
| 7 | xT Map (Expected Threat) |
| 8 | Full Match Report |

### Standalone Visuals (Figs 9–32)
| # | Figure | # | Figure |
|---|---|---|---|
| 9 | Shot Map — Home | 10 | Shot Map — Away |
| 11 | Pass Network — Home | 12 | Pass Network — Away |
| 13 | Danger Zone — Home | 14 | Danger Zone — Away (GK Saves) |
| 15 | xG / xGoT / On Target | 16 | Zone 14 & Half-Spaces — Home |
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
| 33 | Dominating Zone |
| 34 | Box Entries — Home |
| 35 | Box Entries — Away |
| 36 | High Turnovers — Home |
| 37 | High Turnovers — Away |
| 38 | Pass Target Zones — Home |
| 39 | Pass Target Zones — Away |

---

## 🛠 Installation

### Prerequisites
- Python 3.9+
- Google Chrome (for the fallback scraping method)
- ChromeDriver matching your Chrome version

### Install Dependencies

```bash
pip install cloudscraper undetected-chromedriver selenium scipy \
            beautifulsoup4 numpy pandas matplotlib rich
```

Optional (for better stealth scraping):
```bash
pip install selenium-stealth
```

---

## ⚙️ Configuration

Open `Analyzer.py` and edit the **SETTINGS** block at the top:

```python
# ── Match URL ────────────────────────────────────────────────────
MATCH_URL = "https://www.whoscored.com/matches/XXXXXXX/live/..."

# ── Output directory ─────────────────────────────────────────────
SAVE_DIR = "output"

# ── ChromeDriver path (only needed for fallback) ─────────────────
CHROMEDRIVER_PATH = r"C:\path\to\chromedriver.exe"

# ── Your real Chrome profile (keeps cookies / login) ────────────
CHROME_PROFILE_DIR  = r"C:\Users\YourName\AppData\Local\Google\Chrome\User Data"
CHROME_PROFILE_NAME = "Default"   # or "Profile 1", etc.
```

> **Tip — Find your Chrome profile path:**
> ```powershell
> (Get-Item "$env:LOCALAPPDATA\Google\Chrome\User Data").FullName
> ```

---

## 🚀 Usage

```bash
python Analyzer.py
```

The script will:
1. Try to fetch the match page (3 automatic attempts)
2. Parse all event data from the embedded JSON
3. Generate 39 dark-mode figures saved to `output/`
4. Compile a PDF tactical report

All output files are timestamped:
```
output/
├── 09_shot_map_home_20260309_153045.png
├── 10_shot_map_away_20260309_153045.png
├── ...
└── tactical_report_20260309_153045.pdf
```

---

## 🛡️ Anti-Blocking Strategy

WhoScored actively blocks automated access. The script uses a **3-tier fallback** system:

```
Attempt 1 — cloudscraper
  └── Bypasses Cloudflare automatically, no browser needed (fastest)

Attempt 2 — requests + rotating headers
  └── Session-based with human-like delays and realistic User-Agent pool

Attempt 3 — undetected_chromedriver + Chrome stealth
  └── Opens your real Chrome profile (with your cookies/login)
      + CDP webdriver fingerprint removal
      + selenium-stealth applied if installed
      + Human-like scroll simulation
```

If all three methods fail, the script prints actionable troubleshooting steps.

---

## 📐 xG Model

The built-in xG calculator uses an **Opta-style logistic regression** approach:

- **Primary features:** Euclidean distance to goal centre + goal-mouth angle (radians)
- **Situational modifiers:**
  | Situation | Effect |
  |---|---|
  | Penalty | Fixed 0.76 |
  | Header | −35% base |
  | Big Chance | +0.12–0.20 |
  | Counter Attack | +8% |
  | Direct Free Kick | −15% |

**EPL calibration targets:**
- Penalty kick → ~0.76
- 6-yard tap-in → ~0.60
- Box edge centre → ~0.10
- Header from 6 yards → ~0.35

---

## 📁 Project Structure

```
whoscored-analyzer/
│
├── Analyzer.py          # Main script (scraper + visualizations + PDF)
├── README.md            # This file
├── requirements.txt     # Python dependencies
│
└── output/              # Generated figures and reports (gitignored)
    ├── *.png
    └── *.pdf
```

---

## 📝 Credits

**Developed by Mostafa Saad**

- Data source: [WhoScored.com](https://www.whoscored.com)
- xG model calibrated against Opta EPL published benchmarks
- Visualization palette inspired by professional football analytics studios

> ⚠️ **Disclaimer:** This tool is for personal and educational use only. Automated scraping of WhoScored may violate their Terms of Service. Use responsibly.
