# Football Match Analysis

A Python post-match football analysis toolkit for WhoScored/Opta match data. The project produces high-resolution tactical visuals on a dark AMOLED theme, defensive summaries, xG/xT analysis, pass networks, and a full tactical PDF report with written analysis for every visual.

Created by Mostafa Saad.

---

## What This Project Does

The tool reads a WhoScored match page, extracts the embedded match event data, calculates internal metrics such as xG and xT, then exports a complete analysis package:

- Dark-theme visual report
- High-resolution PNG visuals
- CSV event/player/xG outputs
- Full `match_report_<timestamp>.pdf`
- Detailed tactical commentary for every visual inside the PDF
- Advanced multi-page player tables, exported as PNG and CSV files

The main analysis script is:

- `Match_Analysis_Dark.py`

---

## Main Features

### Match Data Pipeline

- Scrapes WhoScored match pages.
- Extracts `matchCentreData`.
- Parses events, players, teams, formations, substitutions, shots, passes, defensive actions and match metadata.
- Uses fallback scraping approaches for difficult pages.

### Internal xG Engine

- Shot-level xG model using distance, angle, body part, shot type and contextual qualifiers.
- Support for penalties, set pieces, big chances, cut-backs, rebounds, crosses and through balls.
- Team-level calibration from available match statistics.
- xG flow, xG summary, xGoT-style finishing context and shot breakdown visuals.

### Tactical Visuals

The report now produces a large visual package, including:

- xG Flow
- Shot Maps
- Shot Breakdown and Goals
- Pass Networks
- xT Maps
- Shot Comparison
- Danger Creation
- Goalkeeper Saves
- xG/xGoT Summary
- Zone 14 and Half-Space Maps
- Match Statistics Comparison
- Territorial Control
- Ball Touches
- Pass Map by Third
- xT per Minute
- Progressive Passes
- Cross Maps
- Defensive Heatmaps
- Defensive Summary
- Average Positions
- Dominating Zones
- Box Entries
- High Turnovers
- Pass Target Zones
- Player Ratings and player stat tables
- Grouped summary boards

---

## Recent Updates

### Visual & Report Overhaul (Latest)

A full pass over the pitch visuals, team colours and PDF report:

**Portrait pitches + AMOLED theme**

- All pitch visuals (pass network, average positions, shot map, xT map) are now drawn on **vertical/portrait pitches with the attack pointing up**, on a true-black AMOLED background.

**Pass network**

- Single **team-shirt colour** for every node; **substitutes are shown as squares**, starters as circles — decided by the **starting XI** (read from match flow), so a starter who is subbed *off* stays a circle and only genuine bench players become squares.
- **Connected-core pruning** (StatsBomb/Opta-style) removes stray, link-less nodes from late cameos.
- Neutral **grey → white** link ramp (independent of team colour), distinct **goalkeeper** and busiest-**hub** markers, shirt numbers inside nodes, and de-overlapped player labels with leader lines.

**Shot map**

- Four-way outcome encoding — **goal · saved · blocked · off-target** — plus a **big-chance halo** and a **penalty ring**.
- Six-card metric strip (xG, Goals, Shots, On-Target %, Big Chances, xG/Shot), penalty-area shading, average shot-distance line, and an attacking-half zoom so shots no longer pile up.

**xT map & average positions**

- xT map uses **white positive-pass arrows** for contrast over the heatmap (gold reserved for the top xT actions).
- Average positions use smaller nodes with node-aware label placement and leader lines to stop names overlapping circles.

**Accurate team & national-team colours**

- Home-kit colours reviewed and corrected, with **all 2026 World Cup national teams** covered, plus many clubs from the Primeira Liga, Eredivisie, Scottish Premiership, Süper Lig, Saudi Pro League, Belgian Pro League, Brazil, Argentina, MLS and Egypt.
- A contrast guard guarantees the two teams in a match never render in clashing/identical colours.

**PDF report navigation & analysis**

- New **Match at a Glance** dashboard page (score + home/away split bars for xG, shots, on-target and possession).
- New **Contents page** with page numbers and **clickable PDF bookmarks** for fast navigation.
- New **Glossary & Methodology** page defining xG, xGoT, xT, PPDA, Big Chance, Zone 14 and the pitch markers.
- Section dividers (Shared / Home / Away), and **each visual's commentary now links to the next page**, so the report reads as one connected tactical argument.

### Dynamic Team Kit Colours

- Team colours are now dynamic and based on the selected kit type.
- Configurable kit options:

```python
HOME_KIT_TYPE = "home"
AWAY_KIT_TYPE = "away"
CUSTOM_KIT_COLORS = {}
```

- The same team colours are applied consistently across visuals:
  - xG charts
  - xT maps
  - pass maps
  - pass networks
  - defensive visuals
  - player tables
  - PDF pages

### Substitutes in Visuals

- Substituted players are now included in pass-network and average-position visuals where data exists.
- Player tables distinguish starters, substitutes and unused players more clearly.

### Better Defensive Metrics

- Blocks are now calculated properly instead of always showing zero.
- Defensive summary and match statistics now include:
  - Tackles
  - Interceptions
  - Blocks
  - Clearances
  - Recoveries
  - Fouls

### Shot Breakdown Improvements

- Goals table now includes assist names where available.
- If the direct assist field is missing, the code tries to infer the assister from the previous successful same-team pass/key pass before the goal.
- Goal type now distinguishes `Open Play`, `Set Piece`, `Penalty`, and the finishing body part where available.
- Set pieces are detected from corners, free kicks and throw-ins in the action sequence before the goal.

### PDF Report Redesign

The unified PDF report is generated through `match_extensions.py` as:

```text
output/match_report_<timestamp>.pdf
```

The PDF now includes:

- A cleaner portrait page layout.
- The visual placed at the top of each analysis page.
- Detailed tactical commentary underneath each visual.
- Human-style football analysis rather than short generic notes.
- Higher-quality embedded visuals inside the PDF.
- Player tables first.
- Shared analysis section immediately after player tables.
- Then home-team analysis.
- Then away-team analysis.
- A minimal final page:

```text
End of report by Mostafa Saad
```

### PDF Quality

PDF visual quality has been increased:

```python
PDF_VISUAL_DPI = 320
PDF_PAGE_DPI = 240
```

This improves clarity for:

- Pitch maps
- Arrows
- Small labels
- Player tables
- xT grids
- Pass networks

---

## Project Files

| File | Purpose |
|---|---|
| `Match_Analysis_Dark.py` | Main analysis script (dark AMOLED theme) |
| `match_extensions.py` | Unified PDF report, PPDA, team stats and extended report pages |
| `viz_v2.py` | Shared visual helpers |
| `viz_v2_charts.py` | Main V2 tactical visual functions |
| `viz_design_system.py` | Shared design system helpers |
| `requirements.txt` | Python dependencies |

---

## Installation

Use Python 3.10 or newer.

```bash
git clone https://github.com/mostafasaad91/Football-Match-Analysis.git
cd Football-Match-Analysis
pip install -r requirements.txt
```

Google Chrome is recommended for Selenium/undetected-chromedriver fallback scraping.

---

## Usage

Edit the match URL in the settings section of the script you want to run:

```python
MATCH_URL = "https://www.whoscored.com/matches/XXXXXXX/live/..."
```

Run the analysis:

```bash
python Match_Analysis_Dark.py
```

Outputs are saved in:

```text
output/
```

## Output Structure

Typical output includes:

```text
output/
└── Team_A_vs_Team_B_score/
    ├── events.csv
    ├── players.csv
    ├── xg.csv
    ├── match_report_<timestamp>.pdf
    ├── 1_xg_flow.png
    ├── 2_shot_map_home.png
    ├── 3_shot_map_away.png
    ├── player_stats_home_attacking.csv
    ├── player_stats_away_passing.csv
    ├── ...
    └── visuals/
```

Extended report files may also include:

```text
output/
├── match_report_<timestamp>.pdf
└── goals_log_<timestamp>.csv
```

Generated PNG files are ignored by Git through `.gitignore`.

---

## Configuration

Important settings are available near the top of the main scripts.

| Setting | Description |
|---|---|
| `MATCH_URL` | WhoScored match page URL |
| `SAVE_DIR` | Output directory |
| `OUTPUT_IMAGE_DPI` | PNG export quality |
| `PDF_EXPORT_DPI` | Tactical PDF export quality for the main scripts |
| `HOME_KIT_TYPE` | Home team kit colour choice |
| `AWAY_KIT_TYPE` | Away team kit colour choice |
| `CUSTOM_KIT_COLORS` | Optional manual team colour override |
| `BROWSER_HEADLESS` | Run browser in headless mode |
| `BROWSER_USE_REAL_PROFILE` | Use real Chrome profile if scraping is blocked |

Example manual colour override:

```python
CUSTOM_KIT_COLORS = {
    "home": "#C8102E",
    "away": "#034694",
}
```

---

## GitHub Notes

The repository is prepared so generated files are not committed:

- `output/`
- `*.png`
- `*.pdf`
- `*.csv`
- `__pycache__/`
- `.claude/`
- temporary/cache files

Only source code, documentation and dependency files should be committed.

---

## Disclaimer

This project is for football analysis, research and educational use. It works with publicly available match data from WhoScored/Opta pages. Please respect the data provider's terms of service and rate limits.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
