# Football Match Analysis

A Python post-match football analysis toolkit for WhoScored/Opta match data. The project produces high-resolution tactical visuals, light and dark report versions, player tables, defensive summaries, xG/xT analysis, pass networks, and a full tactical PDF report with written analysis for every visual.

Created by Mostafa Saad.

---

## What This Project Does

The tool reads a WhoScored match page, extracts the embedded match event data, calculates internal metrics such as xG and xT, then exports a complete analysis package:

- Dark-theme visual report
- Light-theme visual report
- High-resolution PNG visuals
- CSV event/player/xG outputs
- Full `match_report_<timestamp>.pdf`
- Detailed tactical commentary for every visual inside the PDF
- Advanced multi-page player tables, exported as PNG and CSV files

The current version includes both:

- `Match_Analysis_Dark.py`
- `Match_Analysis_Light.py`

Both scripts use the same core analysis logic, with theme-specific styling.

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

### Light and Dark Versions

- Added consistent support for both light and dark visual identities.
- The light version now uses a clean white PDF/report style.
- Dark visuals keep the original dark tactical look.
- Text contrast has been improved across light visuals, especially on tables, maps, arrows and donut charts.

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
- The light version no longer uses dark rows in the goals table.

### Advanced Player Tables

- Player tables are now split into readable category pages:
  - General
  - Attacking
  - Defending
  - Passing
  - Duels
  - Goalkeeping
- Category tables are exported as both visuals and CSV files.
- The report keeps player tables as the first PDF section, followed by shared, home-team and away-team analysis.

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
| `Match_Analysis_Dark.py` | Main dark-theme analysis script |
| `Match_Analysis_Light.py` | Main light-theme analysis script |
| `match_extensions.py` | Unified PDF report, PPDA, player tables, team stats and extended report pages |
| `match_extensions_players.py` | Advanced player-table integration and PDF player-table commentary |
| `advanced_player_stats.py` | Fetching and normalising advanced player statistics |
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

Run the dark version:

```bash
python Match_Analysis_Dark.py
```

Run the light version:

```bash
python Match_Analysis_Light.py
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
