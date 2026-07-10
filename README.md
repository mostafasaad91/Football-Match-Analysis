# Football Match Analysis

A Python toolkit that turns WhoScored/Opta match-event data into a complete
post-match analysis package. It collects the match data, calculates tactical
metrics, renders publication-ready visualizations, and builds a multi-page PDF
report.

Created by **Mostafa Saad**.

## Features

- Multiple data-collection fallbacks: cloudscraper, requests, and Selenium.
- Match, team, and player analysis from a single event stream.
- Expected Goals (xG) and Expected Threat (xT) models implemented locally.
- Shot maps, xG flow, pass networks, heatmaps, territory charts, PPDA, box
  entries, progressive actions, defensive actions, and other tactical views.
- Player radar charts with participation, passing, attacking, threat,
  defensive, and duel metrics.
- Team-specific colors with contrast and kit-clash protection.
- Grouped summary boards for presentation and social publishing.
- A structured PDF report with English, data-driven commentary.
- CSV and PNG exports for further analysis or reuse.

## Requirements

- Python 3.10 or later.
- Google Chrome or Chromium for the Selenium fallback.
- Internet access to retrieve the configured WhoScored match.

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The main dependencies include NumPy, pandas, Matplotlib, SciPy, Rich,
cloudscraper, Beautiful Soup, Selenium, pypdf, and PyMuPDF.

## Configuration

Open `football_match_analysis.py` and update `MATCH_URL` with the required
WhoScored match URL:

```python
MATCH_URL = "https://www.whoscored.com/matches/..."
```

The same settings section also contains the output directory, home and away kit
selection, optional custom kit colors, Chrome profile settings, browser mode,
and official-stat fallback behavior.

For most matches, only `MATCH_URL` needs to change.

## Usage

Run the main entry point from the project directory:

```bash
python football_match_analysis.py
```

The application collects the match data, calculates the metrics, creates the
visualizations, and assembles the report. Network restrictions or changes to
the WhoScored page may cause the application to move through its HTTP and
browser fallback methods.

## Output

Generated artifacts are written to:

```text
output/<home>_vs_<away>_<score>/
```

Depending on the available match data, the directory can contain:

- A full PDF match-analysis report.
- Individual tactical visualization images.
- Grouped dashboard boards.
- Player radar images organized by team.
- Report pages exported as individual images.
- CSV files for events, players, goals, and calculated metrics.

The `output/` directory and generated PDF, PNG, and CSV files are excluded from
Git by default.

## Project Structure

| File | Purpose |
| --- | --- |
| `football_match_analysis.py` | Main entry point, data collection, metric calculation, visualization orchestration, and report generation. |
| `match_report.py` | Report pages, commentary, PPDA analysis, player-stat tables, PDF assembly, and exports. |
| `player_radar.py` | Player participation, player metrics, xT calculations, and radar-chart exports. |
| `tactical_visualizations.py` | Tactical chart renderers and DataFrame adapters. |
| `visualization_components.py` | Reusable plotting components and visualization primitives. |
| `visualization_design.py` | Shared colors, typography, frames, and readability helpers. |
| `requirements.txt` | Runtime Python dependencies. |

## Validation and Code Style

The Python source is formatted with Black using its standard 88-character line
length. Useful local checks are:

```bash
python -m black --check *.py
python -m ruff check *.py --select E9,F63,F7,F82
python -m compileall -q .
```

On PowerShell, pass an expanded file list to Black if the wildcard is not
expanded automatically.

## Method and Limitations

The analysis describes one match and should not be treated as a long-term team
or player evaluation. Positional conclusions are inferred from event locations
and average positions. The xG, xT, and post-shot estimates are transparent local
approximations and do not reproduce proprietary Opta or StatsBomb models.

WhoScored page structure and access controls can change. The fallback collectors
improve resilience, but no scraper can guarantee permanent compatibility with
an external website.

## Data Attribution

WhoScored/Opta is the underlying match-data source. This project independently
processes and visualizes the retrieved event data and is not affiliated with or
endorsed by WhoScored or Opta.

## License

See [LICENSE](LICENSE) for the project license.
