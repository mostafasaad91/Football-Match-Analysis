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

To rebuild the complete dark visual package and connected tactical PDF from
the included France vs England sample data, run:

```bash
python visual_redesign_full.py
```

The reproducible sample inputs are versioned under
`sample_data/France_vs_England_4-6/`.

## Output

Generated artifacts are written to:

```text
output/<home>_vs_<away>_<score>/
```

The bundled sample therefore writes to `output/France_vs_England_4-6/`.

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
| `match_metrics.py` | Canonical possession, regain, transition, progression, crossing, touch, territory, and advanced-team metrics. |
| `match_report.py` | Report pages, commentary, PPDA analysis, player-stat tables, PDF assembly, and exports. |
| `player_radar.py` | Player participation, player metrics, xT calculations, and radar-chart exports. |
| `tactical_visualizations.py` | Tactical chart renderers and DataFrame adapters. |
| `visual_redesign_full.py` | Complete AMOLED visual package, player radars, QA dashboards, and connected tactical PDF. |
| `visual_redesign_preview.py` | Shared visual identity, comparison rows, and reusable statistical pages used by the full renderer. |
| `tactical_pdf_report.py` | Connected performance-analyst and data-analyst commentary report. |
| `build_qa_contact_sheets.py` | Eight curated match-story QA dashboards. |
| `visualization_components.py` | Reusable plotting components and visualization primitives. |
| `visualization_design.py` | Shared colors, typography, frames, and readability helpers. |
| `requirements.txt` | Runtime Python dependencies. |

## Metric Definitions

The report uses one shared implementation for team tables, tactical figures,
and player radars so the same metric does not change between pages.

- **Provider recovery** is an explicit `BallRecovery` event supplied by the
  data provider. It is shown separately because it is not a complete measure
  of every change of possession.
- **Possession regain** is inferred when a team establishes controlled
  possession after the opponent. Restarts and administrative events do not
  count as open-play regains.
- **High regain** is an inferred open-play regain at `x >= 60` on the normalized
  0-100 pitch.
- **Attacking transition** begins with an open-play regain and, within the first
  12 seconds of the same possession, either advances at least 20 pitch units,
  reaches the final third or penalty area, or produces a shot. Provider
  `FastBreak` and `CounterAttack` tags are also respected. Restarts are excluded.
- **Counterpress regain** is a regain close to the location of a loss after the
  opponent controlled the ball for no more than five seconds.
- **Counterpress success rate** is counterpress regains divided by eligible
  open-play losses in the same period. The regain must occur within five
  seconds of opponent control and within 15 normalized pitch units of the loss.
- **Progressive pass** is a completed open-play pass that advances at least
  28.6 pitch units within the team's own half, 14.3 units when crossing halfway,
  or 9.5 units within the opposition half.
- **Cross** uses the provider cross flag or qualifier. A geometric fallback is
  used only when the feed has no cross annotation.
- **Field tilt** is each team's share of completed passes ending in the final
  third. It is not labelled as possession.
- **Deep completion** is a completed open-play pass from outside `x = 80` to a
  central target at or beyond `x = 80` (`y = 15-85`).
- **Build-up success rate** is the share of possessions beginning below `x = 33`
  that subsequently reach the final third in the same possession.
- **Final-third entry efficiency** is the share of possessions containing a
  final-third entry that subsequently reach the penalty area.
- **Box-entry-to-shot conversion** is the share of possessions containing a box
  entry that produce a shot before possession changes.
- **Sequence xT** is the sum of positive expected-threat value inside inferred
  possessions. Both total sequence xT and xT per possession are exported.
- **xGChain** credits every player involved in a shot-producing possession with
  that possession's non-penalty xG.
- **xGBuildup** uses the same non-penalty xG credit but excludes the shot taker
  and the key-pass provider, isolating earlier build-up involvement.
- **Directness** is net forward possession progress divided by the total
  successful pass-and-carry distance, expressed as a percentage.
- **Rest-defence vulnerability** is the share of open-play losses after an
  attack reached the final third that allow the opponent a transition box entry
  or shot within 12 seconds. Lower is better.
- **Game-state splits** assign every possession to leading, drawing, or trailing
  from the score immediately before that possession began. Shootout goals are
  excluded and own goals are credited to the benefiting team.
- **Possession share** is based on the duration of inferred possessions; pass
  share remains a separate diagnostic.

The full team set is exported to `team_advanced_metrics.csv`. Player xGChain,
xGBuildup, and sequence-xT involvement are exported to
`player_sequence_metrics.csv` and xGChain/xGBuildup also appear in player radars.

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
