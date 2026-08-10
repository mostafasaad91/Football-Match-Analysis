# Football Match Analysis

An end-to-end Python workflow for transforming WhoScored/Opta event data into
a publication-ready post-match analysis package. The project combines data
engineering, advanced football metrics, tactical visualisation and structured
report writing in one reproducible pipeline.

The output is designed to read like the work of a performance analyst and a
data analyst: each chart answers a tactical question, the metrics share one
canonical implementation, and the PDF connects the evidence into a coherent
match story.

Created and maintained by **Mostafa Saad**.

## What the project produces

A single run can generate:

- A detailed multi-page tactical PDF report.
- Individual team and match visualisations in a pure-black AMOLED identity.
- Post-match attacking and defensive summary dashboards.
- Eight curated QA/story dashboards covering the main match narrative.
- Player pizza/radar profiles organised by team.
- Reusable CSV exports for events, players and calculated metrics.
- A dedicated output directory named after the fixture.

Generated files are written to:

```text
output/<home>_vs_<away>_<score>/
```

For the included sample, the destination is:

```text
output/France_vs_England_4-6/
```

## Analysis coverage

The visual package includes, when supported by the source data:

- Shot maps, shot quality, shot outcomes, xG, xGoT and xG flow.
- Expected Threat (xT) heatmaps and the top 10 threat-creating passes.
- Pass networks split by half, pass maps and pass distribution by third.
- Average positions, territorial control and field tilt.
- Progressive passes, deep completions, final-third entries and box entries.
- Zone 14, half-space access, crosses and transition threat.
- PPDA, defensive actions, high regains and counterpress outcomes.
- Rest-defence vulnerability, dangerous counters and transition exposure.
- Player contribution profiles, xGChain, xGBuildup and sequence xT.

## Visual system

The project uses a consistent AMOLED design system across all report pages:

- True-black backgrounds with white pitch markings and restrained panel borders.
- Real home-kit colours for roughly 975 clubs and national teams, resolved from
  `team_palettes.py`. When two kits clash, or one fails the contrast floor
  against pure black, the renderer substitutes a readable variant rather than
  shipping two indistinguishable sides.
- White is reserved for all exact values and decisive highlights; neutral low-priority paths are
  thin and dashed instead of competing with the main evidence.
- Single-team maps keep that team's role colour for every main mark. Outcome
  and event type are separated by line style, opacity and marker shape.
- Passing links that always use a separate relationship palette from player
  nodes, so network strength cannot be confused with team identity.
- Collision-aware labels with direct player names and leader lines.
- Contrast-aware text on bright heatmap cells and coloured marks.
- Shared headers, metric strips, notes and chart typography.

The production pipeline resolves both sides once through
`choose_matchup_colors()` in `football_match_analysis.py`, and every surface —
PNGs, player radars, QA dashboards, the tactical PDF — reads that single
decision, so a team never changes colour between pages of one report.

A team name that matches several palette entries is refused rather than
guessed; the unresolved name is reported instead of silently rendering the
wrong club's colours.

Live match runs are routed through the same complete AMOLED renderer used by
the reference package (`USE_COMPLETE_AMOLED_PACKAGE = True`). The fixture
identity, score and output names are configured from the current match before
any visual is rendered, preventing new fixtures from
falling back to the legacy visual style.

## Requirements

- Python 3.10 or newer.
- Internet access when collecting a new WhoScored match.
- Google Chrome or Chromium only as the last collection fallback. Collection
  now tries `curl-cffi` first, which impersonates a browser's TLS and HTTP/2
  fingerprint without launching one; a typical run no longer starts Chrome at
  all.

Install the dependencies from the project directory:

```bash
python -m pip install -r requirements.txt
```

Core dependencies include pandas, NumPy, Matplotlib, SciPy, Rich, curl-cffi,
cloudscraper, Beautiful Soup, Selenium, reportlab, pypdf and PyMuPDF.

## Quick start

### Analyse a new match

Point `MATCH_ANALYSIS_URL` at the fixture and run the pipeline:

```powershell
$env:MATCH_ANALYSIS_URL = "https://www.whoscored.com/matches/1903428/live"
python football_match_analysis.py
```

Without that variable the run falls back to the default URL in the file, which
is the bundled sample — not the match you meant. Set it every time.

`MATCH_URL` in `football_match_analysis.py` remains the fallback, so editing
the file still works if you prefer it. The configuration block also covers
browser settings, output options and official-stat fallbacks.

### Find the URL without a browser

The stored season calendars under `data/fixtures/` carry the WhoScored match id
for every fixture in the Premier League, La Liga and Serie A, so the URL is a
lookup rather than a search:

```bash
python fixtures.py arsenal --next
python fixtures.py --on 2026-08-22
python fixtures.py "aston villa" --last --url
```

Chained into a run:

```powershell
$env:MATCH_ANALYSIS_URL = (python fixtures.py arsenal --last --url)
python football_match_analysis.py
```

An ambiguous name is refused rather than resolved to whichever club sorts
first — `real` reports the four candidates instead of picking Real Madrid.

The calendar is a committed file, not a live feed: a postponed match keeps its
id but not its listed date.

### Rebuild the included sample

The repository includes a reproducible France vs England dataset under:

```text
sample_data/France_vs_England_4-6/
```

Generate the complete visual package, QA dashboards and connected PDF with:

```bash
python visual_redesign_full.py
```

Rebuild only the curated QA contact sheets with:

```bash
python build_qa_contact_sheets.py
```

## Match history

Every run appends its metrics to `output/match_history.db` and archives the
untouched provider payload under `output/raw_snapshots/`. One match is a sample
of one; the history is what makes a claim about a team rather than about an
afternoon.

```bash
python team_history.py matches
python team_history.py team Arsenal --last 6
python team_history.py team Arsenal --last 6 --summary
python team_history.py player "Bukayo Saka" --last 5
python team_history.py export Arsenal --last 10 --out arsenal_last10.csv
```

A fixture is keyed on its provider id, so re-analysing a match replaces its row
instead of double-counting it. The fallback key is competition, season and the
two teams — deliberately not the date, so a postponement does not split one
fixture into two.

Because the raw payloads are kept, a metric added today can be backfilled
across every match already collected without going back to the network:

```bash
python team_history.py replay
```

Percentiles stay silent below ten stored matches rather than dressing noise up
as a ranking.

## Colour configuration

Real kit colours are the default. Set `MATCH_ANALYSIS_TEAM_COLORS` to change
that for a run:

| Value | Behaviour |
| --- | --- |
| `kit` (default) | Each side takes its real home-kit colour from `team_palettes.py`. |
| `roles` | The former fixed pair — home `#2F5BFF`, away `#FFD400` — regardless of fixture. |

```powershell
$env:MATCH_ANALYSIS_TEAM_COLORS = "roles"
```

`team_palettes.py` is organised by domestic league rather than by a single
season's continental entry list, so it does not go stale when a club drops out
of Europe. Hand-picked entries in the main module win over the bulk import.

## Processing pipeline

```text
WhoScored match page
        ↓
HTTP / browser collection fallbacks
        ↓
Canonical event and player tables
        ↓
Possession and advanced-metric engine
        ↓
Team visuals + player radars + summary dashboards
        ↓
Tactical commentary and PDF assembly
        ↓
output/<match_name>/
```

The same metric functions feed the tables, visualisations and report text. This
prevents the same statistic from changing between pages.

## Key metric definitions

| Metric | Definition used by the project |
| --- | --- |
| Possession regain | Controlled possession established after the opponent; restarts and administrative events are excluded. |
| High regain | Open-play regain at `x >= 60` on the normalised 0-100 pitch. |
| Attacking transition | A possession that begins with an open-play regain and quickly advances, enters the final third/box, or produces a shot. |
| Counterpress success | A regain within five seconds of opponent control and within 15 pitch units of the loss. |
| Fouls committed | In paired WhoScored feeds, only the offender's `Unsuccessful` foul row is counted; the opponent's mirrored `Successful` foul-won row is excluded. |
| Defensive blocks | Blocked shots taken by the opponent, detected from `shot_whoscored_type`, `shot_category`, or the exact `Blocked` qualifier, then attributed to the defending team. |
| Progressive pass | A completed open-play pass meeting the distance threshold for its starting zone. |
| Field tilt | Share of completed passes ending in the final third; it is not labelled as possession. |
| Deep completion | Completed open-play pass into the central deep-attacking zone from outside it. |
| Build-up success | Share of possessions beginning below `x = 33` that reach the final third. |
| Box-entry-to-shot | Share of box-entry possessions that produce a shot before possession changes. |
| Sequence xT | Sum of positive expected-threat contribution inside inferred possessions. |
| xGChain | Non-penalty xG credited to every player involved in the shot-producing possession. |
| xGBuildup | xGChain credit excluding the shooter and key-pass provider. |
| Directness | Net forward progress divided by successful pass-and-carry distance. |
| Rest-defence vulnerability | Share of advanced open-play losses that allow, within 12 seconds, a transition shot, box entry, or a 40+ metre break reaching the final third. Lower is better. |

The complete team metric set is exported to `team_advanced_metrics.csv`.
Player sequence metrics are exported to `player_sequence_metrics.csv` and also
feed the player radar pages.

## Project structure

| File | Responsibility |
| --- | --- |
| `football_match_analysis.py` | Main entry point, collection fallbacks, parsing, fixed-role colour mapping and export orchestration. |
| `match_metrics.py` | Canonical possession, transition, progression, territory and advanced-team metrics. |
| `match_report.py` | Report pages, PPDA analysis, player tables and PDF assembly. |
| `tactical_pdf_report.py` | Connected tactical commentary written from the visual evidence. |
| `tactical_visualizations.py` | Metric adapters and legacy-compatible chart helpers. |
| `player_radar.py` | Player participation metrics and pizza/radar exports. |
| `visualization_components.py` | Shared AMOLED chart components and readability helpers. |
| `visualization_design.py` | Global visual tokens, typography and reusable frames. |
| `visual_redesign_full.py` | Unified production AMOLED renderer, PDF package and visual QA build. |
| `visual_redesign_preview.py` | Shared fixture identity and fixed-palette comparison-page helpers. |
| `build_qa_contact_sheets.py` | Eight curated dashboards that summarise the match story. |
| `team_palettes.py` | Real home-kit colours for roughly 975 clubs and national teams, organised by league. |
| `match_store.py` | SQLite match history and the gzipped raw-payload archive. |
| `team_history.py` | Command-line reader for the stored history, including snapshot replay. |
| `fixtures.py` | Season-calendar lookup that resolves a fixture to its WhoScored URL. |
| `data/fixtures/` | Committed EPL, La Liga and Serie A calendars with WhoScored match ids. |
| `scripts/freeze_golden.py` | Re-freezes the reference output the golden test compares against. |
| `tests/` | Metric, substitution, colour, visual-identity and end-to-end golden regression tests. |

## Validation

Run the automated test suite:

```bash
python -m pytest -q
```

Useful static checks:

```bash
python -m compileall -q .
python -m black --check *.py
python -m ruff check *.py --select E9,F63,F7,F82
```

The suite includes an end-to-end golden test: the whole metric engine is run
over the committed France vs England events and every published column is
compared against a frozen reference. The unit tests prove each definition is
implemented as written; the golden test proves the assembled pipeline still
produces the numbers it produced before.

When a metric changes on purpose, read the drift first and then accept it:

```bash
python -m pytest tests/test_metrics_golden.py   # names every column that moved
python scripts/freeze_golden.py                 # rewrite the reference
git diff tests/golden/                          # review the new numbers
```

Visual changes should also be checked in the exported PNG and PDF files. The
automated tests protect calculations and identity rules, but they cannot fully
replace final-context review of label spacing, contrast and page composition.

## Data and model limitations

- The report analyses one match; it is not a replacement for a multi-match
  performance sample.
- Average positions describe the mean location of recorded actions, not a
  continuous tracking-data formation.
- Local xG, xT and post-shot estimates are transparent approximations and do
  not reproduce proprietary Opta or StatsBomb models.
- Event-provider schemas and access controls can change. Collection fallbacks
  improve resilience but cannot guarantee permanent compatibility.
- Tactical interpretation remains evidence-led analysis, not ground truth
  about a coach's intention.

## Data attribution

WhoScored/Opta is the underlying event-data source. This project independently
processes and visualises the retrieved data and is not affiliated with or
endorsed by WhoScored, Opta or Stats Perform.

Use collected data in accordance with the provider's terms and applicable law.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and contribution
guidance. When adding a new metric, define it once in the canonical metric layer
and reuse that implementation in visuals, exports and report commentary.

## License

See [LICENSE](LICENSE).
