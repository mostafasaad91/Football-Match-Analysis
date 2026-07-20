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

- True-black backgrounds with restrained panel borders.
- Team-specific colours resolved from club and national-team palettes.
- Automatic kit-clash protection when both teams have similar colours.
- Stable, deterministic colours for previously unknown teams.
- Passing links that always use a separate relationship palette from player
  nodes, so network strength cannot be confused with team identity.
- Collision-aware labels with direct player names and leader lines.
- Contrast-aware text on bright heatmap cells and coloured marks.
- Shared headers, metric strips, notes and chart typography.

The production pipeline resolves colours through
`choose_matchup_colors()` in `football_match_analysis.py`. The selected colours
are injected into every downstream visual through the match metadata; they are
not fixed home/away colours.

Live match runs are routed through the same complete AMOLED renderer used by
the reference package (`USE_COMPLETE_AMOLED_PACKAGE = True`). The fixture
identity, score, output names and both team palettes are configured from the
current match before any visual is rendered, preventing new fixtures from
falling back to the legacy visual style.

## Requirements

- Python 3.10 or newer.
- Google Chrome or Chromium for the Selenium collection fallback.
- Internet access when collecting a new WhoScored match.

Install the dependencies from the project directory:

```bash
python -m pip install -r requirements.txt
```

Core dependencies include pandas, NumPy, Matplotlib, SciPy, Rich,
cloudscraper, Beautiful Soup, Selenium, pypdf and PyMuPDF.

## Quick start

### Analyse a new match

1. Open `football_match_analysis.py`.
2. Set `MATCH_URL` to the required WhoScored match URL.
3. Run the main pipeline:

```bash
python football_match_analysis.py
```

For most fixtures, changing `MATCH_URL` is sufficient. The configuration block
also supports browser settings, output options, official-stat fallbacks and kit
selection.

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

## Kit and colour configuration

The default colour mode is automatic and fixture-aware. Optional settings in
`football_match_analysis.py` allow explicit kit selection:

| Setting | Purpose |
| --- | --- |
| `HOME_KIT_TYPE` | Select `home`, `accent`, `away` or `auto` for the home team. |
| `AWAY_KIT_TYPE` | Select `home`, `accent`, `away` or `auto` for the away team. |
| `CUSTOM_KIT_COLORS` | Override either team with a specific hexadecimal colour. |

Automatic mode prioritises the team's real palette, removes near-white or
near-black marks that would disappear on AMOLED, and searches alternate kit
colours when the two teams are visually too similar.

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
| `football_match_analysis.py` | Main entry point, collection fallbacks, parsing, colour resolution and export orchestration. |
| `match_metrics.py` | Canonical possession, transition, progression, territory and advanced-team metrics. |
| `match_report.py` | Report pages, PPDA analysis, player tables and PDF assembly. |
| `tactical_pdf_report.py` | Connected tactical commentary written from the visual evidence. |
| `tactical_visualizations.py` | Metric adapters and legacy-compatible chart helpers. |
| `player_radar.py` | Player participation metrics and pizza/radar exports. |
| `visualization_components.py` | Shared AMOLED chart components and readability helpers. |
| `visualization_design.py` | Global visual tokens, typography and reusable frames. |
| `visual_redesign_full.py` | Unified production AMOLED renderer, PDF package and visual QA build. |
| `visual_redesign_preview.py` | Shared dynamic fixture identity and comparison-page helpers. |
| `build_qa_contact_sheets.py` | Eight curated dashboards that summarise the match story. |
| `tests/` | Metric, substitution, colour and visual-identity regression tests. |

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
