<div align="center">

<img src="assets/logo.jpg" width="150" alt="Tactical — Football Data & Analysis">

# Football Match Analysis

**One WhoScored URL in. A 49-visual, 80-page tactical report out.**

An end-to-end Python pipeline that turns Opta event data into a post-match
analysis package: 30 advanced metrics, a pure-black visual identity built from
real kit colours, a written tactical report, and a twelve-post thread of the
whole match story.

Created and maintained by **Mostafa Saad**.

</div>

---

## What one run produces

```powershell
$env:MATCH_ANALYSIS_URL = "https://www.whoscored.com/matches/1873304/live"
python football_match_analysis.py
```

| Output | Detail |
| --- | --- |
| **49 visuals** | Shot maps, pass networks by half, xT surfaces, pressing maps, pitch control, goal-frame keeper plots |
| **80-page PDF** | Every visual with a written tactical reading underneath it, in one analyst voice |
| **12 thread sheets** | Four visuals each, ordered as a narrative — built to be posted one per tweet |
| **Player radars** | A full role profile per player, coloured to their team |
| **CSV exports** | Events, players and every calculated metric |
| **Match history** | Appended to a SQLite database, with the raw payload archived for replay |

Everything lands in `output/<home>_vs_<away>_<score>/`.

---

## Quick start

```bash
python -m pip install -r requirements.txt
```

Point the analyser at a fixture and run it:

```powershell
$env:MATCH_ANALYSIS_URL = "https://www.whoscored.com/matches/1873304/live"
python football_match_analysis.py
```

Without that variable the run falls back to the default URL in the file, which
is not the match you meant. Set it every time.

### Finding the URL without a browser

`data/fixtures/` ships the season calendars for the Premier League, La Liga and
Serie A with each fixture's WhoScored id, so the URL is a lookup:

```bash
python fixtures.py arsenal --next
python fixtures.py --on 2026-08-22
python fixtures.py "aston villa" --last --url
```

Chained:

```powershell
$env:MATCH_ANALYSIS_URL = (python fixtures.py arsenal --last --url)
python football_match_analysis.py
```

An ambiguous name is refused with its candidates rather than resolved to
whichever club sorts first — `real` reports four matches instead of quietly
picking Real Madrid and analysing the wrong game.

### Rebuild the bundled sample

A France vs England dataset is committed under `sample_data/`, so the full
visual package can be produced with no network:

```bash
python visual_redesign_full.py
```

---

## The thread

The twelve contact sheets are written to be posted in order, one per post, and
to carry the match on their own without the report around them:

| | | | |
| --- | --- | --- | --- |
| 01 The Result | 02 The Rhythm | 03 The Shots | 04 The Territory |
| 05 The Networks | 06 How They Passed | 07 Progression | 08 Central Access |
| 09 Into the Box | 10 The Press | 11 Defending | 12 The Difference |

Each sheet holds four visuals with a line under each saying what it answers.
Forty-eight of the forty-nine visuals appear; the omissions are listed in code
with the visual that already tells their story.

---

## Metrics

Thirty metric functions, each defined once in `match_metrics.py` and reused by
the visuals, the exports and the report text — so a number cannot disagree with
itself between two pages.

| Metric | Definition used here |
| --- | --- |
| **Possession regain** | Controlled possession established after the opponent; restarts and administrative events excluded |
| **High regain** | Open-play regain at `x >= 60` on the normalised 0–100 pitch |
| **Attacking transition** | A possession beginning with an open-play regain that quickly advances, enters the final third or box, or produces a shot |
| **Counterpress success** | A regain within five seconds of losing the ball and within 15 pitch units of the loss |
| **Field tilt** | Share of completed passes ending in the final third — territory, not possession |
| **PSxG** | Post-shot expected goals, weighting the chance by how far the placement pulled the keeper |
| **Progressive pass** | A completed open-play pass meeting the distance threshold for its starting zone |
| **Deep completion** | Completed open-play pass into the central deep-attacking zone from outside it |
| **Build-up success** | Share of possessions beginning below `x = 33` that reach the final third |
| **Box-entry-to-shot** | Share of box-entry possessions producing a shot before possession changes |
| **Sequence xT** | Sum of positive expected-threat contribution inside inferred possessions |
| **xGChain** | Non-penalty xG credited to every player involved in the shot-producing possession |
| **xGBuildup** | xGChain credit excluding the shooter and the key-pass provider |
| **Directness** | Net forward progress divided by successful pass-and-carry distance |
| **Rest-defence vulnerability** | Share of advanced open-play losses allowing, within 12 seconds, a transition shot, box entry, or a 40m+ break reaching the final third. Lower is better |
| **Pitch control** | Distance-decayed influence surface, split into held and genuinely contested space |
| **Action value** | Every action priced in goals from a zone-value surface — an explicit model, not a fitted VAEP |

Full team metrics export to `team_advanced_metrics.csv`; player sequence
metrics to `player_sequence_metrics.csv`.

---

## Match history

Every run appends to `output/match_history.db` and archives the untouched
provider payload under `output/raw_snapshots/`. One match is a sample of one;
the history is what makes a claim about a team rather than about an afternoon.

```bash
python team_history.py matches
python team_history.py team Arsenal --last 6
python team_history.py team Arsenal --last 6 --summary
python team_history.py player "Bukayo Saka" --last 5
python team_history.py export Arsenal --last 10 --out arsenal_last10.csv
```

A fixture is keyed on its provider id, so re-analysing a match replaces its row
rather than double-counting it. The fallback key is competition, season and the
two teams — deliberately not the date, so a postponement does not split one
fixture into two.

Because the raw payloads are kept, a metric added today can be backfilled
across every match already collected without going back to the network:

```bash
python team_history.py replay
```

Percentiles stay silent below ten stored matches rather than dressing noise up
as a ranking.

---

## Visual identity

- Pure black grounds with white pitch markings.
- **Real home-kit colours** for roughly 975 clubs and national teams, resolved
  from `team_palettes.py`. When two kits clash, or one fails the contrast floor
  against black, the renderer substitutes a readable variant rather than
  shipping two indistinguishable sides.
- A team name matching several palette entries is **refused, not guessed** —
  the unresolved name is reported instead of rendering the wrong club's colours.
- Both sides are resolved once and every surface reads that decision, so a team
  never changes colour between pages of one report.
- Player-network labels are placed against every other node and penalised for
  running off the pitch, so a name never lands on a neighbour's marker.

Set `MATCH_ANALYSIS_TEAM_COLORS` to change the mode:

| Value | Behaviour |
| --- | --- |
| `kit` (default) | Each side in its real home-kit colour |
| `roles` | The former fixed pair — home `#2F5BFF`, away `#FFD400` |

---

## Requirements

- Python 3.10+
- Internet access when collecting a new match
- Chrome or Chromium **only** as the last collection fallback — collection tries
  `curl-cffi` first, which impersonates a browser's TLS and HTTP/2 fingerprint
  without launching one, so a typical run never starts a browser

---

## Project structure

| File | Responsibility |
| --- | --- |
| `football_match_analysis.py` | Entry point, collection fallbacks, parsing, colour resolution, export orchestration |
| `match_metrics.py` | The thirty canonical metric implementations |
| `match_report.py` | Report pages, PPDA analysis, player tables, PDF assembly |
| `tactical_pdf_report.py` | Cover, tactical commentary and page chrome |
| `tactical_visualizations.py` | Metric adapters and chart helpers |
| `visual_redesign_full.py` | The production renderer and its 49 visuals |
| `visual_redesign_preview.py` | Shared fixture identity and page furniture |
| `player_radar.py` | Player role profiles |
| `visualization_components.py` | Shared chart components and readability helpers |
| `visualization_design.py` | Visual tokens, typography, reusable frames |
| `build_qa_contact_sheets.py` | The twelve thread sheets |
| `team_palettes.py` | Kit colours for ~975 clubs and national teams |
| `match_store.py` | SQLite history and the gzipped payload archive |
| `team_history.py` | Command-line reader for the stored history |
| `fixtures.py` | Season-calendar lookup from fixture to WhoScored URL |
| `scripts/freeze_golden.py` | Re-freezes the reference the golden test compares against |

---

## Validation

```bash
python -m pytest -q
```

The suite includes an **end-to-end golden test**: the whole metric engine runs
over the committed France vs England events and every published column is
compared against a frozen reference. The unit tests prove each definition is
implemented as written; the golden test proves the assembled pipeline still
produces the numbers it produced before.

Other tests check things that render perfectly and are still wrong — a label
that lands on a marker, a panel row drawn through the heading below it, a
verdict sentence that contradicts the numbers printed beside it.

When a metric changes on purpose, read the drift, then accept it:

```bash
python -m pytest tests/test_metrics_golden.py
python scripts/freeze_golden.py
git diff tests/golden/
```

---

## Limitations

- The report analyses one match. It is not a multi-match performance sample.
- Average positions describe the mean location of recorded actions, not
  continuous tracking data.
- Local xG, xT and post-shot estimates are transparent approximations and do
  not reproduce proprietary Opta or StatsBomb models.
- Action value uses an explicit zone-value surface, not a trained VAEP model.
- Event-provider schemas and access controls change. The collection fallbacks
  improve resilience but cannot guarantee permanent compatibility.
- Tactical interpretation is evidence-led analysis, not ground truth about a
  coach's intention.

---

## Data attribution

WhoScored/Opta is the underlying event-data source. This project independently
processes and visualises the retrieved data and is not affiliated with or
endorsed by WhoScored, Opta or Stats Perform. Use collected data in accordance
with the provider's terms and applicable law.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). When adding a metric, define it once in
`match_metrics.py` and reuse that implementation in the visuals, the exports and
the report text.

## License

See [LICENSE](LICENSE).
