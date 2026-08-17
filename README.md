<div align="center">

<img src="assets/logo.jpg" width="150" alt="Tactical — Football Data & Analysis">

# Football Match Analysis

**One WhoScored URL in. A 53-visual, 80-page tactical report out.**

An end-to-end Python pipeline that turns Opta event data into a post-match
analysis package: 30 advanced metrics, a written tactical report opening on
artwork drawn from the match itself, four posters built to be published the
moment the whistle goes, and the whole thing rendered twice — once on black,
once on paper.

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
| **53 visuals** | Shot maps, pass networks by half, xT surfaces, pressing maps, pitch control, goal-frame keeper plots |
| **80-page PDF** | Every visual with a written tactical reading underneath it, in one analyst voice — opening on a cover drawn from the match itself |
| **4 match posters** | Every panel drawn natively at poster scale, club crests included — built to be posted the moment the whistle goes |
| **Player radars** | A full role profile per player, coloured to their team; the report and the article carry the five per side that mattered |
| **Publishable article** | A `.docx` argued from the fixture's own numbers, with every visual and a reading under each |
| **Light copy** | The same package again on `#F5F5F5`, in `light/` |
| **CSV exports** | Events, players and every calculated metric |
| **Match history** | Appended to a SQLite database, with the raw payload archived for replay |

Everything lands in `output/<home>_vs_<away>_<score>/`.

### Two pages

The identity is AMOLED black. A light "Ink & Petrol" palette renders the same
package on `#F5F5F5`, so a fixture can be published on whichever page suits
where it is going. Set `MATCH_ANALYSIS_LIGHT_COPY=0` to skip it.

The theme is read when `visualization_components` is first imported and every
renderer copies its colours into module constants there and then, so one
process renders one page. The light copy is therefore rendered by a child
process — `render_light.py`, which any finished match folder can be handed:

```bash
python render_light.py output/PSG_vs_Aston_Villa_2-1
```

A colour that reads on one page frequently does not read on the other, so the
contrast lift moves kit colours *away from whichever page they are drawn on* —
brightening PSG's navy on black, darkening Manchester City's sky blue on paper.
Written for the black page it only searched upward, which drove City's `#6CABDD`
and Juventus' `#DCE3EC` to pure white against `#F5F5F5`.

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

## The posters

Four 1640x2048 boards, sized to the tallest frame a timeline shows without
cropping, each complete on its own:

| | Left and right columns | Centre |
| --- | --- | --- |
| **1 · Post-match report** | Passing shape, threat zones, defensive work | Sixteen indicators, shot map, game control |
| **2 · How it was played** | Box entries, progression, pressing | Pitch control, zone dominance, sequence leaders |
| **3 · The transition game** | Ball losses, delivery from wide, zone 14 | Game-state splits, twelve transition and press indicators, touch distribution |
| **4 · The final ball** | Shots, restarts, passing profile | The goal frame, the possession-to-goal funnel, twelve shooting indicators |

Forty indicators across the three tables, and no indicator appears on two of
them — a test enforces it, because a repeated cell is a cell the match did not
get.

Every panel is drawn straight onto the poster canvas from the event frame.
An earlier version composited the already-rendered PNGs into a contact sheet,
which scaled each one to a thumbnail and took its type down with it.

---

## Branding

The publisher's badge and both club crests appear on every visual, on all four
posters, and on the report's cover.

Crests come from the provider's own CDN, addressed by the same team id the
event data already carries, so no name matching is involved — the mapping that
usually breaks the moment a club is written "Wolves" in one source and
"Wolverhampton Wanderers" in another. They are cached under `assets/crests`,
and that directory is untracked: club crests are trademarks, fine to render on
an analysis board with attribution, not ours to redistribute. A side whose
crest cannot be fetched gets a monogram roundel in its kit colour instead, and
the page still builds.

A crest earns a backing plate on the share of its pixels that separate from
the page, not on its mean colour — Aston Villa's averages light because of the
pale shield behind the lion, while its claret border and blue field both read.

---

## The cover

The report opens on artwork built from the fixture it introduces: the control
surface both sides held, with every shot of the match on top of it, bleeding
full width from the fold to the top of the sheet. One picture and one
sentence — the statistics are inside.

The artwork is rendered rather than drawn in the PDF, because both layers are
already computed surfaces here and reportlab cannot interpolate a field. It
stays dark on both pages: a dark plate on paper reads as a photograph laid on
the sheet, and a light one would dissolve into it. Its kit colours are
therefore lifted against *its own* ground rather than the page's — on the light
page the report hands over PSG's raw `#004170`, which is correct against paper
and all but invisible on a `#0E1218` image.

A fixture whose artwork cannot render falls back to a typographic cover, which
leads on the match's most lopsided percentage. That is chosen on the
*relative* gap, so a 9.1% against 3.7% conversion rate outranks a 54–46
territory split. Conversion rates draw as two tracks against a common ceiling
rather than one divided bar: each is a share of its own attempts, and the two
do not add up.

---

## The article

The report is a reference: every visual, a paragraph under each. An article is
not that, so `match_article.py` does not walk the visuals and describe them. It
derives a set of findings from the frames, ranks them by how far apart the two
sides actually were, and gives each one a section with the visuals that
evidence it. Behind the argument sits an appendix carrying every remaining
board with the report's own reading under it, and five player radars a side.
The output is a `.docx` with real heading styles and no tables, which is what
pastes cleanly into an editor.

It opens on the report's own cover — not a second layout that tries to look
like it, but page one of the finished PDF rasterised at 200dpi. Two documents
about one match should carry one cover, and rendering the page makes them
identical by construction rather than by two pieces of drawing code agreeing.

The argued part has a floor of 1,200 words and no ceiling: a finding the match
supports is not worth dropping to save words the reader was never promised. The
appendix is extra on top of it.

Two rules hold throughout, both learned from defects the PDF shipped:

- **a sentence names whichever side its numbers name.** Four readings in the
  report said "the away side" and meant "the leader", so they contradicted the
  figures printed beside them whenever the home team led. Every comparison now
  asks which side leads the pair, with a tolerance so a genuine tie is reported
  as one.
- **a number is followed by what it bought.** "70.8% field tilt" is a
  measurement; "seventy per cent of the match in the other half, for 1.08 xG"
  is a finding.

A phase where the two sides were level is still a finding about that phase —
skipping those left an even match with three sections and 773 words against a
1,200 floor.

Nothing is written at all unless the fixture holds together. `match_sanity.py`
asks the questions that cannot be wrong about a real match — a player belongs
to one side, the goals in the events add up to the exported score, the team ids
acting are the two the fixture names — and the article declines with the reason
rather than describing a game that did not happen. It is the artefact most
likely to be published without a second look, so it is the one that should
refuse.

The tests check what is checkable: the length, that no sentence carrying a
number appears in two different matches' articles, that every claim survives
its own mirror image — the same fixture with the sides swapped, which is the
only way to tell "names the away side" apart from "names the leader" — and that
**every figure printed in the prose can be found in the frames**, including the
ratios and shares, recomputed the same way the generator computes them.

What none of that can catch is data that is internally consistent and still
wrong. A squad list that agrees with itself but does not match the real club
passes every check here, because the project has no external roster to compare
against.

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

- Pure black grounds with white pitch markings, and a light `#F5F5F5`
  counterpart that is a second publishing target rather than a recolour.
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
- Every mark is measured against the page it is drawn on. The contrast lift
  moves a kit colour *away* from whichever ground is active — brightening PSG's
  navy on black, darkening Manchester City's sky blue on paper.

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
| `visual_redesign_full.py` | The production renderer and its 53 visuals |
| `visual_redesign_preview.py` | Shared fixture identity and page furniture |
| `player_radar.py` | Player role profiles |
| `visualization_components.py` | Shared chart components and readability helpers |
| `visualization_design.py` | Visual tokens, typography, reusable frames |
| `match_posters.py` | The four post-match posters |
| `render_light.py` | The light-page copy of a finished package |
| `crests.py` | Club crest fetch, cache, plate and fallback |
| `cover_art.py` | The report cover's hero image |
| `match_article.py` | The publishable article and its Word output |
| `match_sanity.py` | Coherence checks a fixture must pass before anything is written about it |
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
verdict sentence that contradicts the numbers printed beside it, an indicator
printed on two posters, a cover that leaves half its page empty. Several run
the same check in a child process on each theme, because the palette is fixed
when `visualization_components` is first imported and one process can only see
one of them.

### What the prose is held to

Every sentence in the report and the article is generated, which means a wrong
one renders as confidently as a right one. Three properties are enforced by
test rather than by reading:

- **A number in the prose exists in the frames.** Every figure is extracted
  from the finished text and matched against the export at the precisions the
  documents print, including ratios and shares recomputed the same way the
  generator computes them. A metric missing from the export used to default to
  zero and print as a measurement — the article stated "PPDA read 0.00", which
  is not a reading a match can produce.
- **A sentence agrees with the numbers beside it.** The recurring defect was a
  branch choosing the wording while an f-string printed the pair in fixed
  home-then-away order: "Man City shot more often, 9 to 12". The same shape
  produced "the same asymmetry" over two figures running opposite ways,
  "finishing ran hot" over three goals from 3.26 xG, and a card crediting
  Arsenal with the stronger field tilt on 29.2% against 70.8%.
- **Every board gets its own reading.** Twenty-nine of the fifty-three visuals
  fell through to two filler sentences, for two independent reasons: the team
  matcher compared `"man city"` against `03_shot_map_man_city.png`, so any
  two-word side lost all fourteen of its boards while a one-word side kept
  them; and fifteen boards had no branch written at all. The tests now assert
  no board carries the generic ending and no coaching note is repeated across
  unrelated boards.

Opta counts the opening minute as minute 0, so a goal from the kick-off was
reported as arriving "in minute 0" in three places. Goal times are now phrased
the way a report phrases them.

### The shapes a match can take

The checks above run against the rendered fixtures, and both of those are home
wins settled without the lead changing hands, level on nothing, with a one-word
home side. So most of the prose had never executed: the level branches, the
draw branches, the away-win branch, the lead-change branch, the goalless
branch. A branch nobody has run is a branch nobody has checked — which is
exactly where the defects were.

`tests/test_scenarios.py` derives eight match shapes from a real fixture by
transformation, so the frames stay internally consistent, and runs every
paragraph writer over every board for each of them:

| shape | what it exercises |
| --- | --- |
| as rendered | the baseline |
| two-word home side | the slug matcher from the other direction |
| one-word both sides | the case where the old substring match worked |
| goalless | no first goal, no score state |
| level on everything | every comparison with nothing to compare |
| away win | winner and home side are not the same team |
| lead changed hands | the match-story card that assumed it never did |
| thin export | optional metric columns simply absent |

Four defects surfaced here that neither rendered fixture could reach. A goal
with no recorded scorer printed "scored through nan" — twice, for two different
reasons: `str(row.get("player", "Goal"))` returns `"nan"` when the column exists
and the cell is empty, and `row.get("player") or ""` returns the NaN itself,
because a float NaN is truthy. The report and the article described the same
goal differently, because the report's goal rows carried no `second` field. A
match level on both territory and shot quality produced "X's stronger field
tilt … yet X used its possessions more efficiently", naming one side twice over
a difference that was not there. And the goal timeline called `.split()[-1]` on
a name that could be empty.

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
