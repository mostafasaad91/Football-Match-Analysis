# Changelog

All notable changes to the WhoScored Post-Match Analyzer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- One typeface across the PDF. The commentary was set in Times while every
  embedded visual is sans, so 85.6% of the document's characters belonged to a
  family none of its images used and each page read as two documents stapled
  together. Weight and colour separate the commentary now, not a second family.
- The amber accent is gone. At 1,398 characters against 648 for both kit
  colours combined, a fixed colour was louder than the fixture — on a page whose
  whole premise is that the two teams own the palette. Structural marks are
  neutral; a brand cyan sampled from `assets/logo.jpg` is reserved for the
  publisher's mark and never touches a value.
- Six-step type scale replaces 23 ad-hoc font sizes, several within a fifth of
  a point of each other. The cover's lead statistic keeps two deliberately
  mismatched sizes, named so they stay a decision rather than drift.
- Rebuilt cover: the publisher's badge, the fixture, a one-line verdict, and
  the match's widest percentage split drawn at full width — the graphic *is* the
  finding. Falls back to a scoreline layout when no split exceeds 25 points, so
  an even match gets an opener that suits it. Blank space fell from 84.3% to
  65%.
- The commentary band shares the visual's pure-black ground instead of
  `#0A0A0A`, removing the seam ruled across every visual page.
- Embedded visuals use the same left margin as the text (42pt, was 4pt), so a
  page has one edge rather than three.
- Executive-summary cards follow their content instead of a fixed 135pt that
  left each one nine tenths empty, and the freed space carries the four
  headline splits.
- Contents markers no longer alternate between the team colours by row number,
  which encoded nothing; leader rules start where the title ends.

### Fixed

- Player-network labels are placed against every other node, not just their
  own. A name cleared its own marker and landed across a neighbour's — one
  midfielder's surname printed over another's circle. Placement now scores the
  four sides for clearance and penalises any that would run off the pitch,
  which is how "Kostic" came to render as "ostic".
- `side_kpis` returns the height it consumed. The block below it was positioned
  at a hand-picked y that held only for the number of KPIs present when it was
  written, so a fourth KPI drove its value straight through the heading beneath.
- A fifth substitution in one half landed exactly on the "Completed pass links"
  footer; row spacing now tightens only when the extra row needs it.
- Legends added where marks were encoded and never explained: Pitch Control
  (colour is which side held the space) and Unlocking the Block (what a dot is,
  and whose defensive line the dashed rule marks).
- Radar labels no longer break inside a word. `CLEARANCES`, `RECOVERIES` and
  `INTERCEPTIONS` were written with the line break mid-word to make them fit and
  rendered as `CLEAR/ANCES`, `RECOV/ERIES`, `INTERCEP/TIONS`.
- Names were shortened to seven characters even in side-panel rows with most of
  a column to spare, turning Locatelli into `Locate…` and Aït-Nouri into
  `Aït-No…`.
- The "Began half" legend ran under the next swatch.
- A subtitle over 115 characters was cut mid-sentence with no ellipsis, so
  Match Momentum ended on "…this shows who was".

### Removed

- 4,323 lines of unreachable code across five modules: 51 top-level
  definitions that nothing in the tree referenced. Most were superseded rather
  than abandoned — the `_rpt_*` and `_panel_*` families belonged to report
  pages the redesigned renderer replaced, `render_pass_network_v2` and
  `make_pass_target_zones_v2` to the v2 chart set, `draw_match_report` was a
  self-described legacy stub, and `text_shadow` predates `label_outline()`.
  Found by name-counting across every tracked file and re-run to a fixed point,
  since deleting a caller orphans its helpers: four passes were needed, and the
  three after the first accounted for 1,028 of the lines.
- Eight imports bound but never used, an empty `.agents/` directory, and a
  `tmp/` scratch directory of palette-review PNGs.
- The import-time banner announcing that `tactical_visualizations` loaded
  correctly. The stale-copy override it reported on is still in place and still
  warns when it fires; only the healthy-path announcement is gone.

Verified by rendering the sample package from the previous commit in a
separate worktree and diffing: 65 files either way, none missing, none added,
and all 64 non-PDF outputs byte-for-byte identical.

### Added

- End-to-end golden test (`tests/test_metrics_golden.py`). The whole metric
  engine is run over the committed France vs England events and all 66
  published team columns plus every player sequence metric are compared against
  a frozen reference in `tests/golden/`. The unit tests prove each definition is
  implemented as written; this proves the assembled pipeline still produces the
  numbers it produced before. `scripts/freeze_golden.py` re-freezes the
  reference (`--check` reports drift without writing).
- `fixtures.py` and `data/fixtures/`: committed season calendars for the Premier
  League, La Liga and Serie A (1,140 fixtures) carrying each match's WhoScored
  id, so the analyser's one manual input becomes a lookup —
  `python fixtures.py arsenal --next --url`. An ambiguous name is refused with
  its candidates rather than resolved to whichever club sorts first.
- `team_palettes.py`: real home-kit colours for ~770 clubs and national teams —
  every UEFA league that supplies Champions League, Europa League or Conference
  League entrants, the non-UEFA continental competitions (CAF, AFC, CONMEBOL,
  CONCACAF), and all FIFA national teams. Merged non-destructively into the
  existing palette so hand-picked entries still win.
- `MATCH_ANALYSIS_TEAM_COLORS` switch. Default `kit` draws each side in its real
  kit colour (clash- and contrast-resolved); `roles` restores the previous fixed
  blue/yellow role pair.
- Shared shot-outcome palette (`SHOT_GOAL` / `SHOT_SAVED` / `SHOT_MISS` /
  `SHOT_BLOCKED` / `SHOT_POST`) plus `shot_outcome_color()`, so every shot map —
  legacy, v2 and redesign — uses one key.
- Goalkeeper goal-frame plot: one goal frame per keeper showing where every
  shot they faced crossed the line, its outcome, the body part that struck it
  and its xG, plus saves / save rate / xGoT faced / goals prevented. Uses the
  provider's `GoalMouthY` / `GoalMouthZ` values when present and falls back to
  the placement qualifiers (`LowLeft`, `HighCentre`, `MissRight`, …).
- `q_value()` in the parser, and `goal_mouth_y` / `goal_mouth_z`, `blocked_x` /
  `blocked_y`, `pass_length` / `pass_angle` columns on parsed events. The
  parser previously kept only qualifier *names* and discarded their values.
- Advanced metrics in `match_metrics.py`, each covered by tests:
  `post_shot_xg` / `shot_placement` / `placement_difficulty` (real PSxG, which
  weights the chance by how far the placement pulled the ball from the
  keeper — the old "xGoT" was the sum of xG over on-target shots and ignored
  placement entirely); `set_piece_breakdown` / `shot_origin`;
  `xg_momentum`; `defensive_line_height`; `team_compactness`;
  `network_centrality` / `pass_links`; `turnover_events`; `duel_map`;
  `shot_placement_zones`; `pass_geometry` / `pass_length_profile`;
  `goalkeeper_distribution`; `press_resistance` / `pressure_mask`;
  `line_breaking_passes`; `win_probability`; `zone_value` / `action_values` /
  `player_action_value` — a unified value in goals for every action, on-ball
  and defensive, so a centre-back and a winger appear in the same ranking.
  This is a zone-value model, not a fitted VAEP: real VAEP trains classifiers
  on labelled sequences and this project holds no such data, so the value
  surface is explicit rather than learned, and the page says so.
- curl-cffi as the first fetch attempt, ahead of cloudscraper, requests and the
  browser. WhoScored refuses a stock Python client during the TLS handshake,
  before a header is read, so no User-Agent change ever helped and the pipeline
  fell through to driving Chrome. Impersonating Chrome's TLS/HTTP2 fingerprint
  returns the same page over a plain GET in under a second. Both the match page
  and the official-stats pages now use it, with the old attempts kept behind it
  as fallbacks. Borrowed from the MatchLab project's `sources/_http/transport`.
- `MATCH_ANALYSIS_BROWSER_STATS` to re-enable the browser capture of the
  official stats table.
- Immutable raw snapshots: every provider payload is gzipped into
  `output/raw_snapshots/` before anything parses it (~90 KB a match), so a
  metric added later can be backfilled across the whole history without
  fetching a page again. `python team_history.py replay` recomputes every
  stored match offline — a full match reparses in about 0.17s. Also borrowed
  from MatchLab.
- `match_store.py` and `team_history.py`: a persistent SQLite history at
  `output/match_history.db`, written automatically at the end of every run.
  Each analysis previously wrote to its own folder and forgot the match, so no
  question spanning more than one game was answerable. One row per team per
  match and one per player per match are appended, keyed on the provider's own
  match id so re-analysing a fixture replaces its rows instead of
  double-counting it. Query it with `python team_history.py team Arsenal
  --last 6`, `--summary`, `--metrics`, or `export`. `metric_percentile()`
  finally makes percentiles measurable against real history rather than
  against the other twenty-one players on the pitch that day, and returns
  None until there is enough of it.
- An explanatory metric batch answering *how* rather than *how much*:
  `sequence_typology` / `classify_sequence` (every possession classified as
  build-up, sustained, direct, counter or set piece, with xG per route),
  `receptions_between_lines`, `switches_of_play`, `time_to_progress`,
  `pressing_triggers`, `rest_defence_structure`, `goal_origin_chains`,
  `substitution_impact`, `third_man_combinations`, `second_ball_recovery`,
  `field_tilt_timeline`. Five pages carry them: how the danger was built, goal
  origins, unlocking the block per team, and press triggers with rest defence.
- `pitch_control` / `average_positions`: a control surface where influence
  decays with distance rather than a hard Voronoi split, so space reads as
  strongly held, weakly held or genuinely contested, plus each side's share. Pressure is inferred from nearby
  opposition events rather than measured, and the defensive line a pass breaks
  is estimated per five-minute window — both are documented approximations,
  not tracking-grade values.
- `MATCH_ANALYSIS_URL` environment variable, so a different fixture can be
  analysed without editing `MATCH_URL` in the source.
- Scorer names on the goal markers of the xG flow curve; the minute alone made
  the reader cross-reference the timeline strip below to learn who scored.
- Regression tests for fixed team-role colours, semantic event styles and
  pass-network relationship palettes.

- Eight visuals built on the new metrics: match momentum (windowed xG
  differential), set-piece contribution, ball-loss maps per team, defensive
  shape over time, a win-probability curve marked at every goal, and a
  "playing through" map per team pairing line-breaking passes with press
  resistance and duel outcomes. Existing pages gained panels for goal-frame
  zones targeted (shot map), pass length and long-ball survival (pass map),
  network connectors by betweenness (pass network) and goalkeeper
  distribution (keeper page).

### Removed

- Ten redundant visuals that repeated numbers already shown elsewhere. The
  shot-profile, xG-summary, match-statistics and advanced-metrics pages were
  each a subset of the post-match dashboard in a different arrangement; the
  ball-touches and pass-by-third pages were flat versions of the
  dominating-zones and pass-target maps; the danger-creation pages were the
  union of the Zone 14 and box-entry pages without their detail. The unique
  rows of the advanced-metrics and defensive-summary pages moved into the
  dashboard, which now carries 32 indicators across four columns and is the
  report's single numeric reference. The set runs 01–34 with no gaps.

### Changed

- The goalkeeper visual is now that goal-frame plot rather than a four-row
  bilateral comparison.
- PDF identity: the page chrome takes the fixture's own kit colours instead of
  a fixed blue/yellow pair, its greys match the values the rendered images use
  rather than a second near-black, the two-tone team rule that tops every
  visual is repeated on the commentary band, and the commentary runs in two
  columns — on a 14-inch page a single measure ran to roughly 830pt at 9pt
  type, far past the length an eye can track back from.
- Each PDF visual now carries one continuous analyst paragraph instead of three
  labelled blocks (`PERFORMANCE ANALYST.` / `DATA ANALYST.` / `INTEGRATED
  READ.`). The labels made every page read like a form being filled in; the
  same three strands — what happened, what the data says, what to do with it —
  now run as prose via `visual_narrative()`.
- Pass-network and average-position nodes carry the player's shirt number
  inside the marker with the name above it. A surname squeezed inside the node
  had to shrink to fit and was clipped on longer names.
- Player radars are drawn in their own team's colour, one lightness step per
  metric group, instead of a fixed category palette.
- Labels drawn ON a coloured fill (heatmap cells, network nodes) take their
  colour from that fill via `text_on_fill()` instead of always being white,
  which was unreadable on a light kit and at the top of a light heat ramp.
- QA contact-sheet chrome no longer overrides the fixture colours it is passed,
  so the sheet header matches the thumbnails it frames.
- A team that plays in white keeps a white identity (soft silver `#DCE3EC`)
  instead of falling through to the next entry in its palette. Juventus and
  Real Madrid were both being rendered in gold.

### Fixed

- A full run died on Windows whenever stdout was redirected, after the match
  had already been fetched, parsed and analysed. The console output uses
  arrows, box drawing and status glyphs throughout, and a redirected Windows
  stdout defaults to cp1252, which cannot encode any of them — so the first
  status line raised `UnicodeEncodeError` and cost the whole analysis. Both
  streams are now reconfigured to UTF-8 with `errors="replace"` at startup, so
  a decorative character can never again take down a completed run. The
  stale-copy guard's messages, one of which crashed the import itself for the
  same reason, are also plain ASCII now.
- `reportlab` was imported by `tactical_pdf_report.py` but absent from
  `requirements.txt`, so a fresh install produced every PNG and then failed at
  PDF assembly.
- The blanket `*.csv` ignore rule swallowed the golden reference, which would
  have kept it out of the repository entirely. The test skips when the
  reference is missing, so a fresh clone would have reported a pass while
  checking nothing.
- README documented the fixed blue/yellow role palette and told the reader to
  edit `MATCH_URL` in the source file. Both had been superseded by real kit
  colours and `MATCH_ANALYSIS_URL`, so following the instructions produced the
  wrong colours on the wrong match.
- The PDF commentary named the wrong team on the advanced-dashboard page. Both
  the tactical paragraph and the data paragraph hard-coded the home side as the
  territorial team and the away side as the efficient one, so any fixture where
  that was reversed printed sentences contradicting the numbers beside them —
  Juventus 2-5 Manchester City read as "Juventus controlled more advanced
  territory" against a field tilt of 8.8%. The leader is now read off the
  values.
- The match key no longer contains the date. A postponed fixture is dated
  differently by different sources, so a date in the key split one match into
  two rows with the events on one and the score on the other. Competition,
  season and the two teams identify it uniquely in league play and survive the
  move; the date is kept on the row as checkable data instead.
- Adding a column to the history schema now migrates an existing database.
  `CREATE TABLE IF NOT EXISTS` leaves an older file on its original columns, so
  every insert against it would have failed.
- Team-name lookup took the first of ~975 palette keys that overlapped the
  name in either direction, so `"Al "` resolved to Arsenal and any `"United"`
  to whichever United came first in the table — a wrong kit rendered with full
  confidence. Partial matching now only resolves when exactly one candidate
  matches; anything ambiguous falls to the deterministic placeholder and is
  reported at the end of the run so an alias can be declared. Fragments under
  four characters are treated as no information at all.
- The browser capture of the official stats table cost roughly four minutes per
  run before failing on this machine, after which the report completed on
  matchCentreData counts regardless. It is off unless asked for, taking a full
  run from 4m15s to 1m54s with identical output.
- `build_tactical_pdf` overwrote the fixture colours it was handed with a
  hard-coded blue/yellow pair, so the PDF chrome never matched the images.
- Pass-network centrality was computed over the whole match on a per-half page,
  so the connectors list named players who were not on the pitch for the half
  being drawn.
- Own goals were credited to the team that struck them across every redesigned
  visual, so the scoreline, the match-state band, the goal timeline, the
  game-state durations and the goalkeeper frames all put them on the wrong
  side. Juventus 2-5 Manchester City rendered as 3-4. A shared
  `credited_team()` helper now resolves the beneficiary from `scoring_team`,
  falling back to flipping on `is_own_goal`.

- Pitch surfaces are true black (`#000000`) and markings are white at a
  controlled alpha, replacing the green legacy lines and the near-invisible
  dark greys.
- xT heatmap: ranks 4–10 are drawn in a contrasting accent instead of the team
  colour, which was the colour of the hot cells they sat on.
- Professional project documentation covering the analytical workflow,
  outputs, metric definitions, colour system and validation process.

### Changed

- Defensive blocks now retain the provider's original blocked-shot
  classification after shot normalization and are attributed to the defending
  team in summaries, heatmaps and report text.
- Fouls committed now use one canonical provider-aware count that excludes the
  mirrored foul-won row emitted by paired WhoScored feeds.
- All production outputs now use one canonical matchup palette: electric blue
  (`#2F5BFF`) for the first-listed/home side and true yellow (`#FFD400`) for the
  second-listed/away side, independent of club, country or kit selection.
- Exact numeric values now render in white across comparison rows, cards,
  sidebars and team pages.
- Both team colours are rendered as the exact approved colours without automatic
  brightening, outline, glow or shadow treatment.
- Arrow-heavy single-team maps retain the team's role colour; success, failure
  and action type use solid/dashed lines, opacity and marker shape. Decisive
  highlights use white.
- Defensive activity uses a smoothed team-colour heatmap with bright,
  shape-coded action marks and committed fouls only.
- Team-specific pass-distribution charts use two readable shades derived from
  the side's fixed role colour.
- Dangerous counters now include 40+ metre transition breaks that reach the
  final third within 12 seconds, alongside transition shots and box entries.
- Remaining comparison legends and xT labels now use the active fixture names,
  and provider score markers are removed from visual headers.
- Live match analysis now routes through the unified complete AMOLED renderer,
  so every fixture receives the same current visual system as the reference
  package instead of the legacy renderer.
- Match names, score, export slugs, QA dashboards and PDF commentary are
  configured dynamically from the active fixture while colour roles stay fixed.
- Pass-network links now use a relationship palette that is always distinct
  from player-node colours.
- Pass-network and average-position player labels use larger bright text,
  collision-aware placement and stronger dark backplates.
- xT maps now highlight the top threat-creating passes with gold and fixed-role
  arrows plus brighter player names.

## [9.4] - 2026-04-30

### Added
- Internal xG engine V7 with logistic regression model
- Three-variant ensemble (Opta-like, SPADL-like, academic)
- 100+ team kit colors across Top-5 European leagues
- Kit-based palette system with automatic contrast selection
- Dark-mode visualizations across all 11+ figures
- PDF tactical report with AI-generated commentary
- Triple fallback scraping (cloudscraper → requests → Selenium)
- Grouped visual category boards for social sharing
- xT (Expected Threat) map visualization
- Pass network with player positioning
- Comprehensive match statistics comparison
- Own goal rendering in benefiting team's color

### Changed
- xG model calibrated with ~10-15% reduction for more realistic values
- Shot xG cap reduced to 0.78
- Penalty xG value set to 0.76
- Ensemble weights adjusted (0.48/0.26/0.26)
- Added local fallback scale (0.88) for additional compression

## [9.3] - 2026-04-15

### Added
- MatchCentreData extraction with robust brace-counting
- Official Opta stat integration from DOM
- HTTP-based stat capture as fallback

### Fixed
- Anti-blocking improvements for WhoScored scraping
- Color contrast issues on dark backgrounds

## [9.2] - 2026-03-20

### Added
- Bundesliga and Ligue 1 team palettes
- Serie A team colors
- La Liga expanded coverage

### Changed
- Improved xG flow chart with period markers
- Enhanced pass network visualization

## [9.0] - 2026-02-01

### Added
- Initial release with Premier League support
- xG flow, shot map, pass map visualizations
- Basic PDF report generation
- cloudscraper-based scraping
