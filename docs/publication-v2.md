# Match publication v2

The article and PDF use `match_editorial.py` for titles, section readings and
result interpretation. Titles include the fixture, score, measured contrast and
date. Finishing differences are assessed separately for each side, so opposing
deviations cannot cancel into a misleading combined-total conclusion.

## Offline reproduction

```powershell
python render_snapshot.py "output/path/to/saved-fixture" --output "output/review/fixture" --theme dark --both-themes
```

The source needs `events.csv`, `players.csv`, `xg.csv`,
`team_advanced_metrics.csv`, `player_sequence_metrics.csv` and `match_info.json`.
Use `--both-themes` with a dark full render to also generate the light copy.
`--publication-only` reuses saved PNGs and rebuilds changed boards, the new
analytical figures and documents; its source PNGs must already use the selected
theme. A full render is required when changing the visual theme.

Generation happens in a sibling staging directory. Only a successful PDF and
article build replaces the published folder. The previous package is retained
in a `.previous-...` sibling; a failed stage remains available for diagnosis.
No old package is deleted before rendering.

## Chart definitions and safeguards

- Player scatters: one player, at least 30 minutes, raw match totals. Median
  guides require 8 valid observations; smaller samples receive a value list.
  Players without shots have no xG-per-shot value. Explicit carries alone enter
  progression counts; no synthetic off-ball carries are invented.
- Profiles: 8 role-specific metrics for outfield players, raw counts and
  denominators printed, single-match comparison pool disclosed. Goalkeepers
  use a separate raw profile. All participants receive a file; the main PDF selects three players per team by
  xG, then positive xT, then touches; this is an editorial selection, not a
  universal player rating.
- Possession scatter: one controlled possession, duration versus positive
  successful-movement xT. Regain-speed charts omit non-reaching regains from
  the vertical time axis but disclose them and retain them in the table.
- Funnel: possessions, final-third presence, subsequent box presence,
  subsequent shot, subsequent shot on target. Each stage is nested in the same
  possession. A long-range shot without box access is deliberately excluded
  from the final stages. These are possession counts, not entry-event counts.
- Entry routes: lane of the entry origin and whether a shot followed in that
  possession. Several entries can precede one shot; do not sum attributed xG
  across these rows. Cut-backs are geometry-based candidates.
- Score states: actual within-period spells, including added time. A goal is
  credited to the state immediately before it. Rates per 30 state minutes are
  suppressed below 5 minutes; zero exposure is unavailable, not zero output.
- Loss consequences: the immediately following opponent possession and the
  first 12 seconds after the loss. Third, box and shot are independent outcome
  counts. The existing turnover map now measures time to the shot rather than
  the duration of the whole opponent possession.
- Receptions: next controlled touch, same team/possession, within 5 seconds and
  5 metres of the endpoint. Unmatched passes stay unknown.
- Substitutions: equal windows of 3–10 minutes in the same period, shortened by
  nearby substitution clusters. Goals/cards are flagged. Differences are not
  estimates of a substitution's causal effect.
- History: at least 20 unique team/date observations, same named competition,
  metric version and explicit `xg_model_version`. Unknown model provenance
  prevents comparison. Reasons for skipped charts are saved, not hidden.

## Model and interpretation limits

The four rebuilt posters are native 2400 × 3000 figures. They cover match story,
progression, pressure/transitions and players/finishing. The same panel structure
is used on AMOLED black and light backgrounds. Each panel records its definition
in `poster_catalog.json`; no fifth poster is generated.

`xGoT` is retained as a storage key for compatibility. Its publication label is
**local post-shot estimate**: a placement-weighted, uncalibrated heuristic with
no ball velocity or actual keeper location. Average-position influence uses
distance in a nominal 105 × 68 metre frame, averaged by duration in short windows
split at substitutions/cards; it is not tracking-based pitch control. The win-probability curve is an
uncalibrated heuristic, with a known terminal result only for a completed feed.

Source xG and xT are retained; no new model is trained or claimed calibrated.
The report separates observed counts from questions requiring video or tracking.
Superseded publication implementations have been removed from production modules.

## Validation

Player points carry their full names next to the original coordinates. A local
placement routine measures label bounds and adds leaders where clustered values
need separation; neither points nor values are jittered. Old radar exports are
replaced by advanced role profiles. `Sub` is an unknown position, never a forward.

Repeated goal summaries, dashboard pages, half-average position pages and the old
unlocking pages were removed from new runs. Their useful information is retained
in goal origins/xG flow, four posters, half-pass networks and verified receptions.
New stage flows, paired-dot comparisons, loss tiles and percentile dot profiles
replace repeated bars. Each chart carries its purpose and denominator.

```powershell
python -m pytest tests/test_insight_publication.py tests/test_match_metrics.py tests/test_radar_meaning.py tests/test_post_shot_xg.py -q
```

Inspect rendered PDFs and Word output as well as tests: typography, long names,
small comparison pools and page breaks need visual verification.
