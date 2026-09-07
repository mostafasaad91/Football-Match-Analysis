# Poster layout

The four 2400 × 3000 PNGs retain the saved match observations and team kit
colours. A fixture masthead, cached crests, explicit team rows for each KPI,
numbered sections and aligned panel headings establish the reading order.
Dark and light exports use the same geometry. Publisher attribution uses the
supplied byline. Missing crests fall back to team monograms.

## Chart contract

| Board | Question | Static Matplotlib views | Data and fallback |
| --- | --- | --- | --- |
| Match story | How did attempts and chance quality compare? | Cumulative step lines, shot maps, paired dots, exact table, goal list | Shot events and score spells; empty goal list is explicit; both xG curves extend to the last recorded minute |
| Progression | How did each side reach the box? | Entry maps, stage and lane paired dots, possession/player scatters | Saved event-derived possessions; fewer than eight scatter observations use labelled values |
| Pressure | What followed recoveries and losses? | Exact tables, paired dots, loss maps | Independent loss outcomes; score-state rates withheld below five minutes |
| Players | Who contributed to creation and finishing? | Player scatters, ranked tables, goalkeeper totals | Players with at least 30 minutes for scatters; five-player contribution lists |

Palette: two team colours plus neutral type and rules; circles/squares identify
the two sides, solid/dashed strokes distinguish cumulative xG. Spatial maps
share the 105 × 68 m pitch. No metric calculations change for the redesign.

Rebuild just the posters, offline, in a separate destination:

```powershell
python render_posters.py output/Arsenal_vs_Coventry_3-0_Final --output output/poster_redesign/dark
python render_posters.py output/Arsenal_vs_Coventry_3-0_Final --output output/poster_redesign/light --theme light
```

Inspect all four full-size images and a reduced contact sheet in both themes.
Panel headings are anchored to the page so pitch aspect ratios cannot shift
them. Long masthead labels are fitted to their reserved width before export.
