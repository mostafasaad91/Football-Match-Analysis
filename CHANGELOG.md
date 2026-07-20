# Changelog

All notable changes to the WhoScored Post-Match Analyzer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Regression tests for fixed team-role colours, semantic event styles and
  pass-network relationship palettes.
- Professional project documentation covering the analytical workflow,
  outputs, metric definitions, colour system and validation process.

### Changed

- Defensive blocks now retain the provider's original blocked-shot
  classification after shot normalization and are attributed to the defending
  team in summaries, heatmaps and report text.
- Fouls committed now use one canonical provider-aware count that excludes the
  mirrored foul-won row emitted by paired WhoScored feeds.
- All production outputs now use one canonical matchup palette: ultraviolet
  (`#7A3DFF`) for the first-listed/home side and chartreuse (`#BEEA24`) for the
  second-listed/away side, independent of club, country or kit selection.
- Both team colours are rendered as the exact approved colours without automatic
  brightening, outline, glow or shadow treatment.
- Arrow-heavy maps use colour plus line style: low-priority paths are light
  dashed lines, successful actions use ultraviolet, failed actions use dashed
  chartreuse and decisive highlights use gold.
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
