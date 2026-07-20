# Changelog

All notable changes to the WhoScored Post-Match Analyzer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Regression tests for dynamic team colours, unknown-team fallbacks and
  pass-network relationship palettes.
- Professional project documentation covering the analytical workflow,
  outputs, metric definitions, colour system and validation process.

### Changed

- Fouls committed now use one canonical provider-aware count that excludes the
  mirrored foul-won row emitted by paired WhoScored feeds.
- Defensive activity uses a smoothed team-colour heatmap with bright,
  shape-coded action marks and committed fouls only.
- Dark team identities are automatically tinted for pitch marks, network nodes
  and transition arrows while the original team hue remains recognisable.
- Team-specific pass-distribution charts now use two shades derived from the
  active team's identity colour instead of a fixed purple comparison series.
- Dangerous counters now include 40+ metre transition breaks that reach the
  final third within 12 seconds, alongside transition shots and box entries.
- Remaining comparison legends and xT labels now use the active fixture names,
  and provider score markers are removed from visual headers.
- Live match analysis now routes through the unified complete AMOLED renderer,
  so every fixture receives the same current visual system as the reference
  package instead of the legacy renderer.
- Match names, score, colours, export slugs, QA dashboards and PDF commentary
  are configured dynamically from the active fixture.
- Pass-network links now use a relationship palette that is always distinct
  from player-node colours.
- Pass-network and average-position player labels use larger bright text,
  collision-aware placement and stronger dark backplates.
- xT maps now highlight the top 10 threat-creating passes with indigo arrows
  and brighter player names.
- Club and national-team colours remain fixture-aware throughout v2 visuals;
  dark blue identities are no longer replaced by a generic light blue.

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
