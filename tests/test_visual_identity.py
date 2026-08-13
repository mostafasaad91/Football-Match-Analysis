import football_match_analysis as analysis
import visual_redesign_full as complete_visuals
import visual_redesign_preview as preview
from football_match_analysis import WHITE_KIT_SILVER, choose_matchup_colors
from tactical_visualizations import _clean_dark_navy, _team_identity_color
from visualization_components import (
    BG_PITCH,
    C_AWAY,
    C_HOME,
    EVENT_FAILURE,
    EVENT_HIGHLIGHT,
    EVENT_NEUTRAL,
    EVENT_SUCCESS,
    FAILURE_DASH,
    PITCH_LINE,
    QUIET_DASH,
    SHOT_BLOCKED,
    SHOT_GOAL,
    SHOT_MISS,
    SHOT_POST,
    SHOT_SAVED,
    USE_REAL_TEAM_KIT_COLORS,
    contrast_ratio,
    metric_card,
    network_link_palette,
    shot_outcome_color,
    TEXT_BR,
)


def test_matchup_colours_follow_the_active_team_colour_mode():
    france_england = choose_matchup_colors("France", "England")
    merseyside = choose_matchup_colors("Liverpool", "Manchester City")

    if USE_REAL_TEAM_KIT_COLORS:
        # Kit mode: each side is drawn in its own real home-kit colour, and the
        # two colours in a fixture are never the same. England plays in white,
        # which is substituted for the soft silver stand-in.
        assert france_england == ("#0055A4", WHITE_KIT_SILVER)
        assert merseyside == ("#C8102E", "#6CABDD")
    else:
        assert france_england == (C_HOME, C_AWAY)
        assert merseyside == (C_HOME, C_AWAY)


def test_white_kit_teams_keep_a_white_identity():
    """A side that plays in white must not be handed a colour from elsewhere in
    its palette — Juventus and Real Madrid used to come out gold."""
    if not USE_REAL_TEAM_KIT_COLORS:
        return
    for home, away in (
        ("Juventus", "Manchester City"),
        ("Real Madrid", "Barcelona"),
        ("Tottenham", "Arsenal"),
    ):
        resolved_home, resolved_away = choose_matchup_colors(home, away)
        assert resolved_home == WHITE_KIT_SILVER, (home, resolved_home)
        assert resolved_home != resolved_away
    # The stand-in must stay clear of the pure white owned by pitch markings
    # and the highlight layer, while still reading as light on the black page.
    assert WHITE_KIT_SILVER.upper() != "#FFFFFF"
    assert contrast_ratio(WHITE_KIT_SILVER, "#000000") >= 12.0


def test_matchup_colours_are_stable_and_distinct_for_unknown_teams():
    first = choose_matchup_colors("Unknown Alpha", "Unknown Beta")
    second = choose_matchup_colors("Unknown Alpha", "Unknown Beta")
    assert first == second
    if USE_REAL_TEAM_KIT_COLORS:
        # No palette entry exists, so both sides fall back to the deterministic
        # pool — but they must still be told apart.
        assert first[0] != first[1]
    else:
        assert first == (C_HOME, C_AWAY)


def test_same_kit_fixtures_fall_back_to_the_fixed_roles():
    """Two teams that resolve to near-identical colours must not both render
    in that colour — the renderers drop back to the role pair instead."""
    from tactical_visualizations import _match_colors

    info = {
        "home_id": 1,
        "away_id": 2,
        "home_color": "#C8102E",
        "away_color": "#CC1030",
    }
    assert _match_colors(info) == (C_HOME, C_AWAY)
    assert _match_colors({"home_id": 1, "away_id": 2}) == (C_HOME, C_AWAY)


def test_single_team_pages_never_share_a_colour():
    from tactical_visualizations import _team_identity_color

    info = {
        "home_id": 7,
        "away_id": 8,
        "home_color": "#C8102E",
        "away_color": "#6CABDD",
    }
    home = _team_identity_color(info, 7)
    away = _team_identity_color(info, 8)
    assert home != away
    if USE_REAL_TEAM_KIT_COLORS:
        assert (home, away) == ("#C8102E", "#6CABDD")
    else:
        assert (home, away) == (C_HOME, C_AWAY)


def test_pitch_reads_as_white_markings_on_true_black():
    from visualization_components import IS_LIGHT_THEME

    if IS_LIGHT_THEME:
        return
    assert BG_PITCH == "#000000"
    assert PITCH_LINE == "#FFFFFF"
    # Markings must clear the page without competing with the data layer.
    assert contrast_ratio(PITCH_LINE, BG_PITCH) >= 15.0


def test_shot_outcomes_have_one_colour_each():
    outcomes = [SHOT_GOAL, SHOT_SAVED, SHOT_MISS, SHOT_BLOCKED, SHOT_POST]
    assert len(set(outcomes)) == len(outcomes)
    for colour in outcomes:
        assert contrast_ratio(colour, "#000000") >= 3.0
    assert shot_outcome_color("SavedShot") == SHOT_SAVED
    assert shot_outcome_color("ShotOnPost") == SHOT_POST
    assert shot_outcome_color("MissedShots") == SHOT_MISS
    assert shot_outcome_color("BlockedShot") == SHOT_BLOCKED
    assert shot_outcome_color("Goal") == SHOT_GOAL
    assert shot_outcome_color("something-unmapped") == SHOT_BLOCKED


def test_team_palette_database_covers_every_competition_tier():
    import football_match_analysis as analysis
    import team_palettes

    palettes = analysis.TOP5_2025_26_TEAM_PALETTES
    # Champions League / Europa League / Conference League regulars, the
    # non-UEFA continental competitions, and national teams.
    for team in (
        "Arsenal",           # Premier League
        "Real Madrid",       # LaLiga
        "Inter Milan",       # Serie A
        "Bayern Munich",     # Bundesliga
        "PSG",               # Ligue 1
        "Sporting CP",       # Primeira Liga
        "Qarabag",           # UCL league phase, outside the big leagues
        "Pafos",             # UCL league phase, outside the big leagues
        "Bodo/Glimt",        # Eliteserien
        "Ferencvaros",       # NB I
        "Al Ahly",           # CAF Champions League
        "Al Hilal",          # AFC Champions League Elite
        "Flamengo",          # Copa Libertadores
        "Club America",      # CONCACAF Champions Cup
        "Brazil",            # CONMEBOL national team
        "Uganda",            # CAF national team
        "Vanuatu",           # OFC national team
    ):
        assert team in palettes, team
        assert analysis._team_palette(team, "#999999")[0].startswith("#")

    # Every alias must resolve to a team that actually exists.
    for alias, target in team_palettes.EXTRA_TEAM_ALIASES.items():
        assert target in team_palettes.ALL_TEAM_PALETTES, (alias, target)


def test_dark_blue_is_not_replaced_with_generic_light_blue():
    assert _clean_dark_navy("#0055A4") == "#0055A4"


def test_network_links_never_reuse_node_color():
    for node_color in ("#0055A4", "#C8102E", "#F1F5F9", "#98A2B3"):
        link_colors = network_link_palette(node_color)
        assert node_color.upper() not in {color.upper() for color in link_colors}
        assert len(set(link_colors)) == 3


def test_live_matches_use_complete_amoled_renderer(tmp_path):
    assert analysis.USE_COMPLETE_AMOLED_PACKAGE is True

    original = {
        "home_id": complete_visuals.HOME_ID,
        "away_id": complete_visuals.AWAY_ID,
        "home_name": complete_visuals.HOME_NAME,
        "away_name": complete_visuals.AWAY_NAME,
        "home_color": complete_visuals.HOME,
        "away_color": complete_visuals.AWAY,
        "score": complete_visuals.MATCH_SCORE,
    }
    original_out = complete_visuals.OUT
    try:
        complete_visuals.configure_match(
            {
                "home_id": 1,
                "away_id": 2,
                "home_name": "Egypt",
                "away_name": "Morocco",
                "home_color": "#CE1126",
                "away_color": "#006233",
                "score": "2 — 1",
            },
            tmp_path / "Egypt_vs_Morocco_2-1",
        )
        expected = (
            ("#CE1126", "#006233") if USE_REAL_TEAM_KIT_COLORS else (C_HOME, C_AWAY)
        )
        assert complete_visuals.TEAM_COLOR == {1: expected[0], 2: expected[1]}
        assert complete_visuals._team_slug(1) == "egypt"
        assert complete_visuals._team_slug(2) == "morocco"
        assert preview.HOME_NAME == "Egypt"
        assert preview.AWAY_NAME == "Morocco"
        assert preview.MATCH_SCORE == "2 — 1"
        assert (preview.HOME, preview.AWAY) == expected
        # Both sides must clear readable contrast against the page they are
        # actually drawn on (black on AMOLED, light paper on light).
        from visualization_components import IS_LIGHT_THEME

        page = preview.BG
        min_side = 2.2 if IS_LIGHT_THEME else 2.4
        assert contrast_ratio(preview.HOME, page) >= min_side
        assert contrast_ratio(preview.AWAY, page) >= min_side
    finally:
        complete_visuals.configure_match(original, original_out)


def test_visual_catalog_accepts_numbered_and_player_named_files(tmp_path):
    original = {
        "home_id": complete_visuals.HOME_ID,
        "away_id": complete_visuals.AWAY_ID,
        "home_name": complete_visuals.HOME_NAME,
        "away_name": complete_visuals.AWAY_NAME,
        "home_color": complete_visuals.HOME,
        "away_color": complete_visuals.AWAY,
        "score": complete_visuals.MATCH_SCORE,
    }
    original_out = complete_visuals.OUT
    try:
        complete_visuals.configure_match(
            {
                "home_id": 1,
                "away_id": 2,
                "home_name": "Spain",
                "away_name": "Argentina",
                "home_color": "#C60B1E",
                "away_color": "#75AADB",
                "score": "1 — 0",
            },
            tmp_path,
        )
        catalog = complete_visuals.build_catalog(
            [
                tmp_path / "01_xg_flow.png",
                tmp_path / "player_radars" / "Spain" / "Pedri.png",
            ]
        )
        assert catalog["number"].tolist() == ["01", "P02"]
        assert catalog["title"].tolist() == ["Xg Flow", "Pedri"]
        assert catalog["file"].tolist() == [
            "01_xg_flow.png",
            "player_radars/Spain/Pedri.png",
        ]
    finally:
        complete_visuals.configure_match(original, original_out)


def test_team_series_palette_and_score_are_fixture_aware():
    primary, secondary = complete_visuals._team_series_palette(C_HOME)

    assert primary.upper() == C_HOME.upper()
    assert contrast_ratio(primary, complete_visuals.BG) >= 3.8
    assert secondary.upper() != primary.upper()
    assert secondary.upper() != complete_visuals.VALUE.upper()
    assert complete_visuals._display_score("*1 : 0") == "1 — 0"


def test_approved_team_colours_are_preserved_without_lift_or_outline():
    from visualization_components import IS_LIGHT_THEME

    primary, _secondary = complete_visuals._team_series_palette(C_AWAY)

    if IS_LIGHT_THEME:
        # "Ink & Burnt Orange" light identity on the #F5F5F5 page.
        assert C_HOME == "#0A0A0A"
        assert C_AWAY == "#E76F51"
        assert contrast_ratio(C_HOME, "#F5F5F5") >= 10.0
        # Burnt orange is a mid-value hue; it clears large-mark visibility
        # rather than text-level contrast.
        assert contrast_ratio(C_AWAY, "#F5F5F5") >= 2.2
    else:
        assert C_HOME == "#2F5BFF"
        assert C_AWAY == "#FFD400"
        assert contrast_ratio(C_HOME, "#000000") >= 3.8
        assert contrast_ratio(C_AWAY, "#000000") >= 14.0
    assert complete_visuals._team_mark_color(complete_visuals.HOME_ID) == C_HOME
    assert primary.upper() == C_AWAY.upper()
    assert complete_visuals._team_mark_color(complete_visuals.AWAY_ID) == C_AWAY


def test_single_team_visuals_resolve_identity_from_fixture_role():
    info = {"home_id": 10, "away_id": 20}
    assert _team_identity_color(info, 10, "#FFFFFF") == C_HOME
    assert _team_identity_color(info, 20, "#FFFFFF") == C_AWAY
    assert _team_identity_color(info, 10, C_AWAY) != _team_identity_color(
        info, 20, C_HOME
    )


def test_team_density_maps_use_distinct_fixture_colours():
    original = {
        "home_id": complete_visuals.HOME_ID,
        "away_id": complete_visuals.AWAY_ID,
        "home_name": complete_visuals.HOME_NAME,
        "away_name": complete_visuals.AWAY_NAME,
        "home_color": complete_visuals.HOME,
        "away_color": complete_visuals.AWAY,
        "score": complete_visuals.MATCH_SCORE,
    }
    original_out = complete_visuals.OUT
    try:
        complete_visuals.configure_match(
            {
                "home_id": 10,
                "away_id": 20,
                "home_name": "Blue Team",
                "away_name": "Yellow Team",
                "score": "0 — 0",
            },
            original_out,
        )
        home_palette = complete_visuals._team_density_palette(10)
        away_palette = complete_visuals._team_density_palette(20)
        assert home_palette[-1] == C_HOME
        assert away_palette[-1] == C_AWAY
        assert home_palette[-1] != away_palette[-1]
    finally:
        complete_visuals.configure_match(original, original_out)


def test_metric_card_values_are_white_even_with_team_accent():
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 2))
    metric_card(
        fig,
        0.1,
        0.1,
        0.8,
        0.8,
        label="EXPECTED GOALS",
        value="1.64",
        accent=C_AWAY,
    )
    value_text = next(text for ax in fig.axes for text in ax.texts if text.get_text() == "1.64")
    assert value_text.get_color().upper() == TEXT_BR.upper()
    plt.close(fig)


def test_semantic_event_styles_share_the_canonical_identity():
    assert EVENT_SUCCESS == C_HOME
    assert EVENT_FAILURE == C_AWAY
    assert EVENT_NEUTRAL not in {C_HOME, C_AWAY, EVENT_HIGHLIGHT}
    assert EVENT_HIGHLIGHT not in {C_HOME, C_AWAY}
    assert QUIET_DASH != "-"
    assert FAILURE_DASH != "-"


def test_ambiguous_team_names_are_refused_not_guessed():
    """Partial matching used to take the first overlapping key out of ~975, so
    "Al " resolved to Arsenal and "United" to whichever United came first."""
    import football_match_analysis as fma

    fma.UNRESOLVED_TEAM_NAMES.clear()
    for ambiguous in ("Real", "United"):
        fma._team_palette(ambiguous, "#999999")
        assert ambiguous in fma.UNRESOLVED_TEAM_NAMES
        assert "ambiguous" in fma.UNRESOLVED_TEAM_NAMES[ambiguous]

    # Fragments too short to carry information are not a lookup at all.
    assert fma._partial_palette_matches("Al") == []

    # Full names still resolve exactly, and quietly.
    fma.UNRESOLVED_TEAM_NAMES.clear()
    assert fma._team_palette("Liverpool", "#999999")[0] == "#C8102E"
    assert fma._team_palette("Bayern Munich", "#999999")[0] == "#DC052D"
    assert not fma.UNRESOLVED_TEAM_NAMES


def test_unique_partial_match_is_accepted_but_recorded():
    import football_match_analysis as fma

    fma.UNRESOLVED_TEAM_NAMES.clear()
    palette = fma._team_palette("Athletic Bilbao FC", "#999999")
    assert palette[0] == "#EE2523"
    # Accepted, but the caller is told so the alias can be declared.
    assert "Athletic Bilbao FC" in fma.UNRESOLVED_TEAM_NAMES
    fma.UNRESOLVED_TEAM_NAMES.clear()
