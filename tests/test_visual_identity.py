import football_match_analysis as analysis
import visual_redesign_full as complete_visuals
import visual_redesign_preview as preview
from football_match_analysis import choose_matchup_colors
from tactical_visualizations import _clean_dark_navy
from visualization_components import (
    C_AWAY,
    C_HOME,
    EVENT_FAILURE,
    EVENT_HIGHLIGHT,
    EVENT_NEUTRAL,
    EVENT_SUCCESS,
    FAILURE_DASH,
    QUIET_DASH,
    contrast_ratio,
    network_link_palette,
)


def test_every_matchup_uses_fixed_visual_roles():
    assert choose_matchup_colors("France", "England") == (C_HOME, C_AWAY)
    assert choose_matchup_colors("Liverpool", "Manchester City") == (C_HOME, C_AWAY)


def test_unknown_teams_receive_the_same_fixed_roles():
    first = choose_matchup_colors("Unknown Alpha", "Unknown Beta")
    second = choose_matchup_colors("Unknown Alpha", "Unknown Beta")
    assert first == second
    assert first == (C_HOME, C_AWAY)


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
        assert complete_visuals.TEAM_COLOR == {1: C_HOME, 2: C_AWAY}
        assert complete_visuals._team_slug(1) == "egypt"
        assert complete_visuals._team_slug(2) == "morocco"
        assert preview.HOME_NAME == "Egypt"
        assert preview.AWAY_NAME == "Morocco"
        assert preview.MATCH_SCORE == "2 — 1"
        assert preview.HOME == C_HOME
        assert preview.AWAY == C_AWAY
        assert contrast_ratio(preview.HOME, "#000000") >= 7.0
        assert 3.0 <= contrast_ratio(preview.AWAY, "#000000") < 4.0
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
    assert contrast_ratio(primary, "#000000") >= 7.0
    assert secondary.upper() != primary.upper()
    assert secondary.upper() != complete_visuals.VALUE.upper()
    assert complete_visuals._display_score("*1 : 0") == "1 — 0"


def test_approved_dark_team_colour_is_preserved_without_lift_or_outline():
    primary, _secondary = complete_visuals._team_series_palette(C_AWAY)

    assert C_AWAY == "#A83246"
    assert primary.upper() == C_AWAY.upper()
    assert complete_visuals._team_mark_color(complete_visuals.AWAY_ID) == C_AWAY
    assert 3.0 <= contrast_ratio(C_AWAY, "#000000") < 4.0


def test_semantic_event_styles_share_the_canonical_identity():
    assert EVENT_SUCCESS == C_HOME
    assert EVENT_FAILURE == C_AWAY
    assert EVENT_NEUTRAL not in {C_HOME, C_AWAY, EVENT_HIGHLIGHT}
    assert EVENT_HIGHLIGHT not in {C_HOME, C_AWAY}
    assert QUIET_DASH != "-"
    assert FAILURE_DASH != "-"
