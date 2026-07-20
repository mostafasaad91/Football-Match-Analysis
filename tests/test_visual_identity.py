from football_match_analysis import choose_matchup_colors
from tactical_visualizations import _clean_dark_navy
from visualization_components import network_link_palette


def test_matchup_colors_follow_team_identity():
    assert choose_matchup_colors("France", "England") == ("#0055A4", "#C8102E")
    assert choose_matchup_colors("Liverpool", "Manchester City") == (
        "#C8102E",
        "#6CABDD",
    )


def test_unknown_teams_receive_stable_distinct_colors():
    first = choose_matchup_colors("Unknown Alpha", "Unknown Beta")
    second = choose_matchup_colors("Unknown Alpha", "Unknown Beta")
    assert first == second
    assert first[0] != first[1]


def test_dark_blue_is_not_replaced_with_generic_light_blue():
    assert _clean_dark_navy("#0055A4") == "#0055A4"


def test_network_links_never_reuse_node_color():
    for node_color in ("#0055A4", "#C8102E", "#F1F5F9", "#98A2B3"):
        link_colors = network_link_palette(node_color)
        assert node_color.upper() not in {color.upper() for color in link_colors}
        assert len(set(link_colors)) == 3
