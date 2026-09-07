import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scatter_labels import label_players
from match_insights import player_observations, possession_observations, stage_counts
from test_match_metrics import event
from insight_visuals import role_group


def test_names_are_at_points_without_label_collisions_even_for_identical_values():
    data=pd.DataFrame({'player':[f'Player long surname {i}' for i in range(18)],
                       'x':[0]*12+[1,2,4,8,10,12], 'y':[0]*12+[.1,.1,.2,.3,.4,.8]})
    fig,ax=plt.subplots(figsize=(12,7));ax.set_xlim(-1,15);ax.set_ylim(-.1,1)
    ax.scatter(data.x,data.y)
    texts=label_players(ax,data,'x','y',color='black',background='white')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    assert {text.get_text() for text in texts}==set(data.player)
    boxes=[text.get_bbox_patch().get_window_extent(renderer) for text in texts]
    assert not any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:])
    for text in texts:
        row=data[data.player.eq(text.get_text())].iloc[0]
        assert tuple(text.xy)==(row.x,row.y)
    plt.close(fig)


def test_blocked_saved_shot_is_not_on_target_in_the_nested_funnel():
    frame=pd.DataFrame([event(1,'Pass',1,0,40,90),
                        event(1,'SavedShot',1,5,90,xG=.3,qualifier_names=['Blocked'])])
    _,possessions=possession_observations(frame)
    assert stage_counts(possessions,1)==[1,1,1,1,0]


def test_unknown_substitute_is_not_in_the_forward_comparison_pool():
    assert role_group('Sub')=='Unknown'


def test_advanced_player_rates_keep_their_denominators():
    frame=pd.DataFrame([event(1,'Pass',1,0,30,70,xT=.2),event(1,'Pass',1,4,70,90,xT=.3),event(1,'Goal',1,8,90,xG=.4)])
    people=player_observations(frame,pd.DataFrame())
    p=people.iloc[0]
    assert p.xT_per_100_touches==100*p.positive_xT/p.touches
    assert p.progressive_pass_pct==100*p.progressive_passes/p.completed_passes
    assert p.xGBuildup<=p.xGChain
