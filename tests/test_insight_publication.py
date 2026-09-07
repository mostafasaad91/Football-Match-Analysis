from pathlib import Path
import pandas as pd
import pytest
from test_match_metrics import event
from match_insights import possession_observations, stage_counts, score_spells, verified_receptions
from match_metrics import turnover_events, win_probability, build_possessions
from match_editorial import headline, result_read, visual_section, reading
from player_radar import _creation_credits
from package_io import transactional_package


def test_funnel_is_nested_and_includes_saved_shots():
    e=pd.DataFrame([event(1,'Pass',1,0,40,75),event(1,'Pass',1,3,75,90),event(1,'SavedShot',1,5,90,xG=.4),event(2,'BallRecovery',1,8,20)])
    _,p=possession_observations(e)
    counts=stage_counts(p,1)
    assert counts==[1,1,1,1,1]


def test_long_shot_without_box_is_outside_nested_funnel():
    _,p=possession_observations(pd.DataFrame([event(1,'Pass',1,0,40,75),event(1,'SavedShot',1,5,75,xG=.1)]))
    assert stage_counts(p,1)==[1,1,0,0,0]


def test_turnover_uses_shot_time_not_possession_end():
    e=pd.DataFrame([event(1,'Pass',0,58,50,60),event(1,'Pass',1,0,60,70,outcome='Unsuccessful'),event(2,'BallRecovery',1,2,30),event(2,'SavedShot',1,6,90,xG=.3),event(2,'Pass',1,30,40,60)])
    losses=turnover_events(e,1)
    assert losses.iloc[0].punished
    assert losses.iloc[0].seconds_to_shot==6
    assert losses.iloc[0].conceded_xG==.3


def test_no_receiver_link_across_opponent_touch():
    e=pd.DataFrame([event(1,'Pass',1,0,30,50,player='A player'),event(2,'BallRecovery',1,1,50,player='Opponent'),event(1,'Pass',1,2,50,60,player='B player')])
    a,_=build_possessions(e)
    assert verified_receptions(a).empty


def test_creation_credit_stays_in_same_possession():
    e=pd.DataFrame([event(1,'Pass',1,0,50,80,player='Creator',is_key_pass=True),event(2,'BallRecovery',1,2,20),event(1,'SavedShot',1,4,90,player='Shooter',xG=.4)])
    assert _creation_credits(e).get('Creator',{}).get('xA',0)==0


def test_goal_ends_level_spell_not_the_final_winner():
    e=pd.DataFrame([event(1,'Pass',0,0,40,50),event(2,'Goal',10,0,90,xG=.4),event(1,'Goal',20,0,90,xG=.3),event(1,'End',45,0,0)])
    spells=score_spells(e,{'home_id':1,'away_id':2})
    home=spells[spells.side.eq('home')]
    assert home.state.tolist()==['level','trailing','level']
    assert home.minutes.sum()==45
    assert home.xG.sum()==pytest.approx(.3)


def test_completed_draw_is_certain_at_terminal_sample():
    e=pd.DataFrame([event(1,'Pass',0,0,40,50),event(1,'Pass',93,0,40,50,period_code='2H'),event(1,'End',93,10,0,period_code='2H')])
    last=win_probability(e,1,2,window=5).iloc[-1]
    assert last.minute==93 and last.draw==1 and last.home_win==last.away_win==0


def test_title_and_reading_handle_draw_and_opposing_finishing_deviations():
    c={'home':'PSG','away':'Monaco','home_goals':1,'away_goals':2,'home_xG':2.72,'away_xG':.59,'date':'2026-08-31'}
    assert 'Monaco won despite PSG' in headline(c)
    assert '-1.72' in result_read(c) and '+1.41' in result_read(c)
    c.update(home_goals=1,away_goals=1)
    assert 'Drawn on goals' in headline(c)
    assert 'winner' not in result_read(c)


def test_semantic_classification_and_chart_contract():
    assert visual_section(Path('44_pitch_control.png'))=='Possession and Progression'
    assert visual_section(Path('player_profiles/Team/Person.png'))=='Player Impact Appendix'
    assert reading(Path('new.png'),{'chart_contracts':{'new.png':{'reading':'A verified reading.'}}}).strip()=='A verified reading.'


def test_failed_generation_preserves_original_package(tmp_path):
    target=tmp_path/'fixture';target.mkdir();(target/'good.txt').write_text('original')
    @transactional_package
    def fail(events,players,xg,team_metrics,player_metrics,match_info,output_dir):
        (Path(output_dir)/'good.txt').write_text('changed')
        raise RuntimeError('render failed')
    with pytest.raises(RuntimeError):fail(pd.DataFrame(),None,None,None,None,{},target)
    assert (target/'good.txt').read_text()=='original'
