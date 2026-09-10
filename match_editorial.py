"""Evidence-led match titles and commentary, shared by PDF and article.

Templates describe observations. Tactical mechanisms are review questions unless
the event sequence itself establishes them. No winner-based score-state inference.
"""
from pathlib import Path
from metric_registry import label, format_value

def value(c, side, key, digits=1, unit=""):
    return format_value(c.get(f"{side}_{key}"), unit, digits)

def comparison(c, key, digits=1, unit=""):
    return f"{c['home']}: {value(c,'home',key,digits,unit)}; {c['away']}: {value(c,'away',key,digits,unit)}."

def headline(c):
    home,away=c['home'],c['away']
    hg,ag=c.get('home_goals',0),c.get('away_goals',0)
    hx,ax=c.get('home_xG',0),c.get('away_xG',0)
    leader=home if hx>=ax else away
    winner=home if hg>ag else away if ag>hg else None
    candidates=[]
    for key, phrase in [('box_entries','box entries'),('final_third_entries','final-third entries'),('transition_xG','transition xG')]:
        h,a=c.get('home_'+key,0),c.get('away_'+key,0)
        if h+a>0:
            side=home if h>a else away
            digits=2 if key.endswith('xG') else 0
            candidates.append((abs(h-a)/max(h+a,1),f"{side} led {phrase} {max(h,a):.{digits}f} to {min(h,a):.{digits}f}"))
    if hg==ag==0:
        thesis=f"No breakthrough from {hx+ax:.2f} expected goals"
    elif winner and winner!=leader and abs(hx-ax)>.15:
        thesis=f"{winner} won despite {leader}'s higher xG"
    elif hg==ag and hx+ax>=4:
        thesis=f"A draw from {hx+ax:.2f} xG of combined chances"
    elif hg==ag:
        thesis=f"Drawn on goals with {hx:.2f} to {ax:.2f} xG"
    elif abs(hg-ag)>=3 and min(hg,ag)==0:
        thesis=f"{winner}'s clean sheet accompanied {max(hx,ax):.2f} xG for the chance leader"
    else:
        thesis=max(candidates, default=(0,f"{winner} won {hg}-{ag}"))[1]
    # Fixture/date is a natural identity, not a random synonym or generic slogan.
    date=str(c.get('date') or c.get('match_id') or '').strip()
    return f"{home} {hg}-{ag} {away}: {thesis}" + (f" — {date}" if date else '')

def result_read(c):
    hg,ag=c['home_goals'],c['away_goals']
    h,a=c['home_xG'],c['away_xG']
    first=f"{c['home']} and {c['away']} finished {hg}–{ag}, with {h:.2f} and {a:.2f} expected goals respectively."
    # Evaluate each side independently: opposite finishing deviations can cancel.
    second=(f"{c['home']} scored {hg-h:+.2f} relative to its xG; "
            f"{c['away']} scored {ag-a:+.2f}. These single-match differences describe execution and variation, not stable finishing ability.")
    return first+' '+second

def state_read(c):
    parts=[]
    for side in ('home','away'):
        states=[]
        for state in ('drawing','leading','trailing'):
            x=c.get(f'{side}_game_state_{state}_xG')
            if x is not None:
                states.append(f"{float(x):.2f} while {'level' if state=='drawing' else state}")
        if states:parts.append(f"{c[side]} generated "+', '.join(states)+'.')
    return ' '.join(parts) or "Score-state splits are unavailable for this dataset."

def visual_section(path):
    s=Path(path).stem.lower()
    if 'player_radars' in Path(path).parts or 'player_profiles' in Path(path).parts:return 'Player Impact Appendix'
    groups=[('Pressing and Rest Defence',['press','regain','loss','defensive','turnover']),
            ('Transitions and Efficiency',['transition','sequence_types','possession_speed','regain_speed']),
            ('Chance Creation',['shot','goalkeeper','zone14','cross','box_entries','funnel','entry_routes','set_piece']),
            ('Possession and Progression',['pass','progress','average_positions','xt_map','dominating','attacking_zones','pitch_control','unlocking','playing_through','reception','player_involvement']),
            ('Match Story',['xg_flow','match_statistic','goal','momentum','game_state','win_probability','dashboard','history','substitution','sequence_story'])]
    for section,tokens in groups:
        if any(t in s for t in tokens):return section
    return 'Player Impact Appendix' if 'player_' in s else 'Match Story'

RULES=[
    ('goalkeeper', [('xGoT',2,''),('on_target',0,'')], 'Local post-shot estimate',
     'The placement model has not been calibrated and does not know shot velocity or the goalkeeper location. Check individual shots before judging shot-stopping.'),
    ('shot_map',[('shots',0,''),('xG',2,''),('xG_per_shot',3,'')], 'Shot volume and chance quality',
     'Separate central close-range attempts from wider and longer shots. Review the preceding actions to establish how each chance was created.'),
    ('xg_flow',[('xG',2,''),('shots',0,'')], 'When shots added chance value',
     'Steps record shots; flat periods mean no additional recorded shot xG, not proof that possession had no attacking value.'),
    ('xt_map',[('sequence_xT',2,'')], 'Where successful movement added threat',
     'Origin zones show accumulated positive movement value. A connected route needs sequence evidence; adjacent hot cells alone do not establish one.'),
    ('pass_network',[('progressive_passes',0,'')], 'Passing connections and participation',
     'Nodes summarize recorded touches and links summarize inferred receivers. Positions are averages, not simultaneous formations. Compare the same half and minutes on the pitch.'),
    ('average_positions',[('touches',0,'')], 'Average locations of recorded touches',
     'An average touch position is not a defensive line or an off-ball position. Substitutes can occupy the same area at different times.'),
    ('pitch_control',[('influence_home',1,'')], 'Average-position influence',
     'This illustrative surface averages short touch windows by duration, split at substitutions and cards. Unobserved off-ball positions remain unknown. Field tilt is a separate pass-based measure.'),
    ('defensive_shape',[('ppda',2,'')], 'Recorded defensive action height and touch spread',
     'Defensive action locations and touch IQR do not directly measure distances between defensive units.'),
    ('playing_through',[('progressive_passes',0,''),('deep_completions',0,'')], 'Progression across an estimated line',
     'The reference line is inferred from defensive actions. Review the sequence or video before calling an action a confirmed line break.'),
    ('unlocking',[('deep_completions',0,'')], 'Receptions near an estimated defensive line',
     'These locations are candidates for review. The event data does not establish receiver body orientation, defender distances or a free player between lines.'),
    ('press_triggers',[('high_regains',0,'')], 'Actions preceding high regains',
     'The preceding opponent event is an association, not evidence of a coached trigger. Receiving orientation and pressing cover require video or tracking.'),
    ('ppda',[('ppda',2,''),('high_regains',0,''),('counterpress_success_rate',1,'%')], 'Pressure frequency and its outcomes',
     'Read passes per defensive action alongside recoveries and subsequent chances. A lower PPDA alone does not establish a better press.'),
    ('high_regains',[('high_regains',0,''),('regain_to_shot_rate',1,'%')], 'What followed high regains',
     'Check whether recoveries produced box access or shots before the opponent reorganised. Low counts should remain visible beside percentages.'),
    ('cross',[('crosses',0,''),('completed_crosses',0,'')], 'Wide delivery and completion',
     'Completion does not establish chance quality. Trace the next action to distinguish a controlled reception, a shot or a recycled attack.'),
    ('box_entries',[('box_entries',0,''),('box_entry_to_shot_rate',1,'%')], 'Access to the penalty area',
     'Compare entry routes with the shots that followed in the same possession. A raw entry total and a possession conversion rate use different denominators.'),
    ('progressive',[('progressive_passes',0,''),('deep_completions',0,'')], 'Progression and deeper access',
     'Distance gained is observable; pressure eliminated and receiver support require additional evidence.'),
    ('ball_losses',[('rest_defence_exposures',0,''),('rest_defence_dangerous_counters',0,'')], 'The consequences of advanced losses',
     'Review the first opponent actions after each highlighted loss. The event record measures consequences, not the exact defensive structure behind the ball.'),
    ('transition',[('transitions',0,''),('transition_shots',0,''),('transition_xG',2,'')], 'Output from inferred transitions',
     'Compare frequency and shot yield separately. A larger total does not by itself establish faster decisions or better off-ball runs.'),
    ('zone14',[('deep_completions',0,''),('box_entries',0,'')], 'Central access and the next action',
     'Review how receptions connect to box entries and shots. Location alone cannot show whether the receiver faced forward or was under pressure.'),
    ('dominating',[('field_tilt',1,'%'),('xG',2,'')], 'Territory and attacking output',
     'Touch density, field tilt and chance quality answer different questions; agreement between them should be checked rather than assumed.'),
    ('pass_targets',[('pass_share',1,'%')], 'Pass destinations',
     'The destinations show where the ball was sent. They do not prove the intended receiver position when a pass failed.'),
    ('pass_map',[('pass_share',1,'%'),('progressive_passes',0,'')], 'Circulation and forward movement',
     'The highlighted movements locate passages for review. Establish the sequence and outcome before assigning a tactical function.'),
    ('action_value',[('sequence_xT',2,'')], 'Illustrative action value',
     'The zone-value model is explicit and uncalibrated. It is not fitted VAEP and should not serve as a universal player rating.'),
    ('win_probability',[], 'Illustrative result state',
     'This score-and-time heuristic is not a calibrated forecast. Probability estimates before full time should not be used as betting or performance claims.'),
]


# One MatchFacts per package, not one per board: reading fifty-one boards would
# otherwise parse the same five CSVs fifty-one times.
_FACTS={}


def _package_root(path):
    """The package a chart belongs to.

    A numbered board sits in the package; a player's card sits two levels down
    in player_profiles/<club>/. Taking the parent for both put the profile hook
    behind a match_info.json that was never going to be there.
    """
    directory = Path(path).parent
    for candidate in (directory, directory.parent, directory.parent.parent):
        if (candidate / 'match_info.json').exists():
            return candidate
    return None


def _narrated(path,c):
    """The narrative paragraph for one board, or "" when it cannot be written."""
    root=_package_root(path)
    if root is None:
        return ""
    if 'player_profiles' in Path(path).parts:
        # Not one of the numbered boards, so report_narrative has no writer for
        # it. Recomputing the caption here means the wording is the current one
        # rather than whatever was current the day the card was drawn.
        try:
            import json

            import pandas as pd

            from advanced_profiles import profile_caption
            info=json.loads((root/'match_info.json').read_text(encoding='utf-8-sig'))
            written=profile_caption(pd.read_csv(root/'events.csv'),
                                    pd.read_csv(root/'players.csv'),
                                    info, Path(path).stem.replace('_',' '))
            if written:
                return written
        except Exception:
            return ""
        return ""
    try:
        from match_facts import load_match
        from report_narrative import resolve, paragraphs_for
    except Exception:
        return ""
    key=str(root.resolve())
    if key not in _FACTS:
        try:
            _FACTS[key]=(load_match(root),sorted(root.glob('[0-9]*.png')))
        except Exception:
            _FACTS[key]=None
    loaded=_FACTS.get(key)
    if not loaded:
        return ""
    facts,boards=loaded
    try:
        names=[b.name for b in boards]
        position=names.index(Path(path).name) if Path(path).name in names else None
        following=resolve(boards[position+1],facts) if (
            position is not None and position+1<len(boards)) else None
        written=paragraphs_for(resolve(Path(path),facts),facts,following)
    except Exception:
        return ""
    return ' '.join(t.strip() for t in written if t and t.strip()).strip()


def reading(path,c):
    path=Path(path);s=path.stem.lower()
    # report_narrative is the only writer. It covers all fifty-one boards,
    # computes its numbers from the package the chart was drawn from, names the
    # players and the minutes, and closes each paragraph on the figure that
    # follows. Where a board has no dedicated writer it falls back inside
    # paragraphs_for to that chart's own stored reading, so this always
    # returns something and the old templated recital below is unreachable.
    #
    # What is below it is kept only because two callers still import the RULES
    # table for its titles. As prose it is a list of the figures already
    # printed on the picture followed by a methodology note identical in every
    # match, which is what this replaced.
    described=_narrated(path,c)
    if described:return described
    contract=c.get('chart_contracts',{}).get(path.name)
    if contract:return (contract.get('reading',contract.get('takeaway',''))+' '+contract.get('interpretation','')).replace(' | ','; ')
    return result_read(c)


def commentary_title(path,c):
    contract=c.get('chart_contracts',{}).get(Path(path).name)
    if contract:return contract['title']
    s=Path(path).stem.lower()
    if 'game_state' in s:return 'Output under each score state'
    if 'player_radars' in Path(path).parts:return Path(path).stem.replace('_',' ')+' in this match'
    for token,_,title,_ in RULES:
        if token in s:return title
    return Path(path).stem.split('_',1)[-1].replace('_',' ').capitalize()

SECTION_METRICS={
 'Match Story':['shots','xG','big_chances','on_target'],
 'Chance Creation':['xG_per_shot','box_entries','deep_completions','completed_crosses'],
 'Possession and Progression':['field_tilt','final_third_entries','progressive_passes','build_up_success_rate'],
 'Pressing and Rest Defence':['ppda','high_regains','counterpress_success_rate','rest_defence_vulnerability'],
 'Transitions and Efficiency':['transition_shot_rate','transition_shots','transition_xG','transitions'],
 'Player Impact Appendix':['shots','progressive_passes','deep_completions','sequence_xT']}

def section_copy(c):
    questions={
        'Match Story':'Locate the phases that produced chances and compare their score states.',
        'Chance Creation':'Trace the strongest chances back to the entry route and the final pass.',
        'Possession and Progression':'Check which recorded passing routes continued into box access.',
        'Pressing and Rest Defence':'Review the first opponent actions after advanced losses and high regains.',
        'Transitions and Efficiency':'Compare the number of transitions with their chance output and exposure time.',
        'Player Impact Appendix':'Read each profile through minutes, role and the size of the comparison pool.'}
    result={}
    for section,keys in SECTION_METRICS.items():
        data=[]
        for key in keys:
            digits=2 if 'xG' in key or key=='ppda' else 1 if any(x in key for x in ['rate','tilt','vulnerability']) else 0
            unit='%' if digits==1 else ''
            data.append((label(key),comparison(c,key,digits,unit)))
        # The report's section page carried three rows that were a metric, a
        # score-state note and a question -- the same three shapes whatever the
        # match did. The tactical reading replaces the first with what this
        # match's own figures mean, and it is the same paragraph the article
        # opens that section on, so the two documents argue one case.
        from match_prose import section_reading, handoff
        reading = section_reading(c, section)
        rows=[('Observed output',data[0][1])]
        if reading:
            rows.append(('What the figures mean', reading))
        rows.append(('Score-state context',state_read(c)))
        rows.append(('Question for review',questions[section]))
        result[section]={'subtitle':questions[section], 'data':data,
            'performance':rows,
            'reading':reading,
            'handoff':handoff(section),
            'implication':questions[section]}
    return result
