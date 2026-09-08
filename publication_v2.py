"""Short, source-led publication assembled from the same chart contracts as PDF."""
from pathlib import Path
from match_editorial import headline, result_read, state_read, section_copy, reading, visual_section


def observed_contrast(c,group):
    pairs={'Chance Creation':('box_entries','final_third_entries','box entries','final-third entries'),
           'Possession and Progression':('progressive_passes','deep_completions','progressive passes','deep completions'),
           'Pressing and Rest Defence':('high_regains','rest_defence_exposures','high regains','advanced-loss exposures'),
           'Transitions and Efficiency':('transitions','transition_shots','transitions','transition shots'),
           'Match Story':('shots','big_chances','shots','source-tagged big chances'),
           'Player Impact Appendix':('touches','progressive_passes','touches','progressive passes')}
    first,second,first_label,second_label=pairs[group]
    h,a=c.get('home_'+first,0),c.get('away_'+first,0)
    leader=c['home'] if h>a else c['away'] if a>h else 'Neither team'
    opening=f"{leader} recorded more {first_label}" if h!=a else f"The teams were level on {first_label}"
    opening+=f": {c['home']} had {h:.0f} and {c['away']} {a:.0f}. The accompanying {second_label} totals were {c.get('home_'+second,0):.0f} and {c.get('away_'+second,0):.0f}. "
    interpretations={
      'Chance Creation':'The distinction is between arriving near the attack and accessing the penalty area. Those event totals can include repeated entries in one possession. The nested funnel below therefore asks a different question: how many distinct possessions continued from territory into the box and then a shot. Read both before deciding whether an attack lacked access or failed after gaining it.',
      'Possession and Progression':'Progression and deep completion are different stages of ball movement. A progressive pass can start far from goal; a deep completion places the next action much nearer the penalty area. The player scatter identifies who contributed to the first stage and who also supplied a shot-creating pass. Their minutes and roles remain part of that comparison.',
      'Pressing and Rest Defence':'Regains describe the ball being recovered, while the loss measure describes the opponent opportunities left behind. They are not opposite ends of one percentage. A useful review pairs the location of a high regain with its next attacking action, then checks the first opponent actions after an advanced loss. The event record cannot identify the spacing of players who never touched the ball.',
      'Transitions and Efficiency':'Frequency alone does not show the return from transition play. The conversion rate asks how many of those transition possessions contained a shot; xG then describes the quality of the attempts. Duration is shown separately so that a short possession with no chance value remains visible alongside the few direct attacks that produced a meaningful opportunity.',
      'Match Story':'Shot counts and big-chance tags give two perspectives on the attempts, and neither substitutes for the score-state timeline. A side can spend very different amounts of time level, ahead or behind. The rates below use the actual time observed in each state, so a short spell is not compared with most of the match as though the opportunities were equal.',
      'Player Impact Appendix':'The player comparisons break those team totals into individual observations. Being above the median on one axis describes this match sample and does not rank players across positions or across a season. Start with the role, then the minutes, and finally inspect the actions on the profile map to see where the contribution occurred.'}
    return opening+interpretations[group]


def review_method(c,group):
    """A distinct, actionable reading method for each analytical section."""
    methods={
      'Chance Creation': 'Use the shot map to connect access with chance quality: compare distance, angle and the final action before the attempt. A route that repeatedly reaches the area but ends in a blocked or low-value shot raises a different review question from a route that never crosses the boundary. Keep crosses and cut-back candidates separate when checking the underlying actions. Several entries may belong to one attack, so entry-to-shot associations must not be added as though they were unique chances.',
      'Possession and Progression': 'The next comparison is between the amount of involvement and the value added by that involvement. Positive xT rewards successful movement into a more threatening zone; it does not deduct every failed action and it is not measured in goals. A player can therefore accumulate a high total through repeated circulation. Inspect xT per 100 touches in the individual profile alongside the raw total, and read the pass network within one half so that different periods of participation are not mistaken for simultaneous positions.',
      'Pressing and Rest Defence': 'PPDA describes the frequency of defensive actions relative to opponent passing in the defined zone. It should be read with regain outcomes because frequent attempts do not necessarily recover the ball. The loss tiles use a different denominator: each tile is a possession loss, with the highlighted subset showing the opponent outcome inside twelve seconds. The final-third, box and shot rows are independent outcomes. Their percentages cannot be summed, and the position of the lost ball does not reveal the cover supplied by teammates.',
      'Transitions and Efficiency': 'Start with the regains that did not reach the final third as well as those that did. The time-to-progress scatter contains only regains with an observed arrival; the table retains the unsuccessful journeys so that the denominator stays visible. For a fast successful attack, inspect whether the value came from one long pass, a carry or several short actions. A direct route can save time without producing a good shooting location, while a longer possession can still end in the strongest chance of the match.',
      'Match Story': 'Locate every goal against the cumulative xG curve before interpreting the next spell. The goal event belongs to the score state immediately before it; subsequent events belong to the new state. The before-and-after substitution chart uses equally observable windows and flags goals or cards inside them. A difference between the two dots describes output in those windows, not the effect of the substitution. Use the timestamps to inspect the change in the video and consider the opponent actions and score state at the same time.',
      'Player Impact Appendix': 'The advanced profile distinguishes individual action value from involvement credit. xGChain credits each participant with the non-penalty xG of a shot possession; xGBuildup removes the shooter and key-pass provider from that credit. Several teammates can receive the same possession value, so their totals cannot be added to reconstruct team xG. Percentile dots appear only with enough eligible players in the same broad role. Raw numbers remain visible beside them, and an unknown substitute position is kept unknown rather than assigned an attacking role.'}
    return methods[group]


def build_article(events, xg, team_metrics, player_metrics, match_info, out_dir, *, max_sections=5, players=None):
    from match_article import Article, Section
    from tactical_pdf_report import build_context
    c = build_context(events, xg, team_metrics, player_metrics, match_info)
    out = Path(out_dir)
    paths = sorted(out.glob('[0-9]*.png'))
    # Each figure has to earn a place in the argument. The PDF remains the reference.
    preferred = ['xg_flow', 'possession_funnel', 'entry_routes', 'game_state_rates',
                 'possession_speed_value', 'player_progression_creation', 'player_involvement_value',
                 'sequence_story', 'ppda_pressing', 'box_entries']
    selected=[]
    for token in preferred:
        candidate=next((p for p in paths if token in p.stem and p not in selected),None)
        if candidate is not None:selected.append(candidate)
    selected=selected[:10]
    sections=[Section('The result and the chances', [result_read(c),
        'Expected goals describe the chances recorded before their outcomes. The difference from the final score is worth explaining through individual attempts, including penalties and own goals where present. It cannot establish repeatable finishing quality from one match.'],
        [p for p in selected if 'xg_flow' in p.stem])]
    copies=section_copy(c)
    for group in ['Chance Creation','Possession and Progression','Pressing and Rest Defence','Transitions and Efficiency','Match Story','Player Impact Appendix']:
        chosen=[p for p in selected if visual_section(p)==group and 'xg_flow' not in p.stem]
        data=copies[group]
        paragraphs=[observed_contrast(c,group)]
        # The reading appears alongside its figure, so the main text develops the question.
        paragraphs.append(review_method(c,group)+' '+data['implication'])
        sections.append(Section('Player involvement' if group=='Player Impact Appendix' else group,paragraphs,chosen))
    profiles=sorted((out/'player_profiles').glob('*/*.png'))
    if profiles:
        # Select two leaders per side from the match observations, then show
        # their advanced profile cards. The ranking is explicitly match-only.
        from match_insights import player_observations
        if players is None:
            import pandas as pd
            players=pd.read_csv(out/'players.csv')
        observations=player_observations(events, players).copy()
        for key in ['positive_xT','xGChain','xA','xG','progressive_passes']:
            if key not in observations: observations[key]=0
        observations['selection_score']=(observations.positive_xT.fillna(0)+observations.xGChain.fillna(0)+2*observations.xA.fillna(0)+observations.xG.fillna(0)+.01*observations.progressive_passes.fillna(0))
        leaders=observations[observations.minutes>=20].sort_values(['team_id','selection_score'],ascending=[True,False]).groupby('team_id').head(2)
        leader_names={str(row.player) for _,row in leaders.iterrows()}
        chosen=[p for p in profiles if p.stem.replace('_',' ') in leader_names or p.stem.lower() in {n.lower().replace(' ','_') for n in leader_names}]
        sections.append(Section('Key players from both teams',[
            'The player cards combine a role-aware radar, raw match values and the player’s action map. The selected names are leaders in this match-only comparison using positive xT, sequence xG credit, xA, xG and progressive passing; minutes are required before selection.',
            'This is a shortlist for video review, not a season rating. The radar compares the player with eligible same-role observations when the sample is large enough, while the map shows where the recorded actions occurred. '+ ' '.join(f"{observations[observations.player.eq(name)].iloc[0].player} ({observations[observations.player.eq(name)].iloc[0].team_id})" for name in leader_names if not observations[observations.player.eq(name)].empty)+'.'],chosen))
    sections.append(Section('What to take into the video review',[state_read(c),
        'Start with the three selected possession chains. Locate the entry, the reception and the shot in order, then review the surrounding video to test the proposed mechanism. Event data shows the recorded actions but cannot establish off-ball spacing, receiving orientation or the coach’s intent.',
        'Separate observed output from an explanation of why it happened. Compare attacks under the same score state, and keep the number of possessions or minutes beside every conversion rate. This makes the next review question specific without presenting an association as a cause.',
        'The complete chart set, role profiles and analysis tables accompany the reference PDF. Local xG, post-shot estimates, expected threat and average-position influence are model outputs, not tracking measurements or calibrated forecasts. Their definitions and versions are recorded in the package manifest.']))
    # The article now documents every exported figure, including the four
    # posters, so the Word copy can stand alone beside the PDF.
    all_visuals=sorted(out.glob('*.png'))+sorted((out/'player_profiles').glob('*/*.png'))
    already={p.resolve() for section in sections for p in section.visuals}
    appendix=[p for p in all_visuals if p.resolve() not in already]
    sections.append(Section('Complete visual guide',[
        'The following pages explain every exported visual in the match package. Read each title with its subtitle and footer: the subtitle defines the denominator or time window, while the footer states the main limitation.'],appendix,gallery=True))
    strap=' · '.join(str(v) for v in [match_info.get('competition',''),match_info.get('date',''),'MATCH STUDY'] if v)
    match_title,standfirst=match_headline(events,xg,team_metrics,player_metrics,match_info,out_dir)
    article=Article(match_title,standfirst or result_read(c),strap,sections,None,c['home'],c['away'],c)
    return article


def write_markdown(article,out):
    out=Path(out)
    lines=[f'# {article.title}',article.strap,article.standfirst]
    for section in article.sections:
        lines.extend(['',f'## {section.heading}',*section.paragraphs])
        for p in section.visuals:
            lines.extend([f'![{p.stem}]({p.relative_to(out).as_posix()})',reading(p,article.context)])
    (out/'match_article.md').write_text('\n\n'.join(lines),encoding='utf-8')


def match_headline(events,xg,team_metrics,player_metrics,info,out_dir='.'):
    """The fixture's own headline and standfirst, from the findings engine.

    This module replaced the v1 article, and in doing so it replaced the
    headline with ``f"{home} {score} {away} — Match study"``. That reads as a
    filename rather than a finding, and because ``match_article`` re-exports
    this module's names from its last line, the thirteen ``_finding_*``
    builders and the ranked ``_title_candidates`` behind them stopped being
    reachable from anywhere. Fifty-five rendered fixtures then shipped one
    sentence between them:

        AssertionError: 55 fixtures share only 1 sentences: X #–# X — Match study

    The section assembly below is still v2's. Only the title comes from the
    older engine, which is the part v2 never replaced.

    Both this and the article go through here so the cover and the Word file
    cannot open on different sentences about the same match.
    """
    if events is None or xg is None or not info.get('home_name') or not info.get('away_name'):
        return '',''
    from match_article import _Match, _title
    return _title(_Match(events,xg,team_metrics,player_metrics,info,out_dir))


def cover_headline(events,xg,team_metrics,player_metrics,info):
    return match_headline(events,xg,team_metrics,player_metrics,info)[0]


def pdf_verdict(self):
    from tactical_pdf_report import PAGE_W, PAGE_H, FOCUS, HOME, AWAY
    from html import escape
    self._start('final_verdict','Match conclusions')
    self._header('Match conclusions','Observed output, score-state context and questions for review','SYNTHESIS')
    self._card_box(42,490,PAGE_W-84,200,'Result and chance balance',FOCUS)
    self._paragraph(escape(result_read(self.context)),60,645,PAGE_W-120,130,self.body)
    self._card_box(42,285,PAGE_W-84,165,'Score-state context',HOME)
    self._paragraph(escape(state_read(self.context)),60,410,PAGE_W-120,110,self.body)
    self._card_box(42,80,PAGE_W-84,165,'Questions for the next review',AWAY)
    self._paragraph(escape('Trace the largest chances back to their entries and the first action after a regain. Compare the same score state and exposure minutes. Review video before attributing the result to off-ball spacing, pressing coordination or substitutions. These are testable review questions; the totals alone do not establish those causes.'),60,205,PAGE_W-120,110,self.body)
    self._finish()
