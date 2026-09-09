"""Editorial charts with explicit observations, denominators and small-sample rules."""
from pathlib import Path
import json
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from match_insights import build_insights, stage_counts, flags, numeric
from visualization_components import C_HOME, C_AWAY, IS_LIGHT_THEME, BG_DARK, TEXT_MAIN, TEXT_DIM
from modern_chart_components import paired_dots, stage_flow

BRAND = '#137F82' if IS_LIGHT_THEME else '#61D7CC'
BG = BG_DARK
FG = TEXT_MAIN
MUTED = TEXT_DIM


class Charts:
    def __init__(self, out, info):
        self.out = Path(out)
        self.info = info
        self.paths = []
        self.contracts = {}
        self.skipped = []
        self.names = {info['home_id']: info['home_name'], info['away_id']: info['away_name']}
        from visual_redesign_full import _resolve_fixture_colors, lift_to_floor
        home, away = _resolve_fixture_colors(info)
        self.colors = {info['home_id']: lift_to_floor(home), info['away_id']: lift_to_floor(away)}

    def figure(self, title, subtitle, columns=1, height=7):
        """One figure, wearing the same head as every other page in the package.

        These charts carried their own header — a teal "MOSTAFA SAAD / MATCH
        STUDY" line, no crests, no scoreline, its own type scale — so a reader
        turning from page 41 to page 42 of one PDF crossed into what looked
        like a different publication. The pages either side of that boundary
        describe the same match.

        The shared strip is the one already on the other forty: badge, section
        label, title, subtitle, both crests and the score. It occupies the top
        of the figure, so the axes start lower than they used to.
        """
        fig, axes = plt.subplots(1, columns, figsize=(14, height), squeeze=False)
        fig.patch.set_facecolor(BG)
        # The shared strip ends at 0.865, so anything below 0.79 leaves a
        # visible band of empty page between the head and the first axis.
        fig.subplots_adjust(left=.09, right=.955, bottom=.17, top=.79, wspace=.30)
        from visual_redesign_preview import amoled_header
        amoled_header(fig, title, subtitle)
        for ax in axes.flat:
            ax.set_facecolor(BG)
            ax.tick_params(colors=MUTED, labelsize=10)
            ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
            for spine in ax.spines.values(): spine.set_color(MUTED)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(alpha=.12, color=MUTED)
            ax.set_axisbelow(True)
        return fig, axes[0]

    def save(self, fig, filename, title, reading, method, n):
        footer = f"{self.info['home_name']} v {self.info['away_name']}  |  {method}"
        fig.text(.055, .055, textwrap.fill(footer, 145), color=MUTED, size=9, va='bottom')
        path = self.out / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor=BG)
        plt.close(fig)
        self.paths.append(path)
        self.contracts[path.name] = {'title': title, 'reading': reading, 'method': method, 'n': int(n)}
        return path

    def scatter(self, frame, x, y, filename, title, xlabel, ylabel, *, player=False, size=None):
        required = [x, y, 'team_id']
        data = frame.dropna(subset=required).copy() if all(k in frame for k in required) else pd.DataFrame()
        if player and not data.empty:
            data = data[(data.minutes >= 30) & ~data.role.astype(str).str.upper().isin(['GK', 'GOALKEEPER'])]
        if len(data) < (8 if player else 5):
            if player and len(data):
                fig, axes = self.figure(title, f'Small sample: {len(data)} eligible players; raw observations replace scatter quadrants')
                ax = axes[0]; ax.axis('off')
                ordered = data.sort_values(y, ascending=False)
                for i, row in enumerate(ordered.itertuples()):
                    ax.text(0, 1-i/max(len(data), 8), f'{row.player}  |  {getattr(row,x):.2f} {xlabel}  |  {getattr(row,y):.2f} {ylabel}',
                            transform=ax.transAxes, color=self.colors[row.team_id], size=11)
                return self.save(fig, filename, title, f'Only {len(data)} players meet the minimum minutes and metric availability. Read individual values; no quadrant ranking is drawn.', 'Minimum 30 minutes; undefined ratios omitted', len(data))
            self.skipped.append({'chart': filename, 'reason': f'Only {len(data)} valid observation{"" if len(data)==1 else "s"}'})
            return
        # "1 observations" reached a shipped article; a single eligible point
        # is rare but legal, so the noun agrees with the count.
        observed=len(data)
        fig, axes = self.figure(title, f'{observed} observation{"" if observed==1 else "s"}  |  '+('Players with at least 30 minutes; match totals' if player else 'One point per possession; both teams share the same axes'),height=8)
        fig.subplots_adjust(bottom=.23)
        ax = axes[0]
        for marker, (tid, group) in zip(['o','s'],data.groupby('team_id')):
            sizes = 35+90*group[size]/max(data[size].max(), 1) if size else 52
            ax.scatter(group[x], group[y], s=sizes, c=self.colors[tid], label=self.names[tid], marker=marker, alpha=.80, edgecolors=FG, linewidths=.5)
        ax.set_xlabel(xlabel, labelpad=8); ax.set_ylabel(ylabel, labelpad=8)
        ax.set_xlim(left=min(0, data[x].min()*.95)); ax.set_ylim(bottom=min(0, data[y].min()*.95))
        if player:
            ax.margins(x=.15,y=.25)
            ax.set_xlim(left=-max(data[x].max(),1)*.04)
            ax.set_ylim(bottom=-max(data[y].max(),.01)*.08)
            ax.axvline(data[x].median(), color=MUTED, ls=':', lw=1)
            ax.axhline(data[y].median(), color=MUTED, ls=':', lw=1)
            from scatter_labels import label_players
            label_players(ax,data,x,y,color=FG,background=BG,fontsize=9)
        ax.legend(facecolor=BG, labelcolor=FG, edgecolor='none', fontsize=9,loc='upper right',bbox_to_anchor=(1,1.14),ncol=2)
        method = ('Dotted lines: eligible-player medians. Bubble area: '+size if size else 'Dotted lines: eligible-player medians.') if player else 'No fitted trend or causal interpretation.'
        if y=='xA':method+=' Inferred xA = shot xG credited to a prior successful key pass in the same possession (≤15 seconds).'
        if y=='positive_xT':method+=' Positive xT = added local zone threat from successful passes/carries; not goals or net value.'
        if y=='xG_per_shot':method+=' xG per shot = total shot xG / attempts; higher means better average pre-shot chance quality.'
        if y=='seconds_to_third':method+=' Time runs from regain to first final-third presence; regains without arrival remain in the source table.'
        med=data.groupby('team_id')[[x,y]].median()
        # This paragraph is what the article and the PDF print under the figure.
        # It was written in Arabic inside an otherwise English document, and
        # what it said was two medians: "Tottenham يميل إلى progressive passes
        # 1.00 وinferred xa 0.00". A scatter's argument is which quadrant the
        # points fall in and whether the two axes agree, so that is what it
        # argues now.
        parts=[]
        if len(med)>=2:
            lead_x=self.names[med[x].idxmax()];lead_y=self.names[med[y].idxmax()]
            if lead_x==lead_y:
                parts.append(f'{lead_x} led on both axes: its typical player did more of the '
                             f'{xlabel.lower()} and got more out of it. Volume and value agreeing '
                             f'is the simple case, and it puts the difference between the sides in '
                             f'the players rather than in the route they were given.')
            else:
                parts.append(f'{lead_x} led {xlabel.lower()} and {lead_y} led {ylabel.lower()}, so the '
                             f'side doing more of it was not the side getting more from it. Doing a '
                             f'thing often and doing it profitably are separate claims, and only the '
                             f'second one is on the vertical axis.')
        if player and len(data):
            both=data[(data[x]>data[x].median())&(data[y]>data[y].median())]
            if len(both):
                picked=both.sort_values([y,x],ascending=False).head(3)
                names=', '.join(str(row.player) for row in picked.itertuples())
                parts.append(f'Above both medians: {names}. Those are the players the two measures '
                             f'agree on. Anyone high on one axis alone was doing half of what the '
                             f'chart asks about, which is the half worth checking on video.')
            else:
                parts.append('Nobody cleared both medians, so no player on either side combined the '
                             'volume with the value. The chart is describing a match in which the '
                             'work and the return sat with different people.')
        parts.append('Minutes and role move a player across this chart on their own, so read a '
                     'position through both.' if player else
                     'Follow the sequence behind a point before reading a tactical cause into it.')
        reading=' '.join(parts)
        return self.save(fig, filename, title, reading, method, len(data))


def role_group(role):
    role = str(role).upper()
    if role in ['SUB','SUBSTITUTE','UNKNOWN','NAN','NONE']: return 'Unknown'
    if role in ['GK', 'GOALKEEPER']: return 'Goalkeeper'
    if role.startswith('D') and not role.startswith('DM'): return 'Defender'
    if role.startswith('F') or role in ['ST','SS','STRIKER']: return 'Forward'
    if role.startswith(('M', 'AM', 'DM')): return 'Midfielder'
    return 'Unknown'


from advanced_profiles import compact_profiles


def build_insight_visuals(events, players, info, out):
    # Remove the old post-49 names from copied staging packages. Otherwise a
    # rebuild would contain both the contiguous series and stale 54–65 files.
    for old_number in range(54,66):
        for legacy in Path(out).glob(f'{old_number}_*.png'):
            legacy.unlink(missing_ok=True)
    data = build_insights(events, players, info)
    charts = Charts(out, info)
    p = data['possessions']; people = data['players']
    charts.scatter(people,'progressions','xA','50_player_progression_creation.png','Progression and chance creation','Progressive passes + explicit carries','Inferred xA',player=True,size='touches')
    charts.scatter(people,'touches','positive_xT','51_player_involvement_value.png','Involvement and added threat','Recorded touches','Positive successful-movement xT',player=True)
    charts.scatter(people[people.shots > 0],'shots','xG_per_shot','52_player_shot_quality.png','Shot volume and average quality','Shots','xG per shot',player=True)
    fig,axes=charts.figure('Possession value by duration','Each box shows the distribution of positive xT; counts retain the number of possessions in each duration band.',columns=2)
    for ax,(tid,name) in zip(axes,charts.names.items()):
        own=p[p.team_id.eq(tid)]
        bands=[own.duration.le(10),own.duration.gt(10)&own.duration.le(20),own.duration.gt(20)&own.duration.le(40),own.duration.gt(40)]
        groups=[own.loc[mask,'positive_xT'].dropna().to_numpy() for mask in bands]
        for i,values in enumerate(groups):
            if len(values)>=4:
                ax.boxplot([values],positions=[i],widths=.45,patch_artist=True,boxprops={'facecolor':charts.colors[tid],'alpha':.5},medianprops={'color':FG},whiskerprops={'color':MUTED},capprops={'color':MUTED},flierprops={'marker':'.','markeredgecolor':charts.colors[tid]})
            elif len(values): ax.scatter([i]*len(values),values,color=charts.colors[tid])
        ax.set_xticks(range(4),[f'{label}\nn={len(v)}' for label,v in zip(['0–10s','10–20s','20–40s','40s+'],groups)])
        ax.set_title(name,color=FG);ax.set_ylabel('Positive xT');ax.set_xlim(-.6,3.6)
    charts.save(fig,'53_possession_speed_value.png','Possession value by duration','Compare typical value and its spread within each duration group.','Box = middle 50%; line = median. Groups below four observations show individual points.',len(p))
    regains = p[p.is_regain]
    fig,axes=charts.figure('Regains and time to reach the final third','All regains included; time starts at the recorded regain, including immediate final-third access.',columns=2)
    for ax,(tid,name) in zip(axes,charts.names.items()):
        times=regains.loc[regains.team_id.eq(tid),'seconds_to_third']
        counts=[int(times.le(5).sum()),int((times.gt(5)&times.le(12)).sum()),int(times.gt(12).sum()),int(times.isna().sum())]
        ax.barh(np.arange(4),counts,color=charts.colors[tid],height=.48)
        ax.set_yticks(np.arange(4),['0–5 seconds','6–12 seconds','Over 12 seconds','Did not reach'],color=FG)
        ax.set_ylim(3.6,-.6);ax.set_xlim(0,max(counts+[1])*1.4)
        ax.set_title(f'{name} · {len(times)} regains',color=FG)
        ax.set_xlabel('Regains')
        for i,n in enumerate(counts): ax.text(n+.2,i,f'{n} / {len(times)}',va='center',color=FG,size=10)
    charts.save(fig,'54_regain_speed.png','Regain progression time','Counts include regains that never reached the final third.','No arrival is retained as a separate category; zero seconds means regained inside the final third.',len(regains))
    fig, axes = charts.figure('From possession to a shot on target','Nested stages: every later stage belongs to the same possession and occurs after the preceding stage',columns=2)
    fig.subplots_adjust(left=.17,right=.96,wspace=.65)
    readings=[]
    for ax,(tid,name) in zip(axes,charts.names.items()):
        counts = stage_counts(p,tid)
        stage_flow(ax,counts,charts.colors[tid],FG,MUTED)
        ax.set_xlim(-4,max(p.groupby('team_id').size().max(),1)*1.5);ax.set_title(name,color=FG,size=14)
        # The noun leads the list. Written the other way round the sentence
        # ended "8, 1 possessions" whenever a funnel narrowed to one, which
        # agrees with the list but reads as a plural on the number beside it.
        readings.append(f'{name}: possessions at successive stages: '+', '.join(map(str,counts))+'.')
    charts.save(fig,'55_possession_funnel.png','Which possessions reached the box and a shot','This identifies where attacks stop: the largest drop between two stages is the access problem to review, rather than possession volume alone. '+' '.join(readings),'Rates use all possessions; one possession is counted once at each stage.',len(p))
    routes=data['routes']
    fig,axes=charts.figure('Box-entry routes and their next outcome','A shot must follow the entry in the same possession; multiple entries can precede one shot',columns=2)
    readings=[]
    for ax,(tid,name) in zip(axes,charts.names.items()):
        g=routes[routes.team_id.eq(tid)]
        counts=g.groupby('lane').shot_followed.agg(['size','sum']).reindex(['Left','Centre','Right'],fill_value=0)
        positions=np.arange(len(counts))
        for i,(lane,row) in enumerate(counts.iterrows()):
            total=float(row['size']); shot=float(row['sum']); rate=100*shot/total if total else 0
            ax.plot([0,total],[i,i],color=MUTED,lw=3,alpha=.45)
            ax.scatter(total,i,s=90,color=MUTED,zorder=3)
            ax.scatter(shot,i,s=90,marker='s',color=charts.colors[tid],zorder=4)
            ax.text(total+.35,i+.13,f'{int(total)} entries',color=FG,size=9,va='bottom')
            ax.text(shot+.35,i-.15,f'{int(shot)} shots · {rate:.0f}%',color=FG,size=9,va='top')
        ax.set_yticks(positions,counts.index,color=FG);ax.set_ylim(2.55,-.55)
        ax.set_xlim(0,max(float(counts['size'].max())*1.38,1));ax.set_title(name,color=FG);ax.set_xlabel('Entries and entries followed by a shot')
        ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=MUTED,markersize=8,label='All entries'),Line2D([0],[0],marker='s',color='none',markerfacecolor=charts.colors[tid],markersize=8,label='Shot followed')],facecolor=BG,labelcolor=FG,edgecolor='none',fontsize=9,loc='upper right')
        readings.append(f'{name}: {len(g)} entries, {int(g.shot_followed.sum())} followed by a shot in that possession.')
    charts.save(fig,'56_entry_routes.png','Entry route and subsequent shot','This compares the quality of access, not only the number of entries. More entries without more following shots means access did not become sustained threat. '+' '.join(readings),'The lane identifies entry origin; a following shot is descriptive, not proof of cause.',len(routes))
    # 57_game_state_rates was folded into the Game-State Output card, which
    # already listed the totals for the same three states. Two pages showing
    # the same split with different denominators made a reader hold both open
    # to answer one question; the rate now sits under the totals it qualifies.
    # Two charts were cut here rather than redrawn.
    #
    # 58_reception_map gave each side a scatter of every attacking-half
    # reception over a second panel of twelve arrows. Two hundred undifferentiated
    # dots on a blank rectangle is a texture, not a finding: nothing in it
    # separated a reception that mattered from one that did not, and the arrows
    # underneath repeated what the progression pages already show.
    #
    # 59_sequence_story drew the three largest chances as three pitches stacked
    # at a fifth of the page each, with the action list set beside them at 9pt.
    # At the size the layout allowed, neither the numbered dots nor the chain
    # text could be read, and the same three possessions are described in prose
    # in the article.
    loss=data['losses']
    fig,axes=charts.figure('Where possession was lost and what followed','Each point is a team loss. Colour shows the most advanced opponent outcome recorded within 12 seconds.',columns=2,height=7)
    summaries=[]
    loss_legend=[]
    for ax,(tid,name) in zip(axes,charts.names.items()):
        g=loss[loss.team_id.eq(tid)]
        values=[len(g),int(g.third_within_12.sum()),int(g.box_within_12.sum()),int(g.shot_within_12.sum())]
        ax.set_xlim(0,105);ax.set_ylim(0,68);ax.set_aspect('equal');ax.set_title(f'{name} · {len(g)} losses',color=FG)
        ax.set_xticks([]);ax.set_yticks([])
        ax.add_patch(Rectangle((0,0),105,68,fill=False,ec=MUTED,lw=.8))
        ax.axvline(52.5,color=MUTED,lw=.6,alpha=.7)
        ax.add_patch(Rectangle((88.5,13.84),16.5,40.32,fill=False,ec=MUTED,lw=.7))
        ax.add_patch(Rectangle((0,13.84),16.5,40.32,fill=False,ec=MUTED,lw=.7))
        # Highest consequence wins the colour, so each loss is plotted once.
        categories=[('No listed outcome',~(g.third_within_12 | g.box_within_12 | g.shot_within_12),MUTED,'o'),('Final third',g.third_within_12 & ~g.box_within_12 & ~g.shot_within_12,charts.colors[tid],'o'),('Box',g.box_within_12 & ~g.shot_within_12,'#F2B84B','s'),('Shot',g.shot_within_12,'#F06A6A','*')]
        for label_text,mask,color,marker in categories:
            h=g[mask]
            if len(h):
                ax.scatter(h.x*1.05,h.y*.68,s=38 if marker=='o' else 105,color=color,marker=marker,edgecolor=FG if marker=='o' else None,linewidth=.25,alpha=.9)
            if len(loss_legend) < len(categories):
                loss_legend.append(Line2D([0],[0],marker=marker,color='none',markerfacecolor=color,markeredgecolor=FG if marker=='o' else 'none',markersize=7 if marker=='o' else 10,label=label_text))
        ax.text(.02,.02,'Loss location · attacking direction →',transform=ax.transAxes,color=MUTED,size=8)
        summaries.append(f'{name}: {values[0]} open-play losses; {values[1]} followed by final-third presence, {values[2]} by box presence and {values[3]} by a shot within 12 seconds, totaling {g.xG_conceded.sum():.2f} xG.')
    fig.legend(handles=loss_legend,loc='upper center',bbox_to_anchor=(.5,.78),ncol=4,facecolor=BG,labelcolor=FG,edgecolor='none',fontsize=8,framealpha=.95)
    charts.save(fig,'60_loss_consequences.png','Loss locations and immediate opponent output',' '.join(summaries),'Each loss is plotted once using the most advanced outcome observed within 12 seconds; coordinates are event-inferred and do not establish pressure.',len(loss))
    # Equal, within-period windows, ending before another substitution cluster.
    a=data['actions']; subs=a[a.type.isin(['SubstitutionOn','SubstitutionOff'])]
    windows=[]
    for (period,clock),cluster in subs.groupby(['_period_order','_clock_seconds']):
        period_actions=a[a._period_order.eq(period)]
        clocks=sorted(subs[subs._period_order.eq(period)]._clock_seconds.unique())
        prev=max([t for t in clocks if t<clock]+[float(period_actions._clock_seconds.min())])
        nxt=min([t for t in clocks if t>clock]+[float(period_actions._clock_seconds.max())])
        duration=min(600,clock-prev,nxt-clock)
        if duration<180:continue
        before=period_actions[period_actions._clock_seconds.between(clock-duration,clock,inclusive='left')]
        after=period_actions[period_actions._clock_seconds.between(clock,clock+duration,inclusive='left')]
        confounds=period_actions[period_actions._clock_seconds.between(clock-duration,clock+duration)&(flags(period_actions,'is_goal')|period_actions.type.eq('Card'))]
        for tid in charts.names:
            for label,frame in [('Before',before),('After',after)]:
                own=frame[frame.team_id.eq(tid)]
                windows.append(dict(team_id=tid,minute=clock/60,window_minutes=duration/60,phase=label,xG=numeric(own[flags(own,'is_shot')],'xG').sum(),goals_or_cards=len(confounds)))
    window_frame=pd.DataFrame(windows)
    data['substitution_windows']=window_frame
    if len(windows):
        fig,axes=charts.figure('Output around substitution windows','Compare equal windows before and after each change. A marker is a descriptive comparison, not a substitution effect.',columns=2,height=7)
        for ax,(tid,name) in zip(axes,charts.names.items()):
            g=window_frame[window_frame.team_id.eq(tid)]
            pivot=g.pivot(index='minute',columns='phase',values='xG');x=np.arange(len(pivot))
            detail=g.drop_duplicates('minute').set_index('minute')
            categories=[f'{t:.0f}\' ±{detail.loc[t,"window_minutes"]:.1f}m'+(' *' if detail.loc[t,'goals_or_cards'] else '') for t in pivot.index]
            if len(pivot)<=2:
                yy=np.arange(len(pivot))
                for j,t in enumerate(pivot.index):
                    b=float(pivot.loc[t,'Before']); a=float(pivot.loc[t,'After'])
                    ax.plot([b,a],[j,j],color=MUTED,lw=2,zorder=1)
                    ax.scatter([b],[j],s=80,color=MUTED,zorder=3,label='Before' if j==0 else None)
                    ax.scatter([a],[j],s=80,color=charts.colors[tid],marker='s',zorder=3,label='After' if j==0 else None)
                    ax.text(b,j+.14,f'{b:.2f}',ha='center',color=FG,size=9)
                    ax.text(a,j-.20,f'{a:.2f}',ha='center',color=FG,size=9)
                ax.set_yticks(np.arange(len(categories)),categories,color=FG)
                ax.set_ylim(-.7,max(len(categories)-.3,.7))
                ax.legend(facecolor=BG,labelcolor=FG,edgecolor='none',loc='upper right')
            else:
                paired_dots(ax,categories,pivot.Before,pivot.After,[MUTED,charts.colors[tid]],FG,MUTED,labels=('Before','After'))
                ax.legend(facecolor=BG,labelcolor=FG,edgecolor='none')
            ax.set_title(name,color=FG);ax.set_xlabel('xG in equal windows')
        charts.save(fig,'61_substitution_windows.png','Equal windows around substitution clusters',f'{window_frame.minute.nunique()} substitution clusters have equal observable windows of at least three minutes before and after. Compare the raw outputs alongside score-state changes; a change is not a measured substitution effect.','* A goal or card occurs in the combined window. Nearby substitutions shorten the available window.',len(windows))
    else:charts.skipped.append({'chart':'61_substitution_windows.png','reason':'No isolated substitution cluster has at least three minutes on each side within the same period.'})
    # Historical comparisons require comparable, versioned snapshots; old files
    # with unknown models must not silently form a benchmark.
    from metric_registry import METRIC_VERSION
    history=[]
    history_root=Path(out).resolve()
    root=next((p for p in history_root.parents if p.name=='output'),None)
    if root and info.get('competition') and info.get('xg_model_version'):
        for manifest_path in root.rglob('package_manifest.json'):
            if any(part.startswith('.') for part in manifest_path.relative_to(root).parts):continue
            try:
                manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
                if manifest.get('metric_version')!=METRIC_VERSION or manifest.get('fixture',{}).get('competition')!=info['competition'] or manifest.get('fixture',{}).get('xg_model_version')!=info['xg_model_version']:continue
                directory=manifest_path.parent
                tm=pd.read_csv(directory/'team_advanced_metrics.csv'); xg=pd.read_csv(directory/'xg.csv')
                for side in ['home','away']:
                    match=manifest['fixture']; name=match[side+'_name']
                    t=tm[tm.side.eq(side)];shot=xg[xg.team.eq(name)]
                    if t.empty or shot.empty:continue
                    from match_report import compute_ppda_both
                    ppda=compute_ppda_both(match,pd.read_csv(directory/'events.csv'))
                    history.append(dict(team_id=match[side+'_id'],team=name,date=match.get('date',''),field_tilt=float(t.iloc[0].field_tilt),xG=float(shot.iloc[0].xG),ppda=ppda[side].get('ppda'),risk=float(t.iloc[0].rest_defence_vulnerability)))
            except (ValueError,KeyError,OSError):continue
    hist=pd.DataFrame(history)
    if len(hist):hist=hist.drop_duplicates(['team','date'])
    data['history_comparisons']=hist
    # Ten matches means at least twenty team observations, not ten points.
    if len(hist)>=20:
        fig,axes=charts.figure('Territory and chances in the saved competition','Comparable metric version; one point per team-match, not the two teams of one fixture')
        ax=axes[0];ax.scatter(hist.field_tilt,hist.xG,color=MUTED,alpha=.45,s=28)
        for tid,name in charts.names.items():
            g=hist[hist.team.eq(name)];ax.scatter(g.field_tilt,g.xG,color=charts.colors[tid],s=55,label=name)
        ax.set_xlabel('Field tilt (%)');ax.set_ylabel('xG');ax.legend(facecolor=BG,labelcolor=FG)
        charts.save(fig,'66_history_territory_chances.png','Territory and chance output across saved matches',f'{len(hist)} unique team-date observations use the same competition and metric version. Differences in opponents, score states and xG source still require checking before causal or forecasting conclusions.','History is descriptive; input-model provenance is listed in each manifest.',len(hist))
        valid=hist.dropna(subset=['ppda','risk'])
        if len(valid)>=20:
            fig,axes=charts.figure('Pressing frequency and danger after losses','Same competition and metric version | one point per team-match')
            ax=axes[0];ax.scatter(valid.ppda,valid.risk,color=MUTED,alpha=.55,s=30)
            for tid,name in charts.names.items():
                g=valid[valid.team.eq(name)];ax.scatter(g.ppda,g.risk,color=charts.colors[tid],s=55,label=name)
            ax.set_xlabel('PPDA (lower = fewer opponent passes per action)');ax.set_ylabel('Dangerous counters / advanced losses (%)');ax.legend(facecolor=BG,labelcolor=FG)
            charts.save(fig,'67_history_pressing_risk.png','Pressing frequency and turnover risk',f'{len(valid)} comparable team-match observations. A relationship here cannot isolate the effect of pressing height or off-ball cover.','Each axis keeps its own denominator; not a causal model.',len(valid))
        else:charts.skipped.append({'chart':'67_history_pressing_risk.png','reason':'Fewer than 20 valid PPDA/risk observations.'})
    else:
        for name in ['66_history_territory_chances.png','67_history_pressing_risk.png']:
            charts.skipped.append({'chart':name,'reason':f'Need 20 comparable team-match observations, named competition and matching explicit xg_model_version; found {len(hist)}.'})
    compact_profiles(charts,people,events)
    target=Path(out)/'analysis_tables';target.mkdir(exist_ok=True)
    for key,frame in data.items():
        if key!='actions':frame.to_csv(target/(key+'.csv'),index=False,encoding='utf-8-sig')
    (target/'chart_contracts.json').write_text(json.dumps(charts.contracts,indent=2,ensure_ascii=False),encoding='utf-8')
    (target/'skipped_charts.json').write_text(json.dumps(charts.skipped,indent=2),encoding='utf-8')
    events.attrs['chart_contracts']=charts.contracts
    # Include every meaningful appearance in the publication. Brief cameos
    # still get a standalone file, while the main PDF/catalog keeps players
    # with at least 30 minutes so every full-match player is represented.
    eligible=set(re.sub(r'[^\w.-]+','_',name)+'.png'
                 for name in people.loc[people.minutes>=30,'player'])
    return [p for p in charts.paths if 'player_profiles' not in p.parts or p.name in eligible], data
