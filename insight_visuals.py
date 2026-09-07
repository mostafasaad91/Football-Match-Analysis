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
        fig, axes = plt.subplots(1, columns, figsize=(12, height), squeeze=False)
        fig.patch.set_facecolor(BG)
        fig.subplots_adjust(left=.09, right=.96, bottom=.19, top=.76, wspace=.30)
        fig.text(.055, .952, 'MOSTAFA SAAD  /  MATCH STUDY', color=BRAND, size=10, weight='bold')
        fig.text(.055, .89, title, color=FG, size=22, weight='bold')
        fig.text(.055, .825, subtitle, color=MUTED, size=10)
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
            self.skipped.append({'chart': filename, 'reason': f'Only {len(data)} valid observations'})
            return
        fig, axes = self.figure(title, f'{len(data)} observations  |  '+('Players with at least 30 minutes; match totals' if player else 'One point per possession; both teams share the same axes'),height=8)
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
        reading='؛ '.join(f'{self.names[tid]} يميل إلى {xlabel.lower()} {row[x]:.2f} و{ylabel.lower()} {row[y]:.2f}' for tid,row in med.iterrows())+'. '
        if len(med)>=2:
            reading += ('التفوق في المؤشرين معًا يربط المشاركة بالقيمة.' if med[x].idxmax()==med[y].idxmax()
                        else 'التفوق في الحجم لا يطابق التفوق في القيمة؛ المشاركة وحدها لا تشرح التأثير.')+' '
        reading += 'هذه قراءة تكتيكية للمباراة؛ الدقائق والدور يغيران مقارنة اللاعبين.' if player else 'تتبع التسلسل الذي أنتج النقطة قبل إسناد سبب تكتيكي.'
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
        readings.append(f'{name}: '+', '.join(map(str,counts))+' possessions at successive stages.')
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
    spells=data['spells']
    fig,axes=charts.figure('Chance production under each score state','Rates per 30 minutes in that state; spells shorter than 5 minutes do not receive rates',columns=2)
    readings=[]
    for ax,(tid,name) in zip(axes,charts.names.items()):
        g=spells[spells.team_id.eq(tid)].groupby('state')[['minutes','shots','xG']].sum().reindex(['level','leading','trailing'],fill_value=0)
        rates=(g.xG*30/g.minutes).where(g.minutes>=5)
        ax.hlines(np.arange(3),0,rates,color=charts.colors[tid],lw=2,alpha=.5)
        ax.scatter(rates,np.arange(3),s=100,color=charts.colors[tid],marker='D',zorder=3)
        ax.set_yticks(np.arange(3),g.index,color=FG);ax.set_ylim(2.6,-.6)
        ax.set_title(name,color=FG); ax.set_xlabel('xG / 30 state minutes')
        ax.set_xlim(0,max(float(rates.max()) if rates.notna().any() else 0,1)*1.5)
        for i,(state,row) in enumerate(g.iterrows()):
            value=rates.loc[state]
            text=f'{value:.2f}' if pd.notna(value) else 'Insufficient exposure'
            ax.annotate(f'{text}\n{row.minutes:.1f} min · {row.xG:.2f} xG',(value if pd.notna(value) else 0,i),xytext=(9,0),textcoords='offset points',va='center',color=FG,size=9)
        readings.append(name+': '+'; '.join(f'{s} for {r.minutes:.1f} minutes, {r.xG:.2f} xG' for s,r in g.iterrows())+'.')
    charts.save(fig,'57_game_state_rates.png','Score-state exposure and chance output','The split shows how chance output changed with the score. Never read the rate without exposure time: a short spell can inflate it. '+' '.join(readings),'Within-period clock; a goal belongs to the state before it. Zero exposure means unavailable.',len(spells))
    receptions=data['receptions']
    fig,axes=charts.figure('Reception locations and forward continuations','Top: all attacking-half receptions. Bottom: up to 12 longest forward actions from those receptions.',columns=2,height=10)
    for ax,(tid,name) in zip(axes,charts.names.items()):
        g=receptions[receptions.team_id.eq(tid)]
        g=g[g.x.between(50,100)&g.y.between(0,100)]
        left=.08 if tid==next(iter(charts.names)) else .56
        ax.set_position([left,.48,.38,.27])
        continuation=fig.add_axes([left,.14,.38,.25],facecolor=BG)
        continuation.set_xlim(50,100);continuation.set_ylim(0,100)
        continuation.set_xticks([]);continuation.set_yticks([])
        for spine in continuation.spines.values(): spine.set_color(MUTED)
        continuation.set_title('Selected forward actions →',color=FG,size=10)
        ax.set_xlim(50,100);ax.set_ylim(0,100)
        if len(g):
            # A raw arrow for every event turns this chart into a hairball. Use
            # a density layer for the complete sample and reserve arrows for
            # the most informative, longest forward continuations.
            ax.scatter(g.x,g.y,s=28,c=charts.colors[tid],alpha=.92,zorder=5,edgecolors=FG,linewidths=.25)
            links=g[g.next_type.isin(['Pass','Carry'])].dropna(subset=['end_x','end_y']).copy()
            if len(links):
                links['advance']=links.end_x-links.x
                # Keep annotation geometry inside the displayed attacking-half
                # panel; otherwise Matplotlib can draw an arrow across the
                # gap into the opponent's subplot.
                links=links[(links.advance>2)&links.x.between(50,100)&links.y.between(0,100)&links.end_x.between(50,100)&links.end_y.between(0,100)].nlargest(12,'advance')
                for r in links.itertuples():
                    continuation.scatter(r.x,r.y,s=24,color=charts.colors[tid],edgecolors=FG,linewidths=.3)
                    continuation.annotate('',(r.end_x,r.end_y),(r.x,r.y),annotation_clip=True,arrowprops={'arrowstyle':'->','lw':1.0,'color':charts.colors[tid],'alpha':.72})
        ax.set_title(f'{name} · {len(g)} receptions · {len(links)} continuations',color=FG,size=11)
        ax.set_xlabel('Attacking x (cropped at halfway)');ax.set_ylabel('Pitch y')
    charts.save(fig,'58_reception_map.png','Reception zones and selected continuations',f'{len(receptions)} pass-to-touch links meet the conservative spatial and time rules. Density shows every link; only the 12 longest forward continuations per team are drawn as arrows so the tactical pattern stays readable.','Event-inferred receiver; does not establish body orientation or pressure.',len(receptions))
    # Select complete real possessions; each row lists ordered actions and their clock.
    best=p.nlargest(3,'xG')
    fig,axes=charts.figure('Three possessions behind the chance totals','Recorded actions up to the final shot in each selected possession. Attacks →',columns=3,height=11)
    summaries=[]
    for i,(_,row) in enumerate(best.iterrows()):
        g=data['actions'];g=g[g.possession_id.eq(row.possession_id)&g.team_id.eq(row.team_id)]
        shot_positions=np.flatnonzero(flags(g,'is_shot').to_numpy())
        if len(shot_positions): g=g.iloc[:shot_positions[-1]+1]
        meaningful=g[g.type.isin(['Pass','Carry','TakeOn','Shot','Goal','SavedShot','MissedShots','ShotOnPost'])|flags(g,'is_shot')].tail(6)
        chain=' → '.join(f"{int(e.minute):02d}:{int(e.second):02d} {e.player} ({e.type})" for e in meaningful.itertuples())
        heading=f'{charts.names[row.team_id]} | {row.xG:.2f} xG | {row.duration:.0f}s | last {len(meaningful)} actions'
        ax=axes[i];ax.set_position([.055,.57-i*.22,.38,.19]);ax.set_xlim(0,105);ax.set_ylim(0,68);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        ax.add_patch(Rectangle((0,0),105,68,fill=False,ec=MUTED,lw=.7))
        ax.add_patch(Rectangle((88.5,13.84),16.5,40.32,fill=False,ec=MUTED,lw=.7))
        for j,(_,e) in enumerate(meaningful.iterrows(),1):
            ax.scatter(e.x*1.05,e.y*.68,s=80,color=charts.colors[row.team_id],edgecolor=FG,zorder=4)
            ax.annotate(str(j),(e.x*1.05,e.y*.68),xytext=(3,8),textcoords='offset points',color=FG,size=10,weight='bold',zorder=5)
            if e.type in ['Pass','Carry'] and pd.notna(e.end_x) and pd.notna(e.end_y):
                ax.annotate('',(e.end_x*1.05,e.end_y*.68),(e.x*1.05,e.y*.68),arrowprops={'arrowstyle':'->','color':charts.colors[row.team_id],'lw':1.5})
        ax.set_title(heading,color=FG,size=10,pad=8)
        from match_clock import event_label
        labels='\n'.join(textwrap.fill(f'{j}. {event_label(e)} {e.player} — {e.type}',35) for j,(_,e) in enumerate(meaningful.iterrows(),1))
        fig.text(.51,.75-i*.22,labels,color=FG,size=9,va='top',linespacing=1.4)
        summaries.append(heading+'. '+chain)
    charts.save(fig,'59_sequence_story.png','The action chains behind the largest chances',' '.join(summaries),'Elapsed event timestamps; arrows mean chronological order, not inferred off-ball runs.',len(best))
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
