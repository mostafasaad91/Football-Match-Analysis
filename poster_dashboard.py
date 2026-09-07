"""Four native statistical posters, with one layout for AMOLED and paper."""
from pathlib import Path
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from match_insights import build_insights, flags, numeric, stage_counts, successful
from visualization_components import BG_DARK, TEXT_MAIN, TEXT_DIM, IS_LIGHT_THEME
from metric_registry import format_value

POSTERS = ('match_poster_1_match_story.png','match_poster_2_progression.png',
           'match_poster_3_pressure_transitions.png','match_poster_4_players_finishing.png')


def _display_number(value):
    """Read the existing formatted value without turning unavailable data into zero."""
    try:
        result=float(str(value).replace(',','').removesuffix('%').removesuffix(' min').strip())
        return result if np.isfinite(result) else np.nan
    except (TypeError,ValueError):
        return np.nan


def _pair_scale(first,second):
    values=[_display_number(first),_display_number(second)]
    valid=[v for v in values if np.isfinite(v)]
    # Percentages with their own denominators are separate tracks out of 100.
    percent=all(str(v).endswith('%') for v in (first,second))
    ceiling=100.0 if percent and all(0<=v<=100 for v in valid) else max(valid+[0.01])
    return values,ceiling


class Poster:
    def __init__(self, info, number, title, subtitle, kpis):
        self.info=info; self.names={info['home_id']:info['home_name'],info['away_id']:info['away_name']}
        from visual_redesign_full import lift_to_floor
        self.colors={info['home_id']:lift_to_floor(info['home_color']),info['away_id']:lift_to_floor(info['away_color'])}
        self.fig=plt.figure(figsize=(16,20),facecolor=BG_DARK)
        self.number=number
        self.rule='#C6CBCB' if IS_LIGHT_THEME else '#303638'
        self.muted='#596266' if IS_LIGHT_THEME else '#ADB5B8'
        self.fit=[]
        self.notes=[]
        self._text(.045,.969,'TACTICAL / MATCH ANALYSIS',size=12,weight='bold',va='center')
        self._text(.955,.969,f'{number:02d} / 04',size=12,ha='right',va='center',color=self.muted)
        self._rule(.045,.955,.952)
        self._text(.045,.929,title,size=30,weight='bold',va='center',width=.91)
        self._text(.045,.904,subtitle,size=10.5,color=self.muted,va='center',width=.91)
        import crests
        for side,x,name_x,ha in [('home',.081,.13,'left'),('away',.919,.87,'right')]:
            tid=info[side+'_id']
            crests.place_crest(self.fig,x,.857,tid,monogram=info[side+'_name'][:3].upper(),
                colour=self.colors[tid],width=.060,background=BG_DARK,
                allow_download=info.get('allow_download',False))
            self._text(name_x,.856,info[side+'_name'],size=22,weight='bold',ha=ha,va='center',width=.245)
        self._text(.5,.859,info['score'].replace(':','–'),size=43,weight='bold',ha='center',va='center',width=.20)
        self._text(.5,.828,'  ·  '.join(filter(None,[info.get('competition'),info.get('date')])),
                   size=9,color=self.muted,ha='center',va='center',width=.82)
        # Mini-bars carry the comparison; the numbers are direct labels.
        for i,(label,h,a) in enumerate(kpis):
            x=.045+i*.232
            self._rule(x,x+.212,.806)
            self._text(x,.790,label.upper(),size=9.5,weight='bold',color=self.muted,width=.21)
            numbers,ceiling=_pair_scale(h,a)
            for yy,value,side,marker,v in [(.766,h,'home','o',numbers[0]),(.741,a,'away','s',numbers[1])]:
                tid=info[side+'_id']
                self.fig.add_artist(Line2D([x+.004],[yy+.003],transform=self.fig.transFigure,
                    marker=marker,color=self.colors[tid],markersize=5,linestyle='none'))
                self._text(x+.015,yy,info[side+'_name'],size=9,color=self.muted,width=.12)
                self._text(x+.21,yy,str(value),size=14,weight='bold',color=self.colors[tid],ha='right',width=.075)
                self.fig.add_artist(Rectangle((x,yy-.010),.21,.003,transform=self.fig.transFigure,facecolor=self.rule,edgecolor='none'))
                if np.isfinite(v):
                    self.fig.add_artist(Rectangle((x,yy-.010),.21*max(v,0)/ceiling,.003,
                        transform=self.fig.transFigure,facecolor=self.colors[tid],edgecolor='none'))
        self.axes=[]
        self.slots=[]
        # Headers use figure coordinates so equal-aspect pitches cannot move them.
        for row in range(3):
            top=.711-row*.219
            for col in range(2):
                left=.045+col*.482
                self.slots.append((left,top))
                self.axes.append(self.fig.add_axes([left+.038,top-.177,.390,.126]))
        for ax in self.axes:
            ax.set_facecolor(BG_DARK);ax.tick_params(colors=self.muted,labelsize=9,length=0,pad=6)
            ax.xaxis.label.set_color(TEXT_DIM);ax.yaxis.label.set_color(TEXT_DIM)
            ax.spines[['top','right']].set_visible(False)
            ax.spines[['left','bottom']].set_color(self.rule)
            ax.grid(color=self.rule,alpha=.55,lw=.6);ax.set_axisbelow(True)

    def _text(self,x,y,text,*,size=11,color=TEXT_MAIN,width=None,**kwargs):
        artist=self.fig.text(x,y,str(text),color=color,size=size,fontfamily='DejaVu Sans',**kwargs)
        if width:self.fit.append((artist,width))
        return artist

    def _rule(self,left,right,y):
        self.fig.add_artist(Line2D([left,right],[y,y],transform=self.fig.transFigure,color=self.rule,lw=.7))

    def panel(self,i,title,note):
        ax=self.axes[i]
        left,top=self.slots[i]
        self._rule(left,left+.428,top)
        self._text(left,top-.018,f'{i+1:02d}',size=9,color=self.muted,va='center')
        self._text(left+.024,top-.018,title,size=15,weight='bold',va='center',width=.404)
        self._text(left,top-.031,textwrap.fill(note,78),size=8.5,color=self.muted,va='top')
        self.notes.append({'panel':i+1,'title':title,'definition':note})
        return ax

    def comparisons(self,i,title,rows,note='Two bars per metric; shared scale within each row; — = unavailable'):
        ax=self.panel(i,title,note);ax.axis('off')
        left,top=self.slots[i]
        ax.set_position([left,top-.177,.428,.126])
        ax.set_xlim(0,1);ax.set_ylim(0,len(rows))
        for j,(label,h,a) in enumerate(rows):
            y=len(rows)-j-.5
            values,ceiling=_pair_scale(h,a)
            ax.text(0,y,textwrap.fill(label,24),color=TEXT_MAIN,size=9,va='center')
            for value,display,tid,dy,hatch in zip(values,[h,a],self.names, [.16,-.16],['','///']):
                ax.barh(y+dy,.48,left=.40,height=.16,color=self.rule,alpha=.30)
                if np.isfinite(value):
                    ax.barh(y+dy,.48*max(value,0)/ceiling,left=.40,height=.16,
                            color=self.colors[tid],edgecolor=BG_DARK,lw=.3,hatch=hatch)
                ax.text(.91,y+dy,str(display),color=self.colors[tid],size=10,weight='bold',va='center')
        self.notes[-1]['scale']='Each metric uses a separate zero-based scale shared by both teams; percentages use 0–100.'

    def ranking(self,i,title,frame,tid,ceiling):
        ax=self.panel(i,title,'Top five by xG, then xA | filled = xG; hatched = inferred xA')
        left,top=self.slots[i]
        ax.set_position([left+.155,top-.177,.273,.126])
        g=frame.sort_values(['xG','xA'],ascending=False).head(5)
        yy=np.arange(len(g))
        for key,dy,hatch in [('xG',-.17,''),('xA',.17,'///')]:
            values=numeric(g,key,np.nan)
            ax.barh(yy+dy,values,height=.25,color=self.colors[tid],edgecolor=BG_DARK,lw=.4,hatch=hatch)
            for y,v in zip(yy+dy,values):
                if np.isfinite(v):ax.text(v+ceiling*.025,y,f'{v:.2f}',color=TEXT_MAIN,size=8,va='center')
        ax.set_yticks(yy,[textwrap.shorten(str(r.player),width=26,placeholder='…')+f'\n{r.minutes:.0f} min' for r in g.itertuples()],size=8)
        ax.set_ylim(len(g)-.5,-.5);ax.set_xlim(0,ceiling*1.23)
        ax.set_xlabel('Expected goals / inferred assists',size=9)
        ax.grid(axis='y',visible=False)
        if g.empty:ax.text(.5,.5,'No player observations',transform=ax.transAxes,ha='center',color=TEXT_DIM)

    def state_durations(self,i,minutes):
        ax=self.panel(i,'Time spent in each score state','Minutes including added time | segment width = time observed')
        left,top=self.slots[i]
        ax.set_position([left+.083,top-.155,.345,.100])
        shades=['#8D9599','#3D454A','#C8CDCF']
        hatches=['','///','...']
        for row,series in enumerate(minutes):
            start=0
            for state,shade,hatch in zip(['level','leading','trailing'],shades,hatches):
                value=float(series.loc[state])
                ax.barh(row,value,left=start,height=.48,color=shade,edgecolor=BG_DARK,lw=.8,hatch=hatch,
                        label=state.title() if row==0 else None)
                if value>=8:
                    ax.text(start+value/2,row,f'{value:.0f}',ha='center',va='center',size=10,
                            color='white' if state=='leading' else '#111111')
                start+=value
        ax.set_yticks([0,1],list(self.names.values()),size=9)
        ax.set_ylim(1.6,-.6);ax.set_xlim(left=0);ax.set_xlabel('Observed minutes',size=9)
        ax.grid(axis='y',visible=False)
        ax.legend(loc='upper left',bbox_to_anchor=(0,-.30),ncol=3,frameon=False,
                  fontsize=8,labelcolor=TEXT_MAIN,handlelength=1.2,columnspacing=1)

    def bars(self,i,title,categories,home,away,note):
        ax=self.panel(i,title,note)
        left,top=self.slots[i]
        ax.set_position([left+.096,top-.177,.332,.126])
        from modern_chart_components import paired_dots
        paired_dots(ax,categories,home,away,list(self.colors.values()),TEXT_MAIN,TEXT_DIM,
                    labels=tuple(self.names.values()))

    def pitch(self,i,title,frame,tid,*,arrows=False):
        purpose=('Arrows = successful box entries' if arrows else
                 'Point area = shot xG; stars = goals' if 'is_shot' in frame else
                 'Point area = opponent xG in the next 12 seconds')
        ax=self.panel(i,title,purpose+' | attacks → | pitch: 105 × 68 m')
        left,top=self.slots[i]
        ax.set_position([left,top-.184,.428,.137])
        ax.set_xlim(-2,107);ax.set_ylim(-2,70);ax.set_aspect('equal');ax.axis('off')
        ax.add_patch(Rectangle((0,0),105,68,fill=False,ec=TEXT_DIM,lw=.8))
        ax.plot([52.5,52.5],[0,68],color=TEXT_DIM,lw=.7)
        ax.add_patch(Circle((52.5,34),9.15,fill=False,ec=TEXT_DIM,lw=.7))
        ax.add_patch(Rectangle((88.5,13.84),16.5,40.32,fill=False,ec=TEXT_DIM,lw=.8))
        ax.add_patch(Rectangle((0,13.84),16.5,40.32,fill=False,ec=TEXT_DIM,lw=.8))
        for xx in (0,99.5):ax.add_patch(Rectangle((xx,24.84),5.5,18.32,fill=False,ec=TEXT_DIM,lw=.7))
        ax.scatter([11,94],[34,34],s=5,color=TEXT_DIM)
        if arrows:
            for row in frame.dropna(subset=['x','y','end_x','end_y']).itertuples():
                ax.annotate('',(row.end_x*1.05,row.end_y*.68),(row.x*1.05,row.y*.68),arrowprops={'arrowstyle':'->','color':self.colors[tid],'lw':1.2,'alpha':.65})
        else:
            size=35+numeric(frame,'xG').fillna(0)*360
            ax.scatter(numeric(frame,'x')*1.05,numeric(frame,'y')*.68,s=size,c=self.colors[tid],edgecolors=TEXT_MAIN,lw=.5,alpha=.8)
            goals=frame[flags(frame,'is_goal')]
            ax.scatter(numeric(goals,'x')*1.05,numeric(goals,'y')*.68,s=100,marker='*',c=TEXT_MAIN)

    def scatter(self,i,title,frame,x,y,xlabel,ylabel,players=False):
        g=frame.dropna(subset=[x,y])
        if players:g=g[(g.minutes>=30)&~g.role.eq('GK')]
        ax=self.panel(i,title,f'{len(g)} '+('players with ≥30 min' if players else 'possessions')+
                      (' | separate measure axes' if len(g)<8 else ' | raw observations, shared axes'))
        if len(g)<8:
            ax.axis('off')
            if g.empty:
                ax.text(.5,.5,'No eligible observations',color=TEXT_DIM,transform=ax.transAxes,ha='center')
                return
            g=g.sort_values(y,ascending=False)
            for col,(key,label) in enumerate([(x,xlabel),(y,ylabel)]):
                mini=ax.inset_axes([.30+col*.38,.02,.28,.86])
                mini.set_facecolor(BG_DARK)
                for j,(_,r) in enumerate(g.iterrows()):
                    mini.hlines(j,0,r[key],color=self.colors[r.team_id],lw=2)
                    mini.scatter(r[key],j,color=self.colors[r.team_id],s=25,
                                 marker='o' if r.team_id==self.info['home_id'] else 's')
                mini.set_xlim(0,max(float(g[key].max()),.01)*1.15)
                mini.set_ylim(len(g)-.5,-.5)
                mini.set_yticks(range(len(g)),[textwrap.shorten(str(r.get('player',self.names.get(r.team_id,''))),width=24,placeholder='…') for _,r in g.iterrows()] if col==0 else [])
                mini.tick_params(colors=TEXT_DIM,labelsize=7,length=0)
                mini.set_title(label,size=8,color=TEXT_MAIN,pad=8)
                mini.spines[['top','right','left']].set_visible(False)
                mini.spines['bottom'].set_color(self.rule)
            return
        for marker,(tid,name) in zip(['o','s'],self.names.items()):
            part=g[g.team_id.eq(tid)]
            ax.scatter(part[x],part[y],c=self.colors[tid],s=35,alpha=.7,marker=marker)
        ax.set_xlabel(xlabel,size=10);ax.set_ylabel(ylabel,size=10)
        ax.margins(.2)
        ax.set_xlim(left=0);ax.set_ylim(bottom=0)
        if players:
            ax.set_xlim(left=-max(g[x].max(),1)*.06)
            ax.set_ylim(bottom=-max(g[y].max(),.01)*.10)
            from scatter_labels import label_players
            label_players(ax,g,x,y,color=TEXT_MAIN,background=BG_DARK,fontsize=8)

    def save(self,path):
        import crests
        self._rule(.045,.955,.057)
        crests.place_logo(self.fig,.061,.034,width=.031,background=BG_DARK)
        self._text(.09,.034,self.info.get('byline','MOSTAFA SAAD').upper(),size=10,weight='bold',va='center',width=.33)
        self._text(.955,.038,'SOURCE / WHOSCORED · OPTA',size=9,ha='right',color=self.muted)
        self._text(.955,.025,'Local model values are estimates',size=8,ha='right',color=self.muted)
        for left,right,tid in [(.045,.5,self.info['home_id']),(.5,.955,self.info['away_id'])]:
            self.fig.add_artist(Line2D([left,right],[.012,.012],transform=self.fig.transFigure,color=self.colors[tid],lw=3))
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        for artist,width in self.fit:
            actual=artist.get_window_extent(renderer).width
            available=width*self.fig.bbox.width
            if actual>available:artist.set_fontsize(artist.get_fontsize()*available/actual)
        self.fig.savefig(path,dpi=150,facecolor=BG_DARK)
        plt.close(self.fig)


def build_match_posters(events,xg,team_metrics,player_metrics,players,*,out_dir,home_id,away_id,home_name,away_name,home_color,away_color,score,competition='MATCH ANALYSIS',match_date='',byline='MOSTAFA SAAD',allow_download=True):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    info=dict(home_id=home_id,away_id=away_id,home_name=home_name,away_name=away_name,home_color=home_color,away_color=away_color,score=score,date=match_date,competition=competition,byline=byline,allow_download=allow_download)
    data=build_insights(events,players,info); people=data['players'];p=data['possessions'];a=data['actions'];routes=data['routes'];spells=data['spells'];losses=data['losses']
    tids=[home_id,away_id];names=[home_name,away_name]
    def metric(key,digits=0,suffix=''):
        result=[]
        for side,name in zip(['home','away'],names):
            rows=xg[xg.team.eq(name)] if key in xg else team_metrics[team_metrics.side.eq(side)]
            result.append(format_value(rows.iloc[0].get(key) if not rows.empty else None,suffix,digits).replace('Unavailable','—'))
        return result
    def kpi(label,key,digits=0,suffix=''):return (label,*metric(key,digits,suffix))
    shots=events[flags(events,'is_shot')&~flags(events,'is_penalty_shootout')&~flags(events,'is_own_goal')]
    paths=[];contracts={}
    def finish(board,n):
        path=out/POSTERS[n-1];board.save(path);paths.append(path);contracts[path.name]=board.notes

    b=Poster(info,1,'The result and the chances','Shot quality, outcomes and the score states in which chances arrived.',[kpi('Expected goals','xG',2),kpi('Shots','shots'),kpi('On target','on_target'),kpi('Big chances','big_chances')])
    ax=b.panel(0,'Expected goals through the match','Cumulative shot xG | steps = attempts | elapsed event minute')
    end_minute=float((numeric(events,'minute')+numeric(events,'second').fillna(0)/60).max())
    for tid,style in zip(tids,['-','--']):
        g=shots[shots.team_id.eq(tid)].sort_values(['minute','second'])
        total=numeric(g,'xG').fillna(0).cumsum()
        ax.step(np.r_[0,numeric(g,'minute')+numeric(g,'second')/60,end_minute],
                np.r_[0,total,total.iloc[-1] if len(total) else 0],where='post',color=b.colors[tid],lw=2.3,ls=style)
    ax.set_xlim(0,max(end_minute,1));ax.set_ylim(bottom=0)
    ax.set_xlabel('Elapsed minute');ax.set_ylabel('Cumulative xG')
    b.comparisons(1,'Chance balance',[kpi('Goals','goals'),kpi('xG per shot','xG_per_shot',3),kpi('Box entries','box_entries'),kpi('Field tilt','field_tilt',1,'%'),kpi('Completed crosses','completed_crosses')])
    for i,tid in enumerate(tids):b.pitch(2+i,names[i]+' · shots',shots[shots.team_id.eq(tid)],tid)
    stages=['Level','Leading','Trailing']; values=[]
    for tid in tids:values.append(spells[spells.team_id.eq(tid)].groupby('state').xG.sum().reindex(['level','leading','trailing'],fill_value=0).round(2).tolist())
    b.bars(4,'xG by score state',stages,*values,'Raw xG; time in each state appears on poster 3')
    outcomes=[kpi(label,key) for label,key in [('Goals','goals'),('Saved','saved'),('Blocked','blocked'),('Off target','off_target')]]
    b.bars(5,'Shot outcomes',[r[0] for r in outcomes],
           *[[_display_number(r[j]) for r in outcomes] for j in [1,2]],
           'Recorded outcome counts | both teams use the same shot-count axis')
    b.axes[5].set_xlabel('Shots',size=9)
    finish(b,1)

    b=Poster(info,2,'Progression and penalty-area access','Separate movement volume, routes into the box and the possessions that reached a shot.',[kpi('Field tilt','field_tilt',1,'%'),kpi('Progressive passes','progressive_passes'),kpi('Third entries','final_third_entries'),kpi('Box entries','box_entries')])
    for i,tid in enumerate(tids):b.pitch(i,names[i]+' · box entries',routes[routes.team_id.eq(tid)],tid,arrows=True)
    counts=[stage_counts(p,tid) for tid in tids]
    b.bars(2,'Nested possession stages',['Possessions','Third','Then box','Then shot','On target'],*counts,'Same possession at every stage; long shots without box access excluded')
    routecounts=[routes[routes.team_id.eq(tid)].groupby('lane').size().reindex(['Left','Centre','Right'],fill_value=0).tolist() for tid in tids]
    b.bars(3,'Entry origin lane',['Left','Centre','Right'],*routecounts,'Entry events; several may belong to one possession')
    b.scatter(4,'Possession duration and value',p,'duration','positive_xT','Seconds','Positive movement xT')
    b.scatter(5,'Players: involvement and value',people,'touches','positive_xT','Touches','Positive movement xT',players=True)
    finish(b,2)

    b=Poster(info,3,'Pressure, losses and transition output','Read recovery outcomes and exposure together; event data does not measure off-ball defensive spacing.',[kpi('High regains','high_regains'),kpi('Transitions','transitions'),kpi('Transition shots','transition_shots'),kpi('Transition xG','transition_xG',2)])
    from match_report import compute_ppda_both
    ppda=compute_ppda_both(info,events)
    b.comparisons(0,'Pressing and second-phase totals',[("PPDA · lower = fewer passes allowed",*(format_value(ppda[s].get('ppda'),digits=2) for s in ['home','away'])),kpi('Counterpress attempts','counterpress_attempts'),kpi('Counterpress regains','counterpress_regains'),kpi('Regain-to-shot rate','regain_to_shot_rate',1,'%'),kpi('Transition-to-shot rate','transition_shot_rate',1,'%')],'PPDA: passes / defensive action | shared scale within each row')
    values=[[len(g)]+[int(g[k].sum()) for k in ['third_within_12','box_within_12','shot_within_12']] for tid in tids for g in [losses[losses.team_id.eq(tid)]]]
    b.bars(1,'Opponent output after losses',['Losses','Third ≤12s','Box ≤12s','Shot ≤12s'],*values,'Immediately following open-play possession; outcome counts are independent')
    for i,tid in enumerate(tids):
        frame=losses[losses.team_id.eq(tid)].copy();frame['xG']=frame.xG_conceded
        b.pitch(2+i,names[i]+' · loss locations',frame,tid)
    values=[];minutes=[]
    for tid in tids:
        g=spells[spells.team_id.eq(tid)].groupby('state')[['minutes','xG']].sum().reindex(['level','leading','trailing'],fill_value=0)
        values.append((g.xG*30/g.minutes).where(g.minutes>=5).round(2));minutes.append(g.minutes)
    b.bars(4,'xG per 30 score-state minutes',['Level','Leading','Trailing'],
           *[v.tolist() for v in values],'Shared xG axis | rate withheld below 5 minutes in a state')
    b.axes[4].set_xlabel('xG / 30 minutes',size=9)
    for j,series in enumerate(values):
        for row,v in enumerate(series):
            if not np.isfinite(v):b.axes[4].text(.98,row+(-.15 if j==0 else .15),
                names[j]+': <5 min',transform=b.axes[4].get_yaxis_transform(),ha='right',color=b.colors[tids[j]],size=8)
    b.state_durations(5,minutes)
    finish(b,3)

    summaries=[]
    for tid in tids:
        squad=people[people.team_id.eq(tid)]
        summaries.append((int(squad.shots.gt(0).sum()),squad.xA.sum(),
                          100*squad.completed_passes.sum()/max(squad.passes.sum(),1),
                          100*squad.positive_xT.sum()/max(squad.touches.sum(),1)))
    kpis=[('Shot takers',*(str(s[0]) for s in summaries)),
          ('Inferred xA',*(f'{s[1]:.2f}' for s in summaries)),
          ('Pass completion',*(f'{s[2]:.1f}%' for s in summaries)),
          ('xT / 100 touches',*(f'{s[3]:.2f}' for s in summaries))]
    b=Poster(info,4,'Player contribution and finishing','xA credits the key pass with shot xG; xT / 100 touches adjusts positive movement value for involvement.',kpis)
    b.scatter(0,'Progression and creation',people,'progressions','xA','Progressions','Inferred xA',players=True)
    b.scatter(1,'Shot volume and average quality',people[people.shots>0],'shots','xG_per_shot','Shots','xG per shot',players=True)
    ceiling=max(float(people[['xG','xA']].max().max()),.01)
    for i,tid in enumerate(tids):
        b.ranking(2+i,names[i]+' · chance involvement',people[people.team_id.eq(tid)],tid,ceiling)
    keeper=people[people.role.eq('GK')]
    rows=[]
    for key,label in [('saves','Recorded saves'),('claims','Claims'),('sweeps','Sweeper actions'),('completed_passes','Completed passes'),('passes','Attempted passes')]:
        rows.append((label,*[int(keeper[keeper.team_id.eq(tid)][key].sum()) for tid in tids]))
    b.bars(4,'Goalkeeper actions',[r[0] for r in rows],*[ [r[j] for r in rows] for j in [1,2]],
           'Recorded action counts | both keepers share the same count axis')
    b.comparisons(5,'Delivery and access',[kpi('Crosses attempted','crosses'),kpi('Crosses completed','completed_crosses'),kpi('Deep completions','deep_completions'),kpi('Build-up success','build_up_success_rate',1,'%'),kpi('Box-to-shot rate','box_entry_to_shot_rate',1,'%')],'Shared scale per row | rates use their own denominators, out of 100%')
    finish(b,4)
    (out/'poster_catalog.json').write_text(json.dumps(contracts,indent=2),encoding='utf-8')
    return paths
