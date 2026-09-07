"""One role-aware player page, replacing duplicated historical radar exports."""
import re
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Wedge, Circle
from matplotlib.lines import Line2D
from match_insights import flags, numeric
from metric_registry import label, format_value


def _role_pizza(fig, rect, labels, values, colour, *, role_average=None):
    """Segmented, role-aware pizza profile; each slice is independently readable."""
    from insight_visuals import FG
    ax=fig.add_axes(rect); ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.35,1.35); ax.set_ylim(-1.35,1.35)
    n=len(labels); gap=2.5; outer=1.0; inner=.16
    for i,(caption,value) in enumerate(zip(labels,values)):
        start=90-i*360/n-360/n+gap/2; end=90-i*360/n-gap/2
        ax.add_patch(Wedge((0,0),outer,start,end,facecolor='#20272A',edgecolor='#536064',lw=.7))
        if np.isfinite(value):
            radius=inner+(outer-inner)*float(np.clip(value,0,100))/100
            ax.add_patch(Wedge((0,0),radius,start,end,facecolor=colour,alpha=.9,edgecolor='#090B0C',lw=1.2))
            ax.text(.86*np.cos(np.deg2rad((start+end)/2)),.86*np.sin(np.deg2rad((start+end)/2)),f'{value:.0f}',color='#fff',ha='center',va='center',fontsize=7,weight='bold')
        if role_average is not None and np.isfinite(role_average[i]):
            avg=inner+(outer-inner)*float(np.clip(role_average[i],0,100))/100
            ax.add_patch(Wedge((0,0),avg,start,end,facecolor='none',edgecolor='#D5DADB',lw=1.1))
        mid=np.deg2rad((start+end)/2)
        ax.text(1.18*np.cos(mid),1.18*np.sin(mid),str(caption).upper(),color=FG,ha='center',va='center',fontsize=6.5,weight='bold')
    ax.add_patch(Circle((0,0),inner,facecolor='#090B0C',edgecolor='#536064',lw=.7))
    return ax


def compact_profiles(charts, players, events):
    from insight_visuals import role_group, FG, BG, MUTED
    people=players.copy();people['role_group']=people.role.map(role_group)
    from match_metrics import line_breaking_passes
    people['line_breaking_passes']=0
    for tid in people.team_id.dropna().unique():
        opponents=[x for x in people.team_id.dropna().unique() if x != tid]
        if opponents:
            br=line_breaking_passes(events,tid,opponents[0])
            if not br.empty and 'player' in br:
                counts=br.groupby('player').size()
                people.loc[people.team_id.eq(tid),'line_breaking_passes']=people.loc[people.team_id.eq(tid),'player'].map(counts).fillna(0)
    roles={
        'Defender':['interceptions','tackles_won','clearances','recoveries'],
        'Midfielder':['xA','box_entries','completed_passes','recoveries'],
        'Forward':['xG','xA','box_entries','takeons_won'],
        'Goalkeeper':['saves','claims','sweeps','completed_passes'],
        'Unknown':['xG','xA','completed_passes','recoveries']}
    advanced=['xGChain','xGBuildup','xT_per_100_touches','progressive_pass_pct']
    for _,p in people[people.minutes>0].sort_values(['team_id','player']).iterrows():
        keys=roles[p.role_group]+advanced
        pool=people[(people.role_group==p.role_group)&(people.minutes>=30)]
        compare=p.minutes>=30 and len(pool)>=8 and p.role_group not in ['Goalkeeper','Unknown']
        fig,axes=charts.figure(p.player,f'{charts.names[p.team_id]} · {p.role_group} · Minutes {p.minutes:.1f} · Touches {p.touches:.0f} · Shots {p.shots:.0f}',columns=2,height=12)
        fig.subplots_adjust(left=.07,right=.95,top=.74,bottom=.29,wspace=.20)
        # A compact, role-aware radar restores the visual player identity while
        # keeping the raw values and the action map on the same card.
        old=axes[0];fig.delaxes(old)
        ax=fig.add_axes([.12,.51,.29,.27])
        radar_keys=(roles[p.role_group]+advanced)
        radar_keys=list(dict.fromkeys(radar_keys))
        vals=[]
        for k in radar_keys:
            value=pd.to_numeric(pd.Series([p[k]]),errors='coerce').iloc[0]
            eligible=pd.to_numeric(pool[k],errors='coerce').dropna()
            # Always draw a radar. If the role pool is too small for a
            # reliable same-role percentile, fall back to all players with a
            # recorded value so the player's raw profile never appears empty.
            if (not compare or len(eligible)<8) and len(eligible)<2:
                eligible=pd.to_numeric(people[k],errors='coerce').dropna()
            if len(eligible)>=2 and pd.notna(value):
                score=100*((eligible<value).sum()+.5*(eligible==value).sum())/len(eligible)
            else: score=0.0 if pd.notna(value) else np.nan
            vals.append(float(np.clip(score,0,100)))
        role_avg=[]
        for k in radar_keys:
            eligible=pd.to_numeric(pool[k],errors='coerce').dropna()
            role_avg.append(float(100*((eligible<eligible.mean()).sum())/len(eligible)) if len(eligible)>=2 else np.nan)
        fig.delaxes(ax)
        _role_pizza(fig,[.105,.49,.32,.32],[label(k) for k in radar_keys],vals,charts.colors[p.team_id],role_average=role_avg)
        fig.text(.265,.455,f'Role percentiles · same-role pool: {len(pool)} players (30+ min) · grey ring = role average',ha='center',color=MUTED,size=8)
        ax=axes[1];ax.set_position([.56,.52,.30,.24]);ax.set_xlim(0,105);ax.set_ylim(0,68);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        ax.add_patch(Rectangle((0,0),105,68,fill=False,ec=MUTED));ax.axvline(52.5,color=MUTED,lw=.6)
        ax.add_patch(Rectangle((88.5,13.84),16.5,40.32,fill=False,ec=MUTED))
        g=events[events.player.eq(p.player)&events.team_id.eq(p.team_id)]
        g=g[numeric(g,'x',np.nan).notna()&numeric(g,'y',np.nan).notna()]
        from match_metrics import touch_mask, progressive_pass_mask
        touches=g[touch_mask(g)]
        # Tactical snapshot keeps the map interpretable without reading every
        # row below it.
        final_touch=int((touches.x>=66.7).sum()) if len(touches) else 0
        defensive_types={'Tackle','Interception','BallRecovery','Clearance','BlockedShot','Aerial','Challenge','Foul'}
        defensive_actions=int(g.type.isin(defensive_types).sum()) if 'type' in g.columns else 0
        avg_x=float(touches.x.mean()) if len(touches) else float('nan')
        avg_y=float(touches.y.mean()) if len(touches) else float('nan')
        thirds=' / '.join(str(int((touches.x.between(lo,hi,inclusive="left")).sum())) for lo,hi in [(0,35),(35,70),(70,105)]) if len(touches) else '0 / 0 / 0'
        fig.text(.56,.485,
                 f'TACTICAL SNAPSHOT   Touch zones D/M/A {thirds}  ·  Final-third touches {final_touch}  ·  Defensive actions {defensive_actions}  ·  Line-breaking passes {int(p.line_breaking_passes)}  ·  Avg touch {avg_x:.0f}, {avg_y:.0f}',
                 color=MUTED,size=7.5,ha='left')
        if len(touches):
            # Smooth team-colour density, matching the defensive-action maps
            # instead of using blocky hexagons.
            from matplotlib.colors import LinearSegmentedColormap
            from scipy.stats import gaussian_kde
            # Purple-to-gold density is deliberately independent of the
            # team's red/blue identity and the defensive marker colours.
            from matplotlib.colors import to_rgb
            is_dark=sum(to_rgb(BG)) < 1.5
            cmap=LinearSegmentedColormap.from_list('player_heat',
                ([(0,'#07111f'),(.35,'#164e63'),(.72,'#38bdf8'),(1,'#fef08a')]
                 if is_dark else
                 [(0,'#ffffff'),(.35,'#e0e7ff'),(.72,'#a78bfa'),(1,'#6d28d9')]))
            tx,ty=touches.x.to_numpy()*1.05,touches.y.to_numpy()*.68
            if len(touches)>=6:
                gx,gy=np.mgrid[0:105:45j,0:68:30j]
                z=gaussian_kde(np.vstack([tx,ty]),bw_method=.20)(np.vstack([gx.ravel(),gy.ravel()])).reshape(gx.shape)
                peak=float(np.nanmax(z))
                if peak>0:
                    # Explicit lower cut-off removes invisible noise while
                    # keeping compact high-density zones visibly filled.
                    levels=np.linspace(peak*.08,peak,12)
                    ax.contourf(gx,gy,z,levels=levels,cmap=cmap,alpha=.88,zorder=1,antialiased=True)
        # Tactical event layer: defensive actions use a separate visual
        # language from the touch heatmap and team colour.
        is_dark=sum(__import__('matplotlib.colors',fromlist=['to_rgb']).to_rgb(BG)) < 1.5
        point_color='#f8fafc' if is_dark else '#1f2937'
        edge_color='#111827' if is_dark else '#ffffff'
        ax.scatter(touches.x*1.05,touches.y*.68,s=28,alpha=.95,color=point_color,edgecolors=edge_color,linewidths=.55,zorder=3)
        defensive_style={
            'Tackle':('#ff4d6d','D'), 'Interception':('#38bdf8','^'),
            'BallRecovery':('#22c55e','o'), 'Clearance':('#f59e0b','s'),
            'BlockedShot':('#c084fc','P'), 'Aerial':('#06b6d4','*'),
            'Challenge':('#fb923c','v'), 'Foul':('#94a3b8','x')}
        for dtype,(colour,marker) in defensive_style.items():
            dg=g[g.type.eq(dtype)] if 'type' in g.columns else g.iloc[0:0]
            dg=dg.dropna(subset=['x','y'])
            if len(dg):
                ax.scatter(dg.x*1.05,dg.y*.68,s=42,marker=marker,color=colour,
                           edgecolors='#111111',linewidths=.45,zorder=5)
        for action in g[progressive_pass_mask(g)].dropna(subset=['end_x','end_y']).itertuples():
            ax.annotate('',(action.end_x*1.05,action.end_y*.68),(action.x*1.05,action.y*.68),arrowprops={'arrowstyle':'->','color':charts.colors[p.team_id],'lw':.8},annotation_clip=True)
        shots=g[flags(g,'is_shot')];ax.scatter(shots.x*1.05,shots.y*.68,s=75,marker='*',color='#ff2d55',edgecolors='#111111',linewidths=.5,zorder=6)
        ax.set_title('Touch heatmap · points: touches · arrows: progressive passes · stars: shots →',color=FG,size=9,pad=14)
        handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor='#f5f5f5',markeredgecolor='#111',label='Touch',markersize=5)]
        handles += [Line2D([0],[0],marker=m,color='none',markerfacecolor=c if m!='x' else 'none',markeredgecolor='#111',label=d,markersize=5)
                    for d,(c,m) in defensive_style.items() if len(g[g.type.eq(d)])]
        handles.append(Line2D([0],[0],marker='*',color='none',markerfacecolor='#ff2d55',markeredgecolor='#111',label='Shot',markersize=6))
        if len(handles)>1:
            ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(1.02,.99),fontsize=5.5,
                      frameon=True,facecolor=BG,labelcolor=FG,edgecolor=MUTED,ncol=2,handletextpad=.3,columnspacing=.5)
        # Four independent columns reserve space for every label and value.
        sections=[
            ('ATTACK', [('Goals','goals'),('Shots','shots'),('xG','xG'),('xG / shot','xG_per_shot'),('Inferred xA','xA'),('xG + xA','xG_xA')]),
            ('PASSING', [('Attempts','passes'),('Completed','completed_passes'),('Completion %','pass_pct'),('Progressive passes','progressive_passes'),('Line-breaking passes','line_breaking_passes'),('Progressive share %','progressive_pass_pct'),('xA / 100 passes','xA_per_100_passes')]),
            ('PROGRESSION', [('Touches','touches'),('Progressive carries','progressive_carries'),('Box entries','box_entries'),('Positive xT','positive_xT'),('xT / 100 touches','xT_per_100_touches'),('xGChain','xGChain'),('xGBuildup','xGBuildup')]),
            ('DEFENDING / DUELS', [('Recoveries','recoveries'),('Interceptions','interceptions'),('Tackles won','tackles_won'),('Clearances','clearances'),('Take-ons attempted','takeons'),('Take-ons won','takeons_won'),('Take-on success %','takeon_success_pct')])]
        if p.role_group=='Goalkeeper':
            sections[0]=('GOALKEEPING',[('Saves','saves'),('Claims','claims'),('Sweeps','sweeps'),('Recoveries','recoveries'),('Clearances','clearances')])
        for i,(heading,metrics) in enumerate(sections):
            panel=fig.add_axes([.055+i*.235,.175,.215,.235],facecolor=BG)
            panel.set_xlim(0,1);panel.set_ylim(0,1);panel.axis('off')
            panel.text(0,1,heading,color=FG,size=9,weight='bold',va='top')
            for j,(caption,key) in enumerate(metrics):
                y=.83-j*.115
                value=p.get(key,np.nan)
                panel.text(0,y,caption,color=MUTED,size=8,va='center')
                panel.text(1,y,('N/A' if pd.isna(value) else format_value(value,digits=2 if key in {'xG','xG_per_shot','xA','xG_xA','positive_xT','xT_per_100_touches','xGChain','xGBuildup','xA_per_100_passes','pass_pct','progressive_pass_pct','takeon_success_pct'} else 0)),color=FG,size=9,weight='bold',ha='right',va='center')
                panel.plot([0,1],[y-.052,y-.052],color=MUTED,lw=.35,alpha=.25)
        definitions=('Raw match totals; unavailable values shown as N/A. Percentiles require 8 same-role players and 30+ minutes.\n'
                     'xGChain: possession chance credit; xGBuildup excludes shooter and key-pass provider. Credits overlap across teammates.\n'
                     'Positive xT: threat added by successful movements. Progressive share uses completed passes; xT / 100 uses touches.')
        fig.text(.055,.105,definitions,color=MUTED,size=8,linespacing=1.5)
        method='Role and minutes govern comparisons; these are match observations, not a season ability rating. Nominal pitch 105 × 68 m.'
        filename='player_profiles/'+re.sub(r'[^\w.-]+','_',charts.names[p.team_id])+'/'+re.sub(r'[^\w.-]+','_',p.player)+'.png'
        reading=f'{p.player}: {p.minutes:.1f} minutes; shots {p.shots:.0f}; progressive passes {p.progressive_passes:.0f}; xGChain {p.xGChain:.2f}; xGBuildup {p.xGBuildup:.2f}. '
        charts.save(fig,filename,p.player+' — advanced role profile',reading+method,method,len(pool))
