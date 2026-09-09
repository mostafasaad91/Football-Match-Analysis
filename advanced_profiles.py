"""One role-aware player page, replacing duplicated historical radar exports."""
import re
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Wedge, Circle
from matplotlib.lines import Line2D
from match_insights import flags, numeric
from metric_registry import label, format_value


# The pizza used to be painted in fixed slate and near-black: track #20272A,
# rim #536064, hub #090B0C, average ring #D5DADB, and the percentile number in
# flat white. Every one of those is a dark-theme value. On the light page the
# hub and track arrived as dark blobs on grey paper, the average ring vanished
# into it, and white digits on a gold wedge were unreadable at any size. The
# colours are derived from the active theme instead, and the digit is solved
# against the wedge it sits on rather than assumed.
_PIZZA = {
    True:  {"track": "#E7E7E7", "rim": "#B4B4B4", "avg": "#4A4A4A"},   # light
    False: {"track": "#16191C", "rim": "#3D454A", "avg": "#D5DADB"},   # dark
}

# Short forms for the ring only. The registry labels are written for table
# rows, where a line can run as long as it likes; on a wedge "Possession-
# adjusted defensive actions" overran its neighbours on both sides.
_WEDGE_LABEL = {
    "padj_defensive_actions": "pAdj def.\nactions",
    "progression_metres": "Progression\nmetres",
    "aerials_won": "Aerials\nwon",
    "defensive_height": "Defensive\nheight",
    "final_third_receptions": "Final-third\nreceptions",
    "line_breaking_passes": "Line-break\npasses",
    "progressive_pass_pct": "Progressive\npass share",
    "xT_per_100_touches": "xT / 100\ntouches",
    "box_entries": "Box\nentries",
    "takeons_won": "Take-ons\nwon",
    "completed_passes": "Completed\npasses",
    "tackles_won": "Tackles\nwon",
    "pass_pct": "Pass %",
    "xGChain": "xGChain",
    "xG": "xG",
    "xA": "xA",
    "saves": "Saves",
    "claims": "Claims",
    "sweeps": "Sweeps",
}


def wedge_label(key):
    """The ring's own name for a metric, wrapped to fit beside its neighbours."""
    import textwrap
    if key in _WEDGE_LABEL:
        return _WEDGE_LABEL[key]
    return "\n".join(textwrap.wrap(str(label(key)), 12)) or str(key)


def _role_pizza(fig, rect, labels, values, colour, *, role_average=None):
    """Segmented, role-aware pizza profile; each slice is independently readable.

    The percentile is printed **outside** the ring, beside its own label, on
    the page background. Inside the wedge it sat on whatever the bar happened
    to reach -- a saturated kit colour, a dark track, or the boundary between
    them -- and no single text colour is readable on all three. Chris Wood's
    "100" was white on Forest red at about four to one. Moving the number off
    the coloured fill removes the whole problem rather than tuning around it:
    one background, one text colour, every card, both themes.
    """
    from insight_visuals import FG, MUTED
    from visualization_components import BG_DARK, IS_LIGHT_THEME
    shades = _PIZZA[bool(IS_LIGHT_THEME)]
    ax = fig.add_axes(rect)
    ax.set_aspect('equal')
    ax.axis('off')
    # Wider than tall on purpose: the labels and their numbers sit out to the
    # left and right of the ring and nothing sits above or below it, so the
    # room they need is horizontal.
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.30, 1.30)
    n = len(labels)
    gap = 2.5
    outer = 1.0
    inner = .16
    for i, (caption, value) in enumerate(zip(labels, values)):
        start = 90 - i * 360 / n - 360 / n + gap / 2
        end = 90 - i * 360 / n - gap / 2
        ax.add_patch(Wedge((0, 0), outer, start, end,
                           facecolor=shades["track"], edgecolor=shades["rim"], lw=.7))
        mid = np.deg2rad((start + end) / 2)
        if np.isfinite(value):
            radius = inner + (outer - inner) * float(np.clip(value, 0, 100)) / 100
            ax.add_patch(Wedge((0, 0), radius, start, end, facecolor=colour,
                               alpha=.9, edgecolor=BG_DARK, lw=1.2))
        if role_average is not None and np.isfinite(role_average[i]):
            avg = inner + (outer - inner) * float(np.clip(role_average[i], 0, 100)) / 100
            ax.add_patch(Wedge((0, 0), avg, start, end, facecolor='none',
                               edgecolor=shades["avg"], lw=1.1))

        # Anchor the block on the side it sits on, so a long label grows away
        # from the ring instead of across the wedge beside it. The number is
        # the anchor and the label stacks above it, which keeps the numbers on
        # a predictable ring however many lines a label runs to.
        across, up = np.cos(mid), np.sin(mid)
        if across > .2:
            align, reach = 'left', 1.07
        elif across < -.2:
            align, reach = 'right', 1.07
        else:
            align, reach = 'center', 1.14
        x, y = reach * across, reach * up
        ax.text(x, y - .015, f'{value:.0f}' if np.isfinite(value) else 'n/a',
                color=FG if np.isfinite(value) else MUTED, ha=align, va='top',
                fontsize=10.5 if np.isfinite(value) else 7,
                weight='bold' if np.isfinite(value) else 'normal',
                style='normal' if np.isfinite(value) else 'italic')
        ax.text(x, y + .045, str(caption).upper(), color=MUTED, ha=align,
                va='bottom', fontsize=6.2, weight='bold', linespacing=1.25)
    ax.add_patch(Circle((0, 0), inner, facecolor=BG_DARK,
                        edgecolor=shades["rim"], lw=.7))
    return ax


# Three columns of rows the role actually generates, instead of one grid of
# twenty-six columns-worth for everybody. Nothing here is already a wedge on
# that role's ring: a reader who has seen a measure as a percentile does not
# need it again as a raw number two inches below.
_TABLE = {
    'Defender': [
        ('ON THE BALL', [('Touches', 'touches'), ('Passes', 'passes'),
                         ('Completed', 'completed_passes'),
                         ('Progressive passes', 'progressive_passes'),
                         ('Line-breaking passes', 'line_breaking_passes')]),
        ('DEFENDING', [('Recoveries', 'recoveries'),
                       ('Interceptions', 'interceptions'),
                       ('Tackles won', 'tackles_won'),
                       ('Clearances', 'clearances'),
                       ('Aerials', 'aerials')]),
        ('THREAT', [('Shots', 'shots'), ('xG', 'xG'), ('Inferred xA', 'xA'),
                    ('Positive xT', 'positive_xT'),
                    ('xGBuildup', 'xGBuildup')]),
    ],
    'Midfielder': [
        ('ON THE BALL', [('Touches', 'touches'), ('Passes', 'passes'),
                         ('Completed', 'completed_passes'),
                         ('Progressive passes', 'progressive_passes'),
                         ('Take-ons won', 'takeons_won')]),
        ('CREATION', [('Shots', 'shots'), ('xG', 'xG'),
                      ('Box entries', 'box_entries'),
                      ('Final-third receptions', 'final_third_receptions'),
                      ('Positive xT', 'positive_xT')]),
        ('WITHOUT THE BALL', [('Recoveries', 'recoveries'),
                              ('Interceptions', 'interceptions'),
                              ('Tackles won', 'tackles_won'),
                              ('Aerials won', 'aerials_won'),
                              ('Dispossessed', 'dispossessed')]),
    ],
    'Forward': [
        ('FINISHING', [('Goals', 'goals'), ('Shots', 'shots'),
                       ('xG / shot', 'xG_per_shot'), ('xG + xA', 'xG_xA'),
                       ('Positive xT', 'positive_xT')]),
        ('INVOLVEMENT', [('Touches', 'touches'),
                         ('Completed passes', 'completed_passes'),
                         ('Progressive passes', 'progressive_passes'),
                         ('Take-ons', 'takeons'),
                         ('Take-ons won', 'takeons_won')]),
        ('THE WORK', [('Aerials won', 'aerials_won'),
                      ('Dispossessed', 'dispossessed'),
                      ('Recoveries', 'recoveries'),
                      ('pAdj def. actions', 'padj_defensive_actions'),
                      ('Progression metres', 'progression_metres')]),
    ],
    'Goalkeeper': [
        ('THE GOAL', [('Saves', 'saves'), ('Claims', 'claims'),
                      ('Sweeps', 'sweeps'), ('Recoveries', 'recoveries'),
                      ('Clearances', 'clearances')]),
        ('DISTRIBUTION', [('Passes', 'passes'),
                          ('Completed', 'completed_passes'),
                          ('Completion %', 'pass_pct'),
                          ('Progressive passes', 'progressive_passes'),
                          ('Line-breaking passes', 'line_breaking_passes')]),
        ('WHAT IT WAS WORTH', [('Positive xT', 'positive_xT'),
                               ('xT / 100 touches', 'xT_per_100_touches'),
                               ('Touches', 'touches'),
                               ('xGBuildup', 'xGBuildup'),
                               ('Progression metres', 'progression_metres')]),
    ],
}
# A player the feed never gave a position gets the midfielder's set: it is the
# only one of the four that asks about every phase.
_TABLE['Unknown'] = _TABLE['Midfielder']


def _verdict(labels, values, minutes):
    """The three things this player did best, for the strip under his name.

    A card of forty-odd numbers at one size says nothing about which of them
    mattered. These three are the top of his own ring, so the strip is a
    summary of the card rather than a fifth opinion on it.
    """
    import numpy as np

    ranked = sorted(
        ((v, l) for l, v in zip(labels, values) if np.isfinite(v)),
        reverse=True)
    return ranked[:3]


def compact_profiles(charts, players, events):
    from insight_visuals import role_group, FG, BG, MUTED
    people=players.copy();people['role_group']=people.role.map(role_group)
    # Possession-adjusted defending, progression distance, aerials, defensive
    # height and final-third receptions. Computed for the article's ranking
    # already; the ring was still drawing raw counts beside them.
    from player_advanced import enrich as _enrich_advanced
    people=_enrich_advanced(people,events,players)
    from match_metrics import line_breaking_passes
    people['line_breaking_passes']=0
    for tid in people.team_id.dropna().unique():
        opponents=[x for x in people.team_id.dropna().unique() if x != tid]
        if opponents:
            br=line_breaking_passes(events,tid,opponents[0])
            if not br.empty and 'player' in br:
                counts=br.groupby('player').size()
                people.loc[people.team_id.eq(tid),'line_breaking_passes']=people.loc[people.team_id.eq(tid),'player'].map(counts).fillna(0)
    # Eight wedges: four that belong to the role, four every outfielder is
    # asked the same question on.
    #
    # The defensive four used to be raw interceptions, tackles, clearances and
    # recoveries. A defensive action can only happen while the other side has
    # the ball, so those four rewarded a player for his team being pinned:
    # Brighton recorded 103 of them to Chelsea's 125 and were still defending
    # twice as hard per opponent touch. They are one possession-adjusted wedge
    # now, with the questions the counts could not answer -- how high he won it,
    # how much ground he actually gained, whether he won the ball in the air --
    # taking the other three.
    roles={
        'Defender':['padj_defensive_actions','aerials_won','defensive_height','progression_metres'],
        'Midfielder':['padj_defensive_actions','progression_metres','line_breaking_passes','xA'],
        'Forward':['xG','xA','box_entries','final_third_receptions'],
        'Goalkeeper':['saves','claims','sweeps','completed_passes'],
        'Unknown':['xG','xA','completed_passes','padj_defensive_actions']}
    # xGBuildup is xGChain minus the shooter and the key-pass provider, and on
    # a player who was neither it is the same number twice. Vuskovic's ring
    # read 93 and 93. It stays in the table, where the pair can be compared;
    # two wedges moving together said nothing the first one had not.
    advanced=['xGChain','xT_per_100_touches','pass_pct','progressive_pass_pct']
    # Below this many players in a role the percentile is a ranking against
    # two or three people, which is a position in a queue rather than a rate.
    # Same floor the article's own ranking uses.
    ROLE_POOL_MINIMUM=4
    for _,p in people[people.minutes>0].sort_values(['team_id','player']).iterrows():
        pool=people[(people.role_group==p.role_group)&(people.minutes>=30)]
        compare=(p.minutes>=30 and len(pool)>=ROLE_POOL_MINIMUM
                 and p.role_group not in ('Goalkeeper','Unknown'))
        # The card used to print "percentiles require 8 same-role players",
        # then draw them anyway off a pool of seven through a fallback that
        # only checked for fewer than two. One basis is chosen here and both
        # the ring and the caption below it are told which one it was.
        basis=pool if compare else people[people.minutes>=30]
        fig,axes=charts.figure(p.player,f'{charts.names[p.team_id]} · {p.role_group} · Touches {p.touches:.0f} · Shots {p.shots:.0f}',columns=2,height=12)
        fig.subplots_adjust(left=.07,right=.95,top=.74,bottom=.29,wspace=.20)
        # A compact, role-aware radar restores the visual player identity while
        # keeping the raw values and the action map on the same card.
        old=axes[0];fig.delaxes(old)
        ax=fig.add_axes([.12,.51,.29,.27])
        radar_keys=(roles[p.role_group]+advanced)
        radar_keys=list(dict.fromkeys(radar_keys))
        vals=[]
        role_avg=[]
        for k in radar_keys:
            value=pd.to_numeric(pd.Series([p.get(k,np.nan)]),errors='coerce').iloc[0]
            eligible=pd.to_numeric(basis.get(k,pd.Series(dtype=float)),errors='coerce').dropna()
            if len(eligible)<2:
                # Last resort so a profile is never an empty ring: everyone who
                # played, at whatever minutes.
                eligible=pd.to_numeric(people.get(k,pd.Series(dtype=float)),errors='coerce').dropna()
            if len(eligible)>=2 and pd.notna(value):
                vals.append(float(np.clip(
                    100*((eligible<value).sum()+.5*(eligible==value).sum())/len(eligible),0,100)))
            else:
                # No comparison and no value are different states, and only one
                # of them is a zero.
                vals.append(0.0 if pd.notna(value) and len(eligible)>=2 else np.nan)
            role_avg.append(
                float(100*((eligible<eligible.mean()).sum())/len(eligible))
                if len(eligible)>=2 else np.nan)
        fig.delaxes(ax)
        _role_pizza(fig,[.065,.435,.35,.30],[wedge_label(k) for k in radar_keys],vals,charts.colors[p.team_id],role_average=role_avg)
        # Three numbers at 26pt under the name, so the card leads with a
        # judgement instead of opening on forty numbers at one size and leaving
        # the reader to find the ones that mattered.
        strip=_verdict([wedge_label(k).replace(chr(10),' ') for k in radar_keys],vals,p.minutes)

        def _tile(x,number,name,under,colour):
            # A 26pt number is about .014 of the page per digit, so a fixed
            # label offset put "MINUTES" through the middle of "98".
            fig.text(x,.812,number,color=colour,size=26,weight='bold',
                     ha='left',va='baseline')
            gap=x+.020+.0155*len(number)
            fig.text(gap,.818,name,color=FG,size=7.5,weight='bold',ha='left',va='baseline')
            fig.text(gap,.803,under,color=MUTED,size=6.8,ha='left',va='baseline')

        for column,(score,name) in enumerate(strip):
            _tile(.068+column*.155,f'{score:.0f}',name.upper(),'percentile',
                  charts.colors[p.team_id])
        if strip:
            _tile(.545,f'{p.minutes:.0f}','MINUTES',
                  'played' if p.minutes>=89 else 'of the match',MUTED)
            fig.text(.068,.782,'His three strongest measures on the ring below.',
                     color=MUTED,size=7.5,ha='left',va='baseline')
        caption=(f'Role percentiles · same-role pool: {len(pool)} players (30+ min)'
                 if compare else
                 f'Match percentiles · only {len(pool)} in this role · compared with all {len(basis)} players over 30 min')
        fig.text(.24,.408,caption+' · grey ring = pool average',ha='center',color=MUTED,size=8)
        ax=axes[1];ax.set_position([.545,.455,.33,.29]);ax.set_xlim(0,105);ax.set_ylim(0,68);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
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
        # One line starting at x=.56 could not hold five readings: it ran past
        # the right edge of the page and lost everything after "Line-breaking".
        # Two lines fit inside the map's own column.
        average=(f'{avg_x:.0f}, {avg_y:.0f}' if np.isfinite(avg_x) and np.isfinite(avg_y)
                 else 'n/a')
        fig.text(.545,.427,
                 f'TACTICAL SNAPSHOT   Touch zones D/M/A {thirds}  ·  Final-third touches {final_touch}\n'
                 f'Defensive actions {defensive_actions}  ·  Line-breaking passes {int(p.line_breaking_passes)}  ·  Avg touch {average}',
                 color=MUTED,size=7.5,ha='left',va='top',linespacing=1.6)
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
        marker_edge='#0b0f14' if is_dark else '#ffffff'
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
                unfilled=marker in {'x','+','|','_'}
                ax.scatter(dg.x*1.05,dg.y*.68,s=42,marker=marker,color=colour,
                           linewidths=.9 if unfilled else .45,zorder=5,
                           **({} if unfilled else {'edgecolors':marker_edge}))
        for action in g[progressive_pass_mask(g)].dropna(subset=['end_x','end_y']).itertuples():
            ax.annotate('',(action.end_x*1.05,action.end_y*.68),(action.x*1.05,action.y*.68),arrowprops={'arrowstyle':'->','color':charts.colors[p.team_id],'lw':.8},annotation_clip=True)
        shots=g[flags(g,'is_shot')];ax.scatter(shots.x*1.05,shots.y*.68,s=75,marker='*',color='#ff2d55',edgecolors=marker_edge,linewidths=.5,zorder=6)
        ax.set_title('Touch heatmap · points: touches · arrows: progressive passes · stars: shots →',color=FG,size=9,pad=14)
        handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=point_color,
                        markeredgecolor=edge_color,label='Touch',markersize=5)]
        handles += [Line2D([0],[0],marker=m,color='none',
                           markerfacecolor='none' if m=='x' else c,
                           markeredgecolor=c if m=='x' else marker_edge,
                           label=d,markersize=5)
                    for d,(c,m) in defensive_style.items() if len(g[g.type.eq(d)])]
        handles.append(Line2D([0],[0],marker='*',color='none',markerfacecolor='#ff2d55',
                              markeredgecolor=marker_edge,label='Shot',markersize=6))
        if len(handles)>1:
            ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(1.02,.99),fontsize=5.5,
                      frameon=True,facecolor=BG,labelcolor=FG,edgecolor=MUTED,ncol=2,handletextpad=.3,columnspacing=.5)
        # Four fixed columns held the same twenty-six rows whatever the player
        # did for a living. Measured across a match, an average card spent 9.9
        # of those rows saying nought -- a centre-back was given three lines of
        # take-on statistics and a goalkeeper thirteen empty rows out of
        # twenty-six. Three columns of role-appropriate rows say more in less
        # space, and anything already drawn on the ring is left off so a reader
        # does not meet the same measure twice in two forms.
        sections=_TABLE[p.role_group if p.role_group in _TABLE else 'Unknown']
        legacy=[
            ('ATTACK', [('Goals','goals'),('Shots','shots'),('xG','xG'),('xG / shot','xG_per_shot'),('Inferred xA','xA'),('xG + xA','xG_xA')]),
            ('PASSING', [('Attempts','passes'),('Completed','completed_passes'),('Completion %','pass_pct'),('Progressive passes','progressive_passes'),('Line-breaking passes','line_breaking_passes'),('Progressive share %','progressive_pass_pct'),('xA / 100 passes','xA_per_100_passes')]),
            ('PROGRESSION', [('Touches','touches'),('Progressive carries','progressive_carries'),('Box entries','box_entries'),('Positive xT','positive_xT'),('xT / 100 touches','xT_per_100_touches'),('xGChain','xGChain'),('xGBuildup','xGBuildup')]),
            ('DEFENDING / DUELS', [('Recoveries','recoveries'),('Interceptions','interceptions'),('Tackles won','tackles_won'),('Clearances','clearances'),('Take-ons attempted','takeons'),('Take-ons won','takeons_won'),('Take-on success %','takeon_success_pct')])]
        for i,(heading,metrics) in enumerate(sections):
            panel=fig.add_axes([.065+i*.30,.185,.255,.20],facecolor=BG)
            panel.set_xlim(0,1);panel.set_ylim(0,1);panel.axis('off')
            panel.text(0,1,heading,color=FG,size=9,weight='bold',va='top')
            for j,(caption,key) in enumerate(metrics):
                y=.82-j*.155
                value=p.get(key,np.nan)
                panel.text(0,y,caption,color=MUTED,size=8,va='center')
                panel.text(1,y,('N/A' if pd.isna(value) else format_value(value,digits=2 if key in {'xG','xG_per_shot','xA','xG_xA','positive_xT','xT_per_100_touches','xGChain','xGBuildup','xA_per_100_passes','pass_pct','progressive_pass_pct','takeon_success_pct'} else 0)),color=FG,size=9,weight='bold',ha='right',va='center')
                panel.plot([0,1],[y-.052,y-.052],color=MUTED,lw=.35,alpha=.25)
        definitions=('Raw match totals; unavailable values shown as N/A. The ring compares a player with others in his role when four or more played 30+ minutes, and with every 30+ minute player otherwise; the caption above says which.\n'
                     'pAdj defensive actions: tackles, interceptions, recoveries, clearances, challenges and blocked passes per 100 opponent touches. Defensive height: mean distance upfield of those actions, blank below three of them.\n'
                     'Progression metres: ground gained toward goal by completed passes and carries. Line-breaking passes: completed forward passes of 20 m or more. Positive xT: threat added by successful movements.\n'
                     'xGChain: possession chance credit; xGBuildup excludes shooter and key-pass provider, and the credit overlaps across teammates. Progressive share uses completed passes; xT / 100 uses touches.')
        # The method footer is drawn at y=.055 and grows upward, so this block
        # has to finish above it. At .132 and five lines it did not, and the
        # two ran through each other.
        fig.text(.055,.152,definitions,color=MUTED,size=6.9,linespacing=1.45,va='top')
        method='Role and minutes govern comparisons; these are match observations, not a season ability rating. Nominal pitch 105 × 68 m.'
        filename='player_profiles/'+re.sub(r'[^\w.-]+','_',charts.names[p.team_id])+'/'+re.sub(r'[^\w.-]+','_',p.player)+'.png'
        reading=f'{p.player}: {p.minutes:.1f} minutes; shots {p.shots:.0f}; progressive passes {p.progressive_passes:.0f}; xGChain {p.xGChain:.2f}; xGBuildup {p.xGBuildup:.2f}. '
        charts.save(fig,filename,p.player+' — advanced role profile',reading+method,method,len(pool))
