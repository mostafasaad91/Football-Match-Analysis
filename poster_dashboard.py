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
    #
    # This was briefly changed to scale a rate against the larger of the pair,
    # because 8.5% and 6.9% draw two short stubs in a long rail. That reads
    # better and says something false: on a shared 0–100 track the row reports
    # that neither side turned even a tenth of its regains into a shot, which
    # is the finding. Max-scaling redraws the same 1.6-point gap as a full bar
    # against a four-fifths one. The rail stays out of 100; the printed value
    # beside it carries the precision.
    percent=all(str(v).endswith('%') for v in (first,second))
    ceiling=100.0 if percent and all(0<=v<=100 for v in valid) else max(valid+[0.01])
    return values,ceiling


class Poster:
    """One 4:5 board.

    ``layout`` picks the panel grid. "grid" is the reference board: six panels,
    three rows of two, which is the density a reader wants on a desktop screen
    or in print.

    "thread" is the same 4:5 frame with three panels instead — one full-width
    above two halves. 4:5 is already the tallest ratio X shows without cropping
    in the timeline, so the frame never needed changing; what made the boards
    unreadable on a phone was six panels inside it. At a timeline width of
    around four hundred points each of those panels renders about a hundred and
    thirty points tall, and no axis label survives that. Three panels get twice
    the area each and the type scales with them.
    """

    def __init__(self, info, number, title, subtitle, kpis, *, layout='grid', total=4):
        self.layout=layout
        self.info=info; self.names={info['home_id']:info['home_name'],info['away_id']:info['away_name']}
        from visual_redesign_full import lift_to_floor
        self.colors={info['home_id']:lift_to_floor(info['home_color']),info['away_id']:lift_to_floor(info['away_color'])}
        self.fig=plt.figure(figsize=(16,20),facecolor=BG_DARK)
        self.number=number
        self.rule='#C6CBCB' if IS_LIGHT_THEME else '#303638'
        self.muted='#596266' if IS_LIGHT_THEME else '#ADB5B8'
        self.fit=[]
        self.notes=[]
        self._panel=0
        self._text(.045,.969,'TACTICAL / MATCH ANALYSIS',size=12,weight='bold',va='center')
        self._text(.955,.969,f'{number:02d} / {total:02d}',size=12,ha='right',va='center',color=self.muted)
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
        if layout=='thread':
            # One full-width panel over two halves. The widths here are the
            # panel's own column, which self.width reports so every draw method
            # can size against the layout it is actually in.
            # The KPI strip above runs to 0.741, so a panel title at 0.760 sat
            # on top of it. Row one starts below that; row two is placed so an
            # equal-aspect pitch fills its column instead of leaving a band of
            # empty page under it.
            geometry=[((.045,.716),[.085,.495,.870,.168]),
                      ((.045,.430),[.085,.148,.390,.227]),
                      ((.525,.430),[.565,.148,.390,.227])]
            for slot,rect in geometry:
                self.slots.append(slot)
                self.axes.append(self.fig.add_axes(rect))
        else:
            for row in range(3):
                top=.711-row*.219
                for col in range(2):
                    left=.045+col*.482
                    self.slots.append((left,top))
                    self.axes.append(self.fig.add_axes([left+.038,top-.177,.390,.126]))
        for ax in self.axes:
            ax.set_facecolor(BG_DARK);ax.tick_params(colors=self.muted,labelsize=11,length=0,pad=6)
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

    @property
    def wide(self):
        """True while drawing a panel that spans the whole board."""
        return self.layout=='thread' and self._panel==0

    def width(self,i=None):
        """The figure-fraction width of panel ``i``'s column."""
        index=self._panel if i is None else i
        if self.layout!='thread':
            return .428
        return .910 if index==0 else .430

    def panel(self,i,title,note):
        self._panel=i
        ax=self.axes[i]
        left,top=self.slots[i]
        span=self.width(i)
        # A thread board carries three panels rather than six, so its type is
        # sized for a phone: the reader is looking at roughly a quarter of the
        # linear scale a printed board gets.
        big=self.layout=='thread'
        # A board is read at timeline width on a phone far more often than at
        # print size, and the old scale assumed the opposite: a 15pt panel
        # title over an 8.5pt note, on a 2400px board shown at 400, renders
        # around 2.5pt. Everything here is one step larger than it was.
        self._rule(left,left+span,top)
        self._text(left,top-.020,f'{i+1:02d}',size=12 if big else 10.5,color=self.muted,va='center')
        self._text(left+(.030 if big else .026),top-.020,title,
                   size=21 if big else 17,weight='bold',va='center',width=span-.032)
        self._text(left,top-.036 if big else top-.033,
                   textwrap.fill(note,90 if big else 70),
                   size=11 if big else 10,color=self.muted,va='top')
        self.notes.append({'panel':i+1,'title':title,'definition':note})
        return ax

    def comparisons(self,i,title,rows,note='Two bars per metric; shared scale within each row; — = unavailable'):
        """Paired bars. A row is (label, home, away) or (label, home, away, lower_better).

        ``lower_better`` exists for PPDA. Bar length reads as "more of the
        thing", and PPDA is the one row where more is worse: Brighton pressed
        at 4.02 against Chelsea's 16.41 and drew a quarter of Chelsea's bar,
        so the harder press looked like the softer one and the caption
        "lower = fewer passes allowed" was left to argue with the picture.
        The bar now shows the *press*, not the number: the lowest PPDA fills
        the rail and the others are drawn as a fraction of it. The printed
        value beside it is still the real PPDA.
        """
        ax=self.panel(i,title,note);ax.axis('off')
        left,top=self.slots[i]
        span=self.width(i)
        big=self.layout=='thread'
        ax.set_position([left,top-(.215 if big else .177),span,(.160 if big else .126)])
        ax.set_xlim(0,1);ax.set_ylim(0,len(rows))
        label_size=13 if big else 11
        value_size=15 if big else 12.5
        for j,row in enumerate(rows):
            label,h,a=row[0],row[1],row[2]
            lower_better=bool(row[3]) if len(row)>3 else False
            y=len(rows)-j-.5
            values,ceiling=_pair_scale(h,a)
            if lower_better:
                positive=[v for v in values if np.isfinite(v) and v>0]
                best=min(positive) if positive else np.nan
                shares=[best/v if np.isfinite(v) and v>0 and np.isfinite(best) else np.nan
                        for v in values]
            else:
                shares=[v/ceiling if np.isfinite(v) else np.nan for v in values]
            ax.text(0,y,textwrap.fill(label,30 if big else 24),color=TEXT_MAIN,size=label_size,va='center')
            for share,display,tid,dy,hatch in zip(shares,[h,a],self.names, [.16,-.16],['','///']):
                ax.barh(y+dy,.48,left=.40,height=.16,color=self.rule,alpha=.30)
                if np.isfinite(share):
                    ax.barh(y+dy,.48*min(max(share,0),1),left=.40,height=.16,
                            color=self.colors[tid],edgecolor=BG_DARK,lw=.3,hatch=hatch)
                ax.text(.91,y+dy,str(display),color=self.colors[tid],size=value_size,weight='bold',va='center')
        self.notes[-1]['scale']=('Each metric uses a separate zero-based scale shared by both teams; '
            'percentages use 0–100. On a row marked lower = better the longest bar is the '
            'best value, not the largest.')

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

    def _heat(self,ax,heat,tid):
        """Touch density under the marks, in the team's own colour.

        A pitch of arrows says how a side got in but not where it lived, so
        eleven entries from a team pinned in its own half looked the same as
        eleven from a team camped on the edge of the box. The density is the
        denominator those marks were missing.

        Drawn as a filled contour rather than an image: an image resamples to
        the panel and leaves a soft rectangle at the touchline, and the pitch
        markings have to stay readable through it.
        """
        from scipy.ndimage import gaussian_filter
        x=numeric(heat,'x').dropna();y=numeric(heat,'y').dropna()
        index=x.index.intersection(y.index)
        if len(index)<12:return
        grid,_,_=np.histogram2d(x.loc[index]*1.05,y.loc[index]*.68,
                                bins=[np.linspace(0,105,36),np.linspace(0,68,24)])
        grid=gaussian_filter(grid,1.6)
        if not np.isfinite(grid).any() or grid.max()<=0:return
        grid=grid/grid.max()
        from matplotlib.colors import LinearSegmentedColormap, to_rgb
        base=to_rgb(self.colors[tid])
        # One alpha does not suit two kits. A navy at 0.6 sits quietly under
        # the pitch markings; Brighton's yellow at the same value buried the
        # lines and the arrows drawn over it. Ceiling scales with how far the
        # colour already is from the page, so a bright kit shades lighter.
        distance=sum(abs(c-(0.96 if IS_LIGHT_THEME else 0.0)) for c in base)/3
        ceiling=float(np.clip(0.62-0.34*distance,0.22,0.55))
        # Transparent at the floor so an empty zone shows the page, not a wash
        # of team colour that would read as presence.
        cmap=LinearSegmentedColormap.from_list('heat',[(*base,0.0),(*base,ceiling)])
        ax.contourf(np.linspace(0,105,35)+105/70,np.linspace(0,68,23)+68/46,
                    grid.T[:23,:35],levels=np.linspace(0.10,1.0,8),cmap=cmap,
                    antialiased=True,zorder=0)

    def pitch(self,i,title,frame,tid,*,arrows=False,heat=None):
        purpose=('Arrows = successful box entries' if arrows else
                 'Point area = shot xG; stars = goals' if 'is_shot' in frame else
                 'Point area = opponent xG in the next 12 seconds')
        if heat is not None:purpose+='; shading = touch density'
        # The caption wraps at 78 characters and the shading clause pushed the
        # pitch dimensions onto a second line that sat on top of the goal line.
        ax=self.panel(i,title,purpose+(' | attacks →' if heat is not None else
                                       ' | attacks → | pitch: 105 × 68 m'))
        left,top=self.slots[i]
        span=self.width(i)
        big=self.layout=='thread'
        # Equal aspect means the drawn height follows the width, so a box that
        # is taller than the pitch only buys air. 0.227 is what a column of
        # span 0.430 on a 16x20 figure needs to hold a 109x72 pitch exactly.
        ax.set_position([left,top-(.282 if big else .184),span,(.227 if big else .137)])
        ax.set_xlim(-2,107);ax.set_ylim(-2,70);ax.set_aspect('equal');ax.axis('off')
        if heat is not None:self._heat(ax,heat,tid)
        ax.add_patch(Rectangle((0,0),105,68,fill=False,ec=TEXT_DIM,lw=.8,zorder=2))
        ax.plot([52.5,52.5],[0,68],color=TEXT_DIM,lw=.7,zorder=2)
        ax.add_patch(Circle((52.5,34),9.15,fill=False,ec=TEXT_DIM,lw=.7,zorder=2))
        ax.add_patch(Rectangle((88.5,13.84),16.5,40.32,fill=False,ec=TEXT_DIM,lw=.8,zorder=2))
        ax.add_patch(Rectangle((0,13.84),16.5,40.32,fill=False,ec=TEXT_DIM,lw=.8,zorder=2))
        for xx in (0,99.5):ax.add_patch(Rectangle((xx,24.84),5.5,18.32,fill=False,ec=TEXT_DIM,lw=.7,zorder=2))
        ax.scatter([11,94],[34,34],s=5,color=TEXT_DIM,zorder=2)
        if arrows:
            for row in frame.dropna(subset=['x','y','end_x','end_y']).itertuples():
                ax.annotate('',(row.end_x*1.05,row.end_y*.68),(row.x*1.05,row.y*.68),
                            arrowprops={'arrowstyle':'->','color':self.colors[tid],'lw':1.2,'alpha':.85},
                            zorder=3)
        else:
            size=35+numeric(frame,'xG').fillna(0)*360
            ax.scatter(numeric(frame,'x')*1.05,numeric(frame,'y')*.68,s=size,c=self.colors[tid],edgecolors=TEXT_MAIN,lw=.5,alpha=.8,zorder=3)
            goals=frame[flags(frame,'is_goal')]
            ax.scatter(numeric(goals,'x')*1.05,numeric(goals,'y')*.68,s=100,marker='*',c=TEXT_MAIN,zorder=4)

    def scatter(self,i,title,frame,x,y,xlabel,ylabel,players=False,qualifier=''):
        """``qualifier`` names any filter the caller applied before calling.

        The shot panel is handed ``people[people.shots>0]`` and printed the
        same "players with ≥30 min" as the panel beside it, so one poster
        carried "20 players with ≥30 min" and "15 players with ≥30 min" for
        the same match under the same stated rule.
        """
        g=frame.dropna(subset=[x,y])
        if players:g=g[(g.minutes>=30)&~g.role.eq('GK')]
        ax=self.panel(i,title,f'{len(g)} '+('players with ≥30 min'+(' and '+qualifier if qualifier else '')
                      if players else 'possessions')+
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

    def press_radar(self, i, rows):
        """The whole press as one shape: six questions, both sides.

        A poster carried the pressing rate as one row of paired bars, which
        answers how hard but not what kind. Six axes -- how early the ball is
        contested, how often it is won high, how often that becomes a shot, how
        often it is won straight back, how far up the side plays, and how much
        of the risk is punished -- carry the shape of a press rather than its
        rate.

        ``rows`` is (label, home, away, floor, ceiling, lower_is_better,
        digits). The two that are better when low are inverted here, so on
        every axis further from the centre is better. That is the one rule that
        makes a radar readable at a glance, and it is why the pressing rate
        alone could never be one.
        """
        import numpy as np

        ax = self.panel(i, 'Pressing profile',
                        'Six measures of the same press | further from the centre is better on every axis')
        left, top = self.slots[i]
        span = self.width(i)
        big = self.layout == 'thread'
        ax.remove()
        ax = self.fig.add_axes([left + span * .22, top - (.215 if big else .180),
                                span * .56, (.160 if big else .126)], projection='polar')
        ax.set_facecolor(BG_DARK)
        angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(rows), endpoint=False)
        for ring in (.25, .5, .75, 1.0):
            ax.plot(np.linspace(0, 2 * np.pi, 180), [ring] * 180,
                    color=self.rule, lw=.6, zorder=1)
        for angle in angles:
            ax.plot([angle, angle], [0, 1], color=self.rule, lw=.6, zorder=1)
        sides = [self.info['home_id'], self.info['away_id']]
        for index, tid in enumerate(sides):
            values = []
            for row in rows:
                low, high = row[3], row[4]
                share = (row[1 + index] - low) / (high - low) if high != low else 0.0
                share = min(max(share, 0.0), 1.0)
                values.append(1.0 - share if row[5] else share)
            loop = list(angles) + [angles[0]]
            colour = self.colors[tid]
            ax.plot(loop, values + values[:1], color=colour, lw=2.2, zorder=4)
            ax.fill(loop, values + values[:1], color=colour, alpha=.16, zorder=3)
            ax.scatter(angles, values, color=colour, s=22, zorder=5,
                       edgecolors=BG_DARK, linewidths=1.0)
        for angle, row in zip(angles, rows):
            ax.text(angle, 1.20, row[0].upper(), ha='center', va='center',
                    color=TEXT_MAIN, size=9 if big else 7.5, weight='bold',
                    linespacing=1.2)
            # One figure a line, each in its own side's colour: side by side
            # they overlap at the top and bottom of the ring, where a small
            # rotation is almost no horizontal distance.
            for radius, value, tid in ((1.40, row[1], sides[0]),
                                       (1.56, row[2], sides[1])):
                ax.text(angle, radius, f'{value:.{row[6]}f}', ha='center',
                        va='center', color=self.colors[tid],
                        size=9 if big else 7.5, weight='bold')
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        return ax


    def sonar(self, i, events, tid, name, reach=None):
        """Direction, distance and completion of one side's passing.

        The same mark as the report's Passing Sonar page. ``pass_angle`` and
        ``pass_length`` are on every pass in the feed and no poster read them,
        so a board could show who passed to whom and how far forward the ball
        went without ever showing what shape the build-up had.
        """
        ax = self.panel(i, f'{name} · passing shape',
                        'Wedge length = median distance | opacity = completion | attacking direction is right')
        left, top = self.slots[i]
        span = self.width(i)
        big = self.layout == 'thread'
        ax.remove()
        ax = self.fig.add_axes([left + span * .18, top - (.205 if big else .172),
                                span * .64, (.150 if big else .118)], projection='polar')
        self.axes[i] = ax
        ax.set_facecolor(BG_DARK)

        passes = events[flags(events, 'is_pass') & events.team_id.eq(tid)].copy()
        passes['pass_angle'] = numeric(passes, 'pass_angle')
        passes['pass_length'] = numeric(passes, 'pass_length')
        passes = passes.dropna(subset=['pass_angle', 'pass_length'])
        if passes.empty:
            ax.text(.5, .5, 'No recorded pass angles', transform=ax.transAxes,
                    ha='center', color=TEXT_DIM)
            return
        ok = successful(passes)
        bins = 16
        edges = np.linspace(0, 2 * np.pi, bins + 1)
        width = edges[1] - edges[0]
        centres = edges[:-1] + width / 2
        # Both sonars on a board have to share a radius. Scaled separately,
        # Chelsea's ran to 61 m and Brighton's to 35, so two sides whose median
        # pass was the same 15 m drew one shrunken rose beside one that filled
        # its circle, and the panel compared the axes rather than the passing.
        if reach is None:
            reach = max(float(passes.pass_length.quantile(.92)), 12.0)
        binned = pd.cut(passes.pass_angle % (2 * np.pi), bins=edges, labels=False, include_lowest=True)
        from matplotlib.colors import LinearSegmentedColormap, to_rgb
        base = to_rgb(self.colors[tid])
        for index in range(bins):
            rows = passes[binned.eq(index)]
            if rows.empty:
                continue
            median_length = float(rows.pass_length.median())
            rate = float(ok[rows.index].mean())
            ax.bar(centres[index], min(median_length, reach), width=width * .88,
                   color=self.colors[tid], alpha=.25 + .65 * rate,
                   edgecolor=BG_DARK, linewidth=.7, zorder=3)
        ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
        ax.set_ylim(0, reach); ax.set_yticks([reach / 2, reach])
        ax.set_yticklabels([f'{reach / 2:.0f}m', f'{reach:.0f}m'], color=self.muted, size=8)
        ax.set_rlabel_position(112)
        ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        ax.set_xticklabels(['FWD', 'LEFT', 'BACK', 'RIGHT'], color=self.muted,
                           size=9 if big else 8, weight='bold')
        ax.grid(color=self.rule, lw=.5, alpha=.8)
        ax.spines['polar'].set_color(self.rule)
        forward = passes[(passes.pass_angle < np.pi / 2) | (passes.pass_angle > 3 * np.pi / 2)]
        self._text(left, top - (.225 if big else .190),
                   f'{len(passes)} passes · {100 * len(forward) / max(len(passes), 1):.0f}% forward · '
                   f'{100 * ok.mean():.0f}% completed',
                   size=11 if big else 8.5, color=self.muted, width=span)

    def finishing(self, i, events, shots):
        """Chance quality, striking quality and goals, as one slope per side.

        Four stages rather than three so every step compares like with like:
        the xG that reached the frame is priced against the post-shot xG of the
        same attempts, not against the xG of every shot including those that
        never arrived.
        """
        ax = self.panel(i, 'From chance to goal',
                        'Pre-shot xG, the share that reached the frame, its post-shot value, and goals')
        from match_metrics import post_shot_xg
        priced = shots.copy()
        priced['_psxg'] = pd.to_numeric(post_shot_xg(events), errors='coerce').reindex(priced.index).fillna(0.0)
        framed = priced.shot_whoscored_type.astype(str).isin(['Goal', 'SavedShot'])
        big = self.layout == 'thread'
        ceiling = 0.0
        for offset, tid in enumerate(self.names):
            own = priced[priced.team_id.eq(tid)]
            hit = own.loc[framed.reindex(own.index, fill_value=False)]
            values = [float(numeric(own, 'xG').sum()), float(numeric(hit, 'xG').sum()),
                      float(hit._psxg.sum()), float(flags(own, 'is_goal').sum())]
            ceiling = max(ceiling, *values)
            ax.plot(range(4), values, color=self.colors[tid], lw=3.0, marker='o',
                    markersize=10, markeredgecolor=BG_DARK, markeredgewidth=1.2, zorder=4)
            for index, value in enumerate(values):
                ax.annotate(f'{value:.2f}' if index < 3 else f'{value:.0f}',
                            (index, value), xytext=(0, 11 if offset == 0 else -18),
                            textcoords='offset points', color=self.colors[tid],
                            size=11 if big else 9, weight='bold', ha='center')
        ax.set_xticks(range(4), ['Created', 'On frame', 'Struck', 'Scored'],
                      size=12 if big else 9)
        ax.set_xlim(-.4, 3.4); ax.set_ylim(0, ceiling * 1.30)
        ax.set_ylabel('Expected goals / goals', size=12 if big else 9)
        ax.tick_params(labelsize=11 if big else 9)

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


def build_match_posters(events, xg, team_metrics, player_metrics, players, *, out_dir,
                         home_id, away_id, home_name, away_name, home_color, away_color,
                         score, competition='MATCH ANALYSIS', match_date='',
                         byline='MOSTAFA SAAD', allow_download=True):
    """Four boards sized to be read on a phone in a timeline.

    Same 4:5 frame as the reference posters, because 4:5 is already the tallest
    ratio X will show uncropped and no ratio change was needed. What changes is
    density: three panels a board instead of six, so each one gets roughly twice
    the area and can carry type a reader can actually see at timeline width.

    Each board answers one question, and the full-width panel at the top is the
    answer. The two below it are the evidence.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = dict(home_id=home_id, away_id=away_id, home_name=home_name, away_name=away_name,
                home_color=home_color, away_color=away_color, score=score, date=match_date,
                competition=competition, byline=byline, allow_download=allow_download)
    data = build_insights(events, players, info)
    people = data['players']; p = data['possessions']; a = data['actions']
    routes = data['routes']; spells = data['spells']; losses = data['losses']
    tids = [home_id, away_id]; names = [home_name, away_name]

    def metric(key, digits=0, suffix=''):
        result = []
        for side, name in zip(['home', 'away'], names):
            rows = xg[xg.team.eq(name)] if key in xg else team_metrics[team_metrics.side.eq(side)]
            result.append(format_value(rows.iloc[0].get(key) if not rows.empty else None,
                                       suffix, digits).replace('Unavailable', '—'))
        return result

    def kpi(label, key, digits=0, suffix=''):
        return (label, *metric(key, digits, suffix))

    shots = events[flags(events, 'is_shot')
                   & ~flags(events, 'is_penalty_shootout')
                   & ~flags(events, 'is_own_goal')]
    paths = []; contracts = {}

    def board(number, title, subtitle, kpis):
        # Six panels, the reference grid. A three-panel version read better on
        # a phone but could not cover the match in four boards, which is what
        # the set is for; the type scale below carries the legibility instead.
        return Poster(info, number, title, subtitle, kpis, total=len(POSTERS))

    def finish(b, n):
        path = out / POSTERS[n - 1]
        b.save(path); paths.append(path); contracts[path.name] = b.notes

    # Six panels a board, twenty-four across the set, so the four cover the
    # match rather than sampling it: the result, how the ball was moved, what
    # the pressure cost, and who did it. Each board reuses the report's own
    # charts -- the sonar and the finishing slope come straight off their
    # reference pages -- rather than inventing a poster-only mark.

    # 1 - the result, and whether the chances agreed with it
    b = board(1, 'The result and the chances',
              'Where the goals came from and whether the chance quality matched the scoreline.',
              [kpi('Expected goals', 'xG', 2), kpi('Shots', 'shots'),
               kpi('On target', 'on_target'), kpi('Big chances', 'big_chances')])
    ax = b.panel(0, 'Expected goals through the match',
                 'Cumulative shot xG | each step is one attempt | elapsed event minute')
    end_minute = float((numeric(events, 'minute') + numeric(events, 'second').fillna(0) / 60).max())
    for tid, style in zip(tids, ['-', '--']):
        g = shots[shots.team_id.eq(tid)].sort_values(['minute', 'second'])
        total = numeric(g, 'xG').fillna(0).cumsum()
        ax.step(np.r_[0, numeric(g, 'minute') + numeric(g, 'second') / 60, end_minute],
                np.r_[0, total, total.iloc[-1] if len(total) else 0],
                where='post', color=b.colors[tid], lw=2.8, ls=style)
        if len(total):
            ax.annotate(f'{b.names[tid]}  {total.iloc[-1]:.2f}', (end_minute, total.iloc[-1]),
                        xytext=(-6, 7), textcoords='offset points', ha='right',
                        color=b.colors[tid], size=11, weight='bold')
    ax.set_xlim(0, max(end_minute, 1)); ax.set_ylim(bottom=0)
    ax.set_xlabel('Elapsed minute', size=11); ax.set_ylabel('Cumulative xG', size=11)
    b.comparisons(1, 'Chance balance',
                  [kpi('Goals', 'goals'), kpi('xG per shot', 'xG_per_shot', 3),
                   kpi('Box entries', 'box_entries'), kpi('Field tilt', 'field_tilt', 1, '%'),
                   kpi('Deep completions', 'deep_completions')])
    for i, tid in enumerate(tids):
        b.pitch(2 + i, names[i] + ' · shots', shots[shots.team_id.eq(tid)], tid,
                heat=a[a.team_id.eq(tid)])
    b.finishing(4, events, shots)
    outcomes = [kpi(label, key) for label, key in
                [('Goals', 'goals'), ('Saved', 'saved'), ('Blocked', 'blocked'), ('Off target', 'off_target')]]
    b.bars(5, 'Shot outcomes', [r[0] for r in outcomes],
           *[[_display_number(r[j]) for r in outcomes] for j in [1, 2]],
           'Recorded outcome counts | both teams use the same shot-count axis')
    b.axes[5].set_xlabel('Shots', size=10)
    finish(b, 1)

    # 2 - how each side moved the ball, and how far it got
    b = board(2, 'Build-up and penalty-area access',
              'The shape of the passing, the routes into the box and the possessions that reached a shot.',
              [kpi('Field tilt', 'field_tilt', 1, '%'), kpi('Progressive passes', 'progressive_passes'),
               kpi('Third entries', 'final_third_entries'), kpi('Box entries', 'box_entries')])
    every_pass = numeric(events[flags(events, 'is_pass')], 'pass_length').dropna()
    shared_reach = max(float(every_pass.quantile(.92)) if len(every_pass) else 0.0, 12.0)
    for i, tid in enumerate(tids):
        b.sonar(i, events, tid, names[i], reach=shared_reach)
    for i, tid in enumerate(tids):
        b.pitch(2 + i, names[i] + ' · box entries', routes[routes.team_id.eq(tid)], tid,
                arrows=True, heat=a[a.team_id.eq(tid)])
    counts = [stage_counts(p, tid) for tid in tids]
    b.bars(4, 'Nested possession stages',
           ['Possessions', 'Third', 'Then box', 'Then shot', 'On target'], *counts,
           'Same possession at every stage; long shots without box access excluded')
    lanes = []
    for tid in tids:
        g = routes[routes.team_id.eq(tid)]
        lanes.append(g.groupby('lane').size().reindex(['Left', 'Centre', 'Right'], fill_value=0).tolist())
    b.bars(5, 'Entry origin lane', ['Left', 'Centre', 'Right'], *lanes,
           'Entry events; several may belong to one possession')
    finish(b, 2)

    # 3 - the press, and what every loss cost
    b = board(3, 'Pressure, losses and transition output',
              'How hard each side pressed, where the ball was lost and what the opponent did with it.',
              [kpi('High regains', 'high_regains'), kpi('Transitions', 'transitions'),
               kpi('Transition shots', 'transition_shots'), kpi('Transition xG', 'transition_xG', 2)])
    from match_report import compute_ppda_both
    ppda = compute_ppda_both(info, events)
    b.comparisons(0, 'Pressing and second-phase totals',
                  [('PPDA · lower = harder press',
                    *(format_value(ppda[s].get('ppda'), digits=2) for s in ['home', 'away']), True),
                   kpi('Counterpress attempts', 'counterpress_attempts'),
                   kpi('Counterpress regains', 'counterpress_regains'),
                   kpi('Regain-to-shot rate', 'regain_to_shot_rate', 1, '%'),
                   kpi('Transition-to-shot rate', 'transition_shot_rate', 1, '%')],
                  'PPDA: passes / defensive action | longest bar = hardest press')
    values = [[len(g)] + [int(g[k].sum()) for k in ['third_within_12', 'box_within_12', 'shot_within_12']]
              for tid in tids for g in [losses[losses.team_id.eq(tid)]]]
    b.bars(1, 'Opponent output after losses', ['Losses', 'Third ≤12s', 'Box ≤12s', 'Shot ≤12s'],
           *values, 'Immediately following open-play possession; outcome counts are independent')
    for i, tid in enumerate(tids):
        frame = losses[losses.team_id.eq(tid)].copy()
        frame['xG'] = frame.xG_conceded
        b.pitch(2 + i, names[i] + ' · loss locations', frame, tid, heat=a[a.team_id.eq(tid)])
    state_xg = []; minutes = []
    for tid in tids:
        g = spells[spells.team_id.eq(tid)].groupby('state')[['minutes', 'xG']].sum().reindex(
            ['level', 'leading', 'trailing'], fill_value=0)
        state_xg.append((g.xG * 30 / g.minutes).where(g.minutes >= 5).round(2))
        minutes.append(g.minutes)
    b.bars(4, 'xG per 30 score-state minutes', ['Level', 'Leading', 'Trailing'],
           *[v.tolist() for v in state_xg], 'Shared xG axis | rate withheld below 5 minutes in a state')
    b.axes[4].set_xlabel('xG / 30 minutes', size=10)
    for j, series in enumerate(state_xg):
        for row, v in enumerate(series):
            if not np.isfinite(v):
                b.axes[4].text(.98, row + (-.15 if j == 0 else .15), names[j] + ': <5 min',
                               transform=b.axes[4].get_yaxis_transform(), ha='right',
                               color=b.colors[tids[j]], size=9)
    # The press as a shape, beside the rate. The paired-bar row above answers
    # how hard; six axes answer what kind.
    def axis_value(side, key, default=0.0):
        """The raw figure for one side, straight off the metric frames.

        metric() formats for display -- it rounds, appends a unit and turns a
        missing value into an em dash -- so a radius has to come from the frame
        rather than from the string the panel would print.
        """
        rows = team_metrics[team_metrics.side.eq(side)]
        if rows.empty or key not in rows:
            return default
        try:
            value = pd.to_numeric(rows.iloc[0].get(key), errors="coerce")
            return default if pd.isna(value) else float(value)
        except (TypeError, ValueError):
            return default

    def punished(side):
        exposures = axis_value(side, 'rest_defence_exposures')
        return (100.0 * axis_value(side, 'rest_defence_dangerous_counters') / exposures
                if exposures else 0.0)

    b.press_radar(5, [
        ("Press\nintensity", float(ppda["home"].get("ppda") or 0),
         float(ppda["away"].get("ppda") or 0), 6.0, 20.0, True, 1),
        ("High\nrecoveries", axis_value("home", "high_regains"),
         axis_value("away", "high_regains"), 0.0, 22.0, False, 0),
        ("Regain\nconversion", axis_value("home", "regain_to_shot_rate"),
         axis_value("away", "regain_to_shot_rate"), 0.0, 20.0, False, 1),
        ("Counter\npress", axis_value("home", "counterpress_success_rate"),
         axis_value("away", "counterpress_success_rate"), 0.0, 25.0, False, 1),
        ("Territory", axis_value("home", "field_tilt"),
         axis_value("away", "field_tilt"), 0.0, 100.0, False, 1),
        ("Rest\ndefence", punished("home"), punished("away"), 0.0, 30.0, True, 1),
    ])
    finish(b, 3)

    # 4 - who decided it
    b = board(4, 'Player contribution and finishing',
              'Chance involvement, how the ball was moved to get there, and what the keepers saw.',
              [kpi('Sequence threat xT', 'sequence_xT', 2), kpi('Progressive passes', 'progressive_passes'),
               kpi('Deep completions', 'deep_completions'), kpi('Box entries', 'box_entries')])
    ceiling = max(float(numeric(people, 'xG').max() or 0), float(numeric(people, 'xA').max() or 0), .01)
    for i, tid in enumerate(tids):
        b.ranking(i, names[i] + ' · chance involvement', people[people.team_id.eq(tid)], tid, ceiling)
    b.scatter(2, 'Progression and creation', people, 'progressions', 'xA',
              'Progressions', 'Inferred xA', players=True)
    b.scatter(3, 'Shot volume and average quality', people[people.shots > 0], 'shots',
              'xG_per_shot', 'Shots', 'xG per shot', players=True, qualifier='at least one shot')
    keeper_rows = [('Recorded saves', 'saves'), ('Claims', 'claims'), ('Sweeper actions', 'sweeps')]
    rows = []
    for label, key in keeper_rows:
        pair = []
        for tid in tids:
            keeper = people[people.team_id.eq(tid) & people.role.astype(str).str.upper().isin(['GK', 'GOALKEEPER'])]
            pair.append(float(keeper[key].sum()) if not keeper.empty and key in keeper else 0.0)
        rows.append((label, *pair))
    b.bars(4, 'Goalkeeper actions', [r[0] for r in rows],
           *[[r[j] for r in rows] for j in [1, 2]],
           'Recorded action counts | both keepers share the same count axis')
    b.comparisons(5, 'Delivery and access',
                  [kpi('Crosses attempted', 'crosses'), kpi('Crosses completed', 'completed_crosses'),
                   kpi('Deep completions', 'deep_completions'),
                   kpi('Build-up success', 'build_up_success_rate', 1, '%'),
                   kpi('Box-to-shot rate', 'box_entry_to_shot_rate', 1, '%')],
                  'Shared scale per row | rates use their own denominators, out of 100%')
    finish(b, 4)

    (out / 'poster_catalog.json').write_text(json.dumps(contracts, indent=2), encoding='utf-8')
    return paths
