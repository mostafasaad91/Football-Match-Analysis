"""Duration-weighted touch-position influence, split at substitutions.

It is an illustrative event model, not a tracking-based control probability.
"""
import numpy as np
import pandas as pd


def surface(events,home_id,away_id,cells_x=60,cells_y=40):
    from match_metrics import touch_mask, live_event_mask
    work=events[live_event_mask(events)].copy()
    work['_t']=pd.to_numeric(work.minute,errors='coerce')*60+pd.to_numeric(work.second,errors='coerce').fillna(0)
    work['_period']=work.get('period_code',pd.Series('1H',index=work.index)).astype(str)
    yy,xx=np.meshgrid(np.linspace(0,100,cells_y),np.linspace(0,100,cells_x),indexing='ij')
    weighted=np.zeros((cells_y,cells_x));exposure=0.
    all_clock=pd.to_numeric(events.minute,errors='coerce')*60+pd.to_numeric(events.second,errors='coerce').fillna(0)
    for period,g in work.groupby('_period'):
        if g.empty:continue
        low=float(g._t.min());high=float(g._t.max())
        changes=events[events.type.isin(['SubstitutionOn','SubstitutionOff','Card']) & events.get('period_code',pd.Series('',index=events.index)).astype(str).eq(period)]
        edges=sorted(set([low,high]+list(np.arange(low,high,300))+all_clock.loc[changes.index].dropna().tolist()))
        for start,end in zip(edges,edges[1:]):
            if end<=start:continue
            chunk=g[g._t.between(start,end,inclusive='left')]
            chunk=chunk[touch_mask(chunk)].dropna(subset=['x','y','player','team_id'])
            positions=chunk.groupby(['team_id','player'])[['x','y']].mean()
            fields=[]
            for tid in [home_id,away_id]:
                if tid not in positions.index.get_level_values(0):break
                field=np.zeros_like(weighted)
                for row in positions.loc[tid].itertuples():
                    px,py=(row.x,row.y) if tid==home_id else (100-row.x,100-row.y)
                    field+=np.exp(-np.hypot((xx-px)*1.05,(yy-py)*.68)/11.)
                fields.append(field)
            if len(fields)!=2:continue
            probability=fields[0]/np.maximum(fields[0]+fields[1],1e-9)
            weighted+=probability*(end-start);exposure+=end-start
    grid=weighted/exposure if exposure else np.full((cells_y,cells_x),.5)
    home=100*np.mean(grid>.55);away=100*np.mean(grid<.45)
    return grid,{'home':round(home,1),'away':round(away,1),'contested':round(100-home-away,1),'observed_minutes':round(exposure/60,1)}
