"""Comparable observations for charts: one player, possession or spell per row.

No off-ball positions, coach intentions or causal substitution effects are inferred.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from match_metrics import (build_possessions, progressive_pass_mask, box_entry_mask,
                           touch_mask, live_event_mask, is_restart_event,
                           turnover_events, FINAL_THIRD_X, BOX_X, BOX_Y_MIN, BOX_Y_MAX)

def numeric(frame, key, default=0.0):
    return pd.to_numeric(frame.get(key, pd.Series(default, index=frame.index)), errors="coerce")

def flags(frame, key):
    return frame.get(key, pd.Series(False, index=frame.index)).astype(str).str.lower().isin(["true", "1", "1.0", "yes"])

def successful(frame):
    return frame.get("outcome", pd.Series("", index=frame.index)).astype(str).str.lower().eq("successful")

def player_observations(events, players):
    from player_radar import _get_participation, _get_credits
    from match_metrics import player_sequence_metrics
    sequence = player_sequence_metrics(events)
    live = events[live_event_mask(events)].copy()
    part = _get_participation(events)
    credits = _get_credits(events)
    live["_touch"] = touch_mask(live)
    live["_prog"] = progressive_pass_mask(live)
    live["_box"] = box_entry_mask(live)
    live["_shot"] = flags(live, "is_shot") & ~flags(live,'is_own_goal')
    # Same exclusion, stated once. An own goal carries is_goal as well as
    # is_shot, and reading is_goal on its own credited João Pedro with two
    # goals — one for Chelsea, one past his own keeper — on a card whose Shots
    # row, built from _shot, already had it right.
    live["_goal"] = flags(live, "is_goal") & ~flags(live,'is_own_goal')
    live["_good"] = successful(live)
    live["_xt"] = numeric(live, "xT", np.nan).where(live["_good"] & live["type"].isin(["Pass", "Carry"])).clip(lower=0)
    # Carries are included only when explicitly supplied, never reconstructed as tracking.
    live["_carry"] = live["type"].eq("Carry") & live["_good"] & (numeric(live, "end_x")-numeric(live, "x") >= 10)
    rows = []
    for (tid, name), g in live.dropna(subset=["player", "team_id"]).groupby(["team_id", "player"], sort=True):
        if not str(name).strip():
            continue
        squad = players[(players["name"].eq(name)) & players["team_id"].eq(tid)] if players is not None and not players.empty else pd.DataFrame()
        role = str(squad.iloc[0].get("position", "Unknown")) if not squad.empty else "Unknown"
        participation = part.get(str(name), {})
        seconds = participation.get("seconds", participation.get("played_seconds", 0))
        minutes = float(seconds or 0)/60
        # Participation helper exposes minutes in current and older saved versions.
        if not minutes:
            minutes = float(participation.get("minutes", 0) or 0)
        passes = g["type"].eq("Pass")
        takeons = g["type"].eq("TakeOn")
        shots = int(g["_shot"].sum())
        xg = numeric(g[g["_shot"]], "xG", np.nan).sum(min_count=1) if shots else 0.0
        creation = credits.get(str(name), {})
        xa = creation.get("xA", creation.get("xa", 0.0)) if isinstance(creation, dict) and 'is_key_pass' in events and 'xG' in events else np.nan
        chain = sequence.get(str(name), {})
        rows.append(dict(player=str(name), team_id=tid, role=role, minutes=minutes,
            xGChain=float(chain.get('xGChain',0)), xGBuildup=float(chain.get('xGBuildup',0)),
            touches=int(g["_touch"].sum()), progressive_passes=int(g["_prog"].sum()),
            progressive_carries=int(g["_carry"].sum()), progressions=int(g["_prog"].sum()+g["_carry"].sum()),
            box_entries=int(g["_box"].sum()), shots=shots, xG=float(xg),
            xG_per_shot=float(xg/shots) if shots else np.nan, xA=float(xa),
            positive_xT=(0.0 if 'xT' in events and not (g['_good'] & g.type.isin(['Pass','Carry'])).any() else g["_xt"].sum(min_count=1)), passes=int(passes.sum()),
            completed_passes=int((passes & g["_good"]).sum()),
            pass_pct=100*(passes & g["_good"]).sum()/passes.sum() if passes.any() else np.nan,
            takeons=int(takeons.sum()), takeons_won=int((takeons & g["_good"]).sum()),
            recoveries=int(g["type"].eq("BallRecovery").sum()),
            interceptions=int(g["type"].eq("Interception").sum()),
            tackles_won=int((g["type"].eq("Tackle") & g["_good"]).sum()),
            clearances=int(g["type"].eq("Clearance").sum()),
            goals=int(g["_goal"].sum()),
            saves=int(g["type"].eq("Save").sum()), claims=int(g["type"].eq("Claim").sum()),
            sweeps=int(g["type"].eq("KeeperSweeper").sum())))
    result=pd.DataFrame(rows)
    if not result.empty:
        result['xT_per_100_touches']=100*result.positive_xT/result.touches.replace(0,np.nan)
        result['progressive_pass_pct']=100*result.progressive_passes/result.completed_passes.replace(0,np.nan)
        result['xA_per_100_passes']=100*result.xA/result.passes.replace(0,np.nan)
        result['xG_xA']=result.xG+result.xA
        result['takeon_success_pct']=100*result.takeons_won/result.takeons.replace(0,np.nan)
    return result

def possession_observations(events):
    actions, possessions = build_possessions(events)
    if possessions.empty:
        return actions, possessions.copy()
    rows = []
    for p in possessions.to_dict("records"):
        g = actions[actions["possession_id"].eq(p["possession_id"]) & actions["team_id"].eq(p["team_id"])].copy()
        good = successful(g)
        move = g["type"].isin(["Pass", "Carry"]) & good
        reach = (numeric(g, "x") >= FINAL_THIRD_X) | (move & (numeric(g, "end_x") >= FINAL_THIRD_X))
        box = ((numeric(g, "x") >= BOX_X) & numeric(g, "y").between(BOX_Y_MIN, BOX_Y_MAX)) | (
            move & (numeric(g, "end_x") >= BOX_X) & numeric(g, "end_y").between(BOX_Y_MIN, BOX_Y_MAX))
        clock = numeric(g, "_clock_seconds")
        reach_at = clock[reach].min() if reach.any() else np.nan
        box_at = clock[box & (clock >= reach_at)].min() if box.any() and reach.any() else np.nan
        # An own goal is not this possession reaching a shot; without the
        # exclusion a side that forced one was credited with the attempt.
        shot = flags(g, "is_shot") & ~flags(g, "is_own_goal") & (clock >= box_at)
        first_shot = clock[shot].min() if shot.any() else np.nan
        from match_metrics import blocked_shot_mask
        target = g['type'].isin(["Goal", "SavedShot"]) & shot & ~blocked_shot_mask(g)
        regain = p["start_reason"] in {"recovery", "opponent_turnover"}
        positive = numeric(g, "xT", np.nan).where(move).clip(lower=0).sum(min_count=1)
        rows.append({**p, "positive_xT":positive, "reached_final_third":bool(reach.any()),
            "reached_box_after_third":bool(pd.notna(box_at)), "shot_after_box":bool(pd.notna(first_shot)),
            "target_after_box":bool(target.any()), "is_regain":regain,
            "seconds_to_third":float(reach_at-p["start_time"]) if pd.notna(reach_at) else np.nan,
            "third_within_12":bool(regain and pd.notna(reach_at) and reach_at-p["start_time"] <= 12),
            "box_within_12":bool(regain and pd.notna(box_at) and box_at-p["start_time"] <= 12),
            "shot_within_12":bool(regain and ((flags(g,"is_shot")) & ((clock-p["start_time"]) <= 12)).any())})
    return actions, pd.DataFrame(rows)

def stage_counts(possessions, team_id):
    own = possessions[possessions["team_id"].eq(team_id)]
    keys = ["reached_final_third", "reached_box_after_third", "shot_after_box", "target_after_box"]
    return [len(own)] + [int(own[k].sum()) for k in keys]

def entry_routes(actions, team_id):
    rows=[]
    entries=actions[actions["team_id"].eq(team_id) & box_entry_mask(actions)]
    for _, e in entries.iterrows():
        tail=actions[actions["possession_id"].eq(e["possession_id"]) & actions["team_id"].eq(team_id) & (actions["_clock_seconds"]>=e["_clock_seconds"])]
        y=float(e["y"]); end_y=float(e["end_y"])
        # Opta y=0 is the right touchline in the normalized attacking frame.
        lane="Right" if y<33.3 else "Left" if y>66.7 else "Centre"
        cross="cross" in str(e.get("qualifier_names", "")).lower()
        cutback=cross and float(e["x"])>=90 and float(e["end_x"])<float(e["x"]) and 21<=end_y<=79
        kind="Carry" if e["type"]=="Carry" else "Cut-back candidate" if cutback else "Cross" if cross else "Pass"
        rows.append(dict(team_id=team_id,minute=e["minute"],player=e["player"],lane=lane,kind=kind,
            x=e["x"],y=y,end_x=e["end_x"],end_y=end_y,shot_followed=bool(flags(tail,"is_shot").any()),
            xG_followed=numeric(tail[flags(tail,"is_shot")],"xG").sum()))
    return pd.DataFrame(rows,columns=["team_id","minute","player","lane","kind","x","y","end_x","end_y","shot_followed","xG_followed"])

def verified_receptions(actions):
    """Conservative receiver inference: next controlled touch, same possession/side,
    <=5 seconds and endpoint within 5m. Unmatched passes stay unassigned."""
    rows=[]
    a=actions.reset_index(drop=True)
    touches=touch_mask(a)
    for i in a.index[a["type"].eq("Pass") & successful(a)]:
        e=a.loc[i]
        candidates=a.loc[i+1:]
        candidates=candidates[touches.loc[candidates.index]]
        if candidates.empty: continue
        nxt=candidates.iloc[0]
        if pd.isna(e["possession_id"]) or pd.isna(nxt["possession_id"]): continue
        dt=nxt["_clock_seconds"]-e["_clock_seconds"]
        distance=np.hypot((nxt["x"]-e["end_x"])*1.05,(nxt["y"]-e["end_y"])*.68)
        if nxt["possession_id"]!=e["possession_id"] or nxt["team_id"]!=e["team_id"] or not 0<=dt<=5 or not np.isfinite(distance) or distance>5: continue
        if not isinstance(nxt.get("player"),str) or nxt["player"]==e["player"]: continue
        rows.append(dict(player=nxt["player"],team_id=nxt["team_id"],minute=nxt["minute"],x=e["end_x"],y=e["end_y"],
                         next_type=nxt["type"],end_x=nxt.get("end_x"),end_y=nxt.get("end_y"),seconds=dt))
    return pd.DataFrame(rows,columns=["player","team_id","minute","x","y","next_type","end_x","end_y","seconds"])

def score_spells(events, info):
    """Nonoverlapping within-period score spells; goal belongs to pre-goal state."""
    a,_=build_possessions(events)
    if a.empty:return pd.DataFrame()
    score={info["home_id"]:0,info["away_id"]:0}; rows=[]
    for period,g in a.groupby("_period_order",sort=True):
        if not g["_period_code"].isin(["1h","2h","et1","et2"]).any():continue
        nominal={1:0,2:45*60,3:90*60,4:105*60}.get(int(period),float(g["_clock_seconds"].min()))
        start=min(nominal,float(g["_clock_seconds"].min())); current=[]
        def append(end):
            frame=g.loc[current]
            for side in ("home","away"):
                tid=info[f"{side}_id"]; opp=info["away_id" if side=="home" else "home_id"]
                state="level" if score[tid]==score[opp] else "leading" if score[tid]>score[opp] else "trailing"
                own=frame[frame["team_id"].eq(tid)]
                shot=flags(own,"is_shot") & ~flags(own,"is_own_goal")
                rows.append(dict(side=side,team_id=tid,period=period,state=state,start=start/60,end=end/60,
                    minutes=max(0,end-start)/60,shots=int(shot.sum()),xG=numeric(own[shot],"xG").sum(),box_entries=int(box_entry_mask(own).sum())))
        for idx,e in g.iterrows():
            current.append(idx)
            if str(e.get("is_goal")).lower() in {"true","1","1.0"}:
                end=float(e["_clock_seconds"]);append(end)
                credited=e.get("scoring_team")
                if pd.isna(credited):
                    credited=e["team_id"]
                    if str(e.get("is_own_goal")).lower() in {"true","1","1.0"}:
                        credited=info["away_id"] if credited==info["home_id"] else info["home_id"]
                if credited in score:score[credited]+=1
                start=end;current=[]
        append(float(g["_clock_seconds"].max()))
    return pd.DataFrame(rows)

def build_insights(events,players,info):
    a,p=possession_observations(events)
    losses=[]
    ordered=p.sort_values(['period_order','start_time']).reset_index(drop=True)
    for i in range(len(ordered)-1):
        current,nxt=ordered.iloc[i],ordered.iloc[i+1]
        if current.team_id==nxt.team_id or current.period_order!=nxt.period_order or nxt.start_reason not in ['recovery','opponent_turnover']:continue
        tail=a[a.possession_id.eq(nxt.possession_id)&a.team_id.eq(nxt.team_id)]
        tail=tail[(tail._clock_seconds-current.end_time).between(0,12)]
        good=successful(tail)&tail.type.isin(['Pass','Carry'])
        third=(numeric(tail,'x')>=FINAL_THIRD_X)|(good&(numeric(tail,'end_x')>=FINAL_THIRD_X))
        box=((numeric(tail,'x')>=BOX_X)&numeric(tail,'y').between(BOX_Y_MIN,BOX_Y_MAX))|box_entry_mask(tail)
        shot=flags(tail,'is_shot')
        losses.append(dict(team_id=current.team_id,minute=current.end_time/60,x=current.end_x,y=current.end_y,
                           third_within_12=bool(third.any()),box_within_12=bool(box.any()),shot_within_12=bool(shot.any()),
                           xG_conceded=numeric(tail[shot],'xG').sum()))
    return {"players":player_observations(events,players),"actions":a,"possessions":p,
            "receptions":verified_receptions(a),"spells":score_spells(events,info),
            "losses":pd.DataFrame(losses,columns=['team_id','minute','x','y','third_within_12','box_within_12','shot_within_12','xG_conceded']),
            "routes":pd.concat([entry_routes(a,info[f"{s}_id"]) for s in ("home","away")],ignore_index=True)}
