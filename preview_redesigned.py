"""Generate preview PNGs of the redesigned visuals."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

np.random.seed(42)

from viz_redesigned import shot_map_v2, xg_flow_v2, pass_network_v2
from viz_design_system import C_HOME, C_AWAY, C_GOLD

os.makedirs("output/visuals", exist_ok=True)

# ── Sample data ──
info = {
    "home_name": "Liverpool", "away_name": "Manchester City",
    "home_id": 1, "away_id": 2,
    "score": "3 - 1",
}

# Players
home_players = ["M. Salah", "V. van Dijk", "T. Alexander-Arnold",
                "A. Mac Allister", "D. Szoboszlai", "C. Gakpo",
                "A. Robertson", "I. Konaté", "R. Gravenberch",
                "Alisson", "D. Jota"]
away_players = ["E. Haaland", "K. De Bruyne", "Rodri", "B. Silva",
                "P. Foden", "R. Dias", "K. Walker", "Ederson",
                "J. Stones", "J. Grealish", "J. Álvarez"]

events_list = []
eid = 0

# Generate shots
shot_types = ["Goal", "SavedShot", "MissedShots", "BlockedShot", "ShotOnPost"]
for team_id, players, scorers in [(1, home_players, ["M. Salah", "V. van Dijk", "D. Núñez"]),
                                    (2, away_players, ["E. Haaland"])]:
    n_shots = 18 if team_id == 1 else 12
    for i in range(n_shots):
        st = np.random.choice(shot_types, p=[0.18, 0.30, 0.25, 0.20, 0.07])
        is_goal = (st == "Goal")
        x = np.random.uniform(70, 99)
        y = np.random.uniform(15, 85)
        xg = round(np.random.uniform(0.02, 0.55) +
                   (0.25 if is_goal else 0), 3)
        player = (np.random.choice(scorers) if is_goal
                  else np.random.choice(players))
        events_list.append({
            "event_id": eid, "minute": int(np.random.uniform(2, 89)),
            "second": 0, "team_id": team_id,
            "player_id": team_id * 100 + players.index(player) if player in players else team_id * 100,
            "player": player, "type": st,
            "outcome": "Successful" if is_goal else "Unsuccessful",
            "x": x, "y": y, "end_x": 100, "end_y": 50,
            "is_shot": True, "is_pass": False, "is_key_pass": False,
            "is_goal": is_goal, "is_own_goal": False,
            "scoring_team": team_id, "is_penalty": False,
            "is_header": False, "is_direct_fk": False,
            "shot_whoscored_type": st, "shot_category": "On Target" if is_goal else "Other",
            "xG": xg, "big_chance": np.random.choice([0, 1], p=[0.85, 0.15]),
            "qualifier_names": [],
        })
        eid += 1

# Generate passes (for pass network)
for team_id, players in [(1, home_players), (2, away_players)]:
    pos_x = np.linspace(20, 75, len(players))
    pos_y = np.random.uniform(15, 85, len(players))
    n_passes = 350 if team_id == 1 else 280
    for i in range(n_passes):
        idx = np.random.randint(0, len(players))
        player = players[idx]
        x = pos_x[idx] + np.random.normal(0, 8)
        y = pos_y[idx] + np.random.normal(0, 12)
        events_list.append({
            "event_id": eid,
            "minute": int(np.random.uniform(0, 90)),
            "second": int(np.random.uniform(0, 60)),
            "team_id": team_id,
            "player_id": team_id * 100 + idx,
            "player": player, "type": "Pass",
            "outcome": np.random.choice(["Successful", "Unsuccessful"],
                                        p=[0.85, 0.15]),
            "x": np.clip(x, 0, 100), "y": np.clip(y, 0, 100),
            "end_x": np.clip(x + np.random.normal(10, 5), 0, 100),
            "end_y": np.clip(y + np.random.normal(0, 10), 0, 100),
            "is_shot": False, "is_pass": True, "is_key_pass": False,
            "is_goal": False, "is_own_goal": False,
            "scoring_team": None, "is_penalty": False,
            "is_header": False, "is_direct_fk": False,
            "shot_whoscored_type": None, "shot_category": None,
            "xG": None, "big_chance": 0,
            "qualifier_names": [],
        })
        eid += 1

events = pd.DataFrame(events_list)

# 1) Shot Map
shot_map_v2(events, info, team_id=1,
            team_color=C_HOME, team_name="Liverpool", accent=C_GOLD,
            save_path="output/visuals/redesign_shot_map.png")
print("Saved: output/visuals/redesign_shot_map.png")

# 2) xG Flow
xg_flow_v2(events, info,
           save_path="output/visuals/redesign_xg_flow.png")
print("Saved: output/visuals/redesign_xg_flow.png")

# 3) Pass Network
pass_network_v2(events, info, team_id=1,
                team_color=C_HOME, team_name="Liverpool", accent=C_GOLD,
                save_path="output/visuals/redesign_pass_network.png")
print("Saved: output/visuals/redesign_pass_network.png")
