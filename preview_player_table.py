"""Generate a preview PNG of the player stats table with sample data."""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match_extensions import draw_player_stats_table, C_HOME

# — 4-3-3 home team
sample = [
    # GK
    {"name": "Alisson Becker",     "position": "GK",  "shirt_no": 1,  "is_first_xi": True,
     "minutesPlayed": 90, "goals": 0, "assists": 0, "shotsTotal": 0, "shotsOnTarget": 0,
     "passesKey": 0, "dribblesWon": 0, "passesTotal": 32, "passesAccurate": 28,
     "touches": 41, "tacklesTotal": 0, "interceptions": 1, "aerialsWon": 0,
     "foulsCommited": 0, "wasFouled": 0, "ratings": 7.12},
    # Defenders
    {"name": "Trent Alexander-Arnold", "position": "DR", "shirt_no": 66, "is_first_xi": True,
     "minutesPlayed": 90, "goals": 0, "assists": 1, "shotsTotal": 1, "shotsOnTarget": 1,
     "passesKey": 4, "dribblesWon": 2, "passesTotal": 78, "passesAccurate": 65,
     "touches": 95, "tacklesTotal": 2, "interceptions": 1, "aerialsWon": 1,
     "foulsCommited": 1, "wasFouled": 2, "ratings": 8.35},
    {"name": "Virgil van Dijk", "position": "DC", "shirt_no": 4, "is_first_xi": True,
     "minutesPlayed": 90, "goals": 1, "assists": 0, "shotsTotal": 2, "shotsOnTarget": 2,
     "passesKey": 0, "dribblesWon": 0, "passesTotal": 88, "passesAccurate": 84,
     "touches": 102, "tacklesTotal": 3, "interceptions": 4, "aerialsWon": 6,
     "foulsCommited": 1, "wasFouled": 1, "ratings": 8.74},
    {"name": "Ibrahima Konaté", "position": "DC", "shirt_no": 5, "is_first_xi": True,
     "minutesPlayed": 90, "goals": 0, "assists": 0, "shotsTotal": 0, "shotsOnTarget": 0,
     "passesKey": 0, "dribblesWon": 0, "passesTotal": 76, "passesAccurate": 71,
     "touches": 89, "tacklesTotal": 2, "interceptions": 3, "aerialsWon": 5,
     "foulsCommited": 2, "wasFouled": 0, "ratings": 7.45},
    {"name": "Andrew Robertson", "position": "DL", "shirt_no": 26, "is_first_xi": True,
     "minutesPlayed": 90, "goals": 0, "assists": 0, "shotsTotal": 1, "shotsOnTarget": 0,
     "passesKey": 2, "dribblesWon": 1, "passesTotal": 65, "passesAccurate": 54,
     "touches": 84, "tacklesTotal": 3, "interceptions": 2, "aerialsWon": 2,
     "foulsCommited": 1, "wasFouled": 1, "ratings": 7.31},
    # Midfielders
    {"name": "Alexis Mac Allister", "position": "DMC", "shirt_no": 10, "is_first_xi": True,
     "minutesPlayed": 90, "goals": 0, "assists": 0, "shotsTotal": 1, "shotsOnTarget": 0,
     "passesKey": 1, "dribblesWon": 2, "passesTotal": 82, "passesAccurate": 76,
     "touches": 98, "tacklesTotal": 4, "interceptions": 2, "aerialsWon": 3,
     "foulsCommited": 2, "wasFouled": 3, "ratings": 7.68},
    {"name": "Dominik Szoboszlai", "position": "MC", "shirt_no": 8, "is_first_xi": True,
     "minutesPlayed": 78, "goals": 0, "assists": 1, "shotsTotal": 3, "shotsOnTarget": 2,
     "passesKey": 3, "dribblesWon": 4, "passesTotal": 54, "passesAccurate": 47,
     "touches": 72, "tacklesTotal": 2, "interceptions": 1, "aerialsWon": 1,
     "foulsCommited": 1, "wasFouled": 2, "ratings": 7.92},
    {"name": "Ryan Gravenberch", "position": "MC", "shirt_no": 38, "is_first_xi": True,
     "minutesPlayed": 85, "goals": 0, "assists": 0, "shotsTotal": 1, "shotsOnTarget": 0,
     "passesKey": 1, "dribblesWon": 3, "passesTotal": 61, "passesAccurate": 55,
     "touches": 78, "tacklesTotal": 3, "interceptions": 2, "aerialsWon": 2,
     "foulsCommited": 1, "wasFouled": 1, "ratings": 7.41},
    # Forwards
    {"name": "Mohamed Salah", "position": "FWR", "shirt_no": 11, "is_first_xi": True,
     "minutesPlayed": 90, "goals": 2, "assists": 1, "shotsTotal": 5, "shotsOnTarget": 4,
     "passesKey": 5, "dribblesWon": 6, "passesTotal": 38, "passesAccurate": 30,
     "touches": 68, "tacklesTotal": 0, "interceptions": 0, "aerialsWon": 0,
     "foulsCommited": 0, "wasFouled": 4, "ratings": 9.21},
    {"name": "Cody Gakpo", "position": "FWL", "shirt_no": 18, "is_first_xi": True,
     "minutesPlayed": 72, "goals": 1, "assists": 0, "shotsTotal": 4, "shotsOnTarget": 2,
     "passesKey": 2, "dribblesWon": 3, "passesTotal": 28, "passesAccurate": 21,
     "touches": 52, "tacklesTotal": 1, "interceptions": 0, "aerialsWon": 1,
     "foulsCommited": 0, "wasFouled": 2, "ratings": 7.85},
    {"name": "Diogo Jota", "position": "FW", "shirt_no": 20, "is_first_xi": True,
     "minutesPlayed": 65, "goals": 1, "assists": 0, "shotsTotal": 3, "shotsOnTarget": 2,
     "passesKey": 1, "dribblesWon": 1, "passesTotal": 19, "passesAccurate": 14,
     "touches": 38, "tacklesTotal": 0, "interceptions": 1, "aerialsWon": 2,
     "foulsCommited": 1, "wasFouled": 1, "ratings": 7.72},
    # Substitutes
    {"name": "Darwin Núñez", "position": "Sub", "shirt_no": 9, "is_first_xi": False,
     "minutesPlayed": 25, "goals": 0, "assists": 1, "shotsTotal": 2, "shotsOnTarget": 1,
     "passesKey": 1, "dribblesWon": 1, "passesTotal": 8, "passesAccurate": 6,
     "touches": 14, "tacklesTotal": 0, "interceptions": 0, "aerialsWon": 1,
     "foulsCommited": 1, "wasFouled": 0, "ratings": 6.94},
    {"name": "Curtis Jones", "position": "Sub", "shirt_no": 17, "is_first_xi": False,
     "minutesPlayed": 18, "goals": 0, "assists": 0, "shotsTotal": 0, "shotsOnTarget": 0,
     "passesKey": 0, "dribblesWon": 1, "passesTotal": 14, "passesAccurate": 12,
     "touches": 19, "tacklesTotal": 1, "interceptions": 0, "aerialsWon": 0,
     "foulsCommited": 0, "wasFouled": 0, "ratings": 6.88},
    {"name": "Joe Gomez", "position": "Sub", "shirt_no": 2, "is_first_xi": False,
     "minutesPlayed": 12, "goals": 0, "assists": 0, "shotsTotal": 0, "shotsOnTarget": 0,
     "passesKey": 0, "dribblesWon": 0, "passesTotal": 9, "passesAccurate": 8,
     "touches": 11, "tacklesTotal": 1, "interceptions": 1, "aerialsWon": 0,
     "foulsCommited": 0, "wasFouled": 0, "ratings": 6.71},
    {"name": "Wataru Endo", "position": "Sub", "shirt_no": 3, "is_first_xi": False,
     "minutesPlayed": 5, "goals": 0, "assists": 0, "shotsTotal": 0, "shotsOnTarget": 0,
     "passesKey": 0, "dribblesWon": 0, "passesTotal": 3, "passesAccurate": 3,
     "touches": 4, "tacklesTotal": 0, "interceptions": 0, "aerialsWon": 0,
     "foulsCommited": 0, "wasFouled": 0, "ratings": 6.55},
]

df = pd.DataFrame(sample)
out_dir = os.path.join("output", "visuals")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "preview_player_stats.png")
draw_player_stats_table(df, "Liverpool (Sample)", team_color=C_HOME, save_path=out)
print(f"Saved: {out}")
