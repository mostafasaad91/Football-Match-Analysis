"""Generate a preview of the PPDA gauge with sample data."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match_extensions import draw_ppda_gauge

ppda_data = {
    "home": {"ppda": 8.42, "passes_allowed": 287, "defensive_actions": 34},
    "away": {"ppda": 12.65, "passes_allowed": 354, "defensive_actions": 28},
}
info = {"home_name": "Liverpool", "away_name": "Manchester City"}

os.makedirs("output/visuals", exist_ok=True)
out = "output/visuals/preview_ppda.png"
draw_ppda_gauge(ppda_data, info, save_path=out)
print(f"Saved: {out}")
