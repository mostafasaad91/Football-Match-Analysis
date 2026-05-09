"""Save preview PNGs of the cover + section divider."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import matplotlib.pyplot as plt

# helpers figure
import match_extensions as ext
from match_extensions import C_GOLD, BG_DARK

# Patch pdf.savefig + plt.close to no-op so we can grab figures
class _FakePdf:
    def __init__(self): self.figs = []
    def savefig(self, fig, **kw): self.figs.append(fig)

_orig_close = plt.close
plt.close = lambda *a, **k: None  # don't close

info = {
    "home_name": "Liverpool", "away_name": "Manchester City",
    "home_id": 1, "away_id": 2,
    "score": "3 - 1",
    "venue": "Anfield Stadium",
    "home_form": "4-3-3", "away_form": "4-2-3-1",
}
goals_df = pd.DataFrame([
    {"minute": 12, "team": "Liverpool", "scorer": "M. Salah",
     "assist": "T. Alexander-Arnold", "category": "Open Play",
     "subtype": "Open Play", "xG": 0.31, "is_own_goal": False, "scoring_team_id": 1},
    {"minute": 38, "team": "Liverpool", "scorer": "V. van Dijk",
     "assist": "A. Robertson", "category": "Set Piece",
     "subtype": "Corner", "xG": 0.18, "is_own_goal": False, "scoring_team_id": 1},
    {"minute": 67, "team": "Manchester City", "scorer": "E. Haaland",
     "assist": "K. De Bruyne", "category": "Open Play",
     "subtype": "Open Play", "xG": 0.42, "is_own_goal": False, "scoring_team_id": 2},
    {"minute": 89, "team": "Liverpool", "scorer": "D. Núñez",
     "assist": "N/A", "category": "Set Piece",
     "subtype": "Penalty", "xG": 0.78, "is_own_goal": False, "scoring_team_id": 1},
])
ppda = {
    "home": {"ppda": 8.42, "passes_allowed": 287, "defensive_actions": 34},
    "away": {"ppda": 12.65, "passes_allowed": 354, "defensive_actions": 28},
}

os.makedirs("output/visuals", exist_ok=True)

# Cover
pdf = _FakePdf()
ext._draw_match_summary_page(pdf, info, goals_df, ppda)
pdf.figs[-1].savefig("output/visuals/preview_cover.png", dpi=150,
                    bbox_inches="tight", facecolor=BG_DARK)

# Section divider
pdf = _FakePdf()
ext._draw_section_divider(pdf, "01", "PRESSING ANALYSIS",
                          "How aggressively each side pressed in the opponent's half",
                          C_GOLD)
pdf.figs[-1].savefig("output/visuals/preview_divider.png", dpi=150,
                    bbox_inches="tight", facecolor=BG_DARK)

print("Saved: output/visuals/preview_cover.png")
print("Saved: output/visuals/preview_divider.png")
